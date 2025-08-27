# Comprehensive audit logging for HIPAA compliance
# Logs all access to PHI and system resources

# Enable audit logs for all services
resource "google_project_iam_audit_config" "all_services_audit" {
  project = var.project_id
  service = "allServices"
  
  audit_log_config {
    log_type = "ADMIN_READ"
  }
  
  audit_log_config {
    log_type = "DATA_READ"
  }
  
  audit_log_config {
    log_type = "DATA_WRITE"
  }
}

# BigQuery datasets for long-term audit log storage
resource "google_bigquery_dataset" "audit_logs_dataset" {
  dataset_id                  = "audit_logs"
  friendly_name               = "Audit Logs Dataset"
  description                 = "Long-term storage for audit logs (HIPAA compliance)"
  location                    = "US"
  
  # 7 years retention for HIPAA compliance
  default_table_expiration_ms = 220752000000 # 7 years in milliseconds

  access {
    role          = "OWNER"
    user_by_email = google_service_account.monitoring_sa.email
  }

  access {
    role         = "roles/bigquery.dataViewer"
    special_group = "projectOwners"
  }

  labels = {
    environment = var.environment
    compliance  = "hipaa"
    purpose     = "audit"
  }
}

resource "google_bigquery_dataset" "security_events_dataset" {
  dataset_id                  = "security_events"
  friendly_name               = "Security Events Dataset"
  description                 = "Security events and incident data"
  location                    = "US"
  
  default_table_expiration_ms = 94608000000 # 3 years

  access {
    role          = "OWNER"
    user_by_email = google_service_account.monitoring_sa.email
  }

  access {
    role         = "roles/bigquery.dataViewer"
    special_group = "projectOwners"
  }

  labels = {
    environment = var.environment
    purpose     = "security"
  }
}

# Log sinks for different types of audit events
resource "google_logging_project_sink" "admin_activity_sink" {
  name = "admin-activity-audit-sink"
  
  destination = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${google_bigquery_dataset.audit_logs_dataset.dataset_id}"
  
  filter = <<EOF
protoPayload.serviceName!="k8s.io"
AND (
  protoPayload.methodName:"iam."
  OR protoPayload.methodName:"SetIamPolicy"
  OR protoPayload.methodName:"admin"
  OR protoPayload.methodName:"insert"
  OR protoPayload.methodName:"delete"
  OR protoPayload.methodName:"update"
  OR protoPayload.methodName:"patch"
)
EOF

  unique_writer_identity = true

  bigquery_options {
    use_partitioned_tables = true
  }
}

resource "google_logging_project_sink" "data_access_sink" {
  name = "data-access-audit-sink"
  
  destination = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${google_bigquery_dataset.audit_logs_dataset.dataset_id}"
  
  filter = <<EOF
protoPayload.serviceName!="k8s.io"
AND protoPayload."@type"="type.googleapis.com/google.cloud.audit.AuditLog"
AND (
  protoPayload.serviceName="storage.googleapis.com"
  OR protoPayload.serviceName="sqladmin.googleapis.com"
  OR protoPayload.serviceName="secretmanager.googleapis.com"
  OR protoPayload.serviceName="healthcare.googleapis.com"
  OR protoPayload.serviceName="aiplatform.googleapis.com"
)
AND (
  protoPayload.methodName:"read"
  OR protoPayload.methodName:"get"
  OR protoPayload.methodName:"list"
  OR protoPayload.methodName:"access"
  OR protoPayload.methodName:"download"
)
EOF

  unique_writer_identity = true

  bigquery_options {
    use_partitioned_tables = true
  }
}

resource "google_logging_project_sink" "security_events_sink" {
  name = "security-events-sink"
  
  destination = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${google_bigquery_dataset.security_events_dataset.dataset_id}"
  
  filter = <<EOF
(
  protoPayload.serviceName="cloudresourcemanager.googleapis.com"
  OR protoPayload.serviceName="iam.googleapis.com"
  OR protoPayload.serviceName="iamcredentials.googleapis.com"
  OR protoPayload.serviceName="cloudkms.googleapis.com"
  OR protoPayload.serviceName="accesscontextmanager.googleapis.com"
)
OR (
  resource.type="gce_firewall_rule"
  OR resource.type="gce_network"
  OR resource.type="gce_subnetwork"
)
OR (
  jsonPayload.type="security"
  OR labels.security!=""
  OR severity>=ERROR
)
EOF

  unique_writer_identity = true

  bigquery_options {
    use_partitioned_tables = true
  }
}

resource "google_logging_project_sink" "application_logs_sink" {
  name = "application-logs-sink"
  
  destination = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${google_bigquery_dataset.audit_logs_dataset.dataset_id}"
  
  filter = <<EOF
resource.type="k8s_container"
AND resource.labels.namespace_name="sadp"
AND (
  jsonPayload.phi_access=true
  OR jsonPayload.user_action!=""
  OR jsonPayload.agent_execution!=""
  OR jsonPayload.data_processing!=""
  OR severity>=WARNING
)
EOF

  unique_writer_identity = true

  bigquery_options {
    use_partitioned_tables = true
  }
}

# Grant BigQuery Data Editor role to log sinks
resource "google_bigquery_dataset_iam_member" "admin_activity_sink_writer" {
  dataset_id = google_bigquery_dataset.audit_logs_dataset.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_logging_project_sink.admin_activity_sink.writer_identity
}

resource "google_bigquery_dataset_iam_member" "data_access_sink_writer" {
  dataset_id = google_bigquery_dataset.audit_logs_dataset.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_logging_project_sink.data_access_sink.writer_identity
}

resource "google_bigquery_dataset_iam_member" "security_events_sink_writer" {
  dataset_id = google_bigquery_dataset.security_events_dataset.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_logging_project_sink.security_events_sink.writer_identity
}

resource "google_bigquery_dataset_iam_member" "application_logs_sink_writer" {
  dataset_id = google_bigquery_dataset.audit_logs_dataset.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_logging_project_sink.application_logs_sink.writer_identity
}

# Cloud DLP for PHI detection in logs
resource "google_data_loss_prevention_inspect_template" "phi_detection" {
  parent       = "projects/${var.project_id}"
  description  = "Template to detect PHI in application logs"
  display_name = "PHI Detection Template"

  inspect_config {
    info_types {
      name = "PERSON_NAME"
    }
    info_types {
      name = "PHONE_NUMBER"
    }
    info_types {
      name = "EMAIL_ADDRESS"
    }
    info_types {
      name = "MEDICAL_RECORD_NUMBER"
    }
    info_types {
      name = "US_SOCIAL_SECURITY_NUMBER"
    }
    info_types {
      name = "DATE_OF_BIRTH"
    }
    info_types {
      name = "US_HEALTHCARE_NPI"
    }
    info_types {
      name = "US_DEA_NUMBER"
    }

    min_likelihood = "POSSIBLE"
    
    limits {
      max_findings_per_item    = 100
      max_findings_per_request = 1000
    }

    include_quote = true
  }
}

# Monitoring alert policies for audit events
resource "google_monitoring_alert_policy" "failed_authentication" {
  display_name = "Failed Authentication Attempts"
  description  = "Alert on multiple failed authentication attempts"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Failed authentication rate"

    condition_threshold {
      filter = <<EOF
resource.type="gce_instance"
AND jsonPayload.authentication="failed"
EOF
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["resource.label.instance_id"]
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = var.notification_channels

  alert_strategy {
    auto_close = "86400s" # 24 hours
  }

  documentation {
    content = "Multiple failed authentication attempts detected. This could indicate a brute force attack."
  }
}

resource "google_monitoring_alert_policy" "admin_activity" {
  display_name = "Suspicious Admin Activity"
  description  = "Alert on unusual administrative activities"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "High rate of admin operations"

    condition_threshold {
      filter = <<EOF
resource.type="audited_resource"
AND protoPayload.methodName:"admin"
AND protoPayload.authenticationInfo.principalEmail!~".*@system.gserviceaccount.com"
EOF
      duration        = "600s"
      comparison      = "COMPARISON_GT"
      threshold_value = 10

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["protoPayload.authenticationInfo.principalEmail"]
      }
    }
  }

  notification_channels = var.notification_channels

  alert_strategy {
    auto_close = "3600s" # 1 hour
  }

  documentation {
    content = "Unusual administrative activity detected. Review the audit logs for potential security incidents."
  }
}

resource "google_monitoring_alert_policy" "phi_access" {
  display_name = "PHI Data Access Alert"
  description  = "Alert on access to PHI data"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "PHI data access"

    condition_threshold {
      filter = <<EOF
resource.type="k8s_container"
AND resource.labels.namespace_name="sadp"
AND jsonPayload.phi_access=true
EOF
      duration        = "0s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = var.notification_channels

  alert_strategy {
    auto_close = "1800s" # 30 minutes
  }

  documentation {
    content = "PHI data access detected. This is logged for HIPAA compliance tracking."
  }
}

# Scheduled queries for audit report generation
resource "google_bigquery_data_transfer_config" "audit_report_transfer" {
  display_name           = "Daily Audit Report"
  location               = "US"
  data_source_id         = "scheduled_query"
  schedule               = "every day 06:00"
  destination_dataset_id = google_bigquery_dataset.audit_logs_dataset.dataset_id

  params = {
    query = <<EOF
SELECT
  timestamp,
  protoPayload.authenticationInfo.principalEmail as user_email,
  protoPayload.serviceName as service,
  protoPayload.methodName as method,
  protoPayload.resourceName as resource,
  protoPayload.requestMetadata.callerIp as source_ip,
  severity
FROM `${var.project_id}.audit_logs.*`
WHERE DATE(_PARTITIONTIME) = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
AND protoPayload.serviceName IN (
  'storage.googleapis.com',
  'sqladmin.googleapis.com',
  'secretmanager.googleapis.com',
  'healthcare.googleapis.com'
)
ORDER BY timestamp DESC
EOF
    destination_table_name_template = "daily_audit_report_{run_date}"
    write_disposition               = "WRITE_TRUNCATE"
  }
}

# Outputs
output "audit_logs_dataset_id" {
  value = google_bigquery_dataset.audit_logs_dataset.dataset_id
}

output "security_events_dataset_id" {
  value = google_bigquery_dataset.security_events_dataset.dataset_id
}

output "phi_detection_template" {
  value = google_data_loss_prevention_inspect_template.phi_detection.id
}