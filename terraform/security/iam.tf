# IAM configuration for HIPAA compliance and least privilege access
# Enhanced service accounts with proper role bindings

# Agent Runtime Service Account with enhanced permissions
resource "google_service_account" "agent_runtime_sa" {
  account_id   = "agent-runtime-sa"
  display_name = "Agent Runtime Service Account"
  description  = "Service account for SADP Agent Runtime Service with minimal required permissions"
}

# Evaluation Service Account
resource "google_service_account" "evaluation_sa" {
  account_id   = "evaluation-sa"
  display_name = "Evaluation Service Account"
  description  = "Service account for SADP Evaluation Service"
}

# Development Service Account
resource "google_service_account" "development_sa" {
  account_id   = "development-sa"
  display_name = "Development Service Account"
  description  = "Service account for SADP Development Service"
}

# Monitoring Service Account
resource "google_service_account" "monitoring_sa" {
  account_id   = "monitoring-sa"
  display_name = "Monitoring Service Account"
  description  = "Service account for SADP Monitoring Service"
}

# Custom IAM roles for fine-grained access control
resource "google_project_iam_custom_role" "agent_runtime_role" {
  role_id     = "sarthi.agentRuntime"
  title       = "SADP Agent Runtime Role"
  description = "Custom role for Agent Runtime Service with minimal required permissions"
  
  permissions = [
    # Pub/Sub permissions
    "pubsub.topics.publish",
    "pubsub.subscriptions.consume",
    "pubsub.messages.ack",
    
    # Storage permissions (read/write to specific buckets)
    "storage.objects.get",
    "storage.objects.create",
    "storage.objects.delete",
    "storage.buckets.get",
    
    # Cloud SQL permissions (connect only)
    "cloudsql.instances.connect",
    
    # Secret Manager permissions (specific secrets only)
    "secretmanager.versions.access",
    
    # AI Platform permissions
    "aiplatform.endpoints.predict",
    "aiplatform.models.predict",
    
    # Monitoring permissions
    "monitoring.metricDescriptors.create",
    "monitoring.metricDescriptors.get",
    "monitoring.timeSeries.create",
    
    # Logging permissions
    "logging.logEntries.create",
  ]
}

resource "google_project_iam_custom_role" "evaluation_role" {
  role_id     = "sarthi.evaluation"
  title       = "SADP Evaluation Role"
  description = "Custom role for Evaluation Service"
  
  permissions = [
    "pubsub.topics.publish",
    "pubsub.subscriptions.consume",
    "pubsub.messages.ack",
    "storage.objects.get",
    "storage.objects.create",
    "cloudsql.instances.connect",
    "secretmanager.versions.access",
    "monitoring.metricDescriptors.create",
    "monitoring.timeSeries.create",
    "logging.logEntries.create",
    "bigquery.datasets.get",
    "bigquery.tables.get",
    "bigquery.tables.create",
    "bigquery.tables.updateData",
  ]
}

resource "google_project_iam_custom_role" "development_role" {
  role_id     = "sarthi.development"
  title       = "SADP Development Role"
  description = "Custom role for Development Service"
  
  permissions = [
    "pubsub.topics.publish",
    "pubsub.subscriptions.consume",
    "pubsub.messages.ack",
    "storage.objects.get",
    "storage.objects.create",
    "storage.objects.update",
    "storage.objects.delete",
    "cloudsql.instances.connect",
    "secretmanager.versions.access",
    "monitoring.metricDescriptors.create",
    "monitoring.timeSeries.create",
    "logging.logEntries.create",
    "artifactregistry.repositories.get",
    "artifactregistry.dockerimages.get",
    "artifactregistry.dockerimages.list",
  ]
}

resource "google_project_iam_custom_role" "monitoring_role" {
  role_id     = "sarthi.monitoring"
  title       = "SADP Monitoring Role"
  description = "Custom role for Monitoring Service"
  
  permissions = [
    "pubsub.topics.publish",
    "pubsub.subscriptions.consume",
    "pubsub.messages.ack",
    "storage.objects.get",
    "cloudsql.instances.connect",
    "secretmanager.versions.access",
    "monitoring.metricDescriptors.create",
    "monitoring.metricDescriptors.get",
    "monitoring.metricDescriptors.list",
    "monitoring.timeSeries.create",
    "monitoring.timeSeries.list",
    "logging.logEntries.create",
    "logging.logEntries.list",
    "logging.sinks.get",
    "logging.sinks.list",
    "bigquery.datasets.get",
    "bigquery.tables.get",
    "bigquery.tables.getData",
  ]
}

# Bind custom roles to service accounts
resource "google_project_iam_binding" "agent_runtime_binding" {
  project = var.project_id
  role    = google_project_iam_custom_role.agent_runtime_role.id

  members = [
    "serviceAccount:${google_service_account.agent_runtime_sa.email}",
  ]
}

resource "google_project_iam_binding" "evaluation_binding" {
  project = var.project_id
  role    = google_project_iam_custom_role.evaluation_role.id

  members = [
    "serviceAccount:${google_service_account.evaluation_sa.email}",
  ]
}

resource "google_project_iam_binding" "development_binding" {
  project = var.project_id
  role    = google_project_iam_custom_role.development_role.id

  members = [
    "serviceAccount:${google_service_account.development_sa.email}",
  ]
}

resource "google_project_iam_binding" "monitoring_binding" {
  project = var.project_id
  role    = google_project_iam_custom_role.monitoring_role.id

  members = [
    "serviceAccount:${google_service_account.monitoring_sa.email}",
  ]
}

# Workload Identity configuration for GKE
resource "google_service_account_iam_binding" "workload_identity_agent_runtime" {
  service_account_id = google_service_account.agent_runtime_sa.name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[sadp/agent-runtime]",
  ]
}

resource "google_service_account_iam_binding" "workload_identity_evaluation" {
  service_account_id = google_service_account.evaluation_sa.name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[sadp/evaluation]",
  ]
}

resource "google_service_account_iam_binding" "workload_identity_development" {
  service_account_id = google_service_account.development_sa.name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[sadp/development]",
  ]
}

resource "google_service_account_iam_binding" "workload_identity_monitoring" {
  service_account_id = google_service_account.monitoring_sa.name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[sadp/monitoring]",
  ]
}

# IAM bindings for specific resource access
# Cloud SQL IAM database user
resource "google_sql_user" "app_users" {
  for_each = toset([
    google_service_account.agent_runtime_sa.email,
    google_service_account.evaluation_sa.email,
    google_service_account.development_sa.email,
    google_service_account.monitoring_sa.email,
  ])
  
  name     = each.value
  instance = google_sql_database_instance.sarthi_sql_instance.name
  type     = "CLOUD_IAM_SERVICE_ACCOUNT"
}

# Storage bucket IAM bindings with specific permissions
resource "google_storage_bucket_iam_binding" "app_data_bucket_binding" {
  bucket = google_storage_bucket.app_data_bucket.name
  role   = "roles/storage.objectAdmin"

  members = [
    "serviceAccount:${google_service_account.agent_runtime_sa.email}",
    "serviceAccount:${google_service_account.evaluation_sa.email}",
    "serviceAccount:${google_service_account.development_sa.email}",
  ]
}

resource "google_storage_bucket_iam_binding" "poml_bucket_binding" {
  bucket = google_storage_bucket.poml_library_bucket.name
  role   = "roles/storage.objectViewer"

  members = [
    "serviceAccount:${google_service_account.agent_runtime_sa.email}",
    "serviceAccount:${google_service_account.development_sa.email}",
  ]
}

resource "google_storage_bucket_iam_binding" "poml_bucket_admin_binding" {
  bucket = google_storage_bucket.poml_library_bucket.name
  role   = "roles/storage.objectAdmin"

  members = [
    "serviceAccount:${google_service_account.development_sa.email}",
  ]
}

# Pub/Sub IAM bindings
resource "google_pubsub_topic_iam_binding" "agent_runtime_topic_binding" {
  topic = google_pubsub_topic.agent_runtime_topic.name
  role  = "roles/pubsub.publisher"

  members = [
    "serviceAccount:${google_service_account.agent_runtime_sa.email}",
    "serviceAccount:${google_service_account.evaluation_sa.email}",
  ]
}

resource "google_pubsub_subscription_iam_binding" "agent_runtime_subscription_binding" {
  subscription = google_pubsub_subscription.agent_runtime_subscription.name
  role         = "roles/pubsub.subscriber"

  members = [
    "serviceAccount:${google_service_account.agent_runtime_sa.email}",
  ]
}

# Audit logging for IAM changes
resource "google_logging_project_sink" "iam_audit_sink" {
  name = "iam-audit-sink"
  
  destination = "bigquery.googleapis.com/projects/${var.project_id}/datasets/security_audit_logs"
  
  filter = <<EOF
protoPayload.serviceName="iam.googleapis.com"
OR protoPayload.serviceName="cloudresourcemanager.googleapis.com"
OR protoPayload.serviceName="serviceusage.googleapis.com"
EOF

  unique_writer_identity = true
}

# BigQuery dataset for audit logs
resource "google_bigquery_dataset" "security_audit_logs" {
  dataset_id                  = "security_audit_logs"
  friendly_name               = "Security Audit Logs"
  description                 = "Audit logs for security compliance"
  location                    = "US"
  default_table_expiration_ms = 2628000000 # 1 month

  access {
    role          = "OWNER"
    user_by_email = google_service_account.monitoring_sa.email
  }

  access {
    role           = "roles/bigquery.dataViewer"
    special_group  = "projectOwners"
  }
}

# Outputs for service account emails
output "agent_runtime_sa_email" {
  value = google_service_account.agent_runtime_sa.email
}

output "evaluation_sa_email" {
  value = google_service_account.evaluation_sa.email
}

output "development_sa_email" {
  value = google_service_account.development_sa.email
}

output "monitoring_sa_email" {
  value = google_service_account.monitoring_sa.email
}