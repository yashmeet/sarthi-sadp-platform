# Enhanced Cloud SQL configuration for HIPAA compliance
# Encrypted, highly available database with comprehensive security

# Random password for database
resource "random_password" "db_password" {
  length  = 32
  special = true
}

# Store database password in Secret Manager
resource "google_secret_manager_secret_version" "db_password" {
  secret      = var.sql_password_secret_id
  secret_data = random_password.db_password.result
}

# Enhanced Cloud SQL instance with encryption and security
resource "google_sql_database_instance" "sarthi_sql_instance" {
  name                = var.instance_name
  database_version    = "POSTGRES_15"
  region              = var.region
  deletion_protection = true

  settings {
    tier                        = var.database_tier
    availability_type          = "REGIONAL" # High availability
    disk_type                  = "PD_SSD"
    disk_size                  = var.disk_size
    disk_autoresize            = true
    disk_autoresize_limit      = var.disk_autoresize_limit
    
    # Enhanced security configuration
    user_labels = {
      environment = var.environment
      compliance  = "hipaa"
      service     = "sadp"
    }

    # Database flags for security
    database_flags {
      name  = "log_checkpoints"
      value = "on"
    }
    
    database_flags {
      name  = "log_connections"
      value = "on"
    }
    
    database_flags {
      name  = "log_disconnections"
      value = "on"
    }
    
    database_flags {
      name  = "log_lock_waits"
      value = "on"
    }
    
    database_flags {
      name  = "log_statement"
      value = "all"
    }
    
    database_flags {
      name  = "log_min_duration_statement"
      value = "1000" # Log queries taking more than 1 second
    }
    
    database_flags {
      name  = "shared_preload_libraries"
      value = "pgaudit"
    }

    # Backup configuration for HIPAA compliance
    backup_configuration {
      enabled                        = true
      start_time                     = "03:00" # 3 AM UTC
      location                       = var.backup_location
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 30 # 30 days retention
        retention_unit   = "COUNT"
      }
    }

    # IP configuration for private access
    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = var.vpc_network_id
      enable_private_path_for_google_cloud_services = true
      require_ssl                                   = true
      
      authorized_networks {
        name  = "management-subnet"
        value = var.management_subnet_cidr
      }
    }

    # Maintenance window
    maintenance_window {
      day          = 1 # Sunday
      hour         = 4 # 4 AM UTC
      update_track = "stable"
    }

    # Enhanced monitoring
    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }
  }

  # Enable customer-managed encryption
  encryption_key_name = var.database_encryption_key

  depends_on = [
    var.private_vpc_connection
  ]

  lifecycle {
    prevent_destroy = true
  }
}

# Create databases
resource "google_sql_database" "sarthi_database" {
  name     = "sarthi"
  instance = google_sql_database_instance.sarthi_sql_instance.name
  charset  = "UTF8"
  collation = "en_US.UTF8"
}

resource "google_sql_database" "audit_database" {
  name     = "audit_logs"
  instance = google_sql_database_instance.sarthi_sql_instance.name
  charset  = "UTF8"
  collation = "en_US.UTF8"
}

# Database users with IAM authentication
resource "google_sql_user" "postgres_admin" {
  name     = "postgres"
  instance = google_sql_database_instance.sarthi_sql_instance.name
  password = random_password.db_password.result
}

# Service account database users (created via IAM module)
resource "google_sql_user" "service_account_users" {
  for_each = var.service_account_emails
  
  name     = each.value
  instance = google_sql_database_instance.sarthi_sql_instance.name
  type     = "CLOUD_IAM_SERVICE_ACCOUNT"
}

# Read replica for improved performance and disaster recovery
resource "google_sql_database_instance" "read_replica" {
  count = var.enable_read_replica ? 1 : 0
  
  name                 = "${var.instance_name}-read-replica"
  master_instance_name = google_sql_database_instance.sarthi_sql_instance.name
  region               = var.replica_region
  database_version     = "POSTGRES_15"

  replica_configuration {
    failover_target = false
  }

  settings {
    tier              = var.replica_tier
    availability_type = "ZONAL"
    disk_type         = "PD_SSD"
    
    user_labels = {
      environment = var.environment
      compliance  = "hipaa"
      service     = "sadp"
      type        = "read-replica"
    }

    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = var.vpc_network_id
      enable_private_path_for_google_cloud_services = true
      require_ssl                                   = true
    }

    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }
  }

  encryption_key_name = var.database_encryption_key
}

# SSL certificates for secure connections
resource "google_sql_ssl_cert" "client_cert" {
  common_name = "sarthi-client-cert"
  instance    = google_sql_database_instance.sarthi_sql_instance.name
}

# Monitoring and alerting
resource "google_monitoring_alert_policy" "database_cpu" {
  display_name = "Database CPU Usage Alert"
  combiner     = "OR"
  
  conditions {
    display_name = "Database CPU usage"
    
    condition_threshold {
      filter          = "resource.type=\"gce_instance\" AND metric.type=\"compute.googleapis.com/instance/cpu/utilization\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.8
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = var.notification_channels

  alert_strategy {
    auto_close = "1800s"
  }
}

resource "google_monitoring_alert_policy" "database_connections" {
  display_name = "Database Connection Count Alert"
  combiner     = "OR"
  
  conditions {
    display_name = "High database connection count"
    
    condition_threshold {
      filter          = "resource.type=\"cloudsql_database\" AND metric.type=\"cloudsql.googleapis.com/database/network/connections\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 80
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = var.notification_channels
}

# Outputs
output "database_instance_name" {
  value = google_sql_database_instance.sarthi_sql_instance.name
}

output "database_connection_name" {
  value = google_sql_database_instance.sarthi_sql_instance.connection_name
}

output "database_private_ip" {
  value = google_sql_database_instance.sarthi_sql_instance.private_ip_address
}

output "database_instance_id" {
  value = google_sql_database_instance.sarthi_sql_instance.id
}

output "ssl_cert" {
  value = google_sql_ssl_cert.client_cert.cert
  sensitive = true
}

output "ssl_key" {
  value = google_sql_ssl_cert.client_cert.private_key
  sensitive = true
}

output "ssl_ca_cert" {
  value = google_sql_ssl_cert.client_cert.server_ca_cert
  sensitive = true
}