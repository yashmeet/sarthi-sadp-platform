# Encryption configuration for HIPAA compliance
# Customer-managed encryption keys (CMEK) for all data at rest

# KMS Key Ring for encryption keys
resource "google_kms_key_ring" "sarthi_keyring" {
  name     = "sarthi-encryption-keys"
  location = var.region
  project  = var.project_id
}

# Database encryption key
resource "google_kms_crypto_key" "database_key" {
  name            = "sarthi-database-key"
  key_ring        = google_kms_key_ring.sarthi_keyring.id
  rotation_period = "7776000s" # 90 days

  lifecycle {
    prevent_destroy = true
  }

  version_template {
    algorithm = "GOOGLE_SYMMETRIC_ENCRYPTION"
  }
}

# Storage encryption key
resource "google_kms_crypto_key" "storage_key" {
  name            = "sarthi-storage-key"
  key_ring        = google_kms_key_ring.sarthi_keyring.id
  rotation_period = "7776000s" # 90 days

  lifecycle {
    prevent_destroy = true
  }

  version_template {
    algorithm = "GOOGLE_SYMMETRIC_ENCRYPTION"
  }
}

# Application-level encryption key for PHI data
resource "google_kms_crypto_key" "application_key" {
  name            = "sarthi-application-key"
  key_ring        = google_kms_key_ring.sarthi_keyring.id
  rotation_period = "2592000s" # 30 days (more frequent for PHI)

  lifecycle {
    prevent_destroy = true
  }

  version_template {
    algorithm = "GOOGLE_SYMMETRIC_ENCRYPTION"
  }
}

# Backup encryption key
resource "google_kms_crypto_key" "backup_key" {
  name            = "sarthi-backup-key"
  key_ring        = google_kms_key_ring.sarthi_keyring.id
  rotation_period = "7776000s" # 90 days

  lifecycle {
    prevent_destroy = true
  }

  version_template {
    algorithm = "GOOGLE_SYMMETRIC_ENCRYPTION"
  }
}

# IAM binding for Cloud SQL service account to use database key
resource "google_kms_crypto_key_iam_binding" "database_key_binding" {
  crypto_key_id = google_kms_crypto_key.database_key.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"

  members = [
    "serviceAccount:service-${data.google_project.project.number}@gcp-sa-cloud-sql.iam.gserviceaccount.com",
  ]
}

# IAM binding for storage service account to use storage key
resource "google_kms_crypto_key_iam_binding" "storage_key_binding" {
  crypto_key_id = google_kms_crypto_key.storage_key.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"

  members = [
    "serviceAccount:service-${data.google_project.project.number}@gs-project-accounts.iam.gserviceaccount.com",
    "serviceAccount:${google_service_account.agent_runtime_sa.email}",
    "serviceAccount:${google_service_account.evaluation_sa.email}",
    "serviceAccount:${google_service_account.development_sa.email}",
    "serviceAccount:${google_service_account.monitoring_sa.email}",
  ]
}

# IAM binding for application services to use application key
resource "google_kms_crypto_key_iam_binding" "application_key_binding" {
  crypto_key_id = google_kms_crypto_key.application_key.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"

  members = [
    "serviceAccount:${google_service_account.agent_runtime_sa.email}",
    "serviceAccount:${google_service_account.evaluation_sa.email}",
    "serviceAccount:${google_service_account.development_sa.email}",
    "serviceAccount:${google_service_account.monitoring_sa.email}",
  ]
}

# Data source to get project information
data "google_project" "project" {
  project_id = var.project_id
}

# Enable required APIs for encryption
resource "google_project_service" "kms_api" {
  project = var.project_id
  service = "cloudkms.googleapis.com"
  
  disable_dependent_services = false
}

# Output encryption keys for use in other modules
output "database_encryption_key" {
  value = google_kms_crypto_key.database_key.id
}

output "storage_encryption_key" {
  value = google_kms_crypto_key.storage_key.id
}

output "application_encryption_key" {
  value = google_kms_crypto_key.application_key.id
}

output "backup_encryption_key" {
  value = google_kms_crypto_key.backup_key.id
}