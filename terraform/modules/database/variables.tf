# Variables for the database module

variable "instance_name" {
  description = "Name of the Cloud SQL instance"
  type        = string
  default     = "sarthi-sql-instance"
}

variable "region" {
  description = "The region where the database will be created"
  type        = string
}

variable "environment" {
  description = "Environment name (development, staging, production)"
  type        = string
  default     = "production"
}

variable "database_tier" {
  description = "The machine type for the database instance"
  type        = string
  default     = "db-custom-2-7680" # 2 vCPUs, 7.5GB RAM - production ready
}

variable "disk_size" {
  description = "The size of the database disk in GB"
  type        = number
  default     = 100
}

variable "disk_autoresize_limit" {
  description = "The maximum size to which storage can be auto-resized (GB)"
  type        = number
  default     = 1000
}

variable "backup_location" {
  description = "The location for database backups"
  type        = string
  default     = "us"
}

variable "vpc_network_id" {
  description = "The ID of the VPC network"
  type        = string
}

variable "management_subnet_cidr" {
  description = "CIDR block for the management subnet"
  type        = string
  default     = "10.0.2.0/24"
}

variable "database_encryption_key" {
  description = "The KMS key for database encryption"
  type        = string
}

variable "sql_password_secret_id" {
  description = "The ID of the Secret Manager secret for database password"
  type        = string
}

variable "service_account_emails" {
  description = "Set of service account emails that need database access"
  type        = set(string)
  default     = []
}

variable "enable_read_replica" {
  description = "Whether to create a read replica"
  type        = bool
  default     = true
}

variable "replica_region" {
  description = "The region for the read replica"
  type        = string
  default     = "us-east1"
}

variable "replica_tier" {
  description = "The machine type for the read replica"
  type        = string
  default     = "db-custom-1-3840" # Smaller tier for read replica
}

variable "notification_channels" {
  description = "List of notification channels for alerts"
  type        = list(string)
  default     = []
}

variable "private_vpc_connection" {
  description = "The private VPC connection for database access"
  type        = any
}