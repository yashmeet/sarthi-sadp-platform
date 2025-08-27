
variable "organization_id" {
  description = "The ID of the GCP organization."
  type        = string
}

variable "project_id" {
  description = "The ID of the GCP project."
  type        = string
  default     = "sarthi-patient-experience-hub"
}

variable "region" {
  description = "The GCP region to deploy the resources in."
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (development, staging, production)"
  type        = string
  default     = "production"
}

variable "enable_vpc_service_controls" {
  description = "Enable VPC Service Controls (requires organization-level permissions)"
  type        = bool
  default     = false
}

variable "notification_channels" {
  description = "List of notification channels for alerts"
  type        = list(string)
  default     = []
}
