# Variables for API Gateway module

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
}

variable "environment" {
  description = "Environment name (development, staging, production)"
  type        = string
  default     = "production"
}

variable "api_domain" {
  description = "Domain for the API Gateway"
  type        = string
  default     = "api.sarthi.ai"
}

variable "security_policy_id" {
  description = "Cloud Armor security policy ID"
  type        = string
}

variable "ssl_policy_id" {
  description = "SSL policy ID"
  type        = string
}

variable "agent_runtime_sa_email" {
  description = "Email of the agent runtime service account"
  type        = string
}

variable "evaluation_sa_email" {
  description = "Email of the evaluation service account"
  type        = string
}

variable "development_sa_email" {
  description = "Email of the development service account"
  type        = string
}

variable "monitoring_sa_email" {
  description = "Email of the monitoring service account"
  type        = string
}

variable "notification_channels" {
  description = "List of notification channels for alerts"
  type        = list(string)
  default     = []
}