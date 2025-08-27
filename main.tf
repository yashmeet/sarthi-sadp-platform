
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "gcp_services" {
  count   = length(local.gcp_apis)
  project = var.project_id
  service = local.gcp_apis[count.index]

  disable_dependent_services = true
}

resource "google_compute_network" "vpc_network" {
  name                    = "sarthi-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "sarthi_subnet" {
  name          = "sarthi-subnet"
  ip_cidr_range = "10.0.0.0/16"
  network       = google_compute_network.vpc_network.id
  region        = var.region
}

resource "google_container_cluster" "sarthi_gke_cluster" {
  name     = "sarthi-gke-cluster"
  location = var.region

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.vpc_network.name
  subnetwork = google_compute_subnetwork.sarthi_subnet.name
}

# Enhanced security modules
# module "encryption" {
#   source = "./terraform/security"
# }

# Will implement enhanced modules in next phase
# For now, using existing simplified infrastructure



resource "google_container_node_pool" "sarthi_gke_node_pool" {
  name       = "sarthi-gke-node-pool"
  location   = var.region
  cluster    = google_container_cluster.sarthi_gke_cluster.name
  node_count = 1

  node_config {
    preemptible  = true
    machine_type = "e2-medium"
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }

  lifecycle {
    ignore_changes = [
      node_config.0.resource_labels,
      node_config.0.kubelet_config
    ]
  }
}

resource "google_storage_bucket" "poml_library_bucket" {
  name          = "sarthi-poml-library-bucket"
  location      = var.region
  force_destroy = true
}

resource "google_storage_bucket" "app_data_bucket" {
  name          = "sarthi-app-data-bucket"
  location      = var.region
  force_destroy = true
}

resource "google_pubsub_topic" "agent_runtime_topic" {
  name = "agent-runtime-topic"
}

resource "google_pubsub_topic" "evaluation_topic" {
  name = "evaluation-topic"
}

resource "google_pubsub_topic" "development_topic" {
  name = "development-topic"
}

resource "google_pubsub_topic" "monitoring_topic" {
  name = "monitoring-topic"
}

resource "google_service_account" "agent_runtime_sa" {
  account_id   = "agent-runtime-sa"
  display_name = "Agent Runtime Service Account"
}

resource "google_service_account" "evaluation_sa" {
  account_id   = "evaluation-sa"
  display_name = "Evaluation Service Account"
}

resource "google_service_account" "development_sa" {
  account_id   = "development-sa"
  display_name = "Development Service Account"
}

resource "google_service_account" "monitoring_sa" {
  account_id   = "monitoring-sa"
  display_name = "Monitoring Service Account"
}

resource "google_secret_manager_secret" "sql_password_secret" {
  secret_id = "sql-password"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "poml_api_key_secret" {
  secret_id = "poml-api-key"

  replication {
    auto {}
  }
}

resource "google_sql_database_instance" "sarthi_sql_instance" {
  name             = "sarthi-sql-instance"
  database_version = "POSTGRES_13"
  region           = var.region

  settings {
    tier = "db-g1-small"
  }

  deletion_protection = false
}

# Access Context Manager requires organization-level permissions
# Uncomment when organization admin access is available
# resource "google_access_context_manager_access_policy" "access_policy" {
#   parent = "organizations/${var.organization_id}"
#   title  = "Sarthi Access Policy"
# }

# resource "google_access_context_manager_service_perimeter" "service_perimeter" {
#   parent  = "accessPolicies/${google_access_context_manager_access_policy.access_policy.name}"
#   name    = "accessPolicies/${google_access_context_manager_access_policy.access_policy.name}/servicePerimeters/sarthi_perimeter"
#   title   = "Sarthi Service Perimeter"
#   status {
#     restricted_services = ["storage.googleapis.com", "sqladmin.googleapis.com"]
#   }
# }

resource "google_artifact_registry_repository" "sarthi_registry" {
  location      = var.region
  repository_id = "sarthi-services"
  description   = "Docker repository for SADP microservices"
  format        = "DOCKER"
}

locals {
  gcp_apis = [
    "serviceusage.googleapis.com",
    "compute.googleapis.com",
    "container.googleapis.com",
    "sqladmin.googleapis.com",
    "storage-component.googleapis.com",
    "pubsub.googleapis.com",
    "iam.googleapis.com",
    "secretmanager.googleapis.com",
    "artifactregistry.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "clouddeploy.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "apigateway.googleapis.com",
    "healthcare.googleapis.com",
    "aiplatform.googleapis.com",
    "binaryauthorization.googleapis.com",
    "vpcaccess.googleapis.com"
  ]
}
