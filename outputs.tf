output "vpc_network_name" {
  description = "The name of the VPC network."
  value       = google_compute_network.vpc_network.name
}

output "sarthi_subnet_name" {
  description = "The name of the subnetwork."
  value       = google_compute_subnetwork.sarthi_subnet.name
}

output "gke_cluster_name" {
  description = "The name of the GKE cluster."
  value       = google_container_cluster.sarthi_gke_cluster.name
}

output "gke_node_pool_name" {
  description = "The name of the GKE node pool."
  value       = google_container_node_pool.sarthi_gke_node_pool.name
}

output "sql_instance_name" {
  description = "The name of the Cloud SQL instance."
  value       = google_sql_database_instance.sarthi_sql_instance.name
}

output "poml_library_bucket_name" {
  description = "The name of the POML library bucket."
  value       = google_storage_bucket.poml_library_bucket.name
}

output "app_data_bucket_name" {
  description = "The name of the application data bucket."
  value       = google_storage_bucket.app_data_bucket.name
}

output "agent_runtime_topic_name" {
  description = "The name of the Agent Runtime Pub/Sub topic."
  value       = google_pubsub_topic.agent_runtime_topic.name
}

output "evaluation_topic_name" {
  description = "The name of the Evaluation Pub/Sub topic."
  value       = google_pubsub_topic.evaluation_topic.name
}

output "development_topic_name" {
  description = "The name of the Development Pub/Sub topic."
  value       = google_pubsub_topic.development_topic.name
}

output "monitoring_topic_name" {
  description = "The name of the Monitoring Pub/Sub topic."
  value       = google_pubsub_topic.monitoring_topic.name
}

output "agent_runtime_sa_email" {
  description = "The email of the Agent Runtime service account."
  value       = google_service_account.agent_runtime_sa.email
}

output "evaluation_sa_email" {
  description = "The email of the Evaluation service account."
  value       = google_service_account.evaluation_sa.email
}

output "development_sa_email" {
  description = "The email of the Development service account."
  value       = google_service_account.development_sa.email
}

output "monitoring_sa_email" {
  description = "The email of the Monitoring service account."
  value       = google_service_account.monitoring_sa.email
}

output "sql_password_secret_name" {
  description = "The name of the Cloud SQL password secret."
  value       = google_secret_manager_secret.sql_password_secret.secret_id
}

output "poml_api_key_secret_name" {
  description = "The name of the POML API key secret."
  value       = google_secret_manager_secret.poml_api_key_secret.secret_id
}

output "artifact_registry_url" {
  description = "The URL of the Artifact Registry repository."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.sarthi_registry.repository_id}"
}