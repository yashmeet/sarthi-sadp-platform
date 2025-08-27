# Network security configuration for HIPAA compliance
# VPC Service Controls, private endpoints, and network segmentation

# Enhanced VPC with security-focused configuration
resource "google_compute_network" "vpc_network" {
  name                    = "sarthi-vpc"
  auto_create_subnetworks = false
  mtu                     = 1460
  
  # Enable flow logs for security monitoring
  description = "SADP VPC with enhanced security controls"
}

# Private subnet for application services
resource "google_compute_subnetwork" "private_subnet" {
  name                     = "sarthi-private-subnet"
  ip_cidr_range           = "10.0.1.0/24"
  region                  = var.region
  network                 = google_compute_network.vpc_network.id
  private_ip_google_access = true
  
  # Enable flow logs for security monitoring
  log_config {
    aggregation_interval = "INTERVAL_10_MIN"
    flow_sampling        = 0.5
    metadata             = "INCLUDE_ALL_METADATA"
  }
  
  secondary_ip_range {
    range_name    = "gke-pods"
    ip_cidr_range = "10.1.0.0/16"
  }
  
  secondary_ip_range {
    range_name    = "gke-services"
    ip_cidr_range = "10.2.0.0/16"
  }
}

# Management subnet for administrative access
resource "google_compute_subnetwork" "management_subnet" {
  name                     = "sarthi-management-subnet"
  ip_cidr_range           = "10.0.2.0/24"
  region                  = var.region
  network                 = google_compute_network.vpc_network.id
  private_ip_google_access = true
  
  log_config {
    aggregation_interval = "INTERVAL_5_MIN"
    flow_sampling        = 1.0
    metadata             = "INCLUDE_ALL_METADATA"
  }
}

# Cloud Router for NAT
resource "google_compute_router" "router" {
  name    = "sarthi-router"
  region  = var.region
  network = google_compute_network.vpc_network.id
}

# Cloud NAT for secure outbound traffic
resource "google_compute_router_nat" "nat" {
  name                               = "sarthi-nat"
  router                            = google_compute_router.router.name
  region                            = var.region
  nat_ip_allocate_option            = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Firewall rules for secure communication
resource "google_compute_firewall" "deny_all_ingress" {
  name      = "sarthi-deny-all-ingress"
  network   = google_compute_network.vpc_network.name
  direction = "INGRESS"
  priority  = 65534

  deny {
    protocol = "all"
  }

  source_ranges = ["0.0.0.0/0"]
  
  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_firewall" "allow_internal" {
  name      = "sarthi-allow-internal"
  network   = google_compute_network.vpc_network.name
  direction = "INGRESS"
  priority  = 1000

  allow {
    protocol = "tcp"
  }

  allow {
    protocol = "udp"
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = [
    "10.0.0.0/16",
    "10.1.0.0/16",
    "10.2.0.0/16"
  ]
  
  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_firewall" "allow_gke_master" {
  name      = "sarthi-allow-gke-master"
  network   = google_compute_network.vpc_network.name
  direction = "INGRESS"
  priority  = 1000

  allow {
    protocol = "tcp"
    ports    = ["443", "10250"]
  }

  source_ranges = ["172.16.0.0/28"] # GKE master range
  target_tags   = ["gke-node"]
  
  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_firewall" "allow_health_checks" {
  name      = "sarthi-allow-health-checks"
  network   = google_compute_network.vpc_network.name
  direction = "INGRESS"
  priority  = 1000

  allow {
    protocol = "tcp"
    ports    = ["8080", "8000"]
  }

  source_ranges = [
    "130.211.0.0/22",
    "35.191.0.0/16"
  ]
  target_tags = ["http-server"]
  
  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_firewall" "allow_load_balancer" {
  name      = "sarthi-allow-load-balancer"
  network   = google_compute_network.vpc_network.name
  direction = "INGRESS"
  priority  = 1000

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["http-server", "https-server"]
  
  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}

# Private Service Connect for Cloud SQL
resource "google_compute_global_address" "private_ip_address" {
  name          = "sarthi-private-ip-address"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc_network.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.vpc_network.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

# VPC Service Controls Access Policy
resource "google_access_context_manager_access_policy" "access_policy" {
  count  = var.enable_vpc_service_controls ? 1 : 0
  parent = "organizations/${var.organization_id}"
  title  = "sarthi-access-policy"
}

# Service Perimeter for HIPAA compliance
resource "google_access_context_manager_service_perimeter" "service_perimeter" {
  count  = var.enable_vpc_service_controls ? 1 : 0
  parent = "accessPolicies/${google_access_context_manager_access_policy.access_policy[0].name}"
  name   = "accessPolicies/${google_access_context_manager_access_policy.access_policy[0].name}/servicePerimeters/sarthi_perimeter"
  title  = "sarthi-service-perimeter"

  status {
    restricted_services = [
      "storage.googleapis.com",
      "sqladmin.googleapis.com",
      "secretmanager.googleapis.com",
      "cloudkms.googleapis.com",
      "pubsub.googleapis.com",
      "aiplatform.googleapis.com",
      "healthcare.googleapis.com",
    ]

    resources = [
      "projects/${data.google_project.project.number}"
    ]

    access_levels = []

    vpc_accessible_services {
      enable_restriction = true
      allowed_services = [
        "storage.googleapis.com",
        "sqladmin.googleapis.com",
        "secretmanager.googleapis.com",
        "cloudkms.googleapis.com",
        "pubsub.googleapis.com",
        "aiplatform.googleapis.com",
        "healthcare.googleapis.com",
        "container.googleapis.com",
        "monitoring.googleapis.com",
        "logging.googleapis.com",
      ]
    }
  }
}

# Cloud Armor security policy for DDoS protection
resource "google_compute_security_policy" "security_policy" {
  name        = "sarthi-security-policy"
  description = "Security policy for SADP with rate limiting and geo-blocking"

  # Default rule
  rule {
    action   = "allow"
    priority = "2147483647"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    description = "Default allow rule"
  }

  # Rate limiting rule
  rule {
    action   = "rate_based_ban"
    priority = "1000"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"
      enforce_on_key = "IP"
      rate_limit_threshold {
        count        = 100
        interval_sec = 60
      }
      ban_duration_sec = 600
    }
    description = "Rate limiting rule"
  }

  # Block known malicious IPs
  rule {
    action   = "deny(403)"
    priority = "500"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = [
          # Add known malicious IP ranges here
        ]
      }
    }
    description = "Block known malicious IPs"
  }

  adaptive_protection_config {
    layer_7_ddos_defense_config {
      enable = true
    }
  }
}

# SSL Policy for enhanced security
resource "google_compute_ssl_policy" "ssl_policy" {
  name            = "sarthi-ssl-policy"
  profile         = "RESTRICTED"
  min_tls_version = "TLS_1_2"
}

# Data source for project information
data "google_project" "project" {
  project_id = var.project_id
}

# Outputs
output "vpc_network_id" {
  value = google_compute_network.vpc_network.id
}

output "private_subnet_id" {
  value = google_compute_subnetwork.private_subnet.id
}

output "management_subnet_id" {
  value = google_compute_subnetwork.management_subnet.id
}

output "security_policy_id" {
  value = google_compute_security_policy.security_policy.id
}

output "ssl_policy_id" {
  value = google_compute_ssl_policy.ssl_policy.id
}