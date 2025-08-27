# API Gateway configuration for SADP
# Centralized API management with authentication and rate limiting

# API Gateway
resource "google_api_gateway_api" "sadp_api" {
  provider = google-beta
  api_id   = "sadp-api"
  project  = var.project_id
  
  labels = {
    environment = var.environment
    service     = "api-gateway"
  }
}

# API Gateway configuration
resource "google_api_gateway_api_config" "sadp_api_config" {
  provider      = google-beta
  api           = google_api_gateway_api.sadp_api.api_id
  api_config_id = "sadp-api-config"
  project       = var.project_id

  openapi_documents {
    document {
      path     = "sadp-api.yaml"
      contents = base64encode(templatefile("${path.module}/sadp-api.yaml", {
        PROJECT_ID = var.project_id
        REGION     = var.region
      }))
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

# API Gateway
resource "google_api_gateway_gateway" "sadp_gateway" {
  provider   = google-beta
  api_config = google_api_gateway_api_config.sadp_api_config.id
  gateway_id = "sadp-gateway"
  region     = var.region
  project    = var.project_id

  labels = {
    environment = var.environment
    service     = "api-gateway"
  }
}

# Load Balancer for API Gateway
resource "google_compute_global_address" "api_gateway_ip" {
  name         = "sadp-api-gateway-ip"
  address_type = "EXTERNAL"
}

resource "google_compute_url_map" "api_gateway_url_map" {
  name            = "sadp-api-gateway-url-map"
  default_service = google_compute_backend_service.api_gateway_backend.id

  host_rule {
    hosts        = [var.api_domain]
    path_matcher = "allpaths"
  }

  path_matcher {
    name            = "allpaths"
    default_service = google_compute_backend_service.api_gateway_backend.id

    path_rule {
      paths   = ["/api/v1/*"]
      service = google_compute_backend_service.api_gateway_backend.id
    }
  }
}

resource "google_compute_backend_service" "api_gateway_backend" {
  name                  = "sadp-api-gateway-backend"
  protocol              = "HTTPS"
  port_name             = "https"
  timeout_sec           = 30
  enable_cdn            = true
  compression_mode      = "AUTOMATIC"
  security_policy       = var.security_policy_id

  backend {
    group = google_compute_network_endpoint_group.api_gateway_neg.id
  }

  health_checks = [google_compute_health_check.api_gateway_health_check.id]

  log_config {
    enable = true
  }

  cdn_policy {
    cache_mode                   = "USE_ORIGIN_HEADERS"
    client_ttl                   = 3600
    default_ttl                  = 3600
    max_ttl                      = 86400
    negative_caching             = true
    serve_while_stale            = 86400
    
    cache_key_policy {
      include_host         = true
      include_protocol     = true
      include_query_string = true
    }
  }
}

resource "google_compute_network_endpoint_group" "api_gateway_neg" {
  name                  = "sadp-api-gateway-neg"
  network_endpoint_type = "INTERNET_FQDN_PORT"
  zone                  = "${var.region}-a"
}

resource "google_compute_network_endpoint" "api_gateway_endpoint" {
  network_endpoint_group = google_compute_network_endpoint_group.api_gateway_neg.name
  zone                   = google_compute_network_endpoint_group.api_gateway_neg.zone
  fqdn                   = google_api_gateway_gateway.sadp_gateway.default_hostname
  port                   = 443
}

resource "google_compute_health_check" "api_gateway_health_check" {
  name               = "sadp-api-gateway-health-check"
  check_interval_sec = 30
  timeout_sec        = 10

  https_health_check {
    port         = "443"
    request_path = "/health"
  }
}

resource "google_compute_target_https_proxy" "api_gateway_proxy" {
  name             = "sadp-api-gateway-proxy"
  url_map          = google_compute_url_map.api_gateway_url_map.id
  ssl_certificates = [google_compute_managed_ssl_certificate.api_gateway_cert.id]
  ssl_policy       = var.ssl_policy_id
}

resource "google_compute_managed_ssl_certificate" "api_gateway_cert" {
  name = "sadp-api-gateway-cert"

  managed {
    domains = [var.api_domain]
  }
}

resource "google_compute_global_forwarding_rule" "api_gateway_forwarding_rule" {
  name                  = "sadp-api-gateway-forwarding-rule"
  ip_protocol           = "TCP"
  load_balancing_scheme = "EXTERNAL"
  port_range            = "443"
  target                = google_compute_target_https_proxy.api_gateway_proxy.id
  ip_address            = google_compute_global_address.api_gateway_ip.id
}

# HTTP to HTTPS redirect
resource "google_compute_url_map" "api_gateway_http_redirect" {
  name = "sadp-api-gateway-http-redirect"

  default_url_redirect {
    https_redirect         = true
    redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
    strip_query            = false
  }
}

resource "google_compute_target_http_proxy" "api_gateway_http_proxy" {
  name    = "sadp-api-gateway-http-proxy"
  url_map = google_compute_url_map.api_gateway_http_redirect.id
}

resource "google_compute_global_forwarding_rule" "api_gateway_http_forwarding_rule" {
  name                  = "sadp-api-gateway-http-forwarding-rule"
  ip_protocol           = "TCP"
  load_balancing_scheme = "EXTERNAL"
  port_range            = "80"
  target                = google_compute_target_http_proxy.api_gateway_http_proxy.id
  ip_address            = google_compute_global_address.api_gateway_ip.id
}

# Cloud Endpoints for enhanced API management
resource "google_endpoints_service" "sadp_endpoints" {
  service_name   = var.api_domain
  project        = var.project_id
  openapi_config = templatefile("${path.module}/sadp-api.yaml", {
    PROJECT_ID = var.project_id
    REGION     = var.region
  })
}

# IAM for API Gateway
resource "google_api_gateway_api_iam_binding" "api_users" {
  provider = google-beta
  api      = google_api_gateway_api.sadp_api.api_id
  role     = "roles/apigateway.viewer"
  
  members = [
    "serviceAccount:${var.agent_runtime_sa_email}",
    "serviceAccount:${var.evaluation_sa_email}",
    "serviceAccount:${var.development_sa_email}",
    "serviceAccount:${var.monitoring_sa_email}",
  ]
}

# Monitoring for API Gateway
resource "google_monitoring_alert_policy" "api_gateway_error_rate" {
  display_name = "API Gateway High Error Rate"
  description  = "Alert when API Gateway error rate exceeds threshold"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "High error rate"

    condition_threshold {
      filter = <<EOF
resource.type="api_gateway"
AND metric.type="apigateway.googleapis.com/api/request_count"
AND metric.labels.response_code_class!="2xx"
EOF
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.05 # 5% error rate

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_MEAN"
      }
    }
  }

  notification_channels = var.notification_channels

  alert_strategy {
    auto_close = "1800s"
  }

  documentation {
    content = "API Gateway is experiencing high error rates. Check service health and logs."
  }
}

resource "google_monitoring_alert_policy" "api_gateway_latency" {
  display_name = "API Gateway High Latency"
  description  = "Alert when API Gateway latency exceeds threshold"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "High latency"

    condition_threshold {
      filter = <<EOF
resource.type="api_gateway"
AND metric.type="apigateway.googleapis.com/api/request_latencies"
EOF
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5000 # 5 seconds

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_PERCENTILE_95"
      }
    }
  }

  notification_channels = var.notification_channels

  alert_strategy {
    auto_close = "1800s"
  }

  documentation {
    content = "API Gateway is experiencing high latency. Check backend service health."
  }
}

# Outputs
output "api_gateway_url" {
  value = "https://${google_api_gateway_gateway.sadp_gateway.default_hostname}"
}

output "api_gateway_ip" {
  value = google_compute_global_address.api_gateway_ip.address
}

output "api_domain" {
  value = var.api_domain
}