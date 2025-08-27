# SADP Deployment Scripts

This directory contains comprehensive scripts for deploying, initializing, and verifying the SADP platform.

## 🚀 Quick Start

### For Cloud Run (Recommended for demos)
```bash
# Deploy everything
./deploy-cloudrun.sh

# Verify deployment
./scripts/verify_cloudrun.sh

# Test specific endpoints
./scripts/verify_cloudrun.sh health
```

### For GKE (Production)
```bash
# Deploy everything
./deploy-gke.sh

# Verify deployment
./scripts/verify_gke.sh

# Check resource usage
./scripts/verify_gke.sh resources
```

## 📁 Script Overview

### Deployment Scripts
- `deploy-cloudrun.sh` - Simple serverless deployment to Cloud Run
- `deploy-gke.sh` - Full Kubernetes deployment to GKE

### Initialization Scripts
- `initialize_marketplace.py` - Populates agent marketplace with sample agents
- `requirements.txt` - Python dependencies for scripts

### Verification Scripts
- `verify_deployment.py` - Comprehensive Python-based API testing
- `verify_cloudrun.sh` - Cloud Run specific verification
- `verify_gke.sh` - GKE/Kubernetes specific verification

## 🔧 Script Usage

### Python Verification Script
```bash
# Install dependencies
pip install -r scripts/requirements.txt

# Run verification
python scripts/verify_deployment.py https://your-service-url.run.app

# Save detailed report
python scripts/verify_deployment.py https://your-service-url.run.app report.json
```

### Cloud Run Verification
```bash
# Full verification
./scripts/verify_cloudrun.sh

# Quick status check
./scripts/verify_cloudrun.sh quick

# View recent logs
./scripts/verify_cloudrun.sh logs

# Health check only
./scripts/verify_cloudrun.sh health

# Performance metrics
./scripts/verify_cloudrun.sh performance

# Get service URL
./scripts/verify_cloudrun.sh url
```

### GKE Verification
```bash
# Full verification
./scripts/verify_gke.sh

# Quick status check
./scripts/verify_gke.sh quick

# View logs
./scripts/verify_gke.sh logs

# Health checks only
./scripts/verify_gke.sh health

# Resource usage
./scripts/verify_gke.sh resources
```

## 📊 Verification Checks

### API Endpoint Tests
- ✅ Health endpoint (`/health`)
- ✅ Agent listing (`/agents/supported`)
- ✅ Marketplace search (`/agents/marketplace/search`)
- ✅ Metrics endpoint (`/metrics`)
- ✅ POML templates (`/poml/templates`)

### Functional Tests
- ✅ Agent execution with sample data
- ✅ Marketplace agent loading
- ✅ POML A/B test creation
- ✅ Dynamic agent registration

### Infrastructure Tests
- ✅ Service health and readiness
- ✅ Resource usage and performance
- ✅ Log analysis for errors
- ✅ Connected services (Firestore, Pub/Sub, etc.)

## 🎯 Expected Results

### Healthy Deployment
```
📊 Overall Results:
   Total Checks: 12
   ✅ Passed: 10
   ⚠️ Warnings: 2
   ❌ Failed: 0
   📈 Success Rate: 83.3%
   ⚡ Avg Response Time: 0.245s
   🏥 Overall Status: HEALTHY
```

### Service Status
```
🔧 Service Status:
   ✅ Core: pass (2 checks, 0.123s avg)
   ✅ Agents: pass (2 checks, 0.234s avg)
   ✅ Marketplace: pass (4 checks, 0.345s avg)
   ⚠️ POML: warning (2 checks, 0.456s avg)
   ✅ Execution: pass (2 checks, 0.678s avg)
```

## 🚨 Troubleshooting

### Common Issues

1. **Service Not Responding**
   ```bash
   # Check if service is deployed
   gcloud run services list --region=us-central1
   
   # Check service logs
   ./scripts/verify_cloudrun.sh logs
   ```

2. **Marketplace Not Initialized**
   ```bash
   # Run marketplace initialization
   python scripts/initialize_marketplace.py
   ```

3. **Authentication Errors**
   ```bash
   # Re-authenticate
   gcloud auth login
   gcloud config set project sarthi-patient-experience-hub
   ```

4. **GKE Connection Issues**
   ```bash
   # Get cluster credentials
   gcloud container clusters get-credentials sarthi-gke-cluster --region=us-central1
   ```

### Error Codes

- **Exit Code 0**: All checks passed
- **Exit Code 1**: Critical failures (service unreachable)
- **Exit Code 2**: Degraded performance (warnings present)

## 📈 Performance Benchmarks

### Expected Response Times
- Health endpoint: < 0.5s
- Agent listing: < 1.0s
- Agent execution: < 3.0s
- Marketplace search: < 2.0s

### Resource Usage (Cloud Run)
- Memory: < 2GB under normal load
- CPU: < 1 vCPU average
- Cold start: < 5s

### Resource Usage (GKE)
- Memory: < 4GB per pod
- CPU: < 2 vCPU per pod
- Pod startup: < 30s

## 🔄 Automated Monitoring

### Continuous Verification
```bash
# Run verification every 5 minutes
*/5 * * * * /path/to/scripts/verify_cloudrun.sh health >> /var/log/sadp_health.log 2>&1
```

### Alert Integration
```bash
# Example: Send Slack alert on failure
if ! ./scripts/verify_cloudrun.sh health; then
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"SADP health check failed!"}' \
        YOUR_SLACK_WEBHOOK_URL
fi
```

## 📝 Report Generation

### JSON Reports
All Python verification scripts can generate detailed JSON reports:
```json
{
  "summary": {
    "total_checks": 12,
    "passed": 10,
    "warnings": 2,
    "failed": 0,
    "success_rate": 83.3,
    "avg_response_time": 0.245,
    "overall_status": "healthy"
  },
  "services": {...},
  "detailed_results": [...],
  "timestamp": "2024-01-20T15:30:45Z"
}
```

### CSV Export
Convert JSON reports to CSV for analysis:
```bash
# Using jq to extract key metrics
cat report.json | jq -r '.detailed_results[] | [.service, .endpoint, .status, .response_time] | @csv' > metrics.csv
```

## 🛠️ Customization

### Adding Custom Checks
1. Extend `verify_deployment.py` with new test methods
2. Add verification endpoints to your service
3. Update expected benchmarks for your use case

### Environment-Specific Configuration
```bash
# Development
export PROJECT_ID="sarthi-dev"
export REGION="us-west1"

# Staging
export PROJECT_ID="sarthi-staging"
export REGION="us-east1"

# Production
export PROJECT_ID="sarthi-production"
export REGION="us-central1"
```

## 📞 Support

For issues with these scripts:
1. Check the troubleshooting section above
2. Review service logs using the verification scripts
3. Ensure all prerequisites are installed
4. Verify GCP authentication and permissions

## 🔐 Security Notes

- Scripts require appropriate GCP IAM permissions
- Sensitive data is handled through Secret Manager
- No hardcoded credentials in any script
- All network communication uses HTTPS
- Verification scripts do not modify production data