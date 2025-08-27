#!/bin/bash

# SADP Production Deployment Script
# Deploys all production services to Google Cloud

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"sarthi-patient-experience-hub"}
REGION=${REGION:-"us-central1"}
ENVIRONMENT=${ENVIRONMENT:-"production"}

echo "🚀 Starting SADP Production Deployment"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Environment: $ENVIRONMENT"

# Authenticate with Google Cloud
echo "📝 Authenticating with Google Cloud..."
gcloud auth configure-docker

# Build and push agent runtime image
echo "🔨 Building agent runtime image..."
cd services/agent-runtime

# Create production Dockerfile
cat > Dockerfile.production << EOF
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PORT=8080
ENV ENVIRONMENT=production

# Run production server
CMD ["uvicorn", "src.main_production:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
EOF

# Build image
docker build -f Dockerfile.production -t gcr.io/$PROJECT_ID/sadp-runtime:latest .
docker push gcr.io/$PROJECT_ID/sadp-runtime:latest

# Deploy to Cloud Run
echo "☁️ Deploying to Cloud Run..."
gcloud run deploy sadp-production \
    --image gcr.io/$PROJECT_ID/sadp-runtime:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars "ENVIRONMENT=$ENVIRONMENT,GCP_PROJECT_ID=$PROJECT_ID" \
    --set-secrets "GEMINI_API_KEY=gemini-api-key:latest,JWT_SECRET_KEY=jwt-secret-key:latest" \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 1 \
    --max-instances 100 \
    --concurrency 1000

# Get service URL
SERVICE_URL=$(gcloud run services describe sadp-production --region $REGION --format 'value(status.url)')

# Deploy frontend
echo "🌐 Building frontend..."
cd ../../simple-demo

# Create production nginx config
cat > nginx-production.conf << EOF
server {
    listen 8080;
    server_name _;
    
    root /usr/share/nginx/html;
    index index-production.html;
    
    # API proxy
    location /api/ {
        proxy_pass $SERVICE_URL/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Content-Security-Policy "default-src 'self' https: data: 'unsafe-inline' 'unsafe-eval'" always;
    
    # Compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    location / {
        try_files \$uri \$uri/ /index-production.html;
    }
}
EOF

# Create frontend Dockerfile
cat > Dockerfile.production << EOF
FROM nginx:alpine

# Copy HTML files
COPY index-production.html /usr/share/nginx/html/
COPY poml-studio.js /usr/share/nginx/html/
COPY agent-management-ui.html /usr/share/nginx/html/

# Copy nginx config
COPY nginx-production.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
EOF

# Build and push frontend
docker build -f Dockerfile.production -t gcr.io/$PROJECT_ID/sadp-frontend:latest .
docker push gcr.io/$PROJECT_ID/sadp-frontend:latest

# Deploy frontend to Cloud Run
gcloud run deploy sadp-frontend \
    --image gcr.io/$PROJECT_ID/sadp-frontend:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 256Mi \
    --cpu 1 \
    --min-instances 1 \
    --max-instances 10

# Create Cloud Scheduler jobs for maintenance
echo "⏰ Setting up Cloud Scheduler..."
gcloud scheduler jobs create http cleanup-executions \
    --location $REGION \
    --schedule "0 2 * * *" \
    --uri "$SERVICE_URL/maintenance/cleanup" \
    --http-method POST \
    --headers "Authorization=Bearer $(gcloud auth print-access-token)"

# Create monitoring dashboard
echo "📊 Creating monitoring dashboard..."
cat > monitoring-dashboard.json << EOF
{
  "displayName": "SADP Production Dashboard",
  "dashboardFilters": [],
  "gridLayout": {
    "widgets": [
      {
        "title": "Request Rate",
        "xyChart": {
          "dataSets": [{
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\""
              }
            }
          }]
        }
      },
      {
        "title": "Request Latency",
        "xyChart": {
          "dataSets": [{
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_latencies\""
              }
            }
          }]
        }
      },
      {
        "title": "Error Rate",
        "xyChart": {
          "dataSets": [{
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.label.response_code_class=\"5xx\""
              }
            }
          }]
        }
      }
    ]
  }
}
EOF

gcloud monitoring dashboards create --config-from-file=monitoring-dashboard.json

# Create alerting policies
echo "🚨 Setting up alerting..."
gcloud alpha monitoring policies create \
    --notification-channels=$(gcloud alpha monitoring channels list --filter="displayName='Email'" --format="value(name)") \
    --display-name="High Error Rate" \
    --condition-display-name="5xx errors > 1%" \
    --condition-threshold-value=0.01 \
    --condition-threshold-duration=60s

# Output deployment information
echo ""
echo "✅ Deployment Complete!"
echo ""
echo "📌 Service URLs:"
echo "   Backend API: $SERVICE_URL"
FRONTEND_URL=$(gcloud run services describe sadp-frontend --region $REGION --format 'value(status.url)')
echo "   Frontend: $FRONTEND_URL"
echo ""
echo "📊 Monitoring:"
echo "   Dashboard: https://console.cloud.google.com/monitoring/dashboards"
echo "   Logs: https://console.cloud.google.com/logs"
echo ""
echo "🔑 Next Steps:"
echo "   1. Set up custom domain (optional)"
echo "   2. Configure identity platform for OAuth"
echo "   3. Set up backup and disaster recovery"
echo "   4. Configure cost alerts"
echo ""
echo "🎉 SADP is now live in production!"