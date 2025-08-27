#!/bin/bash

# Deploy Enhanced POML System - SADP Self-Learning Platform
# This script deploys all new services for the enhanced POML system

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"sarthi-patient-experience-hub"}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_ACCOUNT="sadp-service-account@${PROJECT_ID}.iam.gserviceaccount.com"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if gcloud is installed and authenticated
    if ! command -v gcloud &> /dev/null; then
        error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        error "Please authenticate with gcloud: gcloud auth login"
        exit 1
    fi
    
    # Check if project is set
    CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
    if [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
        warn "Current project is $CURRENT_PROJECT, setting to $PROJECT_ID"
        gcloud config set project $PROJECT_ID
    fi
    
    log "Prerequisites check completed ✓"
}

# Enable required APIs
enable_apis() {
    log "Enabling required Google Cloud APIs..."
    
    apis=(
        "run.googleapis.com"
        "cloudbuild.googleapis.com"
        "firestore.googleapis.com"
        "storage-component.googleapis.com"
        "pubsub.googleapis.com"
        "secretmanager.googleapis.com"
        "apigateway.googleapis.com"
        "servicecontrol.googleapis.com"
        "servicemanagement.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        info "Enabling $api..."
        gcloud services enable $api --quiet
    done
    
    log "APIs enabled ✓"
}

# Create storage buckets if they don't exist
create_storage_buckets() {
    log "Creating storage buckets..."
    
    buckets=(
        "${PROJECT_ID}-poml-templates"
        "${PROJECT_ID}-optimization-results"
        "${PROJECT_ID}-learning-data"
        "${PROJECT_ID}-deployment-artifacts"
    )
    
    for bucket in "${buckets[@]}"; do
        if ! gsutil ls gs://$bucket >/dev/null 2>&1; then
            info "Creating bucket gs://$bucket..."
            gsutil mb -l $REGION gs://$bucket
            gsutil iam ch allUsers:objectViewer gs://$bucket
        else
            info "Bucket gs://$bucket already exists"
        fi
    done
    
    log "Storage buckets created ✓"
}

# Build and deploy service
deploy_service() {
    local service_name=$1
    local service_dir=$2
    local port=${3:-8080}
    
    log "Deploying $service_name..."
    
    # Check if service directory exists
    if [ ! -d "$service_dir" ]; then
        error "Service directory $service_dir not found"
        return 1
    fi
    
    # Build and deploy to Cloud Run
    info "Building and deploying $service_name to Cloud Run..."
    gcloud run deploy $service_name \
        --source $service_dir \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --service-account $SERVICE_ACCOUNT \
        --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID" \
        --set-env-vars "GCP_REGION=$REGION" \
        --set-env-vars "POML_ORCHESTRATOR_URL=https://sadp-poml-orchestrator-xonau6hybq-uc.a.run.app" \
        --set-env-vars "LEARNING_PIPELINE_URL=https://sadp-learning-pipeline-xonau6hybq-uc.a.run.app" \
        --set-env-vars "OPTIMIZATION_SERVICE_URL=https://sadp-prompt-optimization-xonau6hybq-uc.a.run.app" \
        --set-env-vars "KAGGLE_SERVICE_URL=https://sadp-kaggle-integration-xonau6hybq-uc.a.run.app" \
        --set-env-vars "DEPLOYMENT_MANAGER_URL=https://sadp-deployment-manager-xonau6hybq-uc.a.run.app" \
        --memory 1Gi \
        --cpu 1 \
        --timeout 900 \
        --concurrency 100 \
        --max-instances 10 \
        --quiet
    
    if [ $? -eq 0 ]; then
        log "$service_name deployed successfully ✓"
        
        # Get service URL
        SERVICE_URL=$(gcloud run services describe $service_name --region=$REGION --format="value(status.url)")
        echo "$service_name: $SERVICE_URL"
        echo "$SERVICE_URL" > "${service_name}-url.txt"
    else
        error "Failed to deploy $service_name"
        return 1
    fi
}

# Deploy all enhanced services
deploy_enhanced_services() {
    log "Deploying enhanced POML services..."
    
    # Deploy POML Orchestrator
    if [ -d "./services/poml-orchestrator" ]; then
        deploy_service "sadp-poml-orchestrator" "./services/poml-orchestrator"
    else
        warn "POML Orchestrator service directory not found, skipping..."
    fi
    
    # Deploy Learning Pipeline
    if [ -d "./services/learning-pipeline" ]; then
        deploy_service "sadp-learning-pipeline" "./services/learning-pipeline"
    else
        warn "Learning Pipeline service directory not found, skipping..."
    fi
    
    # Deploy Deployment Manager
    if [ -d "./services/deployment-manager" ]; then
        deploy_service "sadp-deployment-manager" "./services/deployment-manager"
    else
        warn "Deployment Manager service directory not found, skipping..."
    fi
    
    log "Enhanced POML services deployed ✓"
}

# Update existing services
update_existing_services() {
    log "Updating existing services..."
    
    # Update Prompt Optimization service (enhanced)
    if [ -d "./services/prompt-optimization" ]; then
        info "Updating Prompt Optimization service with enhanced AutoMedPrompt..."
        deploy_service "sadp-prompt-optimization" "./services/prompt-optimization"
    fi
    
    # Update Kaggle Integration service
    if [ -d "./services/kaggle-integration" ]; then
        info "Updating Kaggle Integration service..."
        deploy_service "sadp-kaggle-integration" "./services/kaggle-integration"
    fi
    
    log "Existing services updated ✓"
}

# Create Firestore indexes
create_firestore_indexes() {
    log "Creating Firestore indexes..."
    
    # Create index configuration file
    cat > firestore.indexes.json << 'EOF'
{
  "indexes": [
    {
      "collectionGroup": "poml_templates_v2",
      "queryScope": "COLLECTION",
      "fields": [
        {"fieldPath": "agent_type", "order": "ASCENDING"},
        {"fieldPath": "medical_domain", "order": "ASCENDING"},
        {"fieldPath": "status", "order": "ASCENDING"},
        {"fieldPath": "updated_at", "order": "DESCENDING"}
      ]
    },
    {
      "collectionGroup": "optimization_jobs",
      "queryScope": "COLLECTION",
      "fields": [
        {"fieldPath": "status", "order": "ASCENDING"},
        {"fieldPath": "strategy", "order": "ASCENDING"},
        {"fieldPath": "started_at", "order": "DESCENDING"}
      ]
    },
    {
      "collectionGroup": "learning_jobs",
      "queryScope": "COLLECTION",
      "fields": [
        {"fieldPath": "agent_type", "order": "ASCENDING"},
        {"fieldPath": "medical_domain", "order": "ASCENDING"},
        {"fieldPath": "status", "order": "ASCENDING"},
        {"fieldPath": "created_at", "order": "DESCENDING"}
      ]
    },
    {
      "collectionGroup": "template_deployments",
      "queryScope": "COLLECTION",
      "fields": [
        {"fieldPath": "template_id", "order": "ASCENDING"},
        {"fieldPath": "status", "order": "ASCENDING"},
        {"fieldPath": "created_at", "order": "DESCENDING"}
      ]
    },
    {
      "collectionGroup": "ab_experiments",
      "queryScope": "COLLECTION",
      "fields": [
        {"fieldPath": "status", "order": "ASCENDING"},
        {"fieldPath": "created_at", "order": "DESCENDING"}
      ]
    }
  ]
}
EOF
    
    # Deploy indexes
    info "Deploying Firestore indexes..."
    gcloud firestore indexes composite create --file=firestore.indexes.json --quiet
    
    # Clean up
    rm -f firestore.indexes.json
    
    log "Firestore indexes created ✓"
}

# Deploy API Gateway
deploy_api_gateway() {
    log "Deploying enhanced API Gateway..."
    
    # Check if the enhanced API gateway config exists
    if [ ! -f "./api-gateway-enhanced.yaml" ]; then
        error "Enhanced API gateway configuration not found"
        return 1
    fi
    
    # Create API config
    info "Creating API configuration..."
    gcloud api-gateway api-configs create sadp-enhanced-config \
        --api=sadp-api \
        --openapi-spec=api-gateway-enhanced.yaml \
        --backend-auth-service-account=$SERVICE_ACCOUNT \
        --quiet
    
    # Update gateway to use new config
    info "Updating API Gateway..."
    gcloud api-gateway gateways update sadp-gateway \
        --api=sadp-api \
        --api-config=sadp-enhanced-config \
        --location=$REGION \
        --quiet
    
    # Get gateway URL
    GATEWAY_URL=$(gcloud api-gateway gateways describe sadp-gateway --location=$REGION --format="value(defaultHostname)")
    
    log "Enhanced API Gateway deployed ✓"
    echo "Gateway URL: https://$GATEWAY_URL"
    echo "https://$GATEWAY_URL" > "api-gateway-url.txt"
}

# Create integration test script
create_integration_tests() {
    log "Creating integration test script..."
    
    cat > test-enhanced-poml-system.sh << 'EOF'
#!/bin/bash

# Integration tests for Enhanced POML System

set -e

GATEWAY_URL="https://sadp-gateway-4jhmplb8.uc.gateway.dev/api/v2"
API_KEY="your-api-key-here"

# Test POML Orchestrator
echo "Testing POML Orchestrator..."
curl -s -H "X-API-Key: $API_KEY" "$GATEWAY_URL/poml/templates" | jq .

# Test Learning Pipeline
echo "Testing Learning Pipeline..."
curl -s -H "X-API-Key: $API_KEY" "$GATEWAY_URL/learning/jobs" | jq .

# Test Optimization Service
echo "Testing Optimization Service..."
curl -s -H "X-API-Key: $API_KEY" "$GATEWAY_URL/optimize/jobs" | jq .

# Test Deployment Manager
echo "Testing Deployment Manager..."
curl -s -H "X-API-Key: $API_KEY" "$GATEWAY_URL/deployments" | jq .

# Test Health Endpoints
echo "Testing health endpoints..."
services=("poml-orchestrator" "learning-pipeline" "prompt-optimization" "deployment-manager" "kaggle-integration")

for service in "${services[@]}"; do
    echo "Checking $service health..."
    curl -s "https://sadp-$service-xonau6hybq-uc.a.run.app/health" | jq .status
done

echo "Integration tests completed!"
EOF
    
    chmod +x test-enhanced-poml-system.sh
    
    log "Integration test script created ✓"
}

# Create monitoring dashboard config
create_monitoring_config() {
    log "Creating monitoring dashboard configuration..."
    
    cat > monitoring-dashboard.json << 'EOF'
{
  "displayName": "SADP Enhanced POML System",
  "mosaicLayout": {
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "POML Template Executions",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE"
                    }
                  }
                }
              }
            ]
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "xPos": 6,
        "widget": {
          "title": "Learning Job Success Rate",
          "scorecard": {
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "resource.type=\"cloud_run_revision\""
              }
            }
          }
        }
      },
      {
        "width": 12,
        "height": 4,
        "yPos": 4,
        "widget": {
          "title": "Service Health Status",
          "text": {
            "content": "Monitor the health of all Enhanced POML services",
            "format": "MARKDOWN"
          }
        }
      }
    ]
  }
}
EOF
    
    log "Monitoring dashboard configuration created ✓"
}

# Print deployment summary
print_summary() {
    log "Deployment Summary"
    echo "==================="
    
    # List deployed services
    echo "Deployed Services:"
    if [ -f "sadp-poml-orchestrator-url.txt" ]; then
        echo "  • POML Orchestrator: $(cat sadp-poml-orchestrator-url.txt)"
    fi
    if [ -f "sadp-learning-pipeline-url.txt" ]; then
        echo "  • Learning Pipeline: $(cat sadp-learning-pipeline-url.txt)"
    fi
    if [ -f "sadp-deployment-manager-url.txt" ]; then
        echo "  • Deployment Manager: $(cat sadp-deployment-manager-url.txt)"
    fi
    if [ -f "sadp-prompt-optimization-url.txt" ]; then
        echo "  • Prompt Optimization: $(cat sadp-prompt-optimization-url.txt)"
    fi
    if [ -f "sadp-kaggle-integration-url.txt" ]; then
        echo "  • Kaggle Integration: $(cat sadp-kaggle-integration-url.txt)"
    fi
    
    echo ""
    if [ -f "api-gateway-url.txt" ]; then
        echo "API Gateway: $(cat api-gateway-url.txt)"
    fi
    
    echo ""
    echo "Next Steps:"
    echo "1. Run integration tests: ./test-enhanced-poml-system.sh"
    echo "2. Set up monitoring dashboards"
    echo "3. Configure KAGGLE_USERNAME and KAGGLE_KEY environment variables"
    echo "4. Create POML templates using the new API endpoints"
    echo ""
    echo "Enhanced POML System Features:"
    echo "• Self-learning with Kaggle datasets"
    echo "• Advanced AutoMedPrompt optimization (genetic algorithms, RL)"
    echo "• Progressive deployment strategies (canary, blue-green, A/B testing)"
    echo "• Automated rollback and health monitoring"
    echo "• POML template versioning and inheritance"
    echo ""
    
    log "Deployment completed successfully! 🎉"
}

# Main execution
main() {
    log "Starting Enhanced POML System Deployment"
    
    check_prerequisites
    enable_apis
    create_storage_buckets
    create_firestore_indexes
    deploy_enhanced_services
    update_existing_services
    deploy_api_gateway
    create_integration_tests
    create_monitoring_config
    print_summary
}

# Run main function
main "$@"