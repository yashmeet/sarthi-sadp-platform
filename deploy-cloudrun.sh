#!/bin/bash

# SADP Cloud Run Deployment Script
# Simpler alternative to GKE - deploys SADP to Google Cloud Run

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=${PROJECT_ID:-"sarthi-patient-experience-hub"}
REGION=${REGION:-"us-central1"}
SERVICE_NAME=${SERVICE_NAME:-"sadp-agent-runtime"}

echo -e "${BLUE}🚀 Starting SADP Cloud Run Deployment${NC}"
echo -e "${BLUE}Project: $PROJECT_ID${NC}"
echo -e "${BLUE}Region: $REGION${NC}"
echo -e "${BLUE}Service: $SERVICE_NAME${NC}"

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}📋 Checking prerequisites...${NC}"
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}❌ gcloud CLI is not installed${NC}"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed${NC}"
        exit 1
    fi
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        echo -e "${RED}❌ Not authenticated with gcloud${NC}"
        echo "Please run: gcloud auth login"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Set up GCP project
setup_project() {
    echo -e "${YELLOW}🔧 Setting up GCP project...${NC}"
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    # Enable required APIs
    echo "Enabling required APIs..."
    gcloud services enable \
        run.googleapis.com \
        cloudbuild.googleapis.com \
        artifactregistry.googleapis.com \
        firestore.googleapis.com \
        secretmanager.googleapis.com \
        pubsub.googleapis.com \
        sqladmin.googleapis.com \
        aiplatform.googleapis.com \
        storage.googleapis.com
    
    echo -e "${GREEN}✅ Project setup completed${NC}"
}

# Initialize Firestore
setup_firestore() {
    echo -e "${YELLOW}🔥 Setting up Firestore...${NC}"
    
    # Check if Firestore is already initialized
    if ! gcloud firestore databases describe --region=$REGION &> /dev/null; then
        echo "Creating Firestore database..."
        gcloud firestore databases create --location=$REGION --type=firestore-native || true
        sleep 10  # Wait for database to be ready
    fi
    
    echo -e "${GREEN}✅ Firestore setup completed${NC}"
}

# Create Pub/Sub subscriptions
setup_pubsub() {
    echo -e "${YELLOW}📡 Setting up Pub/Sub...${NC}"
    
    # Create subscriptions for existing topics
    gcloud pubsub subscriptions create agent-runtime-sub --topic=agent-runtime-topic || true
    gcloud pubsub subscriptions create evaluation-sub --topic=evaluation-topic || true
    gcloud pubsub subscriptions create development-sub --topic=development-topic || true
    gcloud pubsub subscriptions create monitoring-sub --topic=monitoring-topic || true
    
    echo -e "${GREEN}✅ Pub/Sub setup completed${NC}"
}

# Create secrets
create_secrets() {
    echo -e "${YELLOW}🔐 Creating Secret Manager secrets...${NC}"
    
    # Generate JWT secret
    JWT_SECRET=$(openssl rand -hex 32)
    
    # Create or update JWT secret
    if gcloud secrets describe jwt-secret &> /dev/null; then
        echo "$JWT_SECRET" | gcloud secrets versions add jwt-secret --data-file=-
    else
        echo "$JWT_SECRET" | gcloud secrets create jwt-secret --data-file=-
    fi
    
    # Create database password secret if it doesn't exist
    if ! gcloud secrets describe sql-password &> /dev/null; then
        echo "changeme123" | gcloud secrets create sql-password --data-file=-
    fi
    
    # Create POML API key secret if it doesn't exist
    if ! gcloud secrets describe poml-api-key &> /dev/null; then
        echo "demo-api-key-$(date +%s)" | gcloud secrets create poml-api-key --data-file=-
    fi
    
    echo -e "${GREEN}✅ Secrets created${NC}"
}

# Deploy to Cloud Run
deploy_service() {
    echo -e "${YELLOW}☁️ Deploying to Cloud Run...${NC}"
    
    # Navigate to service directory
    cd services/agent-runtime
    
    # Deploy using Cloud Build (simplified for initial deployment)
    gcloud run deploy $SERVICE_NAME \
        --source . \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --port 8000 \
        --memory 4Gi \
        --cpu 2 \
        --timeout 900 \
        --concurrency 10 \
        --min-instances 1 \
        --max-instances 10 \
        --set-env-vars="GCP_PROJECT_ID=$PROJECT_ID,GCP_REGION=$REGION,ENVIRONMENT=demo,JWT_SECRET=demo-jwt-secret,POML_API_KEY=demo-api-key"
    
    cd ../..
    
    echo -e "${GREEN}✅ Cloud Run deployment completed${NC}"
}

# Get service URL
get_service_url() {
    echo -e "${YELLOW}🔍 Getting service URL...${NC}"
    
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format "value(status.url)")
    
    echo -e "${GREEN}🌐 Service URL: $SERVICE_URL${NC}"
    export SERVICE_URL
}

# Initialize marketplace data
initialize_marketplace() {
    echo -e "${YELLOW}🏪 Initializing Agent Marketplace...${NC}"
    
    # Wait for service to be ready
    echo "Waiting for service to be ready..."
    sleep 30
    
    # Health check
    echo "Performing health check..."
    if curl -f "$SERVICE_URL/health" &> /dev/null; then
        echo -e "${GREEN}✅ Service is healthy${NC}"
    else
        echo -e "${YELLOW}⚠️ Service health check failed, but continuing...${NC}"
    fi
    
    # Initialize with sample agents (via API calls)
    echo "Initializing sample agents..."
    
    # Create sample agents via API
    curl -X POST "$SERVICE_URL/agents/marketplace/register" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "Document Processor",
            "version": "1.0.0",
            "description": "OCR and document analysis agent",
            "category": "document",
            "author": "SADP Team",
            "auto_load": true,
            "poml_template": "<prompt><system>Extract key information from medical documents</system><task>{{task}}</task></prompt>"
        }' || true
    
    curl -X POST "$SERVICE_URL/agents/marketplace/register" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "Clinical Assistant",
            "version": "1.0.0", 
            "description": "Clinical analysis and recommendations",
            "category": "clinical",
            "author": "SADP Team",
            "auto_load": true,
            "poml_template": "<prompt><system>Provide clinical insights and recommendations</system><context>{{patient_data}}</context><task>{{clinical_task}}</task></prompt>"
        }' || true
    
    echo -e "${GREEN}✅ Marketplace initialized${NC}"
}

# Deploy demo app to App Engine
deploy_demo_app() {
    echo -e "${YELLOW}🎭 Deploying demo application...${NC}"
    
    cd demo-app
    
    # Create app.yaml for App Engine
    cat > app.yaml << EOF
runtime: nodejs18

env_variables:
  NEXT_PUBLIC_API_URL: $SERVICE_URL
  NEXT_PUBLIC_DEMO_MODE: "false"
  NEXT_PUBLIC_GCP_PROJECT: $PROJECT_ID

automatic_scaling:
  min_instances: 1
  max_instances: 10
EOF
    
    # Build the Next.js app
    npm install
    npm run build
    
    # Deploy to App Engine
    gcloud app deploy --quiet
    
    cd ..
    
    # Get App Engine URL
    DEMO_URL=$(gcloud app describe --format="value(defaultHostname)")
    echo -e "${GREEN}🎭 Demo App URL: https://$DEMO_URL${NC}"
    
    echo -e "${GREEN}✅ Demo app deployed${NC}"
}

# Verify deployment
verify_deployment() {
    echo -e "${YELLOW}🔍 Verifying deployment...${NC}"
    
    echo "Service Information:"
    gcloud run services describe $SERVICE_NAME --region $REGION
    
    echo ""
    echo "Testing endpoints..."
    
    # Test health endpoint
    if curl -f "$SERVICE_URL/health"; then
        echo -e "${GREEN}✅ Health endpoint working${NC}"
    else
        echo -e "${RED}❌ Health endpoint failed${NC}"
    fi
    
    # Test agents endpoint
    if curl -f "$SERVICE_URL/agents/supported"; then
        echo -e "${GREEN}✅ Agents endpoint working${NC}"
    else
        echo -e "${RED}❌ Agents endpoint failed${NC}"
    fi
    
    # Test marketplace endpoint
    if curl -f "$SERVICE_URL/agents/marketplace/search"; then
        echo -e "${GREEN}✅ Marketplace endpoint working${NC}"
    else
        echo -e "${YELLOW}⚠️ Marketplace endpoint may not be ready yet${NC}"
    fi
    
    echo -e "${GREEN}✅ Deployment verification completed${NC}"
}

# Show connection info
show_connection_info() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${GREEN}🎉 SADP Cloud Run Deployment Complete!${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
    echo -e "${GREEN}Service Information:${NC}"
    echo -e "${GREEN}• Service Name: $SERVICE_NAME${NC}"
    echo -e "${GREEN}• Region: $REGION${NC}"
    echo -e "${GREEN}• Service URL: $SERVICE_URL${NC}"
    echo ""
    echo -e "${GREEN}API Endpoints:${NC}"
    echo -e "${GREEN}• Health Check: $SERVICE_URL/health${NC}"
    echo -e "${GREEN}• Agent Execution: $SERVICE_URL/agents/{type}/execute${NC}"
    echo -e "${GREEN}• Marketplace: $SERVICE_URL/agents/marketplace${NC}"
    echo -e "${GREEN}• Metrics: $SERVICE_URL/metrics${NC}"
    echo ""
    echo -e "${GREEN}Useful Commands:${NC}"
    echo "• View logs: gcloud run services logs read $SERVICE_NAME --region $REGION"
    echo "• Update service: gcloud run services update $SERVICE_NAME --region $REGION"
    echo "• Delete service: gcloud run services delete $SERVICE_NAME --region $REGION"
    echo ""
    echo -e "${BLUE}Demo App:${NC}"
    echo "• Update demo-app/.env.local with SERVICE_URL"
    echo "• Run locally: cd demo-app && npm run dev"
    echo ""
}

# Cleanup function
cleanup() {
    if [ "$1" = "true" ]; then
        echo -e "${YELLOW}🧹 Cleaning up failed deployment...${NC}"
        gcloud run services delete $SERVICE_NAME --region $REGION --quiet || true
    fi
}

# Main deployment function
main() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}      SADP Cloud Run Deployment Script    ${NC}"
    echo -e "${BLUE}============================================${NC}"
    
    # Set up error handling
    trap 'cleanup true' ERR
    
    check_prerequisites
    setup_project
    setup_firestore
    setup_pubsub
    create_secrets
    deploy_service
    get_service_url
    initialize_marketplace
    verify_deployment
    show_connection_info
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "deploy-demo")
        get_service_url
        deploy_demo_app
        ;;
    "cleanup")
        echo -e "${YELLOW}🧹 Cleaning up deployment...${NC}"
        gcloud run services delete $SERVICE_NAME --region $REGION --quiet
        gcloud app versions delete --quiet $(gcloud app versions list --format="value(version.id)" --limit=1) || true
        echo -e "${GREEN}✅ Cleanup completed${NC}"
        ;;
    "status")
        echo -e "${BLUE}📊 Deployment Status:${NC}"
        gcloud run services describe $SERVICE_NAME --region $REGION
        ;;
    "logs")
        echo -e "${BLUE}📝 Recent logs:${NC}"
        gcloud run services logs read $SERVICE_NAME --region $REGION --limit=50
        ;;
    "url")
        get_service_url
        echo "$SERVICE_URL"
        ;;
    "test")
        get_service_url
        echo -e "${BLUE}🧪 Testing deployment...${NC}"
        echo "Health check:"
        curl -f "$SERVICE_URL/health" || echo "Health check failed"
        echo ""
        echo "Supported agents:"
        curl -f "$SERVICE_URL/agents/supported" || echo "Agents endpoint failed"
        ;;
    "help")
        echo "Usage: $0 [deploy|deploy-demo|cleanup|status|logs|url|test|help]"
        echo ""
        echo "Commands:"
        echo "  deploy      - Deploy SADP to Cloud Run (default)"
        echo "  deploy-demo - Deploy demo app to App Engine"
        echo "  cleanup     - Remove all SADP resources"
        echo "  status      - Show deployment status"
        echo "  logs        - Show recent logs"
        echo "  url         - Get service URL"
        echo "  test        - Test deployment endpoints"
        echo "  help        - Show this help message"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac