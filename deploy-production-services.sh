#!/bin/bash

# Deploy Production SADP Services
# This script deploys all production-ready services with real AI integration

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
        "secretmanager.googleapis.com"
        "aiplatform.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        info "Enabling $api..."
        gcloud services enable $api --quiet
    done
    
    log "APIs enabled ✓"
}

# Create secrets if they don't exist
create_secrets() {
    log "Creating secrets in Secret Manager..."
    
    secrets=(
        "GEMINI_API_KEY"
        "OPENAI_API_KEY"
        "JWT_SECRET"
        "API_ENCRYPTION_KEY"
    )
    
    for secret in "${secrets[@]}"; do
        if ! gcloud secrets describe $secret >/dev/null 2>&1; then
            info "Creating secret $secret..."
            echo "placeholder_value_change_me" | gcloud secrets create $secret --data-file=-
            warn "Please update secret $secret with actual value: gcloud secrets versions add $secret --data-file=-"
        else
            info "Secret $secret already exists"
        fi
    done
    
    log "Secrets created ✓"
}

# Deploy service
deploy_service() {
    local service_name=$1
    local service_dir=$2
    local dockerfile=${3:-"Dockerfile"}
    local port=${4:-8080}
    
    log "Deploying $service_name..."
    
    # Check if service directory exists
    if [ ! -d "$service_dir" ]; then
        error "Service directory $service_dir not found"
        return 1
    fi
    
    # Create build context that includes common services
    local temp_dir=$(mktemp -d)
    
    # Copy service files
    cp -r "$service_dir"/* "$temp_dir/"
    
    # Copy common services
    if [ -d "./services/common" ]; then
        cp -r "./services/common" "$temp_dir/"
    fi
    
    # Create Dockerfile if using production version
    if [ "$dockerfile" = "Dockerfile.production" ] && [ ! -f "$service_dir/$dockerfile" ]; then
        cat > "$temp_dir/Dockerfile" << EOF
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir -r requirements_production.txt

# Copy common services
COPY common/ ./common/

# Copy source code
COPY src/ ./

# Set environment variables
ENV PORT=$port
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE $port

# Start the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$port"]
EOF
    fi
    
    # Build and deploy to Cloud Run
    info "Building and deploying $service_name to Cloud Run..."
    gcloud run deploy $service_name \
        --source "$temp_dir" \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --service-account $SERVICE_ACCOUNT \
        --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID" \
        --set-env-vars "GCP_REGION=$REGION" \
        --memory 2Gi \
        --cpu 2 \
        --timeout 900 \
        --concurrency 100 \
        --max-instances 10 \
        --quiet
    
    # Clean up temp directory
    rm -rf "$temp_dir"
    
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

# Deploy production services
deploy_production_services() {
    log "Deploying production SADP services..."
    
    # Deploy Authentication Service
    if [ -d "./services/auth-service" ]; then
        deploy_service "sadp-auth-service" "./services/auth-service"
    else
        warn "Authentication service directory not found, skipping..."
    fi
    
    # Deploy PHI Protection Service
    if [ -d "./services/phi-protection" ]; then
        deploy_service "sadp-phi-protection" "./services/phi-protection"
    else
        warn "PHI protection service directory not found, skipping..."
    fi
    
    # Deploy Audit Service
    if [ -d "./services/audit-service" ]; then
        deploy_service "sadp-audit-service" "./services/audit-service"
    else
        warn "Audit service directory not found, skipping..."
    fi
    
    # Deploy Production Prompt Optimization
    if [ -d "./services/prompt-optimization" ]; then
        deploy_service "sadp-prompt-optimization-prod" "./services/prompt-optimization" "Dockerfile.production"
    else
        warn "Prompt optimization service directory not found, skipping..."
    fi
    
    # Deploy existing enhanced services
    if [ -d "./services/poml-orchestrator" ]; then
        deploy_service "sadp-poml-orchestrator-prod" "./services/poml-orchestrator"
    fi
    
    if [ -d "./services/learning-pipeline" ]; then
        deploy_service "sadp-learning-pipeline-prod" "./services/learning-pipeline"
    fi
    
    if [ -d "./services/deployment-manager" ]; then
        deploy_service "sadp-deployment-manager-prod" "./services/deployment-manager"
    fi
    
    log "Production services deployed ✓"
}

# Update API Gateway for production
update_api_gateway_production() {
    log "Updating API Gateway for production services..."
    
    # Create production API gateway config
    cat > api-gateway-production.yaml << 'EOF'
swagger: '2.0'
info:
  title: Sarthi AI Agent Development Platform API - Production
  description: |
    Production API for SADP with real AI integration, PHI protection, and audit logging.
  version: '3.0.0'
  contact:
    name: SADP API Support
    email: support@sarthi.ai

host: sadp-gateway-4jhmplb8.uc.gateway.dev
basePath: /api/v3

schemes:
  - https

consumes:
  - application/json

produces:
  - application/json

securityDefinitions:
  api_key:
    type: apiKey
    name: X-API-Key
    in: header
    description: API key for service access

security:
  - api_key: []

paths:
  # Health check
  /health:
    get:
      summary: Production health check
      description: Check all production services health
      operationId: healthCheck
      security:
        - api_key: []
      responses:
        200:
          description: Services are healthy

  # Authentication
  /auth/login:
    post:
      summary: User login
      operationId: userLogin
      security: []
      x-google-backend:
        address: https://sadp-auth-service-355881591332.us-central1.run.app
        path_translation: APPEND_PATH_TO_ADDRESS
      responses:
        200:
          description: Login successful

  /auth/validate:
    get:
      summary: Validate API key
      operationId: validateApiKey
      x-google-backend:
        address: https://sadp-auth-service-355881591332.us-central1.run.app
        path_translation: APPEND_PATH_TO_ADDRESS
      responses:
        200:
          description: API key valid

  # PHI Protection
  /phi/detect:
    post:
      summary: Detect PHI in text
      operationId: detectPHI
      x-google-backend:
        address: https://sadp-phi-protection-355881591332.us-central1.run.app
        path_translation: APPEND_PATH_TO_ADDRESS
      responses:
        200:
          description: PHI detection result

  /phi/sanitize:
    post:
      summary: Sanitize PHI in text
      operationId: sanitizePHI
      x-google-backend:
        address: https://sadp-phi-protection-355881591332.us-central1.run.app
        path_translation: APPEND_PATH_TO_ADDRESS
      responses:
        200:
          description: PHI sanitization result

  # Audit
  /audit/log:
    post:
      summary: Create audit log
      operationId: createAuditLog
      x-google-backend:
        address: https://sadp-audit-service-355881591332.us-central1.run.app
        path_translation: APPEND_PATH_TO_ADDRESS
      responses:
        200:
          description: Audit log created

  # Production Prompt Optimization
  /optimize/automedprompt:
    post:
      summary: Production AutoMedPrompt optimization
      operationId: optimizeAutoMedPromptProd
      x-google-backend:
        address: https://sadp-prompt-optimization-prod-355881591332.us-central1.run.app
        path_translation: APPEND_PATH_TO_ADDRESS
      responses:
        200:
          description: Optimization result

  /optimize/jobs:
    get:
      summary: List optimization jobs
      operationId: listOptimizationJobs
      x-google-backend:
        address: https://sadp-prompt-optimization-prod-355881591332.us-central1.run.app
        path_translation: APPEND_PATH_TO_ADDRESS
      responses:
        200:
          description: Optimization jobs list
EOF
    
    # Create API config
    info "Creating production API configuration..."
    gcloud api-gateway api-configs create sadp-production-config \
        --api=sadp-api \
        --openapi-spec=api-gateway-production.yaml \
        --backend-auth-service-account=$SERVICE_ACCOUNT \
        --quiet
    
    # Wait for config to be ready
    while [ "$(gcloud api-gateway api-configs describe sadp-production-config --api=sadp-api --format='value(state)' 2>/dev/null)" != "ACTIVE" ]; do
        echo "Waiting for API config..."
        sleep 10
    done
    
    # Update gateway
    info "Updating API Gateway..."
    gcloud api-gateway gateways update sadp-gateway \
        --api=sadp-api \
        --api-config=sadp-production-config \
        --location=$REGION \
        --quiet
    
    log "API Gateway updated for production ✓"
}

# Create integration test script
create_integration_tests() {
    log "Creating integration test script..."
    
    cat > run-integration-tests.sh << 'EOF'
#!/bin/bash

# Run Integration Tests for Production SADP

echo "🧪 Running SADP Production Integration Tests"

# Check if Python and required packages are available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python3."
    exit 1
fi

# Install test dependencies
pip install httpx pytest asyncio

# Set test URLs based on deployed services
export AUTH_SERVICE_URL="https://sadp-auth-service-355881591332.us-central1.run.app"
export PHI_PROTECTION_URL="https://sadp-phi-protection-355881591332.us-central1.run.app"
export AUDIT_SERVICE_URL="https://sadp-audit-service-355881591332.us-central1.run.app"
export PROMPT_OPTIMIZATION_URL="https://sadp-prompt-optimization-prod-355881591332.us-central1.run.app"

# Run integration tests
python3 tests/integration/test_production_system.py

echo "✅ Integration tests completed"
EOF
    
    chmod +x run-integration-tests.sh
    
    log "Integration test script created ✓"
}

# Print deployment summary
print_summary() {
    log "Production Deployment Summary"
    echo "============================"
    
    # List deployed services
    echo "Production Services:"
    if [ -f "sadp-auth-service-url.txt" ]; then
        echo "  • Authentication Service: $(cat sadp-auth-service-url.txt)"
    fi
    if [ -f "sadp-phi-protection-url.txt" ]; then
        echo "  • PHI Protection Service: $(cat sadp-phi-protection-url.txt)"
    fi
    if [ -f "sadp-audit-service-url.txt" ]; then
        echo "  • Audit Service: $(cat sadp-audit-service-url.txt)"
    fi
    if [ -f "sadp-prompt-optimization-prod-url.txt" ]; then
        echo "  • Production Prompt Optimization: $(cat sadp-prompt-optimization-prod-url.txt)"
    fi
    if [ -f "sadp-poml-orchestrator-prod-url.txt" ]; then
        echo "  • Production POML Orchestrator: $(cat sadp-poml-orchestrator-prod-url.txt)"
    fi
    
    echo ""
    echo "API Gateway: https://sadp-gateway-4jhmplb8.uc.gateway.dev/api/v3"
    
    echo ""
    echo "Next Steps:"
    echo "1. Update secrets with actual values:"
    echo "   gcloud secrets versions add GEMINI_API_KEY --data-file=-"
    echo "   gcloud secrets versions add JWT_SECRET --data-file=-"
    echo "2. Run integration tests: ./run-integration-tests.sh"
    echo "3. Test API endpoints with valid API keys"
    echo "4. Monitor service health and logs"
    echo ""
    echo "Production Features:"
    echo "• Real AI integration (Gemini, OpenAI, Vertex AI)"
    echo "• HIPAA-compliant PHI protection"
    echo "• Comprehensive audit logging"
    echo "• Production-grade error handling"
    echo "• Firestore persistence"
    echo "• Authentication and authorization"
    echo ""
    
    log "Production deployment completed successfully! 🎉"
}

# Main execution
main() {
    log "Starting SADP Production Services Deployment"
    
    check_prerequisites
    enable_apis
    create_secrets
    deploy_production_services
    update_api_gateway_production
    create_integration_tests
    print_summary
}

# Run main function
main "$@"