#!/bin/bash

# Enhanced SADP Production Deployment Script
# Deploys security-hardened infrastructure and all services

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"sarthi-patient-experience-hub"}
REGION=${REGION:-"us-central1"}
ENVIRONMENT=${ENVIRONMENT:-"production"}

echo "🚀 Starting Enhanced SADP Production Deployment"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Environment: $ENVIRONMENT"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if gcloud is installed and authenticated
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed"
        exit 1
    fi
    
    # Check if terraform is installed
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is not installed"
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "."; then
        log_error "Not authenticated with gcloud. Run 'gcloud auth login'"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    log_info "Deploying enhanced security infrastructure..."
    
    # Initialize Terraform
    terraform init
    
    # Create terraform.tfvars if it doesn't exist
    if [ ! -f terraform.tfvars ]; then
        log_warning "Creating terraform.tfvars file"
        cat > terraform.tfvars << EOF
project_id = "$PROJECT_ID"
region = "$REGION"
environment = "$ENVIRONMENT"
enable_vpc_service_controls = false
notification_channels = []
EOF
    fi
    
    # Plan infrastructure
    log_info "Planning infrastructure changes..."
    terraform plan -out=tfplan
    
    # Apply infrastructure
    log_info "Applying infrastructure changes..."
    terraform apply tfplan
    
    log_success "Infrastructure deployment completed"
}

# Configure Docker for Artifact Registry
configure_docker() {
    log_info "Configuring Docker for Artifact Registry..."
    
    gcloud auth configure-docker ${REGION}-docker.pkg.dev
    
    log_success "Docker configuration completed"
}

# Build and push container images
build_and_push_images() {
    log_info "Building and pushing container images..."
    
    # Services to build
    services=("agent-runtime" "evaluation" "development" "monitoring")
    
    for service in "${services[@]}"; do
        log_info "Building $service service..."
        
        cd services/$service
        
        # Build image
        docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/sarthi-services/${service}-service:latest .
        docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/sarthi-services/${service}-service:$(date +%Y%m%d-%H%M%S) .
        
        # Push image
        log_info "Pushing $service service image..."
        docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/sarthi-services/${service}-service:latest
        docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/sarthi-services/${service}-service:$(date +%Y%m%d-%H%M%S)
        
        cd ../..
        
        log_success "$service service image built and pushed"
    done
    
    log_success "All container images built and pushed"
}

# Configure kubectl for GKE
configure_kubectl() {
    log_info "Configuring kubectl for GKE..."
    
    gcloud container clusters get-credentials sarthi-gke-cluster --region $REGION --project $PROJECT_ID
    
    log_success "kubectl configuration completed"
}

# Deploy Kubernetes resources
deploy_kubernetes() {
    log_info "Deploying Kubernetes resources..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace sadp --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy in order: secrets, service accounts, deployments, services
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/secrets.yaml
    kubectl apply -f k8s/service-accounts.yaml
    
    # Deploy all services
    kubectl apply -f k8s/agent-runtime-deployment.yaml
    kubectl apply -f k8s/evaluation-deployment.yaml
    kubectl apply -f k8s/development-deployment.yaml
    kubectl apply -f k8s/ingress.yaml
    
    # Wait for deployments to be ready
    log_info "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/agent-runtime -n sadp
    kubectl wait --for=condition=available --timeout=300s deployment/evaluation-service -n sadp
    kubectl wait --for=condition=available --timeout=300s deployment/development-service -n sadp
    
    log_success "Kubernetes resources deployed successfully"
}

# Set up monitoring and alerting
setup_monitoring() {
    log_info "Setting up monitoring and alerting..."
    
    # Deploy monitoring service
    kubectl apply -f k8s/monitoring-deployment.yaml
    kubectl wait --for=condition=available --timeout=300s deployment/monitoring-service -n sadp
    
    # Create default dashboards and alerts
    log_info "Creating default monitoring configuration..."
    
    # TODO: Add script to create default Grafana dashboards and alert rules
    
    log_success "Monitoring setup completed"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Get service endpoints
    agent_runtime_ip=$(kubectl get service agent-runtime-service -n sadp -o jsonpath='{.spec.clusterIP}')
    evaluation_ip=$(kubectl get service evaluation-service -n sadp -o jsonpath='{.spec.clusterIP}')
    development_ip=$(kubectl get service development-service -n sadp -o jsonpath='{.spec.clusterIP}')
    
    # Health check function
    check_health() {
        local service_name=$1
        local service_ip=$2
        
        log_info "Checking health of $service_name service..."
        
        # Run health check from within cluster
        kubectl run health-check-${service_name} --rm -i --restart=Never --image=curlimages/curl -- \
            curl -f http://${service_ip}:80/health
        
        if [ $? -eq 0 ]; then
            log_success "$service_name service is healthy"
        else
            log_warning "$service_name service health check failed"
        fi
    }
    
    # Run health checks
    check_health "agent-runtime" "$agent_runtime_ip"
    check_health "evaluation" "$evaluation_ip"
    check_health "development" "$development_ip"
    
    log_success "Health checks completed"
}

# Setup CI/CD pipeline
setup_cicd() {
    log_info "Setting up CI/CD pipeline..."
    
    # Create Cloud Build configuration
    if [ ! -f cloudbuild.yaml ]; then
        log_info "Creating Cloud Build configuration..."
        cat > cloudbuild.yaml << 'EOF'
steps:
  # Run tests
  - name: 'python:3.11'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        pip install -r services/agent-runtime/requirements.txt
        python -m pytest services/agent-runtime/tests/ -v
  
  # Build images
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '${_REGION}-docker.pkg.dev/$PROJECT_ID/sarthi-services/agent-runtime:$COMMIT_SHA', 'services/agent-runtime']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '${_REGION}-docker.pkg.dev/$PROJECT_ID/sarthi-services/evaluation-service:$COMMIT_SHA', 'services/evaluation']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '${_REGION}-docker.pkg.dev/$PROJECT_ID/sarthi-services/development-service:$COMMIT_SHA', 'services/development']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '${_REGION}-docker.pkg.dev/$PROJECT_ID/sarthi-services/monitoring-service:$COMMIT_SHA', 'services/monitoring']
  
  # Push images
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${_REGION}-docker.pkg.dev/$PROJECT_ID/sarthi-services/agent-runtime:$COMMIT_SHA']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${_REGION}-docker.pkg.dev/$PROJECT_ID/sarthi-services/evaluation-service:$COMMIT_SHA']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${_REGION}-docker.pkg.dev/$PROJECT_ID/sarthi-services/development-service:$COMMIT_SHA']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${_REGION}-docker.pkg.dev/$PROJECT_ID/sarthi-services/monitoring-service:$COMMIT_SHA']
  
  # Deploy to GKE
  - name: 'gcr.io/cloud-builders/gke-deploy'
    args:
    - run
    - --filename=k8s/
    - --cluster=sarthi-gke-cluster
    - --location=${_REGION}
    - --namespace=sadp

substitutions:
  _REGION: us-central1

options:
  logging: CLOUD_LOGGING_ONLY
EOF
    fi
    
    # Create GitHub Actions workflow (if .github directory exists)
    if [ -d .github ]; then
        mkdir -p .github/workflows
        cat > .github/workflows/deploy.yml << 'EOF'
name: Deploy SADP

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Run tests
      run: |
        pip install -r services/agent-runtime/requirements.txt
        python -m pytest services/agent-runtime/tests/ -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Deploy to Cloud Build
      run: |
        gcloud builds submit --config cloudbuild.yaml
EOF
    fi
    
    log_success "CI/CD pipeline setup completed"
}

# Display deployment summary
show_summary() {
    log_info "Deployment Summary"
    echo "=================================="
    echo "Project: $PROJECT_ID"
    echo "Region: $REGION"
    echo "Environment: $ENVIRONMENT"
    echo ""
    echo "Services Deployed:"
    echo "- Agent Runtime Service"
    echo "- Evaluation Service"
    echo "- Development Service"
    echo "- Monitoring Service"
    echo ""
    echo "Security Features:"
    echo "- Customer-managed encryption (CMEK)"
    echo "- IAM with least privilege"
    echo "- VPC with private networking"
    echo "- Comprehensive audit logging"
    echo "- Secret management with rotation"
    echo ""
    echo "Access Information:"
    kubectl get ingress -n sadp
    echo ""
    log_success "Deployment completed successfully!"
}

# Main deployment flow
main() {
    check_prerequisites
    deploy_infrastructure
    configure_docker
    build_and_push_images
    configure_kubectl
    deploy_kubernetes
    setup_monitoring
    run_health_checks
    setup_cicd
    show_summary
}

# Handle script interruption
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Run main deployment
main