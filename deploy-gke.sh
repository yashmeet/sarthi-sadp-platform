#!/bin/bash

# SADP Automatic GKE Deployment Script
# This script deploys the complete SADP platform to Google Kubernetes Engine

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
CLUSTER_NAME=${CLUSTER_NAME:-"sarthi-gke-cluster"}
REGISTRY_URL="$REGION-docker.pkg.dev/$PROJECT_ID/sarthi-services"

echo -e "${BLUE}🚀 Starting SADP GKE Deployment${NC}"
echo -e "${BLUE}Project: $PROJECT_ID${NC}"
echo -e "${BLUE}Region: $REGION${NC}"
echo -e "${BLUE}Cluster: $CLUSTER_NAME${NC}"

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}📋 Checking prerequisites...${NC}"
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}❌ gcloud CLI is not installed${NC}"
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl is not installed${NC}"
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
        container.googleapis.com \
        containerregistry.googleapis.com \
        artifactregistry.googleapis.com \
        cloudbuild.googleapis.com \
        firestore.googleapis.com \
        secretmanager.googleapis.com \
        pubsub.googleapis.com \
        run.googleapis.com \
        aiplatform.googleapis.com
    
    # Install GKE auth plugin (try with current user first)
    if ! gcloud components list --filter="id:gke-gcloud-auth-plugin" --format="value(state.name)" | grep -q "Installed"; then
        echo "Installing GKE auth plugin..."
        if ! gcloud components install gke-gcloud-auth-plugin --quiet; then
            echo -e "${YELLOW}⚠️ Could not install GKE auth plugin with current permissions${NC}"
            echo -e "${YELLOW}Please run as administrator or install manually${NC}"
        fi
    fi
    
    echo -e "${GREEN}✅ Project setup completed${NC}"
}

# Initialize Firestore
setup_firestore() {
    echo -e "${YELLOW}🔥 Setting up Firestore...${NC}"
    
    # Check if Firestore is already initialized
    if ! gcloud firestore databases describe --region=$REGION &> /dev/null; then
        echo "Creating Firestore database..."
        gcloud firestore databases create --location=$REGION --type=firestore-native || true
    fi
    
    echo -e "${GREEN}✅ Firestore setup completed${NC}"
}

# Get GKE credentials
get_gke_credentials() {
    echo -e "${YELLOW}🔑 Getting GKE credentials...${NC}"
    
    # Get cluster credentials
    gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT_ID
    
    # Verify connection
    kubectl cluster-info
    
    echo -e "${GREEN}✅ GKE credentials obtained${NC}"
}

# Build and push Docker images
build_and_push_images() {
    echo -e "${YELLOW}🐳 Building and pushing Docker images...${NC}"
    
    # Configure Docker for Artifact Registry
    gcloud auth configure-docker $REGION-docker.pkg.dev --quiet
    
    # Build Agent Runtime Service
    echo "Building agent-runtime image..."
    cd services/agent-runtime
    docker build -t $REGISTRY_URL/agent-runtime:latest .
    docker push $REGISTRY_URL/agent-runtime:latest
    cd ../..
    
    echo -e "${GREEN}✅ Docker images built and pushed${NC}"
}

# Create secrets
create_secrets() {
    echo -e "${YELLOW}🔐 Creating Kubernetes secrets...${NC}"
    
    # Create namespace if it doesn't exist
    kubectl create namespace sadp --dry-run=client -o yaml | kubectl apply -f -
    
    # Generate random JWT secret if not exists
    JWT_SECRET=$(openssl rand -hex 32)
    
    # Create database connection secret
    kubectl create secret generic database-credentials \
        --from-literal=connection-string="postgresql+asyncpg://postgres:changeme@cloud-sql-proxy:5432/sarthi" \
        --namespace=sadp \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create JWT secret
    kubectl create secret generic jwt-secret \
        --from-literal=secret-key="$JWT_SECRET" \
        --namespace=sadp \
        --dry-run=client -o yaml | kubectl apply -f -
    
    echo -e "${GREEN}✅ Secrets created${NC}"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    echo -e "${YELLOW}☸️ Deploying to Kubernetes...${NC}"
    
    # Apply all Kubernetes manifests
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/service-accounts.yaml
    kubectl apply -f k8s/secrets.yaml
    kubectl apply -f k8s/agent-runtime-deployment.yaml
    kubectl apply -f k8s/ingress.yaml
    
    echo -e "${GREEN}✅ Kubernetes deployment completed${NC}"
}

# Wait for deployment
wait_for_deployment() {
    echo -e "${YELLOW}⏳ Waiting for deployment to be ready...${NC}"
    
    # Wait for agent-runtime deployment
    kubectl wait --for=condition=available --timeout=300s deployment/agent-runtime -n sadp
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready --timeout=300s pod -l app=agent-runtime -n sadp
    
    echo -e "${GREEN}✅ Deployment is ready${NC}"
}

# Initialize marketplace data
initialize_marketplace() {
    echo -e "${YELLOW}🏪 Initializing Agent Marketplace...${NC}"
    
    # Get pod name
    POD_NAME=$(kubectl get pods -l app=agent-runtime -n sadp -o jsonpath="{.items[0].metadata.name}")
    
    # Copy initialization script to pod
    kubectl cp scripts/initialize_marketplace.py sadp/$POD_NAME:/tmp/initialize_marketplace.py
    
    # Run initialization
    kubectl exec -n sadp $POD_NAME -- python /tmp/initialize_marketplace.py
    
    echo -e "${GREEN}✅ Marketplace initialized${NC}"
}

# Verify deployment
verify_deployment() {
    echo -e "${YELLOW}🔍 Verifying deployment...${NC}"
    
    # Check pods
    echo "Pods:"
    kubectl get pods -n sadp
    
    # Check services
    echo "Services:"
    kubectl get services -n sadp
    
    # Check ingress
    echo "Ingress:"
    kubectl get ingress -n sadp
    
    # Get external IP
    EXTERNAL_IP=$(kubectl get service agent-runtime-service -n sadp -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -n "$EXTERNAL_IP" ]; then
        echo -e "${GREEN}🌐 Service available at: http://$EXTERNAL_IP${NC}"
    else
        echo -e "${YELLOW}⚠️ External IP not yet assigned. Check with: kubectl get services -n sadp${NC}"
    fi
    
    # Health check
    echo "Performing health check..."
    if kubectl exec -n sadp deployment/agent-runtime -- curl -f http://localhost:8000/health; then
        echo -e "${GREEN}✅ Health check passed${NC}"
    else
        echo -e "${RED}❌ Health check failed${NC}"
    fi
    
    echo -e "${GREEN}✅ Deployment verification completed${NC}"
}

# Cleanup function
cleanup() {
    if [ "$1" = "true" ]; then
        echo -e "${YELLOW}🧹 Cleaning up failed deployment...${NC}"
        kubectl delete namespace sadp --ignore-not-found=true
        docker rmi $REGISTRY_URL/agent-runtime:latest || true
    fi
}

# Main deployment function
main() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}        SADP GKE Deployment Script        ${NC}"
    echo -e "${BLUE}============================================${NC}"
    
    # Set up error handling
    trap 'cleanup true' ERR
    
    check_prerequisites
    setup_project
    setup_firestore
    get_gke_credentials
    build_and_push_images
    create_secrets
    deploy_kubernetes
    wait_for_deployment
    initialize_marketplace
    verify_deployment
    
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}🎉 SADP Deployment Completed Successfully!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}Next Steps:${NC}"
    echo -e "${GREEN}1. Configure DNS for your domain${NC}"
    echo -e "${GREEN}2. Set up SSL certificates${NC}"
    echo -e "${GREEN}3. Run the demo application${NC}"
    echo -e "${GREEN}4. Load sample data${NC}"
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo "- View logs: kubectl logs -f deployment/agent-runtime -n sadp"
    echo "- Check status: kubectl get all -n sadp"
    echo "- Port forward: kubectl port-forward svc/agent-runtime-service 8000:80 -n sadp"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "cleanup")
        echo -e "${YELLOW}🧹 Cleaning up deployment...${NC}"
        kubectl delete namespace sadp --ignore-not-found=true
        echo -e "${GREEN}✅ Cleanup completed${NC}"
        ;;
    "status")
        echo -e "${BLUE}📊 Deployment Status:${NC}"
        kubectl get all -n sadp
        ;;
    "logs")
        echo -e "${BLUE}📝 Recent logs:${NC}"
        kubectl logs -f deployment/agent-runtime -n sadp --tail=50
        ;;
    "help")
        echo "Usage: $0 [deploy|cleanup|status|logs|help]"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy SADP to GKE (default)"
        echo "  cleanup  - Remove all SADP resources"
        echo "  status   - Show deployment status"
        echo "  logs     - Show recent logs"
        echo "  help     - Show this help message"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac