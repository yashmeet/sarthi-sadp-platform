#!/bin/bash

# GKE-Specific Verification Script for SADP
# Verifies Kubernetes deployment status and pod health

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_ID=${PROJECT_ID:-"sarthi-patient-experience-hub"}
REGION=${REGION:-"us-central1"}
CLUSTER_NAME=${CLUSTER_NAME:-"sarthi-gke-cluster"}
NAMESPACE=${NAMESPACE:-"sadp"}

echo -e "${BLUE}🔍 SADP GKE Deployment Verification${NC}"
echo -e "${BLUE}Project: $PROJECT_ID${NC}"
echo -e "${BLUE}Cluster: $CLUSTER_NAME${NC}"
echo -e "${BLUE}Namespace: $NAMESPACE${NC}"
echo ""

# Function to check kubectl connection
check_kubectl_connection() {
    echo -e "${YELLOW}📡 Checking kubectl connection...${NC}"
    
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}❌ Cannot connect to cluster${NC}"
        echo "Run: gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT_ID"
        exit 1
    fi
    
    echo -e "${GREEN}✅ kubectl connected to cluster${NC}"
}

# Function to verify namespace
verify_namespace() {
    echo -e "${YELLOW}🏗️ Verifying namespace...${NC}"
    
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        echo -e "${GREEN}✅ Namespace '$NAMESPACE' exists${NC}"
    else
        echo -e "${RED}❌ Namespace '$NAMESPACE' not found${NC}"
        exit 1
    fi
}

# Function to check pod status
check_pod_status() {
    echo -e "${YELLOW}🐳 Checking pod status...${NC}"
    
    # Get pod information
    PODS=$(kubectl get pods -n $NAMESPACE -o json)
    TOTAL_PODS=$(echo "$PODS" | jq '.items | length')
    RUNNING_PODS=$(echo "$PODS" | jq '.items[] | select(.status.phase == "Running") | .metadata.name' | wc -l)
    PENDING_PODS=$(echo "$PODS" | jq '.items[] | select(.status.phase == "Pending") | .metadata.name' | wc -l)
    FAILED_PODS=$(echo "$PODS" | jq '.items[] | select(.status.phase == "Failed") | .metadata.name' | wc -l)
    
    echo "   Total Pods: $TOTAL_PODS"
    echo "   Running: $RUNNING_PODS"
    echo "   Pending: $PENDING_PODS"
    echo "   Failed: $FAILED_PODS"
    
    if [ "$RUNNING_PODS" -gt 0 ] && [ "$FAILED_PODS" -eq 0 ]; then
        echo -e "${GREEN}✅ Pod status healthy${NC}"
    elif [ "$PENDING_PODS" -gt 0 ]; then
        echo -e "${YELLOW}⚠️ Some pods still pending${NC}"
        kubectl get pods -n $NAMESPACE
    else
        echo -e "${RED}❌ Pod status unhealthy${NC}"
        kubectl get pods -n $NAMESPACE
        exit 1
    fi
}

# Function to check service status
check_services() {
    echo -e "${YELLOW}🌐 Checking services...${NC}"
    
    SERVICES=$(kubectl get services -n $NAMESPACE -o json)
    SERVICE_COUNT=$(echo "$SERVICES" | jq '.items | length')
    
    echo "   Total Services: $SERVICE_COUNT"
    
    if [ "$SERVICE_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✅ Services configured${NC}"
        
        # Check for external IPs
        EXTERNAL_IPS=$(kubectl get services -n $NAMESPACE -o jsonpath='{.items[*].status.loadBalancer.ingress[*].ip}')
        if [ -n "$EXTERNAL_IPS" ]; then
            echo "   External IPs: $EXTERNAL_IPS"
        else
            echo -e "${YELLOW}⚠️ No external IPs assigned yet${NC}"
        fi
    else
        echo -e "${RED}❌ No services found${NC}"
    fi
}

# Function to check secrets
check_secrets() {
    echo -e "${YELLOW}🔐 Checking secrets...${NC}"
    
    SECRET_COUNT=$(kubectl get secrets -n $NAMESPACE --no-headers | wc -l)
    echo "   Total Secrets: $SECRET_COUNT"
    
    # Check for specific secrets
    REQUIRED_SECRETS=("database-credentials" "jwt-secret")
    
    for secret in "${REQUIRED_SECRETS[@]}"; do
        if kubectl get secret "$secret" -n $NAMESPACE &> /dev/null; then
            echo -e "   ✅ $secret exists"
        else
            echo -e "   ❌ $secret missing"
        fi
    done
    
    echo -e "${GREEN}✅ Secrets verification completed${NC}"
}

# Function to check ingress
check_ingress() {
    echo -e "${YELLOW}🚪 Checking ingress...${NC}"
    
    INGRESS_COUNT=$(kubectl get ingress -n $NAMESPACE --no-headers 2>/dev/null | wc -l)
    echo "   Total Ingress: $INGRESS_COUNT"
    
    if [ "$INGRESS_COUNT" -gt 0 ]; then
        kubectl get ingress -n $NAMESPACE
        echo -e "${GREEN}✅ Ingress configured${NC}"
    else
        echo -e "${YELLOW}⚠️ No ingress found${NC}"
    fi
}

# Function to perform health checks
perform_health_checks() {
    echo -e "${YELLOW}🏥 Performing health checks...${NC}"
    
    # Get the first running pod
    POD_NAME=$(kubectl get pods -n $NAMESPACE -l app=agent-runtime -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [ -n "$POD_NAME" ]; then
        echo "   Testing pod: $POD_NAME"
        
        # Check if health endpoint responds
        if kubectl exec -n $NAMESPACE "$POD_NAME" -- curl -f http://localhost:8000/health &> /dev/null; then
            echo -e "${GREEN}✅ Health endpoint responding${NC}"
        else
            echo -e "${RED}❌ Health endpoint not responding${NC}"
        fi
        
        # Check if agents endpoint responds
        if kubectl exec -n $NAMESPACE "$POD_NAME" -- curl -f http://localhost:8000/agents/supported &> /dev/null; then
            echo -e "${GREEN}✅ Agents endpoint responding${NC}"
        else
            echo -e "${YELLOW}⚠️ Agents endpoint not responding${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️ No running pods found for health checks${NC}"
    fi
}

# Function to check resource usage
check_resource_usage() {
    echo -e "${YELLOW}📊 Checking resource usage...${NC}"
    
    # Get node information
    echo "   Cluster Nodes:"
    kubectl get nodes
    
    echo ""
    echo "   Pod Resource Usage:"
    kubectl top pods -n $NAMESPACE 2>/dev/null || echo "   Metrics not available (metrics-server may not be installed)"
    
    echo ""
    echo "   Node Resource Usage:"
    kubectl top nodes 2>/dev/null || echo "   Metrics not available (metrics-server may not be installed)"
}

# Function to check logs for errors
check_logs() {
    echo -e "${YELLOW}📝 Checking recent logs...${NC}"
    
    # Get pods and check their logs
    PODS=$(kubectl get pods -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}')
    
    for pod in $PODS; do
        echo "   Checking logs for pod: $pod"
        
        # Check for error patterns in recent logs
        ERROR_COUNT=$(kubectl logs "$pod" -n $NAMESPACE --tail=50 | grep -i "error\|exception\|failed" | wc -l)
        
        if [ "$ERROR_COUNT" -eq 0 ]; then
            echo -e "   ✅ No recent errors in $pod"
        else
            echo -e "   ⚠️ Found $ERROR_COUNT potential errors in $pod"
            echo "      Recent errors:"
            kubectl logs "$pod" -n $NAMESPACE --tail=10 | grep -i "error\|exception\|failed" | head -3
        fi
    done
}

# Function to run Python verification script
run_python_verification() {
    echo -e "${YELLOW}🐍 Running detailed Python verification...${NC}"
    
    # Get service URL
    SERVICE_IP=$(kubectl get service agent-runtime-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
    
    if [ -n "$SERVICE_IP" ]; then
        SERVICE_URL="http://$SERVICE_IP"
        echo "   Service URL: $SERVICE_URL"
        
        # Run Python verification if available
        if [ -f "scripts/verify_deployment.py" ]; then
            echo "   Running detailed verification..."
            python3 scripts/verify_deployment.py "$SERVICE_URL" "verification_report.json" || echo "   Python verification completed with warnings"
        else
            echo "   Python verification script not found"
        fi
    else
        echo -e "${YELLOW}⚠️ No external IP available for detailed verification${NC}"
        
        # Try port-forward for local testing
        echo "   Attempting port-forward for local testing..."
        kubectl port-forward svc/agent-runtime-service 8080:80 -n $NAMESPACE &
        PF_PID=$!
        sleep 5
        
        if [ -f "scripts/verify_deployment.py" ]; then
            python3 scripts/verify_deployment.py "http://localhost:8080" "verification_report.json" || echo "   Python verification completed with warnings"
        fi
        
        # Clean up port-forward
        kill $PF_PID 2>/dev/null || true
    fi
}

# Function to generate summary report
generate_summary() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}📋 SADP GKE Verification Summary${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
    echo -e "${GREEN}Cluster Information:${NC}"
    echo "• Project: $PROJECT_ID"
    echo "• Cluster: $CLUSTER_NAME"
    echo "• Namespace: $NAMESPACE"
    echo "• Region: $REGION"
    echo ""
    echo -e "${GREEN}Quick Status Check:${NC}"
    kubectl get all -n $NAMESPACE
    echo ""
    echo -e "${GREEN}Useful Commands:${NC}"
    echo "• View logs: kubectl logs -f deployment/agent-runtime -n $NAMESPACE"
    echo "• Check status: kubectl get all -n $NAMESPACE"
    echo "• Port forward: kubectl port-forward svc/agent-runtime-service 8000:80 -n $NAMESPACE"
    echo "• Scale deployment: kubectl scale deployment agent-runtime --replicas=3 -n $NAMESPACE"
    echo ""
}

# Main verification function
main() {
    check_kubectl_connection
    verify_namespace
    check_pod_status
    check_services
    check_secrets
    check_ingress
    perform_health_checks
    check_resource_usage
    check_logs
    run_python_verification
    generate_summary
    
    echo -e "${GREEN}🎉 GKE verification completed!${NC}"
}

# Handle command line arguments
case "${1:-verify}" in
    "verify")
        main
        ;;
    "quick")
        echo -e "${BLUE}🚀 Quick GKE Status Check${NC}"
        check_kubectl_connection
        kubectl get all -n $NAMESPACE
        ;;
    "logs")
        echo -e "${BLUE}📝 Recent Logs${NC}"
        kubectl logs -f deployment/agent-runtime -n $NAMESPACE --tail=50
        ;;
    "health")
        echo -e "${BLUE}🏥 Health Check Only${NC}"
        check_kubectl_connection
        perform_health_checks
        ;;
    "resources")
        echo -e "${BLUE}📊 Resource Usage${NC}"
        check_kubectl_connection
        check_resource_usage
        ;;
    "help")
        echo "Usage: $0 [verify|quick|logs|health|resources|help]"
        echo ""
        echo "Commands:"
        echo "  verify      - Full verification (default)"
        echo "  quick       - Quick status check"
        echo "  logs        - View recent logs"
        echo "  health      - Health check only"
        echo "  resources   - Resource usage check"
        echo "  help        - Show this help"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac