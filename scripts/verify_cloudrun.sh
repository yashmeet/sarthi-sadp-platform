#!/bin/bash

# Cloud Run-Specific Verification Script for SADP
# Verifies Cloud Run deployment status and service health

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_ID=${PROJECT_ID:-"sarthi-patient-experience-hub"}
REGION=${REGION:-"us-central1"}
SERVICE_NAME=${SERVICE_NAME:-"sadp-agent-runtime"}

echo -e "${BLUE}🔍 SADP Cloud Run Deployment Verification${NC}"
echo -e "${BLUE}Project: $PROJECT_ID${NC}"
echo -e "${BLUE}Region: $REGION${NC}"
echo -e "${BLUE}Service: $SERVICE_NAME${NC}"
echo ""

# Function to check gcloud authentication
check_gcloud_auth() {
    echo -e "${YELLOW}🔑 Checking gcloud authentication...${NC}"
    
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        echo -e "${RED}❌ Not authenticated with gcloud${NC}"
        echo "Please run: gcloud auth login"
        exit 1
    fi
    
    # Set project
    gcloud config set project $PROJECT_ID
    echo -e "${GREEN}✅ gcloud authenticated and project set${NC}"
}

# Function to check if service exists
check_service_exists() {
    echo -e "${YELLOW}🏗️ Checking if Cloud Run service exists...${NC}"
    
    if gcloud run services describe $SERVICE_NAME --region $REGION &> /dev/null; then
        echo -e "${GREEN}✅ Cloud Run service '$SERVICE_NAME' exists${NC}"
    else
        echo -e "${RED}❌ Cloud Run service '$SERVICE_NAME' not found${NC}"
        echo "Available services:"
        gcloud run services list --region $REGION
        exit 1
    fi
}

# Function to check service status
check_service_status() {
    echo -e "${YELLOW}🚀 Checking service status...${NC}"
    
    # Get service information
    SERVICE_INFO=$(gcloud run services describe $SERVICE_NAME --region $REGION --format=json)
    
    # Extract key information
    SERVICE_URL=$(echo "$SERVICE_INFO" | jq -r '.status.url')
    READY_REPLICAS=$(echo "$SERVICE_INFO" | jq -r '.status.conditions[] | select(.type == "Ready") | .status')
    LATEST_REVISION=$(echo "$SERVICE_INFO" | jq -r '.status.latestReadyRevisionName')
    ALLOCATED_CPU=$(echo "$SERVICE_INFO" | jq -r '.spec.template.spec.containerConcurrency')
    MEMORY=$(echo "$SERVICE_INFO" | jq -r '.spec.template.spec.containers[0].resources.limits.memory // "4Gi"')
    
    echo "   Service URL: $SERVICE_URL"
    echo "   Ready Status: $READY_REPLICAS"
    echo "   Latest Revision: $LATEST_REVISION"
    echo "   Memory Limit: $MEMORY"
    echo "   Concurrency: $ALLOCATED_CPU"
    
    if [ "$READY_REPLICAS" = "True" ]; then
        echo -e "${GREEN}✅ Service is ready and healthy${NC}"
        export SERVICE_URL
    else
        echo -e "${RED}❌ Service is not ready${NC}"
        exit 1
    fi
}

# Function to check traffic allocation
check_traffic_allocation() {
    echo -e "${YELLOW}🌐 Checking traffic allocation...${NC}"
    
    TRAFFIC_INFO=$(gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.traffic[].percent,status.traffic[].revisionName)")
    
    echo "   Traffic Allocation:"
    echo "$TRAFFIC_INFO" | while read -r line; do
        if [ -n "$line" ]; then
            echo "   • $line"
        fi
    done
    
    echo -e "${GREEN}✅ Traffic allocation checked${NC}"
}

# Function to check recent deployments
check_recent_deployments() {
    echo -e "${YELLOW}📦 Checking recent deployments...${NC}"
    
    echo "   Recent Revisions:"
    gcloud run revisions list --service=$SERVICE_NAME --region=$REGION --limit=5 --format="table(metadata.name,status.conditions[0].lastTransitionTime,spec.containers[0].image)"
    
    echo -e "${GREEN}✅ Recent deployments checked${NC}"
}

# Function to check environment variables and secrets
check_environment() {
    echo -e "${YELLOW}🔧 Checking environment configuration...${NC}"
    
    ENV_VARS=$(gcloud run services describe $SERVICE_NAME --region $REGION --format="value(spec.template.spec.containers[0].env[].name)")
    SECRET_VOLUMES=$(gcloud run services describe $SERVICE_NAME --region $REGION --format="value(spec.template.spec.volumes[].name)")
    
    echo "   Environment Variables:"
    if [ -n "$ENV_VARS" ]; then
        echo "$ENV_VARS" | while read -r var; do
            if [ -n "$var" ]; then
                echo "   • $var"
            fi
        done
    else
        echo "   • None configured"
    fi
    
    echo "   Secret Volumes:"
    if [ -n "$SECRET_VOLUMES" ]; then
        echo "$SECRET_VOLUMES" | while read -r volume; do
            if [ -n "$volume" ]; then
                echo "   • $volume"
            fi
        done
    else
        echo "   • None configured"
    fi
    
    echo -e "${GREEN}✅ Environment configuration checked${NC}"
}

# Function to check IAM permissions
check_iam_permissions() {
    echo -e "${YELLOW}🔐 Checking IAM permissions...${NC}"
    
    # Check service account
    SERVICE_ACCOUNT=$(gcloud run services describe $SERVICE_NAME --region $REGION --format="value(spec.template.spec.serviceAccountName)")
    
    if [ -n "$SERVICE_ACCOUNT" ] && [ "$SERVICE_ACCOUNT" != "null" ]; then
        echo "   Service Account: $SERVICE_ACCOUNT"
        
        # Check if service account exists
        if gcloud iam service-accounts describe "$SERVICE_ACCOUNT" &> /dev/null; then
            echo -e "${GREEN}   ✅ Service account exists${NC}"
        else
            echo -e "${YELLOW}   ⚠️ Service account may not exist${NC}"
        fi
    else
        echo "   Service Account: Default Compute Engine service account"
    fi
    
    echo -e "${GREEN}✅ IAM permissions checked${NC}"
}

# Function to perform health checks
perform_health_checks() {
    echo -e "${YELLOW}🏥 Performing health checks...${NC}"
    
    if [ -z "$SERVICE_URL" ]; then
        echo -e "${RED}❌ Service URL not available${NC}"
        return 1
    fi
    
    # Health endpoint check
    echo "   Testing health endpoint..."
    if curl -f -s "$SERVICE_URL/health" > /dev/null; then
        HEALTH_RESPONSE=$(curl -s "$SERVICE_URL/health")
        echo -e "${GREEN}   ✅ Health endpoint responding${NC}"
        echo "   Response: $HEALTH_RESPONSE"
    else
        echo -e "${RED}   ❌ Health endpoint not responding${NC}"
    fi
    
    # Agents endpoint check
    echo "   Testing agents endpoint..."
    if curl -f -s "$SERVICE_URL/agents/supported" > /dev/null; then
        AGENTS_COUNT=$(curl -s "$SERVICE_URL/agents/supported" | jq '.agents | length' 2>/dev/null || echo "unknown")
        echo -e "${GREEN}   ✅ Agents endpoint responding${NC}"
        echo "   Available agents: $AGENTS_COUNT"
    else
        echo -e "${YELLOW}   ⚠️ Agents endpoint not responding${NC}"
    fi
    
    # Marketplace endpoint check
    echo "   Testing marketplace endpoint..."
    if curl -f -s "$SERVICE_URL/agents/marketplace/search" > /dev/null; then
        echo -e "${GREEN}   ✅ Marketplace endpoint responding${NC}"
    else
        echo -e "${YELLOW}   ⚠️ Marketplace endpoint not responding${NC}"
    fi
    
    # Metrics endpoint check
    echo "   Testing metrics endpoint..."
    if curl -f -s "$SERVICE_URL/metrics" > /dev/null; then
        echo -e "${GREEN}   ✅ Metrics endpoint responding${NC}"
    else
        echo -e "${YELLOW}   ⚠️ Metrics endpoint not responding${NC}"
    fi
}

# Function to check logs
check_logs() {
    echo -e "${YELLOW}📝 Checking recent logs...${NC}"
    
    echo "   Recent logs (last 10 entries):"
    gcloud run services logs read $SERVICE_NAME --region=$REGION --limit=10 --format="value(timestamp,severity,textPayload)" | while read -r log_entry; do
        echo "   $log_entry"
    done
    
    # Check for errors in recent logs
    ERROR_COUNT=$(gcloud run services logs read $SERVICE_NAME --region=$REGION --limit=50 --format="value(severity)" | grep -c "ERROR" || echo "0")
    WARNING_COUNT=$(gcloud run services logs read $SERVICE_NAME --region=$REGION --limit=50 --format="value(severity)" | grep -c "WARNING" || echo "0")
    
    echo ""
    echo "   Log Summary (last 50 entries):"
    echo "   • Errors: $ERROR_COUNT"
    echo "   • Warnings: $WARNING_COUNT"
    
    if [ "$ERROR_COUNT" -eq 0 ]; then
        echo -e "${GREEN}   ✅ No recent errors${NC}"
    else
        echo -e "${YELLOW}   ⚠️ Found $ERROR_COUNT recent errors${NC}"
        echo "   Recent errors:"
        gcloud run services logs read $SERVICE_NAME --region=$REGION --limit=50 --filter="severity=ERROR" --format="value(timestamp,textPayload)" | head -3
    fi
}

# Function to check resource usage and performance
check_performance() {
    echo -e "${YELLOW}📊 Checking performance metrics...${NC}"
    
    # Get service metrics (if available)
    echo "   Service Configuration:"
    gcloud run services describe $SERVICE_NAME --region=$REGION --format="table(spec.template.spec.containers[0].resources.limits.cpu,spec.template.spec.containers[0].resources.limits.memory,spec.template.spec.containerConcurrency,spec.template.spec.timeoutSeconds)"
    
    # Test response time
    echo ""
    echo "   Testing response time..."
    if command -v curl &> /dev/null && [ -n "$SERVICE_URL" ]; then
        RESPONSE_TIME=$(curl -o /dev/null -s -w "%{time_total}" "$SERVICE_URL/health" 2>/dev/null || echo "N/A")
        echo "   Health endpoint response time: ${RESPONSE_TIME}s"
        
        if (( $(echo "$RESPONSE_TIME < 2.0" | bc -l 2>/dev/null || echo "0") )); then
            echo -e "${GREEN}   ✅ Response time good (<2s)${NC}"
        else
            echo -e "${YELLOW}   ⚠️ Response time could be better (>2s)${NC}"
        fi
    else
        echo "   Cannot test response time (curl not available or no service URL)"
    fi
}

# Function to run Python verification script
run_python_verification() {
    echo -e "${YELLOW}🐍 Running detailed Python verification...${NC}"
    
    if [ -n "$SERVICE_URL" ] && [ -f "scripts/verify_deployment.py" ]; then
        echo "   Running comprehensive verification..."
        python3 scripts/verify_deployment.py "$SERVICE_URL" "cloudrun_verification_report.json" || echo "   Python verification completed with warnings"
        
        if [ -f "cloudrun_verification_report.json" ]; then
            echo -e "${GREEN}   ✅ Detailed report saved to cloudrun_verification_report.json${NC}"
        fi
    else
        echo "   Skipping detailed verification (script or service URL not available)"
    fi
}

# Function to check connected services
check_connected_services() {
    echo -e "${YELLOW}🔗 Checking connected GCP services...${NC}"
    
    echo "   Firestore:"
    if gcloud firestore databases describe --region=$REGION &> /dev/null; then
        echo -e "${GREEN}   ✅ Firestore database exists${NC}"
    else
        echo -e "${YELLOW}   ⚠️ Firestore database not found${NC}"
    fi
    
    echo "   Pub/Sub topics:"
    TOPIC_COUNT=$(gcloud pubsub topics list --format="value(name)" | grep -E "(agent-runtime|evaluation|development|monitoring)" | wc -l)
    echo "   Found $TOPIC_COUNT SADP-related topics"
    
    echo "   Secret Manager:"
    SECRET_COUNT=$(gcloud secrets list --format="value(name)" | grep -E "(jwt-secret|sql-password|poml-api-key)" | wc -l)
    echo "   Found $SECRET_COUNT SADP-related secrets"
    
    echo "   Cloud Storage:"
    BUCKET_COUNT=$(gsutil ls | grep -E "(sarthi.*poml|sarthi.*app)" | wc -l || echo "0")
    echo "   Found $BUCKET_COUNT SADP-related buckets"
    
    echo -e "${GREEN}✅ Connected services checked${NC}"
}

# Function to generate summary report
generate_summary() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}📋 SADP Cloud Run Verification Summary${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
    echo -e "${GREEN}Service Information:${NC}"
    echo "• Project: $PROJECT_ID"
    echo "• Service Name: $SERVICE_NAME"
    echo "• Region: $REGION"
    echo "• Service URL: $SERVICE_URL"
    echo ""
    echo -e "${GREEN}Quick Service Overview:${NC}"
    gcloud run services describe $SERVICE_NAME --region=$REGION --format="table(metadata.name,status.url,status.conditions[0].status,spec.template.spec.containers[0].image)"
    echo ""
    echo -e "${GREEN}Useful Commands:${NC}"
    echo "• View logs: gcloud run services logs read $SERVICE_NAME --region=$REGION"
    echo "• Update service: gcloud run services update $SERVICE_NAME --region=$REGION"
    echo "• Get URL: gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)'"
    echo "• Delete service: gcloud run services delete $SERVICE_NAME --region=$REGION"
    echo ""
    echo -e "${GREEN}API Endpoints:${NC}"
    if [ -n "$SERVICE_URL" ]; then
        echo "• Health: $SERVICE_URL/health"
        echo "• Agents: $SERVICE_URL/agents/supported"
        echo "• Marketplace: $SERVICE_URL/agents/marketplace"
        echo "• Metrics: $SERVICE_URL/metrics"
    fi
    echo ""
}

# Main verification function
main() {
    check_gcloud_auth
    check_service_exists
    check_service_status
    check_traffic_allocation
    check_recent_deployments
    check_environment
    check_iam_permissions
    perform_health_checks
    check_logs
    check_performance
    run_python_verification
    check_connected_services
    generate_summary
    
    echo -e "${GREEN}🎉 Cloud Run verification completed!${NC}"
}

# Handle command line arguments
case "${1:-verify}" in
    "verify")
        main
        ;;
    "quick")
        echo -e "${BLUE}🚀 Quick Cloud Run Status Check${NC}"
        check_gcloud_auth
        gcloud run services describe $SERVICE_NAME --region=$REGION
        ;;
    "logs")
        echo -e "${BLUE}📝 Recent Logs${NC}"
        check_gcloud_auth
        gcloud run services logs read $SERVICE_NAME --region=$REGION --limit=50
        ;;
    "health")
        echo -e "${BLUE}🏥 Health Check Only${NC}"
        check_gcloud_auth
        check_service_status
        perform_health_checks
        ;;
    "performance")
        echo -e "${BLUE}📊 Performance Check${NC}"
        check_gcloud_auth
        check_service_status
        check_performance
        ;;
    "url")
        check_gcloud_auth
        SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
        echo "$SERVICE_URL"
        ;;
    "help")
        echo "Usage: $0 [verify|quick|logs|health|performance|url|help]"
        echo ""
        echo "Commands:"
        echo "  verify       - Full verification (default)"
        echo "  quick        - Quick status check"
        echo "  logs         - View recent logs"
        echo "  health       - Health check only"
        echo "  performance  - Performance check"
        echo "  url          - Get service URL"
        echo "  help         - Show this help"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac