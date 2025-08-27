#!/bin/bash

# Deploy Goal-Driven Self-Tuning Platform Services
# This script deploys the three optimization microservices to Google Cloud Run

set -e

# Configuration
PROJECT_ID="sarthi-patient-experience-hub"
REGION="us-central1"
REGISTRY="us-central1-docker.pkg.dev"

echo "🚀 Deploying Goal-Driven Self-Tuning Platform Services"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Registry: $REGISTRY"

# Authenticate with Google Cloud
echo "🔐 Authenticating with Google Cloud..."
gcloud auth configure-docker $REGISTRY

# Function to deploy a service
deploy_service() {
    local service_name=$1
    local service_path=$2
    local port=${3:-8080}
    
    echo "📦 Building and deploying $service_name..."
    
    # Change to service directory
    cd "$service_path"
    
    # Build the Docker image
    echo "🔨 Building Docker image for $service_name..."
    docker build -t $REGISTRY/$PROJECT_ID/sadp/$service_name:latest .
    
    # Push the image to Container Registry
    echo "⬆️  Pushing image to Container Registry..."
    docker push $REGISTRY/$PROJECT_ID/sadp/$service_name:latest
    
    # Deploy to Cloud Run
    echo "🚀 Deploying to Cloud Run..."
    gcloud run deploy sadp-$service_name-prod \
        --image $REGISTRY/$PROJECT_ID/sadp/$service_name:latest \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --port $port \
        --memory 2Gi \
        --cpu 1 \
        --max-instances 10 \
        --set-env-vars "PROJECT_ID=$PROJECT_ID,REGION=$REGION" \
        --project $PROJECT_ID
    
    # Get the service URL
    SERVICE_URL=$(gcloud run services describe sadp-$service_name-prod --region=$REGION --project=$PROJECT_ID --format="value(status.url)")
    echo "✅ $service_name deployed successfully!"
    echo "   URL: $SERVICE_URL"
    echo ""
    
    # Return to root directory
    cd - > /dev/null
}

# Deploy each service
echo "📋 Starting deployment of optimization services..."

# 1. Goal Definition Service
deploy_service "goal-definition" "./services/goal-definition"

# 2. Kaggle Integration Service  
deploy_service "kaggle-integration" "./services/kaggle-integration"

# 3. Prompt Optimization Engine
deploy_service "prompt-optimization" "./services/prompt-optimization"

echo "🎉 All optimization services deployed successfully!"
echo ""
echo "📍 Service URLs:"
echo "   Goal Definition: https://sadp-goal-definition-prod-355881591332.us-central1.run.app"
echo "   Kaggle Integration: https://sadp-kaggle-integration-prod-355881591332.us-central1.run.app"
echo "   Prompt Optimization: https://sadp-prompt-optimization-prod-355881591332.us-central1.run.app"
echo ""
echo "🔗 Frontend already integrated and deployed at:"
echo "   https://sadp-simple-prod-355881591332.us-central1.run.app"
echo ""
echo "✨ Goal-Driven Self-Tuning Platform is now live!"