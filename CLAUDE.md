# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Sarthi AI Agent Development Platform (SADP) is an enterprise-grade microservice platform for AI agent development, evaluation, and management in healthcare applications. It serves as an external API service for the main Sarthi healthcare application.

## Architecture

- **Platform**: Google Cloud Platform (GCP)
- **Infrastructure**: Event-driven microservices with service mesh (Istio)
- **Infrastructure as Code**: Terraform for all GCP resource provisioning
- **Container Orchestration**: Google Kubernetes Engine (GKE)

## Core Components

### Services (to be implemented)
- **Agent Runtime Service**: Executes AI agents and workflows
- **Evaluation Service**: Evaluates agent performance and accuracy
- **Development Service**: Manages agent development and deployment
- **Monitoring Service**: Monitors agent performance and system health

### AI Agents (per POML Prompt Library)
1. Document Processor Agent
2. Clinical Agent
3. Billing Agent
4. Voice Agent
5. AI Health Assistant Chatbot
6. AI-Assisted Medication Entry
7. Referral Document Processing
8. AI-Assisted Lab Result Entry

## Technology Stack

- **Backend**: Python with FastAPI and Uvicorn
- **AI/ML**: Microsoft POML (Prompt Orchestration Markup Language), Google Cloud AI services (Gemini, Vertex AI)
- **Databases**: Cloud SQL (PostgreSQL), Firestore, BigQuery
- **Messaging**: Google Cloud Pub/Sub
- **Storage**: Google Cloud Storage

## Development Commands

### Terraform Infrastructure Management

```bash
# Initialize Terraform
terraform init

# Plan infrastructure changes
terraform plan

# Apply infrastructure changes
terraform apply

# Destroy infrastructure (use with caution)
terraform destroy

# Format Terraform files
terraform fmt

# Validate Terraform configuration
terraform validate
```

### Required Environment Variables

Before running Terraform commands, ensure these variables are set:
- `TF_VAR_organization_id`: Your GCP organization ID
- `TF_VAR_project_id`: Your GCP project ID (default: sarthi-patient-experience-hub)
- `TF_VAR_region`: GCP region (default: us-central1)

## Current Infrastructure Status

The Terraform configuration provisions:
- VPC network and subnet
- GKE cluster with node pool
- Cloud SQL PostgreSQL instance
- Storage buckets (POML library, app data)
- Pub/Sub topics for each service
- Service accounts with least privilege
- Secret Manager for sensitive data
- VPC Service Controls for network security

## Task Tracking

Active development tasks are tracked in `taskmanager.md`. Current phases:
- Phase 1: Project Setup & Scaffolding (Complete)
- Phase 2: Core Infrastructure (Partially Complete)
- Phase 3-7: Core Services, AI Agents, Integration, POML, Monitoring (Not Started)

## Security Considerations

- All service accounts follow principle of least privilege
- Secrets stored in Google Secret Manager
- VPC Service Controls create security perimeter
- Binary Authorization for container image verification
- Private networking for inter-service communication

## Next Steps

1. Complete GCP project setup and billing configuration
2. Deploy base infrastructure using Terraform
3. Implement core services starting with Agent Runtime Service
4. Set up POML prompt library repository
5. Integrate POML SDK into services