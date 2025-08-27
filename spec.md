
# Sarthi AI Agent Development Platform (SADP) - Specification

## 1. Overview

The Sarthi AI Agent Development Platform (SADP) is a standalone, enterprise-grade microservice that provides comprehensive AI agent development, evaluation, and management capabilities as an external API service for the main Sarthi healthcare application. This document outlines the specifications for the SADP, focusing on a Google Cloud Platform (GCP) architecture.

## 2. Architecture: Google Cloud Platform

The SADP will be built on the Google Cloud Platform, leveraging its robust and scalable services for healthcare applications. The architecture will be event-driven and based on a service mesh.

### Key Components:

*   **API Gateway:** Google Cloud API Gateway to manage, secure, and monitor APIs.
*   **Authentication:** Google Cloud Identity-Aware Proxy (IAP) and OAuth 2.0 for secure authentication.
*   **Service Mesh:** Google Service Mesh (Istio) for managing and securing microservices.
*   **Event Streaming:** Google Cloud Pub/Sub for asynchronous messaging and event-driven architecture.
*   **Data & Infrastructure:**
    *   **Database:** Google Cloud SQL (PostgreSQL) for primary data storage, Firestore for document storage, and BigQuery for analytics.
    *   **Storage:** Google Cloud Storage for object storage.
    *   **AI & ML:** Google Cloud Healthcare APIs, Gemini, and Vertex AI for building and deploying AI models.
*   **CI/CD:** Google Cloud Build and Deploy for continuous integration and deployment.
*   **Monitoring:** Google Cloud Monitoring and Logging for observability.

### Security:

*   **IAM and Service Accounts:** Dedicated IAM service accounts will be created for each microservice with the minimum required permissions (principle of least privilege).
*   **Secrets Management:** All secrets will be stored in Google Secret Manager.
*   **Network Security:** VPC Service Controls will be used to create a service perimeter around our GCP resources. Private networking will be used for communication between services.
*   **Container Security:** Google Artifact Registry will be used for storing container images with vulnerability scanning enabled. Binary Authorization will be enabled to ensure only trusted container images are deployed.

### Scalability:

*   **Autoscaling:** GKE node pools and pods will be configured to autoscale based on traffic.
*   **Load Balancing:** An external HTTP(S) load balancer will be used to route traffic to the API gateway, and internal load balancers will be used for inter-service communication.
*   **Database Scalability:** Cloud SQL will be configured with read replicas to improve read performance.

### Operational Excellence:

*   **Infrastructure as Code (IaC):** Terraform will be used to provision and manage all GCP resources.
*   **Monitoring and Alerting:** A detailed monitoring and alerting plan will be created, including key metrics, dashboards, and alerts.
*   **Cost Management:** Billing alerts and cost allocation labels will be used to manage costs.

## 3. AI Agents

The SADP will support the following AI agents, as defined in the "Sarthi Healthcare POML Prompt Library":

1.  **Document Processor Agent:** OCR, handwriting recognition, form extraction, medication bottle analysis.
2.  **Clinical Agent:** Treatment plan generation, clinical note synthesis, lab result interpretation.
3.  **Billing Agent:** Claim generation, prior authorization, denial management.
4.  **Voice Agent:** Appointment scheduling, medication reminders, symptom triage.
5.  **AI Health Assistant Chatbot:** 24/7 patient portal support and administrative Q&A.
6.  **AI-Assisted Medication Entry:** Medication reconciliation via photo analysis.
7.  **Referral Document Processing:** Referral intake and clinical urgency assessment.
8.  **AI-Assisted Lab Result Entry:** Digitizing and analyzing paper lab results.

## 4. Core Services

The SADP will be composed of the following core services:

*   **Agent Runtime Service:** Executes AI agents and workflows.
*   **Evaluation Service:** Evaluates agent performance and accuracy.
*   **Development Service:** Manages agent development and deployment.
*   **Monitoring Service:** Monitors agent performance and system health.

## 5. Technology Stack

*   **Backend:** Python with FastAPI and Uvicorn.
*   **AI/ML:**
    *   **POML:** Microsoft's Prompt Orchestration Markup Language (POML) for prompt engineering.
    *   **Google Cloud AI/ML Services:** As listed in the architecture section.
*   **Database:** PostgreSQL, Firestore, BigQuery.
*   **Messaging:** Google Cloud Pub/Sub.
*   **Containerization:** Docker.
*   **Orchestration:** Google Kubernetes Engine (GKE).

## 6. POML Infrastructure

The POML infrastructure will consist of the following components:

*   **POML Prompt Library:** A dedicated Git repository will be created to store the POML prompt templates, as described in the "Sarthi Healthcare POML Prompt Library.pdf". This library will be version-controlled and managed separately from the core services.
*   **POML SDK:** The POML Python SDK will be integrated into the **Agent Runtime Service** to load, parse, and execute the POML prompts.
*   **VS Code Extension:** The POML VS Code extension will be used by developers for creating and editing POML prompts.
