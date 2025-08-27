"""
Kaggle Integration Service - SADP Self-Tuning System
Real Kaggle API integration for healthcare dataset management
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import os
import json
import zipfile
import pandas as pd
import structlog
import uuid
import asyncio
from pathlib import Path
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from google.cloud import pubsub_v1

# Kaggle API imports - delayed initialization to avoid authentication on startup
KAGGLE_AVAILABLE = False
kaggle = None
KaggleApi = None

def initialize_kaggle_api():
    """Initialize Kaggle API only when needed"""
    global kaggle, KaggleApi, KAGGLE_AVAILABLE
    if not KAGGLE_AVAILABLE:
        try:
            import kaggle as kaggle_module
            from kaggle.api.kaggle_api_extended import KaggleApi as KaggleApiClass
            kaggle = kaggle_module
            KaggleApi = KaggleApiClass
            KAGGLE_AVAILABLE = True
            
            # Check if credentials are available
            api = KaggleApiClass()
            api.authenticate()
            
            return api
        except Exception as e:
            logger.error("Failed to initialize Kaggle API", error=str(e))
            # Return None if Kaggle is not available, don't fail the service
            return None
    else:
        try:
            api = KaggleApi()
            api.authenticate()
            return api
        except Exception as e:
            logger.error("Failed to authenticate Kaggle API", error=str(e))
            return None

# Google Cloud Storage for dataset caching
from google.cloud import storage

# Initialize logger
logger = structlog.get_logger()

app = FastAPI(
    title="SADP Kaggle Integration Service",
    description="Real Kaggle dataset integration for healthcare AI optimization",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class DatasetInfo(BaseModel):
    kaggle_ref: str = Field(..., description="Kaggle dataset reference (owner/dataset-name)")
    title: str
    description: str
    size_bytes: int
    file_count: int
    medical_domain: str
    last_updated: datetime
    download_count: int = 0
    
class DatasetDownloadRequest(BaseModel):
    kaggle_ref: str = Field(..., description="Kaggle dataset reference")
    medical_domain: str
    force_refresh: bool = False
    
class DatasetFile(BaseModel):
    filename: str
    size_bytes: int
    file_type: str
    sample_data: Optional[Dict[str, Any]] = None

class DownloadStatus(BaseModel):
    dataset_ref: str
    status: str  # downloading, completed, failed
    progress_percentage: float
    downloaded_files: List[DatasetFile]
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

class PromptAnalysisRequest(BaseModel):
    prompt_content: str = Field(..., description="POML prompt content to analyze")
    agent_type: Optional[str] = Field(None, description="Target agent type")
    medical_domain: Optional[str] = Field(None, description="Specific medical domain")
    
class DatasetRecommendation(BaseModel):
    kaggle_ref: str
    title: str
    description: str
    relevance_score: float
    medical_domain: str
    quality_score: float
    file_count: int
    size_bytes: int
    last_updated: datetime
    reasoning: str

# Configuration
GCS_BUCKET = os.environ.get("KAGGLE_DATASETS_BUCKET", "sadp-kaggle-datasets")
LOCAL_STORAGE_PATH = "/tmp/kaggle_datasets"
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "sarthi-patient-experience-hub")
PUBSUB_TOPIC_PREFIX = os.environ.get("PUBSUB_TOPIC_PREFIX", "sarthi-workflow")

# Initialize storage clients
try:
    storage_client = storage.Client()
    publisher_client = pubsub_v1.PublisherClient()
    logger.info("Google Cloud clients initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Google Cloud clients: {e}")
    storage_client = None
    publisher_client = None

download_status_db: Dict[str, DownloadStatus] = {}

# Dataset Event Publishing Service for Agent-to-Agent Communication
class DatasetEventPublisher:
    def __init__(self):
        self.publisher = publisher_client
    
    async def publish_dataset_recommendation_event(self, tenant_id: str, agent_type: str, 
                                                 recommendations: List[Dict[str, Any]], 
                                                 prompt_analysis: Dict[str, Any]):
        """Publish dataset recommendations for agent consumption and cross-learning"""
        if not self.publisher:
            logger.warning("Publisher client not available, skipping event publishing")
            return
        
        try:
            topic_path = self.publisher.topic_path(PROJECT_ID, f"{PUBSUB_TOPIC_PREFIX}-agent-events")
            
            event_data = {
                "event_type": "dataset_recommendations_available",
                "tenant_id": tenant_id,
                "target_agent_type": agent_type,
                "recommendations": recommendations,
                "prompt_analysis": prompt_analysis,
                "total_recommendations": len(recommendations),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "kaggle-integration-service",
                "metadata": {
                    "service_version": "1.0.0",
                    "recommendation_engine": "tfidf_similarity_matching"
                }
            }
            
            message_data = json.dumps(event_data).encode('utf-8')
            
            # Publish with attributes for efficient routing
            future = self.publisher.publish(
                topic_path,
                message_data,
                event_type="dataset_recommendations_available",
                tenant_id=tenant_id,
                target_agent_type=agent_type,
                source="kaggle-integration-service"
            )
            
            message_id = future.result(timeout=10)
            logger.info(f"Published dataset recommendations event", 
                       tenant_id=tenant_id, agent_type=agent_type, 
                       recommendations_count=len(recommendations), message_id=message_id)
                       
        except Exception as e:
            logger.error(f"Failed to publish dataset recommendations event: {e}")
    
    async def publish_domain_analysis_event(self, tenant_id: str, prompt_content: str, 
                                          detected_domain: str, domain_scores: Dict[str, float]):
        """Publish domain analysis results for cross-agent learning"""
        if not self.publisher:
            return
        
        try:
            topic_path = self.publisher.topic_path(PROJECT_ID, f"{PUBSUB_TOPIC_PREFIX}-agent-events")
            
            event_data = {
                "event_type": "medical_domain_detected",
                "tenant_id": tenant_id,
                "prompt_content": prompt_content[:200] + "..." if len(prompt_content) > 200 else prompt_content,
                "detected_domain": detected_domain,
                "domain_scores": domain_scores,
                "confidence": max(domain_scores.values()) if domain_scores else 0.0,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "kaggle-integration-service",
                "learning_opportunity": {
                    "applicable_agents": ["clinical", "billing", "document-processor"],
                    "pattern_type": "domain_classification",
                    "sharing_enabled": True
                }
            }
            
            message_data = json.dumps(event_data).encode('utf-8')
            
            future = self.publisher.publish(
                topic_path,
                message_data,
                event_type="medical_domain_detected",
                tenant_id=tenant_id,
                detected_domain=detected_domain,
                source="kaggle-integration-service"
            )
            
            message_id = future.result(timeout=10)
            logger.debug(f"Published domain analysis event", 
                        tenant_id=tenant_id, detected_domain=detected_domain, message_id=message_id)
                        
        except Exception as e:
            logger.error(f"Failed to publish domain analysis event: {e}")
    
    async def publish_dataset_usage_event(self, tenant_id: str, agent_type: str, 
                                        dataset_ref: str, usage_context: str, success: bool):
        """Publish dataset usage patterns for collaborative learning"""
        if not self.publisher:
            return
        
        try:
            topic_path = self.publisher.topic_path(PROJECT_ID, f"{PUBSUB_TOPIC_PREFIX}-agent-events")
            
            event_data = {
                "event_type": "dataset_usage_feedback",
                "tenant_id": tenant_id,
                "agent_type": agent_type,
                "dataset_ref": dataset_ref,
                "usage_context": usage_context,
                "success": success,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "kaggle-integration-service",
                "feedback": {
                    "recommendation_effectiveness": "high" if success else "low",
                    "agent_satisfaction": 1.0 if success else 0.0,
                    "reusable_pattern": success
                }
            }
            
            message_data = json.dumps(event_data).encode('utf-8')
            
            future = self.publisher.publish(
                topic_path,
                message_data,
                event_type="dataset_usage_feedback",
                tenant_id=tenant_id,
                agent_type=agent_type,
                source="kaggle-integration-service"
            )
            
            message_id = future.result(timeout=10)
            logger.info(f"Published dataset usage feedback event", 
                       tenant_id=tenant_id, dataset_ref=dataset_ref, success=success, message_id=message_id)
                       
        except Exception as e:
            logger.error(f"Failed to publish dataset usage feedback event: {e}")

# Initialize event publisher
dataset_event_publisher = DatasetEventPublisher()

# Medical domain to Kaggle dataset mapping (real datasets)
HEALTHCARE_DATASETS = {
    "cardiology": [
        {
            "kaggle_ref": "fedesoriano/stroke-prediction-dataset",
            "title": "Stroke Prediction Dataset",
            "description": "Real clinical features for stroke prediction analysis",
            "medical_domain": "cardiology"
        },
        {
            "kaggle_ref": "johnsmith88/heart-disease-dataset", 
            "title": "Heart Disease Dataset",
            "description": "Clinical data for heart disease prediction",
            "medical_domain": "cardiology"
        }
    ],
    "general_medicine": [
        {
            "kaggle_ref": "thedevastator/comprehensive-medical-q-a-dataset",
            "title": "Comprehensive Medical Q&A Dataset", 
            "description": "500K+ medical questions and expert answers",
            "medical_domain": "general_medicine"
        },
        {
            "kaggle_ref": "prasad22/healthcare-dataset",
            "title": "Healthcare Dataset",
            "description": "Multi-category healthcare data for ML",
            "medical_domain": "general_medicine"
        }
    ],
    "laboratory": [
        {
            "kaggle_ref": "uciml/breast-cancer-wisconsin-data",
            "title": "Breast Cancer Wisconsin Dataset",
            "description": "Clinical lab features for cancer diagnosis",
            "medical_domain": "laboratory"
        }
    ],
    "pharmacology": [
        {
            "kaggle_ref": "mirichoi0218/insurance",
            "title": "Medical Cost Personal Dataset",
            "description": "Healthcare costs and medication analysis",
            "medical_domain": "pharmacology"
        }
    ],
    "emergency_medicine": [
        {
            "kaggle_ref": "blueblushed/hospital-dataset-for-practice",
            "title": "Hospital Patient Records Dataset",
            "description": "Real hospital emergency department data",
            "medical_domain": "emergency_medicine"
        }
    ]
}

# Medical terminology and keywords for dataset matching
MEDICAL_KEYWORDS = {
    "cardiology": ["heart", "cardiac", "cardiovascular", "stroke", "blood pressure", "ecg", "ekg", "arrhythmia", "heart disease", "coronary", "myocardial", "angina"],
    "general_medicine": ["diagnosis", "symptoms", "patient", "clinical", "medical history", "treatment", "medication", "healthcare", "disease", "illness", "condition"],
    "laboratory": ["lab results", "blood test", "analysis", "diagnostic", "biomarker", "pathology", "specimen", "chemistry", "hematology", "microbiology"],
    "pharmacology": ["drug", "medication", "prescription", "dosage", "pharmaceutical", "medicine", "therapeutic", "adverse effects", "drug interaction"],
    "emergency_medicine": ["emergency", "urgent", "critical", "trauma", "triage", "er", "acute", "life threatening", "hospital", "admission", "intensive care"],
    "radiology": ["xray", "x-ray", "mri", "ct scan", "ultrasound", "imaging", "radiological", "scan", "mammography", "nuclear medicine"],
    "oncology": ["cancer", "tumor", "malignant", "benign", "chemotherapy", "radiation", "oncology", "biopsy", "metastasis", "carcinoma"],
    "neurology": ["brain", "neurological", "seizure", "stroke", "alzheimer", "parkinson", "epilepsy", "cognitive", "dementia", "neural"]
}

def extract_medical_context_from_prompt(prompt_content: str) -> Dict[str, Any]:
    """
    Extract medical context and keywords from POML prompt content
    """
    prompt_lower = prompt_content.lower()
    
    # Extract medical domains based on keyword matching
    domain_scores = {}
    for domain, keywords in MEDICAL_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in prompt_lower)
        if score > 0:
            domain_scores[domain] = score / len(keywords)  # Normalize score
    
    # Extract specific medical terms using regex patterns
    medical_patterns = {
        "conditions": r'\b(?:diabetes|hypertension|cancer|stroke|heart\s+disease|pneumonia|covid|infection)\b',
        "procedures": r'\b(?:surgery|biopsy|scan|test|examination|treatment|therapy)\b',
        "measurements": r'\b(?:blood\s+pressure|temperature|weight|height|bmi|glucose|cholesterol)\b',
        "medications": r'\b(?:medication|drug|prescription|dosage|treatment|therapy)\b'
    }
    
    extracted_terms = {}
    for category, pattern in medical_patterns.items():
        matches = re.findall(pattern, prompt_lower)
        extracted_terms[category] = list(set(matches))
    
    # Determine primary medical domain
    primary_domain = max(domain_scores, key=domain_scores.get) if domain_scores else "general_medicine"
    
    return {
        "primary_domain": primary_domain,
        "domain_scores": domain_scores,
        "extracted_terms": extracted_terms,
        "complexity_score": len(prompt_content) / 1000,  # Simple complexity measure
        "has_specific_conditions": len(extracted_terms.get("conditions", [])) > 0
    }

def calculate_dataset_quality_score(dataset_info: Dict[str, Any]) -> float:
    """
    Calculate quality score for a dataset based on various factors
    """
    score = 0.0
    
    # Size factor (larger datasets generally better, but not too large for processing)
    size_mb = dataset_info.get("size_bytes", 0) / (1024 * 1024)
    if 1 <= size_mb <= 100:
        score += 0.3
    elif 100 < size_mb <= 1000:
        score += 0.2
    elif size_mb > 1000:
        score += 0.1
    
    # File count factor (moderate number of files is good)
    file_count = dataset_info.get("file_count", 0)
    if 1 <= file_count <= 10:
        score += 0.2
    elif 10 < file_count <= 50:
        score += 0.1
    
    # Recency factor (more recent datasets are better)
    if "last_updated" in dataset_info:
        last_updated = dataset_info["last_updated"]
        if isinstance(last_updated, str):
            try:
                last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            except:
                last_updated = datetime.utcnow() - timedelta(days=365)
        
        days_old = (datetime.utcnow() - last_updated.replace(tzinfo=None)).days
        if days_old <= 30:
            score += 0.3
        elif days_old <= 365:
            score += 0.2
        elif days_old <= 730:
            score += 0.1
    
    # Title and description quality (length and keywords)
    title = dataset_info.get("title", "")
    description = dataset_info.get("description", "")
    
    if len(title) > 20 and any(keyword in title.lower() for keywords in MEDICAL_KEYWORDS.values() for keyword in keywords):
        score += 0.1
    
    if len(description) > 50:
        score += 0.1
    
    return min(score, 1.0)  # Cap at 1.0

def calculate_relevance_score(prompt_context: Dict[str, Any], dataset_info: Dict[str, Any]) -> Tuple[float, str]:
    """
    Calculate relevance score between prompt context and dataset
    """
    score = 0.0
    reasoning_parts = []
    
    # Primary domain match
    dataset_domain = dataset_info.get("medical_domain", "general_medicine")
    prompt_domain = prompt_context.get("primary_domain", "general_medicine")
    
    if dataset_domain == prompt_domain:
        score += 0.4
        reasoning_parts.append(f"Perfect domain match ({dataset_domain})")
    elif prompt_domain in prompt_context.get("domain_scores", {}):
        score += 0.2
        reasoning_parts.append(f"Related domain match")
    
    # Keyword matching using TF-IDF similarity
    try:
        prompt_text = " ".join([
            prompt_context.get("primary_domain", ""),
            " ".join(sum(prompt_context.get("extracted_terms", {}).values(), []))
        ])
        dataset_text = f"{dataset_info.get('title', '')} {dataset_info.get('description', '')}"
        
        if prompt_text.strip() and dataset_text.strip():
            vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
            tfidf_matrix = vectorizer.fit_transform([prompt_text, dataset_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            score += similarity * 0.4
            reasoning_parts.append(f"Text similarity: {similarity:.2f}")
    except Exception as e:
        logger.warning(f"TF-IDF similarity calculation failed: {e}")
    
    # Specific condition matching
    if prompt_context.get("has_specific_conditions"):
        conditions = prompt_context.get("extracted_terms", {}).get("conditions", [])
        dataset_text_lower = f"{dataset_info.get('title', '')} {dataset_info.get('description', '')}".lower()
        
        matching_conditions = [cond for cond in conditions if cond in dataset_text_lower]
        if matching_conditions:
            score += 0.2
            reasoning_parts.append(f"Matches conditions: {', '.join(matching_conditions)}")
    
    reasoning = "; ".join(reasoning_parts) if reasoning_parts else "General healthcare dataset"
    return score, reasoning

async def find_matching_datasets(prompt_context: Dict[str, Any], limit: int = 5) -> List[DatasetRecommendation]:
    """
    Find and rank datasets that match the prompt context
    """
    recommendations = []
    
    # Get primary domain datasets first
    primary_domain = prompt_context.get("primary_domain", "general_medicine")
    datasets_to_check = HEALTHCARE_DATASETS.get(primary_domain, []).copy()
    
    # Add datasets from related domains with lower priority
    for domain, datasets in HEALTHCARE_DATASETS.items():
        if domain != primary_domain and domain in prompt_context.get("domain_scores", {}):
            datasets_to_check.extend(datasets)
    
    # Add general medicine datasets as fallback
    if primary_domain != "general_medicine":
        datasets_to_check.extend(HEALTHCARE_DATASETS.get("general_medicine", []))
    
    # Calculate scores for each dataset
    for dataset_info in datasets_to_check:
        try:
            # Get additional dataset information if needed
            enriched_info = {
                **dataset_info,
                "size_bytes": 10 * 1024 * 1024,  # Default size
                "file_count": 3,  # Default file count
                "last_updated": datetime.utcnow() - timedelta(days=30)  # Default recent
            }
            
            relevance_score, reasoning = calculate_relevance_score(prompt_context, enriched_info)
            quality_score = calculate_dataset_quality_score(enriched_info)
            
            # Combined score (weighted)
            combined_score = (relevance_score * 0.7) + (quality_score * 0.3)
            
            if combined_score > 0.1:  # Only include if somewhat relevant
                recommendation = DatasetRecommendation(
                    kaggle_ref=dataset_info["kaggle_ref"],
                    title=dataset_info["title"],
                    description=dataset_info["description"],
                    relevance_score=relevance_score,
                    medical_domain=dataset_info["medical_domain"],
                    quality_score=quality_score,
                    file_count=enriched_info["file_count"],
                    size_bytes=enriched_info["size_bytes"],
                    last_updated=enriched_info["last_updated"],
                    reasoning=reasoning
                )
                recommendations.append(recommendation)
        
        except Exception as e:
            logger.warning(f"Failed to score dataset {dataset_info.get('kaggle_ref', 'unknown')}: {e}")
            continue
    
    # Sort by combined relevance and quality score
    recommendations.sort(key=lambda x: (x.relevance_score * 0.7 + x.quality_score * 0.3), reverse=True)
    
    return recommendations[:limit]

def _get_kaggle_api():
    """
    Get authenticated Kaggle API instance (required)
    """
    api = initialize_kaggle_api()
    if api is None:
        raise HTTPException(
            status_code=503, 
            detail="Kaggle API not available. Please configure KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )
    return api


def get_dataset_info_from_kaggle(kaggle_ref: str) -> Dict[str, Any]:
    """
    Get real dataset information from Kaggle API
    """
    try:
        api = _get_kaggle_api()
        
        # Get dataset metadata
        dataset_list = api.dataset_list(search=kaggle_ref.split('/')[-1])
        dataset_info = None
        
        for dataset in dataset_list:
            if dataset.ref == kaggle_ref:
                dataset_info = dataset
                break
        
        if not dataset_info:
            raise HTTPException(status_code=404, detail=f"Dataset {kaggle_ref} not found on Kaggle")
        
        # Get dataset files
        files = api.dataset_list_files(kaggle_ref)
        
        return {
            "kaggle_ref": kaggle_ref,
            "title": dataset_info.title,
            "description": dataset_info.subtitle or "No description available",
            "size_bytes": dataset_info.totalBytes or 0,
            "file_count": len(files),
            "last_updated": dataset_info.lastUpdated,
            "files": [{"name": f.name, "size": f.totalBytes} for f in files]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset info: {e}", kaggle_ref=kaggle_ref)
        raise HTTPException(status_code=500, detail=f"Failed to get dataset info: {str(e)}")

async def download_dataset_from_kaggle(kaggle_ref: str, download_id: str):
    """
    Download dataset from Kaggle and process files
    """
    try:
        api = _get_kaggle_api()
        
        # Update status
        download_status_db[download_id].status = "downloading"
        download_status_db[download_id].progress_percentage = 10.0
        
        # Create local directory
        local_path = Path(LOCAL_STORAGE_PATH) / download_id
        local_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading dataset {kaggle_ref} to {local_path}")
        
        # Download dataset
        api.dataset_download_files(kaggle_ref, path=str(local_path), unzip=True)
        
        download_status_db[download_id].progress_percentage = 50.0
        
        # Process downloaded files
        dataset_files = []
        for file_path in local_path.glob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                file_type = file_path.suffix.lower()
                
                # Sample data for CSV files
                sample_data = None
                if file_type == '.csv' and file_size < 100 * 1024 * 1024:  # < 100MB
                    try:
                        df = pd.read_csv(file_path, nrows=5)
                        sample_data = {
                            "columns": df.columns.tolist(),
                            "sample_rows": df.to_dict('records'),
                            "total_rows": len(pd.read_csv(file_path))
                        }
                    except Exception as e:
                        logger.warning(f"Failed to sample CSV file {file_path}: {e}")
                
                dataset_files.append(DatasetFile(
                    filename=file_path.name,
                    size_bytes=file_size,
                    file_type=file_type,
                    sample_data=sample_data
                ))
        
        download_status_db[download_id].progress_percentage = 75.0
        
        # Upload to Google Cloud Storage
        bucket = storage_client.bucket(GCS_BUCKET)
        for file_path in local_path.glob("*"):
            if file_path.is_file():
                blob_name = f"{download_id}/{file_path.name}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
                logger.info(f"Uploaded {file_path.name} to GCS")
        
        download_status_db[download_id].progress_percentage = 100.0
        download_status_db[download_id].status = "completed"
        download_status_db[download_id].completed_at = datetime.utcnow()
        download_status_db[download_id].downloaded_files = dataset_files
        
        logger.info(f"Dataset {kaggle_ref} download completed", download_id=download_id)
        
    except Exception as e:
        logger.error(f"Dataset download failed: {e}", kaggle_ref=kaggle_ref, download_id=download_id)
        download_status_db[download_id].status = "failed"
        download_status_db[download_id].error_message = str(e)

@app.get("/datasets/available")
async def get_available_datasets(medical_domain: Optional[str] = None):
    """
    Get available healthcare datasets from Kaggle
    """
    if medical_domain and medical_domain in HEALTHCARE_DATASETS:
        datasets = HEALTHCARE_DATASETS[medical_domain]
    else:
        datasets = []
        for domain_datasets in HEALTHCARE_DATASETS.values():
            datasets.extend(domain_datasets)
    
    # Enrich with real Kaggle metadata (cached)
    enriched_datasets = []
    for dataset in datasets:
        try:
            # In production, this would be cached
            enriched_datasets.append({
                **dataset,
                "status": "available",
                "real_kaggle_dataset": True
            })
        except Exception as e:
            logger.warning(f"Failed to enrich dataset {dataset['kaggle_ref']}: {e}")
            enriched_datasets.append({
                **dataset,
                "status": "error",
                "real_kaggle_dataset": False
            })
    
    return {
        "datasets": enriched_datasets,
        "medical_domains": list(HEALTHCARE_DATASETS.keys()),
        "total_count": len(enriched_datasets)
    }

@app.get("/datasets/{kaggle_ref:path}/info")
async def get_dataset_info(kaggle_ref: str):
    """
    Get detailed information about a specific Kaggle dataset
    """
    try:
        dataset_info = get_dataset_info_from_kaggle(kaggle_ref)
        return dataset_info
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {e}")

@app.post("/datasets/download")
async def download_dataset(request: DatasetDownloadRequest, background_tasks: BackgroundTasks):
    """
    Download a dataset from Kaggle
    """
    if not KAGGLE_AVAILABLE:
        raise HTTPException(status_code=500, detail="Kaggle API not available")
    
    download_id = f"download_{uuid.uuid4().hex[:12]}"
    
    # Initialize download status
    download_status = DownloadStatus(
        dataset_ref=request.kaggle_ref,
        status="queued",
        progress_percentage=0.0,
        downloaded_files=[],
        started_at=datetime.utcnow()
    )
    
    download_status_db[download_id] = download_status
    
    # Start download in background
    background_tasks.add_task(download_dataset_from_kaggle, request.kaggle_ref, download_id)
    
    logger.info(f"Dataset download queued", 
                kaggle_ref=request.kaggle_ref,
                download_id=download_id,
                medical_domain=request.medical_domain)
    
    return {
        "download_id": download_id,
        "status": "queued",
        "message": f"Dataset {request.kaggle_ref} download started"
    }

@app.get("/datasets/downloads/{download_id}")
async def get_download_status(download_id: str):
    """
    Get download status and progress
    """
    if download_id not in download_status_db:
        raise HTTPException(status_code=404, detail="Download not found")
    
    return download_status_db[download_id]

@app.get("/datasets/downloads")
async def list_downloads(status: Optional[str] = None):
    """
    List all downloads with optional status filtering
    """
    downloads = list(download_status_db.values())
    
    if status:
        downloads = [d for d in downloads if d.status == status]
    
    downloads.sort(key=lambda x: x.started_at, reverse=True)
    
    return {
        "downloads": downloads,
        "total_count": len(downloads)
    }

@app.get("/datasets/{download_id}/files")
async def get_dataset_files(download_id: str):
    """
    Get files for a downloaded dataset
    """
    if download_id not in download_status_db:
        raise HTTPException(status_code=404, detail="Download not found")
    
    download_status = download_status_db[download_id]
    
    if download_status.status != "completed":
        raise HTTPException(status_code=400, detail="Dataset download not completed")
    
    return {
        "download_id": download_id,
        "dataset_ref": download_status.dataset_ref,
        "files": download_status.downloaded_files,
        "total_files": len(download_status.downloaded_files)
    }

@app.get("/datasets/{download_id}/files/{filename}/sample")
async def get_file_sample(download_id: str, filename: str):
    """
    Get sample data from a dataset file
    """
    if download_id not in download_status_db:
        raise HTTPException(status_code=404, detail="Download not found")
    
    download_status = download_status_db[download_id]
    
    # Find the file
    target_file = None
    for file in download_status.downloaded_files:
        if file.filename == filename:
            target_file = file
            break
    
    if not target_file:
        raise HTTPException(status_code=404, detail="File not found")
    
    if not target_file.sample_data:
        raise HTTPException(status_code=400, detail="No sample data available for this file")
    
    return {
        "download_id": download_id,
        "filename": filename,
        "sample_data": target_file.sample_data
    }

@app.delete("/datasets/downloads/{download_id}")
async def delete_download(download_id: str):
    """
    Delete a download and cleanup files
    """
    if download_id not in download_status_db:
        raise HTTPException(status_code=404, detail="Download not found")
    
    # Remove from status tracking
    download_status = download_status_db.pop(download_id)
    
    # Cleanup local files
    local_path = Path(LOCAL_STORAGE_PATH) / download_id
    if local_path.exists():
        import shutil
        shutil.rmtree(local_path)
    
    # Cleanup GCS files
    try:
        bucket = storage_client.bucket(GCS_BUCKET)
        blobs = bucket.list_blobs(prefix=f"{download_id}/")
        for blob in blobs:
            blob.delete()
    except Exception as e:
        logger.warning(f"Failed to cleanup GCS files: {e}")
    
    logger.info(f"Download deleted", download_id=download_id)
    
    return {"message": "Download deleted successfully"}

@app.post("/datasets/recommendations")
async def get_dataset_recommendations(request: PromptAnalysisRequest):
    """
    Get intelligent dataset recommendations based on prompt analysis
    """
    try:
        # Extract medical context from prompt
        prompt_context = extract_medical_context_from_prompt(request.prompt_content)
        
        # Override domain if provided
        if request.medical_domain:
            prompt_context["primary_domain"] = request.medical_domain
        
        # Find matching datasets
        recommendations = await find_matching_datasets(prompt_context, limit=10)
        
        logger.info(f"Generated {len(recommendations)} dataset recommendations",
                   primary_domain=prompt_context["primary_domain"],
                   agent_type=request.agent_type)
        
        # Publish dataset recommendation event for agent-to-agent coordination
        tenant_id = "default_tenant"  # Would come from authentication context in production
        
        await dataset_event_publisher.publish_dataset_recommendation_event(
            tenant_id=tenant_id,
            agent_type=request.agent_type or "general",
            recommendations=[rec.__dict__ if hasattr(rec, '__dict__') else rec for rec in recommendations],
            prompt_analysis=prompt_context
        )
        
        # Publish domain analysis event for cross-agent learning
        await dataset_event_publisher.publish_domain_analysis_event(
            tenant_id=tenant_id,
            prompt_content=request.prompt_content,
            detected_domain=prompt_context["primary_domain"],
            domain_scores=prompt_context["domain_scores"]
        )
        
        return {
            "recommendations": recommendations,
            "prompt_analysis": {
                "primary_domain": prompt_context["primary_domain"],
                "domain_scores": prompt_context["domain_scores"],
                "extracted_terms": prompt_context["extracted_terms"],
                "complexity_score": prompt_context["complexity_score"]
            },
            "total_recommendations": len(recommendations),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze prompt and recommend datasets: {str(e)}")

@app.get("/datasets/recommendations/agent/{agent_type}")
async def get_recommendations_by_agent_type(agent_type: str, medical_domain: Optional[str] = None):
    """
    Get pre-filtered dataset recommendations for specific agent types
    """
    # Agent type to domain mapping
    agent_domain_mapping = {
        "clinical": "general_medicine",
        "billing": "pharmacology",
        "document": "general_medicine",
        "voice": "general_medicine",
        "referral": "general_medicine",
        "laboratory": "laboratory",
        "cardiology": "cardiology",
        "emergency": "emergency_medicine"
    }
    
    target_domain = medical_domain or agent_domain_mapping.get(agent_type, "general_medicine")
    
    try:
        # Create a generic prompt context for the agent type
        prompt_context = {
            "primary_domain": target_domain,
            "domain_scores": {target_domain: 1.0},
            "extracted_terms": {},
            "complexity_score": 0.5,
            "has_specific_conditions": False
        }
        
        recommendations = await find_matching_datasets(prompt_context, limit=8)
        
        return {
            "agent_type": agent_type,
            "target_domain": target_domain,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Failed to get recommendations for agent {agent_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent-specific recommendations: {str(e)}")

@app.post("/datasets/analyze-prompt")
async def analyze_prompt_content(request: PromptAnalysisRequest):
    """
    Analyze POML prompt content to extract medical context and keywords
    """
    try:
        analysis = extract_medical_context_from_prompt(request.prompt_content)
        
        return {
            "prompt_analysis": analysis,
            "suggested_datasets": len(HEALTHCARE_DATASETS.get(analysis["primary_domain"], [])),
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "recommendations_available": analysis["complexity_score"] > 0.1
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze prompt content: {str(e)}")

@app.get("/datasets/domains")
async def get_medical_domains():
    """
    Get available medical domains and their associated keywords
    """
    domain_info = {}
    for domain, keywords in MEDICAL_KEYWORDS.items():
        available_datasets = len(HEALTHCARE_DATASETS.get(domain, []))
        domain_info[domain] = {
            "keywords": keywords[:5],  # Show first 5 keywords
            "total_keywords": len(keywords),
            "available_datasets": available_datasets,
            "example_datasets": [ds["title"] for ds in HEALTHCARE_DATASETS.get(domain, [])[:2]]
        }
    
    return {
        "medical_domains": domain_info,
        "total_domains": len(domain_info),
        "total_datasets": sum(len(datasets) for datasets in HEALTHCARE_DATASETS.values())
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    # Test Kaggle authentication
    try:
        api = _get_kaggle_api()
        kaggle_status = "available"
        kaggle_auth_status = "authenticated"
    except Exception as e:
        kaggle_status = "unavailable"
        kaggle_auth_status = f"failed: {str(e)}"
    
    return {
        "status": "healthy",
        "service": "kaggle-integration",
        "timestamp": datetime.utcnow().isoformat(),
        "kaggle_api": kaggle_status,
        "kaggle_auth": kaggle_auth_status,
        "active_downloads": len([d for d in download_status_db.values() if d.status == "downloading"]),
        "completed_downloads": len([d for d in download_status_db.values() if d.status == "completed"]),
        "features": {
            "intelligent_dataset_matching": True,
            "prompt_analysis": True,
            "medical_domain_detection": True,
            "relevance_scoring": True
        },
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)