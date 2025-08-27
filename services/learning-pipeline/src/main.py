"""
SADP Learning Pipeline Service
Integrates Kaggle datasets with POML prompt optimization for continuous learning
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os
import json
import asyncio
import uuid
import structlog
import httpx
from pathlib import Path

# Google Cloud imports
from google.cloud import storage, firestore, pubsub_v1
import google.generativeai as genai

# Initialize logger
logger = structlog.get_logger()

app = FastAPI(
    title="SADP Learning Pipeline Service",
    description="Continuous learning pipeline integrating Kaggle datasets with POML optimization",
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

# Configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "sarthi-patient-experience-hub")
KAGGLE_SERVICE_URL = os.environ.get("KAGGLE_SERVICE_URL", "https://sadp-kaggle-integration-xonau6hybq-uc.a.run.app")
AGENT_RUNTIME_URL = os.environ.get("AGENT_RUNTIME_URL", "https://sadp-agent-runtime-xonau6hybq-uc.a.run.app")
OPTIMIZATION_SERVICE_URL = os.environ.get("OPTIMIZATION_SERVICE_URL", "https://sadp-prompt-optimization-xonau6hybq-uc.a.run.app")

# Initialize clients
storage_client = storage.Client(project=PROJECT_ID)
firestore_client = firestore.Client(project=PROJECT_ID)
publisher = pubsub_v1.PublisherClient()

# Gemini setup
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

# Collections
learning_jobs_collection = firestore_client.collection('learning_jobs')
dataset_mapping_collection = firestore_client.collection('dataset_mappings')
prompt_performance_collection = firestore_client.collection('prompt_performance')
optimization_history_collection = firestore_client.collection('optimization_history')

# Models
class LearningJobRequest(BaseModel):
    agent_type: str = Field(..., description="Type of agent to optimize")
    medical_domain: str = Field(..., description="Medical domain for dataset selection")
    prompt_template_id: str = Field(..., description="POML template to optimize")
    optimization_objective: str = Field(default="accuracy", description="Optimization objective")
    max_iterations: int = Field(default=10, description="Maximum optimization iterations")
    
class DatasetMappingRequest(BaseModel):
    kaggle_ref: str = Field(..., description="Kaggle dataset reference")
    medical_domain: str
    agent_types: List[str]
    prompt_variables: List[str]
    expected_outputs: List[str]
    
class PromptTestCase(BaseModel):
    input_data: Dict[str, Any]
    expected_output: str
    medical_context: str
    confidence_threshold: float = 0.8
    
class LearningJob(BaseModel):
    job_id: str
    agent_type: str
    medical_domain: str
    prompt_template_id: str
    status: str  # queued, running, completed, failed
    datasets_used: List[str]
    iterations_completed: int
    best_performance: float
    optimization_history: List[Dict[str, Any]]
    created_at: datetime
    completed_at: Optional[datetime] = None

# Healthcare domain to agent mapping
AGENT_DOMAIN_MAPPING = {
    "cardiology": ["clinical_agent", "ai_health_assistant"],
    "general_medicine": ["clinical_agent", "ai_health_assistant", "document_processor"],
    "laboratory": ["clinical_agent", "lab_result_agent"],
    "pharmacology": ["medication_agent", "clinical_agent"],
    "emergency_medicine": ["clinical_agent", "ai_health_assistant"],
    "billing": ["billing_agent"],
    "documentation": ["document_processor", "referral_processor"]
}

class LearningPipelineManager:
    """Manages the continuous learning pipeline"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient()
        
    async def start_learning_job(self, request: LearningJobRequest) -> str:
        """Start a new learning job"""
        try:
            job_id = f"learning_job_{uuid.uuid4().hex[:12]}"
            
            # Create learning job document
            job_data = {
                "job_id": job_id,
                "agent_type": request.agent_type,
                "medical_domain": request.medical_domain,
                "prompt_template_id": request.prompt_template_id,
                "optimization_objective": request.optimization_objective,
                "max_iterations": request.max_iterations,
                "status": "queued",
                "datasets_used": [],
                "iterations_completed": 0,
                "best_performance": 0.0,
                "optimization_history": [],
                "created_at": datetime.utcnow(),
                "completed_at": None
            }
            
            learning_jobs_collection.document(job_id).set(job_data)
            
            logger.info("Learning job created", job_id=job_id, agent_type=request.agent_type)
            
            return job_id
            
        except Exception as e:
            logger.error("Failed to start learning job", error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to start learning job: {str(e)}")
    
    async def get_relevant_datasets(self, medical_domain: str, agent_type: str) -> List[Dict[str, Any]]:
        """Get relevant Kaggle datasets for the medical domain and agent type"""
        try:
            # Query Kaggle integration service
            response = await self.http_client.get(
                f"{KAGGLE_SERVICE_URL}/datasets/available",
                params={"medical_domain": medical_domain}
            )
            
            if response.status_code != 200:
                logger.warning("Failed to get datasets from Kaggle service")
                return []
            
            data = response.json()
            datasets = data.get("datasets", [])
            
            # Filter datasets relevant to agent type
            relevant_datasets = []
            for dataset in datasets:
                if self._is_dataset_relevant(dataset, agent_type, medical_domain):
                    relevant_datasets.append(dataset)
            
            logger.info(f"Found {len(relevant_datasets)} relevant datasets", 
                       medical_domain=medical_domain, agent_type=agent_type)
            
            return relevant_datasets
            
        except Exception as e:
            logger.error("Failed to get relevant datasets", error=str(e))
            return []
    
    def _is_dataset_relevant(self, dataset: Dict[str, Any], agent_type: str, medical_domain: str) -> bool:
        """Check if a dataset is relevant for the agent type and medical domain"""
        dataset_domain = dataset.get("medical_domain", "")
        dataset_title = dataset.get("title", "").lower()
        dataset_description = dataset.get("description", "").lower()
        
        # Check domain match
        if dataset_domain != medical_domain:
            return False
        
        # Check agent type relevance
        agent_keywords = {
            "clinical_agent": ["diagnosis", "clinical", "patient", "symptoms", "treatment"],
            "billing_agent": ["billing", "insurance", "cost", "payment", "claim"],
            "document_processor": ["document", "text", "report", "record"],
            "medication_agent": ["medication", "drug", "pharmacy", "prescription"],
            "lab_result_agent": ["lab", "test", "result", "laboratory", "analysis"],
            "ai_health_assistant": ["health", "patient", "medical", "care"],
            "referral_processor": ["referral", "specialist", "appointment"]
        }
        
        keywords = agent_keywords.get(agent_type, [])
        for keyword in keywords:
            if keyword in dataset_title or keyword in dataset_description:
                return True
        
        return False
    
    async def download_and_prepare_dataset(self, dataset_ref: str, medical_domain: str) -> str:
        """Download and prepare a dataset for learning"""
        try:
            # Download dataset via Kaggle service
            response = await self.http_client.post(
                f"{KAGGLE_SERVICE_URL}/datasets/download",
                json={
                    "kaggle_ref": dataset_ref,
                    "medical_domain": medical_domain,
                    "force_refresh": False
                }
            )
            
            if response.status_code != 200:
                raise ValueError(f"Failed to download dataset {dataset_ref}")
            
            download_data = response.json()
            download_id = download_data["download_id"]
            
            # Wait for download completion
            await self._wait_for_download_completion(download_id)
            
            # Process dataset into test cases
            test_cases = await self._generate_test_cases_from_dataset(download_id, medical_domain)
            
            logger.info(f"Generated {len(test_cases)} test cases from dataset", 
                       dataset_ref=dataset_ref)
            
            return download_id
            
        except Exception as e:
            logger.error("Failed to download and prepare dataset", 
                        dataset_ref=dataset_ref, error=str(e))
            raise
    
    async def _wait_for_download_completion(self, download_id: str, timeout: int = 300):
        """Wait for dataset download to complete"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            try:
                response = await self.http_client.get(
                    f"{KAGGLE_SERVICE_URL}/datasets/downloads/{download_id}"
                )
                
                if response.status_code == 200:
                    status_data = response.json()
                    if status_data["status"] == "completed":
                        return
                    elif status_data["status"] == "failed":
                        raise ValueError(f"Dataset download failed: {status_data.get('error_message')}")
                
                await asyncio.sleep(10)  # Wait 10 seconds before checking again
                
            except Exception as e:
                logger.error("Error checking download status", download_id=download_id, error=str(e))
                await asyncio.sleep(10)
        
        raise TimeoutError(f"Dataset download timeout after {timeout} seconds")
    
    async def _generate_test_cases_from_dataset(self, download_id: str, medical_domain: str) -> List[PromptTestCase]:
        """Generate test cases from downloaded dataset"""
        try:
            # Get dataset files
            response = await self.http_client.get(
                f"{KAGGLE_SERVICE_URL}/datasets/{download_id}/files"
            )
            
            if response.status_code != 200:
                return []
            
            files_data = response.json()
            files = files_data.get("files", [])
            
            test_cases = []
            
            # Process each file
            for file_info in files:
                if file_info["file_type"] == ".csv" and file_info.get("sample_data"):
                    sample_data = file_info["sample_data"]
                    columns = sample_data.get("columns", [])
                    sample_rows = sample_data.get("sample_rows", [])
                    
                    # Generate test cases from sample rows
                    for row in sample_rows[:10]:  # Limit to 10 samples per file
                        test_case = await self._create_test_case_from_row(row, columns, medical_domain)
                        if test_case:
                            test_cases.append(test_case)
            
            return test_cases
            
        except Exception as e:
            logger.error("Failed to generate test cases", download_id=download_id, error=str(e))
            return []
    
    async def _create_test_case_from_row(self, row: Dict, columns: List[str], medical_domain: str) -> Optional[PromptTestCase]:
        """Create a test case from a dataset row"""
        try:
            # Extract meaningful input and output based on medical domain
            if medical_domain == "cardiology":
                return await self._create_cardiology_test_case(row, columns)
            elif medical_domain == "general_medicine":
                return await self._create_general_medicine_test_case(row, columns)
            elif medical_domain == "laboratory":
                return await self._create_laboratory_test_case(row, columns)
            elif medical_domain == "pharmacology":
                return await self._create_pharmacology_test_case(row, columns)
            else:
                return await self._create_generic_test_case(row, columns, medical_domain)
                
        except Exception as e:
            logger.error("Failed to create test case from row", error=str(e))
            return None
    
    async def _create_cardiology_test_case(self, row: Dict, columns: List[str]) -> Optional[PromptTestCase]:
        """Create cardiology-specific test case"""
        # Look for common cardiology fields
        input_fields = ["age", "sex", "chest_pain_type", "blood_pressure", "cholesterol", "symptoms"]
        output_fields = ["diagnosis", "heart_disease", "risk_level", "recommendation"]
        
        input_data = {}
        expected_output = ""
        
        for col in columns:
            col_lower = col.lower()
            if any(field in col_lower for field in input_fields):
                if col in row and row[col] is not None:
                    input_data[col] = row[col]
            elif any(field in col_lower for field in output_fields):
                if col in row and row[col] is not None:
                    expected_output = str(row[col])
        
        if input_data and expected_output:
            return PromptTestCase(
                input_data=input_data,
                expected_output=expected_output,
                medical_context="cardiology",
                confidence_threshold=0.8
            )
        
        return None
    
    async def _create_laboratory_test_case(self, row: Dict, columns: List[str]) -> Optional[PromptTestCase]:
        """Create laboratory-specific test case"""
        # Look for lab test fields
        input_fields = ["test_name", "patient_age", "patient_sex", "test_value", "units"]
        output_fields = ["result", "interpretation", "normal_range", "abnormal"]
        
        input_data = {}
        expected_output = ""
        
        for col in columns:
            col_lower = col.lower()
            if any(field in col_lower for field in input_fields):
                if col in row and row[col] is not None:
                    input_data[col] = row[col]
            elif any(field in col_lower for field in output_fields):
                if col in row and row[col] is not None:
                    expected_output = str(row[col])
        
        if input_data and expected_output:
            return PromptTestCase(
                input_data=input_data,
                expected_output=expected_output,
                medical_context="laboratory",
                confidence_threshold=0.85
            )
        
        return None
    
    async def _create_generic_test_case(self, row: Dict, columns: List[str], medical_domain: str) -> Optional[PromptTestCase]:
        """Create generic test case from any medical data"""
        # Take first half of columns as input, second half as potential output
        mid_point = len(columns) // 2
        input_cols = columns[:mid_point]
        output_cols = columns[mid_point:]
        
        input_data = {}
        expected_output = ""
        
        for col in input_cols:
            if col in row and row[col] is not None:
                input_data[col] = row[col]
        
        for col in output_cols:
            if col in row and row[col] is not None:
                expected_output = str(row[col])
                break  # Use first available output
        
        if input_data and expected_output:
            return PromptTestCase(
                input_data=input_data,
                expected_output=expected_output,
                medical_context=medical_domain,
                confidence_threshold=0.7
            )
        
        return None
    
    async def optimize_prompt_with_data(self, job_id: str, prompt_template_id: str, test_cases: List[PromptTestCase]) -> Dict[str, Any]:
        """Optimize prompt using test cases"""
        try:
            # Call prompt optimization service
            response = await self.http_client.post(
                f"{OPTIMIZATION_SERVICE_URL}/optimize/automedprompt",
                json={
                    "prompt_template_id": prompt_template_id,
                    "test_cases": [
                        {
                            "input_data": tc.input_data,
                            "expected_output": tc.expected_output,
                            "medical_context": tc.medical_context
                        } for tc in test_cases
                    ],
                    "optimization_objective": "accuracy",
                    "max_iterations": 5
                }
            )
            
            if response.status_code != 200:
                raise ValueError(f"Optimization service failed: {response.text}")
            
            optimization_result = response.json()
            
            # Update job with optimization results
            await self._update_job_optimization_results(job_id, optimization_result)
            
            return optimization_result
            
        except Exception as e:
            logger.error("Failed to optimize prompt", job_id=job_id, error=str(e))
            raise
    
    async def _update_job_optimization_results(self, job_id: str, optimization_result: Dict[str, Any]):
        """Update learning job with optimization results"""
        try:
            job_ref = learning_jobs_collection.document(job_id)
            job_doc = job_ref.get()
            
            if job_doc.exists:
                job_data = job_doc.to_dict()
                
                # Update optimization history
                optimization_history = job_data.get("optimization_history", [])
                optimization_history.append({
                    "timestamp": datetime.utcnow(),
                    "performance": optimization_result.get("final_performance", 0),
                    "iterations": optimization_result.get("iterations_completed", 0),
                    "best_prompt": optimization_result.get("optimized_prompt", "")
                })
                
                # Update job data
                updates = {
                    "optimization_history": optimization_history,
                    "best_performance": max(job_data.get("best_performance", 0), 
                                          optimization_result.get("final_performance", 0)),
                    "iterations_completed": job_data.get("iterations_completed", 0) + 1,
                    "status": "completed" if optimization_result.get("converged", False) else "running"
                }
                
                if updates["status"] == "completed":
                    updates["completed_at"] = datetime.utcnow()
                
                job_ref.update(updates)
                
                logger.info("Updated job optimization results", job_id=job_id)
                
        except Exception as e:
            logger.error("Failed to update job optimization results", job_id=job_id, error=str(e))

# Initialize learning pipeline manager
learning_manager = LearningPipelineManager()

@app.post("/learning/start")
async def start_learning_job(request: LearningJobRequest, background_tasks: BackgroundTasks):
    """Start a new continuous learning job"""
    try:
        job_id = await learning_manager.start_learning_job(request)
        
        # Run learning job in background
        background_tasks.add_task(run_learning_job, job_id, request)
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": f"Learning job started for {request.agent_type} in {request.medical_domain}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_learning_job(job_id: str, request: LearningJobRequest):
    """Run the complete learning job pipeline"""
    try:
        logger.info("Starting learning job execution", job_id=job_id)
        
        # Update job status
        learning_jobs_collection.document(job_id).update({"status": "running"})
        
        # Get relevant datasets
        datasets = await learning_manager.get_relevant_datasets(
            request.medical_domain, 
            request.agent_type
        )
        
        if not datasets:
            learning_jobs_collection.document(job_id).update({
                "status": "failed",
                "error": "No relevant datasets found"
            })
            return
        
        all_test_cases = []
        datasets_used = []
        
        # Process each dataset
        for dataset in datasets[:3]:  # Limit to 3 datasets
            try:
                download_id = await learning_manager.download_and_prepare_dataset(
                    dataset["kaggle_ref"], 
                    request.medical_domain
                )
                
                test_cases = await learning_manager._generate_test_cases_from_dataset(
                    download_id, 
                    request.medical_domain
                )
                
                all_test_cases.extend(test_cases)
                datasets_used.append(dataset["kaggle_ref"])
                
            except Exception as e:
                logger.warning("Failed to process dataset", 
                             dataset_ref=dataset["kaggle_ref"], error=str(e))
                continue
        
        if not all_test_cases:
            learning_jobs_collection.document(job_id).update({
                "status": "failed",
                "error": "No test cases generated from datasets"
            })
            return
        
        # Update job with datasets used
        learning_jobs_collection.document(job_id).update({
            "datasets_used": datasets_used
        })
        
        # Run optimization
        optimization_result = await learning_manager.optimize_prompt_with_data(
            job_id, 
            request.prompt_template_id, 
            all_test_cases
        )
        
        logger.info("Learning job completed successfully", 
                   job_id=job_id, 
                   performance=optimization_result.get("final_performance", 0))
        
    except Exception as e:
        logger.error("Learning job failed", job_id=job_id, error=str(e))
        learning_jobs_collection.document(job_id).update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow()
        })

@app.get("/learning/jobs/{job_id}")
async def get_learning_job(job_id: str):
    """Get learning job status and results"""
    try:
        doc = learning_jobs_collection.document(job_id).get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Learning job not found")
        
        return doc.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning/jobs")
async def list_learning_jobs(status: Optional[str] = None, limit: int = 50):
    """List learning jobs with optional status filter"""
    try:
        query = learning_jobs_collection.order_by("created_at", direction=firestore.Query.DESCENDING)
        
        if status:
            query = query.where("status", "==", status)
        
        query = query.limit(limit)
        
        jobs = []
        for doc in query.stream():
            job_data = doc.to_dict()
            jobs.append({
                "job_id": job_data["job_id"],
                "agent_type": job_data["agent_type"],
                "medical_domain": job_data["medical_domain"],
                "status": job_data["status"],
                "best_performance": job_data.get("best_performance", 0),
                "iterations_completed": job_data.get("iterations_completed", 0),
                "created_at": job_data["created_at"],
                "completed_at": job_data.get("completed_at")
            })
        
        return {"jobs": jobs, "total": len(jobs)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/datasets/map")
async def create_dataset_mapping(mapping: DatasetMappingRequest):
    """Create mapping between Kaggle dataset and agent types"""
    try:
        mapping_id = f"mapping_{uuid.uuid4().hex[:12]}"
        
        mapping_data = {
            "mapping_id": mapping_id,
            "kaggle_ref": mapping.kaggle_ref,
            "medical_domain": mapping.medical_domain,
            "agent_types": mapping.agent_types,
            "prompt_variables": mapping.prompt_variables,
            "expected_outputs": mapping.expected_outputs,
            "created_at": datetime.utcnow(),
            "active": True
        }
        
        dataset_mapping_collection.document(mapping_id).set(mapping_data)
        
        return {"mapping_id": mapping_id, "status": "created"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets/mappings")
async def list_dataset_mappings(medical_domain: Optional[str] = None, agent_type: Optional[str] = None):
    """List dataset mappings with optional filters"""
    try:
        query = dataset_mapping_collection.where("active", "==", True)
        
        if medical_domain:
            query = query.where("medical_domain", "==", medical_domain)
        
        mappings = []
        for doc in query.stream():
            mapping_data = doc.to_dict()
            
            # Filter by agent_type if specified
            if agent_type and agent_type not in mapping_data.get("agent_types", []):
                continue
            
            mappings.append(mapping_data)
        
        return {"mappings": mappings, "total": len(mappings)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Firestore connection
        learning_jobs_collection.limit(1).get()
        
        # Test external service connections
        services_status = {}
        
        try:
            async with httpx.AsyncClient() as client:
                # Test Kaggle service
                kaggle_response = await client.get(f"{KAGGLE_SERVICE_URL}/health", timeout=5.0)
                services_status["kaggle_service"] = "healthy" if kaggle_response.status_code == 200 else "unhealthy"
        except:
            services_status["kaggle_service"] = "unreachable"
        
        try:
            async with httpx.AsyncClient() as client:
                # Test optimization service
                opt_response = await client.get(f"{OPTIMIZATION_SERVICE_URL}/health", timeout=5.0)
                services_status["optimization_service"] = "healthy" if opt_response.status_code == 200 else "unhealthy"
        except:
            services_status["optimization_service"] = "unreachable"
        
        return {
            "status": "healthy",
            "service": "learning-pipeline",
            "timestamp": datetime.utcnow().isoformat(),
            "services": services_status,
            "version": "1.0.0"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "learning-pipeline",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)