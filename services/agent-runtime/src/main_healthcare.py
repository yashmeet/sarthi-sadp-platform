"""
SADP Healthcare AI Platform - Production API
Real Gemini AI integration with healthcare-specific capabilities
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import os
import base64

# Import healthcare AI service
from healthcare_ai_service import get_healthcare_service
from poml_api import router as poml_router

app = FastAPI(
    title="SADP Healthcare AI Platform",
    version="3.0.0",
    description="Production Healthcare AI Platform with Gemini API Integration"
)

# Include POML Studio routes
app.include_router(poml_router)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get healthcare service
healthcare_service = get_healthcare_service()

# Audit trail storage
audit_trail = []

class ClinicalDocumentRequest(BaseModel):
    document_text: str
    document_type: str = "general"
    patient_id: Optional[str] = None

class MedicalImageRequest(BaseModel):
    image_data: str  # Base64 encoded image
    image_type: str = "general"
    patient_id: Optional[str] = None

class MedicationListRequest(BaseModel):
    medications: List[str]
    patient_id: Optional[str] = None

class MedicalCodesRequest(BaseModel):
    codes: List[str]
    code_type: str = "icd10"

class PatientSummaryRequest(BaseModel):
    patient_data: Dict[str, Any]

@app.get("/")
async def root():
    """Health check and API info"""
    return {
        "service": "SADP Healthcare AI Platform",
        "version": "3.0.0",
        "status": "healthy",
        "ai_enabled": healthcare_service.ai_available,
        "model": "gemini-1.5-flash" if healthcare_service.ai_available else "pattern-based",
        "api_key_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "clinical_document": "/analyze/document",
            "medical_image": "/analyze/image",
            "drug_interactions": "/analyze/medications",
            "medical_coding": "/validate/codes",
            "clinical_summary": "/generate/summary"
        }
    }

@app.get("/setup", response_class=HTMLResponse)
async def setup_page():
    """Provide setup instructions for Gemini API"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SADP Healthcare AI - Setup</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; }
            .warning { background: #fff3cd; color: #856404; }
            .error { background: #f8d7da; color: #721c24; }
            code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
            pre { background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
            .step { margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; }
        </style>
    </head>
    <body>
        <h1>🏥 SADP Healthcare AI Platform Setup</h1>
        
        <div class="status """ + ("success" if healthcare_service.ai_available else "warning") + """">
            <strong>Status:</strong> """ + ("✅ Gemini AI is configured and ready!" if healthcare_service.ai_available else "⚠️ Gemini API key not configured") + """
        </div>
        
        <h2>Setup Instructions</h2>
        
        <div class="step">
            <h3>Step 1: Get a Free Gemini API Key</h3>
            <p>1. Visit <a href="https://makersuite.google.com/app/apikey" target="_blank">Google AI Studio</a></p>
            <p>2. Sign in with your Google account</p>
            <p>3. Click "Create API Key"</p>
            <p>4. Copy your API key</p>
        </div>
        
        <div class="step">
            <h3>Step 2: Set the API Key in Google Secret Manager</h3>
            <pre>echo "YOUR_ACTUAL_API_KEY" | gcloud secrets versions add gemini-api-key --data-file=-</pre>
        </div>
        
        <div class="step">
            <h3>Step 3: Redeploy the Service</h3>
            <pre>gcloud run deploy sadp-agent-runtime-ai \\
    --image=us-central1-docker.pkg.dev/sarthi-patient-experience-hub/sadp-runtime/agent-runtime-ai:latest \\
    --region=us-central1 \\
    --set-secrets="GEMINI_API_KEY=gemini-api-key:latest"</pre>
        </div>
        
        <h2>Available Healthcare AI Features</h2>
        <ul>
            <li>📄 Clinical Document Analysis (discharge summaries, lab reports, radiology)</li>
            <li>🖼️ Medical Image Processing (X-ray, MRI, CT, ECG)</li>
            <li>💊 Drug Interaction Analysis</li>
            <li>🏥 Medical Code Validation (ICD-10, CPT)</li>
            <li>📋 Clinical Summary Generation</li>
            <li>🔍 Medical Entity Extraction</li>
        </ul>
        
        <h2>Test the API</h2>
        <p>Once configured, test with:</p>
        <pre>curl -X POST https://sadp-agent-runtime-ai-355881591332.us-central1.run.app/analyze/document \\
    -H "Content-Type: application/json" \\
    -d '{"document_text": "Patient diagnosed with Type 2 Diabetes (E11.9)", "document_type": "clinical_note"}'</pre>
    </body>
    </html>
    """
    return html_content

@app.post("/analyze/document")
async def analyze_clinical_document(request: ClinicalDocumentRequest):
    """Analyze clinical documents using Gemini AI"""
    
    # Log request
    audit_trail.append({
        "event": "document_analysis_requested",
        "document_type": request.document_type,
        "patient_id": request.patient_id,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    try:
        result = await healthcare_service.analyze_clinical_document(
            request.document_text,
            request.document_type
        )
        
        # Log success
        audit_trail.append({
            "event": "document_analysis_completed",
            "document_type": request.document_type,
            "entities_found": result.get("metadata", {}).get("entities_found", 0),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return result
        
    except Exception as e:
        # Log error
        audit_trail.append({
            "event": "document_analysis_failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/image")
async def analyze_medical_image(request: MedicalImageRequest):
    """Analyze medical images using Gemini Vision API"""
    
    if not healthcare_service.ai_available:
        raise HTTPException(
            status_code=503,
            detail="Gemini AI not configured. Please set up API key first."
        )
    
    try:
        result = await healthcare_service.process_medical_image(
            request.image_data,
            request.image_type
        )
        
        # Log analysis
        audit_trail.append({
            "event": "image_analysis_completed",
            "image_type": request.image_type,
            "patient_id": request.patient_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/medications")
async def analyze_drug_interactions(request: MedicationListRequest):
    """Analyze drug interactions using Gemini AI"""
    
    try:
        result = await healthcare_service.analyze_drug_interactions(
            request.medications
        )
        
        # Log analysis
        audit_trail.append({
            "event": "drug_interaction_analysis",
            "medications_count": len(request.medications),
            "patient_id": request.patient_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate/codes")
async def validate_medical_codes(request: MedicalCodesRequest):
    """Validate medical codes (ICD-10, CPT) using Gemini AI"""
    
    try:
        result = await healthcare_service.validate_medical_coding(
            request.codes,
            request.code_type
        )
        
        # Log validation
        audit_trail.append({
            "event": "medical_code_validation",
            "code_type": request.code_type,
            "codes_count": len(request.codes),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/summary")
async def generate_clinical_summary(request: PatientSummaryRequest):
    """Generate clinical summary from patient data"""
    
    try:
        result = await healthcare_service.generate_clinical_summary(
            request.patient_data
        )
        
        # Log generation
        audit_trail.append({
            "event": "clinical_summary_generated",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/file")
async def analyze_medical_file(
    file: UploadFile = File(...),
    document_type: str = Form("general")
):
    """Analyze uploaded medical documents"""
    
    # Read file content
    content = await file.read()
    
    # Handle different file types
    if file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        # For images and PDFs, encode as base64
        image_data = base64.b64encode(content).decode('utf-8')
        result = await healthcare_service.process_medical_image(
            image_data,
            document_type
        )
    else:
        # For text files
        try:
            text_content = content.decode('utf-8')
            result = await healthcare_service.analyze_clinical_document(
                text_content,
                document_type
            )
        except:
            raise HTTPException(status_code=400, detail="Unable to process file")
    
    return {
        "filename": file.filename,
        "file_size": len(content),
        "analysis": result
    }

@app.get("/audit/trail")
async def get_audit_trail(limit: int = 100):
    """Get audit trail for HIPAA compliance"""
    return {
        "events": audit_trail[-limit:],
        "total_events": len(audit_trail),
        "retrieved": min(limit, len(audit_trail))
    }

@app.get("/ai/status")
async def get_ai_status():
    """Check AI service status"""
    
    test_text = "Patient presents with hypertension (I10) and prescribed Lisinopril 10mg daily."
    
    try:
        # Test the service
        test_result = await healthcare_service.analyze_clinical_document(
            test_text,
            "clinical_note"
        )
        
        operational = test_result.get("status") == "success"
    except:
        operational = False
    
    return {
        "ai_service": "Gemini API",
        "model": "gemini-1.5-flash",
        "api_configured": healthcare_service.ai_available,
        "operational": operational,
        "api_key_present": bool(os.environ.get("GEMINI_API_KEY")),
        "capabilities": [
            "clinical_document_analysis",
            "medical_image_processing",
            "drug_interaction_analysis",
            "medical_code_validation",
            "clinical_summary_generation"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/examples")
async def get_examples():
    """Get example requests for testing"""
    return {
        "clinical_document": {
            "endpoint": "POST /analyze/document",
            "body": {
                "document_text": "Patient: John Doe, 55 years old. Diagnosis: Type 2 Diabetes Mellitus (E11.9). Prescribed Metformin 500mg twice daily. HbA1c: 7.2%. Blood pressure: 130/80 mmHg.",
                "document_type": "clinical_note"
            }
        },
        "drug_interactions": {
            "endpoint": "POST /analyze/medications",
            "body": {
                "medications": ["Warfarin", "Aspirin", "Metformin", "Lisinopril"]
            }
        },
        "medical_codes": {
            "endpoint": "POST /validate/codes",
            "body": {
                "codes": ["E11.9", "I10", "J44.0"],
                "code_type": "icd10"
            }
        },
        "lab_report": {
            "endpoint": "POST /analyze/document",
            "body": {
                "document_text": "CBC Results: WBC 7.5 K/uL (Normal: 4.5-11.0), Hemoglobin 13.2 g/dL (Low, Normal: 14-18), Platelets 250 K/uL (Normal: 150-400)",
                "document_type": "lab_report"
            }
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus-style metrics"""
    
    total_analyses = len([e for e in audit_trail if "analysis" in e.get("event", "")])
    successful_analyses = len([e for e in audit_trail if "completed" in e.get("event", "")])
    failed_analyses = len([e for e in audit_trail if "failed" in e.get("event", "")])
    
    metrics = f"""# HELP sadp_healthcare_analyses_total Total number of healthcare analyses
# TYPE sadp_healthcare_analyses_total counter
sadp_healthcare_analyses_total {total_analyses}

# HELP sadp_healthcare_analyses_successful Successful healthcare analyses
# TYPE sadp_healthcare_analyses_successful counter
sadp_healthcare_analyses_successful {successful_analyses}

# HELP sadp_healthcare_analyses_failed Failed healthcare analyses
# TYPE sadp_healthcare_analyses_failed counter
sadp_healthcare_analyses_failed {failed_analyses}

# HELP sadp_ai_service_up Healthcare AI service availability
# TYPE sadp_ai_service_up gauge
sadp_ai_service_up {1 if healthcare_service.ai_available else 0}
"""
    
    return metrics

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)