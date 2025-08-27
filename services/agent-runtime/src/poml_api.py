"""
POML Studio API Endpoints
Real-time prompt engineering and testing with Gemini AI
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import asyncio
import uuid
from google.cloud import firestore, storage
import google.generativeai as genai
import os
import re
import time

router = APIRouter(prefix="/poml", tags=["POML Studio"])

# Initialize clients
firestore_client = None
storage_client = None
templates_bucket = None

# WebSocket connections for live testing
active_connections: Dict[str, WebSocket] = {}

# Initialize Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

class POMLTemplate(BaseModel):
    name: str
    agent_type: str
    description: str
    content: str
    variables: List[str]
    tags: List[str] = []
    version: str = "1.0"

class POMLTestRequest(BaseModel):
    template_id: Optional[str] = None
    template_content: Optional[str] = None
    variables: Dict[str, Any]
    test_data: str
    use_streaming: bool = False

class POMLABTestRequest(BaseModel):
    name: str
    template_a_id: str
    template_b_id: str
    test_cases: List[Dict[str, Any]]
    iterations: int = 10
    success_metric: str = "accuracy"

class POMLUpdateRequest(BaseModel):
    content: Optional[str] = None
    description: Optional[str] = None
    variables: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    version: Optional[str] = None

def initialize_clients():
    """Initialize GCP clients"""
    global firestore_client, storage_client, templates_bucket
    
    try:
        project_id = os.environ.get("GCP_PROJECT_ID", "sarthi-patient-experience-hub")
        firestore_client = firestore.Client(project=project_id)
        storage_client = storage.Client(project=project_id)
        
        # Create bucket if it doesn't exist
        bucket_name = f"{project_id}-poml-templates"
        try:
            templates_bucket = storage_client.bucket(bucket_name)
            if not templates_bucket.exists():
                templates_bucket = storage_client.create_bucket(bucket_name, location="us-central1")
        except:
            templates_bucket = storage_client.create_bucket(bucket_name, location="us-central1")
            
    except Exception as e:
        print(f"Failed to initialize GCP clients: {e}")

# Initialize on module load
initialize_clients()

def parse_poml_template(content: str) -> Dict[str, Any]:
    """Parse POML XML template and extract components"""
    parsed = {
        'sections': {},
        'variables': [],
        'constraints': [],
        'output_format': 'text',
        'version': '1.0'
    }
    
    # Extract version
    version_match = re.search(r'<prompt\s+version="([^"]+)"', content)
    if version_match:
        parsed['version'] = version_match.group(1)
    
    # Extract main sections
    sections = ['system', 'context', 'task', 'constraints', 'examples', 'output']
    for section in sections:
        pattern = f'<{section}[^>]*>(.*?)</{section}>'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            parsed['sections'][section] = match.group(1).strip()
    
    # Extract variables ({{variable_name}})
    variables = re.findall(r'\{\{(\w+)\}\}', content)
    parsed['variables'] = list(set(variables))
    
    # Extract output format
    output_match = re.search(r'<output\s+format="([^"]+)"', content)
    if output_match:
        parsed['output_format'] = output_match.group(1)
    
    return parsed

def render_poml_template(content: str, variables: Dict[str, Any]) -> str:
    """Render POML template with provided variables"""
    rendered = content
    
    # Replace variables
    for var_name, var_value in variables.items():
        placeholder = f"{{{{{var_name}}}}}"
        rendered = rendered.replace(placeholder, str(var_value))
    
    # Remove POML tags for execution
    rendered = re.sub(r'<prompt[^>]*>', '', rendered)
    rendered = re.sub(r'</prompt>', '', rendered)
    rendered = re.sub(r'<(\w+)[^>]*>', r'\n--- \1 ---\n', rendered)
    rendered = re.sub(r'</\w+>', '', rendered)
    
    return rendered.strip()

@router.post("/templates")
async def create_template(template: POMLTemplate):
    """Create a new POML template"""
    try:
        if not firestore_client:
            raise HTTPException(status_code=503, detail="Database not initialized")
        
        # Generate template ID
        template_id = f"{template.agent_type}_{template.name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        
        # Parse template
        parsed = parse_poml_template(template.content)
        
        # Create document
        doc_data = {
            "id": template_id,
            "name": template.name,
            "agent_type": template.agent_type,
            "description": template.description,
            "content": template.content,
            "variables": parsed['variables'],
            "tags": template.tags,
            "version": template.version,
            "parsed": parsed,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "executions": 0,
            "avg_latency": 0,
            "success_rate": 0
        }
        
        # Store in Firestore
        firestore_client.collection("poml_templates").document(template_id).set(doc_data)
        
        # Store in Cloud Storage
        if templates_bucket:
            blob = templates_bucket.blob(f"templates/{template_id}.poml")
            blob.upload_from_string(template.content)
        
        return {"template_id": template_id, "status": "created", "parsed": parsed}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates")
async def list_templates(agent_type: Optional[str] = None, tag: Optional[str] = None):
    """List all POML templates"""
    try:
        if not firestore_client:
            raise HTTPException(status_code=503, detail="Database not initialized")
        
        query = firestore_client.collection("poml_templates")
        
        if agent_type:
            query = query.where("agent_type", "==", agent_type)
        
        if tag:
            query = query.where("tags", "array_contains", tag)
        
        templates = []
        for doc in query.stream():
            data = doc.to_dict()
            templates.append({
                "id": data["id"],
                "name": data["name"],
                "agent_type": data["agent_type"],
                "description": data["description"],
                "tags": data.get("tags", []),
                "version": data.get("version", "1.0"),
                "executions": data.get("executions", 0),
                "success_rate": data.get("success_rate", 0),
                "created_at": data.get("created_at")
            })
        
        return {"templates": templates, "count": len(templates)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates/{template_id}")
async def get_template(template_id: str):
    """Get a specific POML template"""
    try:
        if not firestore_client:
            raise HTTPException(status_code=503, detail="Database not initialized")
        
        doc = firestore_client.collection("poml_templates").document(template_id).get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return doc.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/templates/{template_id}")
async def update_template(template_id: str, updates: POMLUpdateRequest):
    """Update a POML template"""
    try:
        if not firestore_client:
            raise HTTPException(status_code=503, detail="Database not initialized")
        
        doc_ref = firestore_client.collection("poml_templates").document(template_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Template not found")
        
        # Prepare updates
        update_data = {"updated_at": datetime.utcnow().isoformat()}
        
        if updates.content:
            update_data["content"] = updates.content
            parsed = parse_poml_template(updates.content)
            update_data["parsed"] = parsed
            update_data["variables"] = parsed['variables']
            
            # Update in Cloud Storage
            if templates_bucket:
                blob = templates_bucket.blob(f"templates/{template_id}.poml")
                blob.upload_from_string(updates.content)
        
        if updates.description:
            update_data["description"] = updates.description
        
        if updates.tags is not None:
            update_data["tags"] = updates.tags
        
        if updates.version:
            update_data["version"] = updates.version
            
            # Create version backup
            if templates_bucket and updates.content:
                version_blob = templates_bucket.blob(
                    f"templates/{template_id}_v{updates.version}.poml"
                )
                version_blob.upload_from_string(updates.content)
        
        # Update Firestore
        doc_ref.update(update_data)
        
        return {"template_id": template_id, "status": "updated", "updates": update_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test")
async def test_template(request: POMLTestRequest):
    """Test a POML template with real data using Gemini"""
    try:
        start_time = time.time()
        
        # Get template content
        if request.template_id:
            if not firestore_client:
                raise HTTPException(status_code=503, detail="Database not initialized")
            
            doc = firestore_client.collection("poml_templates").document(request.template_id).get()
            if not doc.exists:
                raise HTTPException(status_code=404, detail="Template not found")
            
            template_content = doc.to_dict()["content"]
        else:
            template_content = request.template_content
        
        if not template_content:
            raise HTTPException(status_code=400, detail="No template provided")
        
        # Render template with variables
        prompt = render_poml_template(template_content, request.variables)
        
        # Add test data to prompt
        full_prompt = f"{prompt}\n\nInput Data:\n{request.test_data}"
        
        # Execute with Gemini
        if gemini_model:
            try:
                response = gemini_model.generate_content(
                    full_prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,
                        top_p=0.8,
                        max_output_tokens=2048,
                    )
                )
                
                result = response.text
                tokens_used = len(full_prompt.split()) + len(result.split())  # Approximate
                success = True
                
            except Exception as e:
                result = f"Gemini API error: {str(e)}"
                tokens_used = 0
                success = False
        else:
            # Fallback to mock response
            result = f"Mock response for prompt:\n{full_prompt[:200]}...\n\nTest data processed successfully."
            tokens_used = len(full_prompt.split())
            success = True
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        # Update metrics if template_id provided
        if request.template_id and firestore_client:
            doc_ref = firestore_client.collection("poml_templates").document(request.template_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                executions = data.get("executions", 0) + 1
                avg_latency = ((data.get("avg_latency", 0) * data.get("executions", 0)) + latency) / executions
                success_rate = ((data.get("success_rate", 0) * data.get("executions", 0)) + (1 if success else 0)) / executions
                
                doc_ref.update({
                    "executions": executions,
                    "avg_latency": avg_latency,
                    "success_rate": success_rate,
                    "last_tested": datetime.utcnow().isoformat()
                })
        
        return {
            "result": result,
            "success": success,
            "latency_ms": latency,
            "tokens_used": tokens_used,
            "prompt_preview": full_prompt[:500] + "..." if len(full_prompt) > 500 else full_prompt
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ab-test")
async def create_ab_test(request: POMLABTestRequest, background_tasks: BackgroundTasks):
    """Create and run an A/B test between two templates"""
    try:
        if not firestore_client:
            raise HTTPException(status_code=503, detail="Database not initialized")
        
        # Get both templates
        doc_a = firestore_client.collection("poml_templates").document(request.template_a_id).get()
        doc_b = firestore_client.collection("poml_templates").document(request.template_b_id).get()
        
        if not doc_a.exists or not doc_b.exists:
            raise HTTPException(status_code=404, detail="One or both templates not found")
        
        template_a = doc_a.to_dict()["content"]
        template_b = doc_b.to_dict()["content"]
        
        # Create test ID
        test_id = f"ab_test_{uuid.uuid4().hex[:12]}"
        
        # Initialize test document
        test_doc = {
            "id": test_id,
            "name": request.name,
            "template_a_id": request.template_a_id,
            "template_b_id": request.template_b_id,
            "status": "running",
            "created_at": datetime.utcnow().isoformat(),
            "results": {
                "template_a": {"executions": 0, "successes": 0, "total_latency": 0, "total_tokens": 0},
                "template_b": {"executions": 0, "successes": 0, "total_latency": 0, "total_tokens": 0}
            }
        }
        
        firestore_client.collection("poml_ab_tests").document(test_id).set(test_doc)
        
        # Run test in background
        background_tasks.add_task(run_ab_test, test_id, template_a, template_b, request.test_cases, request.iterations)
        
        return {"test_id": test_id, "status": "started", "message": "A/B test running in background"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_ab_test(test_id: str, template_a: str, template_b: str, test_cases: List[Dict], iterations: int):
    """Run A/B test in background"""
    try:
        doc_ref = firestore_client.collection("poml_ab_tests").document(test_id)
        
        for i in range(iterations):
            for test_case in test_cases:
                # Test template A
                start_a = time.time()
                prompt_a = render_poml_template(template_a, test_case.get("variables", {}))
                full_prompt_a = f"{prompt_a}\n\nInput: {test_case.get('input', '')}"
                
                if gemini_model:
                    try:
                        response_a = gemini_model.generate_content(full_prompt_a)
                        result_a = response_a.text
                        success_a = True
                    except:
                        result_a = "Error"
                        success_a = False
                else:
                    result_a = "Mock response A"
                    success_a = True
                
                latency_a = (time.time() - start_a) * 1000
                
                # Test template B
                start_b = time.time()
                prompt_b = render_poml_template(template_b, test_case.get("variables", {}))
                full_prompt_b = f"{prompt_b}\n\nInput: {test_case.get('input', '')}"
                
                if gemini_model:
                    try:
                        response_b = gemini_model.generate_content(full_prompt_b)
                        result_b = response_b.text
                        success_b = True
                    except:
                        result_b = "Error"
                        success_b = False
                else:
                    result_b = "Mock response B"
                    success_b = True
                
                latency_b = (time.time() - start_b) * 1000
                
                # Update results
                doc = doc_ref.get()
                if doc.exists:
                    data = doc.to_dict()
                    results = data["results"]
                    
                    results["template_a"]["executions"] += 1
                    results["template_a"]["successes"] += 1 if success_a else 0
                    results["template_a"]["total_latency"] += latency_a
                    
                    results["template_b"]["executions"] += 1
                    results["template_b"]["successes"] += 1 if success_b else 0
                    results["template_b"]["total_latency"] += latency_b
                    
                    doc_ref.update({"results": results})
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
        
        # Mark test as complete
        doc_ref.update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        doc_ref.update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat()
        })

@router.get("/ab-test/{test_id}")
async def get_ab_test_results(test_id: str):
    """Get A/B test results"""
    try:
        if not firestore_client:
            raise HTTPException(status_code=503, detail="Database not initialized")
        
        doc = firestore_client.collection("poml_ab_tests").document(test_id).get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Test not found")
        
        data = doc.to_dict()
        results = data["results"]
        
        # Calculate statistics
        analysis = {
            "test_id": test_id,
            "name": data["name"],
            "status": data["status"],
            "created_at": data["created_at"],
            "template_a": {
                "id": data["template_a_id"],
                "executions": results["template_a"]["executions"],
                "success_rate": results["template_a"]["successes"] / max(results["template_a"]["executions"], 1),
                "avg_latency": results["template_a"]["total_latency"] / max(results["template_a"]["executions"], 1)
            },
            "template_b": {
                "id": data["template_b_id"],
                "executions": results["template_b"]["executions"],
                "success_rate": results["template_b"]["successes"] / max(results["template_b"]["executions"], 1),
                "avg_latency": results["template_b"]["total_latency"] / max(results["template_b"]["executions"], 1)
            }
        }
        
        # Determine winner
        if data["status"] == "completed":
            a_score = analysis["template_a"]["success_rate"] - (analysis["template_a"]["avg_latency"] / 10000)
            b_score = analysis["template_b"]["success_rate"] - (analysis["template_b"]["avg_latency"] / 10000)
            
            if abs(a_score - b_score) > 0.05:
                analysis["winner"] = "template_a" if a_score > b_score else "template_b"
                analysis["confidence"] = abs(a_score - b_score)
            else:
                analysis["winner"] = "no_significant_difference"
                analysis["confidence"] = 0
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/live-test")
async def websocket_live_test(websocket: WebSocket):
    """WebSocket endpoint for live template testing"""
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    active_connections[connection_id] = websocket
    
    try:
        while True:
            # Receive test request
            data = await websocket.receive_json()
            
            if data.get("action") == "test":
                template_content = data.get("template_content", "")
                variables = data.get("variables", {})
                test_data = data.get("test_data", "")
                
                # Send status
                await websocket.send_json({"status": "processing", "message": "Rendering template..."})
                
                # Render template
                prompt = render_poml_template(template_content, variables)
                full_prompt = f"{prompt}\n\nInput: {test_data}"
                
                # Send rendered prompt
                await websocket.send_json({
                    "status": "rendered",
                    "prompt": full_prompt[:500] + "..." if len(full_prompt) > 500 else full_prompt
                })
                
                # Execute with Gemini
                await websocket.send_json({"status": "executing", "message": "Calling Gemini API..."})
                
                if gemini_model:
                    try:
                        response = gemini_model.generate_content(full_prompt)
                        result = response.text
                        success = True
                    except Exception as e:
                        result = f"Error: {str(e)}"
                        success = False
                else:
                    result = f"Mock response for: {test_data[:100]}..."
                    success = True
                
                # Send result
                await websocket.send_json({
                    "status": "complete",
                    "result": result,
                    "success": success
                })
            
            elif data.get("action") == "ping":
                await websocket.send_json({"status": "pong"})
                
    except WebSocketDisconnect:
        del active_connections[connection_id]
    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})
        del active_connections[connection_id]

@router.get("/metrics/{template_id}")
async def get_template_metrics(template_id: str):
    """Get performance metrics for a template"""
    try:
        if not firestore_client:
            raise HTTPException(status_code=503, detail="Database not initialized")
        
        doc = firestore_client.collection("poml_templates").document(template_id).get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Template not found")
        
        data = doc.to_dict()
        
        return {
            "template_id": template_id,
            "name": data["name"],
            "executions": data.get("executions", 0),
            "success_rate": data.get("success_rate", 0),
            "avg_latency_ms": data.get("avg_latency", 0),
            "last_tested": data.get("last_tested"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/marketplace")
async def get_marketplace_templates():
    """Get featured templates from marketplace"""
    try:
        if not firestore_client:
            # Return mock data if database not available
            return {
                "featured": [
                    {
                        "id": "clinical_diagnosis_v2",
                        "name": "Clinical Diagnosis Assistant",
                        "description": "Advanced diagnostic support with ICD-10 coding",
                        "agent_type": "clinical",
                        "downloads": 1250,
                        "rating": 4.8,
                        "author": "SADP Team"
                    },
                    {
                        "id": "medication_reconciliation_v1",
                        "name": "Medication Reconciliation",
                        "description": "Accurate medication history and interaction checking",
                        "agent_type": "medication",
                        "downloads": 890,
                        "rating": 4.9,
                        "author": "PharmaTech"
                    }
                ],
                "categories": ["clinical", "billing", "documentation", "medication", "lab_analysis"]
            }
        
        # Query top templates by downloads/rating
        query = firestore_client.collection("poml_templates").order_by("executions", direction=firestore.Query.DESCENDING).limit(10)
        
        featured = []
        for doc in query.stream():
            data = doc.to_dict()
            featured.append({
                "id": data["id"],
                "name": data["name"],
                "description": data["description"],
                "agent_type": data["agent_type"],
                "executions": data.get("executions", 0),
                "success_rate": data.get("success_rate", 0),
                "tags": data.get("tags", [])
            })
        
        return {"featured": featured, "total": len(featured)}
        
    except Exception as e:
        # Return mock data on error
        return {
            "featured": [],
            "error": str(e),
            "message": "Using mock data"
        }