"""
Simplified POML API for template management
No authentication required - direct access for development
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import uuid
import re
import google.generativeai as genai
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import structlog

# Initialize Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

router = APIRouter(prefix="/poml", tags=["POML Studio"])

# Models
class POMLTemplate(BaseModel):
    name: str
    description: str
    content: str
    category: str = "general"
    tags: List[str] = []
    
class POMLUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    
class TestRequest(BaseModel):
    content: str
    variables: Dict[str, Any]
    test_input: str

# Initialize logger
logger = structlog.get_logger()

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:changeme123@localhost/sarthi")

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def init_database():
    """Initialize database tables"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create templates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS poml_templates (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                content TEXT NOT NULL,
                category VARCHAR(100) DEFAULT 'general',
                tags TEXT[], -- PostgreSQL array type
                version VARCHAR(20) DEFAULT '1.0.0',
                status VARCHAR(20) DEFAULT 'draft',
                published BOOLEAN DEFAULT FALSE,
                archived BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create template versions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS poml_template_versions (
                id SERIAL PRIMARY KEY,
                template_id VARCHAR(255) REFERENCES poml_templates(id) ON DELETE CASCADE,
                version VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                changelog TEXT
            )
        ''')
        
        # Create index for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_templates_category ON poml_templates(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_templates_status ON poml_templates(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_templates_created_at ON poml_templates(created_at DESC)')
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database tables initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Fall back to in-memory storage for demo
        global templates_db, versions_db
        templates_db = {}
        versions_db = {}

# Initialize database on module load
init_database()

# Fallback in-memory storage for demo environments
templates_db = {}
versions_db = {}

def extract_variables(content: str) -> List[Dict[str, str]]:
    """Extract variables from POML template"""
    # Match {{variable}} or {{variable:type}}
    pattern = r'\{\{(\w+)(?::([^}]+))?\}\}'
    matches = re.findall(pattern, content)
    return [{'name': match[0], 'type': match[1] or 'string'} for match in matches]

def render_template(content: str, variables: Dict[str, Any]) -> str:
    """Render POML template with variables"""
    rendered = content
    for key, value in variables.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered

def generate_version(major: int = 1, minor: int = 0, patch: int = 0) -> str:
    """Generate semantic version string"""
    return f"{major}.{minor}.{patch}"

@router.post("/templates", response_model=Dict[str, Any])
async def create_template(template: POMLTemplate):
    """Create a new POML template"""
    template_id = f"tpl_{uuid.uuid4().hex[:12]}"
    
    # Extract variables
    variables = extract_variables(template.content)
    
    try:
        # Try database first
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO poml_templates (id, name, description, content, category, tags)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
        ''', (template_id, template.name, template.description, template.content, 
              template.category, template.tags))
        
        result = cursor.fetchone()
        
        # Add initial version
        cursor.execute('''
            INSERT INTO poml_template_versions (template_id, version, content, changelog)
            VALUES (%s, %s, %s, %s)
        ''', (template_id, "1.0.0", template.content, "Initial version"))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Convert database result to response format
        template_data = dict(result)
        template_data["variables"] = variables
        template_data["created_at"] = template_data["created_at"].isoformat() if template_data["created_at"] else None
        template_data["updated_at"] = template_data["updated_at"].isoformat() if template_data["updated_at"] else None
        
        logger.info(f"Template {template_id} created successfully in database")
        return template_data
        
    except Exception as e:
        logger.warning(f"Database error, falling back to in-memory storage: {e}")
        
        # Fallback to in-memory storage
        template_data = {
            "id": template_id,
            "name": template.name,
            "description": template.description,
            "content": template.content,
            "category": template.category,
            "tags": template.tags,
            "variables": variables,
            "version": "1.0.0",
            "status": "draft",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "published": False,
            "archived": False
        }
        
        # Store template
        templates_db[template_id] = template_data
        
        # Initialize version history
        versions_db[template_id] = [{
            "version": "1.0.0",
            "content": template.content,
            "created_at": datetime.utcnow().isoformat(),
            "message": "Initial version"
        }]
        
        return template_data

@router.get("/templates", response_model=Dict[str, Any])
async def list_templates(
    category: Optional[str] = None,
    status: Optional[str] = None,
    include_archived: bool = False
):
    """List all templates with filtering"""
    try:
        # Try database first
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build query with filters
        query = "SELECT * FROM poml_templates WHERE 1=1"
        params = []
        
        if not include_archived:
            query += " AND archived = FALSE"
        
        if category:
            query += " AND category = %s"
            params.append(category)
            
        if status:
            query += " AND status = %s" 
            params.append(status)
            
        query += " ORDER BY updated_at DESC"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Convert database results to response format
        templates = []
        for row in results:
            template_data = dict(row)
            template_data["created_at"] = template_data["created_at"].isoformat() if template_data["created_at"] else None
            template_data["updated_at"] = template_data["updated_at"].isoformat() if template_data["updated_at"] else None
            template_data["variables"] = extract_variables(template_data["content"])
            templates.append(template_data)
        
        logger.info(f"Retrieved {len(templates)} templates from database")
        return {
            "templates": templates,
            "total": len(templates)
        }
        
    except Exception as e:
        logger.warning(f"Database error, falling back to in-memory storage: {e}")
        
        # Fallback to in-memory storage
        templates = []
        
        for template in templates_db.values():
            # Apply filters
            if not include_archived and template.get("archived", False):
                continue
            if category and template.get("category") != category:
                continue
            if status and template.get("status") != status:
                continue
                
            templates.append(template)
        
        # Sort by updated_at
        templates.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return {
            "templates": templates,
            "total": len(templates)
        }

@router.get("/templates/{template_id}", response_model=Dict[str, Any])
async def get_template(template_id: str):
    """Get a specific template with version history"""
    if template_id not in templates_db:
        raise HTTPException(status_code=404, detail="Template not found")
    
    template = templates_db[template_id].copy()
    template["versions"] = versions_db.get(template_id, [])
    
    return template

@router.put("/templates/{template_id}", response_model=Dict[str, Any])
async def update_template(template_id: str, update: POMLUpdate):
    """Update a template and create new version"""
    if template_id not in templates_db:
        raise HTTPException(status_code=404, detail="Template not found")
    
    template = templates_db[template_id]
    
    # Track if content changed for versioning
    content_changed = False
    
    # Update fields
    if update.name is not None:
        template["name"] = update.name
    if update.description is not None:
        template["description"] = update.description
    if update.content is not None and update.content != template["content"]:
        content_changed = True
        template["content"] = update.content
        template["variables"] = extract_variables(update.content)
    if update.category is not None:
        template["category"] = update.category
    if update.tags is not None:
        template["tags"] = update.tags
    
    template["updated_at"] = datetime.utcnow().isoformat()
    
    # Create new version if content changed
    if content_changed:
        # Parse current version
        current = template["version"]
        parts = current.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        # Increment patch version
        new_version = generate_version(major, minor, patch + 1)
        template["version"] = new_version
        
        # Add to version history
        if template_id not in versions_db:
            versions_db[template_id] = []
            
        versions_db[template_id].append({
            "version": new_version,
            "content": update.content,
            "created_at": datetime.utcnow().isoformat(),
            "message": f"Updated content"
        })
    
    return template

@router.delete("/templates/{template_id}")
async def delete_template(template_id: str):
    """Delete a template"""
    if template_id not in templates_db:
        raise HTTPException(status_code=404, detail="Template not found")
    
    del templates_db[template_id]
    if template_id in versions_db:
        del versions_db[template_id]
    
    return {"message": "Template deleted successfully"}

@router.patch("/templates/{template_id}/archive")
async def toggle_archive(template_id: str):
    """Archive or unarchive a template"""
    if template_id not in templates_db:
        raise HTTPException(status_code=404, detail="Template not found")
    
    template = templates_db[template_id]
    template["archived"] = not template.get("archived", False)
    template["updated_at"] = datetime.utcnow().isoformat()
    
    return {
        "id": template_id,
        "archived": template["archived"],
        "message": f"Template {'archived' if template['archived'] else 'unarchived'} successfully"
    }

@router.patch("/templates/{template_id}/publish")
async def publish_template(template_id: str):
    """Publish a template"""
    if template_id not in templates_db:
        raise HTTPException(status_code=404, detail="Template not found")
    
    template = templates_db[template_id]
    template["published"] = True
    template["status"] = "published"
    template["published_at"] = datetime.utcnow().isoformat()
    template["updated_at"] = datetime.utcnow().isoformat()
    
    # Increment minor version for publish
    parts = template["version"].split(".")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    new_version = generate_version(major, minor + 1, 0)
    template["version"] = new_version
    
    # Add to version history
    if template_id not in versions_db:
        versions_db[template_id] = []
        
    versions_db[template_id].append({
        "version": new_version,
        "content": template["content"],
        "created_at": datetime.utcnow().isoformat(),
        "message": "Published version"
    })
    
    return {
        "id": template_id,
        "published": True,
        "version": new_version,
        "message": "Template published successfully"
    }

@router.get("/templates/{template_id}/versions")
async def get_versions(template_id: str):
    """Get version history for a template"""
    if template_id not in templates_db:
        raise HTTPException(status_code=404, detail="Template not found")
    
    versions = versions_db.get(template_id, [])
    
    return {
        "template_id": template_id,
        "current_version": templates_db[template_id]["version"],
        "versions": versions,
        "total": len(versions)
    }

@router.post("/templates/{template_id}/revert/{version}")
async def revert_to_version(template_id: str, version: str):
    """Revert template to a specific version"""
    if template_id not in templates_db:
        raise HTTPException(status_code=404, detail="Template not found")
    
    versions = versions_db.get(template_id, [])
    target_version = None
    
    for v in versions:
        if v["version"] == version:
            target_version = v
            break
    
    if not target_version:
        raise HTTPException(status_code=404, detail="Version not found")
    
    template = templates_db[template_id]
    
    # Revert content
    template["content"] = target_version["content"]
    template["variables"] = extract_variables(target_version["content"])
    template["updated_at"] = datetime.utcnow().isoformat()
    
    # Create new version for the revert
    parts = template["version"].split(".")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    new_version = generate_version(major, minor, patch + 1)
    template["version"] = new_version
    
    # Add to version history
    versions_db[template_id].append({
        "version": new_version,
        "content": target_version["content"],
        "created_at": datetime.utcnow().isoformat(),
        "message": f"Reverted to version {version}"
    })
    
    return {
        "id": template_id,
        "version": new_version,
        "message": f"Template reverted to version {version}"
    }

@router.post("/templates/{template_id}/duplicate")
async def duplicate_template(template_id: str):
    """Create a copy of a template"""
    if template_id not in templates_db:
        raise HTTPException(status_code=404, detail="Template not found")
    
    original = templates_db[template_id]
    new_id = f"tpl_{uuid.uuid4().hex[:12]}"
    
    # Create copy
    new_template = original.copy()
    new_template["id"] = new_id
    new_template["name"] = f"{original['name']} (Copy)"
    new_template["version"] = "1.0.0"
    new_template["status"] = "draft"
    new_template["published"] = False
    new_template["created_at"] = datetime.utcnow().isoformat()
    new_template["updated_at"] = datetime.utcnow().isoformat()
    
    # Store template
    templates_db[new_id] = new_template
    
    # Initialize version history
    versions_db[new_id] = [{
        "version": "1.0.0",
        "content": new_template["content"],
        "created_at": datetime.utcnow().isoformat(),
        "message": f"Duplicated from {template_id}"
    }]
    
    return new_template

@router.post("/templates/test", response_model=Dict[str, Any])
async def test_template(request: TestRequest):
    """Test a POML template with Gemini"""
    if not gemini_model:
        raise HTTPException(status_code=503, detail="Gemini API not configured")
    
    try:
        # Render template
        rendered = render_template(request.content, request.variables)
        
        # Create full prompt
        full_prompt = f"{rendered}\n\nInput: {request.test_input}"
        
        # Execute with Gemini
        response = gemini_model.generate_content(full_prompt)
        
        return {
            "success": True,
            "rendered_template": rendered,
            "response": response.text,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/templates/import")
async def import_template(content: str, format: str = "json"):
    """Import a template from JSON or XML"""
    try:
        if format == "json":
            data = json.loads(content)
            template = POMLTemplate(**data)
        elif format == "xml":
            # Parse XML format
            import xml.etree.ElementTree as ET
            root = ET.fromstring(content)
            template = POMLTemplate(
                name=root.get("name", "Imported Template"),
                description=root.get("description", ""),
                content=content,
                category=root.get("category", "general")
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
        
        # Create the template
        return await create_template(template)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")

@router.get("/templates/{template_id}/export")
async def export_template(template_id: str, format: str = "json"):
    """Export a template as JSON or XML"""
    if template_id not in templates_db:
        raise HTTPException(status_code=404, detail="Template not found")
    
    template = templates_db[template_id]
    
    if format == "json":
        return {
            "name": template["name"],
            "description": template["description"],
            "content": template["content"],
            "category": template["category"],
            "tags": template["tags"]
        }
    elif format == "xml":
        # Return as XML string
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<template name="{template['name']}" category="{template['category']}" version="{template['version']}">
    <description>{template['description']}</description>
    <content><![CDATA[{template['content']}]]></content>
    <tags>{','.join(template['tags'])}</tags>
</template>"""
        return {"content": xml, "format": "xml"}
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")

# Initialize with sample templates
async def init_sample_templates():
    """Initialize with industry best-practice templates"""
    # Check if templates already exist (try database first, then fallback)
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM poml_templates")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        if count > 0:
            return
    except:
        if templates_db:
            return
        
    # Industry best-practice framework templates
    frameworks = [
        {
            "name": "Clinical Document Analysis",
            "description": "Analyze clinical documents and extract key medical information",
            "category": "clinical",
            "tags": ["clinical", "analysis", "medical"],
            "content": """<prompt version="1.0">
  <role>
    You are an expert clinical document analyst with 20+ years of experience in medical record review and healthcare informatics.
  </role>
  
  <context>
    Analyze {{document_type}} documents to extract structured medical information for healthcare providers.
    Priority: Patient safety and clinical accuracy above all else.
  </context>
  
  <task>
    Extract and structure the following information from the clinical document:
    1. Patient Demographics: Name, Age, Gender, Medical Record Number
    2. Chief Complaint: Primary reason for visit/consultation
    3. Clinical Findings: Vital signs, physical examination findings, symptoms
    4. Diagnoses: Primary and secondary diagnoses with ICD-10 codes if available
    5. Medications: Current medications, dosages, frequency, route
    6. Procedures: Any procedures performed or planned
    7. Follow-up: Recommended follow-up care, referrals, next appointments
    8. Risk Factors: Identified clinical risk factors or concerns
  </task>
  
  <guidelines>
    - Only extract information explicitly stated in the document
    - Use medical terminology accurately and consistently
    - Flag any unclear, incomplete, or potentially concerning information
    - Maintain patient confidentiality standards
    - If information is missing or unclear, explicitly state "Not documented" or "Unclear"
  </guidelines>
  
  <output_format>
    Provide a structured JSON response with the following schema:
    {
      "patient_demographics": {},
      "chief_complaint": "",
      "clinical_findings": {},
      "diagnoses": [],
      "medications": [],
      "procedures": [],
      "follow_up": {},
      "risk_factors": [],
      "data_quality_notes": []
    }
  </output_format>
</prompt>"""
        },
        {
            "name": "Medication Safety Analysis",
            "description": "Comprehensive drug interaction and safety analysis",
            "category": "pharmacy",
            "tags": ["pharmacy", "safety", "drug-interactions"],
            "content": """<prompt version="1.0">
  <role>
    You are a clinical pharmacist with PharmD certification and 15+ years of experience in medication therapy management and drug safety analysis.
  </role>
  
  <context>
    Analyze medication regimens for {{patient_age}} year old {{patient_gender}} patient to identify potential drug interactions, contraindications, and optimization opportunities.
    Patient weight: {{patient_weight}}kg
    Known allergies: {{known_allergies}}
    Medical conditions: {{medical_conditions}}
  </context>
  
  <task>
    Perform comprehensive medication analysis for: {{medication_list}}
    
    Analyze for:
    1. Drug-Drug Interactions (DDI)
    2. Drug-Disease Interactions
    3. Drug-Allergy Interactions  
    4. Dosage Appropriateness
    5. Therapeutic Duplications
    6. Contraindications
    7. Monitoring Requirements
    8. Optimization Opportunities
  </task>
  
  <guidelines>
    - Use evidence-based drug interaction databases (Lexicomp, Micromedex standards)
    - Classify interaction severity: Major, Moderate, Minor
    - Consider patient-specific factors (age, weight, renal/hepatic function)
    - Provide specific monitoring parameters when needed
    - Include mechanism of interaction for major interactions
    - Suggest alternatives when contraindications exist
  </guidelines>
  
  <output_format>
    {
      "overall_risk_assessment": "",
      "major_interactions": [],
      "moderate_interactions": [],
      "minor_interactions": [],
      "contraindications": [],
      "dosage_concerns": [],
      "monitoring_requirements": [],
      "recommendations": [],
      "pharmacist_notes": ""
    }
  </output_format>
</prompt>"""
        },
        {
            "name": "Laboratory Result Interpretation",
            "description": "Interpret lab results with clinical context and recommendations",
            "category": "laboratory",
            "tags": ["laboratory", "diagnostics", "interpretation"],
            "content": """<prompt version="1.0">
  <role>
    You are a clinical pathologist with board certification and 20+ years of experience in laboratory medicine and diagnostic interpretation.
  </role>
  
  <context>
    Interpret {{test_type}} results for {{patient_age}} year old {{patient_gender}} patient.
    Clinical history: {{clinical_history}}
    Current medications: {{current_medications}}
    Previous relevant lab values: {{previous_labs}}
  </context>
  
  <task>
    Provide comprehensive interpretation of laboratory results: {{lab_values}}
    
    Include:
    1. Normal/Abnormal Classification
    2. Clinical Significance Assessment
    3. Potential Causes for Abnormal Values
    4. Recommended Follow-up Actions
    5. Additional Tests if Indicated
    6. Monitoring Recommendations
    7. Clinical Correlation Notes
  </task>
  
  <guidelines>
    - Use age and gender-appropriate reference ranges
    - Consider medication effects on lab values
    - Assess clinical context and patient history
    - Flag critical values requiring immediate attention
    - Provide differential diagnosis for abnormal findings
    - Include timeframe for follow-up testing
    - Note any limitations or interferences
  </guidelines>
  
  <output_format>
    {
      "test_summary": "",
      "critical_values": [],
      "abnormal_findings": [
        {
          "parameter": "",
          "value": "",
          "reference_range": "",
          "clinical_significance": "",
          "potential_causes": [],
          "recommended_actions": []
        }
      ],
      "normal_findings": [],
      "follow_up_recommendations": [],
      "additional_testing": [],
      "clinical_correlation": "",
      "pathologist_notes": ""
    }
  </output_format>
</prompt>"""
        },
        {
            "name": "Health Information Assistant",
            "description": "Provide accurate, evidence-based health information to patients",
            "category": "patient-care",
            "tags": ["patient-education", "health-information", "communication"],
            "content": """<prompt version="1.0">
  <role>
    You are a certified health educator with extensive knowledge in patient communication and health literacy. You provide clear, accurate, and evidence-based health information.
  </role>
  
  <context>
    Patient query: {{patient_question}}
    Patient background: {{patient_background}}
    Health literacy level: {{health_literacy_level}}
  </context>
  
  <task>
    Provide comprehensive, patient-friendly health information that addresses the patient's question while maintaining appropriate medical boundaries.
    
    Include:
    1. Clear explanation in patient-friendly language
    2. Evidence-based information from reputable sources
    3. When to seek professional medical care
    4. Important disclaimers about limitations
    5. Additional reliable resources for more information
  </task>
  
  <guidelines>
    - Use plain language appropriate for the patient's health literacy level
    - Always include disclaimer that this doesn't replace professional medical advice
    - Encourage patients to consult healthcare providers for personalized care
    - Provide accurate, up-to-date medical information
    - Avoid diagnosing or recommending specific treatments
    - Be culturally sensitive and inclusive
    - Include warning signs that require immediate medical attention
  </guidelines>
  
  <disclaimers>
    - This information is for educational purposes only
    - Not a substitute for professional medical advice, diagnosis, or treatment
    - Always consult qualified healthcare providers for medical concerns
    - Seek immediate medical attention for emergency symptoms
  </disclaimers>
  
  <output_format>
    {
      "main_response": "",
      "key_points": [],
      "when_to_see_doctor": [],
      "warning_signs": [],
      "reliable_resources": [],
      "medical_disclaimer": ""
    }
  </output_format>
</prompt>"""
        },
        {
            "name": "General Purpose Assistant",
            "description": "Flexible template for general AI assistant tasks",
            "category": "general",
            "tags": ["general", "assistant", "flexible"],
            "content": """<prompt version="1.0">
  <role>
    You are {{assistant_role}} with expertise in {{domain_expertise}}.
  </role>
  
  <context>
    {{context_description}}
  </context>
  
  <task>
    {{task_description}}
    
    Key requirements:
    - {{requirement_1}}
    - {{requirement_2}}
    - {{requirement_3}}
  </task>
  
  <guidelines>
    - {{guideline_1}}
    - {{guideline_2}}
    - {{guideline_3}}
    - Provide accurate, helpful, and actionable information
    - Be clear and concise in your responses
    - Include relevant examples when helpful
  </guidelines>
  
  <output_format>
    {{output_format_description}}
  </output_format>
</prompt>"""
        }
    ]
    
    for framework in frameworks:
        template = POMLTemplate(**framework)
        await create_template(template)