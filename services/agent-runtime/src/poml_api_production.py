"""
POML Studio Production API
Multi-tenant POML template management with version control and collaboration
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, WebSocket
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import uuid
import asyncio
from google.cloud import firestore, storage
import google.generativeai as genai
import os
import re
import hashlib
import structlog

from auth import get_current_user, TokenData, require_permission, get_current_organization
from auth import Organization

logger = structlog.get_logger()

router = APIRouter(prefix="/poml", tags=["POML Studio"])

# Initialize clients
db = firestore.Client(project=os.environ.get("GCP_PROJECT_ID", "sarthi-patient-experience-hub"))
storage_client = storage.Client(project=os.environ.get("GCP_PROJECT_ID"))

# Initialize Gemini
if os.environ.get("GEMINI_API_KEY"):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

# Models
class POMLTemplate(BaseModel):
    name: str
    description: str
    content: str
    agent_type: Optional[str] = None
    variables: List[str] = []
    tags: List[str] = []
    category: str = "general"
    
class POMLVersion(BaseModel):
    version: str
    content: str
    variables: List[str]
    created_by: str
    created_at: datetime
    commit_message: str
    is_published: bool = False
    
class POMLTestRequest(BaseModel):
    template_id: Optional[str] = None
    template_content: Optional[str] = None
    variables: Dict[str, Any]
    test_data: str
    model: str = "gemini-1.5-flash"
    temperature: float = 0.7
    max_tokens: int = 2000
    
class POMLComment(BaseModel):
    content: str
    line_number: Optional[int] = None
    
class POMLApproval(BaseModel):
    status: str  # approved, rejected, needs_changes
    comment: Optional[str] = None

def parse_poml_template(content: str) -> Dict[str, Any]:
    """Parse and validate POML template"""
    parsed = {
        'sections': {},
        'variables': [],
        'metadata': {},
        'valid': True,
        'errors': []
    }
    
    try:
        # Validate XML structure
        import xml.etree.ElementTree as ET
        root = ET.fromstring(content)
        
        # Extract metadata
        parsed['metadata'] = {
            'version': root.get('version', '1.0'),
            'name': root.get('name', 'Untitled'),
            'type': root.get('type', 'general')
        }
        
        # Extract sections
        for child in root:
            section_name = child.tag.lower()
            parsed['sections'][section_name] = {
                'content': child.text or '',
                'attributes': child.attrib
            }
        
        # Extract variables
        variables = re.findall(r'\{\{(\w+)(?::([^}]+))?\}\}', content)
        parsed['variables'] = [{'name': v[0], 'type': v[1] or 'string'} for v in variables]
        
    except Exception as e:
        parsed['valid'] = False
        parsed['errors'].append(str(e))
    
    return parsed

def render_poml_template(content: str, variables: Dict[str, Any]) -> str:
    """Render POML template with variables and type checking"""
    rendered = content
    
    # Parse template for type checking
    parsed = parse_poml_template(content)
    
    # Validate and replace variables
    for var_info in parsed['variables']:
        var_name = var_info['name']
        var_type = var_info.get('type', 'string')
        
        if var_name in variables:
            value = variables[var_name]
            
            # Type validation
            if var_type == 'number' and not isinstance(value, (int, float)):
                raise ValueError(f"Variable {var_name} must be a number")
            elif var_type == 'boolean' and not isinstance(value, bool):
                raise ValueError(f"Variable {var_name} must be a boolean")
            elif var_type == 'array' and not isinstance(value, list):
                raise ValueError(f"Variable {var_name} must be an array")
            
            # Replace variable
            placeholder = f"{{{{{var_name}{':{}'.format(var_type) if var_type != 'string' else ''}}}}}"
            rendered = rendered.replace(placeholder, str(value))
    
    # Convert POML to prompt format
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(rendered)
        prompt_parts = []
        
        for section in root:
            if section.text and section.text.strip():
                prompt_parts.append(f"--- {section.tag.upper()} ---\n{section.text.strip()}")
        
        return "\n\n".join(prompt_parts)
    except:
        # Fallback to regex-based conversion
        rendered = re.sub(r'<prompt[^>]*>', '', rendered)
        rendered = re.sub(r'</prompt>', '', rendered)
        rendered = re.sub(r'<(\w+)[^>]*>', r'\n--- \1 ---\n', rendered)
        rendered = re.sub(r'</\w+>', '', rendered)
        return rendered.strip()

@router.post("/templates")
async def create_template(
    template: POMLTemplate,
    current_user: TokenData = Depends(require_permission("template:write"))
):
    """Create a new POML template with version control"""
    try:
        # Validate template
        parsed = parse_poml_template(template.content)
        if not parsed['valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid POML template: {', '.join(parsed['errors'])}"
            )
        
        # Generate template ID
        template_id = f"tpl_{uuid.uuid4().hex[:12]}"
        
        # Create template document
        template_data = {
            "id": template_id,
            "name": template.name,
            "description": template.description,
            "agent_type": template.agent_type,
            "category": template.category,
            "tags": template.tags,
            "organization_id": current_user.organization_id,
            "created_by": current_user.user_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "current_version": "1.0.0",
            "is_published": False,
            "collaborators": [current_user.user_id],
            "stats": {
                "executions": 0,
                "avg_latency": 0,
                "success_rate": 0
            }
        }
        
        # Store in Firestore
        db.collection("poml_templates").document(template_id).set(template_data)
        
        # Create first version
        version_data = {
            "version": "1.0.0",
            "content": template.content,
            "variables": [v['name'] for v in parsed['variables']],
            "created_by": current_user.user_id,
            "created_at": datetime.utcnow(),
            "commit_message": "Initial version",
            "is_published": False,
            "hash": hashlib.sha256(template.content.encode()).hexdigest()
        }
        
        db.collection("poml_templates").document(template_id)\
          .collection("versions").document("1.0.0").set(version_data)
        
        # Store in Cloud Storage for backup
        bucket = storage_client.bucket(f"{current_user.organization_id}-poml-templates")
        if not bucket.exists():
            bucket = storage_client.create_bucket(
                f"{current_user.organization_id}-poml-templates",
                location="us-central1"
            )
        
        blob = bucket.blob(f"{template_id}/v1.0.0.xml")
        blob.upload_from_string(template.content)
        
        logger.info(f"Created template {template_id}", 
                   user_id=current_user.user_id,
                   organization_id=current_user.organization_id)
        
        return {
            "id": template_id,
            "name": template.name,
            "version": "1.0.0",
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates")
async def list_templates(
    category: Optional[str] = None,
    tag: Optional[str] = None,
    search: Optional[str] = None,
    current_user: TokenData = Depends(get_current_user),
    org: Organization = Depends(get_current_organization)
):
    """List templates for organization with filtering"""
    try:
        query = db.collection("poml_templates")\
                  .where("organization_id", "==", org.id)
        
        if category:
            query = query.where("category", "==", category)
        
        if tag:
            query = query.where("tags", "array_contains", tag)
        
        templates = []
        for doc in query.stream():
            template_data = doc.to_dict()
            
            # Apply search filter
            if search and search.lower() not in template_data['name'].lower() \
               and search.lower() not in template_data.get('description', '').lower():
                continue
            
            templates.append({
                "id": template_data['id'],
                "name": template_data['name'],
                "description": template_data.get('description'),
                "category": template_data.get('category'),
                "tags": template_data.get('tags', []),
                "version": template_data.get('current_version'),
                "created_by": template_data.get('created_by'),
                "created_at": template_data.get('created_at'),
                "stats": template_data.get('stats', {})
            })
        
        return {"templates": templates, "total": len(templates)}
        
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates/{template_id}")
async def get_template(
    template_id: str,
    version: Optional[str] = None,
    current_user: TokenData = Depends(get_current_user)
):
    """Get template details with specific version"""
    try:
        # Get template
        template_doc = db.collection("poml_templates").document(template_id).get()
        
        if not template_doc.exists:
            raise HTTPException(status_code=404, detail="Template not found")
        
        template_data = template_doc.to_dict()
        
        # Check permissions
        if template_data['organization_id'] != current_user.organization_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get requested version
        version = version or template_data['current_version']
        version_doc = db.collection("poml_templates").document(template_id)\
                       .collection("versions").document(version).get()
        
        if not version_doc.exists:
            raise HTTPException(status_code=404, detail=f"Version {version} not found")
        
        version_data = version_doc.to_dict()
        
        # Get version history
        versions = []
        for v_doc in db.collection("poml_templates").document(template_id)\
                      .collection("versions").order_by("created_at", direction=firestore.Query.DESCENDING).stream():
            v_data = v_doc.to_dict()
            versions.append({
                "version": v_data['version'],
                "created_by": v_data['created_by'],
                "created_at": v_data['created_at'],
                "commit_message": v_data.get('commit_message'),
                "is_published": v_data.get('is_published', False)
            })
        
        return {
            **template_data,
            "content": version_data['content'],
            "variables": version_data['variables'],
            "current_version_details": version_data,
            "versions": versions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/templates/{template_id}")
async def update_template(
    template_id: str,
    content: str,
    commit_message: str,
    current_user: TokenData = Depends(require_permission("template:write"))
):
    """Create new version of template"""
    try:
        # Get template
        template_doc = db.collection("poml_templates").document(template_id).get()
        
        if not template_doc.exists:
            raise HTTPException(status_code=404, detail="Template not found")
        
        template_data = template_doc.to_dict()
        
        # Check permissions
        if template_data['organization_id'] != current_user.organization_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Validate new content
        parsed = parse_poml_template(content)
        if not parsed['valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid POML template: {', '.join(parsed['errors'])}"
            )
        
        # Calculate new version
        current_version = template_data['current_version']
        version_parts = current_version.split('.')
        version_parts[2] = str(int(version_parts[2]) + 1)
        new_version = '.'.join(version_parts)
        
        # Check if content actually changed
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        last_version = db.collection("poml_templates").document(template_id)\
                        .collection("versions").document(current_version).get()
        
        if last_version.exists and last_version.to_dict().get('hash') == content_hash:
            raise HTTPException(status_code=400, detail="No changes detected")
        
        # Create new version
        version_data = {
            "version": new_version,
            "content": content,
            "variables": [v['name'] for v in parsed['variables']],
            "created_by": current_user.user_id,
            "created_at": datetime.utcnow(),
            "commit_message": commit_message,
            "is_published": False,
            "hash": content_hash,
            "parent_version": current_version
        }
        
        # Store version
        db.collection("poml_templates").document(template_id)\
          .collection("versions").document(new_version).set(version_data)
        
        # Update template
        db.collection("poml_templates").document(template_id).update({
            "current_version": new_version,
            "updated_at": datetime.utcnow()
        })
        
        # Store in Cloud Storage
        bucket = storage_client.bucket(f"{current_user.organization_id}-poml-templates")
        blob = bucket.blob(f"{template_id}/v{new_version}.xml")
        blob.upload_from_string(content)
        
        logger.info(f"Updated template {template_id} to version {new_version}",
                   user_id=current_user.user_id)
        
        return {
            "id": template_id,
            "version": new_version,
            "status": "updated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/templates/{template_id}/test")
async def test_template(
    template_id: str,
    request: POMLTestRequest,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(get_current_user)
):
    """Test template execution with real Gemini API"""
    try:
        if not gemini_model:
            raise HTTPException(
                status_code=503,
                detail="Gemini API not configured"
            )
        
        # Get template if ID provided
        if template_id != "inline":
            template_doc = db.collection("poml_templates").document(template_id).get()
            
            if not template_doc.exists:
                raise HTTPException(status_code=404, detail="Template not found")
            
            template_data = template_doc.to_dict()
            
            # Check permissions
            if template_data['organization_id'] != current_user.organization_id:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Get current version content
            version_doc = db.collection("poml_templates").document(template_id)\
                           .collection("versions").document(template_data['current_version']).get()
            
            content = version_doc.to_dict()['content']
        else:
            content = request.template_content
        
        # Render template
        try:
            rendered_prompt = render_poml_template(content, request.variables)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Add test data to prompt
        full_prompt = f"{rendered_prompt}\n\n--- INPUT ---\n{request.test_data}"
        
        # Execute with Gemini
        start_time = datetime.utcnow()
        
        try:
            # Configure generation
            generation_config = genai.GenerationConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
                top_p=0.95,
                top_k=40
            )
            
            # Generate response
            response = gemini_model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = {
                "success": True,
                "response": response.text,
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                },
                "execution_time": execution_time,
                "model": request.model
            }
            
            # Update template stats in background
            if template_id != "inline":
                background_tasks.add_task(
                    update_template_stats,
                    template_id,
                    execution_time,
                    True
                )
            
            # Log execution
            logger.info(f"Template test successful",
                       template_id=template_id,
                       execution_time=execution_time,
                       tokens=result['usage']['total_tokens'])
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            
            # Update failure stats
            if template_id != "inline":
                background_tasks.add_task(
                    update_template_stats,
                    template_id,
                    0,
                    False
                )
            
            raise HTTPException(
                status_code=500,
                detail=f"AI generation failed: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/templates/{template_id}/publish")
async def publish_template(
    template_id: str,
    version: Optional[str] = None,
    current_user: TokenData = Depends(require_permission("template:write"))
):
    """Publish template version for production use"""
    try:
        # Get template
        template_doc = db.collection("poml_templates").document(template_id).get()
        
        if not template_doc.exists:
            raise HTTPException(status_code=404, detail="Template not found")
        
        template_data = template_doc.to_dict()
        
        # Check permissions
        if template_data['organization_id'] != current_user.organization_id:
            if current_user.role not in ["org_admin", "super_admin"]:
                raise HTTPException(status_code=403, detail="Only admins can publish templates")
        
        # Get version to publish
        version = version or template_data['current_version']
        
        # Update version
        db.collection("poml_templates").document(template_id)\
          .collection("versions").document(version).update({
              "is_published": True,
              "published_by": current_user.user_id,
              "published_at": datetime.utcnow()
          })
        
        # Update template
        db.collection("poml_templates").document(template_id).update({
            "is_published": True,
            "published_version": version,
            "published_at": datetime.utcnow()
        })
        
        logger.info(f"Published template {template_id} version {version}",
                   user_id=current_user.user_id)
        
        return {
            "id": template_id,
            "version": version,
            "status": "published"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to publish template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/templates/{template_id}/fork")
async def fork_template(
    template_id: str,
    new_name: str,
    current_user: TokenData = Depends(get_current_user)
):
    """Fork a template to create a new copy"""
    try:
        # Get original template
        template_doc = db.collection("poml_templates").document(template_id).get()
        
        if not template_doc.exists:
            raise HTTPException(status_code=404, detail="Template not found")
        
        original = template_doc.to_dict()
        
        # Get current version content
        version_doc = db.collection("poml_templates").document(template_id)\
                       .collection("versions").document(original['current_version']).get()
        
        version_data = version_doc.to_dict()
        
        # Create new template
        new_template = POMLTemplate(
            name=new_name,
            description=f"Forked from {original['name']}",
            content=version_data['content'],
            agent_type=original.get('agent_type'),
            variables=version_data['variables'],
            tags=original.get('tags', []),
            category=original.get('category', 'general')
        )
        
        # Create the forked template
        result = await create_template(new_template, current_user)
        
        # Add fork metadata
        db.collection("poml_templates").document(result['id']).update({
            "forked_from": template_id,
            "forked_at": datetime.utcnow()
        })
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fork template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def update_template_stats(template_id: str, execution_time: float, success: bool):
    """Update template execution statistics"""
    try:
        template_ref = db.collection("poml_templates").document(template_id)
        template_doc = template_ref.get()
        
        if template_doc.exists:
            stats = template_doc.to_dict().get('stats', {})
            
            # Update stats
            total_executions = stats.get('executions', 0) + 1
            total_latency = stats.get('avg_latency', 0) * stats.get('executions', 0) + execution_time
            avg_latency = total_latency / total_executions if total_executions > 0 else 0
            
            success_count = stats.get('success_rate', 0) * stats.get('executions', 0) / 100
            if success:
                success_count += 1
            success_rate = (success_count / total_executions * 100) if total_executions > 0 else 0
            
            template_ref.update({
                "stats.executions": total_executions,
                "stats.avg_latency": avg_latency,
                "stats.success_rate": success_rate
            })
            
    except Exception as e:
        logger.error(f"Failed to update template stats: {e}")