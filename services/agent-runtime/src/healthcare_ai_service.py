"""
Healthcare AI Service with Real Gemini API and Medical Document Analysis
Integrates Gemini AI with healthcare-specific capabilities
"""

import os
import json
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import google.generativeai as genai
from dataclasses import dataclass
import base64

logger = logging.getLogger(__name__)

# Medical terminology patterns
MEDICAL_PATTERNS = {
    "medications": r'\b(?:mg|mcg|ml|tablet|capsule|injection|dose|daily|twice|three times)\b',
    "diagnoses": r'\b(?:diagnosis|diagnosed|condition|syndrome|disease|disorder|infection)\b',
    "procedures": r'\b(?:surgery|procedure|operation|examination|test|scan|x-ray|MRI|CT|ultrasound)\b',
    "vitals": r'\b(?:blood pressure|BP|heart rate|HR|temperature|oxygen|SpO2|glucose|cholesterol)\b',
    "lab_values": r'\b\d+\.?\d*\s*(?:mg/dL|mmol/L|mcg/dL|g/dL|%|cells/mm3|IU/L)\b',
}

# ICD-10 and CPT code patterns
ICD10_PATTERN = r'\b[A-Z]\d{2}(?:\.\d{1,4})?\b'
CPT_PATTERN = r'\b\d{5}\b'

@dataclass
class MedicalEntity:
    """Represents a medical entity extracted from text"""
    entity_type: str
    value: str
    context: str
    confidence: float
    position: Tuple[int, int]

class HealthcareAIService:
    """Enhanced AI service with healthcare-specific capabilities"""
    
    def __init__(self):
        """Initialize Healthcare AI Service with Gemini"""
        # Initialize Gemini with API key
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        
        if self.api_key and self.api_key != "YOUR_GEMINI_API_KEY_HERE":
            try:
                genai.configure(api_key=self.api_key)
                self.ai_available = True
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini AI configured successfully with API key")
            except Exception as e:
                logger.error(f"Failed to configure Gemini: {e}")
                self.ai_available = False
                self.model = None
        else:
            self.ai_available = False
            self.model = None
            logger.warning("Gemini API key not configured - Please set GEMINI_API_KEY environment variable")
    
    def extract_medical_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities from text using pattern matching"""
        entities = []
        
        # Extract medications
        for match in re.finditer(r'(\w+)\s+(\d+\.?\d*)\s*(mg|mcg|ml)', text, re.IGNORECASE):
            entities.append(MedicalEntity(
                entity_type="medication",
                value=match.group(0),
                context=text[max(0, match.start()-50):min(len(text), match.end()+50)],
                confidence=0.85,
                position=(match.start(), match.end())
            ))
        
        # Extract ICD-10 codes
        for match in re.finditer(ICD10_PATTERN, text):
            entities.append(MedicalEntity(
                entity_type="icd10_code",
                value=match.group(0),
                context=text[max(0, match.start()-50):min(len(text), match.end()+50)],
                confidence=0.90,
                position=(match.start(), match.end())
            ))
        
        # Extract CPT codes
        for match in re.finditer(CPT_PATTERN, text):
            if match.group(0).isdigit() and len(match.group(0)) == 5:
                entities.append(MedicalEntity(
                    entity_type="cpt_code",
                    value=match.group(0),
                    context=text[max(0, match.start()-50):min(len(text), match.end()+50)],
                    confidence=0.85,
                    position=(match.start(), match.end())
                ))
        
        # Extract lab values
        for match in re.finditer(MEDICAL_PATTERNS["lab_values"], text):
            entities.append(MedicalEntity(
                entity_type="lab_value",
                value=match.group(0),
                context=text[max(0, match.start()-50):min(len(text), match.end()+50)],
                confidence=0.80,
                position=(match.start(), match.end())
            ))
        
        return entities
    
    async def analyze_clinical_document(self, document_text: str, document_type: str = "general") -> Dict[str, Any]:
        """Analyze clinical document using Gemini AI with healthcare-specific prompts"""
        
        # Extract medical entities first
        entities = self.extract_medical_entities(document_text)
        
        # Prepare healthcare-specific prompt based on document type
        prompts = {
            "discharge_summary": """You are a medical AI assistant analyzing a discharge summary. 
                Extract and structure the following information:
                1. Patient demographics
                2. Admission and discharge dates
                3. Primary and secondary diagnoses (with ICD-10 codes if mentioned)
                4. Procedures performed (with CPT codes if mentioned)
                5. Medications at discharge
                6. Follow-up instructions
                7. Discharge disposition
                
                Format the output as a structured JSON object.""",
            
            "lab_report": """You are a medical AI assistant analyzing laboratory results.
                Extract and interpret:
                1. Test names and values with units
                2. Normal reference ranges
                3. Abnormal flags (High/Low/Critical)
                4. Clinical significance of abnormal values
                5. Recommended follow-up actions
                
                Provide both extracted data and clinical interpretation.""",
            
            "radiology_report": """You are a medical AI assistant analyzing a radiology report.
                Extract:
                1. Imaging modality and body part
                2. Clinical indication
                3. Key findings (positive and negative)
                4. Impression/conclusion
                5. Recommendations for follow-up
                6. Comparison with prior studies if mentioned
                
                Highlight any critical or urgent findings.""",
            
            "prescription": """You are a medical AI assistant analyzing prescription information.
                Extract:
                1. Medication names with generic equivalents
                2. Dosages and frequencies
                3. Route of administration
                4. Duration of treatment
                5. Drug interactions to watch for
                6. Special instructions or warnings
                
                Flag any potential safety concerns.""",
            
            "clinical_note": """You are a medical AI assistant analyzing a clinical note.
                Extract:
                1. Chief complaint
                2. History of present illness
                3. Review of systems
                4. Physical examination findings
                5. Assessment and plan
                6. Diagnoses with ICD-10 codes
                7. Orders and prescriptions
                
                Summarize key clinical findings and action items.""",
            
            "general": """You are a medical AI assistant analyzing a healthcare document.
                Extract all relevant medical information including:
                1. Patient information
                2. Medical conditions and diagnoses
                3. Medications and treatments
                4. Test results and findings
                5. Clinical recommendations
                
                Ensure HIPAA compliance and maintain medical accuracy."""
        }
        
        prompt = prompts.get(document_type, prompts["general"])
        full_prompt = f"{prompt}\n\nDocument to analyze:\n{document_text}\n\nProvide structured analysis:"
        
        try:
            if self.ai_available and self.model:
                # Use real Gemini API
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,  # Lower temperature for medical accuracy
                        top_p=0.8,
                        top_k=40,
                        max_output_tokens=2048,
                    )
                )
                
                ai_analysis = response.text
                
                # Try to parse JSON from response
                json_data = None
                try:
                    json_match = re.search(r'\{.*\}', ai_analysis, re.DOTALL)
                    if json_match:
                        json_data = json.loads(json_match.group())
                except:
                    pass
                
                result = {
                    "status": "success",
                    "document_type": document_type,
                    "ai_analysis": ai_analysis,
                    "structured_data": json_data,
                    "extracted_entities": [
                        {
                            "type": e.entity_type,
                            "value": e.value,
                            "context": e.context,
                            "confidence": e.confidence
                        } for e in entities
                    ],
                    "medical_codes": {
                        "icd10": [e.value for e in entities if e.entity_type == "icd10_code"],
                        "cpt": [e.value for e in entities if e.entity_type == "cpt_code"]
                    },
                    "medications": [e.value for e in entities if e.entity_type == "medication"],
                    "lab_values": [e.value for e in entities if e.entity_type == "lab_value"],
                    "metadata": {
                        "model": "gemini-1.5-flash",
                        "api_based": True,
                        "timestamp": datetime.utcnow().isoformat(),
                        "document_length": len(document_text),
                        "entities_found": len(entities)
                    }
                }
                
            else:
                # Fallback to enhanced pattern-based extraction
                result = self._generate_pattern_based_analysis(document_text, document_type, entities)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in clinical document analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "fallback_analysis": self._generate_pattern_based_analysis(document_text, document_type, entities)
            }
    
    async def process_medical_image(self, image_data: str, image_type: str = "general") -> Dict[str, Any]:
        """Process medical images using Gemini's vision capabilities"""
        
        if not self.ai_available or not self.model:
            return {
                "status": "error",
                "message": "Gemini AI not available. Please configure API key.",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        try:
            # Decode base64 image if needed
            if image_data.startswith('data:'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            
            # Create prompt based on image type
            prompts = {
                "xray": "Analyze this X-ray image. Identify anatomical structures, any abnormalities, and provide clinical impressions.",
                "mri": "Analyze this MRI scan. Describe the imaging planes, tissue characteristics, and any pathological findings.",
                "ct": "Analyze this CT scan. Identify anatomical structures, density abnormalities, and provide differential diagnoses.",
                "ecg": "Analyze this ECG. Identify rhythm, rate, intervals, and any abnormalities. Provide interpretation.",
                "lab_report": "Extract all text from this lab report image. Identify test names, values, and abnormal results.",
                "prescription": "Extract medication information from this prescription image including drug names, dosages, and instructions.",
                "general": "Analyze this medical image. Extract any visible text and describe medical findings."
            }
            
            prompt = prompts.get(image_type, prompts["general"])
            
            # Generate content with image
            response = self.model.generate_content([prompt, image_bytes])
            
            return {
                "status": "success",
                "image_type": image_type,
                "analysis": response.text,
                "metadata": {
                    "model": "gemini-1.5-flash",
                    "vision_enabled": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing medical image: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to process medical image"
            }
    
    async def validate_medical_coding(self, codes: List[str], code_type: str = "icd10") -> Dict[str, Any]:
        """Validate medical codes using Gemini AI"""
        
        if not self.ai_available or not self.model:
            return {"status": "error", "message": "AI not available"}
        
        prompt = f"""You are a medical coding expert. Validate the following {code_type.upper()} codes:
        
        Codes: {', '.join(codes)}
        
        For each code, provide:
        1. Whether it's a valid {code_type.upper()} code
        2. The full description
        3. Category/chapter
        4. Any coding guidelines or notes
        5. Common related codes
        
        Format as JSON with structure:
        {{
            "code": {{
                "valid": boolean,
                "description": string,
                "category": string,
                "notes": string,
                "related_codes": []
            }}
        }}"""
        
        try:
            response = self.model.generate_content(prompt)
            
            # Parse response
            json_data = None
            try:
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    json_data = json.loads(json_match.group())
            except:
                pass
            
            return {
                "status": "success",
                "code_type": code_type,
                "validation_results": json_data or response.text,
                "codes_validated": codes,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating medical codes: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def generate_clinical_summary(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a clinical summary from patient data using Gemini"""
        
        if not self.ai_available or not self.model:
            return {"status": "error", "message": "AI not available"}
        
        prompt = f"""Generate a comprehensive clinical summary for the following patient data:
        
        Patient Information:
        {json.dumps(patient_data, indent=2)}
        
        Create a professional clinical summary including:
        1. Patient demographics
        2. Chief complaint and HPI
        3. Active problem list with ICD-10 codes
        4. Current medications with dosages
        5. Recent test results and significance
        6. Treatment plan and recommendations
        7. Follow-up care needed
        
        Format in clear sections suitable for healthcare providers."""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    top_p=0.8,
                    max_output_tokens=2048,
                )
            )
            
            return {
                "status": "success",
                "clinical_summary": response.text,
                "generated_at": datetime.utcnow().isoformat(),
                "model": "gemini-1.5-flash"
            }
            
        except Exception as e:
            logger.error(f"Error generating clinical summary: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def analyze_drug_interactions(self, medications: List[str]) -> Dict[str, Any]:
        """Analyze potential drug interactions using Gemini AI"""
        
        if not self.ai_available or not self.model:
            return {"status": "error", "message": "AI not available"}
        
        prompt = f"""You are a clinical pharmacist AI. Analyze potential drug interactions for:
        
        Medications: {', '.join(medications)}
        
        Provide:
        1. Severity of each interaction (Major/Moderate/Minor)
        2. Clinical effects and mechanism
        3. Management recommendations
        4. Monitoring parameters
        5. Alternative medications if needed
        
        Format as structured JSON."""
        
        try:
            response = self.model.generate_content(prompt)
            
            # Parse response
            json_data = None
            try:
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    json_data = json.loads(json_match.group())
            except:
                pass
            
            return {
                "status": "success",
                "medications_analyzed": medications,
                "interaction_analysis": json_data or response.text,
                "timestamp": datetime.utcnow().isoformat(),
                "model": "gemini-1.5-flash"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing drug interactions: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_pattern_based_analysis(self, text: str, document_type: str, entities: List[MedicalEntity]) -> Dict[str, Any]:
        """Generate analysis based on pattern matching when AI is not available"""
        
        return {
            "status": "pattern_based",
            "message": "AI not available - using pattern-based extraction",
            "document_type": document_type,
            "extracted_entities": [
                {
                    "type": e.entity_type,
                    "value": e.value,
                    "context": e.context,
                    "confidence": e.confidence
                } for e in entities
            ],
            "medical_codes": {
                "icd10": [e.value for e in entities if e.entity_type == "icd10_code"],
                "cpt": [e.value for e in entities if e.entity_type == "cpt_code"]
            },
            "medications": [e.value for e in entities if e.entity_type == "medication"],
            "lab_values": [e.value for e in entities if e.entity_type == "lab_value"],
            "metadata": {
                "extraction_method": "pattern_matching",
                "timestamp": datetime.utcnow().isoformat(),
                "document_length": len(text),
                "entities_found": len(entities)
            }
        }

# Singleton instance
_healthcare_service = None

def get_healthcare_service() -> HealthcareAIService:
    """Get or create Healthcare AI service singleton"""
    global _healthcare_service
    if _healthcare_service is None:
        _healthcare_service = HealthcareAIService()
    return _healthcare_service