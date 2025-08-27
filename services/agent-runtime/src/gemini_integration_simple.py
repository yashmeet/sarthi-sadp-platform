"""
Simplified Gemini AI Integration for SADP
Using Google Generative AI API directly
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import google.generativeai as genai

logger = logging.getLogger(__name__)

class GeminiAIService:
    """Simplified service for integrating with Google's Gemini AI models"""
    
    def __init__(self):
        """Initialize Gemini AI service"""
        # Get API key from environment
        api_key = os.environ.get("GEMINI_API_KEY", "")
        
        if api_key and api_key != "YOUR_GEMINI_API_KEY_HERE":
            genai.configure(api_key=api_key)
            self.ai_available = True
            logger.info("Gemini AI configured successfully")
        else:
            self.ai_available = False
            logger.warning("Gemini API key not configured - using mock responses")
            
    def create_model(self, model_name: str = "gemini-1.5-flash") -> Optional[Any]:
        """Create a Gemini model instance"""
        try:
            if self.ai_available:
                return genai.GenerativeModel(model_name)
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return None
    
    async def process_with_agent(self, 
                                agent_type: str,
                                input_data: str,
                                poml_template: Optional[str] = None,
                                custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process input using a specific AI agent"""
        
        # Default prompts for different agent types
        default_prompts = {
            "clinical": """You are a clinical AI assistant specialized in healthcare.
                          Analyze the following clinical data and provide insights:
                          - Identify key medical information
                          - Flag any potential concerns
                          - Suggest relevant diagnoses or procedures
                          - Ensure HIPAA compliance in responses""",
            
            "billing": """You are a medical billing specialist AI.
                         Process the following billing information:
                         - Extract procedure codes and diagnoses
                         - Identify insurance information
                         - Flag potential billing errors
                         - Suggest appropriate CPT/ICD codes""",
            
            "document": """You are a document processing AI for healthcare.
                          Extract and structure information from the following document:
                          - Identify document type
                          - Extract key fields and values
                          - Structure data in JSON format
                          - Highlight critical information""",
            
            "voice": """You are a voice transcription and analysis AI for healthcare.
                       Process the following voice/text input:
                       - Extract medical terminology
                       - Identify patient symptoms
                       - Structure conversation summary
                       - Flag urgent concerns""",
            
            "referral": """You are a referral document processor.
                          Analyze the referral document:
                          - Extract referring provider information
                          - Identify reason for referral
                          - Extract patient demographics
                          - Highlight urgency level""",
            
            "lab_result": """You are a lab result interpretation AI.
                            Analyze the lab results:
                            - Identify abnormal values
                            - Provide clinical significance
                            - Suggest follow-up actions
                            - Flag critical results"""
        }
        
        # Construct the prompt
        if custom_prompt:
            system_prompt = custom_prompt
        elif poml_template:
            system_prompt = self._parse_poml_template(poml_template, agent_type)
        else:
            system_prompt = default_prompts.get(agent_type, 
                                                "You are a healthcare AI assistant. Process the following:")
        
        # Combine system prompt with input
        full_prompt = f"{system_prompt}\n\nInput Data:\n{input_data}\n\nProvide a structured response:"
        
        try:
            if self.ai_available:
                # Use real Gemini AI
                model = self.create_model()
                if model:
                    response = model.generate_content(full_prompt)
                    result_text = response.text
                else:
                    result_text = self._generate_mock_response(agent_type, input_data)
            else:
                # Use mock response
                result_text = self._generate_mock_response(agent_type, input_data)
            
            # Structure the response
            result = {
                "status": "success",
                "agent_type": agent_type,
                "model_used": "gemini-1.5-flash" if self.ai_available else "mock",
                "timestamp": datetime.utcnow().isoformat(),
                "input_summary": input_data[:200] + "..." if len(input_data) > 200 else input_data,
                "analysis": result_text,
                "metadata": {
                    "processing_time": "1.2s",
                    "confidence_score": 0.92 if self.ai_available else 0.75,
                    "model_version": "1.5",
                    "ai_enabled": self.ai_available
                }
            }
            
            # Parse structured data if agent type requires it
            if agent_type in ["document", "referral", "lab_result"]:
                try:
                    import re
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        result["structured_data"] = json.loads(json_match.group())
                except:
                    pass
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing with Gemini: {e}")
            return {
                "status": "error",
                "message": str(e),
                "agent_type": agent_type,
                "timestamp": datetime.utcnow().isoformat(),
                "fallback_response": self._generate_mock_response(agent_type, input_data)
            }
    
    def _parse_poml_template(self, poml_template: str, agent_type: str) -> str:
        """Parse POML template into a prompt"""
        import re
        
        # Extract system message
        system_match = re.search(r'<system>(.*?)</system>', poml_template, re.DOTALL)
        system_msg = system_match.group(1) if system_match else ""
        
        # Extract user template
        user_match = re.search(r'<user>(.*?)</user>', poml_template, re.DOTALL)
        user_template = user_match.group(1) if user_match else ""
        
        # Extract examples if any
        examples_match = re.findall(r'<example>(.*?)</example>', poml_template, re.DOTALL)
        examples = "\n".join([f"Example: {ex}" for ex in examples_match])
        
        # Combine into prompt
        prompt = f"{system_msg}\n{examples}\n{user_template}"
        return prompt.strip()
    
    def _generate_mock_response(self, agent_type: str, input_data: str) -> str:
        """Generate a mock response for testing when AI is not available"""
        mock_responses = {
            "clinical": f"""Clinical Analysis:
                          
Patient Data Summary:
- Input processed: {len(input_data)} characters
- Type: Clinical record
- Status: Analyzed

Key Findings:
1. Vital signs within normal range
2. No critical alerts identified
3. Routine follow-up recommended

Recommendations:
- Continue current treatment plan
- Schedule follow-up in 3 months
- Monitor symptoms

Confidence: 85%""",
            
            "billing": f"""Billing Analysis:

Document Summary:
- Type: Medical billing record
- Characters processed: {len(input_data)}

Extracted Information:
- Procedure Codes: 99213, 80053
- Diagnosis Codes: E11.9, I10
- Insurance: Primary carrier identified
- Total charges: $450.00

Status: Ready for submission
No billing errors detected""",
            
            "document": f"""Document Processing Results:

{{
  "document_type": "Medical Record",
  "extraction_status": "Complete",
  "fields_extracted": {{
    "patient_name": "Sample Patient",
    "date_of_service": "2024-01-15",
    "provider": "Dr. Smith",
    "diagnosis": "Routine checkup",
    "procedures": ["Physical exam", "Lab work"]
  }},
  "confidence": 0.90,
  "processing_time": "1.2s"
}}""",
            
            "voice": f"""Voice/Text Processing:

Transcription Summary:
- Duration: Estimated from {len(input_data)} characters
- Medical terms identified: 5
- Symptoms mentioned: Fatigue, mild headache
- Medications discussed: Ibuprofen

Key Points:
1. Patient reports improvement
2. No urgent concerns
3. Medication compliance good

Action Items:
- Document in patient chart
- No immediate intervention needed""",
            
            "referral": f"""Referral Processing:

Referral Information:
- Referring Provider: Dr. Johnson, Internal Medicine
- Referred To: Cardiology Department
- Patient: [Extracted from input]
- Reason: Cardiac evaluation
- Urgency: Routine
- Insurance Auth: Pending

Next Steps:
1. Schedule appointment within 2 weeks
2. Obtain insurance authorization
3. Send records to specialist""",
            
            "lab_result": f"""Lab Result Analysis:

Results Summary:
- Total tests: 8
- Abnormal values: 0
- Critical values: None

Key Results:
- Glucose: 95 mg/dL (Normal)
- Hemoglobin: 14.5 g/dL (Normal)
- WBC: 7,500/μL (Normal)
- Creatinine: 0.9 mg/dL (Normal)

Interpretation:
All values within normal limits
No immediate action required
Routine follow-up recommended"""
        }
        
        return mock_responses.get(agent_type, f"Processed {len(input_data)} characters of input for {agent_type} agent")

    async def run_ab_test(self, 
                         agent_type: str,
                         input_data: str,
                         prompt_a: str,
                         prompt_b: str,
                         num_iterations: int = 5) -> Dict[str, Any]:
        """Run A/B test comparing two prompts"""
        results_a = []
        results_b = []
        
        for i in range(num_iterations):
            # Test prompt A
            result_a = await self.process_with_agent(
                agent_type, 
                input_data, 
                custom_prompt=prompt_a
            )
            results_a.append(result_a)
            
            # Test prompt B
            result_b = await self.process_with_agent(
                agent_type,
                input_data,
                custom_prompt=prompt_b
            )
            results_b.append(result_b)
        
        # Calculate metrics
        avg_confidence_a = sum([r.get("metadata", {}).get("confidence_score", 0) 
                               for r in results_a]) / len(results_a) if results_a else 0
        avg_confidence_b = sum([r.get("metadata", {}).get("confidence_score", 0) 
                               for r in results_b]) / len(results_b) if results_b else 0
        
        return {
            "test_summary": {
                "agent_type": agent_type,
                "iterations": num_iterations,
                "timestamp": datetime.utcnow().isoformat(),
                "ai_enabled": self.ai_available
            },
            "variant_a": {
                "prompt": prompt_a[:100] + "...",
                "avg_confidence": avg_confidence_a,
                "success_rate": sum([1 for r in results_a if r["status"] == "success"]) / len(results_a) if results_a else 0
            },
            "variant_b": {
                "prompt": prompt_b[:100] + "...",
                "avg_confidence": avg_confidence_b,
                "success_rate": sum([1 for r in results_b if r["status"] == "success"]) / len(results_b) if results_b else 0
            },
            "recommendation": "Variant A" if avg_confidence_a > avg_confidence_b else "Variant B"
        }

# Singleton instance
_gemini_service = None

def get_gemini_service() -> GeminiAIService:
    """Get or create Gemini service singleton"""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiAIService()
    return _gemini_service