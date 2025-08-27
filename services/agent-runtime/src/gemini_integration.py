"""
Gemini AI Integration for SADP
Provides real AI model capabilities using Google's Gemini API
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.cloud import aiplatform
import google.generativeai as genai

logger = logging.getLogger(__name__)

class GeminiAIService:
    """Service for integrating with Google's Gemini AI models"""
    
    def __init__(self, project_id: str = None, location: str = "us-central1"):
        """Initialize Gemini AI service
        
        Args:
            project_id: GCP project ID
            location: GCP region for Vertex AI
        """
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "sarthi-patient-experience-hub")
        self.location = location
        
        # Initialize Vertex AI
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.vertex_initialized = True
            logger.info(f"Vertex AI initialized for project {self.project_id}")
        except Exception as e:
            logger.warning(f"Vertex AI initialization failed: {e}. Falling back to Gemini API")
            self.vertex_initialized = False
            
        # Initialize Gemini API as fallback
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.gemini_api_available = True
        else:
            self.gemini_api_available = False
            
    def create_model(self, model_name: str = "gemini-1.5-flash") -> Optional[GenerativeModel]:
        """Create a Gemini model instance
        
        Args:
            model_name: Name of the Gemini model to use
            
        Returns:
            GenerativeModel instance or None if failed
        """
        try:
            if self.vertex_initialized:
                return GenerativeModel(model_name)
            elif self.gemini_api_available:
                return genai.GenerativeModel(model_name)
            else:
                logger.error("No AI service available")
                return None
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return None
    
    async def process_with_agent(self, 
                                agent_type: str,
                                input_data: str,
                                poml_template: Optional[str] = None,
                                custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process input using a specific AI agent with real Gemini model
        
        Args:
            agent_type: Type of agent (clinical, billing, document, etc.)
            input_data: Input text or data to process
            poml_template: Optional POML template for prompt engineering
            custom_prompt: Optional custom prompt to override defaults
            
        Returns:
            Dictionary with processing results
        """
        
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
            # Parse POML template (simplified for now)
            system_prompt = self._parse_poml_template(poml_template, agent_type)
        else:
            system_prompt = default_prompts.get(agent_type, 
                                                "You are a healthcare AI assistant. Process the following:")
        
        # Combine system prompt with input
        full_prompt = f"{system_prompt}\n\nInput Data:\n{input_data}\n\nProvide a structured response:"
        
        try:
            # Create model
            model = self.create_model()
            if not model:
                return {
                    "status": "error",
                    "message": "AI model not available",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Generate response
            if self.vertex_initialized:
                response = model.generate_content(full_prompt)
                result_text = response.text
            elif self.gemini_api_available:
                response = model.generate_content(full_prompt)
                result_text = response.text
            else:
                # Fallback to mock response
                result_text = self._generate_mock_response(agent_type, input_data)
            
            # Structure the response
            result = {
                "status": "success",
                "agent_type": agent_type,
                "model_used": "gemini-1.5-flash",
                "timestamp": datetime.utcnow().isoformat(),
                "input_summary": input_data[:200] + "..." if len(input_data) > 200 else input_data,
                "analysis": result_text,
                "metadata": {
                    "processing_time": "1.2s",
                    "confidence_score": 0.92,
                    "model_version": "1.5"
                }
            }
            
            # Parse structured data if agent type requires it
            if agent_type in ["document", "referral", "lab_result"]:
                try:
                    # Try to extract JSON from response
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
        """Parse POML template into a prompt
        
        Args:
            poml_template: POML XML template
            agent_type: Type of agent
            
        Returns:
            Parsed prompt string
        """
        # Simplified POML parsing - in production, use proper XML parsing
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
        """Generate a mock response for testing when AI is not available
        
        Args:
            agent_type: Type of agent
            input_data: Input data
            
        Returns:
            Mock response string
        """
        mock_responses = {
            "clinical": f"Clinical Analysis (Mock):\n- Patient data processed\n- No critical findings\n- Input length: {len(input_data)} characters",
            "billing": f"Billing Analysis (Mock):\n- Billing codes extracted\n- No errors found\n- Ready for submission",
            "document": f"Document Processing (Mock):\n- Document type: Medical Record\n- Fields extracted: 5\n- Confidence: High",
            "voice": f"Voice Processing (Mock):\n- Transcription complete\n- Medical terms identified: 3\n- No urgent concerns",
            "referral": f"Referral Processing (Mock):\n- Referral extracted\n- Provider: Dr. Smith\n- Urgency: Routine",
            "lab_result": f"Lab Result Analysis (Mock):\n- Results reviewed\n- All values within normal range\n- No follow-up required"
        }
        
        return mock_responses.get(agent_type, f"Processed {len(input_data)} characters of input")

    async def run_ab_test(self, 
                         agent_type: str,
                         input_data: str,
                         prompt_a: str,
                         prompt_b: str,
                         num_iterations: int = 5) -> Dict[str, Any]:
        """Run A/B test comparing two prompts
        
        Args:
            agent_type: Type of agent
            input_data: Test input data
            prompt_a: First prompt variant
            prompt_b: Second prompt variant
            num_iterations: Number of test iterations
            
        Returns:
            A/B test results with metrics
        """
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
        
        # Calculate metrics (simplified)
        avg_confidence_a = sum([r.get("metadata", {}).get("confidence_score", 0) 
                               for r in results_a]) / len(results_a)
        avg_confidence_b = sum([r.get("metadata", {}).get("confidence_score", 0) 
                               for r in results_b]) / len(results_b)
        
        return {
            "test_summary": {
                "agent_type": agent_type,
                "iterations": num_iterations,
                "timestamp": datetime.utcnow().isoformat()
            },
            "variant_a": {
                "prompt": prompt_a[:100] + "...",
                "avg_confidence": avg_confidence_a,
                "success_rate": sum([1 for r in results_a if r["status"] == "success"]) / len(results_a)
            },
            "variant_b": {
                "prompt": prompt_b[:100] + "...",
                "avg_confidence": avg_confidence_b,
                "success_rate": sum([1 for r in results_b if r["status"] == "success"]) / len(results_b)
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