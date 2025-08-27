from typing import Dict, Any
from datetime import datetime
import structlog

from .base import BaseAgent
from ..models import AgentRequest

logger = structlog.get_logger()

class ClinicalAgent(BaseAgent):
    """Agent for treatment plan generation, clinical note synthesis, and lab result interpretation"""
    
    async def execute(self, request: AgentRequest) -> Dict[str, Any]:
        """Execute clinical analysis"""
        start_time = datetime.utcnow()
        
        try:
            # Validate input
            if not await self.validate_input(request.input_data):
                raise ValueError("Invalid input data")
            
            # Determine clinical task type
            task_type = request.input_data.get("task_type", "general_analysis")
            
            result = {}
            
            if task_type == "treatment_plan":
                result = await self._generate_treatment_plan(request)
            elif task_type == "clinical_note":
                result = await self._synthesize_clinical_note(request)
            elif task_type == "lab_interpretation":
                result = await self._interpret_lab_results(request)
            else:
                result = await self._general_clinical_analysis(request)
            
            # Log execution
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_execution(request.request_id, "success", execution_time_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Clinical analysis failed", 
                        request_id=request.request_id,
                        error=str(e))
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_execution(request.request_id, "failed", execution_time_ms)
            raise
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        required_fields = ["patient_data"]
        for field in required_fields:
            if field not in input_data:
                logger.error(f"Missing required field: {field}")
                return False
        return True
    
    async def _generate_treatment_plan(self, request: AgentRequest) -> Dict[str, Any]:
        """Generate treatment plan"""
        patient_data = request.input_data.get("patient_data", {})
        diagnosis = request.input_data.get("diagnosis", "")
        
        prompt = f"""
        Generate a comprehensive treatment plan for the following patient:
        
        Patient Information:
        - Age: {patient_data.get('age', 'Unknown')}
        - Gender: {patient_data.get('gender', 'Unknown')}
        - Medical History: {patient_data.get('medical_history', 'None provided')}
        - Current Medications: {patient_data.get('medications', 'None')}
        - Allergies: {patient_data.get('allergies', 'None')}
        
        Diagnosis: {diagnosis}
        
        Please provide:
        1. Treatment objectives
        2. Medication recommendations (with dosages)
        3. Lifestyle modifications
        4. Follow-up schedule
        5. Warning signs to watch for
        6. Expected outcomes
        
        Format the response as a structured treatment plan.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "treatment_plan": response,
            "patient_id": patient_data.get("patient_id", ""),
            "diagnosis": diagnosis,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def _synthesize_clinical_note(self, request: AgentRequest) -> Dict[str, Any]:
        """Synthesize clinical note from various inputs"""
        encounter_data = request.input_data.get("encounter_data", {})
        
        prompt = f"""
        Synthesize a clinical note from the following encounter information:
        
        Chief Complaint: {encounter_data.get('chief_complaint', '')}
        History of Present Illness: {encounter_data.get('hpi', '')}
        Review of Systems: {encounter_data.get('ros', '')}
        Physical Examination: {encounter_data.get('physical_exam', '')}
        Assessment: {encounter_data.get('assessment', '')}
        Plan: {encounter_data.get('plan', '')}
        
        Generate a professional clinical note following SOAP format.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "clinical_note": response,
            "encounter_id": encounter_data.get("encounter_id", ""),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def _interpret_lab_results(self, request: AgentRequest) -> Dict[str, Any]:
        """Interpret laboratory results"""
        lab_data = request.input_data.get("lab_results", {})
        patient_data = request.input_data.get("patient_data", {})
        
        prompt = f"""
        Interpret the following laboratory results:
        
        Patient Context:
        - Age: {patient_data.get('age', 'Unknown')}
        - Gender: {patient_data.get('gender', 'Unknown')}
        - Relevant Medical History: {patient_data.get('medical_history', 'None')}
        
        Lab Results:
        {self._format_lab_results(lab_data)}
        
        Please provide:
        1. Interpretation of abnormal values
        2. Clinical significance
        3. Potential diagnoses to consider
        4. Recommended follow-up tests
        5. Urgent findings (if any)
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "interpretation": response,
            "lab_results": lab_data,
            "abnormal_flags": self._identify_abnormal_values(lab_data),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def _general_clinical_analysis(self, request: AgentRequest) -> Dict[str, Any]:
        """Perform general clinical analysis"""
        clinical_data = request.input_data
        
        prompt = f"""
        Perform a clinical analysis of the following data:
        
        {clinical_data}
        
        Provide clinical insights, recommendations, and any concerns.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "analysis": response,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _format_lab_results(self, lab_data: Dict) -> str:
        """Format lab results for prompt"""
        formatted = []
        for test, value in lab_data.items():
            if isinstance(value, dict):
                formatted.append(f"- {test}: {value.get('value', '')} {value.get('unit', '')} (Ref: {value.get('reference', '')})")
            else:
                formatted.append(f"- {test}: {value}")
        return "\n".join(formatted)
    
    def _identify_abnormal_values(self, lab_data: Dict) -> list:
        """Identify abnormal lab values"""
        abnormal = []
        for test, value in lab_data.items():
            if isinstance(value, dict) and value.get("flag"):
                abnormal.append({
                    "test": test,
                    "value": value.get("value"),
                    "flag": value.get("flag")
                })
        return abnormal