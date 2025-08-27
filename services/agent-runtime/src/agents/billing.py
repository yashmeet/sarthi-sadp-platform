from typing import Dict, Any
from datetime import datetime
import structlog

from .base import BaseAgent
from ..models import AgentRequest

logger = structlog.get_logger()

class BillingAgent(BaseAgent):
    """Agent for medical billing code generation, insurance claims processing, and billing optimization"""
    
    async def execute(self, request: AgentRequest) -> Dict[str, Any]:
        """Execute billing analysis"""
        start_time = datetime.utcnow()
        
        try:
            # Validate input
            if not await self.validate_input(request.input_data):
                raise ValueError("Invalid input data")
            
            # Determine billing task type
            task_type = request.input_data.get("task_type", "general_billing")
            
            result = {}
            
            if task_type == "code_generation":
                result = await self._generate_billing_codes(request)
            elif task_type == "claims_processing":
                result = await self._process_insurance_claim(request)
            elif task_type == "billing_optimization":
                result = await self._optimize_billing(request)
            elif task_type == "denial_analysis":
                result = await self._analyze_claim_denial(request)
            else:
                result = await self._general_billing_analysis(request)
            
            # Log execution
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_execution(request.request_id, "success", execution_time_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Billing analysis failed", 
                        request_id=request.request_id,
                        error=str(e))
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_execution(request.request_id, "failed", execution_time_ms)
            raise
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        required_fields = ["encounter_data"]
        for field in required_fields:
            if field not in input_data:
                logger.error(f"Missing required field: {field}")
                return False
        return True
    
    async def _generate_billing_codes(self, request: AgentRequest) -> Dict[str, Any]:
        """Generate ICD-10 and CPT billing codes"""
        encounter_data = request.input_data.get("encounter_data", {})
        patient_data = request.input_data.get("patient_data", {})
        
        prompt = f"""
        Generate appropriate medical billing codes (ICD-10 and CPT) for the following encounter:
        
        Patient Information:
        - Age: {patient_data.get('age', 'Unknown')}
        - Gender: {patient_data.get('gender', 'Unknown')}
        - Insurance: {patient_data.get('insurance_type', 'Unknown')}
        
        Encounter Details:
        - Date of Service: {encounter_data.get('date_of_service', '')}
        - Provider: {encounter_data.get('provider', '')}
        - Chief Complaint: {encounter_data.get('chief_complaint', '')}
        - Diagnosis: {encounter_data.get('diagnosis', '')}
        - Procedures Performed: {encounter_data.get('procedures', '')}
        - Treatment Plan: {encounter_data.get('treatment_plan', '')}
        
        Please provide:
        1. Primary ICD-10 diagnosis codes with descriptions
        2. Secondary ICD-10 codes (if applicable)
        3. CPT procedure codes with descriptions
        4. Modifier codes (if needed)
        5. Units of service
        6. Supporting documentation requirements
        7. Potential bundling considerations
        
        Format as structured billing code recommendations.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "billing_codes": response,
            "encounter_id": encounter_data.get("encounter_id", ""),
            "patient_id": patient_data.get("patient_id", ""),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def _process_insurance_claim(self, request: AgentRequest) -> Dict[str, Any]:
        """Process and validate insurance claim"""
        claim_data = request.input_data.get("claim_data", {})
        patient_data = request.input_data.get("patient_data", {})
        
        prompt = f"""
        Review and validate the following insurance claim for processing:
        
        Patient Information:
        - Name: {patient_data.get('name', '')}
        - DOB: {patient_data.get('date_of_birth', '')}
        - Insurance ID: {patient_data.get('insurance_id', '')}
        - Policy Number: {patient_data.get('policy_number', '')}
        
        Claim Details:
        - Claim Number: {claim_data.get('claim_number', '')}
        - Service Date: {claim_data.get('service_date', '')}
        - Provider: {claim_data.get('provider', '')}
        - Diagnosis Codes: {claim_data.get('diagnosis_codes', '')}
        - Procedure Codes: {claim_data.get('procedure_codes', '')}
        - Billed Amount: {claim_data.get('billed_amount', '')}
        
        Please analyze and provide:
        1. Claim validation status
        2. Required prior authorizations
        3. Coverage verification
        4. Potential rejection reasons
        5. Missing information
        6. Recommended corrections
        7. Submission readiness assessment
        
        Format as a claim processing report.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "claim_analysis": response,
            "claim_number": claim_data.get("claim_number", ""),
            "validation_status": "pending_review",
            "processed_at": datetime.utcnow().isoformat()
        }
    
    async def _optimize_billing(self, request: AgentRequest) -> Dict[str, Any]:
        """Optimize billing for maximum reimbursement"""
        billing_data = request.input_data.get("billing_data", {})
        
        prompt = f"""
        Analyze the following billing scenario for optimization opportunities:
        
        Current Billing:
        - Diagnosis Codes: {billing_data.get('current_diagnosis_codes', '')}
        - Procedure Codes: {billing_data.get('current_procedure_codes', '')}
        - Current Reimbursement: {billing_data.get('current_reimbursement', '')}
        - Payer Mix: {billing_data.get('payer_mix', '')}
        
        Encounter Details:
        - Services Provided: {billing_data.get('services_provided', '')}
        - Documentation Available: {billing_data.get('documentation', '')}
        - Provider Qualifications: {billing_data.get('provider_qualifications', '')}
        
        Please provide optimization recommendations:
        1. Alternative coding strategies
        2. Additional billable services
        3. Documentation improvements
        4. Bundling opportunities
        5. Modifier usage optimization
        6. Potential revenue increase
        7. Compliance considerations
        
        Format as billing optimization recommendations.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "optimization_recommendations": response,
            "current_billing": billing_data,
            "potential_improvement": "To be calculated",
            "analyzed_at": datetime.utcnow().isoformat()
        }
    
    async def _analyze_claim_denial(self, request: AgentRequest) -> Dict[str, Any]:
        """Analyze claim denial and provide resolution strategies"""
        denial_data = request.input_data.get("denial_data", {})
        
        prompt = f"""
        Analyze the following claim denial and provide resolution strategies:
        
        Denial Information:
        - Claim Number: {denial_data.get('claim_number', '')}
        - Denial Code: {denial_data.get('denial_code', '')}
        - Denial Reason: {denial_data.get('denial_reason', '')}
        - Original Submission Date: {denial_data.get('original_submission_date', '')}
        - Denied Amount: {denial_data.get('denied_amount', '')}
        
        Original Claim Details:
        - Diagnosis Codes: {denial_data.get('diagnosis_codes', '')}
        - Procedure Codes: {denial_data.get('procedure_codes', '')}
        - Provider: {denial_data.get('provider', '')}
        - Patient: {denial_data.get('patient_info', '')}
        
        Please provide:
        1. Denial reason analysis
        2. Required corrections
        3. Additional documentation needed
        4. Appeal process steps
        5. Timeline for resolution
        6. Success probability assessment
        7. Alternative billing strategies
        
        Format as a denial resolution action plan.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "denial_analysis": response,
            "claim_number": denial_data.get("claim_number", ""),
            "denial_code": denial_data.get("denial_code", ""),
            "resolution_priority": "high",
            "analyzed_at": datetime.utcnow().isoformat()
        }
    
    async def _general_billing_analysis(self, request: AgentRequest) -> Dict[str, Any]:
        """Perform general billing analysis"""
        billing_data = request.input_data
        
        prompt = f"""
        Perform a general billing analysis of the following data:
        
        {billing_data}
        
        Provide billing insights, optimization opportunities, and compliance recommendations.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "analysis": response,
            "generated_at": datetime.utcnow().isoformat()
        }