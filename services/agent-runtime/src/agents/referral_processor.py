from typing import Dict, Any
from datetime import datetime
import structlog

from .base import BaseAgent
from ..models import AgentRequest

logger = structlog.get_logger()

class ReferralProcessorAgent(BaseAgent):
    """Agent for processing referral documents, extracting key information, and routing to appropriate specialists"""
    
    async def execute(self, request: AgentRequest) -> Dict[str, Any]:
        """Execute referral document processing"""
        start_time = datetime.utcnow()
        
        try:
            # Validate input
            if not await self.validate_input(request.input_data):
                raise ValueError("Invalid input data")
            
            # Determine referral processing task type
            task_type = request.input_data.get("task_type", "document_processing")
            
            result = {}
            
            if task_type == "document_extraction":
                result = await self._extract_referral_information(request)
            elif task_type == "specialist_routing":
                result = await self._route_to_specialist(request)
            elif task_type == "urgency_assessment":
                result = await self._assess_referral_urgency(request)
            elif task_type == "authorization_check":
                result = await self._check_prior_authorization(request)
            elif task_type == "referral_validation":
                result = await self._validate_referral(request)
            else:
                result = await self._general_referral_processing(request)
            
            # Log execution
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_execution(request.request_id, "success", execution_time_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Referral processing failed", 
                        request_id=request.request_id,
                        error=str(e))
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_execution(request.request_id, "failed", execution_time_ms)
            raise
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        # Check for either referral document or referral data
        has_document = "referral_document" in input_data or "document_text" in input_data
        has_data = "referral_data" in input_data
        
        if not (has_document or has_data):
            logger.error("Missing required field: referral_document, document_text, or referral_data")
            return False
        return True
    
    async def _extract_referral_information(self, request: AgentRequest) -> Dict[str, Any]:
        """Extract key information from referral documents"""
        document_text = request.input_data.get("document_text", "")
        document_type = request.input_data.get("document_type", "referral_letter")
        
        prompt = f"""
        Extract and structure key information from the following referral document:
        
        Document Type: {document_type}
        
        Document Content:
        {document_text}
        
        Please extract and structure the following information:
        
        Patient Information:
        1. Patient name
        2. Date of birth
        3. MRN/Patient ID
        4. Contact information
        5. Insurance information
        
        Referring Provider:
        1. Provider name
        2. Specialty
        3. Contact information
        4. NPI number
        5. Practice/institution
        
        Referral Details:
        1. Reason for referral
        2. Specialty requested
        3. Urgency level
        4. Preferred provider/institution
        5. Clinical summary
        6. Relevant history
        7. Current medications
        8. Test results/attachments
        9. Specific questions for specialist
        
        Administrative:
        1. Date of referral
        2. Insurance authorization requirements
        3. Appointment preferences
        4. Follow-up instructions
        
        Format as structured JSON-like data extraction.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "extracted_information": response,
            "document_type": document_type,
            "extraction_confidence": 0.92,  # Simulated confidence score
            "extracted_at": datetime.utcnow().isoformat(),
            "requires_review": False
        }
    
    async def _route_to_specialist(self, request: AgentRequest) -> Dict[str, Any]:
        """Route referral to appropriate specialist based on condition and requirements"""
        referral_data = request.input_data.get("referral_data", {})
        patient_info = request.input_data.get("patient_info", {})
        
        prompt = f"""
        Determine appropriate specialist routing for the following referral:
        
        Patient Information:
        - Age: {patient_info.get('age', 'Unknown')}
        - Gender: {patient_info.get('gender', 'Unknown')}
        - Insurance: {patient_info.get('insurance_type', 'Unknown')}
        - Location: {patient_info.get('location', 'Unknown')}
        
        Referral Details:
        - Chief Complaint: {referral_data.get('chief_complaint', '')}
        - Diagnosis/Reason: {referral_data.get('diagnosis', '')}
        - Symptoms: {referral_data.get('symptoms', '')}
        - Duration: {referral_data.get('symptom_duration', '')}
        - Previous Treatments: {referral_data.get('previous_treatments', '')}
        - Urgency: {referral_data.get('urgency', 'routine')}
        - Requested Specialty: {referral_data.get('requested_specialty', '')}
        
        Please provide specialist routing recommendations:
        1. Primary specialty recommendation
        2. Alternative specialist options
        3. Sub-specialty considerations
        4. Provider characteristics needed
        5. Geographic considerations
        6. Insurance network requirements
        7. Appointment timing recommendations
        8. Multi-disciplinary care needs
        9. Telemedicine suitability
        
        Format as specialist routing plan.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "routing_recommendation": response,
            "primary_specialty": referral_data.get("requested_specialty", ""),
            "patient_id": patient_info.get("patient_id", ""),
            "urgency_level": referral_data.get("urgency", "routine"),
            "routed_at": datetime.utcnow().isoformat()
        }
    
    async def _assess_referral_urgency(self, request: AgentRequest) -> Dict[str, Any]:
        """Assess urgency level of referral based on clinical indicators"""
        clinical_data = request.input_data.get("clinical_data", {})
        symptoms = request.input_data.get("symptoms", [])
        
        prompt = f"""
        Assess the urgency level of this referral based on clinical indicators:
        
        Clinical Information:
        - Primary Diagnosis: {clinical_data.get('primary_diagnosis', '')}
        - Symptoms: {', '.join(symptoms) if isinstance(symptoms, list) else symptoms}
        - Symptom Duration: {clinical_data.get('symptom_duration', '')}
        - Severity (1-10): {clinical_data.get('severity', '')}
        - Red Flag Symptoms: {clinical_data.get('red_flags', '')}
        - Current Status: {clinical_data.get('current_status', '')}
        - Vital Signs: {clinical_data.get('vital_signs', '')}
        - Lab Results: {clinical_data.get('lab_results', '')}
        
        Patient Risk Factors:
        - Age: {clinical_data.get('age', '')}
        - Comorbidities: {clinical_data.get('comorbidities', '')}
        - Current Medications: {clinical_data.get('medications', '')}
        - Previous Hospitalizations: {clinical_data.get('previous_hospitalizations', '')}
        
        Please assess urgency and provide:
        1. Urgency classification (stat/urgent/expedite/routine)
        2. Timeline for specialist consultation
        3. Interim care recommendations
        4. Warning signs for escalation
        5. Risk stratification
        6. Monitoring requirements
        7. Alternative care pathways if delayed
        8. Documentation requirements
        
        Format as urgency assessment report.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "urgency_assessment": response,
            "urgency_level": self._determine_urgency_level(symptoms, clinical_data),
            "recommended_timeline": self._calculate_timeline(clinical_data.get('severity', '5')),
            "assessed_at": datetime.utcnow().isoformat()
        }
    
    async def _check_prior_authorization(self, request: AgentRequest) -> Dict[str, Any]:
        """Check prior authorization requirements for referral"""
        referral_info = request.input_data.get("referral_info", {})
        insurance_info = request.input_data.get("insurance_info", {})
        
        prompt = f"""
        Analyze prior authorization requirements for this referral:
        
        Insurance Information:
        - Plan Type: {insurance_info.get('plan_type', '')}
        - Provider: {insurance_info.get('insurance_provider', '')}
        - Policy Number: {insurance_info.get('policy_number', '')}
        - Group Number: {insurance_info.get('group_number', '')}
        
        Referral Details:
        - Specialty: {referral_info.get('specialty', '')}
        - Procedure/Service: {referral_info.get('requested_service', '')}
        - Diagnosis Code: {referral_info.get('diagnosis_code', '')}
        - CPT Codes: {referral_info.get('cpt_codes', '')}
        
        Provider Information:
        - Referring Provider: {referral_info.get('referring_provider', '')}
        - Requested Provider: {referral_info.get('requested_provider', '')}
        - Facility: {referral_info.get('facility', '')}
        
        Please analyze and provide:
        1. Prior authorization requirement (yes/no)
        2. Required documentation
        3. Submission process steps
        4. Expected processing time
        5. Appeal process if denied
        6. Alternative coverage options
        7. Patient responsibility/costs
        8. Network requirements
        
        Format as prior authorization analysis.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "authorization_analysis": response,
            "requires_authorization": True,  # Would be determined by analysis
            "insurance_plan": insurance_info.get("plan_type", ""),
            "specialty": referral_info.get("specialty", ""),
            "checked_at": datetime.utcnow().isoformat()
        }
    
    async def _validate_referral(self, request: AgentRequest) -> Dict[str, Any]:
        """Validate referral completeness and accuracy"""
        referral_data = request.input_data.get("referral_data", {})
        
        prompt = f"""
        Validate the completeness and accuracy of this referral:
        
        Referral Information:
        - Patient Demographics: {referral_data.get('patient_demographics', '')}
        - Insurance Verification: {referral_data.get('insurance_verified', '')}
        - Clinical Information: {referral_data.get('clinical_information', '')}
        - Reason for Referral: {referral_data.get('reason', '')}
        - Requested Specialty: {referral_data.get('specialty', '')}
        - Supporting Documentation: {referral_data.get('documentation', '')}
        - Provider Information: {referral_data.get('provider_info', '')}
        
        Please validate and check for:
        1. Required fields completion
        2. Clinical information adequacy
        3. Insurance authorization status
        4. Provider credentials verification
        5. Appropriate specialty selection
        6. Supporting documentation sufficiency
        7. Patient consent requirements
        8. Regulatory compliance
        9. Missing information identification
        
        Format as referral validation report.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "validation_report": response,
            "validation_status": "pending_completion",  # Would be determined by analysis
            "missing_items": [],  # Would be populated by analysis
            "validated_at": datetime.utcnow().isoformat()
        }
    
    async def _general_referral_processing(self, request: AgentRequest) -> Dict[str, Any]:
        """Perform general referral processing"""
        referral_data = request.input_data
        
        prompt = f"""
        Process the following referral data:
        
        {referral_data}
        
        Provide referral processing analysis, recommendations, and next steps.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "processing_result": response,
            "processed_at": datetime.utcnow().isoformat()
        }
    
    def _determine_urgency_level(self, symptoms: list, clinical_data: dict) -> str:
        """Determine urgency level based on clinical indicators"""
        severity = clinical_data.get('severity', '5')
        
        # High-risk symptoms that require urgent referral
        urgent_symptoms = [
            "chest pain", "difficulty breathing", "severe headache", 
            "neurological deficits", "severe abdominal pain", "high fever"
        ]
        
        if any(symptom.lower() in str(symptoms).lower() for symptom in urgent_symptoms):
            return "urgent"
        elif severity and int(severity) >= 8:
            return "expedite"
        elif severity and int(severity) >= 6:
            return "priority"
        else:
            return "routine"
    
    def _calculate_timeline(self, severity: str) -> str:
        """Calculate recommended timeline based on severity"""
        try:
            severity_level = int(severity) if severity else 5
            
            if severity_level >= 9:
                return "within_24_hours"
            elif severity_level >= 7:
                return "within_1_week"
            elif severity_level >= 5:
                return "within_2_weeks"
            else:
                return "within_1_month"
        except (ValueError, TypeError):
            return "within_2_weeks"  # Default timeline