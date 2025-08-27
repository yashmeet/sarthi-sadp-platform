from typing import Dict, Any
from datetime import datetime
import structlog

from .base import BaseAgent
from ..models import AgentRequest

logger = structlog.get_logger()

class MedicationEntryAgent(BaseAgent):
    """Agent for AI-assisted medication entry, prescription processing, and medication reconciliation"""
    
    async def execute(self, request: AgentRequest) -> Dict[str, Any]:
        """Execute medication entry processing"""
        start_time = datetime.utcnow()
        
        try:
            # Validate input
            if not await self.validate_input(request.input_data):
                raise ValueError("Invalid input data")
            
            # Determine medication entry task type
            task_type = request.input_data.get("task_type", "medication_entry")
            
            result = {}
            
            if task_type == "prescription_processing":
                result = await self._process_prescription(request)
            elif task_type == "medication_reconciliation":
                result = await self._reconcile_medications(request)
            elif task_type == "dosage_calculation":
                result = await self._calculate_dosage(request)
            elif task_type == "interaction_check":
                result = await self._check_drug_interactions(request)
            elif task_type == "allergy_screening":
                result = await self._screen_allergies(request)
            else:
                result = await self._general_medication_entry(request)
            
            # Log execution
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_execution(request.request_id, "success", execution_time_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Medication entry processing failed", 
                        request_id=request.request_id,
                        error=str(e))
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_execution(request.request_id, "failed", execution_time_ms)
            raise
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        required_fields = ["medication_data"]
        for field in required_fields:
            if field not in input_data:
                logger.error(f"Missing required field: {field}")
                return False
        return True
    
    async def _process_prescription(self, request: AgentRequest) -> Dict[str, Any]:
        """Process prescription and validate medication details"""
        prescription_data = request.input_data.get("prescription_data", {})
        patient_info = request.input_data.get("patient_info", {})
        prescriber_info = request.input_data.get("prescriber_info", {})
        
        prompt = f"""
        Process and validate the following prescription:
        
        Patient Information:
        - Name: {patient_info.get('name', '')}
        - Age: {patient_info.get('age', 'Unknown')}
        - Weight: {patient_info.get('weight', 'Unknown')}
        - Height: {patient_info.get('height', 'Unknown')}
        - Allergies: {patient_info.get('allergies', 'None')}
        - Current Medications: {patient_info.get('current_medications', 'None')}
        - Medical Conditions: {patient_info.get('conditions', 'None')}
        
        Prescriber Information:
        - Name: {prescriber_info.get('name', '')}
        - NPI: {prescriber_info.get('npi', '')}
        - DEA Number: {prescriber_info.get('dea', '')}
        - Specialty: {prescriber_info.get('specialty', '')}
        
        Prescription Details:
        - Medication: {prescription_data.get('medication_name', '')}
        - Strength: {prescription_data.get('strength', '')}
        - Dosage Form: {prescription_data.get('dosage_form', '')}
        - Quantity: {prescription_data.get('quantity', '')}
        - Directions: {prescription_data.get('directions', '')}
        - Refills: {prescription_data.get('refills', '')}
        - Date Prescribed: {prescription_data.get('date_prescribed', '')}
        
        Please validate and provide:
        1. Medication name verification (brand/generic)
        2. Dosage appropriateness for patient
        3. Drug interaction warnings
        4. Allergy contraindications
        5. Dosing frequency validation
        6. Quantity and refill appropriateness
        7. Required monitoring parameters
        8. Patient counseling points
        9. Prescription completeness check
        
        Format as structured prescription validation report.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "prescription_validation": response,
            "medication_name": prescription_data.get("medication_name", ""),
            "patient_id": patient_info.get("patient_id", ""),
            "prescriber_id": prescriber_info.get("prescriber_id", ""),
            "validation_status": "pending_review",
            "processed_at": datetime.utcnow().isoformat()
        }
    
    async def _reconcile_medications(self, request: AgentRequest) -> Dict[str, Any]:
        """Reconcile patient medications across different sources"""
        current_medications = request.input_data.get("current_medications", [])
        new_medications = request.input_data.get("new_medications", [])
        patient_info = request.input_data.get("patient_info", {})
        
        prompt = f"""
        Perform medication reconciliation for the following patient:
        
        Patient Information:
        - Age: {patient_info.get('age', 'Unknown')}
        - Weight: {patient_info.get('weight', 'Unknown')}
        - Kidney Function: {patient_info.get('kidney_function', 'Normal')}
        - Liver Function: {patient_info.get('liver_function', 'Normal')}
        - Allergies: {patient_info.get('allergies', 'None')}
        
        Current Medications:
        {self._format_medication_list(current_medications)}
        
        New/Proposed Medications:
        {self._format_medication_list(new_medications)}
        
        Please provide medication reconciliation analysis:
        1. Duplicate therapy identification
        2. Drug interaction analysis
        3. Contraindication warnings
        4. Dosage appropriateness
        5. Therapeutic alternatives
        6. Discontinuation recommendations
        7. Monitoring requirements
        8. Patient safety concerns
        9. Cost-effectiveness considerations
        
        Format as comprehensive medication reconciliation report.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "reconciliation_report": response,
            "current_medications": current_medications,
            "new_medications": new_medications,
            "patient_id": patient_info.get("patient_id", ""),
            "conflicts_identified": True,  # Would be determined by analysis
            "reconciled_at": datetime.utcnow().isoformat()
        }
    
    async def _calculate_dosage(self, request: AgentRequest) -> Dict[str, Any]:
        """Calculate appropriate medication dosage based on patient parameters"""
        medication_info = request.input_data.get("medication_info", {})
        patient_info = request.input_data.get("patient_info", {})
        indication = request.input_data.get("indication", "")
        
        prompt = f"""
        Calculate appropriate dosage for:
        
        Patient Parameters:
        - Age: {patient_info.get('age', 'Unknown')}
        - Weight: {patient_info.get('weight', 'Unknown')} kg
        - Height: {patient_info.get('height', 'Unknown')} cm
        - BSA (if applicable): {patient_info.get('bsa', 'Calculate if needed')}
        - Kidney Function (CrCl): {patient_info.get('creatinine_clearance', 'Normal')}
        - Liver Function: {patient_info.get('liver_function', 'Normal')}
        
        Medication:
        - Drug: {medication_info.get('drug_name', '')}
        - Indication: {indication}
        - Route: {medication_info.get('route', 'Oral')}
        - Frequency: {medication_info.get('frequency', 'To be determined')}
        
        Special Considerations:
        - Pregnancy: {patient_info.get('pregnancy_status', 'N/A')}
        - Comorbidities: {patient_info.get('comorbidities', 'None')}
        - Concurrent Medications: {patient_info.get('concurrent_medications', 'None')}
        
        Please provide dosage calculation with:
        1. Recommended starting dose
        2. Dose adjustment factors considered
        3. Maximum daily dose
        4. Titration schedule (if applicable)
        5. Monitoring parameters
        6. Dose modifications for special populations
        7. Alternative dosing strategies
        8. Safety considerations
        
        Format as detailed dosage calculation report.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "dosage_calculation": response,
            "medication": medication_info.get("drug_name", ""),
            "patient_id": patient_info.get("patient_id", ""),
            "indication": indication,
            "calculated_dose": "See detailed report",
            "calculated_at": datetime.utcnow().isoformat()
        }
    
    async def _check_drug_interactions(self, request: AgentRequest) -> Dict[str, Any]:
        """Check for drug-drug interactions"""
        medications = request.input_data.get("medications", [])
        patient_info = request.input_data.get("patient_info", {})
        
        prompt = f"""
        Analyze drug interactions for the following medication list:
        
        Patient Information:
        - Age: {patient_info.get('age', 'Unknown')}
        - Kidney Function: {patient_info.get('kidney_function', 'Normal')}
        - Liver Function: {patient_info.get('liver_function', 'Normal')}
        - Conditions: {patient_info.get('conditions', 'None')}
        
        Medications to Check:
        {self._format_medication_list(medications)}
        
        Please provide interaction analysis:
        1. Major drug interactions (contraindicated)
        2. Moderate interactions (monitor closely)
        3. Minor interactions (awareness needed)
        4. Mechanism of interactions
        5. Clinical significance
        6. Management strategies
        7. Alternative medication options
        8. Monitoring recommendations
        9. Patient counseling points
        
        Format as drug interaction screening report.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "interaction_analysis": response,
            "medications_checked": medications,
            "patient_id": patient_info.get("patient_id", ""),
            "interaction_level": "moderate",  # Would be determined by analysis
            "checked_at": datetime.utcnow().isoformat()
        }
    
    async def _screen_allergies(self, request: AgentRequest) -> Dict[str, Any]:
        """Screen for drug allergies and contraindications"""
        medication = request.input_data.get("medication", "")
        allergies = request.input_data.get("allergies", [])
        patient_info = request.input_data.get("patient_info", {})
        
        prompt = f"""
        Screen for allergies and contraindications:
        
        Proposed Medication: {medication}
        
        Patient Allergy Profile:
        {', '.join(allergies) if isinstance(allergies, list) else allergies}
        
        Patient Information:
        - Age: {patient_info.get('age', 'Unknown')}
        - Previous Adverse Reactions: {patient_info.get('adverse_reactions', 'None')}
        - Family History of Allergies: {patient_info.get('family_allergies', 'Unknown')}
        
        Please provide allergy screening:
        1. Direct allergy contraindications
        2. Cross-reactivity concerns
        3. Alternative medications if contraindicated
        4. Allergy testing recommendations
        5. Emergency action plans
        6. Patient education needs
        7. Documentation requirements
        8. Risk mitigation strategies
        
        Format as allergy screening report.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "allergy_screening": response,
            "medication": medication,
            "known_allergies": allergies,
            "patient_id": patient_info.get("patient_id", ""),
            "contraindicated": False,  # Would be determined by analysis
            "screened_at": datetime.utcnow().isoformat()
        }
    
    async def _general_medication_entry(self, request: AgentRequest) -> Dict[str, Any]:
        """Process general medication entry"""
        medication_data = request.input_data.get("medication_data", {})
        
        prompt = f"""
        Process the following medication entry data:
        
        {medication_data}
        
        Provide medication entry validation, safety checks, and recommendations.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "medication_entry": response,
            "processed_at": datetime.utcnow().isoformat()
        }
    
    def _format_medication_list(self, medications: list) -> str:
        """Format medication list for prompt"""
        if not medications:
            return "No medications provided"
        
        formatted = []
        for i, med in enumerate(medications, 1):
            if isinstance(med, dict):
                med_str = f"{i}. {med.get('name', 'Unknown medication')}"
                if med.get('strength'):
                    med_str += f" {med.get('strength')}"
                if med.get('dosage'):
                    med_str += f" - {med.get('dosage')}"
                if med.get('frequency'):
                    med_str += f" - {med.get('frequency')}"
                formatted.append(med_str)
            else:
                formatted.append(f"{i}. {med}")
        
        return "\n".join(formatted)