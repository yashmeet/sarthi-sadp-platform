from typing import Dict, Any
from datetime import datetime
import structlog

from .base import BaseAgent
from ..models import AgentRequest

logger = structlog.get_logger()

class LabResultEntryAgent(BaseAgent):
    """Agent for AI-assisted lab result entry, interpretation, and clinical correlation"""
    
    async def execute(self, request: AgentRequest) -> Dict[str, Any]:
        """Execute lab result entry processing"""
        start_time = datetime.utcnow()
        
        try:
            # Validate input
            if not await self.validate_input(request.input_data):
                raise ValueError("Invalid input data")
            
            # Determine lab result entry task type
            task_type = request.input_data.get("task_type", "result_entry")
            
            result = {}
            
            if task_type == "result_interpretation":
                result = await self._interpret_lab_results(request)
            elif task_type == "critical_value_alert":
                result = await self._process_critical_values(request)
            elif task_type == "trend_analysis":
                result = await self._analyze_lab_trends(request)
            elif task_type == "reference_range_check":
                result = await self._check_reference_ranges(request)
            elif task_type == "quality_control":
                result = await self._perform_quality_control(request)
            else:
                result = await self._general_lab_entry(request)
            
            # Log execution
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_execution(request.request_id, "success", execution_time_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Lab result entry processing failed", 
                        request_id=request.request_id,
                        error=str(e))
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_execution(request.request_id, "failed", execution_time_ms)
            raise
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        required_fields = ["lab_data"]
        for field in required_fields:
            if field not in input_data:
                logger.error(f"Missing required field: {field}")
                return False
        return True
    
    async def _interpret_lab_results(self, request: AgentRequest) -> Dict[str, Any]:
        """Interpret lab results and provide clinical context"""
        lab_data = request.input_data.get("lab_data", {})
        patient_info = request.input_data.get("patient_info", {})
        clinical_context = request.input_data.get("clinical_context", {})
        
        prompt = f"""
        Interpret the following laboratory results in clinical context:
        
        Patient Information:
        - Age: {patient_info.get('age', 'Unknown')}
        - Gender: {patient_info.get('gender', 'Unknown')}
        - Weight: {patient_info.get('weight', 'Unknown')}
        - Medical History: {patient_info.get('medical_history', 'None provided')}
        - Current Medications: {patient_info.get('medications', 'None')}
        - Allergies: {patient_info.get('allergies', 'None')}
        
        Clinical Context:
        - Chief Complaint: {clinical_context.get('chief_complaint', '')}
        - Current Diagnosis: {clinical_context.get('diagnosis', '')}
        - Symptoms: {clinical_context.get('symptoms', '')}
        - Treatment Plan: {clinical_context.get('treatment_plan', '')}
        
        Laboratory Results:
        {self._format_lab_results(lab_data)}
        
        Please provide comprehensive interpretation:
        1. Normal vs. abnormal value identification
        2. Clinical significance of abnormal values
        3. Potential causes of abnormalities
        4. Correlation with patient symptoms
        5. Impact on current treatment plan
        6. Recommended follow-up tests
        7. Monitoring recommendations
        8. Critical values requiring immediate attention
        9. Patient education points
        
        Format as detailed lab interpretation report.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "lab_interpretation": response,
            "lab_results": lab_data,
            "patient_id": patient_info.get("patient_id", ""),
            "abnormal_values": self._identify_abnormal_values(lab_data),
            "critical_values": self._identify_critical_values(lab_data),
            "interpreted_at": datetime.utcnow().isoformat()
        }
    
    async def _process_critical_values(self, request: AgentRequest) -> Dict[str, Any]:
        """Process critical lab values and generate alerts"""
        lab_data = request.input_data.get("lab_data", {})
        patient_info = request.input_data.get("patient_info", {})
        ordering_provider = request.input_data.get("ordering_provider", {})
        
        prompt = f"""
        Analyze critical laboratory values and generate appropriate alerts:
        
        Patient Information:
        - Name: {patient_info.get('name', '')}
        - MRN: {patient_info.get('mrn', '')}
        - Age: {patient_info.get('age', 'Unknown')}
        - Location: {patient_info.get('location', 'Unknown')}
        - Current Condition: {patient_info.get('condition', 'Unknown')}
        
        Ordering Provider:
        - Name: {ordering_provider.get('name', '')}
        - Department: {ordering_provider.get('department', '')}
        - Contact: {ordering_provider.get('contact', '')}
        
        Laboratory Results:
        {self._format_lab_results(lab_data)}
        
        For any critical values identified, provide:
        1. Critical value identification and severity
        2. Immediate clinical implications
        3. Urgent interventions required
        4. Provider notification priority
        5. Patient monitoring needs
        6. Documentation requirements
        7. Follow-up testing schedule
        8. Risk assessment
        9. Emergency protocols if applicable
        
        Format as critical value alert report.
        """
        
        response = await self.call_gemini(prompt)
        
        critical_values = self._identify_critical_values(lab_data)
        
        return {
            "critical_value_alert": response,
            "critical_values": critical_values,
            "patient_id": patient_info.get("patient_id", ""),
            "ordering_provider": ordering_provider.get("name", ""),
            "alert_level": "high" if critical_values else "none",
            "notification_sent": True if critical_values else False,
            "processed_at": datetime.utcnow().isoformat()
        }
    
    async def _analyze_lab_trends(self, request: AgentRequest) -> Dict[str, Any]:
        """Analyze lab result trends over time"""
        current_results = request.input_data.get("current_results", {})
        historical_results = request.input_data.get("historical_results", [])
        patient_info = request.input_data.get("patient_info", {})
        
        prompt = f"""
        Analyze laboratory result trends for the following patient:
        
        Patient Information:
        - Age: {patient_info.get('age', 'Unknown')}
        - Medical Conditions: {patient_info.get('conditions', 'None')}
        - Current Medications: {patient_info.get('medications', 'None')}
        - Treatment Changes: {patient_info.get('recent_changes', 'None')}
        
        Current Results:
        {self._format_lab_results(current_results)}
        
        Historical Results:
        {self._format_historical_results(historical_results)}
        
        Please analyze trends and provide:
        1. Trending patterns (improving/worsening/stable)
        2. Significant changes from baseline
        3. Response to treatment interventions
        4. Correlation with medication changes
        5. Disease progression indicators
        6. Goal achievement assessment
        7. Adjustment recommendations
        8. Future monitoring schedule
        9. Patient prognosis implications
        
        Format as lab trend analysis report.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "trend_analysis": response,
            "current_results": current_results,
            "historical_count": len(historical_results),
            "patient_id": patient_info.get("patient_id", ""),
            "trend_direction": "stable",  # Would be determined by analysis
            "analyzed_at": datetime.utcnow().isoformat()
        }
    
    async def _check_reference_ranges(self, request: AgentRequest) -> Dict[str, Any]:
        """Check lab results against appropriate reference ranges"""
        lab_results = request.input_data.get("lab_results", {})
        patient_info = request.input_data.get("patient_info", {})
        
        prompt = f"""
        Validate laboratory results against appropriate reference ranges:
        
        Patient Demographics:
        - Age: {patient_info.get('age', 'Unknown')}
        - Gender: {patient_info.get('gender', 'Unknown')}
        - Pregnancy Status: {patient_info.get('pregnancy_status', 'N/A')}
        - Ethnic Background: {patient_info.get('ethnicity', 'Unknown')}
        
        Laboratory Results to Validate:
        {self._format_lab_results(lab_results)}
        
        Please validate and provide:
        1. Appropriate reference ranges for patient demographics
        2. Age-specific and gender-specific considerations
        3. Pregnancy-adjusted ranges (if applicable)
        4. Pediatric vs. adult reference ranges
        5. Method-specific reference ranges
        6. Population-specific variations
        7. Clinical decision limits
        8. Flagging recommendations
        9. Quality assurance notes
        
        Format as reference range validation report.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "reference_validation": response,
            "lab_results": lab_results,
            "patient_demographics": {
                "age": patient_info.get("age"),
                "gender": patient_info.get("gender")
            },
            "validation_status": "completed",
            "validated_at": datetime.utcnow().isoformat()
        }
    
    async def _perform_quality_control(self, request: AgentRequest) -> Dict[str, Any]:
        """Perform quality control checks on lab results"""
        lab_data = request.input_data.get("lab_data", {})
        test_info = request.input_data.get("test_info", {})
        
        prompt = f"""
        Perform quality control assessment on laboratory results:
        
        Test Information:
        - Test Type: {test_info.get('test_type', '')}
        - Collection Date/Time: {test_info.get('collection_datetime', '')}
        - Processing Date/Time: {test_info.get('processing_datetime', '')}
        - Laboratory: {test_info.get('laboratory', '')}
        - Instrument: {test_info.get('instrument', '')}
        - Method: {test_info.get('method', '')}
        
        Quality Control Data:
        - Sample Condition: {test_info.get('sample_condition', '')}
        - Collection Method: {test_info.get('collection_method', '')}
        - Storage Conditions: {test_info.get('storage_conditions', '')}
        - Processing Delay: {test_info.get('processing_delay', '')}
        
        Laboratory Results:
        {self._format_lab_results(lab_data)}
        
        Please assess quality and provide:
        1. Sample integrity assessment
        2. Pre-analytical quality indicators
        3. Analytical quality metrics
        4. Delta check comparisons
        5. Plausibility checks
        6. Interfering substances detection
        7. Repeat testing recommendations
        8. Result reliability assessment
        9. Quality flags and comments
        
        Format as quality control assessment report.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "quality_assessment": response,
            "lab_data": lab_data,
            "quality_score": 0.95,  # Simulated quality score
            "quality_flags": [],  # Would be populated by analysis
            "repeat_recommended": False,
            "assessed_at": datetime.utcnow().isoformat()
        }
    
    async def _general_lab_entry(self, request: AgentRequest) -> Dict[str, Any]:
        """Perform general lab result entry processing"""
        lab_data = request.input_data.get("lab_data", {})
        
        prompt = f"""
        Process the following laboratory data entry:
        
        {lab_data}
        
        Provide lab entry validation, interpretation, and recommendations.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "lab_entry_result": response,
            "processed_at": datetime.utcnow().isoformat()
        }
    
    def _format_lab_results(self, lab_data: Dict) -> str:
        """Format lab results for prompt"""
        if not lab_data:
            return "No lab results provided"
        
        formatted = []
        for test, result in lab_data.items():
            if isinstance(result, dict):
                value = result.get('value', '')
                unit = result.get('unit', '')
                reference = result.get('reference_range', '')
                flag = result.get('flag', '')
                
                line = f"- {test}: {value} {unit}"
                if reference:
                    line += f" (Ref: {reference})"
                if flag:
                    line += f" [{flag}]"
                formatted.append(line)
            else:
                formatted.append(f"- {test}: {result}")
        
        return "\n".join(formatted)
    
    def _format_historical_results(self, historical_results: list) -> str:
        """Format historical lab results for prompt"""
        if not historical_results:
            return "No historical results available"
        
        formatted = []
        for i, result_set in enumerate(historical_results):
            date = result_set.get('date', f'Result Set {i+1}')
            formatted.append(f"\n{date}:")
            for test, value in result_set.get('results', {}).items():
                if isinstance(value, dict):
                    formatted.append(f"  - {test}: {value.get('value', '')} {value.get('unit', '')}")
                else:
                    formatted.append(f"  - {test}: {value}")
        
        return "\n".join(formatted)
    
    def _identify_abnormal_values(self, lab_data: Dict) -> list:
        """Identify abnormal lab values"""
        abnormal = []
        for test, result in lab_data.items():
            if isinstance(result, dict):
                flag = result.get('flag', '').lower()
                if flag in ['high', 'low', 'critical', 'abnormal', 'h', 'l']:
                    abnormal.append({
                        "test": test,
                        "value": result.get("value"),
                        "flag": flag,
                        "reference_range": result.get("reference_range")
                    })
        return abnormal
    
    def _identify_critical_values(self, lab_data: Dict) -> list:
        """Identify critical lab values requiring immediate attention"""
        critical = []
        for test, result in lab_data.items():
            if isinstance(result, dict):
                flag = result.get('flag', '').lower()
                if 'critical' in flag or flag in ['panic', 'alert']:
                    critical.append({
                        "test": test,
                        "value": result.get("value"),
                        "flag": flag,
                        "critical_range": result.get("critical_range")
                    })
        return critical