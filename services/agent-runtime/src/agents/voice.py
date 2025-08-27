from typing import Dict, Any
from datetime import datetime
import structlog

from .base import BaseAgent
from ..models import AgentRequest

logger = structlog.get_logger()

class VoiceAgent(BaseAgent):
    """Agent for voice-to-text transcription, medical dictation processing, and voice command interpretation"""
    
    async def execute(self, request: AgentRequest) -> Dict[str, Any]:
        """Execute voice processing"""
        start_time = datetime.utcnow()
        
        try:
            # Validate input
            if not await self.validate_input(request.input_data):
                raise ValueError("Invalid input data")
            
            # Determine voice task type
            task_type = request.input_data.get("task_type", "transcription")
            
            result = {}
            
            if task_type == "transcription":
                result = await self._transcribe_audio(request)
            elif task_type == "medical_dictation":
                result = await self._process_medical_dictation(request)
            elif task_type == "voice_commands":
                result = await self._interpret_voice_commands(request)
            elif task_type == "clinical_note_dictation":
                result = await self._process_clinical_dictation(request)
            else:
                result = await self._general_voice_processing(request)
            
            # Log execution
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_execution(request.request_id, "success", execution_time_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Voice processing failed", 
                        request_id=request.request_id,
                        error=str(e))
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_execution(request.request_id, "failed", execution_time_ms)
            raise
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        # Check for either audio data or transcribed text
        has_audio = "audio_data" in input_data or "audio_url" in input_data
        has_text = "transcribed_text" in input_data
        
        if not (has_audio or has_text):
            logger.error("Missing required field: audio_data, audio_url, or transcribed_text")
            return False
        return True
    
    async def _transcribe_audio(self, request: AgentRequest) -> Dict[str, Any]:
        """Transcribe audio to text"""
        audio_data = request.input_data.get("audio_data", "")
        audio_url = request.input_data.get("audio_url", "")
        audio_format = request.input_data.get("audio_format", "mp3")
        
        # Note: In a real implementation, this would use Google Cloud Speech-to-Text API
        # For now, we'll simulate transcription processing
        
        prompt = f"""
        Process the following audio transcription request:
        
        Audio Information:
        - Format: {audio_format}
        - Source: {'URL provided' if audio_url else 'Direct upload'}
        - Context: {request.input_data.get('context', 'General medical')}
        
        If transcribed text is provided for processing:
        {request.input_data.get('transcribed_text', 'Audio file to be processed')}
        
        Please provide:
        1. Cleaned and formatted transcription
        2. Medical terminology corrections
        3. Punctuation and capitalization
        4. Confidence assessment
        5. Identified medical entities
        6. Suggested follow-up actions
        
        Format as a structured transcription result.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "transcription": response,
            "audio_source": audio_url if audio_url else "uploaded_file",
            "audio_format": audio_format,
            "confidence_score": 0.95,  # Simulated confidence score
            "processed_at": datetime.utcnow().isoformat()
        }
    
    async def _process_medical_dictation(self, request: AgentRequest) -> Dict[str, Any]:
        """Process medical dictation and format as clinical documentation"""
        dictation_text = request.input_data.get("transcribed_text", "")
        provider_info = request.input_data.get("provider_info", {})
        patient_context = request.input_data.get("patient_context", {})
        
        prompt = f"""
        Process the following medical dictation into structured clinical documentation:
        
        Provider Information:
        - Name: {provider_info.get('name', 'Unknown')}
        - Specialty: {provider_info.get('specialty', 'Unknown')}
        - NPI: {provider_info.get('npi', 'Unknown')}
        
        Patient Context:
        - Patient ID: {patient_context.get('patient_id', '')}
        - Visit Type: {patient_context.get('visit_type', '')}
        - Date of Service: {patient_context.get('date_of_service', '')}
        
        Dictated Content:
        {dictation_text}
        
        Please format this into:
        1. Structured clinical note (SOAP format)
        2. Corrected medical terminology
        3. Proper medical abbreviations
        4. ICD-10 and CPT code suggestions
        5. Missing information flagged
        6. Documentation quality assessment
        7. Compliance with documentation standards
        
        Format as professional medical documentation.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "formatted_documentation": response,
            "provider_id": provider_info.get("provider_id", ""),
            "patient_id": patient_context.get("patient_id", ""),
            "original_dictation": dictation_text,
            "processed_at": datetime.utcnow().isoformat()
        }
    
    async def _interpret_voice_commands(self, request: AgentRequest) -> Dict[str, Any]:
        """Interpret voice commands for EHR navigation and actions"""
        command_text = request.input_data.get("transcribed_text", "")
        context = request.input_data.get("context", {})
        
        prompt = f"""
        Interpret the following voice command for EHR system interaction:
        
        Current Context:
        - User Role: {context.get('user_role', 'Provider')}
        - Current Screen: {context.get('current_screen', 'Unknown')}
        - Patient Context: {context.get('patient_id', 'None')}
        
        Voice Command:
        "{command_text}"
        
        Please provide:
        1. Command interpretation
        2. Required actions/navigation
        3. Parameters extracted
        4. Confirmation required (yes/no)
        5. Potential ambiguities
        6. Security considerations
        7. Alternative interpretations
        
        Format as structured command analysis.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "command_interpretation": response,
            "original_command": command_text,
            "confidence": 0.90,  # Simulated confidence
            "requires_confirmation": True,
            "processed_at": datetime.utcnow().isoformat()
        }
    
    async def _process_clinical_dictation(self, request: AgentRequest) -> Dict[str, Any]:
        """Process clinical note dictation with medical context awareness"""
        dictation_text = request.input_data.get("transcribed_text", "")
        note_type = request.input_data.get("note_type", "progress_note")
        patient_info = request.input_data.get("patient_info", {})
        
        prompt = f"""
        Process clinical dictation for a {note_type}:
        
        Patient Information:
        - Name: {patient_info.get('name', '')}
        - MRN: {patient_info.get('mrn', '')}
        - Age: {patient_info.get('age', '')}
        - Primary Diagnosis: {patient_info.get('primary_diagnosis', '')}
        
        Dictated Content:
        {dictation_text}
        
        Please provide:
        1. Structured clinical note based on note type
        2. Medical terminology validation
        3. Missing critical information
        4. Suggested template completion
        5. Quality metrics assessment
        6. Billing code recommendations
        7. Required signatures/attestations
        
        Ensure compliance with medical documentation standards.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "structured_note": response,
            "note_type": note_type,
            "patient_mrn": patient_info.get("mrn", ""),
            "dictation_quality": "high",  # Simulated quality assessment
            "requires_review": False,
            "processed_at": datetime.utcnow().isoformat()
        }
    
    async def _general_voice_processing(self, request: AgentRequest) -> Dict[str, Any]:
        """Perform general voice processing"""
        voice_data = request.input_data
        
        prompt = f"""
        Process the following voice-related data:
        
        {voice_data}
        
        Provide voice processing insights, transcription quality assessment, and recommendations.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "analysis": response,
            "processed_at": datetime.utcnow().isoformat()
        }