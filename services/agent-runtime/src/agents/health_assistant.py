from typing import Dict, Any
from datetime import datetime
import structlog

from .base import BaseAgent
from ..models import AgentRequest

logger = structlog.get_logger()

class HealthAssistantAgent(BaseAgent):
    """Agent for patient health assistance, symptom assessment, health education, and care coordination"""
    
    async def execute(self, request: AgentRequest) -> Dict[str, Any]:
        """Execute health assistant functionality"""
        start_time = datetime.utcnow()
        
        try:
            # Validate input
            if not await self.validate_input(request.input_data):
                raise ValueError("Invalid input data")
            
            # Determine health assistant task type
            task_type = request.input_data.get("task_type", "general_assistance")
            
            result = {}
            
            if task_type == "symptom_assessment":
                result = await self._assess_symptoms(request)
            elif task_type == "health_education":
                result = await self._provide_health_education(request)
            elif task_type == "medication_reminders":
                result = await self._manage_medication_reminders(request)
            elif task_type == "appointment_scheduling":
                result = await self._assist_appointment_scheduling(request)
            elif task_type == "wellness_coaching":
                result = await self._provide_wellness_coaching(request)
            else:
                result = await self._general_health_assistance(request)
            
            # Log execution
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_execution(request.request_id, "success", execution_time_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Health assistant processing failed", 
                        request_id=request.request_id,
                        error=str(e))
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_execution(request.request_id, "failed", execution_time_ms)
            raise
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        required_fields = ["patient_query"]
        for field in required_fields:
            if field not in input_data:
                logger.error(f"Missing required field: {field}")
                return False
        return True
    
    async def _assess_symptoms(self, request: AgentRequest) -> Dict[str, Any]:
        """Assess patient symptoms and provide guidance"""
        symptoms = request.input_data.get("symptoms", [])
        patient_info = request.input_data.get("patient_info", {})
        duration = request.input_data.get("symptom_duration", "")
        severity = request.input_data.get("severity", "")
        
        prompt = f"""
        Provide a symptom assessment for the following patient inquiry:
        
        Patient Information:
        - Age: {patient_info.get('age', 'Unknown')}
        - Gender: {patient_info.get('gender', 'Unknown')}
        - Medical History: {patient_info.get('medical_history', 'None provided')}
        - Current Medications: {patient_info.get('medications', 'None')}
        - Allergies: {patient_info.get('allergies', 'None')}
        
        Symptoms Reported:
        {', '.join(symptoms) if isinstance(symptoms, list) else symptoms}
        
        Symptom Details:
        - Duration: {duration}
        - Severity (1-10): {severity}
        
        Please provide:
        1. Symptom analysis and potential causes
        2. Red flag symptoms requiring immediate attention
        3. Self-care recommendations
        4. When to seek medical attention
        5. Questions for healthcare provider
        6. Lifestyle modifications
        7. Follow-up recommendations
        
        IMPORTANT: Always recommend consulting healthcare providers for medical advice.
        Format as patient-friendly health guidance.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "symptom_assessment": response,
            "urgency_level": self._assess_urgency(symptoms, severity),
            "patient_id": patient_info.get("patient_id", ""),
            "symptoms": symptoms,
            "assessed_at": datetime.utcnow().isoformat()
        }
    
    async def _provide_health_education(self, request: AgentRequest) -> Dict[str, Any]:
        """Provide health education and information"""
        topic = request.input_data.get("education_topic", "")
        patient_info = request.input_data.get("patient_info", {})
        education_level = request.input_data.get("education_level", "general")
        
        prompt = f"""
        Provide health education on the following topic for a patient:
        
        Topic: {topic}
        
        Patient Context:
        - Age: {patient_info.get('age', 'Adult')}
        - Education Level: {education_level}
        - Language Preference: {patient_info.get('language', 'English')}
        - Current Conditions: {patient_info.get('conditions', 'None')}
        
        Please provide:
        1. Clear explanation of the topic
        2. Why this information is important
        3. Practical tips and recommendations
        4. Common misconceptions to address
        5. Resources for further learning
        6. When to consult healthcare providers
        7. Action steps the patient can take
        
        Tailor the language and complexity to the patient's education level.
        Use encouraging and supportive tone.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "health_education": response,
            "topic": topic,
            "education_level": education_level,
            "patient_id": patient_info.get("patient_id", ""),
            "provided_at": datetime.utcnow().isoformat()
        }
    
    async def _manage_medication_reminders(self, request: AgentRequest) -> Dict[str, Any]:
        """Manage medication reminders and adherence"""
        medications = request.input_data.get("medications", [])
        patient_info = request.input_data.get("patient_info", {})
        reminder_preferences = request.input_data.get("reminder_preferences", {})
        
        prompt = f"""
        Create medication management plan and reminders for:
        
        Patient Information:
        - Age: {patient_info.get('age', 'Unknown')}
        - Timezone: {patient_info.get('timezone', 'UTC')}
        - Lifestyle: {patient_info.get('lifestyle_notes', '')}
        
        Current Medications:
        {self._format_medications(medications)}
        
        Reminder Preferences:
        - Preferred Times: {reminder_preferences.get('preferred_times', 'Morning, Evening')}
        - Reminder Method: {reminder_preferences.get('method', 'App notification')}
        - Frequency: {reminder_preferences.get('frequency', 'Daily')}
        
        Please provide:
        1. Optimal medication schedule
        2. Reminder timing recommendations
        3. Drug interaction warnings
        4. Side effect monitoring
        5. Adherence tips and strategies
        6. Storage and handling instructions
        7. What to do if dose is missed
        
        Format as a comprehensive medication management plan.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "medication_plan": response,
            "medications": medications,
            "next_reminder": self._calculate_next_reminder(medications),
            "patient_id": patient_info.get("patient_id", ""),
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def _assist_appointment_scheduling(self, request: AgentRequest) -> Dict[str, Any]:
        """Assist with appointment scheduling and preparation"""
        appointment_type = request.input_data.get("appointment_type", "")
        patient_info = request.input_data.get("patient_info", {})
        preferences = request.input_data.get("scheduling_preferences", {})
        reason = request.input_data.get("reason_for_visit", "")
        
        prompt = f"""
        Assist with appointment scheduling and preparation:
        
        Appointment Details:
        - Type: {appointment_type}
        - Reason: {reason}
        - Urgency: {request.input_data.get('urgency', 'routine')}
        
        Patient Information:
        - Age: {patient_info.get('age', 'Unknown')}
        - Insurance: {patient_info.get('insurance', 'Unknown')}
        - Preferred Provider: {patient_info.get('preferred_provider', 'Any')}
        - Location Preference: {patient_info.get('location_preference', 'Any')}
        
        Scheduling Preferences:
        - Preferred Days: {preferences.get('preferred_days', 'Weekdays')}
        - Preferred Times: {preferences.get('preferred_times', 'Morning')}
        - Transportation: {preferences.get('transportation', 'Personal vehicle')}
        
        Please provide:
        1. Appointment scheduling guidance
        2. Questions to ask when scheduling
        3. Information to have ready
        4. Pre-appointment preparation
        5. What to bring to the appointment
        6. Insurance verification steps
        7. Follow-up recommendations
        
        Format as appointment assistance guide.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "appointment_assistance": response,
            "appointment_type": appointment_type,
            "patient_id": patient_info.get("patient_id", ""),
            "scheduling_priority": self._determine_priority(appointment_type, reason),
            "provided_at": datetime.utcnow().isoformat()
        }
    
    async def _provide_wellness_coaching(self, request: AgentRequest) -> Dict[str, Any]:
        """Provide wellness coaching and lifestyle guidance"""
        wellness_goals = request.input_data.get("wellness_goals", [])
        patient_info = request.input_data.get("patient_info", {})
        current_habits = request.input_data.get("current_habits", {})
        
        prompt = f"""
        Provide personalized wellness coaching for:
        
        Patient Profile:
        - Age: {patient_info.get('age', 'Unknown')}
        - Activity Level: {patient_info.get('activity_level', 'Moderate')}
        - Health Conditions: {patient_info.get('conditions', 'None')}
        - Stress Level: {patient_info.get('stress_level', 'Moderate')}
        
        Wellness Goals:
        {', '.join(wellness_goals) if isinstance(wellness_goals, list) else wellness_goals}
        
        Current Habits:
        - Exercise: {current_habits.get('exercise', 'Occasional')}
        - Diet: {current_habits.get('diet', 'Average')}
        - Sleep: {current_habits.get('sleep_hours', '7-8 hours')}
        - Stress Management: {current_habits.get('stress_management', 'Limited')}
        
        Please provide:
        1. Personalized wellness plan
        2. Achievable short-term goals
        3. Long-term wellness strategy
        4. Habit formation techniques
        5. Progress tracking methods
        6. Motivation and accountability tips
        7. Resources and tools
        
        Format as encouraging wellness coaching plan.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "wellness_coaching": response,
            "wellness_goals": wellness_goals,
            "patient_id": patient_info.get("patient_id", ""),
            "coaching_plan_duration": "30_days",
            "provided_at": datetime.utcnow().isoformat()
        }
    
    async def _general_health_assistance(self, request: AgentRequest) -> Dict[str, Any]:
        """Provide general health assistance"""
        patient_query = request.input_data.get("patient_query", "")
        patient_info = request.input_data.get("patient_info", {})
        
        prompt = f"""
        Provide helpful health assistance for the following patient inquiry:
        
        Patient Query: {patient_query}
        
        Patient Context:
        - Age: {patient_info.get('age', 'Unknown')}
        - General Health: {patient_info.get('health_status', 'Unknown')}
        
        Provide supportive, informative, and encouraging guidance while always
        recommending consultation with healthcare providers for medical advice.
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "assistance": response,
            "query": patient_query,
            "patient_id": patient_info.get("patient_id", ""),
            "provided_at": datetime.utcnow().isoformat()
        }
    
    def _format_medications(self, medications: list) -> str:
        """Format medications for prompt"""
        if not medications:
            return "No medications provided"
        
        formatted = []
        for med in medications:
            if isinstance(med, dict):
                formatted.append(f"- {med.get('name', 'Unknown')}: {med.get('dosage', '')} {med.get('frequency', '')}")
            else:
                formatted.append(f"- {med}")
        return "\n".join(formatted)
    
    def _assess_urgency(self, symptoms: list, severity: str) -> str:
        """Assess urgency level of symptoms"""
        # Simple urgency assessment logic
        high_risk_symptoms = ["chest pain", "difficulty breathing", "severe headache", "high fever"]
        
        if any(symptom.lower() in str(symptoms).lower() for symptom in high_risk_symptoms):
            return "high"
        elif severity and int(severity) >= 8:
            return "high"
        elif severity and int(severity) >= 6:
            return "medium"
        else:
            return "low"
    
    def _calculate_next_reminder(self, medications: list) -> str:
        """Calculate next medication reminder time"""
        # Simplified calculation - would be more complex in real implementation
        return datetime.now().replace(hour=8, minute=0, second=0).isoformat()
    
    def _determine_priority(self, appointment_type: str, reason: str) -> str:
        """Determine scheduling priority"""
        urgent_types = ["emergency", "urgent care", "follow-up"]
        if appointment_type.lower() in urgent_types or "urgent" in reason.lower():
            return "high"
        elif "routine" in appointment_type.lower():
            return "low"
        else:
            return "medium"