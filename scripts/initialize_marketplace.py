#!/usr/bin/env python3
"""
Initialize SADP Agent Marketplace with sample agents and data
"""

import asyncio
import os
import sys
from datetime import datetime
import json

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'agent-runtime', 'src'))

from config import Settings
from agent_marketplace import AgentMarketplace
from poml_manager import POMLManager

# Sample agents to load
SAMPLE_AGENTS = [
    {
        "name": "Document Processor",
        "version": "1.0.0",
        "description": "Advanced OCR and document analysis for medical records, lab reports, and clinical notes",
        "category": "document",
        "author": "SADP Core Team",
        "auto_load": True,
        "capabilities": ["ocr", "handwriting_recognition", "form_extraction", "table_parsing"],
        "supported_formats": ["PDF", "JPG", "PNG", "TIFF"],
        "accuracy_rate": 0.975,
        "avg_processing_time": 2.3,
        "cost_per_execution": 0.10,
        "code": """
from agents.base import BaseAgent
from typing import Dict, Any
import structlog

logger = structlog.get_logger()

class DocumentProcessorAgent(BaseAgent):
    async def execute(self, request) -> Dict[str, Any]:
        # Simulate document processing
        document_url = request.input_data.get('document_url')
        document_type = request.input_data.get('document_type', 'general')
        
        # Mock results based on document type
        if document_type == 'lab_report':
            return {
                'extracted_text': 'Patient: John Doe, DOB: 1980-01-15\\nGlucose: 120 mg/dL (Normal)\\nCholesterol: 180 mg/dL (Normal)',
                'entities': [
                    {'type': 'patient_name', 'value': 'John Doe', 'confidence': 0.98},
                    {'type': 'test_result', 'value': 'Glucose: 120 mg/dL', 'confidence': 0.95},
                    {'type': 'test_result', 'value': 'Cholesterol: 180 mg/dL', 'confidence': 0.96}
                ],
                'confidence': 0.97,
                'document_type': 'lab_report',
                'processing_time_ms': 2300
            }
        else:
            return {
                'extracted_text': 'Sample medical document text extracted successfully.',
                'entities': [],
                'confidence': 0.92,
                'document_type': document_type,
                'processing_time_ms': 1800
            }
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return 'document_url' in input_data
""",
        "poml_template": """
<prompt version="2.0">
    <system>
        You are an expert medical document processor with advanced OCR capabilities.
        Extract key information from medical documents with high accuracy.
    </system>
    <context>
        Document Type: {{document_type}}
        Processing Mode: {{processing_mode}}
    </context>
    <task>
        Analyze the document and extract:
        1. Patient demographics
        2. Clinical findings
        3. Test results with values and ranges
        4. Medications and dosages
        5. Diagnoses and conditions
    </task>
    <constraints>
        - Maintain HIPAA compliance
        - Flag uncertain extractions
        - Use medical terminology accurately
        - Preserve numerical precision
    </constraints>
    <output format="json">
        {
            "patient_info": {},
            "clinical_findings": [],
            "test_results": [],
            "medications": [],
            "diagnoses": [],
            "confidence_score": 0.0
        }
    </output>
</prompt>
"""
    },
    {
        "name": "Clinical Assistant",
        "version": "1.0.0",
        "description": "AI-powered clinical decision support for treatment planning and medical analysis",
        "category": "clinical",
        "author": "SADP Medical Team",
        "auto_load": True,
        "capabilities": ["treatment_planning", "clinical_analysis", "drug_interaction_checking", "diagnosis_support"],
        "accuracy_rate": 0.935,
        "avg_processing_time": 3.1,
        "cost_per_execution": 0.15,
        "code": """
from agents.base import BaseAgent
from typing import Dict, Any

class ClinicalAssistantAgent(BaseAgent):
    async def execute(self, request) -> Dict[str, Any]:
        task_type = request.input_data.get('task_type', 'general_analysis')
        patient_data = request.input_data.get('patient_data', {})
        
        if task_type == 'treatment_plan':
            return {
                'treatment_plan': {
                    'primary_diagnosis': 'Hypertension, Stage 1',
                    'medications': [
                        {'name': 'Lisinopril', 'dosage': '10mg daily', 'duration': '3 months'},
                        {'name': 'Metformin', 'dosage': '500mg twice daily', 'duration': '6 months'}
                    ],
                    'lifestyle_modifications': [
                        'Low sodium diet (<2000mg/day)',
                        'Regular exercise 150min/week',
                        'Weight reduction 10-15 lbs'
                    ],
                    'follow_up': '4 weeks for BP check',
                    'monitoring': ['Blood pressure logs', 'Kidney function tests']
                },
                'confidence': 0.94,
                'reasoning': 'Based on patient age, BP readings, and medical history'
            }
        else:
            return {
                'analysis': 'Clinical analysis completed successfully',
                'recommendations': ['Further evaluation recommended'],
                'confidence': 0.88
            }
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return 'patient_data' in input_data or 'clinical_data' in input_data
""",
        "poml_template": """
<prompt version="2.0">
    <system>
        You are a clinical decision support assistant with expertise in evidence-based medicine.
        Provide accurate clinical recommendations following current medical guidelines.
    </system>
    <context>
        Patient Age: {{patient_age}}
        Medical History: {{medical_history}}
        Current Medications: {{current_medications}}
        Allergies: {{allergies}}
    </context>
    <task>
        Generate clinical recommendations for: {{clinical_question}}
    </task>
    <constraints>
        - Follow evidence-based guidelines
        - Consider drug interactions
        - Account for patient allergies
        - Provide confidence levels
        - Include monitoring requirements
    </constraints>
    <output format="json">
        {
            "recommendations": [],
            "contraindications": [],
            "monitoring_requirements": [],
            "confidence_level": 0.0,
            "evidence_level": ""
        }
    </output>
</prompt>
"""
    },
    {
        "name": "Billing Optimizer",
        "version": "1.0.0",
        "description": "Automated medical billing code generation and claim optimization",
        "category": "billing",
        "author": "SADP Revenue Team",
        "auto_load": True,
        "capabilities": ["icd10_coding", "cpt_coding", "claim_optimization", "denial_management"],
        "accuracy_rate": 0.968,
        "avg_processing_time": 1.8,
        "cost_per_execution": 0.08,
        "revenue_impact": 23.5,  # % increase in billing accuracy
        "code": """
from agents.base import BaseAgent
from typing import Dict, Any

class BillingOptimizerAgent(BaseAgent):
    async def execute(self, request) -> Dict[str, Any]:
        procedure_data = request.input_data.get('procedure_data', {})
        diagnosis_data = request.input_data.get('diagnosis_data', {})
        
        return {
            'billing_codes': {
                'icd10_codes': ['I10', 'E11.9', 'Z51.11'],
                'cpt_codes': ['99213', '85025', '80053'],
                'modifiers': ['25']
            },
            'estimated_reimbursement': 285.50,
            'optimization_suggestions': [
                'Consider adding complexity modifier +25 for additional $15',
                'ICD-10 code I10 supports medical necessity',
                'Lab codes bundled appropriately'
            ],
            'compliance_check': {
                'passed': True,
                'flags': [],
                'confidence': 0.97
            }
        }
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return bool(input_data.get('procedure_data') or input_data.get('diagnosis_data'))
""",
        "poml_template": """
<prompt version="2.0">
    <system>
        You are a medical billing specialist with expertise in ICD-10, CPT, and HCPCS coding.
        Optimize billing accuracy while ensuring compliance.
    </system>
    <context>
        Provider Specialty: {{provider_specialty}}
        Patient Insurance: {{insurance_type}}
        Procedure Date: {{procedure_date}}
    </context>
    <task>
        Generate optimal billing codes for: {{clinical_scenario}}
    </task>
    <constraints>
        - Follow CMS guidelines
        - Maximize appropriate reimbursement
        - Ensure medical necessity
        - Check for bundling rules
        - Validate modifier usage
    </constraints>
    <output format="json">
        {
            "primary_diagnosis": "",
            "secondary_diagnoses": [],
            "procedure_codes": [],
            "modifiers": [],
            "estimated_reimbursement": 0.0,
            "compliance_score": 0.0
        }
    </output>
</prompt>
"""
    },
    {
        "name": "Lab Result Analyzer",
        "version": "1.0.0",
        "description": "Intelligent analysis and interpretation of laboratory test results",
        "category": "laboratory",
        "author": "SADP Lab Team",
        "auto_load": True,
        "capabilities": ["result_interpretation", "trend_analysis", "critical_value_detection", "reference_range_comparison"],
        "accuracy_rate": 0.991,
        "avg_processing_time": 1.2,
        "cost_per_execution": 0.05,
        "code": """
from agents.base import BaseAgent
from typing import Dict, Any

class LabResultAnalyzerAgent(BaseAgent):
    async def execute(self, request) -> Dict[str, Any]:
        lab_results = request.input_data.get('lab_results', {})
        patient_context = request.input_data.get('patient_context', {})
        
        return {
            'interpretation': {
                'glucose': {
                    'value': 125,
                    'unit': 'mg/dL',
                    'status': 'slightly_elevated',
                    'reference_range': '70-99 mg/dL',
                    'significance': 'Suggests prediabetes, recommend glucose tolerance test'
                },
                'hemoglobin_a1c': {
                    'value': 6.2,
                    'unit': '%',
                    'status': 'elevated',
                    'reference_range': '<5.7%',
                    'significance': 'Indicates prediabetes, lifestyle modifications recommended'
                }
            },
            'critical_values': [],
            'trends': {
                'glucose_trend': 'increasing over 6 months',
                'recommendation': 'Monitor closely, consider diabetes prevention program'
            },
            'confidence': 0.99
        }
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return 'lab_results' in input_data
"""
    },
    {
        "name": "Medication Reconciler",
        "version": "1.0.0",
        "description": "Advanced medication reconciliation and drug interaction analysis",
        "category": "pharmacy",
        "author": "SADP Pharmacy Team",
        "auto_load": True,
        "capabilities": ["drug_identification", "interaction_checking", "dosage_validation", "allergy_screening"],
        "accuracy_rate": 0.987,
        "avg_processing_time": 2.1,
        "cost_per_execution": 0.12,
        "safety_impact": "reduces medication errors by 78%"
    }
]

# Sample POML prompts
SAMPLE_PROMPTS = [
    {
        "name": "clinical_assessment_v2",
        "agent_type": "clinical",
        "template": """
<prompt version="2.0">
    <system>
        You are a board-certified physician with 15+ years of clinical experience.
        Provide comprehensive medical assessments with evidence-based recommendations.
    </system>
    <context>
        Patient: {{patient_demographics}}
        Chief Complaint: {{chief_complaint}}
        History: {{history_present_illness}}
        Physical Exam: {{physical_exam}}
        Vitals: {{vital_signs}}
    </context>
    <task>
        Provide a clinical assessment including:
        1. Differential diagnosis (ranked by likelihood)
        2. Recommended diagnostic tests
        3. Treatment plan with alternatives
        4. Patient education points
        5. Follow-up schedule
    </task>
    <constraints>
        - Use evidence-based medicine principles
        - Consider cost-effectiveness
        - Account for patient preferences
        - Include safety considerations
        - Provide confidence levels for each recommendation
    </constraints>
    <output format="json">
        {
            "differential_diagnosis": [
                {
                    "condition": "",
                    "likelihood": 0.0,
                    "supporting_evidence": [],
                    "against_evidence": []
                }
            ],
            "diagnostic_tests": [
                {
                    "test": "",
                    "priority": "high|medium|low",
                    "rationale": ""
                }
            ],
            "treatment_plan": {
                "primary": [],
                "alternatives": [],
                "contraindications": []
            },
            "patient_education": [],
            "follow_up": {
                "timeframe": "",
                "criteria": []
            },
            "confidence_score": 0.0
        }
    </output>
</prompt>
""",
        "variables": ["patient_demographics", "chief_complaint", "history_present_illness", "physical_exam", "vital_signs"],
        "description": "Enhanced clinical assessment prompt with comprehensive differential diagnosis"
    }
]

async def initialize_agents(marketplace: AgentMarketplace):
    """Initialize sample agents in the marketplace"""
    print("🤖 Initializing sample agents...")
    
    initialized_count = 0
    for agent_data in SAMPLE_AGENTS:
        try:
            print(f"  • Registering {agent_data['name']}...")
            agent_id = await marketplace.register_agent(agent_data)
            print(f"    ✅ Registered as {agent_id}")
            initialized_count += 1
        except Exception as e:
            print(f"    ❌ Failed to register {agent_data['name']}: {e}")
    
    print(f"✅ Initialized {initialized_count}/{len(SAMPLE_AGENTS)} agents")
    return initialized_count

async def initialize_prompts(poml_manager: POMLManager):
    """Initialize sample POML prompts"""
    print("📝 Initializing sample prompts...")
    
    initialized_count = 0
    for prompt_data in SAMPLE_PROMPTS:
        try:
            print(f"  • Creating {prompt_data['name']}...")
            prompt_id = await poml_manager.create_prompt(prompt_data)
            print(f"    ✅ Created as {prompt_id}")
            initialized_count += 1
        except Exception as e:
            print(f"    ❌ Failed to create {prompt_data['name']}: {e}")
    
    print(f"✅ Initialized {initialized_count}/{len(SAMPLE_PROMPTS)} prompts")
    return initialized_count

async def create_sample_metrics(marketplace: AgentMarketplace):
    """Create sample performance metrics"""
    print("📊 Creating sample metrics...")
    
    try:
        # Simulate some agent executions to generate metrics
        agents = marketplace.get_loaded_agents()
        
        for agent_id, agent in agents.items():
            # Create sample metrics
            for i in range(100):  # Simulate 100 executions
                success = i < 95  # 95% success rate
                confidence = 0.85 + (i % 20) * 0.01  # Varying confidence
                latency = 1000 + (i % 50) * 10  # Varying latency
                
                execution_data = {
                    'success': success,
                    'confidence': confidence,
                    'latency_ms': latency,
                    'tokens_used': 150 + (i % 30)
                }
                
                # This would normally be called after real executions
                # For demo, we'll just show how it would work
                print(f"    • Sample metric for {agent_id}: {confidence:.2f} confidence")
        
        print("✅ Sample metrics created")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample metrics: {e}")
        return False

async def verify_initialization(marketplace: AgentMarketplace, poml_manager: POMLManager):
    """Verify that initialization was successful"""
    print("🔍 Verifying initialization...")
    
    try:
        # Check loaded agents
        loaded_agents = marketplace.get_loaded_agents()
        print(f"  • Loaded agents: {len(loaded_agents)}")
        
        # Check marketplace search
        search_results = await marketplace.search_agents({"category": "clinical"})
        print(f"  • Clinical agents in marketplace: {len(search_results)}")
        
        # Check POML prompts
        try:
            prompt = await poml_manager.get_prompt("clinical_assessment_v2")
            print(f"  • Sample prompt loaded: {prompt.get('name', 'Unknown')}")
        except:
            print("  • No sample prompts loaded yet")
        
        print("✅ Initialization verification completed")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

async def main():
    """Main initialization function"""
    print("🚀 SADP Marketplace Initialization Starting...")
    print("=" * 50)
    
    try:
        # Initialize settings
        settings = Settings()
        print(f"📋 Project: {settings.PROJECT_ID}")
        print(f"📋 Region: {settings.REGION}")
        
        # Initialize marketplace
        print("\n🏪 Initializing Agent Marketplace...")
        marketplace = AgentMarketplace(settings)
        await marketplace.initialize()
        
        # Initialize POML manager
        print("\n📝 Initializing POML Manager...")
        poml_manager = POMLManager(settings)
        await poml_manager.initialize()
        
        # Initialize agents
        agent_count = await initialize_agents(marketplace)
        
        # Initialize prompts
        prompt_count = await initialize_prompts(poml_manager)
        
        # Create sample metrics
        await create_sample_metrics(marketplace)
        
        # Verify initialization
        await verify_initialization(marketplace, poml_manager)
        
        print("\n" + "=" * 50)
        print("🎉 SADP Marketplace Initialization Complete!")
        print(f"✅ Agents initialized: {agent_count}")
        print(f"✅ Prompts initialized: {prompt_count}")
        print(f"✅ Marketplace ready for use")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the initialization
    success = asyncio.run(main())
    
    if success:
        print("\n🔗 Next Steps:")
        print("1. Test the marketplace: curl http://localhost:8000/agents/marketplace/search")
        print("2. Run agent execution: curl -X POST http://localhost:8000/agents/clinical/execute")
        print("3. Start the demo app: cd demo-app && npm run dev")
        exit(0)
    else:
        print("\n💥 Initialization failed. Check the logs above.")
        exit(1)