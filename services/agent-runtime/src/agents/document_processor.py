from typing import Dict, Any
from datetime import datetime
import structlog
from google.cloud import documentai_v1 as documentai
from google.cloud import storage

from .base import BaseAgent
from ..models import AgentRequest

logger = structlog.get_logger()

class DocumentProcessorAgent(BaseAgent):
    """Agent for OCR, handwriting recognition, and form extraction"""
    
    def __init__(self, settings):
        super().__init__(settings)
        self.doc_ai_client = None
        self.storage_client = None
        self.processor_id = None
    
    async def initialize(self):
        """Initialize Document AI and Storage clients"""
        await super().initialize()
        try:
            self.doc_ai_client = documentai.DocumentProcessorServiceClient()
            self.storage_client = storage.Client(project=self.settings.PROJECT_ID)
            
            # Create or get processor
            self.processor_id = await self._get_or_create_processor()
            
        except Exception as e:
            logger.error(f"Failed to initialize DocumentProcessorAgent", error=str(e))
            raise
    
    async def execute(self, request: AgentRequest) -> Dict[str, Any]:
        """Process document using Document AI"""
        start_time = datetime.utcnow()
        
        try:
            # Validate input
            if not await self.validate_input(request.input_data):
                raise ValueError("Invalid input data")
            
            # Preprocess
            processed_input = await self.preprocess(request.input_data)
            
            # Get document content
            document_content = await self._get_document_content(
                processed_input.get("document_url")
            )
            
            # Process with Document AI
            document = await self._process_document(
                document_content,
                processed_input.get("document_type", "general")
            )
            
            # Extract information based on document type
            extracted_data = await self._extract_information(
                document,
                processed_input.get("document_type", "general")
            )
            
            # Use Gemini for additional analysis if needed
            if processed_input.get("enable_ai_analysis", True):
                ai_analysis = await self._analyze_with_gemini(
                    extracted_data,
                    request.context
                )
                extracted_data["ai_analysis"] = ai_analysis
            
            # Postprocess
            result = await self.postprocess(extracted_data)
            
            # Log execution
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_execution(request.request_id, "success", execution_time_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed", 
                        request_id=request.request_id,
                        error=str(e))
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.log_execution(request.request_id, "failed", execution_time_ms)
            raise
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        required_fields = ["document_url"]
        for field in required_fields:
            if field not in input_data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate document URL format
        doc_url = input_data["document_url"]
        if not (doc_url.startswith("gs://") or doc_url.startswith("http")):
            logger.error(f"Invalid document URL format: {doc_url}")
            return False
        
        return True
    
    async def _get_or_create_processor(self) -> str:
        """Get or create Document AI processor"""
        # For now, return a placeholder
        # In production, this would create/retrieve an actual processor
        return f"projects/{self.settings.PROJECT_ID}/locations/us/processors/doc-processor"
    
    async def _get_document_content(self, document_url: str) -> bytes:
        """Get document content from URL"""
        if document_url.startswith("gs://"):
            # Get from Cloud Storage
            bucket_name = document_url.split("/")[2]
            blob_name = "/".join(document_url.split("/")[3:])
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return blob.download_as_bytes()
        else:
            # Get from HTTP URL
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(document_url)
                return response.content
    
    async def _process_document(self, content: bytes, document_type: str) -> Any:
        """Process document with Document AI"""
        # Create document object
        document = documentai.Document(content=content)
        
        # Create request
        request = documentai.ProcessRequest(
            name=self.processor_id,
            raw_document=documentai.RawDocument(
                content=content,
                mime_type="application/pdf"
            )
        )
        
        # Process document
        result = self.doc_ai_client.process_document(request=request)
        return result.document
    
    async def _extract_information(self, document: Any, document_type: str) -> Dict[str, Any]:
        """Extract information from processed document"""
        extracted = {
            "text": document.text if hasattr(document, 'text') else "",
            "pages": len(document.pages) if hasattr(document, 'pages') else 0,
            "entities": [],
            "tables": [],
            "form_fields": []
        }
        
        # Extract entities
        if hasattr(document, 'entities'):
            for entity in document.entities:
                extracted["entities"].append({
                    "type": entity.type_,
                    "mention_text": entity.mention_text,
                    "confidence": entity.confidence
                })
        
        # Extract form fields
        if hasattr(document, 'pages'):
            for page in document.pages:
                if hasattr(page, 'form_fields'):
                    for field in page.form_fields:
                        extracted["form_fields"].append({
                            "field_name": field.field_name.text_anchor.content if field.field_name else "",
                            "field_value": field.field_value.text_anchor.content if field.field_value else "",
                            "confidence": field.confidence
                        })
        
        return extracted
    
    async def _analyze_with_gemini(self, extracted_data: Dict, context: Dict) -> Dict[str, Any]:
        """Analyze extracted data with Gemini"""
        prompt = f"""
        Analyze the following extracted document data and provide insights:
        
        Document Text: {extracted_data.get('text', '')[:2000]}
        Entities Found: {extracted_data.get('entities', [])}
        Form Fields: {extracted_data.get('form_fields', [])}
        
        Context: {context}
        
        Please provide:
        1. Document summary
        2. Key information extracted
        3. Any data quality issues
        4. Recommendations for further processing
        """
        
        response = await self.call_gemini(prompt)
        
        return {
            "analysis": response,
            "timestamp": datetime.utcnow().isoformat()
        }