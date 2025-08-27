"""
Simplified PHI Protection Service for SADP - Production Ready
Basic PHI detection and sanitization
"""

import os
import logging
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SADP PHI Protection Service",
    description="HIPAA-compliant PHI detection and sanitization",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PHIType(str, Enum):
    SSN = "ssn"
    PHONE = "phone"
    EMAIL = "email"
    DATE_OF_BIRTH = "date_of_birth"
    MEDICAL_RECORD_NUMBER = "medical_record_number"
    NAME = "name"

class SanitizationLevel(str, Enum):
    MASK = "mask"
    REMOVE = "remove"
    HASH = "hash"

class PHIDetection(BaseModel):
    phi_type: PHIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float

class DetectPHIRequest(BaseModel):
    text: str
    detection_types: Optional[List[PHIType]] = None

class DetectPHIResponse(BaseModel):
    detections: List[PHIDetection]
    has_phi: bool
    processed_at: str

class SanitizePHIRequest(BaseModel):
    text: str
    sanitization_level: SanitizationLevel = SanitizationLevel.MASK

class SanitizePHIResponse(BaseModel):
    sanitized_text: str
    phi_detected: bool
    detections_count: int
    sanitization_level: str
    processed_at: str

# PHI detection patterns
PHI_PATTERNS = {
    PHIType.SSN: r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b',
    PHIType.PHONE: r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b',
    PHIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    PHIType.DATE_OF_BIRTH: r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b',
    PHIType.MEDICAL_RECORD_NUMBER: r'\bMRN\s*:?\s*\d+\b|\b[A-Z]{2,3}\d{6,}\b'
}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "phi-protection",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.post("/phi/detect", response_model=DetectPHIResponse)
async def detect_phi(request: DetectPHIRequest):
    """Detect PHI in text"""
    detections = []
    text = request.text
    
    detection_types = request.detection_types or list(PHIType)
    
    for phi_type in detection_types:
        if phi_type in PHI_PATTERNS:
            pattern = PHI_PATTERNS[phi_type]
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                detection = PHIDetection(
                    phi_type=phi_type,
                    value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.95  # High confidence for pattern matching
                )
                detections.append(detection)
    
    return DetectPHIResponse(
        detections=detections,
        has_phi=len(detections) > 0,
        processed_at=datetime.utcnow().isoformat()
    )

@app.post("/phi/sanitize", response_model=SanitizePHIResponse)
async def sanitize_phi(request: SanitizePHIRequest):
    """Sanitize PHI in text"""
    text = request.text
    sanitization_level = request.sanitization_level
    detections_count = 0
    
    # Detect and sanitize PHI
    for phi_type, pattern in PHI_PATTERNS.items():
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        detections_count += len(matches)
        
        if matches:
            if sanitization_level == SanitizationLevel.MASK:
                # Replace with asterisks
                for match in reversed(matches):  # Reverse to maintain positions
                    replacement = "*" * len(match.group())
                    text = text[:match.start()] + replacement + text[match.end():]
            elif sanitization_level == SanitizationLevel.REMOVE:
                # Remove completely
                for match in reversed(matches):
                    text = text[:match.start()] + text[match.end():]
            elif sanitization_level == SanitizationLevel.HASH:
                # Replace with hash placeholder
                for match in reversed(matches):
                    replacement = f"[{phi_type.value.upper()}_REDACTED]"
                    text = text[:match.start()] + replacement + text[match.end():]
    
    return SanitizePHIResponse(
        sanitized_text=text,
        phi_detected=detections_count > 0,
        detections_count=detections_count,
        sanitization_level=sanitization_level.value,
        processed_at=datetime.utcnow().isoformat()
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SADP PHI Protection Service",
        "version": "1.0.0",
        "status": "running",
        "features": ["phi_detection", "phi_sanitization", "hipaa_compliance"],
        "endpoints": ["/health", "/phi/detect", "/phi/sanitize"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)