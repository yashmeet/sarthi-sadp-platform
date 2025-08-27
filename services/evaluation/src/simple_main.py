"""
SADP Evaluation Service - Simple Working Version
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

app = FastAPI(
    title="SADP Evaluation Service",
    description="Agent performance evaluation and metrics collection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluationRequest(BaseModel):
    agent_id: str
    test_cases: List[Dict[str, Any]]
    config: Optional[Dict[str, Any]] = None

class EvaluationResult(BaseModel):
    evaluation_id: str
    agent_id: str
    overall_accuracy: float
    results: List[Dict[str, Any]]
    timestamp: str

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "evaluation"}

@app.get("/")
async def root():
    return {"message": "SADP Evaluation Service is running"}

@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate_agent(request: EvaluationRequest):
    # Simple mock evaluation
    import uuid
    from datetime import datetime
    
    evaluation_id = str(uuid.uuid4())
    results = []
    total_score = 0
    
    for i, test_case in enumerate(request.test_cases):
        score = 0.85 + (i % 3) * 0.05  # Mock varying scores
        results.append({
            "test_case_id": f"test_{i}",
            "score": score,
            "passed": score > 0.8
        })
        total_score += score
    
    overall_accuracy = total_score / len(request.test_cases) if request.test_cases else 0
    
    return EvaluationResult(
        evaluation_id=evaluation_id,
        agent_id=request.agent_id,
        overall_accuracy=overall_accuracy,
        results=results,
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/evaluations/{evaluation_id}")
async def get_evaluation_result(evaluation_id: str):
    # Mock response
    return {
        "evaluation_id": evaluation_id,
        "status": "completed",
        "results": {"accuracy": 0.87, "score": 87}
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)