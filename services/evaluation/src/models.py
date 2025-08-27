"""
Pydantic models for the evaluation service
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator

class ValidationMode(str, Enum):
    """Validation modes for evaluation"""
    EXACT = "exact"
    SEMANTIC = "semantic" 
    CUSTOM = "custom"
    FUZZY = "fuzzy"

class DifficultyLevel(str, Enum):
    """Test case difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

class EvaluationConfig(BaseModel):
    """Configuration for evaluation"""
    validation_mode: ValidationMode = ValidationMode.EXACT
    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    timeout_ms: int = Field(default=30000, gt=0)
    max_retries: int = Field(default=3, ge=0)
    include_confidence: bool = True
    include_metadata: bool = True
    parallel_execution: bool = False
    
    class Config:
        use_enum_values = True

class TestCase(BaseModel):
    """Individual test case for evaluation"""
    case_id: str
    input_data: Any
    expected_output: Any
    category: str
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    description: Optional[str] = None
    tags: List[str] = []
    weight: float = Field(default=1.0, gt=0.0)
    
    class Config:
        use_enum_values = True

class TestCaseResult(BaseModel):
    """Result of executing a test case"""
    case_id: str
    input_data: Any
    expected_output: Any
    actual_output: Any
    passed: bool
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    execution_time_ms: int
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    timestamp: datetime

class EvaluationRequest(BaseModel):
    """Request to evaluate an agent"""
    agent_id: str
    test_cases: List[TestCase]
    config: EvaluationConfig = EvaluationConfig()
    description: Optional[str] = None
    tags: List[str] = []
    
    @validator('test_cases')
    def validate_test_cases(cls, v):
        if not v:
            raise ValueError('At least one test case is required')
        if len(v) > 1000:
            raise ValueError('Maximum 1000 test cases per evaluation')
        return v

class EvaluationResult(BaseModel):
    """Result of agent evaluation"""
    evaluation_id: str
    agent_id: str
    organization_id: str
    overall_accuracy: float = Field(ge=0.0, le=1.0)
    test_results: List[TestCaseResult]
    evaluation_duration_ms: int
    average_latency_ms: float
    average_confidence: float = Field(ge=0.0, le=1.0)
    failed_cases: int
    total_cases: int
    metrics: Dict[str, float]
    config: EvaluationConfig
    timestamp: datetime
    
    @validator('failed_cases')
    def validate_failed_cases(cls, v, values):
        if 'total_cases' in values and v > values['total_cases']:
            raise ValueError('Failed cases cannot exceed total cases')
        return v

class AgentPerformanceMetrics(BaseModel):
    """Performance metrics for an agent over time"""
    agent_id: str
    organization_id: str
    time_period_start: datetime
    time_period_end: datetime
    total_evaluations: int
    average_accuracy: float = Field(ge=0.0, le=1.0)
    accuracy_trend: List[float]
    average_latency_ms: float
    latency_trend: List[float]
    average_confidence: float = Field(ge=0.0, le=1.0)
    confidence_trend: List[float]
    error_rate: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    improvement_rate: float
    category_performance: Dict[str, float] = {}

class AgentComparison(BaseModel):
    """Comparison results between agents"""
    best_agent_id: str
    accuracy_ranking: List[str]
    latency_ranking: List[str]
    confidence_ranking: List[str]
    overall_scores: Dict[str, float]
    statistical_significance: Dict[str, bool] = {}

class BenchmarkResult(BaseModel):
    """Result of running a benchmark"""
    benchmark_id: str
    agent_ids: List[str]
    agent_results: Dict[str, EvaluationResult]
    comparison: AgentComparison
    best_performing_agent: str
    benchmark_duration_ms: int
    config: Dict[str, Any]
    timestamp: datetime

class ValidationResult(BaseModel):
    """Result of validating evaluation results"""
    evaluation_id: str
    is_valid: bool
    validation_score: float = Field(ge=0.0, le=1.0)
    issues_found: List[str] = []
    recommendations: List[str] = []
    validated_at: datetime

class PerformanceTrend(BaseModel):
    """Performance trend analysis"""
    metric_name: str
    time_series: List[Tuple[datetime, float]]
    trend_direction: str  # 'improving', 'declining', 'stable'
    trend_strength: float = Field(ge=0.0, le=1.0)
    forecast: Optional[List[float]] = None
    anomalies: List[datetime] = []

class AccuracyReport(BaseModel):
    """Comprehensive accuracy analysis report"""
    organization_id: str
    report_period_start: datetime
    report_period_end: datetime
    overall_accuracy: float = Field(ge=0.0, le=1.0)
    agent_accuracies: Dict[str, float]
    category_accuracies: Dict[str, float]
    accuracy_distribution: Dict[str, int]  # bins -> counts
    top_performing_agents: List[str]
    underperforming_agents: List[str]
    recommendations: List[str]
    generated_at: datetime

class EvaluationSummary(BaseModel):
    """Summary statistics for evaluations"""
    organization_id: str
    time_period: str
    total_evaluations: int
    total_test_cases: int
    average_accuracy: float = Field(ge=0.0, le=1.0)
    average_latency_ms: float
    most_evaluated_agent: str
    best_performing_agent: str
    evaluation_frequency: float  # evaluations per day
    resource_usage: Dict[str, Any]

class AlertRule(BaseModel):
    """Alert rule for evaluation monitoring"""
    rule_id: str
    name: str
    description: str
    metric: str  # accuracy, latency, error_rate, etc.
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    window_minutes: int
    severity: str  # 'low', 'medium', 'high', 'critical'
    enabled: bool = True
    organization_id: str
    created_by: str
    created_at: datetime
    
class Alert(BaseModel):
    """Generated alert from monitoring"""
    alert_id: str
    rule_id: str
    agent_id: Optional[str] = None
    organization_id: str
    severity: str
    message: str
    metric_value: float
    threshold_value: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

class EvaluationJob(BaseModel):
    """Scheduled evaluation job"""
    job_id: str
    name: str
    description: str
    agent_id: str
    organization_id: str
    test_suite_id: str
    schedule_cron: str
    config: EvaluationConfig
    enabled: bool = True
    next_run: datetime
    last_run: Optional[datetime] = None
    last_result_id: Optional[str] = None
    created_by: str
    created_at: datetime

class TestSuite(BaseModel):
    """Collection of test cases for reuse"""
    suite_id: str
    name: str
    description: str
    test_cases: List[TestCase]
    organization_id: str
    version: str = "1.0.0"
    tags: List[str] = []
    created_by: str
    created_at: datetime
    updated_at: datetime
    
    @validator('test_cases')
    def validate_test_cases(cls, v):
        if not v:
            raise ValueError('Test suite must contain at least one test case')
        return v