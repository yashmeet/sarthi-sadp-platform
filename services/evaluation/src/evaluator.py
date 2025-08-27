"""
Core evaluation engine for agent performance assessment
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import statistics

import structlog
from google.cloud import firestore, pubsub_v1
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from models import (
    EvaluationRequest, EvaluationResult, EvaluationConfig, TestCase,
    TestCaseResult, BenchmarkResult, AgentComparison
)
from config import Settings

logger = structlog.get_logger()

class EvaluationEngine:
    """Core engine for evaluating agent performance"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.firestore_client = firestore.Client(project=settings.PROJECT_ID)
        self.publisher = pubsub_v1.PublisherClient()
        
        # Collections
        self.evaluations_collection = self.firestore_client.collection('evaluations')
        self.agents_collection = self.firestore_client.collection('agents')
        self.benchmarks_collection = self.firestore_client.collection('benchmarks')
        
        # Pub/Sub topics
        self.evaluation_topic = self.publisher.topic_path(
            settings.PROJECT_ID, "evaluation-events"
        )
        
        # Evaluation metrics
        self.supported_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'latency', 'confidence', 'consistency', 'robustness'
        ]
        
    async def evaluate_agent(
        self,
        agent_id: str,
        test_cases: List[TestCase],
        config: EvaluationConfig,
        user_id: str,
        organization_id: str
    ) -> EvaluationResult:
        """Evaluate an agent against test cases"""
        
        evaluation_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        logger.info(
            "Starting agent evaluation",
            evaluation_id=evaluation_id,
            agent_id=agent_id,
            test_case_count=len(test_cases)
        )
        
        try:
            # Initialize evaluation record
            evaluation_record = {
                'evaluation_id': evaluation_id,
                'agent_id': agent_id,
                'organization_id': organization_id,
                'user_id': user_id,
                'status': 'running',
                'start_time': start_time,
                'config': config.dict(),
                'test_case_count': len(test_cases),
                'created_at': start_time
            }
            
            # Store initial record
            self.evaluations_collection.document(evaluation_id).set(evaluation_record)
            
            # Publish evaluation started event
            await self._publish_evaluation_event({
                'evaluation_id': evaluation_id,
                'event_type': 'evaluation_started',
                'agent_id': agent_id,
                'timestamp': start_time.isoformat()
            })
            
            # Execute test cases
            test_results = []
            failed_cases = 0
            total_latency = 0
            confidence_scores = []
            
            for i, test_case in enumerate(test_cases):
                logger.debug(
                    "Executing test case",
                    evaluation_id=evaluation_id,
                    case_index=i,
                    case_id=test_case.case_id
                )
                
                case_result = await self._execute_test_case(
                    agent_id, test_case, config, organization_id
                )
                
                test_results.append(case_result)
                
                if not case_result.passed:
                    failed_cases += 1
                
                total_latency += case_result.execution_time_ms
                if case_result.confidence_score is not None:
                    confidence_scores.append(case_result.confidence_score)
                
                # Update progress
                progress = (i + 1) / len(test_cases) * 100
                await self._publish_evaluation_event({
                    'evaluation_id': evaluation_id,
                    'event_type': 'progress_update',
                    'progress': progress,
                    'completed_cases': i + 1,
                    'total_cases': len(test_cases)
                })
            
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Calculate metrics
            overall_accuracy = (len(test_cases) - failed_cases) / len(test_cases)
            average_latency = total_latency / len(test_cases)
            average_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0
            
            # Advanced metrics calculation
            advanced_metrics = await self._calculate_advanced_metrics(
                test_results, config
            )
            
            # Create evaluation result
            result = EvaluationResult(
                evaluation_id=evaluation_id,
                agent_id=agent_id,
                organization_id=organization_id,
                overall_accuracy=overall_accuracy,
                test_results=test_results,
                evaluation_duration_ms=duration_ms,
                average_latency_ms=average_latency,
                average_confidence=average_confidence,
                failed_cases=failed_cases,
                total_cases=len(test_cases),
                metrics=advanced_metrics,
                config=config,
                timestamp=end_time
            )
            
            # Update evaluation record
            evaluation_record.update({
                'status': 'completed',
                'end_time': end_time,
                'duration_ms': duration_ms,
                'overall_accuracy': overall_accuracy,
                'average_latency_ms': average_latency,
                'failed_cases': failed_cases,
                'metrics': advanced_metrics
            })
            
            self.evaluations_collection.document(evaluation_id).update(evaluation_record)
            
            # Publish completion event
            await self._publish_evaluation_event({
                'evaluation_id': evaluation_id,
                'event_type': 'evaluation_completed',
                'agent_id': agent_id,
                'accuracy': overall_accuracy,
                'duration_ms': duration_ms,
                'timestamp': end_time.isoformat()
            })
            
            logger.info(
                "Agent evaluation completed",
                evaluation_id=evaluation_id,
                accuracy=overall_accuracy,
                duration_ms=duration_ms
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Agent evaluation failed",
                evaluation_id=evaluation_id,
                error=str(e)
            )
            
            # Update evaluation record with error
            self.evaluations_collection.document(evaluation_id).update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.utcnow()
            })
            
            # Publish failure event
            await self._publish_evaluation_event({
                'evaluation_id': evaluation_id,
                'event_type': 'evaluation_failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
            
            raise
    
    async def _execute_test_case(
        self,
        agent_id: str,
        test_case: TestCase,
        config: EvaluationConfig,
        organization_id: str
    ) -> TestCaseResult:
        """Execute a single test case against an agent"""
        
        case_start = datetime.utcnow()
        
        try:
            # Make request to agent runtime service
            agent_request = {
                'agent_id': agent_id,
                'input_data': test_case.input_data,
                'organization_id': organization_id,
                'metadata': {
                    'evaluation_mode': True,
                    'test_case_id': test_case.case_id
                }
            }
            
            # Simulate agent execution (replace with actual agent runtime call)
            # This would normally call the agent runtime service
            agent_response = await self._call_agent_runtime(agent_request)
            
            case_end = datetime.utcnow()
            execution_time = int((case_end - case_start).total_seconds() * 1000)
            
            # Validate response
            validation_result = await self._validate_response(
                agent_response, test_case.expected_output, config
            )
            
            return TestCaseResult(
                case_id=test_case.case_id,
                input_data=test_case.input_data,
                expected_output=test_case.expected_output,
                actual_output=agent_response.get('output'),
                passed=validation_result['passed'],
                confidence_score=agent_response.get('confidence'),
                execution_time_ms=execution_time,
                error_message=validation_result.get('error'),
                metadata=agent_response.get('metadata', {}),
                timestamp=case_end
            )
            
        except Exception as e:
            case_end = datetime.utcnow()
            execution_time = int((case_end - case_start).total_seconds() * 1000)
            
            logger.error(
                "Test case execution failed",
                case_id=test_case.case_id,
                error=str(e)
            )
            
            return TestCaseResult(
                case_id=test_case.case_id,
                input_data=test_case.input_data,
                expected_output=test_case.expected_output,
                actual_output=None,
                passed=False,
                confidence_score=0.0,
                execution_time_ms=execution_time,
                error_message=str(e),
                metadata={},
                timestamp=case_end
            )
    
    async def _call_agent_runtime(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Call the agent runtime service"""
        # TODO: Implement actual HTTP call to agent runtime service
        # For now, simulate a response
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            'output': 'Simulated agent response',
            'confidence': 0.85,
            'metadata': {
                'processing_time_ms': 100,
                'model_version': '1.0.0'
            }
        }
    
    async def _validate_response(
        self,
        actual: Dict[str, Any],
        expected: Any,
        config: EvaluationConfig
    ) -> Dict[str, Any]:
        """Validate agent response against expected output"""
        
        try:
            actual_output = actual.get('output')
            
            if config.validation_mode == 'exact':
                passed = actual_output == expected
            elif config.validation_mode == 'semantic':
                # Implement semantic similarity check
                passed = await self._semantic_similarity(actual_output, expected) > config.similarity_threshold
            elif config.validation_mode == 'custom':
                # Use custom validation function
                passed = await self._custom_validation(actual_output, expected, config)
            else:
                passed = str(actual_output).strip().lower() == str(expected).strip().lower()
            
            return {
                'passed': passed,
                'similarity_score': await self._semantic_similarity(actual_output, expected) if config.validation_mode == 'semantic' else None
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    async def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        # TODO: Implement semantic similarity using embeddings
        # For now, simple string similarity
        if not text1 or not text2:
            return 0.0
        
        text1 = str(text1).lower().strip()
        text2 = str(text2).lower().strip()
        
        if text1 == text2:
            return 1.0
        
        # Simple Jaccard similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _custom_validation(
        self, 
        actual: Any, 
        expected: Any, 
        config: EvaluationConfig
    ) -> bool:
        """Custom validation logic"""
        # TODO: Implement custom validation based on config
        return str(actual) == str(expected)
    
    async def _calculate_advanced_metrics(
        self,
        test_results: List[TestCaseResult],
        config: EvaluationConfig
    ) -> Dict[str, float]:
        """Calculate advanced evaluation metrics"""
        
        metrics = {}
        
        # Basic metrics
        passed_count = sum(1 for result in test_results if result.passed)
        total_count = len(test_results)
        
        metrics['accuracy'] = passed_count / total_count if total_count > 0 else 0.0
        
        # Latency metrics
        latencies = [result.execution_time_ms for result in test_results]
        metrics['avg_latency_ms'] = statistics.mean(latencies)
        metrics['median_latency_ms'] = statistics.median(latencies)
        metrics['p95_latency_ms'] = np.percentile(latencies, 95)
        metrics['p99_latency_ms'] = np.percentile(latencies, 99)
        
        # Confidence metrics
        confidences = [
            result.confidence_score for result in test_results 
            if result.confidence_score is not None
        ]
        if confidences:
            metrics['avg_confidence'] = statistics.mean(confidences)
            metrics['min_confidence'] = min(confidences)
            metrics['max_confidence'] = max(confidences)
            metrics['confidence_std'] = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        
        # Consistency metrics
        if len(confidences) > 1:
            metrics['consistency_score'] = 1.0 - (statistics.stdev(confidences) / statistics.mean(confidences))
        else:
            metrics['consistency_score'] = 1.0
        
        # Error analysis
        error_count = sum(1 for result in test_results if result.error_message)
        metrics['error_rate'] = error_count / total_count if total_count > 0 else 0.0
        
        return metrics
    
    async def run_benchmark(
        self,
        agent_ids: List[str],
        config: Dict[str, Any],
        organization_id: str
    ) -> BenchmarkResult:
        """Run performance benchmark across multiple agents"""
        
        benchmark_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        logger.info(
            "Starting benchmark",
            benchmark_id=benchmark_id,
            agent_count=len(agent_ids)
        )
        
        try:
            # Load benchmark test cases
            test_cases = await self._load_benchmark_test_cases(config)
            
            # Run evaluations for each agent
            agent_results = {}
            for agent_id in agent_ids:
                evaluation_config = EvaluationConfig(
                    validation_mode=config.get('validation_mode', 'exact'),
                    similarity_threshold=config.get('similarity_threshold', 0.8),
                    timeout_ms=config.get('timeout_ms', 30000)
                )
                
                result = await self.evaluate_agent(
                    agent_id=agent_id,
                    test_cases=test_cases,
                    config=evaluation_config,
                    user_id="benchmark",
                    organization_id=organization_id
                )
                
                agent_results[agent_id] = result
            
            # Compare results
            comparison = await self._compare_agents(agent_results)
            
            end_time = datetime.utcnow()
            duration = int((end_time - start_time).total_seconds() * 1000)
            
            # Create benchmark result
            benchmark_result = BenchmarkResult(
                benchmark_id=benchmark_id,
                agent_ids=agent_ids,
                agent_results=agent_results,
                comparison=comparison,
                best_performing_agent=comparison.best_agent_id,
                benchmark_duration_ms=duration,
                config=config,
                timestamp=end_time
            )
            
            # Store benchmark result
            benchmark_record = {
                'benchmark_id': benchmark_id,
                'organization_id': organization_id,
                'agent_ids': agent_ids,
                'config': config,
                'duration_ms': duration,
                'best_agent': comparison.best_agent_id,
                'results': {
                    agent_id: {
                        'accuracy': result.overall_accuracy,
                        'avg_latency': result.average_latency_ms,
                        'confidence': result.average_confidence
                    }
                    for agent_id, result in agent_results.items()
                },
                'created_at': start_time,
                'completed_at': end_time
            }
            
            self.benchmarks_collection.document(benchmark_id).set(benchmark_record)
            
            logger.info(
                "Benchmark completed",
                benchmark_id=benchmark_id,
                winner=comparison.best_agent_id,
                duration_ms=duration
            )
            
            return benchmark_result
            
        except Exception as e:
            logger.error("Benchmark failed", benchmark_id=benchmark_id, error=str(e))
            raise
    
    async def _load_benchmark_test_cases(self, config: Dict[str, Any]) -> List[TestCase]:
        """Load test cases for benchmark"""
        # TODO: Load test cases from configuration or database
        # For now, create sample test cases
        return [
            TestCase(
                case_id=f"bench_case_{i}",
                input_data=f"Sample input {i}",
                expected_output=f"Expected output {i}",
                category="benchmark",
                difficulty="medium"
            )
            for i in range(config.get('test_case_count', 10))
        ]
    
    async def _compare_agents(
        self, 
        agent_results: Dict[str, EvaluationResult]
    ) -> AgentComparison:
        """Compare agent performance results"""
        
        # Calculate rankings
        rankings = {}
        
        # Rank by accuracy
        accuracy_ranking = sorted(
            agent_results.items(),
            key=lambda x: x[1].overall_accuracy,
            reverse=True
        )
        
        # Rank by latency (lower is better)
        latency_ranking = sorted(
            agent_results.items(),
            key=lambda x: x[1].average_latency_ms
        )
        
        # Rank by confidence
        confidence_ranking = sorted(
            agent_results.items(),
            key=lambda x: x[1].average_confidence,
            reverse=True
        )
        
        # Calculate overall score (weighted combination)
        for agent_id, result in agent_results.items():
            accuracy_score = result.overall_accuracy * 0.5
            latency_score = (1.0 - min(result.average_latency_ms / 10000, 1.0)) * 0.3
            confidence_score = result.average_confidence * 0.2
            
            rankings[agent_id] = accuracy_score + latency_score + confidence_score
        
        # Find best agent
        best_agent_id = max(rankings.keys(), key=lambda k: rankings[k])
        
        return AgentComparison(
            best_agent_id=best_agent_id,
            accuracy_ranking=[agent_id for agent_id, _ in accuracy_ranking],
            latency_ranking=[agent_id for agent_id, _ in latency_ranking],
            confidence_ranking=[agent_id for agent_id, _ in confidence_ranking],
            overall_scores=rankings
        )
    
    async def get_evaluation_result(
        self, 
        evaluation_id: str, 
        organization_id: str
    ) -> Optional[EvaluationResult]:
        """Get evaluation result by ID"""
        
        try:
            doc = self.evaluations_collection.document(evaluation_id).get()
            
            if not doc.exists:
                return None
            
            data = doc.to_dict()
            
            # Verify organization access
            if data.get('organization_id') != organization_id:
                return None
            
            # TODO: Reconstruct EvaluationResult from stored data
            # For now, return a simplified version
            
            return None  # Placeholder
            
        except Exception as e:
            logger.error("Failed to get evaluation result", error=str(e))
            return None
    
    async def start_continuous_evaluation(self):
        """Start continuous evaluation background task"""
        logger.info("Starting continuous evaluation service")
        
        try:
            while True:
                # Check for scheduled evaluations
                await self._process_scheduled_evaluations()
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
        except asyncio.CancelledError:
            logger.info("Continuous evaluation service stopped")
        except Exception as e:
            logger.error("Continuous evaluation service error", error=str(e))
    
    async def _process_scheduled_evaluations(self):
        """Process scheduled evaluations"""
        # TODO: Implement scheduled evaluation processing
        pass
    
    async def _publish_evaluation_event(self, event_data: Dict[str, Any]):
        """Publish evaluation event to Pub/Sub"""
        try:
            message_data = json.dumps(event_data).encode('utf-8')
            future = self.publisher.publish(self.evaluation_topic, message_data)
            future.result()  # Wait for publish confirmation
            
        except Exception as e:
            logger.error("Failed to publish evaluation event", error=str(e))