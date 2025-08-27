#!/usr/bin/env python3
"""
Load Testing Framework for SADP Production Services
Tests performance, scalability, and reliability under load
"""

import asyncio
import aiohttp
import time
import statistics
import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
from datetime import datetime
import argparse

@dataclass
class LoadTestConfig:
    """Configuration for load testing"""
    concurrent_users: int = 10
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    target_rps: float = 10.0  # requests per second
    timeout_seconds: int = 30

@dataclass
class TestResult:
    """Individual test result"""
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    timestamp: datetime
    success: bool
    error_message: str = None

@dataclass
class LoadTestStats:
    """Aggregated load test statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0
    min_response_time_ms: float = 0
    max_response_time_ms: float = 0
    p95_response_time_ms: float = 0
    p99_response_time_ms: float = 0
    requests_per_second: float = 0
    error_rate: float = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)

class LoadTester:
    """
    Load testing framework for SADP services
    """
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.active_sessions = 0
        
    async def run_test_scenario(self, scenario_name: str, test_functions: List[Callable]):
        """Run a complete load test scenario"""
        print(f"Starting load test scenario: {scenario_name}")
        print(f"   Concurrent users: {self.config.concurrent_users}")
        print(f"   Duration: {self.config.duration_seconds}s")
        print(f"   Target RPS: {self.config.target_rps}")
        print()
        
        start_time = time.time()
        
        # Create semaphore to limit concurrent users
        semaphore = asyncio.Semaphore(self.config.concurrent_users)
        
        # Start workers
        tasks = []
        for i in range(self.config.concurrent_users):
            task = asyncio.create_task(
                self._worker(f"user_{i}", test_functions, semaphore, start_time)
            )
            tasks.append(task)
        
        # Wait for test duration
        await asyncio.sleep(self.config.duration_seconds)
        
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate and display results
        stats = self._calculate_stats(start_time)
        self._display_results(scenario_name, stats)
        
        return stats
    
    async def _worker(self, user_id: str, test_functions: List[Callable], 
                     semaphore: asyncio.Semaphore, start_time: float):
        """Individual worker that runs test functions"""
        
        # Ramp-up delay
        if self.config.ramp_up_seconds > 0:
            delay = random.uniform(0, self.config.ramp_up_seconds)
            await asyncio.sleep(delay)
        
        async with semaphore:
            self.active_sessions += 1
            
            try:
                while time.time() - start_time < self.config.duration_seconds:
                    # Select random test function
                    test_func = random.choice(test_functions)
                    
                    try:
                        result = await test_func(user_id)
                        if isinstance(result, TestResult):
                            self.results.append(result)
                    except Exception as e:
                        # Record the error
                        error_result = TestResult(
                            endpoint="unknown",
                            method="unknown",
                            status_code=0,
                            response_time_ms=0,
                            timestamp=datetime.utcnow(),
                            success=False,
                            error_message=str(e)
                        )
                        self.results.append(error_result)
                    
                    # Rate limiting
                    if self.config.target_rps > 0:
                        await asyncio.sleep(1.0 / self.config.target_rps)
                    
            finally:
                self.active_sessions -= 1
    
    def _calculate_stats(self, start_time: float) -> LoadTestStats:
        """Calculate aggregated statistics from test results"""
        if not self.results:
            return LoadTestStats()
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        response_times = [r.response_time_ms for r in successful_results]
        
        total_duration = time.time() - start_time
        
        stats = LoadTestStats(
            total_requests=len(self.results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results)
        )
        
        if response_times:
            stats.avg_response_time_ms = statistics.mean(response_times)
            stats.min_response_time_ms = min(response_times)
            stats.max_response_time_ms = max(response_times)
            
            sorted_times = sorted(response_times)
            if len(sorted_times) >= 20:  # Need enough samples for percentiles
                stats.p95_response_time_ms = sorted_times[int(len(sorted_times) * 0.95)]
                stats.p99_response_time_ms = sorted_times[int(len(sorted_times) * 0.99)]
        
        stats.requests_per_second = stats.total_requests / total_duration if total_duration > 0 else 0
        stats.error_rate = (stats.failed_requests / stats.total_requests * 100) if stats.total_requests > 0 else 0
        
        # Count errors by type
        for result in failed_results:
            error_type = f"HTTP_{result.status_code}" if result.status_code > 0 else "Connection_Error"
            stats.errors_by_type[error_type] = stats.errors_by_type.get(error_type, 0) + 1
        
        return stats
    
    def _display_results(self, scenario_name: str, stats: LoadTestStats):
        """Display test results"""
        print(f"Load Test Results: {scenario_name}")
        print("=" * 60)
        print(f"Total Requests:       {stats.total_requests}")
        print(f"Successful Requests:  {stats.successful_requests}")
        print(f"Failed Requests:      {stats.failed_requests}")
        print(f"Success Rate:         {100 - stats.error_rate:.2f}%")
        print(f"Requests/Second:      {stats.requests_per_second:.2f}")
        print()
        
        if stats.avg_response_time_ms > 0:
            print("Response Times (ms):")
            print(f"  Average:            {stats.avg_response_time_ms:.2f}")
            print(f"  Minimum:            {stats.min_response_time_ms:.2f}")
            print(f"  Maximum:            {stats.max_response_time_ms:.2f}")
            print(f"  95th Percentile:    {stats.p95_response_time_ms:.2f}")
            print(f"  99th Percentile:    {stats.p99_response_time_ms:.2f}")
            print()
        
        if stats.errors_by_type:
            print("Errors by Type:")
            for error_type, count in stats.errors_by_type.items():
                print(f"  {error_type}: {count}")
            print()

# Test Functions for different services
class SADPLoadTests:
    """Load test functions for SADP services"""
    
    def __init__(self):
        self.base_urls = {
            "auth": "https://sadp-auth-service-prod-xonau6hybq-uc.a.run.app",
            "phi_protection": "https://sadp-phi-protection-prod-xonau6hybq-uc.a.run.app",
            "audit": "https://sadp-audit-service-prod-xonau6hybq-uc.a.run.app",
            "prompt_optimization": "https://sadp-prompt-optimization-355881591332.us-central1.run.app"
        }
    
    async def test_health_endpoints(self, user_id: str) -> TestResult:
        """Test health endpoints of all services"""
        service_name = random.choice(list(self.base_urls.keys()))
        url = f"{self.base_urls[service_name]}/health"
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    await response.read()  # Consume response
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    return TestResult(
                        endpoint=f"/{service_name}/health",
                        method="GET",
                        status_code=response.status,
                        response_time_ms=response_time_ms,
                        timestamp=datetime.utcnow(),
                        success=response.status == 200
                    )
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return TestResult(
                endpoint=f"/{service_name}/health",
                method="GET",
                status_code=0,
                response_time_ms=response_time_ms,
                timestamp=datetime.utcnow(),
                success=False,
                error_message=str(e)
            )
    
    async def test_phi_detection(self, user_id: str) -> TestResult:
        """Test PHI detection functionality"""
        url = f"{self.base_urls['phi_protection']}/phi/detect"
        
        test_texts = [
            "Patient John Doe SSN 123-45-6789",
            "Contact info: john.doe@email.com, Phone: 555-123-4567",
            "Medical Record Number: MRN123456789",
            "Date of birth: 01/15/1980"
        ]
        
        data = {"text": random.choice(test_texts)}
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    await response.read()
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    return TestResult(
                        endpoint="/phi/detect",
                        method="POST",
                        status_code=response.status,
                        response_time_ms=response_time_ms,
                        timestamp=datetime.utcnow(),
                        success=response.status == 200
                    )
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return TestResult(
                endpoint="/phi/detect",
                method="POST",
                status_code=0,
                response_time_ms=response_time_ms,
                timestamp=datetime.utcnow(),
                success=False,
                error_message=str(e)
            )
    
    async def test_audit_logging(self, user_id: str) -> TestResult:
        """Test audit logging functionality"""
        url = f"{self.base_urls['audit']}/audit/log"
        
        data = {
            "event_type": "prompt_execution",
            "user_id": user_id,
            "tenant_id": "load_test_tenant",
            "action": "load_test",
            "outcome": "success"
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    await response.read()
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    return TestResult(
                        endpoint="/audit/log",
                        method="POST",
                        status_code=response.status,
                        response_time_ms=response_time_ms,
                        timestamp=datetime.utcnow(),
                        success=response.status in [200, 201]
                    )
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return TestResult(
                endpoint="/audit/log",
                method="POST",
                status_code=0,
                response_time_ms=response_time_ms,
                timestamp=datetime.utcnow(),
                success=False,
                error_message=str(e)
            )

async def main():
    """Main load testing function"""
    parser = argparse.ArgumentParser(description="SADP Load Testing Framework")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--rps", type=float, default=5.0, help="Target requests per second")
    parser.add_argument("--scenario", choices=["health", "phi", "audit", "mixed"], 
                       default="mixed", help="Test scenario to run")
    
    args = parser.parse_args()
    
    config = LoadTestConfig(
        concurrent_users=args.users,
        duration_seconds=args.duration,
        target_rps=args.rps,
        ramp_up_seconds=10
    )
    
    tester = LoadTester(config)
    tests = SADPLoadTests()
    
    if args.scenario == "health":
        test_functions = [tests.test_health_endpoints]
    elif args.scenario == "phi":
        test_functions = [tests.test_phi_detection]
    elif args.scenario == "audit":
        test_functions = [tests.test_audit_logging]
    else:  # mixed
        test_functions = [
            tests.test_health_endpoints,
            tests.test_phi_detection,
            tests.test_audit_logging
        ]
    
    stats = await tester.run_test_scenario(f"SADP {args.scenario.title()} Load Test", test_functions)
    
    # Performance thresholds
    print("Performance Analysis:")
    print("-" * 30)
    
    if stats.error_rate < 1.0:
        print("PASS Error rate within acceptable limits (<1%)")
    else:
        print(f"FAIL Error rate too high: {stats.error_rate:.2f}%")
    
    if stats.p95_response_time_ms < 1000:
        print("PASS P95 response time within target (<1s)")
    else:
        print(f"FAIL P95 response time too slow: {stats.p95_response_time_ms:.2f}ms")
    
    if stats.requests_per_second >= config.target_rps * 0.8:
        print("PASS Throughput within target range")
    else:
        print(f"FAIL Throughput below target: {stats.requests_per_second:.2f} RPS")

if __name__ == "__main__":
    asyncio.run(main())