#!/usr/bin/env python3
"""
SADP Deployment Verification Script
Comprehensive verification of SADP platform deployment
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional
import aiohttp
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Results of a verification check"""
    service: str
    endpoint: str
    status: str  # "pass", "fail", "warning"
    response_time: float
    details: str
    timestamp: datetime


class SADPVerifier:
    """Main verification class for SADP deployment"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.results: List[VerificationResult] = []
        
    async def verify_all(self) -> Dict[str, any]:
        """Run all verification checks"""
        logger.info("🔍 Starting comprehensive SADP deployment verification")
        logger.info(f"Target URL: {self.base_url}")
        
        # Core service checks
        await self.verify_health_endpoint()
        await self.verify_agents_endpoint()
        await self.verify_marketplace_endpoints()
        await self.verify_metrics_endpoint()
        await self.verify_poml_endpoints()
        
        # Load and performance tests
        await self.verify_agent_execution()
        await self.verify_marketplace_functionality()
        await self.verify_poml_ab_testing()
        
        # Generate summary
        return self.generate_summary()
    
    async def verify_health_endpoint(self):
        """Verify basic health endpoint"""
        endpoint = f"{self.base_url}/health"
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(endpoint) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        if data.get('status') == 'healthy':
                            self.results.append(VerificationResult(
                                service="core",
                                endpoint="/health",
                                status="pass",
                                response_time=response_time,
                                details=f"Health check passed. Uptime: {data.get('uptime', 'unknown')}",
                                timestamp=datetime.now()
                            ))
                        else:
                            self.results.append(VerificationResult(
                                service="core",
                                endpoint="/health",
                                status="warning",
                                response_time=response_time,
                                details=f"Health endpoint accessible but status: {data.get('status', 'unknown')}",
                                timestamp=datetime.now()
                            ))
                    else:
                        self.results.append(VerificationResult(
                            service="core",
                            endpoint="/health",
                            status="fail",
                            response_time=response_time,
                            details=f"HTTP {response.status}: Health endpoint returned error",
                            timestamp=datetime.now()
                        ))
        except Exception as e:
            self.results.append(VerificationResult(
                service="core",
                endpoint="/health",
                status="fail",
                response_time=time.time() - start_time,
                details=f"Failed to reach health endpoint: {str(e)}",
                timestamp=datetime.now()
            ))
    
    async def verify_agents_endpoint(self):
        """Verify agents listing endpoint"""
        endpoint = f"{self.base_url}/agents/supported"
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(endpoint) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        agent_count = len(data.get('agents', []))
                        
                        if agent_count > 0:
                            self.results.append(VerificationResult(
                                service="agents",
                                endpoint="/agents/supported",
                                status="pass",
                                response_time=response_time,
                                details=f"Found {agent_count} supported agents",
                                timestamp=datetime.now()
                            ))
                        else:
                            self.results.append(VerificationResult(
                                service="agents",
                                endpoint="/agents/supported",
                                status="warning",
                                response_time=response_time,
                                details="No agents found - marketplace may not be initialized",
                                timestamp=datetime.now()
                            ))
                    else:
                        self.results.append(VerificationResult(
                            service="agents",
                            endpoint="/agents/supported",
                            status="fail",
                            response_time=response_time,
                            details=f"HTTP {response.status}: Agents endpoint error",
                            timestamp=datetime.now()
                        ))
        except Exception as e:
            self.results.append(VerificationResult(
                service="agents",
                endpoint="/agents/supported",
                status="fail",
                response_time=time.time() - start_time,
                details=f"Failed to reach agents endpoint: {str(e)}",
                timestamp=datetime.now()
            ))
    
    async def verify_marketplace_endpoints(self):
        """Verify marketplace endpoints"""
        endpoints = [
            "/agents/marketplace/search",
            "/agents/marketplace/categories",
            "/agents/marketplace/featured"
        ]
        
        for endpoint_path in endpoints:
            endpoint = f"{self.base_url}{endpoint_path}"
            start_time = time.time()
            
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.get(endpoint) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            data = await response.json()
                            
                            if endpoint_path == "/agents/marketplace/search":
                                agent_count = len(data.get('agents', []))
                                details = f"Marketplace search working, {agent_count} agents available"
                            elif endpoint_path == "/agents/marketplace/categories":
                                cat_count = len(data.get('categories', []))
                                details = f"Found {cat_count} agent categories"
                            else:  # featured
                                featured_count = len(data.get('featured', []))
                                details = f"Found {featured_count} featured agents"
                            
                            self.results.append(VerificationResult(
                                service="marketplace",
                                endpoint=endpoint_path,
                                status="pass",
                                response_time=response_time,
                                details=details,
                                timestamp=datetime.now()
                            ))
                        else:
                            self.results.append(VerificationResult(
                                service="marketplace",
                                endpoint=endpoint_path,
                                status="fail",
                                response_time=response_time,
                                details=f"HTTP {response.status}: Marketplace endpoint error",
                                timestamp=datetime.now()
                            ))
            except Exception as e:
                self.results.append(VerificationResult(
                    service="marketplace",
                    endpoint=endpoint_path,
                    status="fail",
                    response_time=time.time() - start_time,
                    details=f"Failed to reach marketplace endpoint: {str(e)}",
                    timestamp=datetime.now()
                ))
    
    async def verify_metrics_endpoint(self):
        """Verify metrics endpoint"""
        endpoint = f"{self.base_url}/metrics"
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(endpoint) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        # Metrics endpoint should return text/plain
                        content = await response.text()
                        if "sadp_" in content:  # Check for SADP-specific metrics
                            self.results.append(VerificationResult(
                                service="monitoring",
                                endpoint="/metrics",
                                status="pass",
                                response_time=response_time,
                                details="Prometheus metrics endpoint working with SADP metrics",
                                timestamp=datetime.now()
                            ))
                        else:
                            self.results.append(VerificationResult(
                                service="monitoring",
                                endpoint="/metrics",
                                status="warning",
                                response_time=response_time,
                                details="Metrics endpoint accessible but no SADP-specific metrics found",
                                timestamp=datetime.now()
                            ))
                    else:
                        self.results.append(VerificationResult(
                            service="monitoring",
                            endpoint="/metrics",
                            status="fail",
                            response_time=response_time,
                            details=f"HTTP {response.status}: Metrics endpoint error",
                            timestamp=datetime.now()
                        ))
        except Exception as e:
            self.results.append(VerificationResult(
                service="monitoring",
                endpoint="/metrics",
                status="fail",
                response_time=time.time() - start_time,
                details=f"Failed to reach metrics endpoint: {str(e)}",
                timestamp=datetime.now()
            ))
    
    async def verify_poml_endpoints(self):
        """Verify POML management endpoints"""
        endpoints = [
            "/poml/templates",
            "/poml/versions"
        ]
        
        for endpoint_path in endpoints:
            endpoint = f"{self.base_url}{endpoint_path}"
            start_time = time.time()
            
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.get(endpoint) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            data = await response.json()
                            
                            if endpoint_path == "/poml/templates":
                                template_count = len(data.get('templates', []))
                                details = f"Found {template_count} POML templates"
                            else:  # versions
                                version_count = len(data.get('versions', []))
                                details = f"Found {version_count} template versions"
                            
                            self.results.append(VerificationResult(
                                service="poml",
                                endpoint=endpoint_path,
                                status="pass",
                                response_time=response_time,
                                details=details,
                                timestamp=datetime.now()
                            ))
                        else:
                            self.results.append(VerificationResult(
                                service="poml",
                                endpoint=endpoint_path,
                                status="fail",
                                response_time=response_time,
                                details=f"HTTP {response.status}: POML endpoint error",
                                timestamp=datetime.now()
                            ))
            except Exception as e:
                self.results.append(VerificationResult(
                    service="poml",
                    endpoint=endpoint_path,
                    status="fail",
                    response_time=time.time() - start_time,
                    details=f"Failed to reach POML endpoint: {str(e)}",
                    timestamp=datetime.now()
                ))
    
    async def verify_agent_execution(self):
        """Verify actual agent execution functionality"""
        endpoint = f"{self.base_url}/agents/document/execute"
        start_time = time.time()
        
        # Test payload
        test_payload = {
            "document": {
                "content": "Patient: John Doe, Age: 45, BP: 140/90, Temp: 98.6F",
                "type": "clinical_note"
            },
            "task": "extract_vitals"
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(endpoint, json=test_payload) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('status') == 'completed' and data.get('results'):
                            self.results.append(VerificationResult(
                                service="execution",
                                endpoint="/agents/document/execute",
                                status="pass",
                                response_time=response_time,
                                details=f"Agent execution successful. Confidence: {data.get('confidence', 'unknown')}",
                                timestamp=datetime.now()
                            ))
                        else:
                            self.results.append(VerificationResult(
                                service="execution",
                                endpoint="/agents/document/execute",
                                status="warning",
                                response_time=response_time,
                                details=f"Agent execution completed but with issues: {data.get('status')}",
                                timestamp=datetime.now()
                            ))
                    else:
                        self.results.append(VerificationResult(
                            service="execution",
                            endpoint="/agents/document/execute",
                            status="fail",
                            response_time=response_time,
                            details=f"HTTP {response.status}: Agent execution failed",
                            timestamp=datetime.now()
                        ))
        except Exception as e:
            self.results.append(VerificationResult(
                service="execution",
                endpoint="/agents/document/execute",
                status="fail",
                response_time=time.time() - start_time,
                details=f"Failed agent execution test: {str(e)}",
                timestamp=datetime.now()
            ))
    
    async def verify_marketplace_functionality(self):
        """Verify marketplace loading and registration"""
        # Test marketplace agent loading
        endpoint = f"{self.base_url}/agents/marketplace/load"
        start_time = time.time()
        
        test_payload = {
            "agent_id": "document-processor",
            "version": "1.0.0"
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(endpoint, json=test_payload) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('status') == 'loaded':
                            self.results.append(VerificationResult(
                                service="marketplace",
                                endpoint="/agents/marketplace/load",
                                status="pass",
                                response_time=response_time,
                                details="Dynamic agent loading successful",
                                timestamp=datetime.now()
                            ))
                        else:
                            self.results.append(VerificationResult(
                                service="marketplace",
                                endpoint="/agents/marketplace/load",
                                status="warning",
                                response_time=response_time,
                                details=f"Agent loading completed with status: {data.get('status')}",
                                timestamp=datetime.now()
                            ))
                    else:
                        self.results.append(VerificationResult(
                            service="marketplace",
                            endpoint="/agents/marketplace/load",
                            status="fail",
                            response_time=response_time,
                            details=f"HTTP {response.status}: Agent loading failed",
                            timestamp=datetime.now()
                        ))
        except Exception as e:
            self.results.append(VerificationResult(
                service="marketplace",
                endpoint="/agents/marketplace/load",
                status="fail",
                response_time=time.time() - start_time,
                details=f"Failed marketplace loading test: {str(e)}",
                timestamp=datetime.now()
            ))
    
    async def verify_poml_ab_testing(self):
        """Verify POML A/B testing functionality"""
        endpoint = f"{self.base_url}/poml/ab-tests"
        start_time = time.time()
        
        # Test creating an A/B test
        test_payload = {
            "name": "verification_test",
            "description": "Verification test for deployment",
            "control_template": "<prompt><system>Test control</system></prompt>",
            "variant_template": "<prompt><system>Test variant</system></prompt>",
            "sample_size": 10
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(endpoint, json=test_payload) as response:
                    response_time = time.time() - start_time
                    
                    if response.status in [200, 201]:
                        data = await response.json()
                        
                        if data.get('test_id'):
                            self.results.append(VerificationResult(
                                service="poml",
                                endpoint="/poml/ab-tests",
                                status="pass",
                                response_time=response_time,
                                details=f"A/B test creation successful. Test ID: {data.get('test_id')}",
                                timestamp=datetime.now()
                            ))
                        else:
                            self.results.append(VerificationResult(
                                service="poml",
                                endpoint="/poml/ab-tests",
                                status="warning",
                                response_time=response_time,
                                details="A/B test endpoint accessible but response format unexpected",
                                timestamp=datetime.now()
                            ))
                    else:
                        self.results.append(VerificationResult(
                            service="poml",
                            endpoint="/poml/ab-tests",
                            status="fail",
                            response_time=response_time,
                            details=f"HTTP {response.status}: A/B test creation failed",
                            timestamp=datetime.now()
                        ))
        except Exception as e:
            self.results.append(VerificationResult(
                service="poml",
                endpoint="/poml/ab-tests",
                status="fail",
                response_time=time.time() - start_time,
                details=f"Failed POML A/B testing: {str(e)}",
                timestamp=datetime.now()
            ))
    
    def generate_summary(self) -> Dict[str, any]:
        """Generate verification summary"""
        total_checks = len(self.results)
        passed = len([r for r in self.results if r.status == "pass"])
        warnings = len([r for r in self.results if r.status == "warning"])
        failed = len([r for r in self.results if r.status == "fail"])
        
        # Calculate average response time
        avg_response_time = sum(r.response_time for r in self.results) / total_checks if total_checks > 0 else 0
        
        # Group results by service
        services = {}
        for result in self.results:
            if result.service not in services:
                services[result.service] = []
            services[result.service].append(result)
        
        return {
            "summary": {
                "total_checks": total_checks,
                "passed": passed,
                "warnings": warnings,
                "failed": failed,
                "success_rate": (passed / total_checks * 100) if total_checks > 0 else 0,
                "avg_response_time": round(avg_response_time, 3),
                "overall_status": "healthy" if failed == 0 and warnings <= 2 else "degraded" if failed <= 2 else "unhealthy"
            },
            "services": {
                service: {
                    "status": "pass" if all(r.status == "pass" for r in results) else 
                             "warning" if any(r.status == "warning" for r in results) and not any(r.status == "fail" for r in results) else "fail",
                    "checks": len(results),
                    "avg_response_time": round(sum(r.response_time for r in results) / len(results), 3)
                }
                for service, results in services.items()
            },
            "detailed_results": [
                {
                    "service": r.service,
                    "endpoint": r.endpoint,
                    "status": r.status,
                    "response_time": round(r.response_time, 3),
                    "details": r.details,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.results
            ],
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url
        }


def print_summary(summary: Dict[str, any]):
    """Print verification summary to console"""
    print("\n" + "="*60)
    print("🔍 SADP DEPLOYMENT VERIFICATION REPORT")
    print("="*60)
    
    print(f"\n📊 Overall Results:")
    print(f"   Target URL: {summary['base_url']}")
    print(f"   Total Checks: {summary['summary']['total_checks']}")
    print(f"   ✅ Passed: {summary['summary']['passed']}")
    print(f"   ⚠️  Warnings: {summary['summary']['warnings']}")
    print(f"   ❌ Failed: {summary['summary']['failed']}")
    print(f"   📈 Success Rate: {summary['summary']['success_rate']:.1f}%")
    print(f"   ⚡ Avg Response Time: {summary['summary']['avg_response_time']}s")
    print(f"   🏥 Overall Status: {summary['summary']['overall_status'].upper()}")
    
    print(f"\n🔧 Service Status:")
    for service, status in summary['services'].items():
        status_icon = "✅" if status['status'] == "pass" else "⚠️" if status['status'] == "warning" else "❌"
        print(f"   {status_icon} {service.title()}: {status['status']} ({status['checks']} checks, {status['avg_response_time']}s avg)")
    
    print(f"\n📝 Detailed Results:")
    for result in summary['detailed_results']:
        status_icon = "✅" if result['status'] == "pass" else "⚠️" if result['status'] == "warning" else "❌"
        print(f"   {status_icon} [{result['service']}] {result['endpoint']}")
        print(f"      {result['details']} ({result['response_time']}s)")
    
    print(f"\n⏰ Report generated at: {summary['timestamp']}")
    print("="*60)


async def main():
    """Main verification function"""
    if len(sys.argv) < 2:
        print("Usage: python verify_deployment.py <base_url> [output_file]")
        print("Example: python verify_deployment.py https://your-service-url.run.app")
        sys.exit(1)
    
    base_url = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Create verifier and run all checks
    verifier = SADPVerifier(base_url)
    summary = await verifier.verify_all()
    
    # Print results to console
    print_summary(summary)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Exit with appropriate code
    if summary['summary']['overall_status'] == 'unhealthy':
        sys.exit(1)
    elif summary['summary']['overall_status'] == 'degraded':
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())