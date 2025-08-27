"""
Integration Tests for Production SADP System
Tests real AI integration, persistence, authentication, PHI protection, and audit logging
"""

import asyncio
import pytest
import httpx
import json
from datetime import datetime
from typing import Dict, Any

# Test configuration
TEST_CONFIG = {
    "auth_service_url": "http://localhost:8001",
    "phi_protection_url": "http://localhost:8002", 
    "audit_service_url": "http://localhost:8003",
    "prompt_optimization_url": "http://localhost:8004",
    "timeout": 30.0
}

class ProductionSystemTests:
    """Integration tests for the production SADP system"""
    
    def __init__(self):
        self.test_user_email = "test@sadp.ai"
        self.test_password = "TestPassword123!"
        self.access_token = None
        self.api_key = None
        
    async def setup_test_environment(self):
        """Set up test environment"""
        print("🔧 Setting up test environment...")
        
        # Create test user
        await self.create_test_user()
        
        # Authenticate user
        await self.authenticate_user()
        
        # Create API key
        await self.create_api_key()
        
        print("✅ Test environment ready")
    
    async def create_test_user(self):
        """Create a test user through the auth service"""
        
        user_data = {
            "email": self.test_user_email,
            "name": "Test User",
            "password": self.test_password,
            "role": "developer",
            "tenant_id": "test_tenant"
        }
        
        async with httpx.AsyncClient(timeout=TEST_CONFIG["timeout"]) as client:
            try:
                response = await client.post(
                    f"{TEST_CONFIG['auth_service_url']}/auth/users",
                    json=user_data
                )
                
                if response.status_code in [200, 201, 409]:  # 409 = user already exists
                    print("👤 Test user created/exists")
                else:
                    print(f"⚠️ User creation failed: {response.status_code}")
                    
            except Exception as e:
                print(f"⚠️ Could not create test user: {e}")
    
    async def authenticate_user(self):
        """Authenticate test user and get access token"""
        
        auth_data = {
            "email": self.test_user_email,
            "password": self.test_password
        }
        
        async with httpx.AsyncClient(timeout=TEST_CONFIG["timeout"]) as client:
            try:
                response = await client.post(
                    f"{TEST_CONFIG['auth_service_url']}/auth/login",
                    data=auth_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.access_token = result.get("access_token")
                    print("🔑 User authenticated successfully")
                else:
                    print(f"⚠️ Authentication failed: {response.status_code}")
                    
            except Exception as e:
                print(f"⚠️ Could not authenticate user: {e}")
    
    async def create_api_key(self):
        """Create API key for testing"""
        
        if not self.access_token:
            print("⚠️ No access token available")
            return
        
        headers = {"Authorization": f"Bearer {self.access_token}"}
        key_data = {
            "name": "Integration Test Key",
            "permissions": ["read", "write", "execute"],
            "expires_in_days": 30
        }
        
        async with httpx.AsyncClient(timeout=TEST_CONFIG["timeout"]) as client:
            try:
                response = await client.post(
                    f"{TEST_CONFIG['auth_service_url']}/auth/api-keys",
                    json=key_data,
                    headers=headers
                )
                
                if response.status_code in [200, 201]:
                    result = response.json()
                    self.api_key = result.get("api_key")
                    print("🔐 API key created successfully")
                else:
                    print(f"⚠️ API key creation failed: {response.status_code}")
                    
            except Exception as e:
                print(f"⚠️ Could not create API key: {e}")
    
    async def test_health_endpoints(self):
        """Test health endpoints for all services"""
        
        print("\n🏥 Testing health endpoints...")
        
        services = [
            ("Auth Service", TEST_CONFIG["auth_service_url"]),
            ("PHI Protection", TEST_CONFIG["phi_protection_url"]),
            ("Audit Service", TEST_CONFIG["audit_service_url"]),
            ("Prompt Optimization", TEST_CONFIG["prompt_optimization_url"])
        ]
        
        results = {}
        
        async with httpx.AsyncClient(timeout=TEST_CONFIG["timeout"]) as client:
            for service_name, url in services:
                try:
                    response = await client.get(f"{url}/health")
                    
                    if response.status_code == 200:
                        health_data = response.json()
                        status = health_data.get("status", "unknown")
                        results[service_name] = {"status": status, "healthy": status == "healthy"}
                        
                        if status == "healthy":
                            print(f"✅ {service_name}: Healthy")
                        else:
                            print(f"⚠️ {service_name}: {status}")
                    else:
                        results[service_name] = {"status": "error", "healthy": False}
                        print(f"❌ {service_name}: HTTP {response.status_code}")
                        
                except Exception as e:
                    results[service_name] = {"status": "unreachable", "healthy": False}
                    print(f"❌ {service_name}: Unreachable ({e})")
        
        return results
    
    async def test_phi_detection(self):
        """Test PHI detection service"""
        
        print("\n🔍 Testing PHI detection...")
        
        test_text = """
        Patient John Doe (SSN: 123-45-6789) was born on 01/15/1980.
        His phone number is (555) 123-4567 and email is john.doe@email.com.
        Medical Record Number: MRN12345678
        """
        
        request_data = {
            "text": test_text,
            "detection_types": ["ssn", "mrn", "dob", "phone", "email"],
            "include_context": True
        }
        
        async with httpx.AsyncClient(timeout=TEST_CONFIG["timeout"]) as client:
            try:
                response = await client.post(
                    f"{TEST_CONFIG['phi_protection_url']}/phi/detect",
                    json=request_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    detections = result.get("detections", [])
                    phi_found = result.get("phi_found", False)
                    
                    print(f"✅ PHI Detection: Found {len(detections)} PHI elements")
                    print(f"   PHI types detected: {[d['phi_type'] for d in detections]}")
                    
                    return {
                        "success": True,
                        "detections_count": len(detections),
                        "phi_found": phi_found
                    }
                else:
                    print(f"❌ PHI Detection failed: HTTP {response.status_code}")
                    return {"success": False, "error": response.status_code}
                    
            except Exception as e:
                print(f"❌ PHI Detection error: {e}")
                return {"success": False, "error": str(e)}
    
    async def test_phi_sanitization(self):
        """Test PHI sanitization service"""
        
        print("\n🧹 Testing PHI sanitization...")
        
        test_text = "Patient SSN 123-45-6789 needs follow-up."
        
        request_data = {
            "text": test_text,
            "sanitization_level": "mask",
            "preserve_format": True,
            "audit_log": True,
            "user_id": "test_user"
        }
        
        async with httpx.AsyncClient(timeout=TEST_CONFIG["timeout"]) as client:
            try:
                response = await client.post(
                    f"{TEST_CONFIG['phi_protection_url']}/phi/sanitize",
                    json=request_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    sanitized_text = result.get("sanitized_text", "")
                    phi_removed = result.get("phi_removed", [])
                    
                    print(f"✅ PHI Sanitization successful")
                    print(f"   Original: {test_text}")
                    print(f"   Sanitized: {sanitized_text}")
                    print(f"   PHI removed: {phi_removed}")
                    
                    return {
                        "success": True,
                        "sanitized": sanitized_text,
                        "phi_removed": phi_removed
                    }
                else:
                    print(f"❌ PHI Sanitization failed: HTTP {response.status_code}")
                    return {"success": False, "error": response.status_code}
                    
            except Exception as e:
                print(f"❌ PHI Sanitization error: {e}")
                return {"success": False, "error": str(e)}
    
    async def test_prompt_optimization(self):
        """Test production prompt optimization"""
        
        print("\n🚀 Testing prompt optimization...")
        
        test_prompt = "Analyze patient symptoms and provide recommendations."
        
        request_data = {
            "prompt": test_prompt,
            "strategy": "automedprompt",
            "objective": "accuracy",
            "user_id": "test_user",
            "tenant_id": "test_tenant"
        }
        
        async with httpx.AsyncClient(timeout=TEST_CONFIG["timeout"]) as client:
            try:
                response = await client.post(
                    f"{TEST_CONFIG['prompt_optimization_url']}/optimize/automedprompt",
                    json=request_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    optimized_prompt = result.get("optimized_prompt", "")
                    improvement = result.get("performance_improvement", 0.0)
                    
                    print(f"✅ Prompt optimization successful")
                    print(f"   Improvement: {improvement:.2%}")
                    print(f"   Job ID: {result.get('job_id', 'N/A')}")
                    
                    return {
                        "success": True,
                        "job_id": result.get("job_id"),
                        "improvement": improvement,
                        "optimized_prompt": optimized_prompt
                    }
                else:
                    print(f"❌ Prompt optimization failed: HTTP {response.status_code}")
                    return {"success": False, "error": response.status_code}
                    
            except Exception as e:
                print(f"❌ Prompt optimization error: {e}")
                return {"success": False, "error": str(e)}
    
    async def test_audit_logging(self):
        """Test audit logging service"""
        
        print("\n📝 Testing audit logging...")
        
        audit_data = {
            "event_type": "agent_execution",
            "severity": "info",
            "user_id": "test_user",
            "tenant_id": "test_tenant",
            "service_name": "integration_test",
            "agent_type": "test_agent",
            "execution_time_ms": 1500,
            "status_code": 200,
            "metadata": {"test": True}
        }
        
        async with httpx.AsyncClient(timeout=TEST_CONFIG["timeout"]) as client:
            try:
                response = await client.post(
                    f"{TEST_CONFIG['audit_service_url']}/audit/log",
                    json=audit_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    audit_id = result.get("audit_id")
                    
                    print(f"✅ Audit logging successful")
                    print(f"   Audit ID: {audit_id}")
                    
                    # Test audit query
                    query_data = {
                        "user_id": "test_user",
                        "limit": 10
                    }
                    
                    query_response = await client.post(
                        f"{TEST_CONFIG['audit_service_url']}/audit/query",
                        json=query_data
                    )
                    
                    if query_response.status_code == 200:
                        query_result = query_response.json()
                        count = query_result.get("count", 0)
                        print(f"✅ Audit query successful: {count} records found")
                    
                    return {
                        "success": True,
                        "audit_id": audit_id,
                        "query_count": count
                    }
                else:
                    print(f"❌ Audit logging failed: HTTP {response.status_code}")
                    return {"success": False, "error": response.status_code}
                    
            except Exception as e:
                print(f"❌ Audit logging error: {e}")
                return {"success": False, "error": str(e)}
    
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        
        print("\n🔄 Testing end-to-end workflow...")
        
        # Step 1: PHI detection and sanitization
        input_text = "Patient John Smith (SSN: 123-45-6789) has diabetes. Recommend treatment."
        
        # Sanitize PHI
        sanitize_request = {
            "text": input_text,
            "sanitization_level": "mask",
            "audit_log": True,
            "user_id": "test_user"
        }
        
        async with httpx.AsyncClient(timeout=TEST_CONFIG["timeout"]) as client:
            # Step 1: Sanitize input
            sanitize_response = await client.post(
                f"{TEST_CONFIG['phi_protection_url']}/phi/sanitize",
                json=sanitize_request
            )
            
            if sanitize_response.status_code != 200:
                print("❌ E2E: PHI sanitization failed")
                return {"success": False, "step": "phi_sanitization"}
            
            sanitized_text = sanitize_response.json().get("sanitized_text", "")
            print(f"✅ E2E Step 1: PHI sanitized")
            
            # Step 2: Optimize prompt
            optimize_request = {
                "prompt": f"You are a medical AI. {sanitized_text}",
                "strategy": "automedprompt",
                "user_id": "test_user",
                "tenant_id": "test_tenant"
            }
            
            optimize_response = await client.post(
                f"{TEST_CONFIG['prompt_optimization_url']}/optimize/automedprompt",
                json=optimize_request
            )
            
            if optimize_response.status_code != 200:
                print("❌ E2E: Prompt optimization failed")
                return {"success": False, "step": "prompt_optimization"}
            
            optimization_result = optimize_response.json()
            print(f"✅ E2E Step 2: Prompt optimized")
            
            # Step 3: Log the complete workflow
            audit_request = {
                "event_type": "poml_execution",
                "severity": "info",
                "user_id": "test_user",
                "tenant_id": "test_tenant",
                "template_id": "e2e_test_template",
                "phi_detected": True,
                "phi_types": ["ssn"],
                "execution_time_ms": 2500,
                "metadata": {
                    "workflow": "e2e_test",
                    "phi_sanitized": True,
                    "prompt_optimized": True
                }
            }
            
            audit_response = await client.post(
                f"{TEST_CONFIG['audit_service_url']}/audit/log",
                json=audit_request
            )
            
            if audit_response.status_code != 200:
                print("❌ E2E: Audit logging failed")
                return {"success": False, "step": "audit_logging"}
            
            print("✅ E2E Step 3: Workflow audited")
            
            return {
                "success": True,
                "sanitized_text": sanitized_text,
                "optimization_improvement": optimization_result.get("performance_improvement", 0),
                "audit_id": audit_response.json().get("audit_id")
            }
    
    async def run_all_tests(self):
        """Run all integration tests"""
        
        print("🧪 Starting SADP Production System Integration Tests")
        print("=" * 60)
        
        # Setup
        await self.setup_test_environment()
        
        # Track results
        test_results = {}
        
        # Run tests
        test_results["health"] = await self.test_health_endpoints()
        test_results["phi_detection"] = await self.test_phi_detection()
        test_results["phi_sanitization"] = await self.test_phi_sanitization()
        test_results["prompt_optimization"] = await self.test_prompt_optimization()
        test_results["audit_logging"] = await self.test_audit_logging()
        test_results["end_to_end"] = await self.test_end_to_end_workflow()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        
        total_tests = len(test_results)
        passed_tests = 0
        
        for test_name, result in test_results.items():
            if isinstance(result, dict):
                success = result.get("success", False)
                if test_name == "health":
                    # Special handling for health check
                    success = all(service.get("healthy", False) for service in result.values())
                
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"{test_name.replace('_', ' ').title()}: {status}")
                
                if success:
                    passed_tests += 1
            else:
                print(f"{test_name.replace('_', ' ').title()}: ❓ UNKNOWN")
        
        print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! Production system is ready.")
        else:
            print("⚠️ Some tests failed. Review configuration and services.")
        
        return test_results

async def main():
    """Main test runner"""
    
    tests = ProductionSystemTests()
    results = await tests.run_all_tests()
    
    # Save results to file
    with open("integration_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to integration_test_results.json")

if __name__ == "__main__":
    asyncio.run(main())