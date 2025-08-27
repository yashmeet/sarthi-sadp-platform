"""
Circuit Breaker Pattern Implementation for SADP Services
Provides fault tolerance and resilience for external service calls
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Number of failures before opening
    recovery_timeout: int = 60          # Seconds before attempting recovery
    expected_exception: type = Exception  # Exception type to catch
    success_threshold: int = 3          # Successes needed to close from half-open

@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None

class CircuitBreaker:
    """
    Circuit breaker implementation for external service calls
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.stats = CircuitBreakerStats()
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        """
        self.stats.total_calls += 1
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker {self.name} moving to HALF_OPEN")
                self.stats.state_changes += 1
            else:
                logger.warning(f"Circuit breaker {self.name} is OPEN, failing fast")
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
        
        try:
            # Execute the function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._on_success()
            return result
            
        except self.config.expected_exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        """Handle successful call"""
        self.stats.successful_calls += 1
        self.stats.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} CLOSED after successful recovery")
                self.stats.state_changes += 1
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset failure count on success
    
    async def _on_failure(self):
        """Handle failed call"""
        self.stats.failed_calls += 1
        self.stats.last_failure_time = time.time()
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Go back to open on any failure in half-open state
            self.state = CircuitState.OPEN
            self.success_count = 0
            logger.warning(f"Circuit breaker {self.name} back to OPEN after failure")
            self.stats.state_changes += 1
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(f"Circuit breaker {self.name} OPENED after {self.failure_count} failures")
                self.stats.state_changes += 1
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.last_failure_time is None:
            return True
        
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        success_rate = (self.stats.successful_calls / self.stats.total_calls * 100) if self.stats.total_calls > 0 else 0
        
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "stats": {
                "total_calls": self.stats.total_calls,
                "successful_calls": self.stats.successful_calls,
                "failed_calls": self.stats.failed_calls,
                "success_rate": round(success_rate, 2),
                "state_changes": self.stats.state_changes,
                "last_failure_time": self.stats.last_failure_time,
                "last_success_time": self.stats.last_success_time
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold
            }
        }

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different services
    """
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all circuit breakers"""
        return {
            "circuit_breakers": {name: breaker.get_stats() for name, breaker in self.breakers.items()},
            "total_breakers": len(self.breakers),
            "open_breakers": len([b for b in self.breakers.values() if b.state == CircuitState.OPEN]),
            "half_open_breakers": len([b for b in self.breakers.values() if b.state == CircuitState.HALF_OPEN])
        }

# Global circuit breaker manager
circuit_manager = CircuitBreakerManager()

def with_circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """
    Decorator to add circuit breaker protection to a function
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            breaker = circuit_manager.get_breaker(name, config)
            return await breaker.call(func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            breaker = circuit_manager.get_breaker(name, config)
            return asyncio.run(breaker.call(func, *args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator