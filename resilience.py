# Enhanced Error Handling and Robustness

import logging
import traceback
import asyncio
from functools import wraps
from typing import Callable, Any, Optional, Type
from enum import Enum
import time
from dataclasses import dataclass
import sys

# 1. Custom Exception Hierarchy
class JarvisException(Exception):
    """Base exception for JARVIS"""
    pass

class AIModelException(JarvisException):
    """AI model related errors"""
    pass

class VoiceProcessingException(JarvisException):
    """Voice processing errors"""
    pass

class AutomationException(JarvisException):
    """Automation related errors"""
    pass

class SecurityException(JarvisException):
    """Security related errors"""
    pass

class ConfigurationException(JarvisException):
    """Configuration errors"""
    pass

# 2. Circuit Breaker Pattern
class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    timeout: int = 60  # seconds
    expected_exception: Type[Exception] = Exception

class CircuitBreaker:
    """Circuit breaker for external service calls"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise JarvisException("Circuit breaker is OPEN - service unavailable")
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except self.config.expected_exception as e:
                self._on_failure()
                raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and 
            time.time() - self.last_failure_time >= self.config.timeout
        )
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN

# 3. Retry Decorator with Exponential Backoff
def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """Retry decorator with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries")
                        raise e
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator

# 4. Enhanced Logging System
class ContextualLogger:
    """Logger with contextual information"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._context = {}
    
    def set_context(self, **kwargs):
        """Set contextual information"""
        self._context.update(kwargs)
    
    def clear_context(self):
        """Clear contextual information"""
        self._context.clear()
    
    def _format_message(self, message: str) -> str:
        """Format message with context"""
        if self._context:
            context_str = " | ".join(f"{k}={v}" for k, v in self._context.items())
            return f"[{context_str}] {message}"
        return message
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(self._format_message(message), **kwargs)
    
    def info(self, message: str, **kwargs):
        self.logger.info(self._format_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(self._format_message(message), **kwargs)
    
    def error(self, message: str, exc_info=None, **kwargs):
        self.logger.error(self._format_message(message), exc_info=exc_info, **kwargs)

# 5. Health Check System
class HealthCheck:
    """System health monitoring"""
    
    def __init__(self):
        self.checks = {}
        self.logger = ContextualLogger(__name__)
    
    def register_check(self, name: str, check_func: Callable, timeout: int = 5):
        """Register a health check function"""
        self.checks[name] = {
            'func': check_func,
            'timeout': timeout,
            'last_status': None,
            'last_check': None
        }
    
    async def run_checks(self) -> dict:
        """Run all health checks"""
        results = {}
        
        for name, check in self.checks.items():
            try:
                start_time = time.time()
                
                # Run check with timeout
                status = await asyncio.wait_for(
                    check['func'](),
                    timeout=check['timeout']
                )
                
                duration = time.time() - start_time
                
                results[name] = {
                    'status': 'healthy',
                    'response_time': duration,
                    'details': status
                }
                
                check['last_status'] = 'healthy'
                check['last_check'] = time.time()
                
            except asyncio.TimeoutError:
                results[name] = {
                    'status': 'unhealthy',
                    'error': f'Check timed out after {check["timeout"]}s'
                }
                check['last_status'] = 'timeout'
                
            except Exception as e:
                results[name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                check['last_status'] = 'error'
                
                self.logger.error(f"Health check {name} failed: {e}")
        
        return results

# 6. Graceful Degradation
class ServiceRegistry:
    """Registry for services with fallback capabilities"""
    
    def __init__(self):
        self.services = {}
        self.fallbacks = {}
        self.logger = ContextualLogger(__name__)
    
    def register_service(self, name: str, service: Any, fallback: Optional[Any] = None):
        """Register a service with optional fallback"""
        self.services[name] = service
        if fallback:
            self.fallbacks[name] = fallback
    
    async def call_service(self, name: str, method: str, *args, **kwargs):
        """Call service method with automatic fallback"""
        if name not in self.services:
            raise JarvisException(f"Service {name} not registered")
        
        try:
            service = self.services[name]
            method_func = getattr(service, method)
            return await method_func(*args, **kwargs)
            
        except Exception as e:
            self.logger.warning(f"Primary service {name}.{method} failed: {e}")
            
            # Try fallback
            if name in self.fallbacks:
                try:
                    fallback_service = self.fallbacks[name]
                    fallback_method = getattr(fallback_service, method)
                    self.logger.info(f"Using fallback for {name}.{method}")
                    return await fallback_method(*args, **kwargs)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback also failed: {fallback_error}")
            
            raise e

# 7. Safe Command Execution
class SafeCommandExecutor:
    """Execute commands safely with validation"""
    
    def __init__(self):
        self.allowed_commands = {
            'file_operations': ['ls', 'dir', 'pwd', 'cat', 'head', 'tail'],
            'system_info': ['ps', 'top', 'df', 'free', 'uname'],
            'network': ['ping', 'wget', 'curl']
        }
        
        self.dangerous_patterns = [
            r'rm\s+-rf',
            r'del\s+/s',
            r'format\s+',
            r'shutdown',
            r'reboot',
            r'halt',
            r'init\s+0',
            r'dd\s+if=',
            r':\(\)\{\s*:\|\:&\s*\}\s*;:',  # Fork bomb
            r'sudo\s+',
            r'su\s+',
            r'chmod\s+777',
            r'chown\s+',
            r'passwd\s+',
        ]
        
        self.logger = ContextualLogger(__name__)
    
    def validate_command(self, command: str) -> bool:
        """Validate if command is safe to execute"""
        import re
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                self.logger.warning(f"Dangerous command pattern detected: {pattern}")
                return False
        
        # Check if command is in allowed list
        cmd_parts = command.split()
        if cmd_parts:
            base_cmd = cmd_parts[0]
            for category, commands in self.allowed_commands.items():
                if base_cmd in commands:
                    return True
        
        self.logger.warning(f"Command not in allowed list: {command}")
        return False
    
    async def execute_safe_command(self, command: str) -> str:
        """Execute command safely"""
        if not self.validate_command(command):
            raise SecurityException(f"Command not allowed: {command}")
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024  # 1MB limit
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30  # 30 second timeout
            )
            
            if process.returncode == 0:
                return stdout.decode('utf-8', errors='ignore')
            else:
                raise AutomationException(f"Command failed: {stderr.decode('utf-8', errors='ignore')}")
                
        except asyncio.TimeoutError:
            raise AutomationException("Command execution timed out")
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            raise AutomationException(f"Execution error: {e}")

# 8. Configuration Validation
from typing import Dict, Any, List
import jsonschema

class ConfigValidator:
    """Validate configuration with schema"""
    
    JARVIS_CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "master_name": {"type": "string", "minLength": 1},
            "ai_name": {"type": "string", "minLength": 1},
            "wake_word": {"type": "string", "minLength": 1},
            "voice_model": {"type": "string", "enum": ["whisper-base", "whisper-small", "whisper-medium"]},
            "llm_model": {"type": "string", "enum": ["llama2", "mistral", "codellama"]},
            "voice_rate": {"type": "integer", "minimum": 50, "maximum": 400},
            "voice_volume": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "language": {"type": "string", "pattern": "^[a-z]{2}$"},
            "theme": {"type": "string", "enum": ["dark", "light"]},
            "api_timeout": {"type": "integer", "minimum": 5, "maximum": 300},
            "max_context_length": {"type": "integer", "minimum": 512, "maximum": 8192},
            "enable_vision": {"type": "boolean"},
            "enable_automation": {"type": "boolean"},
            "enable_learning": {"type": "boolean"},
            "debug_mode": {"type": "boolean"}
        },
        "required": ["master_name", "ai_name", "wake_word"],
        "additionalProperties": False
    }
    
    @classmethod
    def validate_config(cls, config_dict: Dict[str, Any]) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        try:
            jsonschema.validate(config_dict, cls.JARVIS_CONFIG_SCHEMA)
        except jsonschema.ValidationError as e:
            errors.append(f"Configuration validation error: {e.message}")
        except jsonschema.SchemaError as e:
            errors.append(f"Schema error: {e.message}")
        
        return errors

# 9. Resource Management
class ResourceManager:
    """Manage system resources and limits"""
    
    def __init__(self):
        self.resource_limits = {
            'max_memory_mb': 1024,  # 1GB
            'max_cpu_percent': 80,
            'max_open_files': 100,
            'max_threads': 50
        }
        self.current_usage = {
            'memory_mb': 0,
            'cpu_percent': 0,
            'open_files': 0,
            'threads': 0
        }
        self.logger = ContextualLogger(__name__)
    
    def check_resource_limits(self) -> List[str]:
        """Check if resource limits are exceeded"""
        warnings = []
        
        import psutil
        import threading
        
        try:
            # Get current process
            process = psutil.Process()
            
            # Memory usage
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.current_usage['memory_mb'] = memory_mb
            
            if memory_mb > self.resource_limits['max_memory_mb']:
                warnings.append(f"Memory usage ({memory_mb:.1f}MB) exceeds limit ({self.resource_limits['max_memory_mb']}MB)")
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            self.current_usage['cpu_percent'] = cpu_percent
            
            if cpu_percent > self.resource_limits['max_cpu_percent']:
                warnings.append(f"CPU usage ({cpu_percent:.1f}%) exceeds limit ({self.resource_limits['max_cpu_percent']}%)")
            
            # Open files
            try:
                open_files = len(process.open_files())
                self.current_usage['open_files'] = open_files
                
                if open_files > self.resource_limits['max_open_files']:
                    warnings.append(f"Open files ({open_files}) exceeds limit ({self.resource_limits['max_open_files']})")
            except psutil.AccessDenied:
                pass
            
            # Thread count
            thread_count = threading.active_count()
            self.current_usage['threads'] = thread_count
            
            if thread_count > self.resource_limits['max_threads']:
                warnings.append(f"Thread count ({thread_count}) exceeds limit ({self.resource_limits['max_threads']})")
        
        except Exception as e:
            self.logger.error(f"Resource check failed: {e}")
            warnings.append(f"Resource monitoring error: {e}")
        
        return warnings

# 10. Exception Handler Decorator
def handle_exceptions(
    default_return=None,
    log_errors=True,
    reraise=False,
    custom_handler=None
):
    """Decorator for consistent exception handling"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger = logging.getLogger(func.__module__)
                    logger.error(
                        f"Exception in {func.__name__}: {e}",
                        exc_info=True
                    )
                
                if custom_handler:
                    return await custom_handler(e, *args, **kwargs)
                
                if reraise:
                    raise
                
                return default_return
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger = logging.getLogger(func.__module__)
                    logger.error(
                        f"Exception in {func.__name__}: {e}",
                        exc_info=True
                    )
                
                if custom_handler:
                    return custom_handler(e, *args, **kwargs)
                
                if reraise:
                    raise
                
                return default_return
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# 11. Integration Example
class RobustJARVIS:
    """JARVIS with enhanced error handling"""
    
    def __init__(self):
        self.logger = ContextualLogger(__name__)
        self.health_check = HealthCheck()
        self.service_registry = ServiceRegistry()
        self.command_executor = SafeCommandExecutor()
        self.resource_manager = ResourceManager()
        
        # Circuit breakers for external services
        self.weather_circuit = CircuitBreaker(
            CircuitBreakerConfig(failure_threshold=3, timeout=30)
        )
        self.news_circuit = CircuitBreaker(
            CircuitBreakerConfig(failure_threshold=5, timeout=60)
        )
        
        self._setup_health_checks()
    
    def _setup_health_checks(self):
        """Setup health checks"""
        self.health_check.register_check("memory", self._check_memory_health)
        self.health_check.register_check("disk_space", self._check_disk_health)
        self.health_check.register_check("ai_model", self._check_ai_model_health)
    
    async def _check_memory_health(self) -> dict:
        """Check memory health"""
        import psutil
        memory = psutil.virtual_memory()
        
        return {
            "available_gb": memory.available / (1024**3),
            "percent_used": memory.percent,
            "status": "healthy" if memory.percent < 90 else "warning"
        }
    
    async def _check_disk_health(self) -> dict:
        """Check disk health"""
        import psutil
        disk = psutil.disk_usage('/')
        
        percent_used = (disk.used / disk.total) * 100
        
        return {
            "free_gb": disk.free / (1024**3),
            "percent_used": percent_used,
            "status": "healthy" if percent_used < 90 else "warning"
        }
    
    async def _check_ai_model_health(self) -> dict:
        """Check AI model health"""
        try:
            # Test if AI model is responsive
            test_response = await self.generate_simple_response("test")
            return {
                "status": "healthy",
                "response_length": len(test_response)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @handle_exceptions(default_return="I'm having trouble with that request.")
    @retry_with_backoff(max_retries=2)
    async def process_command_safely(self, user_input: str) -> str:
        """Process command with robust error handling"""
        self.logger.set_context(command=user_input[:50])
        
        try:
            # Check resource limits
            resource_warnings = self.resource_manager.check_resource_limits()
            if resource_warnings:
                self.logger.warning(f"Resource warnings: {resource_warnings}")
            
            # Process the command
            response = await self._internal_process_command(user_input)
            
            self.logger.info("Command processed successfully")
            return response
            
        except AIModelException as e:
            self.logger.error(f"AI model error: {e}")
            return "I'm experiencing AI model issues. Please try again later."
        
        except VoiceProcessingException as e:
            self.logger.error(f"Voice processing error: {e}")
            return "I had trouble understanding your voice input."
        
        except AutomationException as e:
            self.logger.error(f"Automation error: {e}")
            return "I couldn't complete that automation task safely."
        
        except SecurityException as e:
            self.logger.error(f"Security error: {e}")
            return "I cannot complete that request for security reasons."
        
        finally:
            self.logger.clear_context()
    
    @weather_circuit
    async def get_weather_safely(self, location: str) -> str:
        """Get weather with circuit breaker protection"""
        # Weather API call implementation
        pass
    
    @news_circuit  
    async def get_news_safely(self) -> str:
        """Get news with circuit breaker protection"""
        # News API call implementation
        pass
    
    async def _internal_process_command(self, user_input: str) -> str:
        """Internal command processing"""
        # Implementation would go here
        return "Command processed"
    
    async def generate_simple_response(self, prompt: str) -> str:
        """Simple response generation for health checks"""
        return "Test response"
    
    async def shutdown_gracefully(self):
        """Graceful shutdown with cleanup"""
        self.logger.info("Starting graceful shutdown...")
        
        try:
            # Run final health check
            health_results = await self.health_check.run_checks()
            self.logger.info(f"Final health check: {health_results}")
            
            # Close resources
            await self._cleanup_resources()
            
            self.logger.info("Shutdown completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def _cleanup_resources(self):
        """Clean up resources"""
        # Implementation for resource cleanup
        pass