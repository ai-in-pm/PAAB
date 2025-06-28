#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Base Tool Classes
Evolved from PsychoPy's tools system for AI agent capabilities
"""

import uuid
import time
import inspect
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Types of tools"""
    LLM = "llm"
    DATA_PROCESSING = "data_processing"
    WEB_SCRAPING = "web_scraping"
    FILE_SYSTEM = "file_system"
    API_INTEGRATION = "api_integration"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    COMMUNICATION = "communication"
    CUSTOM = "custom"


class ToolStatus(Enum):
    """Tool execution status"""
    IDLE = "idle"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass
class ToolParameter:
    """Tool parameter definition"""
    name: str
    type: Type
    description: str
    required: bool = True
    default: Any = None
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    examples: List[Any] = field(default_factory=list)


@dataclass
class ToolResult:
    """Tool execution result"""
    success: bool
    output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolMetrics:
    """Tool performance metrics"""
    execution_count: int = 0
    success_rate: float = 0.0
    average_execution_time: float = 0.0
    total_execution_time: float = 0.0
    error_count: int = 0
    last_execution_time: float = 0.0
    last_updated: float = field(default_factory=time.time)


class BaseTool(ABC):
    """
    Base class for AI agent tools, evolved from PsychoPy tools
    Provides standardized interface for agent capabilities
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        tool_type: ToolType = ToolType.CUSTOM,
        parameters: Optional[List[ToolParameter]] = None,
        return_type: Optional[Type] = None,
        async_execution: bool = False,
        cache_results: bool = False,
        timeout: Optional[float] = None,
        retry_count: int = 3,
        validation_enabled: bool = True
    ):
        """
        Initialize a base tool
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            tool_type: Type/category of the tool
            parameters: List of tool parameters
            return_type: Expected return type
            async_execution: Whether tool supports async execution
            cache_results: Whether to cache execution results
            timeout: Maximum execution time in seconds
            retry_count: Number of retry attempts on failure
            validation_enabled: Whether to validate inputs/outputs
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.tool_type = tool_type
        self.parameters = parameters or []
        self.return_type = return_type
        self.async_execution = async_execution
        self.cache_results = cache_results
        self.timeout = timeout
        self.retry_count = retry_count
        self.validation_enabled = validation_enabled
        
        # State management
        self.status = ToolStatus.IDLE
        self.current_execution = None
        
        # Execution tracking
        self.execution_history = []
        self.metrics = ToolMetrics()
        self.cache = {} if cache_results else None
        
        # Dependencies and requirements
        self.dependencies = []
        self.requirements = []
        
        logger.info(f"Tool {self.name} ({self.id}) initialized")
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            ToolResult containing execution outcome
        """
        pass
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate tool input parameters
        
        Args:
            **kwargs: Input parameters to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        if not self.validation_enabled:
            return True
        
        try:
            # Check required parameters
            for param in self.parameters:
                if param.required and param.name not in kwargs:
                    logger.error(f"Required parameter '{param.name}' missing for tool {self.name}")
                    return False
            
            # Validate parameter types and rules
            for param_name, param_value in kwargs.items():
                param_def = self._get_parameter_definition(param_name)
                if param_def:
                    if not self._validate_parameter(param_def, param_value):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed for tool {self.name}: {str(e)}")
            return False
    
    def validate_output(self, output: Any) -> bool:
        """
        Validate tool output
        
        Args:
            output: Output to validate
            
        Returns:
            True if output is valid, False otherwise
        """
        if not self.validation_enabled:
            return True
        
        try:
            # Check return type if specified
            if self.return_type and not isinstance(output, self.return_type):
                logger.error(f"Output type mismatch for tool {self.name}: expected {self.return_type}, got {type(output)}")
                return False
            
            # Additional output validation can be implemented by subclasses
            return self._validate_output_content(output)
            
        except Exception as e:
            logger.error(f"Output validation failed for tool {self.name}: {str(e)}")
            return False
    
    def execute_with_validation(self, **kwargs) -> ToolResult:
        """
        Execute tool with input/output validation
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            ToolResult containing execution outcome
        """
        start_time = time.time()
        self.status = ToolStatus.EXECUTING
        
        try:
            # Validate input
            if not self.validate_input(**kwargs):
                return ToolResult(
                    success=False,
                    output=None,
                    error_message="Input validation failed",
                    execution_time=time.time() - start_time
                )
            
            # Check cache if enabled
            if self.cache_results:
                cache_key = self._generate_cache_key(**kwargs)
                if cache_key in self.cache:
                    logger.info(f"Cache hit for tool {self.name}")
                    cached_result = self.cache[cache_key]
                    cached_result.metadata["from_cache"] = True
                    return cached_result
            
            # Execute tool
            result = self._execute_with_retry(**kwargs)
            
            # Validate output
            if result.success and not self.validate_output(result.output):
                result.success = False
                result.error_message = "Output validation failed"
            
            # Update execution time
            result.execution_time = time.time() - start_time
            
            # Cache result if enabled and successful
            if self.cache_results and result.success:
                cache_key = self._generate_cache_key(**kwargs)
                self.cache[cache_key] = result
            
            # Update metrics
            self._update_metrics(result)
            
            # Record execution history
            self._record_execution(kwargs, result)
            
            self.status = ToolStatus.COMPLETED if result.success else ToolStatus.FAILED
            return result
            
        except Exception as e:
            self.status = ToolStatus.FAILED
            error_result = ToolResult(
                success=False,
                output=None,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
            self._update_metrics(error_result)
            return error_result
    
    def _execute_with_retry(self, **kwargs) -> ToolResult:
        """Execute tool with retry logic"""
        last_error = None
        
        for attempt in range(self.retry_count + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retrying tool {self.name} (attempt {attempt + 1})")
                
                result = self.execute(**kwargs)
                
                if result.success:
                    return result
                else:
                    last_error = result.error_message
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"Tool {self.name} execution attempt {attempt + 1} failed: {str(e)}")
        
        return ToolResult(
            success=False,
            output=None,
            error_message=f"Tool execution failed after {self.retry_count + 1} attempts. Last error: {last_error}"
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get tool schema for LLM function calling
        
        Returns:
            Dict containing tool schema
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": self._get_json_type(param.type),
                "description": param.description
            }
            
            if param.examples:
                properties[param.name]["examples"] = param.examples
            
            if param.validation_rules:
                properties[param.name].update(param.validation_rules)
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get tool status and metrics"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tool_type": self.tool_type.value,
            "status": self.status.value,
            "parameters_count": len(self.parameters),
            "async_execution": self.async_execution,
            "cache_enabled": self.cache_results,
            "cache_size": len(self.cache) if self.cache else 0,
            "metrics": self.metrics.__dict__,
            "dependencies": self.dependencies,
            "requirements": self.requirements
        }
    
    def clear_cache(self) -> None:
        """Clear tool cache"""
        if self.cache:
            self.cache.clear()
            logger.info(f"Cache cleared for tool {self.name}")
    
    def add_parameter(self, parameter: ToolParameter) -> None:
        """Add a parameter to the tool"""
        if parameter.name not in [p.name for p in self.parameters]:
            self.parameters.append(parameter)
            logger.info(f"Parameter '{parameter.name}' added to tool {self.name}")
    
    def remove_parameter(self, parameter_name: str) -> None:
        """Remove a parameter from the tool"""
        self.parameters = [p for p in self.parameters if p.name != parameter_name]
        logger.info(f"Parameter '{parameter_name}' removed from tool {self.name}")
    
    def _get_parameter_definition(self, param_name: str) -> Optional[ToolParameter]:
        """Get parameter definition by name"""
        for param in self.parameters:
            if param.name == param_name:
                return param
        return None
    
    def _validate_parameter(self, param_def: ToolParameter, value: Any) -> bool:
        """Validate a single parameter"""
        try:
            # Type validation
            if not isinstance(value, param_def.type):
                # Try type conversion for basic types
                if param_def.type in [int, float, str, bool]:
                    try:
                        value = param_def.type(value)
                    except (ValueError, TypeError):
                        logger.error(f"Parameter '{param_def.name}' type validation failed")
                        return False
                else:
                    logger.error(f"Parameter '{param_def.name}' type validation failed")
                    return False
            
            # Validation rules
            for rule_name, rule_value in param_def.validation_rules.items():
                if not self._apply_validation_rule(rule_name, rule_value, value):
                    logger.error(f"Parameter '{param_def.name}' validation rule '{rule_name}' failed")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Parameter validation error: {str(e)}")
            return False
    
    def _apply_validation_rule(self, rule_name: str, rule_value: Any, value: Any) -> bool:
        """Apply a validation rule"""
        try:
            if rule_name == "min_length" and hasattr(value, "__len__"):
                return len(value) >= rule_value
            elif rule_name == "max_length" and hasattr(value, "__len__"):
                return len(value) <= rule_value
            elif rule_name == "min_value" and isinstance(value, (int, float)):
                return value >= rule_value
            elif rule_name == "max_value" and isinstance(value, (int, float)):
                return value <= rule_value
            elif rule_name == "pattern" and isinstance(value, str):
                import re
                return bool(re.match(rule_value, value))
            elif rule_name == "enum":
                return value in rule_value
            else:
                return True  # Unknown rule, assume valid
                
        except Exception:
            return False
    
    def _validate_output_content(self, output: Any) -> bool:
        """Validate output content - to be overridden by subclasses"""
        return True
    
    def _generate_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters"""
        import hashlib
        import json
        
        # Sort parameters for consistent key generation
        sorted_params = sorted(kwargs.items())
        key_string = json.dumps(sorted_params, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_metrics(self, result: ToolResult) -> None:
        """Update tool metrics"""
        self.metrics.execution_count += 1
        self.metrics.last_execution_time = result.execution_time
        self.metrics.total_execution_time += result.execution_time
        self.metrics.average_execution_time = self.metrics.total_execution_time / self.metrics.execution_count
        
        if result.success:
            success_count = self.metrics.execution_count - self.metrics.error_count
            self.metrics.success_rate = success_count / self.metrics.execution_count
        else:
            self.metrics.error_count += 1
            self.metrics.success_rate = (self.metrics.execution_count - self.metrics.error_count) / self.metrics.execution_count
        
        self.metrics.last_updated = time.time()
    
    def _record_execution(self, inputs: Dict[str, Any], result: ToolResult) -> None:
        """Record execution in history"""
        execution_record = {
            "timestamp": time.time(),
            "inputs": inputs,
            "result": {
                "success": result.success,
                "execution_time": result.execution_time,
                "error_message": result.error_message
            }
        }
        
        self.execution_history.append(execution_record)
        
        # Keep only last 100 executions
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def _get_json_type(self, python_type: Type) -> str:
        """Convert Python type to JSON schema type"""
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        return type_mapping.get(python_type, "string")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tool_type": self.tool_type.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type.__name__,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default
                }
                for p in self.parameters
            ],
            "return_type": self.return_type.__name__ if self.return_type else None,
            "async_execution": self.async_execution,
            "cache_results": self.cache_results,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "metrics": self.metrics.__dict__,
            "status": self.status.value
        }
