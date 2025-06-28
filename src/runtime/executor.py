#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Runtime Executor
Main execution engine for AI agent crews and workflows
"""

import uuid
import time
import asyncio
import threading
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, Future

logger = logging.getLogger(__name__)


class ExecutorStatus(Enum):
    """Executor status"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ExecutionMode(Enum):
    """Execution modes"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"


@dataclass
class ExecutionContext:
    """Execution context for crews and workflows"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    security_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Execution result"""
    success: bool
    session_id: str
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    artifacts: Dict[str, Any] = field(default_factory=dict)


class RuntimeExecutor:
    """
    Main execution engine for AI agent crews and workflows
    Manages the lifecycle and execution of AI agent systems
    """
    
    def __init__(
        self,
        max_concurrent_crews: int = 10,
        max_workers: int = 50,
        execution_timeout: Optional[float] = None,
        enable_monitoring: bool = True,
        enable_security: bool = True,
        resource_limits: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the runtime executor
        
        Args:
            max_concurrent_crews: Maximum number of concurrent crew executions
            max_workers: Maximum number of worker threads
            execution_timeout: Default execution timeout in seconds
            enable_monitoring: Whether to enable execution monitoring
            enable_security: Whether to enable security features
            resource_limits: Resource usage limits
        """
        self.id = str(uuid.uuid4())
        self.max_concurrent_crews = max_concurrent_crews
        self.max_workers = max_workers
        self.execution_timeout = execution_timeout
        self.enable_monitoring = enable_monitoring
        self.enable_security = enable_security
        self.resource_limits = resource_limits or {}
        
        # State management
        self.status = ExecutorStatus.IDLE
        self.created_at = time.time()
        self.started_at = None
        
        # Execution tracking
        self.active_sessions = {}
        self.execution_history = []
        self.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "total_execution_time": 0.0,
            "peak_concurrent_crews": 0,
            "resource_usage": {}
        }
        
        # Thread management
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.event_loop = None
        self.loop_thread = None
        
        # Monitoring and callbacks
        self.execution_callbacks = []
        self.error_handlers = []
        self.monitoring_enabled = enable_monitoring
        
        # Security
        self.security_manager = None
        if enable_security:
            self._initialize_security()
        
        logger.info(f"Runtime executor {self.id} initialized")
    
    def start(self) -> None:
        """Start the runtime executor"""
        if self.status != ExecutorStatus.IDLE:
            raise RuntimeError(f"Executor is not idle, current status: {self.status}")
        
        self.status = ExecutorStatus.INITIALIZING
        
        try:
            # Start event loop in separate thread
            self._start_event_loop()
            
            # Initialize monitoring if enabled
            if self.monitoring_enabled:
                self._start_monitoring()
            
            self.status = ExecutorStatus.RUNNING
            self.started_at = time.time()
            
            logger.info(f"Runtime executor {self.id} started")
            
        except Exception as e:
            self.status = ExecutorStatus.ERROR
            logger.error(f"Failed to start runtime executor: {str(e)}")
            raise
    
    def stop(self) -> None:
        """Stop the runtime executor"""
        if self.status not in [ExecutorStatus.RUNNING, ExecutorStatus.PAUSED]:
            logger.warning(f"Executor is not running, current status: {self.status}")
            return
        
        self.status = ExecutorStatus.STOPPING
        
        try:
            # Stop all active sessions
            self._stop_all_sessions()
            
            # Stop monitoring
            if self.monitoring_enabled:
                self._stop_monitoring()
            
            # Stop event loop
            self._stop_event_loop()
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            self.status = ExecutorStatus.STOPPED
            logger.info(f"Runtime executor {self.id} stopped")
            
        except Exception as e:
            self.status = ExecutorStatus.ERROR
            logger.error(f"Error stopping runtime executor: {str(e)}")
            raise
    
    def execute_crew(
        self,
        crew: 'BaseCrew',
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[ExecutionContext] = None,
        mode: ExecutionMode = ExecutionMode.ASYNCHRONOUS,
        timeout: Optional[float] = None
    ) -> Union[ExecutionResult, Future[ExecutionResult]]:
        """
        Execute a crew
        
        Args:
            crew: Crew to execute
            inputs: Initial inputs for the crew
            context: Execution context
            mode: Execution mode
            timeout: Execution timeout
            
        Returns:
            ExecutionResult or Future[ExecutionResult] depending on mode
        """
        if self.status != ExecutorStatus.RUNNING:
            raise RuntimeError(f"Executor is not running, current status: {self.status}")
        
        # Check concurrent crew limit
        if len(self.active_sessions) >= self.max_concurrent_crews:
            raise RuntimeError(f"Maximum concurrent crews ({self.max_concurrent_crews}) reached")
        
        # Create execution context
        if context is None:
            context = ExecutionContext()
        
        # Set timeout
        execution_timeout = timeout or self.execution_timeout
        
        # Execute based on mode
        if mode == ExecutionMode.SYNCHRONOUS:
            return self._execute_crew_sync(crew, inputs, context, execution_timeout)
        elif mode == ExecutionMode.ASYNCHRONOUS:
            return self._execute_crew_async(crew, inputs, context, execution_timeout)
        elif mode == ExecutionMode.PARALLEL:
            return self._execute_crew_parallel(crew, inputs, context, execution_timeout)
        else:
            raise ValueError(f"Unsupported execution mode: {mode}")
    
    def execute_workflow(
        self,
        workflow: 'BaseFlow',
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[ExecutionContext] = None,
        mode: ExecutionMode = ExecutionMode.ASYNCHRONOUS
    ) -> Union[ExecutionResult, Future[ExecutionResult]]:
        """
        Execute a workflow
        
        Args:
            workflow: Workflow to execute
            inputs: Initial inputs for the workflow
            context: Execution context
            mode: Execution mode
            
        Returns:
            ExecutionResult or Future[ExecutionResult] depending on mode
        """
        if self.status != ExecutorStatus.RUNNING:
            raise RuntimeError(f"Executor is not running, current status: {self.status}")
        
        # Create execution context
        if context is None:
            context = ExecutionContext()
        
        # Execute workflow
        return self._execute_workflow_internal(workflow, inputs, context, mode)
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an execution session"""
        session = self.active_sessions.get(session_id)
        if session:
            return {
                "session_id": session_id,
                "status": session.get("status", "unknown"),
                "started_at": session.get("started_at"),
                "progress": session.get("progress", {}),
                "current_task": session.get("current_task"),
                "execution_time": time.time() - session.get("started_at", time.time())
            }
        return None
    
    def cancel_session(self, session_id: str, reason: str = "User cancelled") -> bool:
        """Cancel an execution session"""
        session = self.active_sessions.get(session_id)
        if session:
            try:
                # Cancel the crew/workflow execution
                if "crew" in session:
                    session["crew"].cancel_execution(reason)
                elif "workflow" in session:
                    session["workflow"].cancel_execution(reason)
                
                # Mark session as cancelled
                session["status"] = "cancelled"
                session["cancelled_at"] = time.time()
                session["cancellation_reason"] = reason
                
                logger.info(f"Session {session_id} cancelled: {reason}")
                return True
                
            except Exception as e:
                logger.error(f"Error cancelling session {session_id}: {str(e)}")
                return False
        
        return False
    
    def pause_session(self, session_id: str) -> bool:
        """Pause an execution session"""
        session = self.active_sessions.get(session_id)
        if session:
            try:
                if "crew" in session:
                    session["crew"].pause_execution()
                elif "workflow" in session:
                    session["workflow"].pause_execution()
                
                session["status"] = "paused"
                session["paused_at"] = time.time()
                
                logger.info(f"Session {session_id} paused")
                return True
                
            except Exception as e:
                logger.error(f"Error pausing session {session_id}: {str(e)}")
                return False
        
        return False
    
    def resume_session(self, session_id: str) -> bool:
        """Resume a paused execution session"""
        session = self.active_sessions.get(session_id)
        if session and session.get("status") == "paused":
            try:
                if "crew" in session:
                    session["crew"].resume_execution()
                elif "workflow" in session:
                    session["workflow"].resume_execution()
                
                session["status"] = "running"
                session["resumed_at"] = time.time()
                
                logger.info(f"Session {session_id} resumed")
                return True
                
            except Exception as e:
                logger.error(f"Error resuming session {session_id}: {str(e)}")
                return False
        
        return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get executor performance metrics"""
        current_time = time.time()
        uptime = current_time - self.created_at
        
        metrics = self.performance_metrics.copy()
        metrics.update({
            "executor_id": self.id,
            "status": self.status.value,
            "uptime": uptime,
            "active_sessions": len(self.active_sessions),
            "thread_pool_size": self.max_workers,
            "memory_usage": self._get_memory_usage(),
            "cpu_usage": self._get_cpu_usage(),
            "last_updated": current_time
        })
        
        return metrics
    
    def add_execution_callback(self, callback: Callable) -> None:
        """Add execution callback"""
        if callback not in self.execution_callbacks:
            self.execution_callbacks.append(callback)
    
    def remove_execution_callback(self, callback: Callable) -> None:
        """Remove execution callback"""
        if callback in self.execution_callbacks:
            self.execution_callbacks.remove(callback)
    
    def add_error_handler(self, handler: Callable) -> None:
        """Add error handler"""
        if handler not in self.error_handlers:
            self.error_handlers.append(handler)
    
    def _execute_crew_sync(
        self,
        crew: 'BaseCrew',
        inputs: Optional[Dict[str, Any]],
        context: ExecutionContext,
        timeout: Optional[float]
    ) -> ExecutionResult:
        """Execute crew synchronously"""
        start_time = time.time()
        session_id = context.session_id
        
        try:
            # Create session
            session = self._create_session(session_id, crew, context)
            
            # Execute crew
            result = crew.kickoff(inputs)
            
            # Create execution result
            execution_result = ExecutionResult(
                success=result.get("success", False),
                session_id=session_id,
                outputs=result,
                execution_time=time.time() - start_time
            )
            
            # Update metrics
            self._update_performance_metrics(execution_result)
            
            # Clean up session
            self._cleanup_session(session_id)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Synchronous crew execution failed: {str(e)}")
            self._cleanup_session(session_id)
            return ExecutionResult(
                success=False,
                session_id=session_id,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    def _execute_crew_async(
        self,
        crew: 'BaseCrew',
        inputs: Optional[Dict[str, Any]],
        context: ExecutionContext,
        timeout: Optional[float]
    ) -> Future[ExecutionResult]:
        """Execute crew asynchronously"""
        def execute():
            return self._execute_crew_sync(crew, inputs, context, timeout)
        
        return self.thread_pool.submit(execute)
    
    def _execute_crew_parallel(
        self,
        crew: 'BaseCrew',
        inputs: Optional[Dict[str, Any]],
        context: ExecutionContext,
        timeout: Optional[float]
    ) -> Future[ExecutionResult]:
        """Execute crew with parallel task processing"""
        def execute():
            # Set crew to parallel processing mode
            original_process_type = crew.process_type
            crew.process_type = crew.ProcessType.PARALLEL
            
            try:
                result = self._execute_crew_sync(crew, inputs, context, timeout)
                return result
            finally:
                # Restore original process type
                crew.process_type = original_process_type
        
        return self.thread_pool.submit(execute)
    
    def _execute_workflow_internal(
        self,
        workflow: 'BaseFlow',
        inputs: Optional[Dict[str, Any]],
        context: ExecutionContext,
        mode: ExecutionMode
    ) -> Union[ExecutionResult, Future[ExecutionResult]]:
        """Internal workflow execution"""
        # Placeholder for workflow execution
        # This would integrate with the flow system
        
        def execute():
            start_time = time.time()
            session_id = context.session_id
            
            try:
                # Create session
                session = self._create_session(session_id, workflow, context)
                
                # Execute workflow
                result = workflow.execute(inputs)
                
                # Create execution result
                execution_result = ExecutionResult(
                    success=result.get("success", False),
                    session_id=session_id,
                    outputs=result,
                    execution_time=time.time() - start_time
                )
                
                # Clean up session
                self._cleanup_session(session_id)
                
                return execution_result
                
            except Exception as e:
                logger.error(f"Workflow execution failed: {str(e)}")
                self._cleanup_session(session_id)
                return ExecutionResult(
                    success=False,
                    session_id=session_id,
                    errors=[str(e)],
                    execution_time=time.time() - start_time
                )
        
        if mode == ExecutionMode.SYNCHRONOUS:
            return execute()
        else:
            return self.thread_pool.submit(execute)
    
    def _create_session(
        self,
        session_id: str,
        execution_target: Union['BaseCrew', 'BaseFlow'],
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Create execution session"""
        session = {
            "session_id": session_id,
            "status": "running",
            "started_at": time.time(),
            "context": context,
            "progress": {}
        }
        
        if hasattr(execution_target, 'agents'):  # It's a crew
            session["crew"] = execution_target
            session["type"] = "crew"
        else:  # It's a workflow
            session["workflow"] = execution_target
            session["type"] = "workflow"
        
        self.active_sessions[session_id] = session
        
        # Update peak concurrent crews metric
        current_count = len(self.active_sessions)
        if current_count > self.performance_metrics["peak_concurrent_crews"]:
            self.performance_metrics["peak_concurrent_crews"] = current_count
        
        return session
    
    def _cleanup_session(self, session_id: str) -> None:
        """Clean up execution session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session["completed_at"] = time.time()
            session["status"] = "completed"
            
            # Move to history
            self.execution_history.append(session)
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            # Keep only last 1000 executions in history
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
    
    def _update_performance_metrics(self, result: ExecutionResult) -> None:
        """Update performance metrics"""
        self.performance_metrics["total_executions"] += 1
        
        if result.success:
            self.performance_metrics["successful_executions"] += 1
        else:
            self.performance_metrics["failed_executions"] += 1
        
        self.performance_metrics["total_execution_time"] += result.execution_time
        self.performance_metrics["average_execution_time"] = (
            self.performance_metrics["total_execution_time"] / 
            self.performance_metrics["total_executions"]
        )
    
    def _start_event_loop(self) -> None:
        """Start event loop in separate thread"""
        def run_loop():
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            self.event_loop.run_forever()
        
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
    
    def _stop_event_loop(self) -> None:
        """Stop event loop"""
        if self.event_loop:
            self.event_loop.call_soon_threadsafe(self.event_loop.stop)
            if self.loop_thread:
                self.loop_thread.join(timeout=5.0)
    
    def _stop_all_sessions(self) -> None:
        """Stop all active sessions"""
        for session_id in list(self.active_sessions.keys()):
            self.cancel_session(session_id, "Executor shutdown")
    
    def _start_monitoring(self) -> None:
        """Start monitoring system"""
        # Placeholder for monitoring implementation
        logger.info("Monitoring started")
    
    def _stop_monitoring(self) -> None:
        """Stop monitoring system"""
        # Placeholder for monitoring implementation
        logger.info("Monitoring stopped")
    
    def _initialize_security(self) -> None:
        """Initialize security manager"""
        # Placeholder for security implementation
        logger.info("Security manager initialized")
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0
