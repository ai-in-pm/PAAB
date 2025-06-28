#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Monitoring and Observability System
Comprehensive monitoring for AI agent systems
"""

import time
import json
import asyncio
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert"""
    id: str
    level: AlertLevel
    message: str
    source: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_agents: int = 0
    active_crews: int = 0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0


class BaseMonitor(ABC):
    """Base class for monitoring components"""
    
    def __init__(self, name: str, collection_interval: float = 60.0):
        """
        Initialize monitor
        
        Args:
            name: Monitor name
            collection_interval: Data collection interval in seconds
        """
        self.name = name
        self.collection_interval = collection_interval
        self.is_running = False
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.alerts: List[Alert] = []
        
        # Callbacks
        self.on_metric_collected: Optional[Callable] = None
        self.on_alert_triggered: Optional[Callable] = None
    
    @abstractmethod
    async def collect_metrics(self) -> List[Metric]:
        """Collect metrics"""
        pass
    
    @abstractmethod
    def check_alerts(self, metrics: List[Metric]) -> List[Alert]:
        """Check for alert conditions"""
        pass
    
    async def start(self) -> None:
        """Start monitoring"""
        self.is_running = True
        logger.info(f"Monitor {self.name} started")
        
        while self.is_running:
            try:
                # Collect metrics
                metrics = await self.collect_metrics()
                
                # Store metrics
                for metric in metrics:
                    self.metrics[metric.name].append(metric)
                    
                    # Limit metric history
                    if len(self.metrics[metric.name]) > 1000:
                        self.metrics[metric.name] = self.metrics[metric.name][-1000:]
                
                # Check alerts
                new_alerts = self.check_alerts(metrics)
                self.alerts.extend(new_alerts)
                
                # Trigger callbacks
                if self.on_metric_collected:
                    for metric in metrics:
                        self.on_metric_collected(metric)
                
                if self.on_alert_triggered:
                    for alert in new_alerts:
                        self.on_alert_triggered(alert)
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitor {self.name}: {str(e)}")
                await asyncio.sleep(self.collection_interval)
    
    def stop(self) -> None:
        """Stop monitoring"""
        self.is_running = False
        logger.info(f"Monitor {self.name} stopped")
    
    def get_latest_metrics(self, metric_name: str, count: int = 10) -> List[Metric]:
        """Get latest metrics for a specific metric name"""
        return self.metrics[metric_name][-count:]
    
    def get_metric_summary(self, metric_name: str, duration: float = 3600) -> Dict[str, Any]:
        """Get summary statistics for a metric over a time period"""
        cutoff_time = time.time() - duration
        recent_metrics = [
            m for m in self.metrics[metric_name] 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"count": 0}
        
        values = [m.value for m in recent_metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else 0,
            "trend": "increasing" if len(values) > 1 and values[-1] > values[0] else "stable"
        }


class SystemMonitor(BaseMonitor):
    """System resource monitoring"""
    
    def __init__(self, **kwargs):
        super().__init__(name="system", **kwargs)
        self.process = None
        
        try:
            import psutil
            self.process = psutil.Process()
        except ImportError:
            logger.warning("psutil not available, system monitoring will be limited")
    
    async def collect_metrics(self) -> List[Metric]:
        """Collect system metrics"""
        metrics = []
        current_time = time.time()
        
        if self.process:
            try:
                # CPU usage
                cpu_percent = self.process.cpu_percent()
                metrics.append(Metric(
                    name="system_cpu_usage",
                    value=cpu_percent,
                    metric_type=MetricType.GAUGE,
                    timestamp=current_time
                ))
                
                # Memory usage
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()
                
                metrics.append(Metric(
                    name="system_memory_usage",
                    value=memory_percent,
                    metric_type=MetricType.GAUGE,
                    timestamp=current_time
                ))
                
                metrics.append(Metric(
                    name="system_memory_rss",
                    value=memory_info.rss,
                    metric_type=MetricType.GAUGE,
                    timestamp=current_time
                ))
                
                # Thread count
                thread_count = self.process.num_threads()
                metrics.append(Metric(
                    name="system_thread_count",
                    value=thread_count,
                    metric_type=MetricType.GAUGE,
                    timestamp=current_time
                ))
                
                # File descriptors (Unix only)
                try:
                    fd_count = self.process.num_fds()
                    metrics.append(Metric(
                        name="system_fd_count",
                        value=fd_count,
                        metric_type=MetricType.GAUGE,
                        timestamp=current_time
                    ))
                except (AttributeError, OSError):
                    pass  # Not available on Windows
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {str(e)}")
        
        return metrics
    
    def check_alerts(self, metrics: List[Metric]) -> List[Alert]:
        """Check for system alerts"""
        alerts = []
        
        for metric in metrics:
            if metric.name == "system_cpu_usage" and metric.value > 90:
                alerts.append(Alert(
                    id=f"high_cpu_{metric.timestamp}",
                    level=AlertLevel.WARNING,
                    message=f"High CPU usage: {metric.value:.1f}%",
                    source="system_monitor"
                ))
            
            elif metric.name == "system_memory_usage" and metric.value > 85:
                alerts.append(Alert(
                    id=f"high_memory_{metric.timestamp}",
                    level=AlertLevel.WARNING,
                    message=f"High memory usage: {metric.value:.1f}%",
                    source="system_monitor"
                ))
            
            elif metric.name == "system_thread_count" and metric.value > 1000:
                alerts.append(Alert(
                    id=f"high_threads_{metric.timestamp}",
                    level=AlertLevel.WARNING,
                    message=f"High thread count: {metric.value}",
                    source="system_monitor"
                ))
        
        return alerts


class AgentMonitor(BaseMonitor):
    """Agent performance monitoring"""
    
    def __init__(self, agent_manager, **kwargs):
        super().__init__(name="agent", **kwargs)
        self.agent_manager = agent_manager
    
    async def collect_metrics(self) -> List[Metric]:
        """Collect agent metrics"""
        metrics = []
        current_time = time.time()
        
        try:
            # Get all agents
            agents = getattr(self.agent_manager, 'agents', [])
            
            # Agent count by state
            state_counts = defaultdict(int)
            total_agents = len(agents)
            
            for agent in agents:
                state = getattr(agent, 'state', 'unknown')
                state_counts[str(state)] += 1
            
            metrics.append(Metric(
                name="agent_total_count",
                value=total_agents,
                metric_type=MetricType.GAUGE,
                timestamp=current_time
            ))
            
            for state, count in state_counts.items():
                metrics.append(Metric(
                    name="agent_state_count",
                    value=count,
                    metric_type=MetricType.GAUGE,
                    timestamp=current_time,
                    labels={"state": state}
                ))
            
            # Agent performance metrics
            total_tasks_completed = 0
            total_tasks_failed = 0
            total_execution_time = 0.0
            
            for agent in agents:
                if hasattr(agent, 'metrics'):
                    total_tasks_completed += getattr(agent.metrics, 'tasks_completed', 0)
                    total_tasks_failed += getattr(agent.metrics, 'tasks_failed', 0)
                    total_execution_time += getattr(agent.metrics, 'total_execution_time', 0.0)
            
            metrics.append(Metric(
                name="agent_tasks_completed_total",
                value=total_tasks_completed,
                metric_type=MetricType.COUNTER,
                timestamp=current_time
            ))
            
            metrics.append(Metric(
                name="agent_tasks_failed_total",
                value=total_tasks_failed,
                metric_type=MetricType.COUNTER,
                timestamp=current_time
            ))
            
            if total_tasks_completed > 0:
                avg_execution_time = total_execution_time / total_tasks_completed
                metrics.append(Metric(
                    name="agent_avg_execution_time",
                    value=avg_execution_time,
                    metric_type=MetricType.GAUGE,
                    timestamp=current_time
                ))
            
        except Exception as e:
            logger.error(f"Error collecting agent metrics: {str(e)}")
        
        return metrics
    
    def check_alerts(self, metrics: List[Metric]) -> List[Alert]:
        """Check for agent alerts"""
        alerts = []
        
        for metric in metrics:
            if metric.name == "agent_tasks_failed_total":
                # Check if failure rate is too high
                completed_metric = next(
                    (m for m in metrics if m.name == "agent_tasks_completed_total"), 
                    None
                )
                
                if completed_metric and completed_metric.value > 0:
                    failure_rate = metric.value / (metric.value + completed_metric.value)
                    
                    if failure_rate > 0.2:  # 20% failure rate
                        alerts.append(Alert(
                            id=f"high_failure_rate_{metric.timestamp}",
                            level=AlertLevel.WARNING,
                            message=f"High agent failure rate: {failure_rate:.1%}",
                            source="agent_monitor"
                        ))
            
            elif metric.name == "agent_avg_execution_time" and metric.value > 300:  # 5 minutes
                alerts.append(Alert(
                    id=f"slow_execution_{metric.timestamp}",
                    level=AlertLevel.WARNING,
                    message=f"Slow agent execution: {metric.value:.1f}s average",
                    source="agent_monitor"
                ))
        
        return alerts


class CrewMonitor(BaseMonitor):
    """Crew execution monitoring"""
    
    def __init__(self, crew_manager, **kwargs):
        super().__init__(name="crew", **kwargs)
        self.crew_manager = crew_manager
    
    async def collect_metrics(self) -> List[Metric]:
        """Collect crew metrics"""
        metrics = []
        current_time = time.time()
        
        try:
            # Get all crews
            crews = getattr(self.crew_manager, 'crews', [])
            
            # Crew count by status
            status_counts = defaultdict(int)
            total_crews = len(crews)
            
            for crew in crews:
                status = getattr(crew, 'status', 'unknown')
                status_counts[str(status)] += 1
            
            metrics.append(Metric(
                name="crew_total_count",
                value=total_crews,
                metric_type=MetricType.GAUGE,
                timestamp=current_time
            ))
            
            for status, count in status_counts.items():
                metrics.append(Metric(
                    name="crew_status_count",
                    value=count,
                    metric_type=MetricType.GAUGE,
                    timestamp=current_time,
                    labels={"status": status}
                ))
            
            # Crew performance metrics
            total_executions = 0
            total_execution_time = 0.0
            successful_executions = 0
            
            for crew in crews:
                if hasattr(crew, 'metrics'):
                    total_executions += getattr(crew.metrics, 'total_executions', 0)
                    total_execution_time += getattr(crew.metrics, 'total_execution_time', 0.0)
                    successful_executions += getattr(crew.metrics, 'successful_executions', 0)
            
            metrics.append(Metric(
                name="crew_executions_total",
                value=total_executions,
                metric_type=MetricType.COUNTER,
                timestamp=current_time
            ))
            
            metrics.append(Metric(
                name="crew_successful_executions",
                value=successful_executions,
                metric_type=MetricType.COUNTER,
                timestamp=current_time
            ))
            
            if total_executions > 0:
                success_rate = successful_executions / total_executions
                avg_execution_time = total_execution_time / total_executions
                
                metrics.append(Metric(
                    name="crew_success_rate",
                    value=success_rate,
                    metric_type=MetricType.GAUGE,
                    timestamp=current_time
                ))
                
                metrics.append(Metric(
                    name="crew_avg_execution_time",
                    value=avg_execution_time,
                    metric_type=MetricType.GAUGE,
                    timestamp=current_time
                ))
            
        except Exception as e:
            logger.error(f"Error collecting crew metrics: {str(e)}")
        
        return metrics
    
    def check_alerts(self, metrics: List[Metric]) -> List[Alert]:
        """Check for crew alerts"""
        alerts = []
        
        for metric in metrics:
            if metric.name == "crew_success_rate" and metric.value < 0.8:  # 80% success rate
                alerts.append(Alert(
                    id=f"low_success_rate_{metric.timestamp}",
                    level=AlertLevel.WARNING,
                    message=f"Low crew success rate: {metric.value:.1%}",
                    source="crew_monitor"
                ))
            
            elif metric.name == "crew_avg_execution_time" and metric.value > 1800:  # 30 minutes
                alerts.append(Alert(
                    id=f"slow_crew_execution_{metric.timestamp}",
                    level=AlertLevel.WARNING,
                    message=f"Slow crew execution: {metric.value:.1f}s average",
                    source="crew_monitor"
                ))
        
        return alerts


class MonitoringSystem:
    """Central monitoring system"""
    
    def __init__(self):
        """Initialize monitoring system"""
        self.monitors: Dict[str, BaseMonitor] = {}
        self.is_running = False
        self.performance_history: deque = deque(maxlen=1000)
        
        # Aggregated metrics
        self.aggregated_metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.all_alerts: List[Alert] = []
        
        # Event loop for async operations
        self.loop = None
        self.monitor_tasks = []
        
        logger.info("Monitoring system initialized")
    
    def add_monitor(self, monitor: BaseMonitor) -> None:
        """Add a monitor to the system"""
        self.monitors[monitor.name] = monitor
        
        # Set up callbacks
        monitor.on_metric_collected = self._on_metric_collected
        monitor.on_alert_triggered = self._on_alert_triggered
        
        logger.info(f"Added monitor: {monitor.name}")
    
    def remove_monitor(self, monitor_name: str) -> None:
        """Remove a monitor from the system"""
        if monitor_name in self.monitors:
            monitor = self.monitors[monitor_name]
            monitor.stop()
            del self.monitors[monitor_name]
            logger.info(f"Removed monitor: {monitor_name}")
    
    async def start(self) -> None:
        """Start all monitors"""
        if self.is_running:
            return
        
        self.is_running = True
        self.loop = asyncio.get_event_loop()
        
        # Start all monitors
        for monitor in self.monitors.values():
            task = asyncio.create_task(monitor.start())
            self.monitor_tasks.append(task)
        
        # Start performance snapshot collection
        snapshot_task = asyncio.create_task(self._collect_performance_snapshots())
        self.monitor_tasks.append(snapshot_task)
        
        logger.info("Monitoring system started")
    
    def stop(self) -> None:
        """Stop all monitors"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop all monitors
        for monitor in self.monitors.values():
            monitor.stop()
        
        # Cancel all tasks
        for task in self.monitor_tasks:
            task.cancel()
        
        self.monitor_tasks.clear()
        
        logger.info("Monitoring system stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        current_time = time.time()
        
        # Get latest performance snapshot
        latest_snapshot = self.performance_history[-1] if self.performance_history else None
        
        # Count active alerts by level
        active_alerts = [alert for alert in self.all_alerts if not alert.resolved]
        alert_counts = defaultdict(int)
        for alert in active_alerts:
            alert_counts[alert.level.value] += 1
        
        # Calculate uptime
        uptime = current_time - (self.performance_history[0].timestamp if self.performance_history else current_time)
        
        return {
            "timestamp": current_time,
            "uptime": uptime,
            "monitors_active": len([m for m in self.monitors.values() if m.is_running]),
            "total_monitors": len(self.monitors),
            "active_alerts": len(active_alerts),
            "alert_breakdown": dict(alert_counts),
            "latest_performance": latest_snapshot.__dict__ if latest_snapshot else None,
            "system_healthy": len(active_alerts) == 0 or all(
                alert.level in [AlertLevel.INFO, AlertLevel.WARNING] for alert in active_alerts
            )
        }
    
    def get_metrics_summary(self, duration: float = 3600) -> Dict[str, Any]:
        """Get summary of all metrics over a time period"""
        summary = {}
        
        for monitor_name, monitor in self.monitors.items():
            monitor_summary = {}
            
            for metric_name in monitor.metrics.keys():
                metric_summary = monitor.get_metric_summary(metric_name, duration)
                monitor_summary[metric_name] = metric_summary
            
            summary[monitor_name] = monitor_summary
        
        return summary
    
    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        resolved: Optional[bool] = None,
        limit: int = 100
    ) -> List[Alert]:
        """Get alerts with optional filtering"""
        
        filtered_alerts = self.all_alerts
        
        if level is not None:
            filtered_alerts = [alert for alert in filtered_alerts if alert.level == level]
        
        if resolved is not None:
            filtered_alerts = [alert for alert in filtered_alerts if alert.resolved == resolved]
        
        # Sort by timestamp (newest first) and limit
        filtered_alerts.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_alerts[:limit]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.all_alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = time.time()
                logger.info(f"Resolved alert: {alert_id}")
                return True
        
        return False
    
    async def _collect_performance_snapshots(self) -> None:
        """Collect periodic performance snapshots"""
        while self.is_running:
            try:
                snapshot = PerformanceSnapshot()
                
                # Collect data from monitors
                for monitor in self.monitors.values():
                    if monitor.name == "system":
                        cpu_metrics = monitor.get_latest_metrics("system_cpu_usage", 1)
                        memory_metrics = monitor.get_latest_metrics("system_memory_usage", 1)
                        
                        if cpu_metrics:
                            snapshot.cpu_usage = cpu_metrics[0].value
                        if memory_metrics:
                            snapshot.memory_usage = memory_metrics[0].value
                    
                    elif monitor.name == "agent":
                        agent_metrics = monitor.get_latest_metrics("agent_total_count", 1)
                        completed_metrics = monitor.get_latest_metrics("agent_tasks_completed_total", 1)
                        failed_metrics = monitor.get_latest_metrics("agent_tasks_failed_total", 1)
                        
                        if agent_metrics:
                            snapshot.active_agents = int(agent_metrics[0].value)
                        if completed_metrics:
                            snapshot.completed_tasks = int(completed_metrics[0].value)
                        if failed_metrics:
                            snapshot.failed_tasks = int(failed_metrics[0].value)
                    
                    elif monitor.name == "crew":
                        crew_metrics = monitor.get_latest_metrics("crew_total_count", 1)
                        
                        if crew_metrics:
                            snapshot.active_crews = int(crew_metrics[0].value)
                
                # Calculate derived metrics
                total_tasks = snapshot.completed_tasks + snapshot.failed_tasks
                if total_tasks > 0:
                    snapshot.error_rate = snapshot.failed_tasks / total_tasks
                
                self.performance_history.append(snapshot)
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Error collecting performance snapshot: {str(e)}")
                await asyncio.sleep(60)
    
    def _on_metric_collected(self, metric: Metric) -> None:
        """Handle metric collection"""
        self.aggregated_metrics[metric.name].append(metric)
        
        # Limit aggregated metric history
        if len(self.aggregated_metrics[metric.name]) > 10000:
            self.aggregated_metrics[metric.name] = self.aggregated_metrics[metric.name][-10000:]
    
    def _on_alert_triggered(self, alert: Alert) -> None:
        """Handle alert triggering"""
        self.all_alerts.append(alert)
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }.get(alert.level, logging.INFO)
        
        logger.log(log_level, f"Alert triggered: {alert.message} (Source: {alert.source})")
        
        # Limit alert history
        if len(self.all_alerts) > 10000:
            self.all_alerts = self.all_alerts[-10000:]
