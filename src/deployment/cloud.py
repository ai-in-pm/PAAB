#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Cloud Deployment System
Cloud deployment and scaling capabilities
"""

import os
import json
import yaml
import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    LOCAL = "local"


class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    SCALING = "scaling"
    UPDATING = "updating"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    name: str
    provider: CloudProvider
    region: str = "us-east-1"
    instance_type: str = "medium"
    min_instances: int = 1
    max_instances: int = 10
    auto_scaling: bool = True
    environment_variables: Dict[str, str] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    networking: Dict[str, Any] = field(default_factory=dict)
    storage: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentInfo:
    """Deployment information"""
    id: str
    name: str
    status: DeploymentStatus
    provider: CloudProvider
    endpoint: Optional[str] = None
    instances: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseCloudProvider(ABC):
    """Base class for cloud providers"""
    
    def __init__(self, provider_type: CloudProvider, credentials: Optional[Dict[str, str]] = None):
        """
        Initialize cloud provider
        
        Args:
            provider_type: Type of cloud provider
            credentials: Provider credentials
        """
        self.provider_type = provider_type
        self.credentials = credentials or {}
        self.deployments: Dict[str, DeploymentInfo] = {}
        
        logger.info(f"Cloud provider {provider_type.value} initialized")
    
    @abstractmethod
    async def deploy(self, config: DeploymentConfig) -> DeploymentInfo:
        """Deploy application to cloud"""
        pass
    
    @abstractmethod
    async def scale(self, deployment_id: str, instances: int) -> bool:
        """Scale deployment"""
        pass
    
    @abstractmethod
    async def update(self, deployment_id: str, config: DeploymentConfig) -> bool:
        """Update deployment"""
        pass
    
    @abstractmethod
    async def stop(self, deployment_id: str) -> bool:
        """Stop deployment"""
        pass
    
    @abstractmethod
    async def get_status(self, deployment_id: str) -> DeploymentInfo:
        """Get deployment status"""
        pass
    
    @abstractmethod
    async def get_logs(self, deployment_id: str, lines: int = 100) -> List[str]:
        """Get deployment logs"""
        pass
    
    def list_deployments(self) -> List[DeploymentInfo]:
        """List all deployments"""
        return list(self.deployments.values())


class AWSProvider(BaseCloudProvider):
    """AWS cloud provider implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(CloudProvider.AWS, **kwargs)
        self.session = None
        self.ecs_client = None
        self.ec2_client = None
        
        # Initialize AWS clients
        self._initialize_aws_clients()
    
    def _initialize_aws_clients(self) -> None:
        """Initialize AWS clients"""
        try:
            import boto3
            
            # Create session with credentials
            session_kwargs = {}
            if self.credentials.get("access_key_id"):
                session_kwargs["aws_access_key_id"] = self.credentials["access_key_id"]
            if self.credentials.get("secret_access_key"):
                session_kwargs["aws_secret_access_key"] = self.credentials["secret_access_key"]
            if self.credentials.get("region"):
                session_kwargs["region_name"] = self.credentials["region"]
            
            self.session = boto3.Session(**session_kwargs)
            self.ecs_client = self.session.client('ecs')
            self.ec2_client = self.session.client('ec2')
            
            logger.info("AWS clients initialized successfully")
            
        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {str(e)}")
    
    async def deploy(self, config: DeploymentConfig) -> DeploymentInfo:
        """Deploy to AWS ECS"""
        if not self.ecs_client:
            raise RuntimeError("AWS clients not initialized")
        
        try:
            deployment_id = f"paab-{config.name}-{int(time.time())}"
            
            # Create task definition
            task_definition = await self._create_task_definition(config, deployment_id)
            
            # Create service
            service = await self._create_service(config, deployment_id, task_definition)
            
            # Create deployment info
            deployment_info = DeploymentInfo(
                id=deployment_id,
                name=config.name,
                status=DeploymentStatus.DEPLOYING,
                provider=CloudProvider.AWS,
                metadata={
                    "task_definition": task_definition,
                    "service": service,
                    "cluster": config.metadata.get("cluster", "default")
                }
            )
            
            self.deployments[deployment_id] = deployment_info
            
            logger.info(f"Started AWS deployment: {deployment_id}")
            return deployment_info
            
        except Exception as e:
            logger.error(f"AWS deployment failed: {str(e)}")
            raise
    
    async def scale(self, deployment_id: str, instances: int) -> bool:
        """Scale ECS service"""
        if deployment_id not in self.deployments:
            return False
        
        try:
            deployment = self.deployments[deployment_id]
            cluster = deployment.metadata.get("cluster", "default")
            service_name = deployment.metadata["service"]["serviceName"]
            
            # Update service desired count
            response = await asyncio.to_thread(
                self.ecs_client.update_service,
                cluster=cluster,
                service=service_name,
                desiredCount=instances
            )
            
            deployment.status = DeploymentStatus.SCALING
            deployment.updated_at = time.time()
            
            logger.info(f"Scaled AWS deployment {deployment_id} to {instances} instances")
            return True
            
        except Exception as e:
            logger.error(f"AWS scaling failed: {str(e)}")
            return False
    
    async def update(self, deployment_id: str, config: DeploymentConfig) -> bool:
        """Update ECS service"""
        if deployment_id not in self.deployments:
            return False
        
        try:
            # Create new task definition
            new_task_definition = await self._create_task_definition(config, deployment_id)
            
            deployment = self.deployments[deployment_id]
            cluster = deployment.metadata.get("cluster", "default")
            service_name = deployment.metadata["service"]["serviceName"]
            
            # Update service with new task definition
            response = await asyncio.to_thread(
                self.ecs_client.update_service,
                cluster=cluster,
                service=service_name,
                taskDefinition=new_task_definition["taskDefinitionArn"]
            )
            
            deployment.status = DeploymentStatus.UPDATING
            deployment.updated_at = time.time()
            deployment.metadata["task_definition"] = new_task_definition
            
            logger.info(f"Updated AWS deployment: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"AWS update failed: {str(e)}")
            return False
    
    async def stop(self, deployment_id: str) -> bool:
        """Stop ECS service"""
        if deployment_id not in self.deployments:
            return False
        
        try:
            deployment = self.deployments[deployment_id]
            cluster = deployment.metadata.get("cluster", "default")
            service_name = deployment.metadata["service"]["serviceName"]
            
            # Delete service
            await asyncio.to_thread(
                self.ecs_client.delete_service,
                cluster=cluster,
                service=service_name,
                force=True
            )
            
            deployment.status = DeploymentStatus.STOPPED
            deployment.updated_at = time.time()
            
            logger.info(f"Stopped AWS deployment: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"AWS stop failed: {str(e)}")
            return False
    
    async def get_status(self, deployment_id: str) -> DeploymentInfo:
        """Get ECS service status"""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.deployments[deployment_id]
        
        try:
            cluster = deployment.metadata.get("cluster", "default")
            service_name = deployment.metadata["service"]["serviceName"]
            
            # Get service status
            response = await asyncio.to_thread(
                self.ecs_client.describe_services,
                cluster=cluster,
                services=[service_name]
            )
            
            if response["services"]:
                service = response["services"][0]
                
                # Update deployment status
                if service["status"] == "ACTIVE":
                    if service["runningCount"] > 0:
                        deployment.status = DeploymentStatus.RUNNING
                    else:
                        deployment.status = DeploymentStatus.PENDING
                else:
                    deployment.status = DeploymentStatus.FAILED
                
                # Update instance information
                deployment.instances = [
                    {
                        "id": task["taskArn"].split("/")[-1],
                        "status": task["lastStatus"],
                        "health": task.get("healthStatus", "UNKNOWN")
                    }
                    for task in service.get("tasks", [])
                ]
            
            deployment.updated_at = time.time()
            
        except Exception as e:
            logger.error(f"Failed to get AWS status: {str(e)}")
            deployment.status = DeploymentStatus.FAILED
        
        return deployment
    
    async def get_logs(self, deployment_id: str, lines: int = 100) -> List[str]:
        """Get CloudWatch logs"""
        if deployment_id not in self.deployments:
            return []
        
        try:
            # This would integrate with CloudWatch Logs
            # For now, return placeholder logs
            return [
                f"[{time.ctime()}] AWS deployment {deployment_id} log entry {i}"
                for i in range(min(lines, 10))
            ]
            
        except Exception as e:
            logger.error(f"Failed to get AWS logs: {str(e)}")
            return []
    
    async def _create_task_definition(self, config: DeploymentConfig, deployment_id: str) -> Dict[str, Any]:
        """Create ECS task definition"""
        
        # Build container definition
        container_definition = {
            "name": f"paab-{config.name}",
            "image": config.metadata.get("image", "paab:latest"),
            "memory": config.resource_limits.get("memory", 512),
            "cpu": config.resource_limits.get("cpu", 256),
            "essential": True,
            "environment": [
                {"name": key, "value": value}
                for key, value in config.environment_variables.items()
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": f"/ecs/paab-{config.name}",
                    "awslogs-region": config.region,
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
        
        # Add port mappings if specified
        if config.networking.get("ports"):
            container_definition["portMappings"] = [
                {
                    "containerPort": port,
                    "protocol": "tcp"
                }
                for port in config.networking["ports"]
            ]
        
        # Register task definition
        response = await asyncio.to_thread(
            self.ecs_client.register_task_definition,
            family=f"paab-{config.name}",
            containerDefinitions=[container_definition],
            requiresCompatibilities=["FARGATE"],
            networkMode="awsvpc",
            cpu=str(config.resource_limits.get("cpu", 256)),
            memory=str(config.resource_limits.get("memory", 512)),
            executionRoleArn=config.security.get("execution_role_arn")
        )
        
        return response["taskDefinition"]
    
    async def _create_service(self, config: DeploymentConfig, deployment_id: str, task_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Create ECS service"""
        
        service_config = {
            "serviceName": f"paab-{config.name}-service",
            "cluster": config.metadata.get("cluster", "default"),
            "taskDefinition": task_definition["taskDefinitionArn"],
            "desiredCount": config.min_instances,
            "launchType": "FARGATE",
            "networkConfiguration": {
                "awsvpcConfiguration": {
                    "subnets": config.networking.get("subnets", []),
                    "securityGroups": config.networking.get("security_groups", []),
                    "assignPublicIp": "ENABLED" if config.networking.get("public", True) else "DISABLED"
                }
            }
        }
        
        # Add load balancer configuration if specified
        if config.networking.get("load_balancer"):
            service_config["loadBalancers"] = [config.networking["load_balancer"]]
        
        # Add auto scaling if enabled
        if config.auto_scaling:
            service_config["enableAutoScaling"] = True
        
        response = await asyncio.to_thread(
            self.ecs_client.create_service,
            **service_config
        )
        
        return response["service"]


class DockerProvider(BaseCloudProvider):
    """Docker provider implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(CloudProvider.DOCKER, **kwargs)
        self.docker_client = None
        
        # Initialize Docker client
        self._initialize_docker_client()
    
    def _initialize_docker_client(self) -> None:
        """Initialize Docker client"""
        try:
            import docker
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
            
        except ImportError:
            logger.error("docker library not installed. Install with: pip install docker")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {str(e)}")
    
    async def deploy(self, config: DeploymentConfig) -> DeploymentInfo:
        """Deploy using Docker"""
        if not self.docker_client:
            raise RuntimeError("Docker client not initialized")
        
        try:
            deployment_id = f"paab-{config.name}-{int(time.time())}"
            
            # Build container configuration
            container_config = {
                "image": config.metadata.get("image", "paab:latest"),
                "name": f"paab-{config.name}",
                "environment": config.environment_variables,
                "detach": True,
                "restart_policy": {"Name": "unless-stopped"}
            }
            
            # Add port mappings
            if config.networking.get("ports"):
                container_config["ports"] = {
                    f"{port}/tcp": port for port in config.networking["ports"]
                }
            
            # Add volume mounts
            if config.storage.get("volumes"):
                container_config["volumes"] = config.storage["volumes"]
            
            # Add resource limits
            if config.resource_limits:
                container_config["mem_limit"] = config.resource_limits.get("memory", "512m")
                container_config["cpu_quota"] = config.resource_limits.get("cpu_quota", 50000)
            
            # Run container
            container = await asyncio.to_thread(
                self.docker_client.containers.run,
                **container_config
            )
            
            # Create deployment info
            deployment_info = DeploymentInfo(
                id=deployment_id,
                name=config.name,
                status=DeploymentStatus.RUNNING,
                provider=CloudProvider.DOCKER,
                endpoint=f"http://localhost:{config.networking.get('ports', [8080])[0]}" if config.networking.get("ports") else None,
                instances=[{
                    "id": container.id,
                    "name": container.name,
                    "status": container.status
                }],
                metadata={
                    "container_id": container.id,
                    "container_name": container.name
                }
            )
            
            self.deployments[deployment_id] = deployment_info
            
            logger.info(f"Started Docker deployment: {deployment_id}")
            return deployment_info
            
        except Exception as e:
            logger.error(f"Docker deployment failed: {str(e)}")
            raise
    
    async def scale(self, deployment_id: str, instances: int) -> bool:
        """Scale Docker deployment (using Docker Swarm or Compose)"""
        # For single container Docker, scaling would require Docker Swarm
        # This is a simplified implementation
        logger.warning("Docker scaling requires Docker Swarm mode")
        return False
    
    async def update(self, deployment_id: str, config: DeploymentConfig) -> bool:
        """Update Docker deployment"""
        if deployment_id not in self.deployments:
            return False
        
        try:
            # Stop current container
            await self.stop(deployment_id)
            
            # Deploy new version
            new_deployment = await self.deploy(config)
            
            # Update deployment ID mapping
            self.deployments[deployment_id] = new_deployment
            self.deployments[deployment_id].id = deployment_id
            
            logger.info(f"Updated Docker deployment: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Docker update failed: {str(e)}")
            return False
    
    async def stop(self, deployment_id: str) -> bool:
        """Stop Docker container"""
        if deployment_id not in self.deployments:
            return False
        
        try:
            deployment = self.deployments[deployment_id]
            container_id = deployment.metadata["container_id"]
            
            # Get and stop container
            container = self.docker_client.containers.get(container_id)
            await asyncio.to_thread(container.stop)
            await asyncio.to_thread(container.remove)
            
            deployment.status = DeploymentStatus.STOPPED
            deployment.updated_at = time.time()
            
            logger.info(f"Stopped Docker deployment: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Docker stop failed: {str(e)}")
            return False
    
    async def get_status(self, deployment_id: str) -> DeploymentInfo:
        """Get Docker container status"""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.deployments[deployment_id]
        
        try:
            container_id = deployment.metadata["container_id"]
            container = self.docker_client.containers.get(container_id)
            
            # Update status
            container.reload()
            if container.status == "running":
                deployment.status = DeploymentStatus.RUNNING
            elif container.status == "exited":
                deployment.status = DeploymentStatus.STOPPED
            else:
                deployment.status = DeploymentStatus.FAILED
            
            # Update instance info
            deployment.instances = [{
                "id": container.id,
                "name": container.name,
                "status": container.status,
                "created": container.attrs["Created"],
                "image": container.image.tags[0] if container.image.tags else "unknown"
            }]
            
            deployment.updated_at = time.time()
            
        except Exception as e:
            logger.error(f"Failed to get Docker status: {str(e)}")
            deployment.status = DeploymentStatus.FAILED
        
        return deployment
    
    async def get_logs(self, deployment_id: str, lines: int = 100) -> List[str]:
        """Get Docker container logs"""
        if deployment_id not in self.deployments:
            return []
        
        try:
            deployment = self.deployments[deployment_id]
            container_id = deployment.metadata["container_id"]
            
            container = self.docker_client.containers.get(container_id)
            logs = await asyncio.to_thread(
                container.logs,
                tail=lines,
                timestamps=True
            )
            
            return logs.decode('utf-8').split('\n')
            
        except Exception as e:
            logger.error(f"Failed to get Docker logs: {str(e)}")
            return []


class CloudDeploymentManager:
    """Manager for cloud deployments"""
    
    def __init__(self):
        """Initialize deployment manager"""
        self.providers: Dict[CloudProvider, BaseCloudProvider] = {}
        self.default_provider = None
        
        # Auto-initialize available providers
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize available cloud providers"""
        
        # Docker (always available if Docker is installed)
        try:
            docker_provider = DockerProvider()
            self.providers[CloudProvider.DOCKER] = docker_provider
            if not self.default_provider:
                self.default_provider = CloudProvider.DOCKER
            logger.info("Docker provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Docker provider: {str(e)}")
        
        # AWS (if credentials available)
        if os.getenv("AWS_ACCESS_KEY_ID") or os.path.exists(os.path.expanduser("~/.aws/credentials")):
            try:
                aws_provider = AWSProvider(credentials={
                    "access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                    "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                    "region": os.getenv("AWS_DEFAULT_REGION", "us-east-1")
                })
                self.providers[CloudProvider.AWS] = aws_provider
                if not self.default_provider:
                    self.default_provider = CloudProvider.AWS
                logger.info("AWS provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize AWS provider: {str(e)}")
    
    def add_provider(self, provider: BaseCloudProvider) -> None:
        """Add a cloud provider"""
        self.providers[provider.provider_type] = provider
        if not self.default_provider:
            self.default_provider = provider.provider_type
        logger.info(f"Added {provider.provider_type.value} provider")
    
    def get_provider(self, provider_type: Optional[CloudProvider] = None) -> BaseCloudProvider:
        """Get a cloud provider"""
        if provider_type is None:
            provider_type = self.default_provider
        
        if provider_type not in self.providers:
            raise ValueError(f"Provider {provider_type.value} not available")
        
        return self.providers[provider_type]
    
    def get_available_providers(self) -> List[CloudProvider]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    async def deploy(
        self,
        config: DeploymentConfig,
        provider: Optional[CloudProvider] = None
    ) -> DeploymentInfo:
        """Deploy to cloud"""
        cloud_provider = self.get_provider(provider)
        return await cloud_provider.deploy(config)
    
    async def scale(
        self,
        deployment_id: str,
        instances: int,
        provider: Optional[CloudProvider] = None
    ) -> bool:
        """Scale deployment"""
        cloud_provider = self.get_provider(provider)
        return await cloud_provider.scale(deployment_id, instances)
    
    async def update(
        self,
        deployment_id: str,
        config: DeploymentConfig,
        provider: Optional[CloudProvider] = None
    ) -> bool:
        """Update deployment"""
        cloud_provider = self.get_provider(provider)
        return await cloud_provider.update(deployment_id, config)
    
    async def stop(
        self,
        deployment_id: str,
        provider: Optional[CloudProvider] = None
    ) -> bool:
        """Stop deployment"""
        cloud_provider = self.get_provider(provider)
        return await cloud_provider.stop(deployment_id)
    
    async def get_status(
        self,
        deployment_id: str,
        provider: Optional[CloudProvider] = None
    ) -> DeploymentInfo:
        """Get deployment status"""
        cloud_provider = self.get_provider(provider)
        return await cloud_provider.get_status(deployment_id)
    
    async def get_logs(
        self,
        deployment_id: str,
        lines: int = 100,
        provider: Optional[CloudProvider] = None
    ) -> List[str]:
        """Get deployment logs"""
        cloud_provider = self.get_provider(provider)
        return await cloud_provider.get_logs(deployment_id, lines)
    
    def list_all_deployments(self) -> Dict[CloudProvider, List[DeploymentInfo]]:
        """List all deployments across all providers"""
        all_deployments = {}
        
        for provider_type, provider in self.providers.items():
            all_deployments[provider_type] = provider.list_deployments()
        
        return all_deployments
    
    def create_deployment_config(
        self,
        name: str,
        provider: CloudProvider = CloudProvider.DOCKER,
        **kwargs
    ) -> DeploymentConfig:
        """Create a deployment configuration"""
        
        # Default configurations by provider
        defaults = {
            CloudProvider.DOCKER: {
                "instance_type": "container",
                "region": "local",
                "resource_limits": {"memory": "512m", "cpu_quota": 50000},
                "networking": {"ports": [8080]},
                "metadata": {"image": "paab:latest"}
            },
            CloudProvider.AWS: {
                "instance_type": "t3.medium",
                "region": "us-east-1",
                "resource_limits": {"memory": 512, "cpu": 256},
                "networking": {"ports": [8080], "public": True},
                "metadata": {"cluster": "default", "image": "paab:latest"}
            }
        }
        
        # Merge defaults with provided kwargs
        config_data = defaults.get(provider, {})
        config_data.update(kwargs)
        
        return DeploymentConfig(
            name=name,
            provider=provider,
            **config_data
        )


# Global deployment manager instance
deployment_manager = CloudDeploymentManager()
