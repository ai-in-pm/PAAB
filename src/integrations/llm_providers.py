#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - LLM Provider Integrations
Comprehensive integration with various LLM providers
"""

import os
import json
import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    COHERE = "cohere"
    REPLICATE = "replicate"


class ModelCapability(Enum):
    """Model capabilities"""
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    IMAGE_ANALYSIS = "image_analysis"
    CODE_GENERATION = "code_generation"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"


@dataclass
class ModelInfo:
    """Information about a language model"""
    name: str
    provider: LLMProvider
    capabilities: List[ModelCapability]
    max_tokens: int
    context_window: int
    cost_per_1k_tokens: Optional[float] = None
    supports_system_message: bool = True
    supports_streaming: bool = False
    supports_function_calling: bool = False
    description: str = ""


@dataclass
class LLMRequest:
    """Request to an LLM"""
    messages: List[Dict[str, str]]
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None


@dataclass
class LLMResponse:
    """Response from an LLM"""
    content: str
    model: str
    provider: LLMProvider
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    function_call: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3
    ):
        """
        Initialize LLM provider
        
        Args:
            api_key: API key for the provider
            base_url: Base URL for API calls
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }
    
    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models"""
        pass
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream response from LLM"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics"""
        return self.stats.copy()


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                logger.error("OpenAI library not installed. Install with: pip install openai")
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get OpenAI models"""
        return [
            ModelInfo(
                name="gpt-4",
                provider=LLMProvider.OPENAI,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING
                ],
                max_tokens=4096,
                context_window=8192,
                cost_per_1k_tokens=0.03,
                supports_function_calling=True,
                supports_streaming=True,
                description="Most capable GPT-4 model"
            ),
            ModelInfo(
                name="gpt-4-turbo",
                provider=LLMProvider.OPENAI,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING
                ],
                max_tokens=4096,
                context_window=128000,
                cost_per_1k_tokens=0.01,
                supports_function_calling=True,
                supports_streaming=True,
                description="Latest GPT-4 Turbo model with larger context"
            ),
            ModelInfo(
                name="gpt-3.5-turbo",
                provider=LLMProvider.OPENAI,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING
                ],
                max_tokens=4096,
                context_window=16384,
                cost_per_1k_tokens=0.001,
                supports_function_calling=True,
                supports_streaming=True,
                description="Fast and efficient GPT-3.5 model"
            )
        ]
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI"""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            self.stats["total_requests"] += 1
            
            # Prepare request
            kwargs = {
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty
            }
            
            if request.stop:
                kwargs["stop"] = request.stop
            
            if request.functions:
                kwargs["functions"] = request.functions
                if request.function_call:
                    kwargs["function_call"] = request.function_call
            
            # Make API call
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **kwargs
            )
            
            # Extract response
            message = response.choices[0].message
            content = message.content or ""
            
            # Handle function calls
            function_call = None
            if hasattr(message, 'function_call') and message.function_call:
                function_call = {
                    "name": message.function_call.name,
                    "arguments": message.function_call.arguments
                }
            
            # Update statistics
            self.stats["successful_requests"] += 1
            if response.usage:
                self.stats["total_tokens"] += response.usage.total_tokens
                
                # Estimate cost (rough calculation)
                model_info = next((m for m in self.get_available_models() if m.name == request.model), None)
                if model_info and model_info.cost_per_1k_tokens:
                    cost = (response.usage.total_tokens / 1000) * model_info.cost_per_1k_tokens
                    self.stats["total_cost"] += cost
            
            return LLMResponse(
                content=content,
                model=request.model,
                provider=LLMProvider.OPENAI,
                usage=response.usage.dict() if response.usage else {},
                finish_reason=response.choices[0].finish_reason,
                function_call=function_call,
                metadata={"response_id": response.id}
            )
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI"""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            # Prepare streaming request
            kwargs = {
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": True
            }
            
            # Make streaming API call
            stream = await asyncio.to_thread(
                self.client.chat.completions.create,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {str(e)}")
            raise


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.error("Anthropic library not installed. Install with: pip install anthropic")
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get Anthropic models"""
        return [
            ModelInfo(
                name="claude-3-opus-20240229",
                provider=LLMProvider.ANTHROPIC,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.IMAGE_ANALYSIS,
                    ModelCapability.STREAMING
                ],
                max_tokens=4096,
                context_window=200000,
                cost_per_1k_tokens=0.015,
                supports_streaming=True,
                description="Most capable Claude model"
            ),
            ModelInfo(
                name="claude-3-sonnet-20240229",
                provider=LLMProvider.ANTHROPIC,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.IMAGE_ANALYSIS,
                    ModelCapability.STREAMING
                ],
                max_tokens=4096,
                context_window=200000,
                cost_per_1k_tokens=0.003,
                supports_streaming=True,
                description="Balanced Claude model"
            ),
            ModelInfo(
                name="claude-3-haiku-20240307",
                provider=LLMProvider.ANTHROPIC,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.STREAMING
                ],
                max_tokens=4096,
                context_window=200000,
                cost_per_1k_tokens=0.00025,
                supports_streaming=True,
                description="Fast and efficient Claude model"
            )
        ]
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Anthropic"""
        if not self.client:
            raise ValueError("Anthropic client not initialized")
        
        try:
            self.stats["total_requests"] += 1
            
            # Convert messages format
            system_message = ""
            messages = []
            
            for msg in request.messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    messages.append(msg)
            
            # Prepare request
            kwargs = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature
            }
            
            if system_message:
                kwargs["system"] = system_message
            
            if request.stop:
                kwargs["stop_sequences"] = request.stop
            
            # Make API call
            response = await asyncio.to_thread(
                self.client.messages.create,
                **kwargs
            )
            
            # Extract response
            content = response.content[0].text
            
            # Update statistics
            self.stats["successful_requests"] += 1
            if response.usage:
                total_tokens = response.usage.input_tokens + response.usage.output_tokens
                self.stats["total_tokens"] += total_tokens
                
                # Estimate cost
                model_info = next((m for m in self.get_available_models() if m.name == request.model), None)
                if model_info and model_info.cost_per_1k_tokens:
                    cost = (total_tokens / 1000) * model_info.cost_per_1k_tokens
                    self.stats["total_cost"] += cost
            
            return LLMResponse(
                content=content,
                model=request.model,
                provider=LLMProvider.ANTHROPIC,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                } if response.usage else {},
                finish_reason=response.stop_reason,
                metadata={"response_id": response.id}
            )
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"Anthropic API call failed: {str(e)}")
            raise
    
    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream response from Anthropic"""
        if not self.client:
            raise ValueError("Anthropic client not initialized")
        
        try:
            # Convert messages format
            system_message = ""
            messages = []
            
            for msg in request.messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    messages.append(msg)
            
            # Prepare streaming request
            kwargs = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": True
            }
            
            if system_message:
                kwargs["system"] = system_message
            
            # Make streaming API call
            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming failed: {str(e)}")
            raise


class LLMProviderManager:
    """Manager for multiple LLM providers"""
    
    def __init__(self):
        """Initialize provider manager"""
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self.default_provider = None
        
        # Auto-initialize available providers
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize available providers based on API keys"""
        
        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.providers[LLMProvider.OPENAI] = OpenAIProvider()
                if not self.default_provider:
                    self.default_provider = LLMProvider.OPENAI
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI provider: {str(e)}")
        
        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.providers[LLMProvider.ANTHROPIC] = AnthropicProvider()
                if not self.default_provider:
                    self.default_provider = LLMProvider.ANTHROPIC
                logger.info("Anthropic provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic provider: {str(e)}")
    
    def add_provider(self, provider_type: LLMProvider, provider: BaseLLMProvider) -> None:
        """Add a provider"""
        self.providers[provider_type] = provider
        if not self.default_provider:
            self.default_provider = provider_type
        logger.info(f"Added {provider_type.value} provider")
    
    def get_provider(self, provider_type: Optional[LLMProvider] = None) -> BaseLLMProvider:
        """Get a provider"""
        if provider_type is None:
            provider_type = self.default_provider
        
        if provider_type not in self.providers:
            raise ValueError(f"Provider {provider_type.value} not available")
        
        return self.providers[provider_type]
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    def get_all_models(self) -> List[ModelInfo]:
        """Get all available models from all providers"""
        models = []
        for provider in self.providers.values():
            models.extend(provider.get_available_models())
        return models
    
    def get_models_by_capability(self, capability: ModelCapability) -> List[ModelInfo]:
        """Get models that support a specific capability"""
        models = []
        for model in self.get_all_models():
            if capability in model.capabilities:
                models.append(model)
        return models
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        provider: Optional[LLMProvider] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using specified or default provider"""
        
        # Determine provider from model name if not specified
        if provider is None:
            for model_info in self.get_all_models():
                if model_info.name == model:
                    provider = model_info.provider
                    break
            
            if provider is None:
                provider = self.default_provider
        
        if provider not in self.providers:
            raise ValueError(f"Provider {provider.value} not available")
        
        # Create request
        request = LLMRequest(
            messages=messages,
            model=model,
            **kwargs
        )
        
        # Generate response
        return await self.providers[provider].generate(request)
    
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        provider: Optional[LLMProvider] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response using specified or default provider"""
        
        # Determine provider from model name if not specified
        if provider is None:
            for model_info in self.get_all_models():
                if model_info.name == model:
                    provider = model_info.provider
                    break
            
            if provider is None:
                provider = self.default_provider
        
        if provider not in self.providers:
            raise ValueError(f"Provider {provider.value} not available")
        
        # Create request
        request = LLMRequest(
            messages=messages,
            model=model,
            stream=True,
            **kwargs
        )
        
        # Stream response
        async for chunk in self.providers[provider].stream_generate(request):
            yield chunk
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all providers"""
        combined_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "providers": {}
        }
        
        for provider_type, provider in self.providers.items():
            stats = provider.get_stats()
            combined_stats["total_requests"] += stats["total_requests"]
            combined_stats["successful_requests"] += stats["successful_requests"]
            combined_stats["failed_requests"] += stats["failed_requests"]
            combined_stats["total_tokens"] += stats["total_tokens"]
            combined_stats["total_cost"] += stats["total_cost"]
            combined_stats["providers"][provider_type.value] = stats
        
        return combined_stats


# Global provider manager instance
llm_manager = LLMProviderManager()
