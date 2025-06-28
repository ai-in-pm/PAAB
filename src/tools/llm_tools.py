#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - LLM Tools
Tools for integrating with various Language Model providers
"""

import os
import json
from typing import Dict, Any, List, Optional, Union
from .base import BaseTool, ToolType, ToolParameter, ToolResult
import logging

logger = logging.getLogger(__name__)


class OpenAITool(BaseTool):
    """Tool for OpenAI API integration"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        super().__init__(
            name="openai_chat",
            description="Generate text using OpenAI's language models",
            tool_type=ToolType.LLM,
            parameters=[
                ToolParameter(
                    name="prompt",
                    type=str,
                    description="The prompt to send to the model",
                    required=True
                ),
                ToolParameter(
                    name="model",
                    type=str,
                    description="The model to use",
                    required=False,
                    default=model
                ),
                ToolParameter(
                    name="temperature",
                    type=float,
                    description="Sampling temperature (0-2)",
                    required=False,
                    default=0.7,
                    validation_rules={"min_value": 0, "max_value": 2}
                ),
                ToolParameter(
                    name="max_tokens",
                    type=int,
                    description="Maximum tokens to generate",
                    required=False,
                    default=1000,
                    validation_rules={"min_value": 1, "max_value": 4000}
                )
            ],
            return_type=str
        )
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.default_model = model
        
        if not self.api_key:
            logger.warning("OpenAI API key not provided. Tool will not function properly.")
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute OpenAI API call"""
        try:
            # Import OpenAI here to avoid dependency issues
            try:
                import openai
            except ImportError:
                return ToolResult(
                    success=False,
                    output=None,
                    error_message="OpenAI library not installed. Install with: pip install openai"
                )
            
            if not self.api_key:
                return ToolResult(
                    success=False,
                    output=None,
                    error_message="OpenAI API key not provided"
                )
            
            # Set up client
            client = openai.OpenAI(api_key=self.api_key)
            
            # Extract parameters
            prompt = kwargs.get("prompt")
            model = kwargs.get("model", self.default_model)
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1000)
            
            # Make API call
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract response
            content = response.choices[0].message.content
            
            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "model": model,
                    "usage": response.usage.dict() if response.usage else {},
                    "finish_reason": response.choices[0].finish_reason
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            return ToolResult(
                success=False,
                output=None,
                error_message=str(e)
            )


class AnthropicTool(BaseTool):
    """Tool for Anthropic Claude API integration"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        super().__init__(
            name="anthropic_chat",
            description="Generate text using Anthropic's Claude models",
            tool_type=ToolType.LLM,
            parameters=[
                ToolParameter(
                    name="prompt",
                    type=str,
                    description="The prompt to send to Claude",
                    required=True
                ),
                ToolParameter(
                    name="model",
                    type=str,
                    description="The Claude model to use",
                    required=False,
                    default=model
                ),
                ToolParameter(
                    name="temperature",
                    type=float,
                    description="Sampling temperature (0-1)",
                    required=False,
                    default=0.7,
                    validation_rules={"min_value": 0, "max_value": 1}
                ),
                ToolParameter(
                    name="max_tokens",
                    type=int,
                    description="Maximum tokens to generate",
                    required=False,
                    default=1000,
                    validation_rules={"min_value": 1, "max_value": 4000}
                )
            ],
            return_type=str
        )
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.default_model = model
        
        if not self.api_key:
            logger.warning("Anthropic API key not provided. Tool will not function properly.")
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute Anthropic API call"""
        try:
            # Import Anthropic here to avoid dependency issues
            try:
                import anthropic
            except ImportError:
                return ToolResult(
                    success=False,
                    output=None,
                    error_message="Anthropic library not installed. Install with: pip install anthropic"
                )
            
            if not self.api_key:
                return ToolResult(
                    success=False,
                    output=None,
                    error_message="Anthropic API key not provided"
                )
            
            # Set up client
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Extract parameters
            prompt = kwargs.get("prompt")
            model = kwargs.get("model", self.default_model)
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1000)
            
            # Make API call
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract response
            content = response.content[0].text
            
            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "model": model,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    },
                    "stop_reason": response.stop_reason
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {str(e)}")
            return ToolResult(
                success=False,
                output=None,
                error_message=str(e)
            )


class LocalLLMTool(BaseTool):
    """Tool for local LLM integration (e.g., Ollama, local transformers)"""
    
    def __init__(self, model_name: str = "llama2", endpoint: str = "http://localhost:11434"):
        super().__init__(
            name="local_llm",
            description="Generate text using local language models",
            tool_type=ToolType.LLM,
            parameters=[
                ToolParameter(
                    name="prompt",
                    type=str,
                    description="The prompt to send to the model",
                    required=True
                ),
                ToolParameter(
                    name="model",
                    type=str,
                    description="The local model to use",
                    required=False,
                    default=model_name
                ),
                ToolParameter(
                    name="temperature",
                    type=float,
                    description="Sampling temperature",
                    required=False,
                    default=0.7
                ),
                ToolParameter(
                    name="max_tokens",
                    type=int,
                    description="Maximum tokens to generate",
                    required=False,
                    default=1000
                )
            ],
            return_type=str
        )
        
        self.model_name = model_name
        self.endpoint = endpoint
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute local LLM call"""
        try:
            import requests
            
            # Extract parameters
            prompt = kwargs.get("prompt")
            model = kwargs.get("model", self.model_name)
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1000)
            
            # Prepare request for Ollama-style API
            payload = {
                "model": model,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                "stream": False
            }
            
            # Make API call
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")
                
                return ToolResult(
                    success=True,
                    output=content,
                    metadata={
                        "model": model,
                        "endpoint": self.endpoint,
                        "done": result.get("done", False)
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error_message=f"Local LLM API error: {response.status_code} - {response.text}"
                )
                
        except Exception as e:
            logger.error(f"Local LLM call failed: {str(e)}")
            return ToolResult(
                success=False,
                output=None,
                error_message=str(e)
            )


class EmbeddingTool(BaseTool):
    """Tool for generating text embeddings"""
    
    def __init__(self, provider: str = "openai", model: str = "text-embedding-ada-002"):
        super().__init__(
            name="text_embedding",
            description="Generate embeddings for text using various providers",
            tool_type=ToolType.LLM,
            parameters=[
                ToolParameter(
                    name="text",
                    type=str,
                    description="Text to generate embeddings for",
                    required=True
                ),
                ToolParameter(
                    name="provider",
                    type=str,
                    description="Embedding provider (openai, sentence-transformers)",
                    required=False,
                    default=provider
                ),
                ToolParameter(
                    name="model",
                    type=str,
                    description="Embedding model to use",
                    required=False,
                    default=model
                )
            ],
            return_type=list
        )
        
        self.provider = provider
        self.model = model
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute embedding generation"""
        try:
            text = kwargs.get("text")
            provider = kwargs.get("provider", self.provider)
            model = kwargs.get("model", self.model)
            
            if provider == "openai":
                return self._openai_embedding(text, model)
            elif provider == "sentence-transformers":
                return self._sentence_transformers_embedding(text, model)
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error_message=f"Unsupported embedding provider: {provider}"
                )
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return ToolResult(
                success=False,
                output=None,
                error_message=str(e)
            )
    
    def _openai_embedding(self, text: str, model: str) -> ToolResult:
        """Generate OpenAI embeddings"""
        try:
            import openai
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return ToolResult(
                    success=False,
                    output=None,
                    error_message="OpenAI API key not provided"
                )
            
            client = openai.OpenAI(api_key=api_key)
            
            response = client.embeddings.create(
                model=model,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            return ToolResult(
                success=True,
                output=embedding,
                metadata={
                    "model": model,
                    "provider": "openai",
                    "dimension": len(embedding),
                    "usage": response.usage.dict() if response.usage else {}
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error_message=f"OpenAI embedding failed: {str(e)}"
            )
    
    def _sentence_transformers_embedding(self, text: str, model: str) -> ToolResult:
        """Generate sentence-transformers embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load model
            embedding_model = SentenceTransformer(model)
            
            # Generate embedding
            embedding = embedding_model.encode(text).tolist()
            
            return ToolResult(
                success=True,
                output=embedding,
                metadata={
                    "model": model,
                    "provider": "sentence-transformers",
                    "dimension": len(embedding)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error_message=f"Sentence-transformers embedding failed: {str(e)}"
            )


# Factory function to create LLM tools
def create_llm_tool(provider: str, **kwargs) -> BaseTool:
    """
    Factory function to create LLM tools
    
    Args:
        provider: LLM provider (openai, anthropic, local, embedding)
        **kwargs: Additional arguments for the tool
        
    Returns:
        BaseTool instance
    """
    if provider == "openai":
        return OpenAITool(**kwargs)
    elif provider == "anthropic":
        return AnthropicTool(**kwargs)
    elif provider == "local":
        return LocalLLMTool(**kwargs)
    elif provider == "embedding":
        return EmbeddingTool(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# Pre-configured tool instances
def get_default_llm_tools() -> List[BaseTool]:
    """Get a list of default LLM tools"""
    tools = []
    
    # Add OpenAI tool if API key is available
    if os.getenv("OPENAI_API_KEY"):
        tools.append(OpenAITool())
        tools.append(EmbeddingTool(provider="openai"))
    
    # Add Anthropic tool if API key is available
    if os.getenv("ANTHROPIC_API_KEY"):
        tools.append(AnthropicTool())
    
    # Add local LLM tool (always available, but may not work without local setup)
    tools.append(LocalLLMTool())
    
    # Add sentence-transformers embedding tool
    try:
        import sentence_transformers
        tools.append(EmbeddingTool(provider="sentence-transformers", model="all-MiniLM-L6-v2"))
    except ImportError:
        logger.info("sentence-transformers not available, skipping embedding tool")
    
    return tools
