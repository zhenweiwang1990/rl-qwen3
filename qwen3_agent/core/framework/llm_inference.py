"""Unified LLM inference interface for both training and external models."""

from typing import Any, Dict, List, Optional, Union
from litellm import acompletion
from litellm.types.utils import ModelResponse
import art
import os


class LLMInference:
    """Unified LLM inference interface for both training and external models.
    
    This class provides a consistent interface for calling LLMs regardless of whether
    they are ART models (during training) or external models (for comparison).
    
    Key features:
    - Automatically detects model type (ART trainable, ART frozen, or external)
    - Configures LiteLLM parameters correctly
    - Handles caching appropriately (disabled for trainable models)
    - Supports both native tool calling and text-based tool calling
    
    Example:
        ```python
        # With ART model
        llm = LLMInference(art_model)
        response = await llm.complete(messages, tools=tools)
        
        # With external model
        llm = LLMInference("openai/gpt-4", {"api_key": "..."})
        response = await llm.complete(messages, tools=tools)
        ```
    """
    
    def __init__(
        self,
        model: Union[art.Model, art.TrainableModel, str],
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize LLM inference.
        
        Args:
            model: ART model instance or model name string (e.g., "openai/gpt-4")
            model_config: Optional config for external models (api_key, base_url, etc.)
        """
        self.model = model
        self.model_config = model_config or {}
        
        # Detect model type and configure
        if isinstance(model, (art.Model, art.TrainableModel)):
            self.is_art_model = True
            self.model_name = self._get_litellm_name(model)
            self.base_url = getattr(model, 'inference_base_url', None) or getattr(model, 'base_url', None)
            self.api_key = getattr(model, 'inference_api_key', None) or getattr(model, 'api_key', None)
            self.is_trainable = getattr(model, 'trainable', False)
        else:
            self.is_art_model = False
            self.model_name = model
            self.base_url = model_config.get('base_url')
            self.api_key = model_config.get('api_key')
            self.is_trainable = False
    
    def _get_litellm_name(self, model: Union[art.Model, art.TrainableModel]) -> str:
        """Get LiteLLM-compatible model name from ART model.
        
        Args:
            model: ART model instance
            
        Returns:
            LiteLLM model name string
        """
        # Check if model config specifies a custom litellm name
        if hasattr(model, 'config') and hasattr(model.config, 'litellm_model_name'):
            if model.config.litellm_model_name:
                return model.config.litellm_model_name
        
        # Detect based on model attributes
        base_url = getattr(model, 'inference_base_url', None) or getattr(model, 'base_url', None)
        if base_url:
            if hasattr(model, 'trainable') and model.trainable:
                return f"hosted_vllm/{model.name}"
            else:
                return f"openai/{model.get_inference_name()}"
        
        # Fallback
        return f"hosted_vllm/{model.name}"
    
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> ModelResponse:
        """Generate completion from LLM.
        
        Args:
            messages: Conversation messages in OpenAI format
            tools: Optional tool definitions for function calling
            max_tokens: Maximum completion tokens
            temperature: Sampling temperature
            **kwargs: Additional arguments for litellm.acompletion
            
        Returns:
            ModelResponse from LiteLLM
            
        Raises:
            Exception: If LLM call fails
        """
        completion_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "caching": not self.is_trainable,  # Disable caching for trainable models
            # Configurable timeout and retries
            "timeout": int(os.environ.get("LITELLM_TIMEOUT", "60")),
            "max_retries": int(os.environ.get("LITELLM_MAX_RETRIES", "0")),
            **kwargs
        }
        
        # Add authentication if available
        if self.base_url:
            completion_kwargs["base_url"] = self.base_url
        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        
        # Add tools if provided
        if tools:
            completion_kwargs["tools"] = tools
        
        return await acompletion(**completion_kwargs)  # type: ignore

