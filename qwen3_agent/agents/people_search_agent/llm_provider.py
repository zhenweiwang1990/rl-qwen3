#!/usr/bin/env python3
"""LLM provider abstraction supporting Ollama and OpenRouter."""

import json
import os
import requests
from typing import List, Dict, Any
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Send chat request to LLM.
        
        Args:
            messages: List of message dicts with role and content
            tools: Optional list of tool schemas
            temperature: Sampling temperature
            
        Returns:
            Response dict with message content and tool calls in Ollama format
        """
        pass


class OllamaLLM(BaseLLM):
    """Ollama LLM client for local inference."""
    
    def __init__(
        self,
        model: str = "qwen3:14b",
        base_url: str = "http://127.0.0.1:11434",
        debug: bool = False,
    ):
        """Initialize Ollama client.
        
        Args:
            model: Model name (e.g., "qwen3:14b")
            base_url: Ollama API base URL
            debug: If True, print full request/response payloads
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/chat"
        self.debug = debug
        self._last_request = None
        self._last_response = None
        
    def chat(
        self, 
        messages: List[Dict[str, Any]], 
        tools: List[Dict[str, Any]] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Send chat request to Ollama.
        
        Args:
            messages: List of message dicts with role and content
            tools: Optional list of tool schemas
            temperature: Sampling temperature
            
        Returns:
            Response dict with message content and tool calls
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if tools:
            payload["tools"] = tools
        
        self._last_request = payload
        
        if self.debug:
            print("\n===== Ollama Request =====")
            print(json.dumps(payload, indent=2))
            print("==========================")
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            self._last_response = data
            
            if self.debug:
                print("\n===== Ollama Response =====")
                print(json.dumps(data, indent=2)[:20000])
                print("==========================")
            
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            raise


class OpenRouterLLM(BaseLLM):
    """OpenRouter LLM client using OpenAI API format."""
    
    def __init__(
        self,
        model: str,
        api_key: str = None,
        debug: bool = False,
        site_url: str = None,
        site_name: str = None,
    ):
        """Initialize OpenRouter client.
        
        Args:
            model: Model name (e.g., "qwen/qwen3-30b-a3b-instruct-2507")
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            debug: If True, print full request/response payloads
            site_url: Optional site URL for rankings
            site_name: Optional site name for rankings
        """
        from openai import OpenAI
        
        self.model = model
        self.debug = debug
        self.site_url = site_url or os.getenv("OPENROUTER_SITE_URL", "")
        self.site_name = site_name or os.getenv("OPENROUTER_SITE_NAME", "")
        
        # Get API key from env if not provided
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key parameter."
                )
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self._last_request = None
        self._last_response = None
    
    def _convert_tools_to_openai_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Ollama tool format to OpenAI tool format.
        
        Ollama format:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "...",
                "parameters": {...}
            }
        }
        
        OpenAI format is the same, so we can pass through.
        """
        return tools
    
    def _convert_messages_to_openai_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI format.
        
        Ollama uses 'tool' role, OpenAI uses 'tool' role too, so mostly compatible.
        """
        converted = []
        for msg in messages:
            new_msg = dict(msg)
            
            # OpenAI expects tool messages to have 'tool_call_id'
            if msg.get("role") == "tool":
                # If tool_call_id is missing, generate a simple one
                if "tool_call_id" not in new_msg:
                    new_msg["tool_call_id"] = f"call_{len(converted)}"
            
            converted.append(new_msg)
        
        return converted
    
    def _convert_response_to_ollama_format(self, response) -> Dict[str, Any]:
        """Convert OpenAI response to Ollama format.
        
        OpenAI response format:
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "...",
                    "tool_calls": [{
                        "id": "call_xxx",
                        "type": "function",
                        "function": {
                            "name": "tool_name",
                            "arguments": "{...}"  # JSON string
                        }
                    }]
                }
            }]
        }
        
        Ollama format:
        {
            "message": {
                "role": "assistant",
                "content": "...",
                "tool_calls": [{
                    "function": {
                        "name": "tool_name",
                        "arguments": {...}  # Can be dict or string
                    }
                }]
            }
        }
        """
        choice = response.choices[0]
        message = choice.message
        
        ollama_message = {
            "role": message.role,
            "content": message.content or "",
        }
        
        # Convert tool calls if present
        if message.tool_calls:
            ollama_tool_calls = []
            for tc in message.tool_calls:
                ollama_tc = {
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,  # Keep as string or parse to dict
                    }
                }
                # Store tool_call_id for response mapping
                if hasattr(tc, 'id'):
                    ollama_tc["id"] = tc.id
                ollama_tool_calls.append(ollama_tc)
            
            ollama_message["tool_calls"] = ollama_tool_calls
        
        return {
            "message": ollama_message,
            "done": True,
        }
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Send chat request to OpenRouter.
        
        Args:
            messages: List of message dicts with role and content
            tools: Optional list of tool schemas in Ollama format
            temperature: Sampling temperature
            
        Returns:
            Response dict in Ollama format
        """
        # Convert to OpenAI format
        openai_messages = self._convert_messages_to_openai_format(messages)
        
        # Build request
        request_kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
        }
        
        # Add extra headers if provided
        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra_headers["X-Title"] = self.site_name
        
        if extra_headers:
            request_kwargs["extra_headers"] = extra_headers
        
        # Add tools if provided
        if tools:
            openai_tools = self._convert_tools_to_openai_format(tools)
            request_kwargs["tools"] = openai_tools
        
        self._last_request = request_kwargs
        
        if self.debug:
            print("\n===== OpenRouter Request =====")
            print(json.dumps(request_kwargs, indent=2, default=str))
            print("==============================")
        
        try:
            response = self.client.chat.completions.create(**request_kwargs)
            
            if self.debug:
                print("\n===== OpenRouter Response =====")
                print(response.model_dump_json(indent=2)[:20000])
                print("================================")
            
            # Convert to Ollama format
            ollama_response = self._convert_response_to_ollama_format(response)
            self._last_response = ollama_response
            
            return ollama_response
            
        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
            raise


def create_llm(
    provider: str = "ollama",
    model: str = "qwen3:14b",
    debug: bool = False,
    **kwargs
) -> BaseLLM:
    """Factory function to create LLM client based on provider.
    
    Args:
        provider: "ollama" or "openrouter"
        model: Model name
        debug: Debug mode
        **kwargs: Additional provider-specific arguments
        
    Returns:
        LLM client instance
    """
    provider = provider.lower()
    
    if provider == "ollama":
        return OllamaLLM(
            model=model,
            base_url=kwargs.get("base_url", "http://127.0.0.1:11434"),
            debug=debug,
        )
    
    elif provider == "openrouter":
        # Map short model names to full OpenRouter model names
        model_mapping = {
            "qwen3-14b": "qwen/qwen-2.5-14b-instruct",
            "qwen3-30b-a3b-instruct-2507": "qwen/qwen3-30b-a3b-instruct-2507",
            "qwen2.5-72b": "qwen/qwen-2.5-72b-instruct",
            # Add more mappings as needed
        }
        
        # Use mapping if available, otherwise use model name as-is
        full_model_name = model_mapping.get(model, model)
        
        return OpenRouterLLM(
            model=full_model_name,
            api_key=kwargs.get("api_key"),
            debug=debug,
            site_url=kwargs.get("site_url"),
            site_name=kwargs.get("site_name"),
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'openrouter'.")

