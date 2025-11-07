"""Local model wrapper using transformers for Mac-compatible training."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing import List, Dict, Any, Optional
import json
from qwen3_agent.config import get_device
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalLLM:
    """Local LLM wrapper using transformers.
    
    This class provides a simple interface for loading and using language models
    locally with transformers, supporting CPU, MPS (Mac), and CUDA.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        max_length: int = 2048,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """Initialize the local LLM.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use (auto-detected if None)
            max_length: Maximum sequence length
            load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
        """
        self.model_name = model_name
        self.device = device or get_device()
        self.max_length = max_length
        
        logger.info(f"Loading model {model_name} on device {self.device}")
        
        # Load tokenizer
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate configuration
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
        }
        
        # Add quantization if requested (only works on CUDA)
        if self.device == "cuda":
            if load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif load_in_4bit:
                model_kwargs["load_in_4bit"] = True
        
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        
        # Move to device if not quantized
        if not (load_in_8bit or load_in_4bit):
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate text from messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
        
        Returns:
            Generated text string
        """
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        
        return generated_text
    
    def parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse tool call from model response.
        
        Args:
            response: Model response text
        
        Returns:
            Dictionary with 'tool_name' and 'tool_args', or None if parsing fails
        """
        # Try to find JSON in response
        start_idx = response.find("{")
        end_idx = response.rfind("}")
        
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return None
        
        json_str = response[start_idx:end_idx + 1]
        
        try:
            tool_call = json.loads(json_str)
            
            # Validate structure
            if "tool_name" in tool_call and "tool_args" in tool_call:
                return tool_call
            
            return None
        except json.JSONDecodeError:
            return None
    
    def generate_tool_call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ) -> Optional[Dict[str, Any]]:
        """Generate and parse a tool call.
        
        Args:
            messages: Conversation history
            temperature: Sampling temperature
        
        Returns:
            Parsed tool call or None if parsing fails
        """
        response = self.generate(
            messages,
            max_new_tokens=512,
            temperature=temperature,
            do_sample=True,
        )
        
        return self.parse_tool_call(response)
    
    def enable_training_mode(self):
        """Enable training mode for the model."""
        self.model.train()
    
    def enable_eval_mode(self):
        """Enable evaluation mode for the model."""
        self.model.eval()
    
    def get_trainable_parameters(self):
        """Get trainable parameters count."""
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        all_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percentage": 100 * trainable_params / all_params,
        }
    
    def save_model(self, output_dir: str):
        """Save model and tokenizer.
        
        Args:
            output_dir: Directory to save model
        """
        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Model saved successfully")
    
    def load_adapter(self, adapter_path: str):
        """Load a LoRA adapter.
        
        Args:
            adapter_path: Path to the adapter
        """
        from peft import PeftModel
        
        logger.info(f"Loading adapter from {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        logger.info("Adapter loaded successfully")


class LLMAgent:
    """Agent that uses LocalLLM to interact with the environment."""
    
    def __init__(
        self,
        llm: LocalLLM,
        verbose: bool = False,
    ):
        """Initialize the agent.
        
        Args:
            llm: LocalLLM instance
            verbose: Whether to print verbose logs
        """
        self.llm = llm
        self.verbose = verbose
    
    def act(
        self,
        observation: Dict[str, Any],
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate action from observation.
        
        Args:
            observation: Environment observation
            temperature: Sampling temperature
        
        Returns:
            Action dictionary with 'tool_name' and 'tool_args'
        """
        # Parse conversation from observation
        text = observation["text"]
        
        # Convert text back to messages
        messages = self._parse_text_to_messages(text)
        
        # Generate tool call
        tool_call = self.llm.generate_tool_call(messages, temperature=temperature)
        
        if tool_call is None:
            # Fallback: return a default action
            if self.verbose:
                logger.warning("Failed to parse tool call, using fallback")
            return {
                "tool_name": "return_final_answer",
                "tool_args": {"answer": "I don't know", "sources": []},
            }
        
        if self.verbose:
            logger.info(f"Generated action: {tool_call['tool_name']}")
        
        return tool_call
    
    def _parse_text_to_messages(self, text: str) -> List[Dict[str, str]]:
        """Parse observation text back to messages.
        
        Args:
            text: Observation text
        
        Returns:
            List of message dictionaries
        """
        messages = []
        lines = text.split("\n\n")
        
        for line in lines:
            if ": " not in line:
                continue
            
            role_end = line.index(": ")
            role = line[:role_end].lower()
            content = line[role_end + 2:]
            
            messages.append({"role": role, "content": content})
        
        return messages

