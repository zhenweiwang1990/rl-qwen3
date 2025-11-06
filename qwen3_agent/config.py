"""Configuration classes for Qwen3 Agent training and evaluation."""

from pydantic import BaseModel
from typing import Optional
import os
import torch


class TrainingConfig(BaseModel):
    """Training hyperparameters configuration."""
    
    trajectories_per_group: int = 6
    groups_per_step: int = 8
    learning_rate: float = 1.2e-5
    eval_steps: int = 30
    val_set_size: int = 100
    training_dataset_size: int = 4000
    num_epochs: int = 4


class PolicyConfig(BaseModel):
    """Policy configuration for the email agent."""
    
    max_turns: int = 10
    max_tokens: int = 2048
    litellm_model_name: Optional[str] = None
    use_tools: bool = True
    stupid_simple_reward_fn: bool = False
    verbose: bool = False
    
    training_config: Optional[TrainingConfig] = None


def get_device() -> str:
    """Auto-detect the best available device (CUDA, MPS, or CPU)."""
    device = os.environ.get("DEVICE")
    if device:
        return device
    
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_training_config_from_env() -> TrainingConfig:
    """Load training configuration from environment variables."""
    return TrainingConfig(
        trajectories_per_group=int(os.environ.get("TRAJECTORIES_PER_GROUP", "6")),
        groups_per_step=int(os.environ.get("GROUPS_PER_STEP", "8")),
        learning_rate=float(os.environ.get("LEARNING_RATE", "1.2e-5")),
        eval_steps=int(os.environ.get("EVAL_STEPS", "30")),
        val_set_size=int(os.environ.get("VAL_SET_SIZE", "100")),
        training_dataset_size=int(os.environ.get("TRAINING_DATASET_SIZE", "4000")),
        num_epochs=int(os.environ.get("NUM_EPOCHS", "4")),
    )


def get_policy_config_from_env() -> PolicyConfig:
    """Load policy configuration from environment variables."""
    training_config = get_training_config_from_env()
    verbose = os.environ.get("VERBOSE", "false").lower() == "true"
    
    return PolicyConfig(
        max_turns=int(os.environ.get("MAX_TURNS", "10")),
        max_tokens=int(os.environ.get("MAX_TOKENS", "2048")),
        use_tools=True,
        verbose=verbose,
        training_config=training_config,
    )


def create_model_config(
    run_id: str = "001",
    max_turns: int = 10,
    custom_training_config: Optional[TrainingConfig] = None,
) -> PolicyConfig:
    """Create a model configuration with specific parameters. Returns PolicyConfig with the specified parameters.
    
    Args:
        run_id: Unique identifier for this run
        max_turns: Maximum number of turns for the agent
        custom_training_config: Custom training configuration (optional)
    """
    if custom_training_config is None:
        custom_training_config = TrainingConfig()
    
    verbose = os.environ.get("VERBOSE", "false").lower() == "true"
    
    return PolicyConfig(
        max_turns=max_turns,
        max_tokens=2048,
        use_tools=True,
        verbose=verbose,
        training_config=custom_training_config,
    )

