"""Training script for Qwen3 email agent.

This is a simplified standalone training script that doesn't require the full ART library.
It provides basic training functionality for the email agent.
"""

import asyncio
from dotenv import load_dotenv
from typing import List
import os

from qwen3_agent.rollout import rollout
from qwen3_agent.data import load_synthetic_queries
from qwen3_agent.data.types import SyntheticQuery
from qwen3_agent.core import TrainableModel, gather_trajectory_groups, TrajectoryGroup
from qwen3_agent.utils import iterate_dataset
from qwen3_agent.config import PolicyConfig, TrainingConfig, get_device
from qwen3_agent.benchmark import benchmark_model

load_dotenv()


def create_trainable_model(
    run_id: str = "001",
    base_model: str = "Qwen/Qwen2.5-14B-Instruct",
    training_config: TrainingConfig | None = None,
    max_turns: int = 10,
    verbose: bool = False,
) -> TrainableModel:
    """Create a trainable model configuration. Returns TrainableModel instance.
    
    Args:
        run_id: Unique identifier for this training run
        base_model: Base model to use (default: Qwen2.5-14B-Instruct)
        training_config: Training hyperparameters
        max_turns: Maximum turns for the agent
        verbose: Whether to print verbose logs
    """
    if training_config is None:
        training_config = TrainingConfig()
    
    # For standalone operation, we use LiteLLM for inference
    return TrainableModel(
        name=f"qwen3-email-agent-{run_id}",
        project="qwen3_email_agent",
        base_model=base_model,
        config=PolicyConfig(
            max_turns=max_turns,
            verbose=verbose,
            training_config=training_config,
        ),
        # Set up inference via LiteLLM
        inference_api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        inference_base_url=os.getenv("INFERENCE_BASE_URL", "http://localhost:8000/v1"),
        inference_model_name=base_model,
    )


async def run_training(
    model: TrainableModel,
    verbose: bool = False,
):
    """Run the training loop.
    
    NOTE: This is a simplified standalone version that focuses on data generation
    and evaluation. For full training with gradient updates, you would need to
    integrate with a training backend like vLLM + LoRA fine-tuning.
    
    Args:
        model: The trainable model to train
        verbose: Whether to print verbose logs
    """
    assert isinstance(model.config, PolicyConfig)
    if model.config.training_config is None:
        raise ValueError("Training config is not set")
    
    training_config = model.config.training_config
    device = get_device()
    
    if verbose or model.config.verbose:
        print(f"\n{'='*60}")
        print(f"Starting training: {model.name}")
        print(f"Base model: {model.base_model}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")
        print("Training Configuration:")
        print(f"  - Trajectories per group: {training_config.trajectories_per_group}")
        print(f"  - Groups per step: {training_config.groups_per_step}")
        print(f"  - Learning rate: {training_config.learning_rate}")
        print(f"  - Eval steps: {training_config.eval_steps}")
        print(f"  - Training dataset size: {training_config.training_dataset_size}")
        print(f"  - Validation set size: {training_config.val_set_size}")
        print(f"  - Number of epochs: {training_config.num_epochs}")
        print(f"{'='*60}\n")
        print("\nNOTE: This is a simplified standalone training script.")
        print("For full RL fine-tuning, you need to set up a training backend.")
        print("This script will generate training data and run evaluations.\n")

    # Load training and validation data
    if verbose or model.config.verbose:
        print("\nLoading training data...")
    train_scenarios: List[SyntheticQuery] = load_synthetic_queries(
        split="train", limit=training_config.training_dataset_size
    )
    
    if verbose or model.config.verbose:
        print("Loading validation data...")
    val_scenarios: List[SyntheticQuery] = load_synthetic_queries(
        split="test", limit=training_config.val_set_size
    )

    if verbose or model.config.verbose:
        print(f"\nDataset sizes:")
        print(f"  - Training: {len(train_scenarios)}")
        print(f"  - Validation: {len(val_scenarios)}")
        print(f"{'='*60}\n")

    # Training loop
    current_step = 0
    train_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=training_config.groups_per_step,
        num_epochs=training_config.num_epochs,
        initial_step=current_step,
    )

    for batch, epoch, global_step, epoch_step in train_iterator:
        if verbose or model.config.verbose:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{training_config.num_epochs}, "
                  f"Step {epoch_step}, Global Step {global_step}")
            print(f"{'='*60}")
        
        # Evaluation
        if global_step % training_config.eval_steps == 0:
            if verbose or model.config.verbose:
                print(f"\n--- Evaluating at Global Step {global_step} ---")
            
            await benchmark_model(model, limit=training_config.val_set_size, verbose=verbose or model.config.verbose)

        # Generate trajectories for training
        if verbose or model.config.verbose:
            print(f"\nGenerating {len(batch)} trajectory groups...")
        
        groups = await gather_trajectory_groups(
            (
                TrajectoryGroup(
                    (
                        rollout(model, scenario)
                        for _ in range(training_config.trajectories_per_group)
                    )
                )
                for scenario in batch
            )
        )

        # If verbose, surface any exceptions collected for better debugging
        if verbose or model.config.verbose:
            total_exceptions = sum(len(group.exceptions) for group in groups)
            if total_exceptions:
                print(f"\nEncountered {total_exceptions} exceptions during trajectory generation:")
                for gi, group in enumerate(groups):
                    for ei, ex in enumerate(group.exceptions[:3]):
                        # Print a few examples per group to avoid spam
                        print("-" * 80)
                        print(f"[Group {gi}] Exception {ei + 1}: {ex.type}")
                        print(f"Message: {ex.message}")
                        print(ex.traceback)

        # In a full implementation, you would train on these trajectories here
        # For now, we just log the statistics
        if verbose or model.config.verbose:
            total_trajectories = sum(len(group) for group in groups)
            avg_reward = sum(
                t.reward for group in groups for t in group
            ) / max(total_trajectories, 1)
            print(f"Generated {total_trajectories} trajectories")
            print(f"Average reward: {avg_reward:.3f}")
            print(f"\nNOTE: Training step skipped - implement gradient updates for full training")
            print(f"Completed step {global_step}")

    # Final evaluation
    if verbose or model.config.verbose:
        print(f"\n{'='*60}")
        print("Training loop complete! Running final evaluation...")
        print(f"{'='*60}\n")
    
    await benchmark_model(model, limit=training_config.val_set_size, verbose=verbose or model.config.verbose)
    
    if verbose or model.config.verbose:
        print(f"\n{'='*60}")
        print(f"Training finished for {model.name}")
        print(f"{'='*60}\n")


async def main():
    """Main training entry point."""
    # Get configuration from environment
    run_id = os.environ.get("RUN_ID", "001")
    verbose = os.environ.get("VERBOSE", "false").lower() == "true"
    base_model = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct")
    
    # Create training config from environment
    training_config = TrainingConfig(
        trajectories_per_group=int(os.environ.get("TRAJECTORIES_PER_GROUP", "6")),
        groups_per_step=int(os.environ.get("GROUPS_PER_STEP", "8")),
        learning_rate=float(os.environ.get("LEARNING_RATE", "1.2e-5")),
        eval_steps=int(os.environ.get("EVAL_STEPS", "30")),
        val_set_size=int(os.environ.get("VAL_SET_SIZE", "100")),
        training_dataset_size=int(os.environ.get("TRAINING_DATASET_SIZE", "4000")),
        num_epochs=int(os.environ.get("NUM_EPOCHS", "4")),
    )
    
    max_turns = int(os.environ.get("MAX_TURNS", "10"))
    
    # Create model
    model = create_trainable_model(
        run_id=run_id,
        base_model=base_model,
        training_config=training_config,
        max_turns=max_turns,
        verbose=verbose,
    )
    
    # Run training
    await run_training(model, verbose=verbose)


if __name__ == "__main__":
    asyncio.run(main())
