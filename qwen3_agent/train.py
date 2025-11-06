"""Training script for Qwen3 email agent with full RL gradient updates.

This script uses the OpenPipe ART library for complete reinforcement learning training,
including gradient updates, checkpoint management, and model optimization.
"""

import art
import asyncio
from dotenv import load_dotenv
from typing import List
import os

from qwen3_agent.rollout import rollout
from qwen3_agent.data import load_synthetic_queries
from qwen3_agent.data.types import SyntheticQuery
from qwen3_agent.data.local_email_db import generate_database
from art.utils import iterate_dataset
from qwen3_agent.config import PolicyConfig, TrainingConfig
from qwen3_agent.benchmark import benchmark_model

load_dotenv()


def create_trainable_model(
    run_id: str = "001",
    base_model: str = "OpenPipe/Qwen3-14B-Instruct",
    training_config: TrainingConfig | None = None,
    max_turns: int = 10,
    verbose: bool = False,
) -> art.TrainableModel:
    """Create a trainable model configuration using ART framework.
    
    Args:
        run_id: Unique identifier for this training run
        base_model: Base model to use (default: OpenPipe/Qwen3-14B-Instruct)
        training_config: Training hyperparameters
        max_turns: Maximum turns for the agent
        verbose: Whether to print verbose logs
        
    Returns:
        art.TrainableModel instance ready for training
    """
    if training_config is None:
        training_config = TrainingConfig()
    
    return art.TrainableModel(
        name=f"qwen3-email-agent-{run_id}",
        project="qwen3_email_agent",
        base_model=base_model,
        config=PolicyConfig(
            max_turns=max_turns,
            verbose=verbose,
            training_config=training_config,
        ),
    )


async def run_training(model: art.TrainableModel, verbose: bool = False):
    """Run the full training loop with gradient updates.
    
    This function implements a complete RL training pipeline:
    1. Generate email database if needed
    2. Initialize ART LocalAPI
    3. Load/restore model from S3 if available
    4. Load training and validation data
    5. Training loop:
       - Run evaluation at specified intervals
       - Generate trajectory groups via rollout
       - Train model with gradient updates (PPO/GRPO)
       - Save checkpoints and sync to S3
    
    Args:
        model: The trainable model to train
        verbose: Whether to print verbose logs
    """
    assert isinstance(model.config, PolicyConfig)
    if model.config.training_config is None:
        raise ValueError("Training config is not set")
    
    training_config = model.config.training_config
    
    # Generate database if it doesn't exist
    if verbose or model.config.verbose:
        print("Checking email database...")
    generate_database()
    
    # Initialize ART LocalAPI
    api = art.LocalAPI()
    await model.register(api)
    
    if verbose or model.config.verbose:
        print(f"\n{'='*60}")
        print(f"Starting training: {model.name}")
        print(f"Base model: {model.base_model}")
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
    
    # Pull from S3 if backup bucket is configured
    if backup_bucket := os.environ.get("BACKUP_BUCKET"):
        if verbose or model.config.verbose:
            print(f"Pulling existing checkpoints from S3 bucket: `{backup_bucket}`")
        try:
            await api._experimental_pull_from_s3(
                model,
                s3_bucket=backup_bucket,
                verbose=verbose or model.config.verbose,
            )
            if verbose or model.config.verbose:
                print("✓ Successfully restored from S3")
        except Exception as e:
            if verbose or model.config.verbose:
                print(f"⚠ Could not pull from S3 (may be first run): {e}")
    
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
    current_step = await model.get_step()
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
        
        # Evaluation at specified intervals
        if global_step % training_config.eval_steps == 0:
            if verbose or model.config.verbose:
                print(f"\n--- Evaluating at Global Step {global_step} ---")
            
            eval_results = await benchmark_model(
                model, 
                limit=training_config.val_set_size, 
                verbose=verbose or model.config.verbose
            )
            
            # Log evaluation results to ART
            await model.log([art.Trajectory(
                messages_and_choices=[],
                reward=eval_results["reward"][0] if "reward" in eval_results else 0.0,
                metrics={k: v[0] for k, v in eval_results.to_dict().items() if k != "reward"}
            )], split="val")
            
            # Clean up old checkpoints (keep best and latest)
            await model.delete_checkpoints(best_checkpoint_metric="val/reward")
            
            # Push to S3 if configured
            if backup_bucket := os.environ.get("BACKUP_BUCKET"):
                if verbose or model.config.verbose:
                    print(f"Backing up to S3 bucket: {backup_bucket}")
                await api._experimental_push_to_s3(
                    model,
                    s3_bucket=backup_bucket,
                )

        # Generate trajectories for training
        if verbose or model.config.verbose:
            print(f"\nGenerating {len(batch)} trajectory groups...")
        
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    (
                        rollout(model, scenario)
                        for _ in range(training_config.trajectories_per_group)
                    )
                )
                for scenario in batch
            )
        )

        # Log trajectory statistics
        if verbose or model.config.verbose:
            total_exceptions = sum(len(group.exceptions) for group in groups)
            if total_exceptions:
                print(f"\nEncountered {total_exceptions} exceptions during trajectory generation:")
                for gi, group in enumerate(groups):
                    for ei, ex in enumerate(group.exceptions[:3]):
                        print("-" * 80)
                        print(f"[Group {gi}] Exception {ei + 1}: {ex.type}")
                        print(f"Message: {ex.message}")
                        print(ex.traceback[:500])  # Truncate long tracebacks

        total_trajectories = sum(len(group) for group in groups)
        avg_reward = sum(
            t.reward for group in groups for t in group
        ) / max(total_trajectories, 1) if total_trajectories > 0 else 0.0
        
        if verbose or model.config.verbose:
            print(f"\nGenerated {total_trajectories} trajectories")
            print(f"Average reward: {avg_reward:.3f}")
        
        # ⭐ THIS IS THE KEY PART - Actually train the model with gradient updates!
        if verbose or model.config.verbose:
            print(f"\n🔥 Training model with gradient updates...")
        
        await model.train(
            groups,
            config=art.TrainConfig(
                learning_rate=training_config.learning_rate
            ),
        )
        
        if verbose or model.config.verbose:
            print(f"✓ Completed training step {global_step}")

    # Final evaluation
    if verbose or model.config.verbose:
        print(f"\n{'='*60}")
        print("Training loop complete! Running final evaluation...")
        print(f"{'='*60}\n")
    
    final_eval_results = await benchmark_model(
        model, 
        limit=training_config.val_set_size, 
        verbose=verbose or model.config.verbose
    )
    
    # Log final evaluation
    await model.log([art.Trajectory(
        messages_and_choices=[],
        reward=final_eval_results["reward"][0] if "reward" in final_eval_results else 0.0,
        metrics={k: v[0] for k, v in final_eval_results.to_dict().items() if k != "reward"}
    )], split="val")
    
    # Final S3 push
    if backup_bucket := os.environ.get("BACKUP_BUCKET"):
        if verbose or model.config.verbose:
            print(f"Final backup to S3 bucket: {backup_bucket}")
        await api._experimental_push_to_s3(
            model,
            s3_bucket=backup_bucket,
        )
    
    if verbose or model.config.verbose:
        print(f"\n{'='*60}")
        print(f"🎉 Training finished for {model.name}")
        print(f"{'='*60}\n")


async def main():
    """Main training entry point."""
    # Get configuration from environment
    run_id = os.environ.get("RUN_ID", "001")
    verbose = os.environ.get("VERBOSE", "false").lower() == "true"
    base_model = os.environ.get("MODEL_NAME", "OpenPipe/Qwen3-14B-Instruct")
    
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
