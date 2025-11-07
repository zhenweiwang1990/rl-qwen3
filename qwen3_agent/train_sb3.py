"""Training script for Qwen3 email agent using local training on Mac.

This script uses:
- Local transformers models (no vLLM)
- LoRA for efficient fine-tuning
- REINFORCE algorithm for RL training
- Gymnasium environment for task definition
- Compatible with Mac (MPS/CPU) and Linux (CUDA)
"""

import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
from tqdm import tqdm
from dotenv import load_dotenv

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from qwen3_agent.env import EmailSearchEnv
from qwen3_agent.local_model import LocalLLM, LLMAgent
from qwen3_agent.data import load_synthetic_queries
from qwen3_agent.data.types import SyntheticQuery
from qwen3_agent.data.local_email_db import generate_database
from qwen3_agent.config import TrainingConfig, get_device
import wandb

load_dotenv()


class REINFORCETrainer:
    """REINFORCE algorithm trainer for LLM agent.
    
    This implements a simple policy gradient method (REINFORCE) that's
    suitable for training LLMs on structured tasks.
    """
    
    def __init__(
        self,
        llm: LocalLLM,
        env: EmailSearchEnv,
        learning_rate: float = 1e-5,
        gamma: float = 0.99,
        use_wandb: bool = True,
        verbose: bool = False,
    ):
        """Initialize the trainer.
        
        Args:
            llm: Local LLM instance
            env: Email search environment
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            use_wandb: Whether to log to wandb
            verbose: Whether to print verbose logs
        """
        self.llm = llm
        self.env = env
        self.gamma = gamma
        self.use_wandb = use_wandb
        self.verbose = verbose
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.llm.model.parameters(),
            lr=learning_rate,
        )
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        
    def collect_trajectory(
        self,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Collect a single trajectory.
        
        Args:
            temperature: Sampling temperature
        
        Returns:
            Dictionary with trajectory data
        """
        observation, info = self.env.reset()
        done = False
        
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        agent = LLMAgent(self.llm, verbose=self.verbose)
        
        while not done:
            # Get action from agent
            action = agent.act(observation, temperature=temperature)
            
            # Store state and action
            states.append(observation)
            actions.append(action)
            
            # Compute log probability (needed for REINFORCE)
            # For simplicity, we'll compute this during the backward pass
            # Here we'll store the conversation state
            log_probs.append(None)  # Placeholder
            
            # Take step in environment
            observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            rewards.append(reward)
        
        # Compute discounted returns
        returns = self._compute_returns(rewards)
        
        trajectory = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "returns": returns,
            "log_probs": log_probs,
            "final_reward": sum(rewards),
            "episode_length": len(rewards),
            "info": info,
        }
        
        return trajectory
    
    def _compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns.
        
        Args:
            rewards: List of rewards
        
        Returns:
            List of discounted returns
        """
        returns = []
        G = 0
        
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def train_step(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Perform one training step.
        
        Args:
            trajectories: List of collected trajectories
        
        Returns:
            Dictionary with training metrics
        """
        self.llm.enable_training_mode()
        
        total_loss = 0.0
        num_steps = 0
        
        for traj in trajectories:
            for state, action, return_val in zip(
                traj["states"],
                traj["actions"],
                traj["returns"],
            ):
                # Reconstruct messages from state
                messages = self._state_to_messages(state)
                
                # Create prompt for tool call
                prompt = self.llm.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                
                # Create target (the action)
                target_text = json.dumps({
                    "tool_name": action["tool_name"],
                    "tool_args": action["tool_args"],
                })
                
                # Tokenize
                inputs = self.llm.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.llm.max_length,
                ).to(self.llm.device)
                
                target_tokens = self.llm.tokenizer(
                    target_text,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(self.llm.device)
                
                # Forward pass
                outputs = self.llm.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                )
                
                logits = outputs.logits
                
                # Compute log probability of the action
                # Take the last logit position and compute cross-entropy with target
                last_logits = logits[:, -1, :]
                target_id = target_tokens.input_ids[:, 0]
                
                log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)
                action_log_prob = log_probs[0, target_id]
                
                # REINFORCE loss: -log_prob * return
                loss = -action_log_prob * return_val
                
                # Backward pass
                loss.backward()
                
                total_loss += loss.item()
                num_steps += 1
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.llm.model.parameters(), 1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.llm.enable_eval_mode()
        
        avg_loss = total_loss / max(num_steps, 1)
        
        return {
            "loss": avg_loss,
            "num_steps": num_steps,
        }
    
    def _state_to_messages(self, state: Dict[str, Any]) -> List[Dict[str, str]]:
        """Convert state observation to messages.
        
        Args:
            state: State observation
        
        Returns:
            List of message dictionaries
        """
        text = state["text"]
        messages = []
        
        for line in text.split("\n\n"):
            if ": " not in line:
                continue
            
            role_end = line.index(": ")
            role = line[:role_end].lower()
            content = line[role_end + 2:]
            
            messages.append({"role": role, "content": content})
        
        return messages
    
    def evaluate(
        self,
        num_episodes: int = 10,
    ) -> Dict[str, float]:
        """Evaluate the agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.llm.enable_eval_mode()
        
        rewards = []
        correct_answers = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            trajectory = self.collect_trajectory(temperature=0.3)
            
            rewards.append(trajectory["final_reward"])
            episode_lengths.append(trajectory["episode_length"])
            
            # Check if answer was correct
            rubric = trajectory["info"]["rubric"]
            correct_answers.append(rubric["answer_correct"])
        
        return {
            "eval/mean_reward": np.mean(rewards),
            "eval/std_reward": np.std(rewards),
            "eval/mean_length": np.mean(episode_lengths),
            "eval/accuracy": np.mean(correct_answers),
        }


def train(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "./output",
    training_config: Optional[TrainingConfig] = None,
    max_turns: int = 10,
    num_episodes: int = 1000,
    eval_frequency: int = 50,
    save_frequency: int = 100,
    batch_size: int = 4,
    use_lora: bool = True,
    verbose: bool = False,
):
    """Main training function.
    
    Args:
        model_name: HuggingFace model name
        output_dir: Output directory for checkpoints
        training_config: Training configuration
        max_turns: Maximum turns per episode
        num_episodes: Number of training episodes
        eval_frequency: Evaluate every N episodes
        save_frequency: Save checkpoint every N episodes
        batch_size: Number of trajectories to collect per batch
        use_lora: Whether to use LoRA
        verbose: Whether to print verbose logs
    """
    if training_config is None:
        training_config = TrainingConfig()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    use_wandb = os.environ.get("USE_WANDB", "true").lower() == "true"
    if use_wandb:
        wandb.init(
            project="qwen3-email-agent-sb3",
            name=f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "model_name": model_name,
                "max_turns": max_turns,
                "num_episodes": num_episodes,
                "batch_size": batch_size,
                "use_lora": use_lora,
                "learning_rate": training_config.learning_rate,
            },
        )
    
    # Generate database
    print("Generating email database...")
    generate_database()
    
    # Load data
    print("Loading training data...")
    train_scenarios: List[SyntheticQuery] = load_synthetic_queries(
        split="train",
        limit=training_config.training_dataset_size,
    )
    
    print("Loading validation data...")
    val_scenarios: List[SyntheticQuery] = load_synthetic_queries(
        split="test",
        limit=training_config.val_set_size,
    )
    
    print(f"Training scenarios: {len(train_scenarios)}")
    print(f"Validation scenarios: {len(val_scenarios)}")
    
    # Create environment
    env = EmailSearchEnv(
        scenarios=train_scenarios,
        max_turns=max_turns,
        verbose=verbose,
    )
    
    # Load model
    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")
    
    llm = LocalLLM(
        model_name=model_name,
        device=device,
        load_in_8bit=False,
        load_in_4bit=False,
    )
    
    # Apply LoRA if requested
    if use_lora:
        print("Applying LoRA configuration...")
        
        # Prepare model for k-bit training if quantized
        # llm.model = prepare_model_for_kbit_training(llm.model)
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        llm.model = get_peft_model(llm.model, lora_config)
        llm.model.print_trainable_parameters()
    
    # Create trainer
    trainer = REINFORCETrainer(
        llm=llm,
        env=env,
        learning_rate=training_config.learning_rate,
        use_wandb=use_wandb,
        verbose=verbose,
    )
    
    # Training loop
    print("\nStarting training...")
    print(f"{'='*60}")
    
    episode = 0
    pbar = tqdm(total=num_episodes, desc="Training")
    
    while episode < num_episodes:
        # Collect batch of trajectories
        trajectories = []
        batch_rewards = []
        
        for _ in range(batch_size):
            traj = trainer.collect_trajectory(temperature=0.7)
            trajectories.append(traj)
            batch_rewards.append(traj["final_reward"])
            episode += 1
            
            pbar.update(1)
        
        # Training step
        metrics = trainer.train_step(trajectories)
        metrics["train/mean_reward"] = np.mean(batch_rewards)
        metrics["train/episode"] = episode
        
        # Log to wandb
        if use_wandb:
            wandb.log(metrics)
        
        # Evaluation
        if episode % eval_frequency == 0:
            print(f"\n{'='*60}")
            print(f"Evaluating at episode {episode}...")
            
            # Switch to validation scenarios
            env.scenarios = val_scenarios
            eval_metrics = trainer.evaluate(num_episodes=10)
            env.scenarios = train_scenarios  # Switch back
            
            print(f"Eval mean reward: {eval_metrics['eval/mean_reward']:.3f}")
            print(f"Eval accuracy: {eval_metrics['eval/accuracy']:.3f}")
            print(f"{'='*60}\n")
            
            if use_wandb:
                wandb.log(eval_metrics)
        
        # Save checkpoint
        if episode % save_frequency == 0:
            checkpoint_dir = output_path / f"checkpoint-{episode}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            print(f"Saving checkpoint to {checkpoint_dir}...")
            llm.save_model(str(checkpoint_dir))
    
    pbar.close()
    
    # Final evaluation
    print("\n" + "="*60)
    print("Training complete! Running final evaluation...")
    
    env.scenarios = val_scenarios
    final_metrics = trainer.evaluate(num_episodes=training_config.val_set_size)
    
    print(f"\nFinal Results:")
    print(f"Mean reward: {final_metrics['eval/mean_reward']:.3f}")
    print(f"Accuracy: {final_metrics['eval/accuracy']:.3f}")
    print("="*60 + "\n")
    
    if use_wandb:
        wandb.log(final_metrics)
        wandb.finish()
    
    # Save final model
    final_model_dir = output_path / "final_model"
    final_model_dir.mkdir(exist_ok=True)
    llm.save_model(str(final_model_dir))
    
    print(f"Final model saved to {final_model_dir}")


def main():
    """Main entry point."""
    # Configuration from environment
    model_name = os.environ.get(
        "MODEL_NAME",
        "Qwen/Qwen2.5-1.5B-Instruct",
    )
    output_dir = os.environ.get("OUTPUT_DIR", "./output")
    verbose = os.environ.get("VERBOSE", "false").lower() == "true"
    
    # Training parameters
    num_episodes = int(os.environ.get("NUM_EPISODES", "1000"))
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    max_turns = int(os.environ.get("MAX_TURNS", "10"))
    eval_frequency = int(os.environ.get("EVAL_FREQUENCY", "50"))
    save_frequency = int(os.environ.get("SAVE_FREQUENCY", "100"))
    use_lora = os.environ.get("USE_LORA", "true").lower() == "true"
    
    # Create training config
    training_config = TrainingConfig(
        learning_rate=float(os.environ.get("LEARNING_RATE", "1e-5")),
        training_dataset_size=int(os.environ.get("TRAINING_DATASET_SIZE", "1000")),
        val_set_size=int(os.environ.get("VAL_SET_SIZE", "100")),
    )
    
    # Run training
    train(
        model_name=model_name,
        output_dir=output_dir,
        training_config=training_config,
        max_turns=max_turns,
        num_episodes=num_episodes,
        eval_frequency=eval_frequency,
        save_frequency=save_frequency,
        batch_size=batch_size,
        use_lora=use_lora,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()

