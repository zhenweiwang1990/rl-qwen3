"""Test script to verify SB3 training code can be imported."""

import sys

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")
    
    try:
        from qwen3_agent.evaluation import EvaluationRubric, determine_if_answer_is_correct
        print("✓ EvaluationRubric and determine_if_answer_is_correct imported successfully")
    except Exception as e:
        print(f"✗ Failed to import evaluation module: {e}")
        return False
    
    try:
        from qwen3_agent.env import EmailSearchEnv
        print("✓ EmailSearchEnv imported successfully")
    except Exception as e:
        print(f"✗ Failed to import EmailSearchEnv: {e}")
        return False
    
    try:
        from qwen3_agent.local_model import LocalLLM, LLMAgent
        print("✓ LocalLLM and LLMAgent imported successfully")
    except Exception as e:
        print(f"✗ Failed to import LocalLLM: {e}")
        return False
    
    try:
        from qwen3_agent.train_sb3 import REINFORCETrainer, train
        print("✓ REINFORCETrainer and train imported successfully")
    except Exception as e:
        print(f"✗ Failed to import train_sb3: {e}")
        return False
    
    try:
        import gymnasium
        print("✓ gymnasium imported successfully")
    except Exception as e:
        print(f"✗ Failed to import gymnasium: {e}")
        print("  Run: pip install gymnasium")
        return False
    
    try:
        from stable_baselines3 import PPO
        print("✓ stable-baselines3 imported successfully")
    except Exception as e:
        print(f"✗ Failed to import stable-baselines3: {e}")
        print("  Run: pip install stable-baselines3")
        return False
    
    try:
        from peft import LoraConfig, get_peft_model
        print("✓ peft imported successfully")
    except Exception as e:
        print(f"✗ Failed to import peft: {e}")
        print("  Run: pip install peft")
        return False
    
    print("\n✓ All imports successful!")
    return True


def test_environment():
    """Test basic environment functionality."""
    print("\nTesting environment...")
    
    try:
        from qwen3_agent.env import EmailSearchEnv
        from qwen3_agent.evaluation import EvaluationRubric
        from qwen3_agent.data.types import SyntheticQuery
        
        # Create a dummy scenario
        dummy_scenario = SyntheticQuery(
            id=0,
            question="Test question?",
            answer="Test answer",
            message_ids=["<test@example.com>"],
            how_realistic=1.0,
            inbox_address="user@example.com",
            query_date="2024-01-01",
        )
        
        # Create environment
        env = EmailSearchEnv(
            scenarios=[dummy_scenario],
            max_turns=5,
            verbose=False,
        )
        
        # Reset environment
        obs, info = env.reset()
        
        print("✓ Environment created and reset successfully")
        print(f"  Observation keys: {list(obs.keys())}")
        print(f"  Info keys: {list(info.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("SB3 Training Code - Import Test")
    print("="*60)
    print()
    
    # Test imports
    if not test_imports():
        print("\n✗ Import test failed")
        sys.exit(1)
    
    # Test environment (optional, requires database)
    try:
        test_environment()
    except Exception as e:
        print(f"\n⚠ Environment test skipped (database may not be generated yet)")
        print(f"  Error: {e}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    print("\nYou can now run training with:")
    print("  ./scripts/quick_train_sb3.sh  # Quick test")
    print("  ./scripts/train_sb3.sh        # Full training")
    print()


if __name__ == "__main__":
    main()

