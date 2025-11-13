#!/usr/bin/env python3
"""Quick test to verify all imports work correctly."""

import sys
import os

def test_imports():
    """Test that all main modules can be imported."""
    print("Testing imports...")
    
    errors = []
    
    try:
        print("  ✓ Importing qwen3_agent...")
        import qwen3_agent
        print(f"    Version: {qwen3_agent.__version__}")
    except Exception as e:
        errors.append(f"qwen3_agent: {e}")
        print(f"  ✗ qwen3_agent failed: {e}")
    
    try:
        print("  ✓ Importing qwen3_agent.config...")
        from qwen3_agent.config import PolicyConfig, TrainingConfig, get_device
        print(f"    Detected device: {get_device()}")
    except Exception as e:
        errors.append(f"qwen3_agent.config: {e}")
        print(f"  ✗ qwen3_agent.config failed: {e}")
    
    try:
        print("  ✓ Importing qwen3_agent.data...")
        from qwen3_agent.data import SyntheticQuery, Email, load_synthetic_queries
    except Exception as e:
        errors.append(f"qwen3_agent.data: {e}")
        print(f"  ✗ qwen3_agent.data failed: {e}")
    
    try:
        print("  ✓ Importing qwen3_agent.tools...")
        from qwen3_agent.tools import search_emails, read_email, SearchResult
    except Exception as e:
        errors.append(f"qwen3_agent.tools: {e}")
        print(f"  ✗ qwen3_agent.tools failed: {e}")
    
    try:
        print("  ✓ Importing qwen3_agent.core.framework...")
        from qwen3_agent.core.framework import (
            BaseAgent, BaseTask, BaseEvaluator, 
            LLMInference, generic_rollout
        )
    except Exception as e:
        errors.append(f"qwen3_agent.core.framework: {e}")
        print(f"  ✗ qwen3_agent.core.framework failed: {e}")
    
    try:
        print("  ✓ Importing qwen3_agent.agents.email_agent...")
        from qwen3_agent.agents.email_agent import (
            EmailAgent, EmailTask, EmailEvaluator
        )
    except Exception as e:
        errors.append(f"qwen3_agent.agents.email_agent: {e}")
        print(f"  ✗ qwen3_agent.agents.email_agent failed: {e}")
    
    try:
        print("  ✓ Importing qwen3_agent.benchmark...")
        from qwen3_agent.benchmark import benchmark_model
    except Exception as e:
        errors.append(f"qwen3_agent.benchmark: {e}")
        print(f"  ✗ qwen3_agent.benchmark failed: {e}")
    
    try:
        print("  ✓ Importing qwen3_agent.train...")
        from qwen3_agent.train import create_trainable_model, run_training
    except Exception as e:
        errors.append(f"qwen3_agent.train: {e}")
        print(f"  ✗ qwen3_agent.train failed: {e}")
    
    print("\n" + "="*60)
    if errors:
        print(f"✗ {len(errors)} import error(s) found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✓ All imports successful!")
        return True


def test_config():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("Testing configuration...")
    
    try:
        from qwen3_agent.config import PolicyConfig, TrainingConfig
        
        # Test default config
        config = PolicyConfig()
        print(f"  ✓ Default PolicyConfig created")
        print(f"    - max_turns: {config.max_turns}")
        print(f"    - max_tokens: {config.max_tokens}")
        
        # Test training config
        training_config = TrainingConfig()
        print(f"  ✓ Default TrainingConfig created")
        print(f"    - trajectories_per_group: {training_config.trajectories_per_group}")
        print(f"    - learning_rate: {training_config.learning_rate}")
        
        return True
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Qwen3 Email Agent - Import Test")
    print("="*60)
    print()
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test config
    if not test_config():
        success = False
    
    print("\n" + "="*60)
    if success:
        print("✓ All tests passed!")
        print("="*60)
        return 0
    else:
        print("✗ Some tests failed")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

