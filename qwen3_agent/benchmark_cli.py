"""Benchmark CLI entry point."""

import asyncio
import os
from dotenv import load_dotenv
from qwen3_agent.config import PolicyConfig, get_device
from qwen3_agent.benchmark import benchmark_model
from qwen3_agent.core import Model
import polars as pl


def initialize_model(
    model_name: str,
    vllm_base_url: str,
    verbose: bool = False,
) -> Model:
    """Initialize a model for benchmarking.
    
    Args:
        model_name: Name of the model to use
        vllm_base_url: Base URL for vLLM or OpenAI-compatible server
        verbose: Whether to enable verbose logging
    
    Returns:
        Initialized Model instance
    """
    # Pure standalone model using LiteLLM + OpenAI-compatible server
    if "Qwen" in model_name or "qwen" in model_name:
        model = Model(
            name=model_name,
            project="qwen3_email_agent",
            config=PolicyConfig(
                litellm_model_name=f"openai/{model_name}",
                verbose=verbose,
            ),
            inference_api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            inference_base_url=vllm_base_url,
            inference_model_name=model_name,
        )
    else:
        # External OpenAI model
        model = Model(
            name=model_name,
            project="qwen3_email_agent",
            config=PolicyConfig(
                litellm_model_name=f"openai/{model_name}",
                verbose=verbose,
            ),
            inference_api_key=os.getenv("OPENAI_API_KEY"),
            inference_base_url="https://api.openai.com/v1",
            inference_model_name=model_name,
        )
    
    return model


async def main():
    """Main entry point for benchmark CLI."""
    load_dotenv()
    
    run_id = os.environ.get("RUN_ID", "001")
    test_set_size = int(os.environ.get("TEST_SET_SIZE", "100"))
    verbose = os.environ.get("VERBOSE", "false").lower() == "true"
    device = get_device()

    model_name = os.environ.get("MODEL_NAME", "OpenPipe/Qwen3-14B-Instruct")
    # Prefer explicit VLLM_BASE_URL if provided, else use INFERENCE_BASE_URL
    vllm_base_url = os.environ.get("VLLM_BASE_URL") or os.environ.get("INFERENCE_BASE_URL", "http://localhost:8000/v1")

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print("")

    # Initialize model
    model = initialize_model(model_name, vllm_base_url, verbose)

    # Run benchmark
    results = await benchmark_model(model, limit=test_set_size, verbose=verbose)

    print("\n" + "="*60)
    print("Benchmark Results")
    print("="*60)
    print(results)

    # Save results
    results.write_csv(f"benchmark_results_{run_id}.csv")
    print(f"\nResults saved to benchmark_results_{run_id}.csv")


if __name__ == "__main__":
    asyncio.run(main())

