"""Benchmark evaluation utilities."""

import art
from typing import Union
from art import Trajectory
from art import gather_trajectories
from qwen3_agent.rollout import rollout
from qwen3_agent.data import load_synthetic_queries
import polars as pl
import os

# Support both ART models and any compatible model type
Model = Union[art.Model, art.TrainableModel]


async def benchmark_model(
    model: Model, 
    limit: int = 100, 
    swallow_exceptions: bool = True,
    verbose: bool = False,
) -> pl.DataFrame:
    """Benchmark a model on the test set. Returns DataFrame with aggregated metrics.
    
    Args:
        model: The model to benchmark
        limit: Number of test scenarios to evaluate
        swallow_exceptions: Whether to continue on exceptions
        verbose: Whether to print detailed logs
    """
    if verbose:
        print(f"\n=== Benchmarking {model.name} ===")
        print(f"Evaluating on {limit} test scenarios...")
    
    val_scenarios = load_synthetic_queries(split="test", limit=limit)
    # Optionally run sequentially to avoid loading the server with many pending tasks
    if os.environ.get("BENCH_SEQUENTIAL", "false").lower() == "true":
        if verbose:
            print("Running benchmark sequentially (BENCH_SEQUENTIAL=true)")
        val_trajectories = []
        for i, scenario in enumerate(val_scenarios, 1):
            try:
                t = await rollout(model, scenario)
                val_trajectories.append(t)
            except BaseException as e:
                val_trajectories.append(e)
            if verbose:
                print(f"Completed {i}/{limit}")
    else:
        val_trajectories = await gather_trajectories(
            (rollout(model, scenario) for scenario in val_scenarios),
            pbar_desc=f"validation {model.name}",
            max_exceptions=limit if swallow_exceptions else 0,
        )

    valid_trajectories = [t for t in val_trajectories if isinstance(t, Trajectory)]

    if verbose:
        errored = [e for e in val_trajectories if not isinstance(e, Trajectory)]
        if errored:
            print(f"\nEncountered {len(errored)} exceptions during benchmark (showing up to 5):")
            for i, e in enumerate(errored[:5], 1):
                print("-" * 80)
                print(f"Exception {i}: {type(e)}\n{e}")

    if verbose:
        print(f"Completed {len(valid_trajectories)}/{len(val_scenarios)} trajectories successfully")

    metrics = pl.DataFrame(
        [{**t.metrics, "reward": t.reward} for t in valid_trajectories]
    )

    avg_metrics = metrics.select(
        [pl.mean(c).alias(c) for c in metrics.columns]
    ).with_columns(pl.lit(len(valid_trajectories)).alias("n_trajectories"))

    if verbose:
        print("\n=== Benchmark Results ===")
        print(avg_metrics)
        print()

    return avg_metrics

