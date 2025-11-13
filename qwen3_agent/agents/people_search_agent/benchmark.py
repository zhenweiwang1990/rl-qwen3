#!/usr/bin/env python3
"""Benchmark evaluation for People Search Agent using Ollama or OpenRouter."""

import asyncio
import json
import os
import random
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

import polars as pl
from tqdm import tqdm

from .agent import PeopleSearchAgent
from .evaluator import PeopleSearchEvaluator
from .tasks import PeopleSearchTask
from .data_loader import load_default_benchmark
from .llm_provider import create_llm, BaseLLM


async def run_single_evaluation(
    task: PeopleSearchTask,
    agent: PeopleSearchAgent,
    evaluator: PeopleSearchEvaluator,
    llm: BaseLLM,
    max_turns: int = 10,
    verbose: bool = False,
    debug_dir: str | None = None,
) -> Dict[str, Any]:
    """Run evaluation on a single task.
    
    Args:
        task: Task to evaluate
        agent: Agent instance
        evaluator: Evaluator instance
        llm: LLM client
        max_turns: Maximum turns
        verbose: Whether to print details
        
    Returns:
        Dictionary with evaluation results
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Task {task.id}: {task.query[:80]}...")
        print(f"Expected profiles: {len(task.expected_profiles)}")
    
    # Debug trace container (always collect when debug_dir provided)
    trace: Dict[str, Any] = {
        "task_id": task.id,
        "query": task.query,
        "turns": [],
        "final_answer": None,
        "error": None,
    }
    
    # Initialize conversation
    messages = [
        {"role": "system", "content": agent.get_system_prompt(task)},
        {"role": "user", "content": task.query}
    ]
    
    # Get tools
    tools = agent.get_tools_schema()
    
    # Create rubric
    rubric = evaluator.create_rubric()
    rubric.expected_profiles = set(task.get_ground_truth()["expected_profiles"])
    
    final_answer = None
    error = None
    
    try:
        for turn in range(max_turns):
            rubric.num_turns = turn + 1
            
            if verbose:
                print(f"\nTurn {turn + 1}/{max_turns}")
            
            # Get LLM response
            try:
                response = llm.chat(messages, tools=tools, temperature=0.7)
            except Exception as e:
                error = f"LLM error: {str(e)}"
                if verbose:
                    print(f"❌ {error}")
                evaluator.on_parsing_error(rubric, "llm_error", error)
                break
            
            message = response.get("message", {})
            tool_calls = message.get("tool_calls", [])
            # Record LLM output for this turn
            turn_record: Dict[str, Any] = {
                "turn": turn + 1,
                "ollama_request": getattr(llm, "_last_request", None),
                "ollama_response_raw": response,
                "llm_message": message,
                "tool_exec": [],
            }
            
            if not tool_calls:
                # No tool calls, might be final text response
                content = message.get("content", "")
                if content:
                    if verbose:
                        print(f"LLM response: {content[:100]}...")
                    turn_record["final_text"] = content
                break
            
            # Add assistant message
            messages.append(message)
            
            # Execute tool calls
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                tool_name = function.get("name", "")
                
                # Parse arguments
                try:
                    tool_args = function.get("arguments", {})
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args)
                except json.JSONDecodeError as e:
                    error = f"Parse error: {str(e)}"
                    if verbose:
                        print(f"❌ {error}")
                    evaluator.on_parsing_error(rubric, "parse_error", error)
                    turn_record["tool_exec"].append({
                        "tool": tool_name,
                        "args": function.get("arguments", {}),
                        "error": error,
                    })
                    continue
                
                if verbose:
                    print(f"  Tool: {tool_name}")
                    print(f"  Args: {json.dumps(tool_args, indent=2)[:200]}...")
                
                # Execute tool
                result = agent.execute_action(tool_name, tool_args, task)
                
                if not result.success:
                    error = result.error
                    if verbose:
                        print(f"  ❌ Error: {error}")
                    evaluator.on_parsing_error(rubric, "action_error", error)
                    turn_record["tool_exec"].append({
                        "tool": tool_name,
                        "args": tool_args,
                        "error": error,
                    })
                    continue
                
                if verbose:
                    if isinstance(result.data, list):
                        print(f"  ✓ Result: {len(result.data)} items")
                    elif isinstance(result.data, dict):
                        print(f"  ✓ Result: {list(result.data.keys())}")
                turn_record["tool_exec"].append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result.data,
                })
                
                # Update rubric
                evaluator.on_action_executed(
                    rubric,
                    tool_name,
                    tool_args,
                    result.data,
                    task
                )
                
                # Add tool result to messages
                tool_message = {
                    "role": "tool",
                    "content": json.dumps(result.data) if result.data else "{}"
                }
                messages.append(tool_message)
                
                # Check if this is final answer
                if agent.is_terminal_action(tool_name):
                    final_answer = result.data
                    trace["final_answer"] = result.data
                    break
            
            # Break if we have final answer
            if final_answer:
                break
            
            trace["turns"].append(turn_record)
    
    except Exception as e:
        error = f"Execution error: {str(e)}"
        if verbose:
            print(f"❌ {error}")
        trace["error"] = error
    
    # Evaluate trajectory
    reward = await evaluator.evaluate_trajectory(None, task, rubric)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Results:")
        print(f"  Reward: {reward:.2f}")
        print(f"  Turns: {rubric.num_turns}")
        print(f"  Answer correct: {rubric.answer_correct}")
        print(f"  Profiles found: {len(rubric.profiles_found_in_search)}/{len(rubric.expected_profiles)}")
        print(f"  Profiles read: {len(rubric.profiles_read_correct)}/{len(rubric.expected_profiles)}")
        if rubric.answer_profiles:
            overlap = len(rubric.answer_profiles & rubric.expected_profiles)
            print(f"  Answer overlap: {overlap}/{len(rubric.expected_profiles)} ({overlap/len(rubric.expected_profiles)*100:.1f}%)")
    
    # Persist debug trace if requested
    if debug_dir:
        try:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            trace_path = Path(debug_dir) / f"task_{task.id}_trace.json"
            with open(trace_path, "w") as f:
                json.dump(trace, f, indent=2)
            if verbose:
                print(f"\n📝 Saved debug trace: {trace_path}")
        except Exception as _:
            pass
    
    # Calculate overlap ratio
    overlap_count = len(rubric.answer_profiles & rubric.expected_profiles) if rubric.answer_profiles else 0
    overlap_ratio = overlap_count / len(rubric.expected_profiles) if len(rubric.expected_profiles) > 0 else 0.0
    
    # Determine error categories
    has_format_error = rubric.cant_parse_tool_call or rubric.bad_tool_call_name or rubric.bad_tool_call_args
    has_wrong_answer = rubric.attempted_answer and not rubric.answer_correct
    no_answer_given = not rubric.attempted_answer or rubric.returned_empty_list
    
    # Collect metrics with detailed intermediate values
    metrics = {
        "task_id": task.id,
        "query": task.query,
        "reward": reward,
        "num_turns": rubric.num_turns,
        
        # Correctness metrics
        "answer_correct": rubric.answer_correct,
        "answer_perfect_match": rubric.answer_perfect_match,
        "attempted_answer": rubric.attempted_answer,
        "has_wrong_answer": has_wrong_answer,
        "no_answer_given": no_answer_given,
        
        # Profile metrics
        "profiles_expected": len(rubric.expected_profiles),
        "profiles_found": len(rubric.profiles_found_in_search),
        "profiles_read": len(rubric.profiles_read_correct),
        "profiles_returned": len(rubric.answer_profiles) if rubric.answer_profiles else 0,
        "overlap_count": overlap_count,
        "overlap_ratio": overlap_ratio,
        
        # Error metrics
        "has_format_error": has_format_error,
        "cant_parse_tool_call": rubric.cant_parse_tool_call,
        "bad_tool_call_name": rubric.bad_tool_call_name,
        "bad_tool_call_args": rubric.bad_tool_call_args,
        "ever_tried_to_read_invalid_profile": rubric.ever_tried_to_read_invalid_profile,
        "ran_out_of_turns": rubric.ran_out_of_turns,
        "returned_empty_list": rubric.returned_empty_list,
        
        # System metrics
        "error": error if error else "",
    }
    
    return metrics


async def benchmark_agent(
    num_samples: int = 100,
    max_turns: int = 10,
    model: str = "qwen3:14b",
    provider: str = "ollama",
    random_seed: int = 42,
    output_dir: str = "./benchmark_results",
    verbose: bool = True,
    benchmark_file: str | None = None,
    debug: bool = False,
    debug_count: int = 1,
) -> pl.DataFrame:
    """Benchmark People Search Agent on random sample of tasks.
    
    Args:
        num_samples: Number of tasks to sample
        max_turns: Maximum turns per task
        model: Model name (e.g., "qwen3:14b" for Ollama, "qwen3-30b-a3b-instruct-2507" for OpenRouter)
        provider: LLM provider ("ollama" or "openrouter")
        random_seed: Random seed for sampling
        output_dir: Directory to save results
        verbose: Whether to print progress
        benchmark_file: Optional custom benchmark file path
        debug: Debug mode
        debug_count: Number of tasks to debug
        
    Returns:
        DataFrame with evaluation results
    """
    if verbose:
        print("\n" + "="*80)
        print("People Search Agent Benchmark")
        print("="*80)
        print(f"Provider: {provider}")
        print(f"Model: {model}")
        print(f"Samples: {num_samples}")
        print(f"Max turns: {max_turns}")
        print(f"Random seed: {random_seed}")
        print()
    
    # Load all benchmark tasks
    if benchmark_file:
        if verbose:
            print(f"Loading custom benchmark: {benchmark_file}")
        from .data_loader import load_benchmark_queries
        all_tasks = load_benchmark_queries(benchmark_file)
    else:
        all_tasks = load_default_benchmark()
        if verbose:
            print(f"Using training benchmark")
    
    if verbose:
        print(f"Loaded {len(all_tasks)} total tasks")
    
    # Sample tasks
    random.seed(random_seed)
    if num_samples > len(all_tasks):
        num_samples = len(all_tasks)
        if verbose:
            print(f"Warning: Requested {num_samples} samples, but only {len(all_tasks)} available")
    
    sampled_tasks = random.sample(all_tasks, num_samples)
    if verbose:
        print(f"Sampled {len(sampled_tasks)} tasks")
        print()
    
    # Create agent, evaluator, LLM
    evaluator = PeopleSearchEvaluator(
        simple_reward=False,
        verbose=False,  # Don't print for each task
        max_turns=max_turns,
    )
    agent = PeopleSearchAgent(evaluator=evaluator)
    
    # Create LLM client based on provider
    llm = create_llm(
        provider=provider,
        model=model,
        debug=debug,
    )
    
    # Run evaluations
    results = []
    
    if verbose:
        print("Running evaluations...")
        pbar = tqdm(total=num_samples, desc="Evaluating")
    
    for i, task in enumerate(sampled_tasks):
        try:
            metrics = await run_single_evaluation(
                task=task,
                agent=agent,
                evaluator=evaluator,
                llm=llm,
                max_turns=max_turns,
                verbose=(debug and i < debug_count),
                debug_dir=(str(Path(output_dir) / "debug") if debug and i < debug_count else None),
            )
            results.append(metrics)
            
            if verbose:
                pbar.update(1)
                # Update progress bar description with current average reward
                if results:
                    avg_reward = sum(r["reward"] for r in results) / len(results)
                    pbar.set_description(f"Evaluating (avg reward: {avg_reward:.2f})")
        
        except Exception as e:
            if verbose:
                print(f"\n❌ Error on task {task.id}: {e}")
            results.append({
                "task_id": task.id,
                "query": task.query,
                "reward": 0.0,
                "error": str(e),
            })
            if verbose:
                pbar.update(1)
    
    if verbose:
        pbar.close()
    
    # Create DataFrame
    df = pl.DataFrame(results)
    
    # Calculate summary statistics
    if verbose:
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        total_tasks = len(df)
        
        # Overall metrics
        print("\n📊 Overall Metrics:")
        print(f"  Total tasks evaluated: {total_tasks}")
        successful = len(df.filter(pl.col('error') == ''))
        failed = len(df.filter(pl.col('error') != ''))
        print(f"  Successful: {successful} ({successful/total_tasks*100:.1f}%)")
        print(f"  Failed: {failed} ({failed/total_tasks*100:.1f}%)")
        
        # Reward statistics
        print("\n💰 Reward Statistics:")
        reward_mean = df['reward'].mean()
        reward_median = df['reward'].median()
        reward_std = df['reward'].std()
        reward_min = df['reward'].min()
        reward_max = df['reward'].max()
        
        print(f"  Mean:   {reward_mean:.3f}" if reward_mean is not None else "  Mean:   N/A")
        print(f"  Median: {reward_median:.3f}" if reward_median is not None else "  Median: N/A")
        print(f"  Std:    {reward_std:.3f}" if reward_std is not None else "  Std:    N/A")
        print(f"  Min:    {reward_min:.3f}" if reward_min is not None else "  Min:    N/A")
        print(f"  Max:    {reward_max:.3f}" if reward_max is not None else "  Max:    N/A")
        
        # Reward distribution (updated for new scoring: -3 to 4)
        print(f"\n  Reward Distribution:")
        reward_excellent = len(df.filter(pl.col('reward') >= 3.0))  # Perfect match or near-perfect
        reward_great = len(df.filter((pl.col('reward') >= 2.0) & (pl.col('reward') < 3.0)))  # High coverage (>=0.6)
        reward_good = len(df.filter((pl.col('reward') >= 1.0) & (pl.col('reward') < 2.0)))  # Medium coverage (>=0.3)
        reward_ok = len(df.filter((pl.col('reward') >= 0.0) & (pl.col('reward') < 1.0)))  # Low coverage or partial progress
        reward_poor = len(df.filter((pl.col('reward') >= -1.0) & (pl.col('reward') < 0.0)))  # No answer but some progress
        reward_bad = len(df.filter(pl.col('reward') < -1.0))  # Serious errors
        print(f"    Excellent (>=3.0):  {reward_excellent} ({reward_excellent/total_tasks*100:.1f}%)  [Perfect/near-perfect]")
        print(f"    Great (2.0-3.0):    {reward_great} ({reward_great/total_tasks*100:.1f}%)  [High coverage >=60%]")
        print(f"    Good (1.0-2.0):     {reward_good} ({reward_good/total_tasks*100:.1f}%)  [Medium coverage >=30%]")
        print(f"    OK (0.0-1.0):       {reward_ok} ({reward_ok/total_tasks*100:.1f}%)  [Low coverage/partial]")
        print(f"    Poor (-1.0-0.0):    {reward_poor} ({reward_poor/total_tasks*100:.1f}%)  [No answer+progress]")
        print(f"    Bad (<-1.0):        {reward_bad} ({reward_bad/total_tasks*100:.1f}%)  [Serious errors]")
        
        # Core correctness metrics (matching your requirements)
        print("\n✅ Answer Correctness Metrics:")
        
        # 1. Format error rate
        if "has_format_error" in df.columns:
            format_errors = df.filter(pl.col("has_format_error") == True)
            format_error_rate = len(format_errors) / total_tasks * 100
            print(f"  1. Format errors:          {len(format_errors):3d}/{total_tasks} ({format_error_rate:5.1f}%)")
            
            # Breakdown
            parse_errors = df.filter(pl.col("cant_parse_tool_call") == True)
            bad_names = df.filter(pl.col("bad_tool_call_name") == True)
            bad_args = df.filter(pl.col("bad_tool_call_args") == True)
            print(f"     - Parse errors:         {len(parse_errors)}")
            print(f"     - Bad tool names:       {len(bad_names)}")
            print(f"     - Bad tool args:        {len(bad_args)}")
        
        # 2. Wrong answer rate
        if "has_wrong_answer" in df.columns:
            wrong_answers = df.filter(pl.col("has_wrong_answer") == True)
            wrong_answer_rate = len(wrong_answers) / total_tasks * 100
            print(f"  2. Wrong answers:          {len(wrong_answers):3d}/{total_tasks} ({wrong_answer_rate:5.1f}%)")
        
        # 3. No answer given rate
        if "no_answer_given" in df.columns:
            no_answers = df.filter(pl.col("no_answer_given") == True)
            no_answer_rate = len(no_answers) / total_tasks * 100
            print(f"  3. No answer given:        {len(no_answers):3d}/{total_tasks} ({no_answer_rate:5.1f}%)")
            
            # Breakdown
            ran_out = df.filter(pl.col("ran_out_of_turns") == True)
            empty_list = df.filter(pl.col("returned_empty_list") == True)
            print(f"     - Ran out of turns:     {len(ran_out)}")
            print(f"     - Returned empty list:  {len(empty_list)}")
        
        # 4. Correct answer rate
        if "answer_correct" in df.columns:
            correct = df.filter(pl.col("answer_correct") == True)
            correct_rate = len(correct) / total_tasks * 100
            print(f"  4. Correct answers:        {len(correct):3d}/{total_tasks} ({correct_rate:5.1f}%)")
        
        # 5. Perfect match rate
        if "answer_perfect_match" in df.columns:
            perfect = df.filter(pl.col("answer_perfect_match") == True)
            perfect_rate = len(perfect) / total_tasks * 100
            print(f"  5. Perfect matches:        {len(perfect):3d}/{total_tasks} ({perfect_rate:5.1f}%)")
        
        # 6. Average overlap ratio
        if "overlap_ratio" in df.columns:
            # Only calculate for tasks that gave an answer
            answered = df.filter(pl.col("attempted_answer") == True)
            if len(answered) > 0:
                avg_overlap_ratio = answered['overlap_ratio'].mean() * 100
                median_overlap_ratio = answered['overlap_ratio'].median() * 100
                print(f"  6. Avg overlap ratio:      {avg_overlap_ratio:5.1f}% (median: {median_overlap_ratio:.1f}%)")
                
                # Overlap distribution
                high_overlap = len(answered.filter(pl.col('overlap_ratio') > 0.7))
                med_overlap = len(answered.filter((pl.col('overlap_ratio') >= 0.3) & (pl.col('overlap_ratio') <= 0.7)))
                low_overlap = len(answered.filter(pl.col('overlap_ratio') < 0.3))
                print(f"     - High (>70%):          {high_overlap} ({high_overlap/len(answered)*100:.1f}% of answered)")
                print(f"     - Medium (30-70%):      {med_overlap} ({med_overlap/len(answered)*100:.1f}% of answered)")
                print(f"     - Low (<30%):           {low_overlap} ({low_overlap/len(answered)*100:.1f}% of answered)")
        
        # 7. Average turns
        if "num_turns" in df.columns:
            avg_turns = df['num_turns'].mean()
            median_turns = df['num_turns'].median()
            print(f"  7. Avg turns per task:     {avg_turns:5.1f} (median: {median_turns:.1f})")
        
        # Profile statistics
        if "profiles_expected" in df.columns:
            print(f"\n👥 Profile Statistics:")
            print(f"  Expected per task:    {df['profiles_expected'].mean():.1f}")
            print(f"  Found in search:      {df['profiles_found'].mean():.1f} ({df['profiles_found'].mean()/df['profiles_expected'].mean()*100:.1f}%)")
            print(f"  Read correctly:       {df['profiles_read'].mean():.1f} ({df['profiles_read'].mean()/df['profiles_expected'].mean()*100:.1f}%)")
            print(f"  Returned in answer:   {df['profiles_returned'].mean():.1f}")
            
            # Invalid reads
            invalid_reads = df.filter(pl.col("ever_tried_to_read_invalid_profile") == True)
            print(f"  Invalid profile reads: {len(invalid_reads)} tasks ({len(invalid_reads)/total_tasks*100:.1f}%)")
    
    # Save detailed results to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model.replace(':', '_')
    
    # Save CSV with all task details
    csv_filename = f"benchmark_{model_safe}_{num_samples}samples_{timestamp}.csv"
    csv_file = output_path / csv_filename
    df.write_csv(csv_file)
    
    # Save summary statistics to JSON
    total_tasks = len(df)
    summary = {
        "timestamp": timestamp,
        "provider": provider,
        "model": model,
        "num_samples": num_samples,
        "max_turns": max_turns,
        "random_seed": random_seed,
        
        # Overall metrics
        "total_tasks": total_tasks,
        "successful_tasks": len(df.filter(pl.col('error') == '')),
        "failed_tasks": len(df.filter(pl.col('error') != '')),
        
        # Reward statistics
        "reward_mean": float(df['reward'].mean()),
        "reward_median": float(df['reward'].median()),
        "reward_std": float(df['reward'].std()),
        "reward_min": float(df['reward'].min()),
        "reward_max": float(df['reward'].max()),
        
        # Core metrics (matching requirements)
        "format_error_count": int(df.filter(pl.col("has_format_error") == True).shape[0]) if "has_format_error" in df.columns else 0,
        "format_error_rate": float(df.filter(pl.col("has_format_error") == True).shape[0] / total_tasks) if "has_format_error" in df.columns else 0,
        
        "wrong_answer_count": int(df.filter(pl.col("has_wrong_answer") == True).shape[0]) if "has_wrong_answer" in df.columns else 0,
        "wrong_answer_rate": float(df.filter(pl.col("has_wrong_answer") == True).shape[0] / total_tasks) if "has_wrong_answer" in df.columns else 0,
        
        "no_answer_count": int(df.filter(pl.col("no_answer_given") == True).shape[0]) if "no_answer_given" in df.columns else 0,
        "no_answer_rate": float(df.filter(pl.col("no_answer_given") == True).shape[0] / total_tasks) if "no_answer_given" in df.columns else 0,
        
        "correct_answer_count": int(df.filter(pl.col("answer_correct") == True).shape[0]) if "answer_correct" in df.columns else 0,
        "correct_answer_rate": float(df.filter(pl.col("answer_correct") == True).shape[0] / total_tasks) if "answer_correct" in df.columns else 0,
        
        "perfect_match_count": int(df.filter(pl.col("answer_perfect_match") == True).shape[0]) if "answer_perfect_match" in df.columns else 0,
        "perfect_match_rate": float(df.filter(pl.col("answer_perfect_match") == True).shape[0] / total_tasks) if "answer_perfect_match" in df.columns else 0,
        
        "avg_turns": float(df['num_turns'].mean()) if "num_turns" in df.columns else 0,
        "median_turns": float(df['num_turns'].median()) if "num_turns" in df.columns else 0,
    }
    
    # Add overlap ratio statistics (only for answered tasks)
    if "overlap_ratio" in df.columns and "attempted_answer" in df.columns:
        answered = df.filter(pl.col("attempted_answer") == True)
        if len(answered) > 0:
            summary["avg_overlap_ratio"] = float(answered['overlap_ratio'].mean())
            summary["median_overlap_ratio"] = float(answered['overlap_ratio'].median())
            summary["answered_tasks_count"] = len(answered)
        else:
            summary["avg_overlap_ratio"] = 0.0
            summary["median_overlap_ratio"] = 0.0
            summary["answered_tasks_count"] = 0
    
    # Add profile statistics
    if "profiles_expected" in df.columns:
        summary["avg_profiles_expected"] = float(df['profiles_expected'].mean())
        summary["avg_profiles_found"] = float(df['profiles_found'].mean())
        summary["avg_profiles_read"] = float(df['profiles_read'].mean())
        summary["avg_profiles_returned"] = float(df['profiles_returned'].mean())
    
    # Save summary JSON
    json_filename = f"benchmark_{model_safe}_{num_samples}samples_{timestamp}_summary.json"
    json_file = output_path / json_filename
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"\n💾 Files saved:")
        print(f"  Detailed results: {csv_file}")
        print(f"  Summary stats:    {json_file}")
        print("="*80)
    
    return df


async def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark People Search Agent on random sample of tasks"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=100,
        help="Number of tasks to sample (default: 100)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum turns per task (default: 10)"
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider: 'ollama' or 'openrouter' (default: from LLM_PROVIDER env or 'ollama')"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: from LLM_MODEL env or 'qwen3:14b')"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        default="./benchmark_results",
        help="Output directory for results (default: ./benchmark_results)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (minimal output)"
    )
    parser.add_argument(
        "--benchmark-file",
        type=str,
        default=None,
        help="Path to custom benchmark CSV file (default: training benchmark)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed logs for the first few tasks and save JSON traces",
    )
    parser.add_argument(
        "--debug-count",
        type=int,
        default=1,
        help="How many tasks to debug in detail (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Get provider and model from args or environment variables
    provider = args.provider or os.getenv("LLM_PROVIDER", "ollama")
    model = args.model or os.getenv("LLM_MODEL", "qwen3:14b")
    
    # Run benchmark
    df = await benchmark_agent(
        num_samples=args.num_samples,
        max_turns=args.max_turns,
        model=model,
        provider=provider,
        random_seed=args.seed,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        benchmark_file=args.benchmark_file,
        debug=args.debug,
        debug_count=args.debug_count,
    )
    
    return df


if __name__ == "__main__":
    asyncio.run(main())

