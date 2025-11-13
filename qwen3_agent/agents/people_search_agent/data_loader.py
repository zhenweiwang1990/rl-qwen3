"""Data loader for people search benchmark queries."""

import csv
import json
from pathlib import Path
from typing import List

from .tasks import PeopleSearchTask


def load_benchmark_queries(csv_path: str | Path) -> List[PeopleSearchTask]:
    """Load benchmark queries from flattened CSV file.
    
    Args:
        csv_path: Path to benchmark-queries-flattened.csv
        
    Returns:
        List of PeopleSearchTask objects
    """
    tasks = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            task_id = row['id']
            query = row['query']
            content_json = row['content']
            batch = row.get('batch', '')
            
            # Parse content JSON to get list of linkedin handles
            try:
                expected_profiles = json.loads(content_json)
                
                # Create task
                task = PeopleSearchTask(
                    id=task_id,
                    query=query,
                    expected_profiles=expected_profiles,
                    batch=batch,
                )
                
                tasks.append(task)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse content for task {task_id}: {e}")
                continue
    
    return tasks


def load_default_benchmark() -> List[PeopleSearchTask]:
    """Load training benchmark queries.
    
    Returns:
        List of PeopleSearchTask objects
    """
    script_dir = Path(__file__).parent
    csv_path = script_dir / "data" / "benchmark-queries-training.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training benchmark not found at {csv_path}\n"
            f"Please run: python3 filter_benchmark.py"
        )
    
    return load_benchmark_queries(csv_path)


if __name__ == "__main__":
    # Test loading
    tasks = load_default_benchmark()
    print(f"Loaded {len(tasks)} benchmark tasks")
    
    # Show first few
    for i, task in enumerate(tasks[:3]):
        print(f"\nTask {i+1}:")
        print(f"  ID: {task.id}")
        print(f"  Query: {task.query[:80]}...")
        print(f"  Expected profiles: {len(task.expected_profiles)}")
        print(f"  First 5: {task.expected_profiles[:5]}")

