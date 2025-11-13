#!/usr/bin/env python3
"""Filter benchmark queries to only include those with profiles in training database."""

import csv
import json
import sqlite3
from pathlib import Path


def load_training_profiles(db_path: str) -> set:
    """Load set of linkedin handles from training database.
    
    Args:
        db_path: Path to training database
        
    Returns:
        Set of linkedin handles in the database
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT linkedin_handle FROM profiles")
    handles = {row[0] for row in cursor.fetchall()}
    
    conn.close()
    
    print(f"Loaded {len(handles)} profiles from training database")
    return handles


def filter_benchmark_queries(
    input_csv: str,
    output_csv: str,
    training_handles: set,
    verbose: bool = True
) -> tuple[int, int]:
    """Filter benchmark queries to only those answerable with training database.
    
    Args:
        input_csv: Input benchmark CSV path
        output_csv: Output filtered CSV path
        training_handles: Set of handles in training database
        verbose: Print progress
        
    Returns:
        Tuple of (kept_count, filtered_count)
    """
    kept_count = 0
    filtered_count = 0
    
    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            task_id = row['id']
            content_json = row['content']
            
            try:
                # Parse expected profiles
                expected_profiles = json.loads(content_json)
                
                # Check if all expected profiles are in training database
                all_in_training = all(
                    handle in training_handles 
                    for handle in expected_profiles
                )
                
                if all_in_training:
                    # Keep this query
                    writer.writerow(row)
                    kept_count += 1
                    
                    if verbose and kept_count % 100 == 0:
                        print(f"  Kept {kept_count} queries...")
                else:
                    # Filter out this query
                    filtered_count += 1
                    
                    if verbose:
                        missing = [
                            h for h in expected_profiles 
                            if h not in training_handles
                        ]
                        print(f"  Filtered query {task_id}: {len(missing)} missing profiles")
                        if len(missing) <= 3:
                            print(f"    Missing: {missing}")
                    
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse content for task {task_id}: {e}")
                filtered_count += 1
                continue
    
    return kept_count, filtered_count


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    
    # Paths
    db_path = script_dir / "profiles_training.db"
    input_csv = script_dir / "data" / "benchmark-queries-flattened.csv"
    output_csv = script_dir / "data" / "benchmark-queries-training.csv"
    
    # Check files exist
    if not db_path.exists():
        print(f"❌ Training database not found: {db_path}")
        print("   Run 'python3 create_training_db.py' first.")
        return
    
    if not input_csv.exists():
        print(f"❌ Benchmark file not found: {input_csv}")
        return
    
    print(f"Filtering benchmark queries...")
    print(f"  Input:  {input_csv}")
    print(f"  Output: {output_csv}")
    print()
    
    # Load training profiles
    training_handles = load_training_profiles(str(db_path))
    print()
    
    # Filter benchmark
    kept, filtered = filter_benchmark_queries(
        str(input_csv),
        str(output_csv),
        training_handles,
        verbose=True
    )
    
    print()
    print("="*80)
    print("✓ Filtering complete!")
    print(f"  - Kept: {kept} queries")
    print(f"  - Filtered out: {filtered} queries")
    print(f"  - Retention rate: {kept/(kept+filtered)*100:.1f}%")
    print(f"  - Output: {output_csv}")
    print("="*80)
    
    # Show stats
    total = kept + filtered
    if total > 0:
        print()
        print(f"Original benchmark: {total} queries")
        print(f"Training benchmark: {kept} queries ({kept/total*100:.1f}%)")
        print()
        print("Use this filtered benchmark for training evaluation:")
        print(f"  python3 benchmark.py --benchmark-file data/benchmark-queries-training.csv")


if __name__ == "__main__":
    main()

