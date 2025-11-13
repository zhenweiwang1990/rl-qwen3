#!/usr/bin/env python3
"""Create a filtered training database with only profiles from the linkedin handle list."""

import sqlite3
import csv
import os
import sys
from pathlib import Path

# Increase CSV field size limit to handle large fields
csv.field_size_limit(sys.maxsize)


def load_linkedin_handles(handle_csv_path: str) -> set:
    """Load the list of linkedin handles to filter.
    
    Args:
        handle_csv_path: Path to CSV containing linkedin_handle column
        
    Returns:
        Set of linkedin handles to include
    """
    handles = set()
    
    with open(handle_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            handle = row.get('linkedin_handle', '').strip()
            if handle:
                handles.add(handle)
    
    print(f"Loaded {len(handles)} linkedin handles from filter list")
    return handles


def create_filtered_database(csv_path: str, db_path: str, filter_handles: set):
    """Create SQLite database with only filtered profiles from CSV.
    
    Args:
        csv_path: Path to profile_detail.csv
        db_path: Path to output SQLite database
        filter_handles: Set of linkedin handles to include
    """
    print(f"Creating filtered profile database from {csv_path}...")
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    # Create database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create profiles table
    cursor.execute("""
        CREATE TABLE profiles (
            id TEXT PRIMARY KEY,
            linkedin_handle TEXT UNIQUE NOT NULL,
            name TEXT,
            about TEXT,
            summary TEXT,
            experiences TEXT,
            education TEXT,
            skills TEXT,
            profile_image TEXT,
            full_screenshot TEXT,
            aria_snapshot TEXT,
            meta TEXT,
            updated_at TEXT
        )
    """)
    
    # Create FTS5 virtual table for full-text search
    cursor.execute("""
        CREATE VIRTUAL TABLE profiles_fts USING fts5(
            name,
            about,
            summary,
            experiences,
            education,
            skills,
            content=profiles,
            content_rowid=rowid
        )
    """)
    
    # Create triggers to keep FTS table in sync
    cursor.execute("""
        CREATE TRIGGER profiles_ai AFTER INSERT ON profiles BEGIN
            INSERT INTO profiles_fts(rowid, name, about, summary, experiences, education, skills)
            VALUES (new.rowid, new.name, new.about, new.summary, new.experiences, new.education, new.skills);
        END
    """)
    
    cursor.execute("""
        CREATE TRIGGER profiles_ad AFTER DELETE ON profiles BEGIN
            DELETE FROM profiles_fts WHERE rowid = old.rowid;
        END
    """)
    
    cursor.execute("""
        CREATE TRIGGER profiles_au AFTER UPDATE ON profiles BEGIN
            DELETE FROM profiles_fts WHERE rowid = old.rowid;
            INSERT INTO profiles_fts(rowid, name, about, summary, experiences, education, skills)
            VALUES (new.rowid, new.name, new.about, new.summary, new.experiences, new.education, new.skills);
        END
    """)
    
    # Create index on linkedin_handle for fast lookups
    cursor.execute("CREATE INDEX idx_linkedin_handle ON profiles(linkedin_handle)")
    
    print("Database schema created.")
    
    # Read and insert CSV data (only filtered profiles)
    print("Importing filtered CSV data...")
    inserted_count = 0
    skipped_count = 0
    not_in_filter_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            # Progress indicator for large files
            if (i + 1) % 100000 == 0:
                print(f"  Processed {i + 1} rows, inserted {inserted_count} profiles...")
            
            try:
                # Extract linkedin_handle from linkedin_handle field, fallback to id
                linkedin_handle = row.get('linkedin_handle') or row.get('id')
                
                if not linkedin_handle:
                    skipped_count += 1
                    continue
                
                # Check if this handle is in our filter list
                if linkedin_handle not in filter_handles:
                    not_in_filter_count += 1
                    continue
                
                # Insert into profiles table
                cursor.execute("""
                    INSERT OR IGNORE INTO profiles (
                        id, linkedin_handle, name, about, summary,
                        experiences, education, skills, profile_image,
                        full_screenshot, aria_snapshot, meta, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row.get('id'),
                    linkedin_handle,
                    row.get('name'),
                    row.get('about'),
                    row.get('summary'),
                    row.get('experiences'),
                    row.get('education'),
                    row.get('skills'),
                    row.get('profile_image'),
                    row.get('full_screenshot'),
                    row.get('aria_snapshot'),
                    row.get('meta'),
                    row.get('updated_at')
                ))
                
                inserted_count += 1
                
                if inserted_count % 1000 == 0:
                    print(f"  Inserted {inserted_count} profiles...")
                    conn.commit()
                    
            except Exception as e:
                print(f"  Error inserting row: {e}")
                skipped_count += 1
                continue
    
    # Final commit
    conn.commit()
    
    print(f"\n✓ Filtered database created successfully!")
    print(f"  - Filter list size: {len(filter_handles)}")
    print(f"  - Inserted: {inserted_count} profiles")
    print(f"  - Not in filter: {not_in_filter_count} profiles")
    print(f"  - Skipped (errors): {skipped_count} profiles")
    print(f"  - Database: {db_path}")
    
    # Print some stats
    cursor.execute("SELECT COUNT(*) FROM profiles")
    total = cursor.fetchone()[0]
    print(f"  - Total profiles in DB: {total}")
    
    # Calculate coverage
    if len(filter_handles) > 0:
        coverage = (total / len(filter_handles)) * 100
        print(f"  - Coverage: {coverage:.1f}% ({total}/{len(filter_handles)})")
        
        if total < len(filter_handles):
            missing = len(filter_handles) - total
            print(f"  - Warning: {missing} profiles from filter list not found in source CSV")
    
    conn.close()


def main():
    # Paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / "data" / "profile_detail.csv"
    handle_list_path = script_dir / "data" / "10000-training-linkedin-handle.csv"
    db_path = script_dir / "profiles_training.db"
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    if not handle_list_path.exists():
        print(f"Error: LinkedIn handle list not found at {handle_list_path}")
        return
    
    # Load filter list
    filter_handles = load_linkedin_handles(str(handle_list_path))
    
    # Create filtered database
    create_filtered_database(str(csv_path), str(db_path), filter_handles)
    
    print("\n✓ Training database ready!")
    print(f"  Database location: {db_path}")


if __name__ == "__main__":
    main()

