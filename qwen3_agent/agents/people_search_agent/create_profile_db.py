#!/usr/bin/env python3
"""Create SQLite database from profile_detail.csv for people search agent."""

import sqlite3
import csv
import json
import os
import sys
from pathlib import Path

# Increase CSV field size limit to handle large fields
csv.field_size_limit(sys.maxsize)


def create_profile_database(csv_path: str, db_path: str):
    """Create SQLite database with profile data from CSV.
    
    Args:
        csv_path: Path to profile_detail.csv
        db_path: Path to output SQLite database
    """
    print(f"Creating profile database from {csv_path}...")
    
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
    
    # Read and insert CSV data
    print("Importing CSV data...")
    inserted_count = 0
    skipped_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                # Extract linkedin_handle from linkedin_handle field, fallback to id
                linkedin_handle = row.get('linkedin_handle') or row.get('id')
                
                if not linkedin_handle:
                    skipped_count += 1
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
                
                if inserted_count % 10000 == 0:
                    print(f"  Inserted {inserted_count} profiles...")
                    conn.commit()
                    
            except Exception as e:
                print(f"  Error inserting row: {e}")
                skipped_count += 1
                continue
    
    # Final commit
    conn.commit()
    
    print(f"\n✓ Database created successfully!")
    print(f"  - Inserted: {inserted_count} profiles")
    print(f"  - Skipped: {skipped_count} profiles")
    print(f"  - Database: {db_path}")
    
    # Print some stats
    cursor.execute("SELECT COUNT(*) FROM profiles")
    total = cursor.fetchone()[0]
    print(f"  - Total profiles in DB: {total}")
    
    conn.close()


def main():
    # Paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / "data" / "profile_detail.csv"
    db_path = script_dir / "profiles.db"
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    create_profile_database(str(csv_path), str(db_path))


if __name__ == "__main__":
    main()

