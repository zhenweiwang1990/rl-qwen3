#!/usr/bin/env python3
"""Test that the training database is working correctly (legacy test).

NOTE: This is a legacy test for local database. For MCP integration, use test_mcp.py.
"""

import sys
import os
import sqlite3
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

# Local DB path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "profiles_training.db")

@dataclass
class SearchResult:
    linkedin_handle: str
    snippet: str

@dataclass
class Profile:
    linkedin_handle: str
    name: Optional[str]
    about: Optional[str]
    summary: Optional[str]
    experiences: Optional[str]
    education: Optional[str]
    skills: Optional[str]

def get_conn():
    return sqlite3.connect(f"file:{DEFAULT_DB_PATH}?mode=ro", uri=True, check_same_thread=False)

def search_profiles(keywords: List[str], max_results: int = 10) -> List[SearchResult]:
    """Local DB search for testing."""
    conn = get_conn()
    cursor = conn.cursor()
    fts_query = " ".join(f'"{k.replace(chr(34), chr(34) * 2)}"' for k in keywords)
    sql = """
        SELECT p.linkedin_handle, snippet(profiles_fts, -1, '<b>', '</b>', ' ... ', 20) as snippet
        FROM profiles p JOIN profiles_fts fts ON p.rowid = fts.rowid
        WHERE profiles_fts MATCH ? LIMIT ?;
    """
    cursor.execute(sql, (fts_query, max_results))
    return [SearchResult(linkedin_handle=row[0], snippet=row[1]) for row in cursor.fetchall()]

def read_profile(linkedin_handle: str) -> Optional[Profile]:
    """Local DB read for testing."""
    conn = get_conn()
    cursor = conn.cursor()
    sql = "SELECT linkedin_handle, name, about, summary, experiences, education, skills FROM profiles WHERE linkedin_handle = ?;"
    cursor.execute(sql, (linkedin_handle,))
    row = cursor.fetchone()
    if not row:
        return None
    return Profile(linkedin_handle=row[0], name=row[1], about=row[2], summary=row[3], experiences=row[4], education=row[5], skills=row[6])


def test_database():
    """Test the training database."""
    print(f"Testing database: {DEFAULT_DB_PATH}")
    print()
    
    # Check if database exists
    db_path = Path(DEFAULT_DB_PATH)
    if not db_path.exists():
        print(f"❌ Database not found at {DEFAULT_DB_PATH}")
        return False
    
    print(f"✓ Database exists")
    print(f"  Size: {db_path.stat().st_size / (1024*1024):.2f} MB")
    print()
    
    # Test search
    print("Testing search_profiles()...")
    try:
        results = search_profiles(keywords=["software", "engineer"], max_results=5)
        print(f"✓ Search returned {len(results)} results")
        
        if results:
            print("\nFirst search result:")
            print(f"  LinkedIn: {results[0].linkedin_handle}")
            print(f"  Snippet: {results[0].snippet[:100]}...")
            
            # Test read_profile with first result
            print(f"\nTesting read_profile() with '{results[0].linkedin_handle}'...")
            profile = read_profile(results[0].linkedin_handle)
            
            if profile:
                print(f"✓ Profile found")
                print(f"  Name: {profile.name}")
                print(f"  About: {profile.about[:100] if profile.about else 'N/A'}...")
            else:
                print(f"❌ Profile not found")
                return False
        else:
            print("⚠️  No results found for search")
    
    except Exception as e:
        print(f"❌ Error during search: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("✓ All tests passed! Training database is working correctly.")
    print(f"✓ Database location: {DEFAULT_DB_PATH}")
    print("="*80)
    return True


if __name__ == "__main__":
    success = test_database()
    sys.exit(0 if success else 1)

