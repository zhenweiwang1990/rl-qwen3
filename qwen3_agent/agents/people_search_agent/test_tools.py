#!/usr/bin/env python3
"""Test script for people search agent tools (legacy local DB test).

NOTE: This test uses local database functions for backward compatibility testing.
For MCP integration testing, use test_mcp.py instead.
"""

# Import local DB functions for testing (kept for backward compatibility)
import sqlite3
import os
from typing import List, Optional
from dataclasses import dataclass

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


def test_search():
    """Test search_profiles function."""
    print("Testing search_profiles...")
    
    # Search for AI researchers
    results = search_profiles(
        keywords=["AI", "researcher", "machine learning"],
        max_results=5
    )
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.linkedin_handle}")
        print(f"   Snippet: {result.snippet[:150]}...")
    
    return results


def test_read(linkedin_handle: str):
    """Test read_profile function."""
    print(f"\n\nTesting read_profile for: {linkedin_handle}")
    
    profile = read_profile(linkedin_handle)
    
    if profile:
        print(f"Name: {profile.name}")
        print(f"About: {profile.about[:200] if profile.about else 'N/A'}...")
        print(f"Summary: {profile.summary[:200] if profile.summary else 'N/A'}...")
    else:
        print("Profile not found")
    
    return profile


def test_invalid_profile():
    """Test reading an invalid profile."""
    print("\n\nTesting invalid profile...")
    
    profile = read_profile("non-existent-profile-xyz-123")
    
    if profile is None:
        print("✓ Correctly returned None for invalid profile")
    else:
        print("✗ Should have returned None for invalid profile")


if __name__ == "__main__":
    # Test search
    results = test_search()
    
    # Test read with first result
    if results:
        test_read(results[0].linkedin_handle)
    
    # Test invalid profile
    test_invalid_profile()
    
    print("\n\n✓ All tests completed!")

