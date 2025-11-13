#!/usr/bin/env python3
"""Test MCP tools integration for people search agent."""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from qwen3_agent.agents.people_search_agent.mcp_tools import (
    get_mcp_client,
    get_search_and_read_tools,
    call_mcp_tool,
)


def test_list_tools():
    """Test listing available MCP tools."""
    print("=" * 60)
    print("Test 1: List MCP Tools")
    print("=" * 60)
    
    try:
        client = get_mcp_client()
        tools = client.list_tools()
        
        print(f"\n✓ Found {len(tools)} tools:")
        for tool in tools:
            name = tool.get("name", "N/A")
            description = tool.get("description", "N/A")
            print(f"\n  - {name}")
            print(f"    Description: {description}")
            
            # Show input schema
            input_schema = tool.get("inputSchema", {})
            properties = input_schema.get("properties", {})
            if properties:
                print(f"    Parameters:")
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    required = param_name in input_schema.get("required", [])
                    req_str = " (required)" if required else ""
                    print(f"      - {param_name}: {param_type}{req_str}")
                    if param_desc:
                        print(f"        {param_desc}")
        
        return True
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        return False


def test_get_openai_schemas():
    """Test getting OpenAI-format tool schemas."""
    print("\n" + "=" * 60)
    print("Test 2: Get OpenAI Tool Schemas")
    print("=" * 60)
    
    try:
        schemas = get_search_and_read_tools()
        
        print(f"\n✓ Got {len(schemas)} tool schemas in OpenAI format:")
        for schema in schemas:
            print(f"\n{json.dumps(schema, indent=2)}")
        
        return True
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        return False


def test_search_profiles():
    """Test searchProfileTool."""
    print("\n" + "=" * 60)
    print("Test 3: Search Profiles")
    print("=" * 60)
    
    try:
        # Test with simple keywords
        result = call_mcp_tool("searchProfileTool", {
            "keywords": ["software", "engineer"],
            "max_results": 3
        })
        
        print(f"\n✓ Search result:")
        print(json.dumps(result, indent=2))
        
        return True
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_read_profile():
    """Test readProfileTool."""
    print("\n" + "=" * 60)
    print("Test 4: Read Profile")
    print("=" * 60)
    
    try:
        # First search to get a valid handle
        search_result = call_mcp_tool("searchProfileTool", {
            "keywords": ["engineer"],
            "max_results": 1
        })
        
        # Extract linkedin_handle from search result
        linkedin_handle = None
        if isinstance(search_result, dict):
            results = search_result.get("results", search_result.get("data", []))
            if isinstance(results, list) and len(results) > 0:
                first_result = results[0]
                if isinstance(first_result, dict):
                    linkedin_handle = first_result.get("linkedin_handle")
        
        if not linkedin_handle:
            print("\n✗ Could not get linkedin_handle from search")
            return False
        
        print(f"\nReading profile: {linkedin_handle}")
        
        # Read the profile
        result = call_mcp_tool("readProfileTool", {
            "linkedin_handle": linkedin_handle
        })
        
        print(f"\n✓ Read profile:")
        print(json.dumps(result, indent=2))
        
        return True
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MCP Tools Integration Test")
    print("=" * 60)
    
    tests = [
        ("List Tools", test_list_tools),
        ("Get OpenAI Schemas", test_get_openai_schemas),
        ("Search Profiles", test_search_profiles),
        ("Read Profile", test_read_profile),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

