#!/usr/bin/env python3
"""Test script for CLI (without requiring Ollama)."""

import sys


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        from .cli import OllamaLLM, InteractiveCLI
        print("✓ CLI imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_tool_formatting():
    """Test tool result formatting."""
    print("\nTesting tool result formatting...")
    
    from .cli import InteractiveCLI
    from .tools import SearchResult
    from dataclasses import asdict
    
    cli = InteractiveCLI()
    
    # Test search result formatting
    search_results = [
        asdict(SearchResult(
            linkedin_handle="test-profile-1",
            snippet="Test snippet with <b>keywords</b>"
        ))
    ]
    
    formatted = cli.format_tool_result("search_profiles", search_results)
    assert "test-profile-1" in formatted
    assert "Found 1 profiles" in formatted
    print("✓ Search result formatting works")
    
    # Test profile result formatting
    profile_result = {
        "linkedin_handle": "test-profile",
        "name": "Test User",
        "about": "Test about section",
    }
    
    formatted = cli.format_tool_result("read_profile", profile_result)
    assert "Test User" in formatted
    assert "test-profile" in formatted
    print("✓ Profile result formatting works")
    
    # Test final answer formatting
    answer_result = {
        "profiles": ["profile-1", "profile-2", "profile-3"]
    }
    
    formatted = cli.format_tool_result("return_final_answer", answer_result)
    assert "3 matching profiles" in formatted
    assert "profile-1" in formatted
    print("✓ Final answer formatting works")
    
    return True


def test_tool_execution():
    """Test tool execution (actual tools, not LLM)."""
    print("\nTesting tool execution...")
    
    from .cli import InteractiveCLI
    
    cli = InteractiveCLI()
    
    # Test search
    try:
        result, display = cli.execute_tool(
            "search_profiles",
            {"keywords": ["AI", "researcher"], "max_results": 3}
        )
        assert isinstance(result, list)
        assert "profiles" in display or "results" in display.lower()
        print(f"✓ Search executed successfully (found {len(result)} results)")
    except Exception as e:
        print(f"✗ Search execution failed: {e}")
        return False
    
    # Test read (with a likely valid profile from search)
    if result:
        first_handle = result[0].get("linkedin_handle")
        if first_handle:
            try:
                profile_result, profile_display = cli.execute_tool(
                    "read_profile",
                    {"linkedin_handle": first_handle}
                )
                assert isinstance(profile_result, dict)
                print(f"✓ Read profile executed successfully")
            except Exception as e:
                print(f"✗ Read profile failed: {e}")
                return False
    
    # Test return_final_answer
    try:
        answer_result, answer_display = cli.execute_tool(
            "return_final_answer",
            {"profiles": ["test-1", "test-2"]}
        )
        assert "profiles" in answer_result
        print("✓ Return final answer executed successfully")
    except Exception as e:
        print(f"✗ Return final answer failed: {e}")
        return False
    
    return True


def test_system_prompt():
    """Test system prompt generation."""
    print("\nTesting system prompt generation...")
    
    from .cli import InteractiveCLI
    
    cli = InteractiveCLI()
    
    assert cli.system_prompt
    assert "people search agent" in cli.system_prompt.lower()
    assert "tools" in cli.system_prompt.lower()
    print("✓ System prompt generated")
    print(f"  Length: {len(cli.system_prompt)} characters")
    
    return True


def test_tools_schema():
    """Test tools schema."""
    print("\nTesting tools schema...")
    
    from .cli import InteractiveCLI
    
    cli = InteractiveCLI()
    
    assert len(cli.tools) == 3
    tool_names = [t["function"]["name"] for t in cli.tools]
    assert "search_profiles" in tool_names
    assert "read_profile" in tool_names
    assert "return_final_answer" in tool_names
    print("✓ All 3 tools available")
    print(f"  Tools: {tool_names}")
    
    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("CLI Test Suite (No Ollama Required)")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Tool Formatting", test_tool_formatting),
        ("Tool Execution", test_tool_execution),
        ("System Prompt", test_system_prompt),
        ("Tools Schema", test_tools_schema),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 80)
    print("Test Results:")
    print("=" * 80)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! CLI is ready to use.")
        print("\nTo use the CLI:")
        print("  1. Make sure Ollama is running: ollama serve")
        print("  2. Pull the model: ollama pull qwen3:14b")
        print("  3. Run the CLI: ./scripts/people_search_cli.sh")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

