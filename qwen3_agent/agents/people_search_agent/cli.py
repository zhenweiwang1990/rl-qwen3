#!/usr/bin/env python3
"""Interactive CLI for People Search Agent using local Ollama."""

import json
import sys
import requests
from typing import List, Dict, Any

from .mcp_tools import call_mcp_tool
from .agent import PeopleSearchAgent


class OllamaLLM:
    """Ollama LLM client for local inference."""
    
    def __init__(
        self,
        model: str = "qwen3:14b",
        base_url: str = "http://127.0.0.1:11434",
        debug: bool = False,
    ):
        """Initialize Ollama client.
        
        Args:
            model: Model name (e.g., "qwen3:14b")
            base_url: Ollama API base URL
            debug: If True, print full request/response payloads
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/chat"
        self.debug = debug
        self._last_request = None
        self._last_response = None
        
    def chat(
        self, 
        messages: List[Dict[str, Any]], 
        tools: List[Dict[str, Any]] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Send chat request to Ollama.
        
        Args:
            messages: List of message dicts with role and content
            tools: Optional list of tool schemas
            temperature: Sampling temperature
            
        Returns:
            Response dict with message content and tool calls
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if tools:
            payload["tools"] = tools
        
        self._last_request = payload
        
        if self.debug:
            print("\n===== Ollama Request =====")
            print(json.dumps(payload, indent=2))
            print("==========================")
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            self._last_response = data
            
            if self.debug:
                print("\n===== Ollama Response =====")
                print(json.dumps(data, indent=2)[:20000])
                print("==========================")
            
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            raise


class InteractiveCLI:
    """Interactive CLI for people search."""
    
    def __init__(self, model: str = "qwen3:14b", max_turns: int = 10):
        """Initialize CLI.
        
        Args:
            model: Ollama model name
            max_turns: Maximum conversation turns
        """
        self.llm = OllamaLLM(model=model)
        self.agent = PeopleSearchAgent()
        self.max_turns = max_turns
        self.messages = []
        
        # Get system prompt
        from .tasks import PeopleSearchTask
        dummy_task = PeopleSearchTask(
            id="cli",
            query="Interactive search",
            expected_profiles=[],
        )
        self.system_prompt = self.agent.get_system_prompt(dummy_task)
        
        # Get tools
        self.tools = self.agent.get_tools_schema()
        
        print(f"🤖 People Search Agent CLI")
        print(f"📡 Using Ollama model: {model}")
        print(f"🔧 Tools available: search_profiles, read_profile, return_final_answer")
        print(f"💡 Max turns per query: {max_turns}")
        print()
    
    def format_tool_result(self, tool_name: str, result: Any) -> str:
        """Format tool result for display.
        
        Args:
            tool_name: Name of tool
            result: Tool result
            
        Returns:
            Formatted string
        """
        if tool_name == "search_profiles":
            if isinstance(result, dict) and result.get("error"):
                return f"❌ Error: {result['error']}"
            
            if isinstance(result, list):
                if len(result) == 0:
                    return "No results found"
                
                output = f"Found {len(result)} profiles:\n"
                for i, item in enumerate(result, 1):
                    handle = item.get("linkedin_handle", "unknown")
                    snippet = item.get("snippet", "")
                    # Clean up snippet
                    snippet = snippet.replace("<b>", "").replace("</b>", "")
                    output += f"\n{i}. {handle}\n"
                    output += f"   {snippet[:150]}...\n"
                return output
        
        elif tool_name == "read_profile":
            if isinstance(result, dict):
                if result.get("error"):
                    return f"❌ Profile not found"
                
                output = "Profile Details:\n"
                output += f"  Handle: {result.get('linkedin_handle', 'N/A')}\n"
                output += f"  Name: {result.get('name', 'N/A')}\n"
                
                about = result.get('about', '')
                if about:
                    output += f"  About: {about[:200]}...\n"
                
                summary = result.get('summary', '')
                if summary:
                    output += f"  Summary: {summary[:200]}...\n"
                
                return output
        
        elif tool_name == "return_final_answer":
            if isinstance(result, dict):
                profiles = result.get("profiles", [])
                if len(profiles) == 0:
                    return "✅ No matching profiles found"
                
                output = f"✅ Found {len(profiles)} matching profiles:\n"
                for i, handle in enumerate(profiles, 1):
                    output += f"  {i}. {handle}\n"
                return output
        
        return str(result)
    
    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> tuple[Any, str]:
        """Execute a tool.
        
        Args:
            tool_name: Name of tool
            tool_args: Tool arguments
            
        Returns:
            Tuple of (result, formatted_display)
        """
        print(f"\n🔧 Executing: {tool_name}")
        print(f"   Args: {json.dumps(tool_args, indent=2)}")
        
        try:
            if tool_name in ("search_profiles", "searchProfileTool"):
                # Call MCP tool directly
                result_data = call_mcp_tool("searchProfileTool", tool_args)
                display = self.format_tool_result(tool_name, result_data)
                return result_data, display
            
            elif tool_name in ("read_profile", "readProfileTool"):
                # Call MCP tool directly
                result_data = call_mcp_tool("readProfileTool", tool_args)
                display = self.format_tool_result(tool_name, result_data)
                return result_data, display
            
            elif tool_name == "return_final_answer":
                result_data = tool_args
                display = self.format_tool_result(tool_name, tool_args)
                return result_data, display
            
            else:
                error_msg = f"Unknown tool: {tool_name}"
                print(f"❌ {error_msg}")
                return {"error": error_msg}, error_msg
        
        except Exception as e:
            error_msg = f"Error executing tool: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}, error_msg
    
    def process_query(self, query: str) -> str:
        """Process a user query.
        
        Args:
            query: User's search query
            
        Returns:
            Final answer
        """
        # Reset messages for new query
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        
        print(f"\n💭 Processing query: {query}")
        print("=" * 80)
        
        final_answer = None
        
        for turn in range(self.max_turns):
            print(f"\n🔄 Turn {turn + 1}/{self.max_turns}")
            
            # Get LLM response
            try:
                response = self.llm.chat(self.messages, tools=self.tools)
            except Exception as e:
                print(f"\n❌ Error getting LLM response: {e}")
                return "Sorry, I encountered an error processing your query."
            
            # Extract message
            message = response.get("message", {})
            
            # Check for tool calls
            tool_calls = message.get("tool_calls", [])
            
            if not tool_calls:
                # No tool calls, check if we have a text response
                content = message.get("content", "")
                if content:
                    print(f"\n💬 LLM says: {content}")
                    return content
                else:
                    print("\n⚠️  No tool calls or response from LLM")
                    continue
            
            # Add assistant message to history
            self.messages.append(message)
            
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
                    print(f"❌ Error parsing tool arguments: {e}")
                    continue
                
                # Execute tool
                result, display = self.execute_tool(tool_name, tool_args)
                
                # Display result
                print(f"\n📊 Result:")
                print(display)
                
                # Add tool result to messages
                tool_message = {
                    "role": "tool",
                    "content": json.dumps(result)
                }
                self.messages.append(tool_message)
                
                # Check if this is final answer
                if tool_name == "return_final_answer":
                    final_answer = display
                    break
            
            # If we have final answer, break
            if final_answer:
                break
        
        if final_answer:
            return final_answer
        else:
            return "⚠️  Reached maximum turns without finding an answer."
    
    def run(self):
        """Run interactive CLI."""
        print("=" * 80)
        print("🚀 Ready! Enter your people search queries.")
        print("💡 Examples:")
        print("   - 'Find AI researchers with machine learning experience'")
        print("   - 'Show me CTOs in fintech companies'")
        print("   - 'Search for data scientists in San Francisco'")
        print("\nType 'quit' or 'exit' to quit.\n")
        print("=" * 80)
        
        while True:
            try:
                # Get user input
                query = input("\n🔍 Your query: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Process query
                answer = self.process_query(query)
                
                print("\n" + "=" * 80)
                print("✨ Final Answer:")
                print(answer)
                print("=" * 80)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interactive People Search Agent CLI using local Ollama"
    )
    parser.add_argument(
        "--model",
        default="qwen3:14b",
        help="Ollama model name (default: qwen3:14b)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum turns per query (default: 10)"
    )
    parser.add_argument(
        "--query",
        help="Single query to process (non-interactive mode)"
    )
    
    args = parser.parse_args()
    
    # Create CLI
    cli = InteractiveCLI(model=args.model, max_turns=args.max_turns)
    
    # Single query mode or interactive mode
    if args.query:
        answer = cli.process_query(args.query)
        print("\n" + "=" * 80)
        print("✨ Final Answer:")
        print(answer)
        print("=" * 80)
    else:
        cli.run()


if __name__ == "__main__":
    main()

