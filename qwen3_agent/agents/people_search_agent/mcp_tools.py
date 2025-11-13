"""MCP-based tools for people search agent.

This module provides direct MCP tool integration without additional wrapping layers.
It fetches tool schemas from the MCP server and converts them to OpenAI function calling format.
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional
from uuid import uuid4
from urllib import request, error
from urllib.parse import urljoin
from http.cookiejar import CookieJar

logger = logging.getLogger(__name__)

# MCP Server configuration
# HTTP JSON-RPC endpoint
DEFAULT_MCP_HTTP_BASE = os.environ.get(
    "PROFILE_MCP_BASE_URL",
    "http://localhost:4111/api/mcp/profile-mcp-server/mcp",
)
# SSE initialization endpoint (session/bootstrap). Can be set explicitly, or
# inferred by replacing a trailing '/mcp' with '/sse'.
DEFAULT_MCP_SSE_BASE = os.environ.get(
    "PROFILE_MCP_SSE_URL",
    DEFAULT_MCP_HTTP_BASE.rstrip("/").rsplit("/", 1)[0] + "/sse"
    if DEFAULT_MCP_HTTP_BASE.rstrip("/").endswith("/mcp")
    else "http://localhost:4111/api/mcp/profile-mcp-server/sse",
)

# Mastra dev playground header toggle (optional)
MASTRA_DEV_PLAYGROUND = os.environ.get("MASTRA_DEV_PLAYGROUND", "false").lower() in ("1", "true", "yes")
MASTRA_COOKIES = os.environ.get("MASTRA_COOKIES", "")


class MCPToolsClient:
    """Client for interacting with MCP server tools."""
    
    def __init__(
        self,
        base_url: str = DEFAULT_MCP_HTTP_BASE,
        timeout: float = 20.0,
        sse_url: str = DEFAULT_MCP_SSE_BASE,
    ):
        """Initialize MCP tools client.
        
        Args:
            base_url: Base URL for MCP HTTP endpoint
            timeout: Request timeout in seconds
            sse_url: SSE URL for initialize handshake
        """
        self.base_url = base_url
        self.sse_url = sse_url
        self.timeout = timeout
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        # Maintain cookies (for session id set by initialize)
        self._cookie_jar = CookieJar()
        self._opener = request.build_opener(request.HTTPCookieProcessor(self._cookie_jar))
        self._initialized = False
        self._session_id: Optional[str] = None
    
    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            # Accept both JSON and SSE for initialize flows
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["x-session-id"] = self._session_id
        if MASTRA_DEV_PLAYGROUND:
            headers["x-mastra-dev-playground"] = "true"
        if MASTRA_COOKIES:
            headers["Cookie"] = MASTRA_COOKIES
        return headers
    
    def _ensure_initialized(self) -> None:
        """Perform MCP initialize handshake once to establish a session."""
        if self._initialized:
            return
        # Ensure we have a client session id ready before initialize
        if not self._session_id:
            self._session_id = str(uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "rl-qwen3", "version": "0.1.0"},
                "capabilities": {},
                # Many gateways accept/expect a session id during initialize
                "sessionId": self._session_id,
            },
        }
        errors: list[str] = []
        # Prefer SSE initialize endpoint first, then fall back to HTTP variants
        init_urls = [
            self.sse_url,
            urljoin(self.base_url.rstrip("/") + "/", "initialize"),
            self.base_url,
        ]
        for url in init_urls:
            try:
                # For initialize, include session header too
                result = self._http_post_json(url, payload)
                # Extract session id from JSON result if present
                if isinstance(result, dict):
                    res_obj = result.get("result") or {}
                    sid = (
                        res_obj.get("sessionId")
                        or result.get("sessionId")
                        or res_obj.get("session_id")
                        or result.get("session_id")
                    )
                    if isinstance(sid, str) and sid:
                        self._session_id = sid
                # Fallback to cookie-based session ids if provided
                if not self._session_id:
                    for c in self._cookie_jar:
                        if c.name.lower() in ("sessionid", "session_id", "mcp_session", "mcpsessionid"):
                            self._session_id = c.value
                self._initialized = True
                return
            except Exception as e:
                errors.append(f"{url} init failed: {e}")
        raise RuntimeError("MCP initialize failed: " + " | ".join(errors))
    
    def _http_post_json(self, url: str, payload: dict) -> dict:
        """POST JSON to url and return parsed JSON with detailed errors."""
        data = json.dumps(payload).encode("utf-8")
        headers = self._build_headers()
        req = request.Request(url=url, data=data, headers=headers, method="POST")
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                raw = resp.read()
                text = raw.decode("utf-8")
                ctype = resp.headers.get("Content-Type", "")
                # Capture session id header if provided by server
                sid_hdr = (
                    resp.headers.get("x-session-id")
                    or resp.headers.get("X-Session-Id")
                    or resp.headers.get("X-Session-ID")
                )
                if sid_hdr and not self._session_id:
                    self._session_id = sid_hdr
                # Handle SSE (text/event-stream) by extracting JSON from 'data:' lines
                if "text/event-stream" in ctype or text.lstrip().startswith(("event:", "data:")):
                    # Parse Server-Sent Events; take the latest complete 'data:' JSON block
                    last_json = None
                    data_buf = []
                    for line in text.splitlines():
                        if line.startswith("data:"):
                            data_line = line[len("data:"):].strip()
                            data_buf.append(data_line)
                        elif line.strip() == "" and data_buf:
                            # End of one event
                            candidate = "\n".join(data_buf).strip()
                            data_buf = []
                            try:
                                last_json = json.loads(candidate)
                            except Exception:
                                # ignore and continue to next event
                                pass
                    # Flush remaining buffer
                    if data_buf:
                        candidate = "\n".join(data_buf).strip()
                        try:
                            last_json = json.loads(candidate)
                        except Exception:
                            pass
                    if last_json is not None:
                        return last_json
                    # If we couldn't parse, surface a helpful error
                    raise RuntimeError(f"MCP SSE response not JSON-decodable from {url}: {text[:500]}")
                # Normal JSON response
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    raise RuntimeError(f"MCP non-JSON response from {url}: {text[:500]}")
        except error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
            raise RuntimeError(f"MCP HTTP error {e.code} at {url}: {body or e.reason}")
        except error.URLError as e:
            raise RuntimeError(f"MCP connection error to {url}: {e.reason}")
        except Exception as e:
            raise RuntimeError(f"MCP request failed to {url}: {e}")
    
    def list_tools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """List available tools from MCP server.
        
        Args:
            force_refresh: Force refresh cache
            
        Returns:
            List of tool definitions in MCP format
        """
        if self._tools_cache is not None and not force_refresh:
            return self._tools_cache
        
        errors: list[str] = []
        tools: list[Dict[str, Any]] | None = None
        
        # Strategy 0: Try Mastra REST API first (simplest, no session required)
        # GET http://localhost:4111/api/tools
        try:
            # Extract base without MCP path (e.g., http://localhost:4111)
            base_parts = self.base_url.split("/api/mcp/")
            if len(base_parts) >= 2:
                mastra_base = base_parts[0]
                mastra_tools_url = f"{mastra_base}/api/tools"
                
                req = request.Request(
                    mastra_tools_url,
                    headers={
                        "Accept": "application/json",
                    }
                )
                
                with request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode())
                    
                    # Convert Mastra format to MCP format
                    # Mastra returns: {toolName: {id, description, inputSchema, outputSchema}, ...}
                    # MCP expects: [{name, description, inputSchema}, ...]
                    tools = []
                    for tool_name, tool_def in data.items():
                        input_schema_str = tool_def.get("inputSchema", "{}")
                        # Parse the JSON string
                        if isinstance(input_schema_str, str):
                            try:
                                # Format: '{"json": {...}}'
                                schema_wrapper = json.loads(input_schema_str)
                                input_schema = schema_wrapper.get("json", {})
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse inputSchema for {tool_name}")
                                input_schema = {"type": "object", "properties": {}}
                        else:
                            input_schema = input_schema_str
                        
                        tools.append({
                            "name": tool_name,
                            "description": tool_def.get("description", ""),
                            "inputSchema": input_schema,
                        })
                    
                    logger.info(f"Retrieved {len(tools)} tools from Mastra REST API")
                    self._tools_cache = tools
                    return tools
        except Exception as e:
            errors.append(f"Mastra REST API failed: {e}")
        
        # Fallback: Try MCP JSON-RPC protocols
        # Ensure session cookie exists
        try:
            self._ensure_initialized()
        except Exception as e:
            errors.append(str(e))
        
        # Strategy 1: JSON-RPC to base URL
        try:
            response = self._http_post_json(self.base_url, {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {
                    # Some gateways require explicit session id in params
                    **({"sessionId": self._session_id} if self._session_id else {})
                }
            })
            if "error" in response:
                raise RuntimeError(str(response["error"]))
            result = response.get("result", {})
            tools = result.get("tools")
        except Exception as e:
            errors.append(f"base JSON-RPC failed: {e}")
        
        # Strategy 2: JSON-RPC to {base}/tools/list
        if tools is None:
            try:
                url = urljoin(self.base_url.rstrip("/") + "/", "tools/list")
                response = self._http_post_json(url, {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {
                        **({"sessionId": self._session_id} if self._session_id else {})
                    }
                })
                if "error" in response:
                    raise RuntimeError(str(response["error"]))
                result = response.get("result", {})
                tools = result.get("tools")
            except Exception as e:
                errors.append(f"{url} JSON-RPC failed: {e}")
        
        # Strategy 3: Plain POST {base}/tools/list with empty/body {}
        if tools is None:
            try:
                url = urljoin(self.base_url.rstrip("/") + "/", "tools/list")
                body = {}  # some gateways accept empty body with cookie/header session
                if self._session_id:
                    body = {"sessionId": self._session_id}
                response = self._http_post_json(url, body)
                # Accept formats: {"tools":[...]}, or {"result":{"tools":[...]}}
                if isinstance(response, dict):
                    if "tools" in response and isinstance(response["tools"], list):
                        tools = response["tools"]
                    elif "result" in response and isinstance(response["result"], dict):
                        maybe = response["result"].get("tools")
                        if isinstance(maybe, list):
                            tools = maybe
            except Exception as e:
                errors.append(f"{url} plain POST failed: {e}")
        
        if tools is None:
            raise RuntimeError("Unable to list MCP tools. Tried multiple endpoints: " + " | ".join(errors))
        
        self._tools_cache = tools
        logger.info(f"Retrieved {len(tools)} tools from MCP server")
        return tools
    
    def _tools_root(self) -> str:
        """Derive the '/tools/' root from HTTP base (strip trailing '/mcp' or '/sse')."""
        base = self.base_url.rstrip("/")
        if base.endswith("/mcp") or base.endswith("/sse"):
            base = base.rsplit("/", 1)[0]
        return base + "/tools/"
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        errors: list[str] = []
        
        # Strategy 0: Mastra-style HTTP execute endpoint (no JSON-RPC)
        # POST {root}/tools/{tool}/execute  body: {"data": {...}, "runtimeContext": {}}
        try:
            url = urljoin(self._tools_root(), f"{tool_name}/execute")
            body = {
                "data": arguments,
                "runtimeContext": {},
            }
            # For execute, Accept pure JSON
            data = json.dumps(body).encode("utf-8")
            headers = self._build_headers()
            headers["Accept"] = "application/json"
            req = request.Request(url=url, data=data, headers=headers, method="POST")
            with self._opener.open(req, timeout=self.timeout) as resp:
                raw = resp.read()
                text = raw.decode("utf-8")
                try:
                    direct = json.loads(text)
                except json.JSONDecodeError:
                    raise RuntimeError(f"Non-JSON execute response at {url}: {text[:500]}")
                # Mastra playground returns result under 'result' or direct object
                result_payload = direct.get("result", direct) if isinstance(direct, dict) else direct
                return result_payload
        except Exception as e:
            errors.append(f"{url} execute failed: {e}")
        # Ensure session cookie exists
        try:
            self._ensure_initialized()
        except Exception as e:
            errors.append(str(e))
        
        # Strategy 1: JSON-RPC to base URL
        try:
            response = self._http_post_json(self.base_url, {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                    **({"sessionId": self._session_id} if self._session_id else {})
                }
            })
            if "error" in response:
                err = response["error"]
                raise RuntimeError(err.get("message", str(err)))
            result = response.get("result", {})
        except Exception as e:
            errors.append(f"base JSON-RPC failed: {e}")
            result = None
        
        # Strategy 2: JSON-RPC to {base}/tools/call
        if result is None:
            try:
                url = urljoin(self.base_url.rstrip("/") + "/", "tools/call")
                response = self._http_post_json(url, {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments,
                        **({"sessionId": self._session_id} if self._session_id else {})
                    }
                })
                if "error" in response:
                    err = response["error"]
                    raise RuntimeError(err.get("message", str(err)))
                result = response.get("result", {})
            except Exception as e:
                errors.append(f"{url} JSON-RPC failed: {e}")
        
        # Strategy 3: Plain POST {base}/tools/{name} with {"arguments": {...}}
        if result is None:
            try:
                url = urljoin(self.base_url.rstrip("/") + "/", f"tools/{tool_name}")
                body = {"arguments": arguments}
                if self._session_id:
                    body["sessionId"] = self._session_id
                response = self._http_post_json(url, body)
                # In this style, response itself may be the payload/result
                # Normalize to a common dict
                if isinstance(response, dict):
                    result = response
                else:
                    result = {"content": response}
            except Exception as e:
                errors.append(f"{url} plain POST failed: {e}")
        
        if result is None:
            raise RuntimeError("Unable to call MCP tool. Attempts: " + " | ".join(errors))
        
        # Normalize content extraction
        content = result.get("content")
        if content is None and isinstance(result, dict):
            # Some gateways return raw fields
            content = result.get("data") or result.get("result") or result
        
        # If content is typical MCP message list, extract text
        if isinstance(content, list) and len(content) > 0:
            first_item = content[0]
            if isinstance(first_item, dict) and first_item.get("type") == "text":
                text_content = first_item.get("text", "")
                try:
                    return json.loads(text_content)
                except json.JSONDecodeError:
                    return text_content
        
        return content
    
    def mcp_to_openai_tool_schema(self, mcp_tool: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MCP tool schema to OpenAI function calling format.
        
        Args:
            mcp_tool: MCP tool definition
            
        Returns:
            OpenAI-format tool schema
        """
        name = mcp_tool.get("name", "")
        description = mcp_tool.get("description", "")
        input_schema = mcp_tool.get("inputSchema", {})
        
        # MCP uses JSON Schema, which is compatible with OpenAI's format
        # Just need to wrap it in the right structure
        openai_schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": input_schema
            }
        }
        
        return openai_schema
    
    def get_openai_tools_schemas(
        self,
        tool_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get OpenAI-format tool schemas for specified tools.
        
        Args:
            tool_names: List of tool names to include. If None, include all tools.
            
        Returns:
            List of OpenAI-format tool schemas
        """
        try:
            mcp_tools = self.list_tools()
            if tool_names:
                mcp_tools = [t for t in mcp_tools if t.get("name") in tool_names]
            return [self.mcp_to_openai_tool_schema(t) for t in mcp_tools]
        except Exception as e:
            # Fallback: return permissive schemas for known tools
            logger.warning(f"Falling back to permissive schemas due to list_tools failure: {e}")
            fallback_names = tool_names or ["searchProfileTool", "readProfileTool"]
            schemas: List[Dict[str, Any]] = []
            for name in fallback_names:
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": f"Direct MCP tool: {name}",
                        "parameters": {
                            "type": "object",
                            # Accept any fields to avoid mismatches with server schema
                            "properties": {},
                            "additionalProperties": True,
                        }
                    }
                })
            return schemas


# Global client instance
_mcp_client: Optional[MCPToolsClient] = None


def get_mcp_client() -> MCPToolsClient:
    """Get or create global MCP client instance."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPToolsClient()
    return _mcp_client


def get_search_and_read_tools() -> List[Dict[str, Any]]:
    """Get OpenAI-format schemas for searchProfileTool and readProfileTool.
    
    Returns:
        List of tool schemas in OpenAI function calling format
    """
    client = get_mcp_client()
    return client.get_openai_tools_schemas(
        tool_names=["searchProfileTool", "readProfileTool"]
    )


def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Call an MCP tool directly.
    
    Args:
        tool_name: Name of the MCP tool
        arguments: Tool arguments
        
    Returns:
        Tool execution result
    """
    client = get_mcp_client()
    return client.call_tool(tool_name, arguments)

