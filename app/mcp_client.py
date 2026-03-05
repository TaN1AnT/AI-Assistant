"""
MCP Client — 2-Server Connection Manager.

Connects to Knowledge (:8081/SSE) and Suppa CRM (suppa-mcp-server via stdio).
Discovers all tools at startup and builds a routing map so call_tool()
sends requests to the correct server automatically.

Architecture:
  connect() → for each server → get_tools() → build _tool_route_map
  get_tools() → returns flat list of all tools from all servers
  call_tool(name, args) → looks up server in _tool_route_map → session.call_tool()
"""
import os
import logging
from typing import List, Dict
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger("mcp_client")


class UnifiedMCPClient:
    """
    Connects to multiple MCP servers and provides a single get_tools() interface.
    The LangGraph agent sees all tools as a flat list — this client handles routing.
    """

    def __init__(self):
        self._client: MultiServerMCPClient | None = None
        self._tools_cache: List[BaseTool] = []
        self._tool_route_map: Dict[str, str] = {}  # {"tool_name": "server_name"}

        # Server config from environment
        suppa_env = {
            **os.environ,
            "SUPPA_API_KEY": os.getenv("SUPPA_API_KEY", ""),
            "SUPPA_API_URL": os.getenv("SUPPA_API_URL", "https://sp.modern-expo.com"),
        }

        self.servers = {
            "knowledge":  {"transport": "sse", "url": os.getenv("MCP_KNOWLEDGE_URL",  "http://127.0.0.1:8081/sse")},
            "crm":        {
                "transport": "stdio",
                "command": "node",
                "args": [os.path.join(os.path.dirname(os.path.dirname(__file__)), "suppa-mcp-server", "dist", "index.js")],
                "env": suppa_env,
            },
        }

    @property
    def client(self) -> MultiServerMCPClient:
        """Lazy-initialize the underlying MultiServerMCPClient."""
        if self._client is None:
            self._client = MultiServerMCPClient(self.servers)
            logger.info("MultiServerMCPClient created for: %s", list(self.servers.keys()))
        return self._client

    async def connect(self):
        """
        Discover tools from each server and build the routing map.
        Called once at FastAPI startup.
        """
        all_tools: List[BaseTool] = []

        for server_name in self.servers:
            try:
                server_tools = await self.client.get_tools(server_name=server_name)
                for tool in server_tools:
                    self._tool_route_map[tool.name] = server_name
                all_tools.extend(server_tools)
                logger.info("  %s: %d tools → %s",
                            server_name, len(server_tools),
                            [t.name for t in server_tools])
            except Exception as e:
                logger.warning("  %s: FAILED — %s", server_name, e)

        self._tools_cache = all_tools
        logger.info("Total tools discovered: %d", len(all_tools))

    async def disconnect(self):
        """Clear caches on shutdown."""
        self._tools_cache.clear()
        self._tool_route_map.clear()

    async def get_tools(self) -> List[BaseTool]:
        """
        Return all tools from all servers (cached after first connect()).
        """
        if not self._tools_cache:
            await self.connect()
        return self._tools_cache

    async def call_tool(self, name: str, arguments: dict) -> str:
        """
        Execute a tool by name. Automatically routes to the correct server
        using the _tool_route_map built during connect().
        """
        target = self._tool_route_map.get(name)
        if not target:
            logger.warning("Tool '%s' not in route map — defaulting to 'knowledge'", name)
            target = "knowledge"

        try:
            async with self.client.session(target) as session:
                result = await session.call_tool(name, arguments)
                if result.content:
                    return "\n".join(
                        block.text for block in result.content
                        if hasattr(block, "text")
                    )
                return "Tool executed but returned no content."
        except Exception as e:
            logger.error("Tool '%s' on '%s' failed: %s", name, target, e)
            return f"Error executing '{name}': {e}"

    async def read_resource(self, uri: str, server_name: str = "knowledge") -> str:
        """Read a resource from a specific server (defaults to knowledge)."""
        try:
            async with self.client.session(server_name) as session:
                response = await session.read_resource(uri)
                if response.contents:
                    first = response.contents[0]
                    return first.text if hasattr(first, "text") else str(first)
                return ""
        except Exception as e:
            logger.error("Resource '%s' on '%s' failed: %s", uri, server_name, e)
            raise


# Singleton
mcp_client = UnifiedMCPClient()
