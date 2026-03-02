"""
Automation MCP Server — Action workflows via n8n webhooks.

Port: 8083
Tools: create_task, send_notification

Run standalone: python -m mcp_server.automation.server
"""
import logging
from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)
mcp = FastMCP("Automation_MCP", port=8083)


def start_server():
    """Starts the Automation MCP server on port 8083."""
    from mcp_server.automation.tools import register_tools
    register_tools(mcp)
    logger.info("Automation MCP → http://127.0.0.1:8083/sse")
    mcp.run(transport="sse")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_server()
