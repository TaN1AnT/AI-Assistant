"""
CRM MCP Server — Raw data queries via n8n webhooks.

Port: 8082
Tools (all return raw JSON, no LLM formatting):
  get_task_comments, get_checklists,
  get_subtasks, get_approvals, get_time_tracking

The supervisor LLM orchestrates multi-step data gathering
and generates one comprehensive answer from all raw results.

Run standalone: python -m mcp_server.crm.server
"""
import logging
from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)
mcp = FastMCP("CRM_MCP", port=8082)


def start_server():
    """Starts the CRM MCP server on port 8082."""
    from mcp_server.crm.tools import register_tools
    register_tools(mcp)
    logger.info("CRM MCP → http://127.0.0.1:8082/sse")
    mcp.run(transport="sse")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_server()
