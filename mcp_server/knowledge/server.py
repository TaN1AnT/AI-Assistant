"""
Knowledge MCP Server — Internal document search via Vertex AI RAG.

Port: 8081
Tools: rag_search

Run standalone: python -m mcp_server.knowledge.server
"""
import logging
from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)
mcp = FastMCP("Knowledge_MCP", port=8081)


def start_server():
    """Starts the Knowledge MCP server on port 8081."""
    from mcp_server.knowledge.tools import register_tools
    register_tools(mcp)
    logger.info("Knowledge MCP → http://127.0.0.1:8081/sse")
    mcp.run(transport="sse")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_server()
