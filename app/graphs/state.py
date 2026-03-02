"""
Agent State — Unified state for the ReAct-style supervisor agent.

The access_token is injected by main.py from the user's request
and used by MCP tools that need to authenticate with external APIs via n8n.
"""
import operator
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    State passed through the supervisor → tool_node → supervisor loop.

    Fields:
        messages:     Chat history (accumulated via operator.add).
        user_id:      Authenticated user's identifier.
        user_email:   Email for notifications and audit logging.
        user_role:    Role (admin, sales_rep, guest) for permission checks.
        permissions:  List of allowed tool names.
        access_token: API token from the user, forwarded to n8n webhooks.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    user_id: str
    user_email: str
    user_role: str
    permissions: List[str]
    access_token: str
    # Internal loop-control fields (managed by supervisor graph)
    _iteration: int    # Current loop count (max 10)
    _validation: str   # "COMPLETE" or "INCOMPLETE"