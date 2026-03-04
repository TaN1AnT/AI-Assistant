"""
CRM Tools — Raw data fetchers for querying business data.

Each tool:
1. Accepts access_token + query parameters from the LangGraph agent.
2. POSTs to a dedicated n8n webhook URL (from .env) via async httpx.
3. Returns the RAW JSON response from n8n (no LLM formatting).

The supervisor LLM is responsible for:
- Deciding which tools to call (and in what order)
- Calling multiple tools for complex requests
- Generating ONE final answer from all accumulated raw data

═══════════════════════════════════════════════════════════════
DECOMPOSITION GUIDE — How the supervisor breaks down requests
═══════════════════════════════════════════════════════════════

The supervisor LLM reads the user's question and decides which
raw data tools to call. Here are common decomposition patterns:

Simple requests (1 tool call):
  "Show me the subtasks for T-501"
    → get_subtasks(task_id="T-501")

  "Who approved task T-501?"
    → get_approvals(task_id="T-501")

  "How much time was logged on T-501?"
    → get_time_tracking(task_id="T-501")

Medium requests (2-3 tool calls):
  "What's the status of task T-501?"
    → get_subtasks + get_checklists
      (progress = subtask completion + checklist completion)

  "Is task T-501 blocked?"
    → get_subtasks + get_approvals + get_task_comments
      (check pending approvals, stalled subtasks, blocking comments)

  "Show me task T-501 progress with discussion"
    → get_subtasks + get_checklists + get_task_comments

Complex requests (4-5 tool calls):
  "Give me a full summary of task T-501"
    → get_task_comments + get_subtasks + get_checklists
      + get_approvals + get_time_tracking

  "Prepare a status report for task T-501"
    → get_subtasks + get_checklists + get_approvals
      + get_time_tracking + get_task_comments

  "What work has been done on T-501 and what's left?"
    → get_subtasks + get_checklists + get_time_tracking
      + get_task_comments

Cross-task requests (multiple task IDs):
  "Compare progress on T-501 and T-502"
    → get_subtasks(T-501) + get_subtasks(T-502)
      + get_checklists(T-501) + get_checklists(T-502)

  "Which tasks have pending approvals: T-501, T-502, T-503?"
    → get_approvals(T-501) + get_approvals(T-502)
      + get_approvals(T-503)

Time & resource analysis:
  "How much time did the team spend on T-501?"
    → get_time_tracking(task_id="T-501")

  "Show time breakdown for T-501 with subtask status"
    → get_time_tracking + get_subtasks

Approval workflows:
  "Is T-501 ready for release?"
    → get_approvals + get_subtasks + get_checklists
      (all approvals done? all subtasks done? all checklists checked?)

  "Who still needs to approve T-501?"
    → get_approvals(task_id="T-501")
"""
import os
import logging
from mcp_server.shared.webhook_helper import call_n8n_webhook

logger = logging.getLogger("mcp_server.crm.tools")


# ── Tool registration ──────────────────────────────────────────────────────

def register_tools(mcp):
    """Register CRM query tools with the MCP server."""

    @mcp.tool()
    async def get_task_comments(access_token: str, task_id: str) -> str:
        """
        Retrieve comments/discussion thread for a specific task. Returns raw JSON.

        Use when the user asks about comments, notes, discussion,
        activity, or recent updates on a task.

        Args:
            access_token: API token for authenticating the CRM request.
            task_id: The task ID to fetch comments for (e.g. 'T-501').

        Returns:
            Raw JSON array of comments with author, text, timestamp.
        """
        url = os.getenv("N8N_WEBHOOK_CRM_GET_COMMENTS", "")
        return await call_n8n_webhook(url, access_token, {"task_id": task_id})

    @mcp.tool()
    async def get_checklists(access_token: str, task_id: str) -> str:
        """
        Retrieve checklists for a specific task. Returns raw JSON.

        Use when the user asks about checklists, checkbox items,
        completion progress, or remaining items on a task.

        Args:
            access_token: API token for authenticating the CRM request.
            task_id: The task ID to fetch checklists for (e.g. 'T-501').

        Returns:
            Raw JSON array of checklist items with name, completed status.
        """
        url = os.getenv("N8N_WEBHOOK_CRM_GET_CHECKLISTS", "")
        return await call_n8n_webhook(url, access_token, {"task_id": task_id})

    @mcp.tool()
    async def get_subtasks(access_token: str, task_id: str) -> str:
        """
        Retrieve subtasks (child tasks) for a specific task. Returns raw JSON.

        Use when the user asks about subtasks, sub-items, breakdown of work,
        progress, or status of a task (subtask completion = progress).

        Args:
            access_token: API token for authenticating the CRM request.
            task_id: The parent task ID to fetch subtasks for.

        Returns:
            Raw JSON array of subtasks with id, title, status, assignee.
        """
        url = os.getenv("N8N_WEBHOOK_CRM_GET_SUBTASKS", "")
        return await call_n8n_webhook(url, access_token, {"task_id": task_id})

    @mcp.tool()
    async def get_approvals(access_token: str, task_id: str) -> str:
        """
        Retrieve approval requests and statuses for a task. Returns raw JSON.

        Use when the user asks about approvals, sign-offs,
        review status, who approved/rejected, or readiness for release.

        Args:
            access_token: API token for authenticating the CRM request.
            task_id: The task ID to fetch approvals for.

        Returns:
            Raw JSON array of approvals with approver, status, date.
        """
        url = os.getenv("N8N_WEBHOOK_CRM_GET_APPROVALS", "")
        return await call_n8n_webhook(url, access_token, {"task_id": task_id})

    @mcp.tool()
    async def get_time_tracking(access_token: str, task_id: str) -> str:
        """
        Retrieve time tracking entries for a specific task. Returns raw JSON.

        Use when the user asks about time spent, hours logged,
        time estimates, resource allocation, or team workload.

        Args:
            access_token: API token for authenticating the CRM request.
            task_id: The task ID to fetch time entries for.

        Returns:
            Raw JSON array of time entries with person, hours, date, description.
        """
        url = os.getenv("N8N_WEBHOOK_CRM_GET_TIME", "")
        return await call_n8n_webhook(url, access_token, {"task_id": task_id})
