"""
CRM Tools — Webhook-proxy tools for querying business data.

Each tool:
1. Accepts access_token + query parameters from the LangGraph agent.
2. POSTs a JSON payload to a DEDICATED n8n webhook URL (from .env).
3. n8n uses the token to call the real CRM HTTP API.
4. Receives the JSON response from n8n.
5. Uses Gemini (via llm_helper) to format a human-readable answer.
"""
import os
import logging
from mcp_server.shared.llm_helper import generate_answer
from mcp_server.shared.webhook_helper import call_n8n_webhook

logger = logging.getLogger("mcp_server.crm.tools")


async def _call_n8n_crm(webhook_url: str, token: str, params: dict) -> str:
    """Proxy to the shared async webhook helper for CRM-specific calls."""
    return await call_n8n_webhook(webhook_url, token, params)




# ── Tool registration ──────────────────────────────────────────────────────

def register_tools(mcp):
    """Register CRM query tools with the MCP server."""

    @mcp.tool()
    async def get_tasks(access_token: str, deal_id: str) -> str:
        """
        Retrieve tasks linked to a specific deal.

        Use when the user asks about tasks, to-do items, or action items for a deal.

        Args:
            access_token: API token for authenticating the CRM request.
            deal_id: The deal ID to fetch tasks for (e.g. 'D-100').

        Returns:
            A formatted list of tasks.
        """
        url = os.getenv("N8N_WEBHOOK_CRM_GET_TASKS")
        raw = await _call_n8n_crm(url, access_token, {"deal_id": deal_id})
        if raw.startswith("Error:"):
            return raw

        return generate_answer(
            "You are a project manager. List tasks with bullet points showing "
            "ID, title, status, and assignee. If empty, say 'No tasks found.'",
            raw, f"Get tasks for deal {deal_id}"
        )

    @mcp.tool()
    async def get_task_comments(access_token: str, task_id: str) -> str:
        """
        Retrieve comments/discussion thread for a specific task.

        Use when the user asks about comments, notes, or discussion on a task.

        Args:
            access_token: API token for authenticating the CRM request.
            task_id: The task ID to fetch comments for (e.g. 'T-501').

        Returns:
            A chronological summary of the comment thread.
        """
        url = os.getenv("N8N_WEBHOOK_CRM_GET_COMMENTS")
        raw = await _call_n8n_crm(url, access_token, {"task_id": task_id})
        if raw.startswith("Error:"):
            return raw

        return generate_answer(
            "Summarize the comment thread chronologically. "
            "If empty, say 'No comments on this task yet.'",
            raw, f"Get comments for task {task_id}"
        )

    # ── Checklists ─────────────────────────────────────────────────────────

    @mcp.tool()
    async def get_checklists(access_token: str, task_id: str) -> str:
        """
        Retrieve checklists for a specific task.

        Use when the user asks about checklists, checkbox items,
        or completion progress on a task.

        Args:
            access_token: API token for authenticating the CRM request.
            task_id: The task ID to fetch checklists for (e.g. 'T-501').

        Returns:
            A formatted list of checklist items with their completion status.
        """
        url = os.getenv("N8N_WEBHOOK_CRM_GET_CHECKLISTS")
        raw = await _call_n8n_crm(url, access_token, {"task_id": task_id})
        if raw.startswith("Error:"):
            return raw

        return generate_answer(
            "You are a project manager. List each checklist item with a checkbox "
            "(✅ for done, ⬜ for pending). Show completion percentage at the top. "
            "If empty, say 'No checklists found for this task.'",
            raw, f"Get checklists for task {task_id}"
        )

    # ── Subtasks ───────────────────────────────────────────────────────────

    @mcp.tool()
    async def get_subtasks(access_token: str, task_id: str) -> str:
        """
        Retrieve subtasks (child tasks) for a specific task.

        Use when the user asks about subtasks, sub-items,
        or breakdown of work within a task.

        Args:
            access_token: API token for authenticating the CRM request.
            task_id: The parent task ID to fetch subtasks for.

        Returns:
            A formatted list of subtasks with status and assignee.
        """
        url = os.getenv("N8N_WEBHOOK_CRM_GET_SUBTASKS")
        raw = await _call_n8n_crm(url, access_token, {"task_id": task_id})
        if raw.startswith("Error:"):
            return raw

        return generate_answer(
            "You are a project manager. List subtasks with bullet points showing "
            "ID, title, status, and assignee. Show count of completed vs total. "
            "If empty, say 'No subtasks found for this task.'",
            raw, f"Get subtasks for task {task_id}"
        )

    # ── Approvals ──────────────────────────────────────────────────────────

    @mcp.tool()
    async def get_approvals(access_token: str, task_id: str) -> str:
        """
        Retrieve approval requests and their statuses for a task.

        Use when the user asks about approvals, sign-offs,
        review status, or who approved/rejected something.

        Args:
            access_token: API token for authenticating the CRM request.
            task_id: The task ID to fetch approvals for.

        Returns:
            A summary of approval statuses per approver.
        """
        url = os.getenv("N8N_WEBHOOK_CRM_GET_APPROVALS")
        raw = await _call_n8n_crm(url, access_token, {"task_id": task_id})
        if raw.startswith("Error:"):
            return raw

        return generate_answer(
            "You are a compliance assistant. For each approval, show: "
            "approver name, status (✅ Approved / ❌ Rejected / ⏳ Pending), "
            "and date. If empty, say 'No approval requests found for this task.'",
            raw, f"Get approvals for task {task_id}"
        )

    # ── Time Tracking ──────────────────────────────────────────────────────

    @mcp.tool()
    async def get_time_tracking(access_token: str, task_id: str) -> str:
        """
        Retrieve time tracking entries for a specific task.

        Use when the user asks about time spent, hours logged,
        time estimates, or time tracking data for a task.

        Args:
            access_token: API token for authenticating the CRM request.
            task_id: The task ID to fetch time entries for.

        Returns:
            A summary of time logged with totals.
        """
        url = os.getenv("N8N_WEBHOOK_CRM_GET_TIME")
        raw = await _call_n8n_crm(url, access_token, {"task_id": task_id})
        if raw.startswith("Error:"):
            return raw

        return generate_answer(
            "You are a project analyst. Summarize time tracking data: "
            "list each entry with person, hours, date, and description. "
            "Show total hours at the end. Format durations as Xh Ym. "
            "If empty, say 'No time entries found for this task.'",
            raw, f"Get time tracking for task {task_id}"
        )
