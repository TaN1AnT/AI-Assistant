"""
Automation Tools — Webhook-proxy tools for triggering actions.

Each tool:
1. Accepts access_token + action parameters from the LangGraph agent.
2. POSTs a JSON payload to a dedicated n8n webhook URL (from .env).
3. n8n executes the action (create ticket, send notification, etc.).
4. Receives the JSON response from n8n.
5. Uses Gemini (via llm_helper) to format a confirmation message.
"""
import os
import logging
from mcp_server.shared.llm_helper import generate_answer
from mcp_server.shared.webhook_helper import call_n8n_webhook

logger = logging.getLogger("mcp_server.automation.tools")


async def _call_n8n_automation(webhook_url: str, token: str, data: dict) -> str:
    """Proxy to the shared async webhook helper for Automation-specific calls."""
    return await call_n8n_webhook(webhook_url, token, {"data": data})



# ── Tool registration ──────────────────────────────────────────────────────

def register_tools(mcp):
    """Register automation tools with the MCP server."""

    @mcp.tool()
    async def create_task(access_token: str, deal_id: str, title: str, description: str = "") -> str:
        """
        Create a new task in the CRM/project management system.

        Use when the user wants to create a ticket, task, or action item.

        Args:
            access_token: API token for authenticating the request.
            deal_id: The deal or project ID this task belongs to.
            title: Short title for the task.
            description: Detailed description of what needs to be done.

        Returns:
            Confirmation with the new task details.
        """
        url = os.getenv("N8N_WEBHOOK_AUTOMATION_CREATE_TASK")
        raw = await _call_n8n_automation(url, access_token, {
            "deal_id": deal_id,
            "title": title,
            "description": description,
        })
        if raw.startswith("Error:"):
            return raw

        return generate_answer(
            "Confirm the task was created. Show the Task ID and Title. Be brief.",
            raw, f"Create task '{title}' for deal {deal_id}"
        )

    @mcp.tool()
    async def send_notification(access_token: str, recipient: str, message: str, channel: str = "email") -> str:
        """
        Send a notification to a person or channel.

        Use when the user wants to email someone, post to Slack, or send an alert.

        Args:
            access_token: API token for authenticating the request.
            recipient: Who to notify (email address, Slack channel, etc.).
            message: The notification content.
            channel: Delivery channel: 'email', 'slack', or 'sms'. Default: 'email'.

        Returns:
            Confirmation that the notification was sent.
        """
        url = os.getenv("N8N_WEBHOOK_AUTOMATION_SEND_NOTIFICATION")
        raw = await _call_n8n_automation(url, access_token, {
            "recipient": recipient,
            "message": message,
            "channel": channel,
        })
        if raw.startswith("Error:"):
            return raw

        return generate_answer(
            "Confirm the notification was sent. State to whom and via what channel. Be brief.",
            raw, f"Send {channel} to {recipient}"
        )
