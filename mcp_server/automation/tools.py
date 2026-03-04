"""
Automation Tools — Webhook-proxy tools for triggering actions.

Each tool:
1. Accepts access_token + action parameters from the LangGraph agent.
2. POSTs a JSON payload to a dedicated n8n webhook URL (from .env).
3. n8n executes the action (create ticket, send notification, etc.).
4. Returns the raw JSON response from n8n.
"""
import os
import logging
from mcp_server.shared.webhook_helper import call_n8n_webhook

logger = logging.getLogger("mcp_server.automation.tools")


async def _call_n8n_automation(webhook_url: str, token: str, data: dict) -> str:
    """Proxy to the shared async webhook helper for Automation-specific calls."""
    return await call_n8n_webhook(webhook_url, token, {"data": data})



# ── Tool registration ──────────────────────────────────────────────────────

def register_tools(mcp):
    """Register automation tools with the MCP server."""

    @mcp.tool()
    async def create_task(access_token: str, title: str, description: str = "") -> str:
        """
        Create a new task in the project management system. Returns raw JSON.

        Use when the user wants to create a ticket, task, or action item.

        Args:
            access_token: API token for authenticating the request.
            title: Short title for the task.
            description: Detailed description of what needs to be done.

        Returns:
            Raw JSON confirmation with the new task details.
        """
        url = os.getenv("N8N_WEBHOOK_AUTOMATION_CREATE_TASK")
        return await _call_n8n_automation(url, access_token, {
            "title": title,
            "description": description,
        })

    @mcp.tool()
    async def send_notification(access_token: str, recipient: str, message: str, channel: str = "email") -> str:
        """
        Send a notification to a person or channel. Returns raw JSON.

        Use when the user wants to email someone, post to Slack, or send an alert.

        Args:
            access_token: API token for authenticating the request.
            recipient: Who to notify (email address, Slack channel, etc.).
            message: The notification content.
            channel: Delivery channel: 'email', 'slack', or 'sms'. Default: 'email'.

        Returns:
            Raw JSON confirmation that the notification was sent.
        """
        url = os.getenv("N8N_WEBHOOK_AUTOMATION_SEND_NOTIFICATION")
        return await _call_n8n_automation(url, access_token, {
            "recipient": recipient,
            "message": message,
            "channel": channel,
        })
