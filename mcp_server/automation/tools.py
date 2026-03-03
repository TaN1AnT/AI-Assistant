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
import json
import logging
import requests
from mcp_server.shared.llm_helper import generate_answer

logger = logging.getLogger("mcp_server.automation.tools")

# ── Shared webhook caller ──────────────────────────────────────────────────

def _call_n8n_automation(webhook_url: str, token: str, data: dict) -> str:
    """
    Sends a request to a specific Automation n8n webhook.

    Args:
        webhook_url: The specific URL for the automation action.
        token:       The user's access token, forwarded to n8n for API auth.
        data:        Action-specific data payload.

    Returns:
        Raw JSON string from n8n, or an error message.
    """
    if not webhook_url:
        return "Error: Webhook URL is not configured in .env."

    secret = os.getenv("N8N_WEBHOOK_SECRET", "")
    payload = {"token": token, "data": data}
    headers = {"Content-Type": "application/json", "X-Webhook-Secret": secret}

    try:
        resp = requests.post(webhook_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2) if resp.headers.get("content-type", "").startswith("application/json") else resp.text[:3000]
    except requests.Timeout:
        return "Error: Automation webhook timed out (30s)."
    except requests.ConnectionError:
        return f"Error: Cannot reach n8n at {webhook_url}."
    except requests.HTTPError as e:
        code = e.response.status_code if e.response else "?"
        body = e.response.text[:300] if e.response else ""
        return f"Error: Webhook returned HTTP {code}. {body}"
    except Exception as e:
        logger.error(f"Automation webhook error: {e}", exc_info=True)
        return f"Error: {type(e).__name__}: {e}"


# ── Tool registration ──────────────────────────────────────────────────────

def register_tools(mcp):
    """Register automation tools with the MCP server."""

    @mcp.tool()
    def create_task(access_token: str, deal_id: str, title: str, description: str = "") -> str:
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
        raw = _call_n8n_automation(url, access_token, {
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
    def send_notification(access_token: str, recipient: str, message: str, channel: str = "email") -> str:
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
        raw = _call_n8n_automation(url, access_token, {
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
