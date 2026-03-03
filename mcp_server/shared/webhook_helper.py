"""
Shared Webhook Helper for MCP Servers.

Consolidates the logic for calling n8n webhooks from CRM and Automation
MCP servers. Ensures consistent error handling, timeouts, retries, and logging.
"""
import os
import json
import asyncio
import logging
import httpx

logger = logging.getLogger("mcp_server.shared.webhook_helper")

# Global async client for connection pooling
_client = httpx.AsyncClient(timeout=30.0, limits=httpx.Limits(max_connections=100, max_keepalive_connections=20))

# Retry configuration
MAX_RETRIES = 1
RETRY_DELAY = 0.5  # seconds


async def call_n8n_webhook(webhook_url: str, token: str, payload_data: dict, timeout: int = 30) -> str:
    """
    Sends a secure POST request to an n8n webhook asynchronously.

    Features:
      - Automatic retry (1 retry) on transient failures (timeout, connection).
      - Connection pooling via a shared httpx.AsyncClient.
      - X-Webhook-Secret header for n8n authentication.

    Args:
        webhook_url:  The target n8n webhook URL.
        token:        The user's CRM/App access token.
        payload_data: The specific data for the action.
        timeout:      Request timeout in seconds.

    Returns:
        A JSON string from n8n or a clear error message starting with 'Error:'.
    """
    if not webhook_url:
        return "Error: Target webhook URL is not configured in .env."

    secret = os.getenv("N8N_WEBHOOK_SECRET", "")

    payload = {
        "token": token,
        **payload_data
    }

    headers = {
        "Content-Type": "application/json",
        "X-Webhook-Secret": secret,
    }

    last_error = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            resp = await _client.post(webhook_url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()

            if resp.headers.get("content-type", "").startswith("application/json"):
                return json.dumps(resp.json(), indent=2)
            else:
                return resp.text[:10000]

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                logger.warning(f"Webhook attempt {attempt+1} failed ({type(e).__name__}), retrying in {RETRY_DELAY}s...")
                await asyncio.sleep(RETRY_DELAY)
                continue
            # Final attempt failed
            if isinstance(e, httpx.TimeoutException):
                return f"Error: Webhook timed out after {timeout}s (retried {MAX_RETRIES}x)."
            return f"Error: Could not connect to n8n at {webhook_url} (retried {MAX_RETRIES}x)."

        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            body = e.response.text[:500]
            # Retry on 502/503/504 (n8n behind proxy restarting)
            if code in (502, 503, 504) and attempt < MAX_RETRIES:
                logger.warning(f"Webhook returned HTTP {code}, retrying...")
                await asyncio.sleep(RETRY_DELAY)
                continue
            return f"Error: n8n returned HTTP {code}. {body}"

        except Exception as e:
            logger.error(f"Webhook helper error: {e}", exc_info=True)
            return f"Error: {type(e).__name__}: {str(e)}"

    return f"Error: All {1 + MAX_RETRIES} attempts failed: {last_error}"


