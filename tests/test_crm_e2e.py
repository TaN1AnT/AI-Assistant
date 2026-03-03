import os
import sys
import json
import asyncio
import logging
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Ensure the root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_crm_e2e")


def _make_mock_httpx_response(data, status_code=200):
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"content-type": "application/json"}
    resp.json.return_value = data
    resp.text = json.dumps(data)
    resp.raise_for_status = MagicMock()
    return resp


class TestCRME2E(unittest.IsolatedAsyncioTestCase):
    """
    E2E test for CRM integration with async httpx and ChatGoogleGenerativeAI.
    Simulates:
    1. User Request (via async webhook)
    2. Background Processing (_process_and_callback)
    3. Supervisor Node Tool Selection
    4. Tool Node Execution -> n8n Webhook (Mocked via httpx)
    5. Final Answer Generation
    6. POST result to Callback URL (Mocked via httpx)
    """

    @patch("app.utils.ChatGoogleGenerativeAI")
    async def test_crm_full_process_with_callback(self, mock_chat_class):
        """
        Tests the full background process including the final callback to n8n.
        Uses get_tasks (CRM tool) since get_deals has been removed.
        """
        from app.graphs.supervisor import build_agent_graph
        from main import _process_and_callback
        from app.mcp_client import mcp_client
        from langchain_core.tools import StructuredTool

        # --- 1. SETUP MOCK LLM ---
        mock_llm_instance = MagicMock()
        mock_chat_class.return_value = mock_llm_instance

        mock_ainvoke = AsyncMock()
        mock_llm_instance.ainvoke = mock_ainvoke
        mock_llm_instance.bind_tools.return_value = mock_llm_instance

        resp_step1 = AIMessage(
            content="Fetching tasks for the deal...",
            tool_calls=[{
                "name": "get_tasks",
                "args": {"access_token": "token_abc", "deal_id": "DEAL-42"},
                "id": "call_tasks_01"
            }]
        )
        resp_step2 = AIMessage(content="I have processed the CRM data. Here is your task report.")
        resp_validator = AIMessage(content="COMPLETE")

        mock_ainvoke.side_effect = [resp_step1, resp_step2, resp_validator]
        mock_llm_instance.invoke.return_value = AIMessage(content="Processed tool data.")

        # --- 2. SETUP MOCK HTTPX ---
        MOCK_TASKS = [
            {"id": "T-101", "title": "Review PR", "status": "in_progress", "assignee": "dev@co.com"},
            {"id": "T-102", "title": "Deploy staging", "status": "todo", "assignee": "ops@co.com"},
        ]

        posted_urls = []

        async def mock_httpx_post(url, **kwargs):
            posted_urls.append(url)
            if "crm/tasks" in url:
                return _make_mock_httpx_response(MOCK_TASKS)
            return _make_mock_httpx_response({"status": "ok"})

        # Mock the shared webhook helper's global client
        mock_webhook_client = MagicMock()
        mock_webhook_client.post = AsyncMock(side_effect=mock_httpx_post)

        # Mock httpx.AsyncClient for the callback in _process_and_callback
        mock_callback_client = MagicMock()
        mock_callback_client.post = AsyncMock(side_effect=mock_httpx_post)
        mock_callback_client.__aenter__ = AsyncMock(return_value=mock_callback_client)
        mock_callback_client.__aexit__ = AsyncMock(return_value=False)

        # --- 3. SETUP REAL CRM TOOLS ---
        from mcp_server.crm.tools import register_tools as register_crm_tools

        class MockMCP:
            def __init__(self):
                self.tools = {}
            def tool(self):
                def decorator(func):
                    self.tools[func.__name__] = func
                    return func
                return decorator

        mock_mcp = MockMCP()
        register_crm_tools(mock_mcp)

        real_get_tasks = mock_mcp.tools["get_tasks"]
        mock_tools = [
            StructuredTool.from_function(
                coroutine=real_get_tasks,
                name="get_tasks",
                description="Get tasks for a deal"
            ),
        ]

        os.environ["N8N_WEBHOOK_CRM_GET_TASKS"] = "https://n8n.modern-expo.com/webhook/crm/tasks"

        with patch.object(mcp_client, "get_tools", return_value=mock_tools), \
             patch("mcp_server.shared.webhook_helper._client", mock_webhook_client), \
             patch("httpx.AsyncClient", return_value=mock_callback_client):

            agent_workflow = await build_agent_graph()
            graph = agent_workflow.compile()

            # --- 4. EXECUTE FULL BACKGROUND FLOW ---
            USER_QUERY = "What are the tasks for deal DEAL-42?"
            ACCESS_TOKEN = "token_abc"
            CALLBACK_URL = "https://n8n.internal/webhook/my-callback"
            REQUEST_ID = "req-e2e-999"

            logger.info(f"Starting E2E Background processing for: {USER_QUERY}")

            await _process_and_callback(
                graph=graph,
                query=USER_QUERY,
                access_token=ACCESS_TOKEN,
                callback_url=CALLBACK_URL,
                request_id=REQUEST_ID
            )

        # --- 5. VERIFY OUTCOMES ---
        logger.info("Verifying final outcomes...")
        logger.info(f"POSTed URLs: {posted_urls}")

        # A. Verify n8n CRM Webhook was called
        self.assertTrue(any("webhook/crm/tasks" in u for u in posted_urls), "Tasks webhook should be called")

        # B. Verify Callback was sent
        self.assertTrue(any(CALLBACK_URL in u for u in posted_urls), "Final callback should be sent")

        # C. Verify callback payload
        callback_call = None
        for call in mock_callback_client.post.call_args_list:
            args, kwargs = call
            if CALLBACK_URL in args[0]:
                callback_call = kwargs
                break

        self.assertIsNotNone(callback_call)
        payload = callback_call.get("json", {})
        self.assertEqual(payload.get("status"), "completed")
        self.assertIn("get_tasks", payload.get("tools_used", []))

        logger.info("Full CRM E2E process with REAL tool logic: PASSED")

if __name__ == "__main__":
    unittest.main()
