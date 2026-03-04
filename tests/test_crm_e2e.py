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
    E2E test for CRM integration.
    Simulates the full flow: query → supervisor → CRM tool → n8n → callback.
    Uses get_subtasks (task-level tool) since deals were removed.
    """

    @patch("app.utils.ChatGoogleGenerativeAI")
    async def test_crm_full_process_with_callback(self, mock_chat_class):
        """
        Tests the full background process including the final callback to n8n.
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
            content="Fetching subtasks...",
            tool_calls=[{
                "name": "get_subtasks",
                "args": {"access_token": "token_abc", "task_id": "T-501"},
                "id": "call_subtasks_01"
            }]
        )
        resp_step2 = AIMessage(content="Task T-501 has 3 subtasks. Here is the breakdown.")
        resp_validator = AIMessage(content="COMPLETE")

        mock_ainvoke.side_effect = [resp_step1, resp_step2, resp_validator]
        mock_llm_instance.invoke.return_value = AIMessage(content="Processed tool data.")

        # --- 2. SETUP MOCK HTTPX ---
        MOCK_SUBTASKS = [
            {"id": "ST-1", "title": "Design review", "status": "done", "assignee": "dev@co.com"},
            {"id": "ST-2", "title": "Implementation", "status": "in_progress", "assignee": "dev2@co.com"},
            {"id": "ST-3", "title": "QA testing", "status": "todo", "assignee": "qa@co.com"},
        ]

        posted_urls = []

        async def mock_httpx_post(url, **kwargs):
            posted_urls.append(url)
            if "crm" in url and "subtasks" in url:
                return _make_mock_httpx_response(MOCK_SUBTASKS)
            return _make_mock_httpx_response({"status": "ok"})

        mock_webhook_client = MagicMock()
        mock_webhook_client.post = AsyncMock(side_effect=mock_httpx_post)

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

        real_get_subtasks = mock_mcp.tools["get_subtasks"]
        mock_tools = [
            StructuredTool.from_function(
                coroutine=real_get_subtasks,
                name="get_subtasks",
                description="Get subtasks for a task"
            ),
        ]

        os.environ["N8N_WEBHOOK_CRM_GET_SUBTASKS"] = "https://n8n.modern-expo.com/webhook/crm/subtasks"

        # Both CRM tools and callback now use the same shared httpx client
        with patch.object(mcp_client, "get_tools", return_value=mock_tools), \
             patch("mcp_server.shared.webhook_helper._client", mock_webhook_client):

            agent_workflow = await build_agent_graph()
            graph = agent_workflow.compile()

            # --- 4. EXECUTE FULL BACKGROUND FLOW ---
            USER_QUERY = "What are the subtasks for task T-501?"
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

        self.assertTrue(any("crm" in u and "subtasks" in u for u in posted_urls), "Subtasks webhook should be called")
        self.assertTrue(any(CALLBACK_URL in u for u in posted_urls), "Final callback should be sent")

        callback_call = None
        for call in mock_webhook_client.post.call_args_list:
            args, kwargs = call
            if CALLBACK_URL in args[0]:
                callback_call = kwargs
                break

        self.assertIsNotNone(callback_call, "Callback should be sent via shared httpx client")
        payload = callback_call.get("json", {})
        self.assertEqual(payload.get("status"), "completed")
        self.assertIn("get_subtasks", payload.get("tools_used", []))
        # Verify new execution metrics are present
        self.assertGreater(payload.get("execution_time_ms", 0), 0)
        self.assertGreater(payload.get("tools_called_count", 0), 0)

        logger.info("Full CRM E2E process: PASSED")

if __name__ == "__main__":
    unittest.main()
