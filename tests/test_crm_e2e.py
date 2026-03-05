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
logger = logging.getLogger("test_suppa_e2e")


class TestSuppaE2E(unittest.IsolatedAsyncioTestCase):
    """
    E2E test for Suppa CRM integration.
    Simulates: query → supervisor → suppa_search_instances tool → callback.
    The suppa tools are mocked as StructuredTools to test the full graph flow.
    """

    @patch("app.utils.ChatVertexAI")
    async def test_suppa_full_process_with_callback(self, mock_chat_class):
        """
        Tests the full background process:
        1. Supervisor calls suppa_search_instances
        2. Tool returns mock data
        3. Supervisor generates answer
        4. Callback sent with result + metrics
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

        # Step 1: LLM calls suppa_search_instances
        resp_step1 = AIMessage(
            content="Searching for tasks...",
            tool_calls=[{
                "name": "suppa_search_instances",
                "args": {
                    "entity_id": "entity-uuid-tasks",
                    "fields": ["name", "status", "responsible"],
                    "limit": 10,
                },
                "id": "call_search_01"
            }]
        )
        # Step 2: LLM generates final answer from tool data
        resp_step2 = AIMessage(content="Found 3 tasks. Here is the summary:\n- Design review (done)\n- Implementation (in progress)\n- QA testing (todo)")
        # Step 3: Validator says COMPLETE
        resp_validator = AIMessage(content="COMPLETE")

        mock_ainvoke.side_effect = [resp_step1, resp_step2, resp_validator]

        # --- 2. SETUP MOCK SUPPA TOOL ---
        MOCK_SEARCH_RESULT = json.dumps({
            "entity_id": "entity-uuid-tasks",
            "count": 3,
            "instances": [
                {"id": "inst-1", "name": "Design review", "status": {"name": "done"}, "responsible": {"name": "dev@co.com"}},
                {"id": "inst-2", "name": "Implementation", "status": {"name": "in_progress"}, "responsible": {"name": "dev2@co.com"}},
                {"id": "inst-3", "name": "QA testing", "status": {"name": "todo"}, "responsible": {"name": "qa@co.com"}},
            ]
        })

        async def mock_suppa_search(entity_id: str, fields: list, limit: int = 20, offset: int = 0, **kwargs) -> str:
            """Mock suppa_search_instances that returns predefined data."""
            return MOCK_SEARCH_RESULT

        mock_tools = [
            StructuredTool.from_function(
                coroutine=mock_suppa_search,
                name="suppa_search_instances",
                description="Search records in a Suppa entity"
            ),
        ]

        # Mock the webhook client for the callback
        posted_callbacks = []

        async def mock_callback_post(url, **kwargs):
            posted_callbacks.append({"url": url, **kwargs})
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            return resp

        mock_httpx_client = MagicMock()
        mock_httpx_client.post = AsyncMock(side_effect=mock_callback_post)

        with patch.object(mcp_client, "get_tools", return_value=mock_tools), \
             patch("mcp_server.shared.webhook_helper._client", mock_httpx_client):

            agent_workflow = await build_agent_graph()
            graph = agent_workflow.compile()

            # --- 3. EXECUTE FULL BACKGROUND FLOW ---
            USER_QUERY = "Show me all tasks"
            ACCESS_TOKEN = "token_abc"
            CALLBACK_URL = "https://n8n.internal/webhook/my-callback"
            REQUEST_ID = "req-suppa-001"

            logger.info(f"Starting Suppa E2E for: {USER_QUERY}")

            await _process_and_callback(
                graph=graph,
                query=USER_QUERY,
                access_token=ACCESS_TOKEN,
                callback_url=CALLBACK_URL,
                request_id=REQUEST_ID
            )

        # --- 4. VERIFY OUTCOMES ---
        logger.info(f"Callbacks sent: {len(posted_callbacks)}")

        # Verify callback was sent
        self.assertTrue(len(posted_callbacks) > 0, "At least one callback should be sent")

        callback = posted_callbacks[-1]
        self.assertEqual(callback["url"], CALLBACK_URL)

        payload = callback.get("json", {})
        self.assertEqual(payload.get("status"), "completed")
        self.assertIn("suppa_search_instances", payload.get("tools_used", []))

        # Verify execution metrics
        self.assertGreater(payload.get("execution_time_ms", 0), 0)
        self.assertGreater(payload.get("tools_called_count", 0), 0)

        logger.info("Suppa E2E process: PASSED")

if __name__ == "__main__":
    unittest.main()
