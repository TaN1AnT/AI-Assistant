import os
import sys
import json
import asyncio
import logging
import unittest
from unittest.mock import patch, MagicMock
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

class TestCRME2E(unittest.IsolatedAsyncioTestCase):
    """
    Detailed E2E test for CRM integration.
    Simulates: 
    1. User Request (via async webhook)
    2. Background Processing (_process_and_callback)
    3. Supervisor Node Tool Selection
    4. Tool Node Execution -> n8n Webhook (Mocked)
    5. Final Answer Generation
    6. POST result to Callback URL (Mocked)
    """

    @patch("app.utils.ChatVertexAI")  # Patch where it's used
    @patch("requests.post")
    async def test_crm_full_process_with_callback(self, mock_post, mock_chat_class):
        """
        Tests the full background process including the final callback to n8n.
        """
        from app.graphs.supervisor import build_agent_graph
        from main import _process_and_callback
        from app.mcp_client import mcp_client
        from langchain_core.tools import StructuredTool
        from unittest.mock import AsyncMock

        # --- 1. SETUP MOCK LLM ---
        mock_llm_instance = MagicMock()
        mock_chat_class.return_value = mock_llm_instance
        
        # Ensure ainvoke is an AsyncMock
        mock_ainvoke = AsyncMock()
        mock_llm_instance.ainvoke = mock_ainvoke
        
        # When bind_tools is called, return the same mock so we can control ainvoke
        mock_llm_instance.bind_tools.return_value = mock_llm_instance

        # Responses for the supervisor graph
        # Supervisor picks get_deals
        resp_step1 = AIMessage(
            content="Checking deals...",
            tool_calls=[{
                "name": "get_deals",
                "args": {"access_token": "token_abc", "status_filter": "open"},
                "id": "call_deals_02"
            }]
        )
        # Supervisor decides it's enough after one tool
        resp_step2 = AIMessage(content="I have processed the CRM data. Here is your CRM report.")
        
        # Validator says COMPLETE
        resp_validator = AIMessage(content="COMPLETE")
        
        # Sequence: [Step 1 (Supervisor - tool call), Step 2 (Supervisor - final answer), Step 3 (Validator)]
        mock_ainvoke.side_effect = [resp_step1, resp_step2, resp_validator]
        
        # Sync call for generate_answer (called inside the tool)
        mock_llm_instance.invoke.return_value = AIMessage(content="Processed tool data.")

        # --- 2. SETUP MOCK N8N ---
        MOCK_DEALS = [{"id": "D-99", "title": "Test Deal", "value": 1000}]

        def mocked_requests_post(url, **kwargs):
            response = MagicMock()
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            
            if "crm/deals" in url:
                response.json.return_value = MOCK_DEALS
                return response
            
            # This is the callback catch
            if "my-callback" in url:
                return response
                
            return response

        mock_post.side_effect = mocked_requests_post

        # --- 3. SETUP REAL CRM TOOLS (Captured via MockMCP) ---
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
        
        # Now mock_mcp.tools contains the REAL functions: get_deals, get_tasks, etc.
        # We wrap them as StructuredTool for the Agent
        
        real_get_deals = mock_mcp.tools["get_deals"]
        real_get_tasks = mock_mcp.tools["get_tasks"]

        mock_tools = [
            StructuredTool.from_function(
                func=real_get_deals,
                name="get_deals",
                description="Get deals"
            ),
            StructuredTool.from_function(
                func=real_get_tasks,
                name="get_tasks",
                description="Get tasks"
            )
        ]
        
        with patch.object(mcp_client, "get_tools", return_value=mock_tools):
            agent_workflow = await build_agent_graph()
            graph = agent_workflow.compile()

        # --- 4. EXECUTE FULL BACKGROUND FLOW ---
        USER_QUERY = "Give me a report on open deals."
        ACCESS_TOKEN = "token_abc"
        CALLBACK_URL = "https://n8n.internal/webhook/my-callback"
        REQUEST_ID = "req-e2e-999"

        # Set environment variables for the tools to find the URLs
        os.environ["N8N_WEBHOOK_CRM_GET_DEALS"] = "https://n8n.modern-expo.com/webhook/crm/deals"
        os.environ["N8N_WEBHOOK_CRM_GET_TASKS"] = "https://n8n.modern-expo.com/webhook/crm/tasks"

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
        
        # A. Verify n8n Webhook was reached (Real tool code now calls requests.post)
        post_urls = [args[0] for args, kwargs in mock_post.call_args_list]
        logger.info(f"POSTed URLs: {post_urls}")
        
        self.assertTrue(any("webhook/crm/deals" in u for u in post_urls), "Deals webhook should be called")
        
        # B. Verify Callback was sent back
        self.assertTrue(any(CALLBACK_URL in u for u in post_urls), "Final callback should be sent")
        
        # C. Verify the content of the callback
        callback_payload = None
        for args, kwargs in mock_post.call_args_list:
            if CALLBACK_URL in args[0]:
                callback_payload = kwargs.get("json", {})
                
        self.assertIsNotNone(callback_payload)
        self.assertEqual(callback_payload.get("status"), "completed")
        self.assertIn("Here is your CRM report.", callback_payload.get("answer", ""))
        self.assertIn("get_deals", callback_payload.get("tools_used", []))

        logger.info("Full CRM E2E process with REAL tool logic: PASSED")

if __name__ == "__main__":
    unittest.main()
