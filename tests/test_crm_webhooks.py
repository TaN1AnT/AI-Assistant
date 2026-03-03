import os
import json
import logging
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("test_crm_webhooks")

class MockMCP:
    """A minimal mock of the FastMCP object to capture tool registrations."""
    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func
        return decorator

def test_crm_integration():
    from mcp_server.crm.tools import register_tools
    
    mock_mcp = MockMCP()
    register_tools(mock_mcp)
    
    tools = mock_mcp.tools
    logger.info(f"Discovered {len(tools)} CRM tools: {list(tools.keys())}")

    # Configuration for testing
    ACCESS_TOKEN = "test_token_123"
    TEST_DEAL_ID = "D-100"
    TEST_TASK_ID = "T-501"

    print("\n" + "="*60)
    print("  CRM Webhook Integration Test")
    print("="*60)

    # 1. Test get_deals
    print(f"\n[1] Testing get_deals...")
    if "get_deals" in tools:
        try:
            result = tools["get_deals"](ACCESS_TOKEN, status_filter="open")
            print(f"Result:\n{result}")
        except Exception as e:
            print(f"❌ Error calling get_deals: {e}")
    else:
        print("❌ get_deals tool not found!")

    # 2. Test get_tasks
    print(f"\n[2] Testing get_tasks for deal {TEST_DEAL_ID}...")
    if "get_tasks" in tools:
        try:
            result = tools["get_tasks"](ACCESS_TOKEN, TEST_DEAL_ID)
            print(f"Result:\n{result}")
        except Exception as e:
            print(f"❌ Error calling get_tasks: {e}")
    else:
        print("❌ get_tasks tool not found!")

    # 3. Test get_task_comments
    print(f"\n[3] Testing get_task_comments for task {TEST_TASK_ID}...")
    if "get_task_comments" in tools:
        try:
            result = tools["get_task_comments"](ACCESS_TOKEN, TEST_TASK_ID)
            print(f"Result:\n{result}")
        except Exception as e:
            print(f"❌ Error calling get_task_comments: {e}")
    else:
        print("❌ get_task_comments tool not found!")

    print("\n" + "="*60)
    print("  Test Complete")
    print("="*60)
    print("Note: If you see 'Cannot reach n8n' errors, make sure n8n is running")
    print("and the URLs in .env are correct and accessible.")

if __name__ == "__main__":
    test_crm_integration()
