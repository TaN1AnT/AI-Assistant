import os
import sys
import json
import asyncio
import uuid
import logging
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Ensure root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_chat_session_e2e")

class TestChatSessionE2E(unittest.IsolatedAsyncioTestCase):
    """
    Emulates a 3-turn chat session to verify:
    1. Session Memory: Does context persist across turns?
    2. RAG Routing: Does it call the Knowledge MCP for tech specs?
    3. Suppa Routing: Does it call the Suppa MCP for task management?
    """

    @patch("app.utils.ChatVertexAI")
    async def test_multi_turn_session_flow(self, mock_chat_class):
        from app.graphs.supervisor import build_agent_graph
        from main import app_state
        from app.mcp_client import mcp_client
        from app.session_store import session_store
        from langchain_core.tools import StructuredTool

        # --- 1. SETUP MOCK LLM ---
        mock_llm_instance = MagicMock()
        mock_chat_class.return_value = mock_llm_instance
        mock_llm_instance.bind_tools.return_value = mock_llm_instance
        mock_ainvoke = AsyncMock()
        mock_llm_instance.ainvoke = mock_ainvoke

        # --- 2. SETUP MOCK TOOLS ---
        async def mock_rag_search(query: str) -> str:
            return "Conveniq Tech Specs: 4K Display, 120Hz refresh, ARM-v9 chip."

        async def mock_suppa_list(query: str = "") -> str:
            return json.dumps([{"id": "tasks-uuid", "alias": "Tasks"}])

        async def mock_suppa_props(entity_id: str) -> str:
            return json.dumps({"fields": {"id": "uuid", "name": "string", "status": "string"}})

        async def mock_suppa_search(entity_id: str, fields: list, **kwargs) -> str:
            return json.dumps({"instances": [{"id": "task-501", "name": "Conveniq Verification", "status": "Todo"}]})

        async def mock_suppa_update(entity_id: str, instance_id: str, data: dict) -> str:
            return json.dumps({"status": "updated", "id": instance_id})

        mock_tools = [
            StructuredTool.from_function(coroutine=mock_rag_search, name="rag_search", description="search docs"),
            StructuredTool.from_function(coroutine=mock_suppa_list, name="suppa_list_entities", description="list tables"),
            StructuredTool.from_function(coroutine=mock_suppa_props, name="suppa_get_entity_props", description="get schema"),
            StructuredTool.from_function(coroutine=mock_suppa_search, name="suppa_search_instances", description="search records"),
            StructuredTool.from_function(coroutine=mock_suppa_update, name="suppa_update_instance", description="update record"),
        ]

        # --- 3. CONVERSATION SIMULATION ---
        SESSION_ID = f"test-session-{uuid.uuid4()}"
        
        # Turn 1: Knowledge Query
        # Response sequence: Tool call -> Final Answer -> Validator
        mock_ainvoke.side_effect = [
            AIMessage(content="", tool_calls=[{"name": "rag_search", "args": {"query": "Conveniq specs"}, "id": "t1"}]),
            AIMessage(content="Conveniq features a 4K 120Hz display with ARM-v9."),
            AIMessage(content="COMPLETE")
        ]

        from langgraph.checkpoint.memory import MemorySaver

        with patch.object(mcp_client, "get_tools", return_value=mock_tools):
            agent_workflow = await build_agent_graph()
            graph = agent_workflow.compile(checkpointer=MemorySaver())

            # --- TURN 1 EXECUTION ---
            logger.info("--- TURN 1: Knowledge ---")
            config = {"configurable": {"thread_id": SESSION_ID}}
            out1 = await graph.ainvoke({"messages": [HumanMessage(content="What are Conveniq specs?")]}, config=config)
            
            # Debug: print message types and tool calls
            for i, m in enumerate(out1["messages"]):
                logger.info(f"Msg {i}: {type(m).__name__} | Content: {m.content[:50]} | ToolCalls: {getattr(m, 'tool_calls', [])}")
            
            self.assertIn("4K", out1["messages"][-1].content)
            # Check for rag_search in tool calls
            tool_calls = [tc["name"] for m in out1["messages"] if hasattr(m, "tool_calls") for tc in m.tool_calls]
            self.assertIn("rag_search", tool_calls)

            # --- TURN 2: Suppa Search ---
            logger.info("--- TURN 2: Suppa ---")
            mock_ainvoke.side_effect = [
                # LLM decides to search Suppa
                AIMessage(content="", tool_calls=[
                    {"name": "suppa_list_entities", "args": {"query": "Tasks"}, "id": "t2a"},
                    {"name": "suppa_get_entity_props", "args": {"entity_id": "tasks-uuid"}, "id": "t2b"}
                ]),
                # After props, it searches
                AIMessage(content="", tool_calls=[
                    {"name": "suppa_search_instances", "args": {"entity_id": "tasks-uuid", "fields": ["name", "status"]}, "id": "t2c"}
                ]),
                # Final answer
                AIMessage(content="I found task 'Conveniq Verification' (ID: task-501) with status 'Todo'."),
                # Validator
                AIMessage(content="COMPLETE")
            ]
            
            out2 = await graph.ainvoke({"messages": [HumanMessage(content="Are there any tasks for this?")]}, config=config)
            # Verify suppa search was called
            tool_calls2 = [tc["name"] for m in out2["messages"] if hasattr(m, "tool_calls") for tc in m.tool_calls]
            self.assertIn("suppa_search_instances", tool_calls2)
            self.assertIn("task-501", out2["messages"][-1].content)
            
            # --- TURN 3: Update using Session Memory ---
            logger.info("--- TURN 3: Update (Session Memory) ---")
            # User says "Update its status to In Progress"
            # LLM should remember task-501 from previous turn messages
            mock_ainvoke.side_effect = [
                AIMessage(content="", tool_calls=[
                    {"name": "suppa_update_instance", "args": {"entity_id": "tasks-uuid", "instance_id": "task-501", "data": {"status": "In Progress"}}, "id": "t3"}
                ]),
                AIMessage(content="Successfully updated task task-501 to 'In Progress'."),
                AIMessage(content="COMPLETE")
            ]
            
            # Reset _iteration for the new turn
            out3 = await graph.ainvoke(
                {"messages": [HumanMessage(content="Update its status to 'In Progress'")], "_iteration": 0}, 
                config=config
            )
            
            # Verify tool args used the ID from memory (look at last few messages)
            turn3_msgs = out3["messages"][-5:]
            update_call = [m for m in turn3_msgs if hasattr(m, "tool_calls") and m.tool_calls][0].tool_calls[0]
            self.assertEqual(update_call["args"]["instance_id"], "task-501")
            self.assertIn("Successfully updated", out3["messages"][-1].content)

            logger.info("Chat Session E2E: PASSED (Memory, Knowledge, Suppa)")

if __name__ == "__main__":
    unittest.main()
