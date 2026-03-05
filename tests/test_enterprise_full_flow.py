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
logger = logging.getLogger("test_enterprise_full_flow")

class TestEnterpriseFullFlow(unittest.IsolatedAsyncioTestCase):
    """
    Comprehensive verification for:
    1. Parallel RAG + Suppa Discovery
    2. Relational Search (nested field syntax)
    3. Multi-turn Memory Consistency
    4. Error Handling (partial failure in parallel calls)
    """

    @patch("app.utils.ChatVertexAI")
    async def test_full_enterprise_scenario(self, mock_chat_class):
        from app.graphs.supervisor import build_agent_graph
        from app.mcp_client import mcp_client
        from langchain_core.tools import StructuredTool
        from langgraph.checkpoint.memory import MemorySaver

        # --- MOCK TOOL DEFINITIONS ---
        async def mock_rag_search(query: str) -> str:
            if "cleaning" in query.lower():
                return "SOP-101: Warehouse cleaning must happen every 24h."
            return "General warehouse info."

        async def mock_suppa_list(query: str = "") -> str:
            return json.dumps([{"id": "tasks-uuid", "alias": "Tasks"}])

        async def mock_suppa_props(entity_id: str) -> str:
            return json.dumps({
                "fields": {
                    "name": "string",
                    "status": "relation",
                    "responsible": "relation",
                    "priority": "string"
                }
            })

        async def mock_suppa_search(entity_id: str, fields: dict, filter: dict = None, **kwargs) -> str:
            # Complex result with relational data
            return json.dumps({
                "instances": [{
                    "id": "task-abc",
                    "name": "Warehouse Floor Cleaning",
                    "priority": "High",
                    "status": {"id": "s1", "name": "Todo"},
                    "responsible": {"id": "u1", "name": "John Cleaner"}
                }]
            })

        async def mock_suppa_failing_tool(**kwargs) -> str:
            raise Exception("Suppa API Timeout Exception")

        mock_tools = [
            StructuredTool.from_function(coroutine=mock_rag_search, name="rag_search", description="rag"),
            StructuredTool.from_function(coroutine=mock_suppa_list, name="suppa_list_entities", description="list"),
            StructuredTool.from_function(coroutine=mock_suppa_props, name="suppa_get_entity_props", description="props"),
            StructuredTool.from_function(coroutine=mock_suppa_search, name="suppa_search_instances", description="search"),
            StructuredTool.from_function(coroutine=mock_suppa_failing_tool, name="suppa_get_mentions", description="fail"),
        ]

        # --- SETUP GRAPH ---
        SESSION_ID = f"ent-test-{uuid.uuid4()}"
        config = {"configurable": {"thread_id": SESSION_ID}}
        
        mock_llm_instance = MagicMock()
        mock_chat_class.return_value = mock_llm_instance
        mock_llm_instance.bind_tools.return_value = mock_llm_instance
        mock_ainvoke = AsyncMock()
        mock_llm_instance.ainvoke = mock_ainvoke

        with patch.object(mcp_client, "get_tools", return_value=mock_tools):
            agent_workflow = await build_agent_graph()
            graph = agent_workflow.compile(checkpointer=MemorySaver())

            # ════════════════════════════════════════════════════════════════
            # TURN 1: Parallel RAG + Suppa Discovery
            # ════════════════════════════════════════════════════════════════
            logger.info("TURN 1: Parallel RAG + Suppa Discovery")
            mock_ainvoke.side_effect = [
                # LLM decides to search both
                AIMessage(content="", tool_calls=[
                    {"name": "rag_search", "args": {"query": "warehouse cleaning schedule"}, "id": "t1a"},
                    {"name": "suppa_list_entities", "args": {"query": "Tasks"}, "id": "t1b"}
                ]),
                # Final Turn 1 result
                AIMessage(content="According to SOP-101, cleaning is every 24h. I've found the 'Tasks' table."),
                AIMessage(content="COMPLETE")
            ]
            out1 = await graph.ainvoke({"messages": [HumanMessage(content="Find warehouse cleaning schedule and the tasks table.")]}, config=config)
            
            tool_calls1 = [tc["name"] for m in out1["messages"] if hasattr(m, "tool_calls") for tc in m.tool_calls]
            self.assertIn("rag_search", tool_calls1)
            self.assertIn("suppa_list_entities", tool_calls1)

            # ════════════════════════════════════════════════════════════════
            # TURN 2: Schema Discovery + Relational Search (Nested fields)
            # ════════════════════════════════════════════════════════════════
            logger.info("TURN 2: Relational Search")
            mock_ainvoke.side_effect = [
                # Discover props first
                AIMessage(content="", tool_calls=[
                    {"name": "suppa_get_entity_props", "args": {"entity_id": "tasks-uuid"}, "id": "t2a"}
                ]),
                # Now perform search with NESTED fields (as taught in prompt)
                AIMessage(content="", tool_calls=[
                    {"name": "suppa_search_instances", "args": {
                        "entity_id": "tasks-uuid",
                        "fields": {"name": {}, "status": {"name": {}}, "responsible": {"name": {}}},
                        "filter": {"conditions": [{"field": "priority", "operator": "=", "value": "High"}]}
                    }, "id": "t2b"}
                ]),
                # Final Answer
                AIMessage(content="High priority task 'Warehouse Floor Cleaning' is assigned to John Cleaner (Status: Todo)."),
                AIMessage(content="COMPLETE")
            ]
            out2 = await graph.ainvoke({"messages": [HumanMessage(content="Who is responsible for the high priority cleaning task?")], "_iteration": 0}, config=config)
            
            self.assertIn("John Cleaner", out2["messages"][-1].content)
            
            # ════════════════════════════════════════════════════════════════
            # TURN 3: Error Recovery (Parallel Tool Failure)
            # ════════════════════════════════════════════════════════════════
            logger.info("TURN 3: Error Recovery")
            mock_ainvoke.side_effect = [
                # Call one good tool and one failing tool in parallel
                AIMessage(content="", tool_calls=[
                    {"name": "rag_search", "args": {"query": "Warehouse layout"}, "id": "t3_good"},
                    {"name": "suppa_get_mentions", "args": {}, "id": "t3_fail"}
                ]),
                # LLM should see the error from tool 2 but have data from tool 1
                AIMessage(content="I retrieved the warehouse layout, but the Suppa mention service is currently unavailable."),
                AIMessage(content="COMPLETE")
            ]
            out3 = await graph.ainvoke({"messages": [HumanMessage(content="Check layout and my mentions.")], "_iteration": 0}, config=config)
            
            # Verify the failing tool response exists
            tool_outputs = [m.content for m in out3["messages"] if isinstance(m, ToolMessage)]
            self.assertTrue(any("Suppa API Timeout" in str(content) for content in tool_outputs))
            self.assertIn("Suppa mention service is currently unavailable", out3["messages"][-1].content)

            logger.info("Enterprise Full Flow Test: PASSED")

if __name__ == "__main__":
    unittest.main()
