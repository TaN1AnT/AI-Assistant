"""
Session Memory Tests — Tests conversation continuity, multi-session isolation,
and session security boundaries.

Test 1: Session Continuity
    Two sequential messages with the SAME session_id.
    Verifies the second message sees context from the first.

Test 2: Multi-Session Isolation
    Two concurrent sessions with DIFFERENT session_ids.
    Verifies each session stores its own context independently.

Test 3: Session Security
    Verifies that one session cannot read another session's data.
    Verifies sessions auto-expire after TTL.
"""
import os
import sys
import json
import time
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_session")


# ── Helper: mock httpx response ────────────────────────────────────────────

def _make_mock_httpx_response(data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"content-type": "application/json"}
    resp.json.return_value = data
    resp.text = json.dumps(data)
    resp.raise_for_status = MagicMock()
    return resp


def _setup_mock_llm(mock_chat_class, responses):
    """Configure the mock LLM with a sequence of responses."""
    mock_llm = MagicMock()
    mock_chat_class.return_value = mock_llm
    mock_ainvoke = AsyncMock()
    mock_llm.ainvoke = mock_ainvoke
    mock_llm.bind_tools.return_value = mock_llm
    mock_ainvoke.side_effect = responses
    mock_llm.invoke.return_value = AIMessage(content="Processed tool data.")
    return mock_llm


def _make_httpx_mocks(task_data=None):
    """Create mock httpx clients for webhook helper and callback."""
    async def mock_post(url, **kwargs):
        if "crm/tasks" in url:
            return _make_mock_httpx_response(task_data or [])
        return _make_mock_httpx_response({"status": "ok"})

    # Mock for the shared webhook helper global client
    mock_webhook_client = MagicMock()
    mock_webhook_client.post = AsyncMock(side_effect=mock_post)

    # Mock for httpx.AsyncClient used in _process_and_callback
    mock_callback_client = MagicMock()
    mock_callback_client.post = AsyncMock(side_effect=mock_post)
    mock_callback_client.__aenter__ = AsyncMock(return_value=mock_callback_client)
    mock_callback_client.__aexit__ = AsyncMock(return_value=False)

    return mock_webhook_client, mock_callback_client


async def _build_test_graph(mcp_client):
    """Build a test graph with a mock Suppa tool."""
    from app.graphs.supervisor import build_agent_graph
    from langchain_core.tools import StructuredTool

    async def mock_suppa_search(entity_id: str, fields: list, limit: int = 20, **kwargs) -> str:
        """Mock suppa_search_instances for testing."""
        import json
        return json.dumps({"entity_id": entity_id, "count": 0, "instances": []})

    mock_tools = [
        StructuredTool.from_function(
            coroutine=mock_suppa_search,
            name="suppa_search_instances",
            description="Search records in a Suppa entity"
        ),
    ]

    with patch.object(mcp_client, "get_tools", return_value=mock_tools):
        agent_workflow = await build_agent_graph()
        return agent_workflow.compile()


# ══════════════════════════════════════════════════════════════════════════
# Test 1: Session Continuity — same session, two messages
# ══════════════════════════════════════════════════════════════════════════

class TestSessionContinuity(unittest.IsolatedAsyncioTestCase):

    @patch("app.utils.ChatVertexAI")
    async def test_follow_up_retains_context(self, mock_chat_class):
        from main import _process_and_callback
        from app.mcp_client import mcp_client
        from app.session_store import session_store

        SESSION_ID = "session-continuity-test"
        CALLBACK = "https://n8n.internal/webhook/callback"
        os.environ["N8N_WEBHOOK_CRM_GET_TASKS"] = "https://n8n.modern-expo.com/webhook/crm/tasks"

        mock_webhook_client, mock_callback_client = _make_httpx_mocks([
            {"id": "T-101", "title": "Review PR", "status": "in_progress"},
            {"id": "T-102", "title": "Deploy staging", "status": "todo"},
        ])

        mock_llm = _setup_mock_llm(mock_chat_class, [
            AIMessage(content="Fetching tasks...", tool_calls=[{
                "name": "get_tasks",
                "args": {"access_token": "tok", "deal_id": "DEAL-42"},
                "id": "call_1"
            }]),
            AIMessage(content="Found 2 tasks: T-101 (Review PR) and T-102 (Deploy staging)."),
            AIMessage(content="COMPLETE"),
        ])

        with patch("mcp_server.shared.webhook_helper._client", mock_webhook_client), \
             patch("httpx.AsyncClient", return_value=mock_callback_client):

            graph = await _build_test_graph(mcp_client)

            # Message 1
            await _process_and_callback(graph, "What are the tasks for DEAL-42?", "tok", CALLBACK, "req-1", SESSION_ID)

            history = session_store.load_history(SESSION_ID)
            self.assertTrue(len(history) > 0, "Session should have stored the first turn")
            logger.info("✅ Message 1 completed. Session has %d history messages.", len(history))

            # Message 2
            mock_llm.ainvoke = AsyncMock(side_effect=[
                AIMessage(content="Getting details for T-101...", tool_calls=[{
                    "name": "get_tasks",
                    "args": {"access_token": "tok", "deal_id": "DEAL-42"},
                    "id": "call_2"
                }]),
                AIMessage(content="Task T-101 'Review PR' is in progress."),
                AIMessage(content="COMPLETE"),
            ])

            await _process_and_callback(graph, "Tell me more about T-101", "tok", CALLBACK, "req-2", SESSION_ID)

            first_call_messages = mock_llm.ainvoke.call_args_list[0][0][0]
            message_contents = [m.content for m in first_call_messages if isinstance(m, (HumanMessage, AIMessage))]
            has_prior_context = any("DEAL-42" in c for c in message_contents if c)
            self.assertTrue(has_prior_context, "Second message should see context from first message (DEAL-42)")

            history_after = session_store.load_history(SESSION_ID)
            human_msgs = [m for m in history_after if isinstance(m, HumanMessage)]
            self.assertEqual(len(human_msgs), 2, "Session should have 2 human messages (2 turns)")

            logger.info("✅ Message 2 completed. Continuity verified!")

        session_store.clear_session(SESSION_ID)


# ══════════════════════════════════════════════════════════════════════════
# Test 2: Multi-Session Isolation — two sessions, independent context
# ══════════════════════════════════════════════════════════════════════════

class TestMultiSessionIsolation(unittest.IsolatedAsyncioTestCase):

    @patch("app.utils.ChatVertexAI")
    async def test_two_sessions_are_independent(self, mock_chat_class):
        from main import _process_and_callback
        from app.mcp_client import mcp_client
        from app.session_store import session_store

        SESSION_A = "session-alpha"
        SESSION_B = "session-beta"
        CALLBACK = "https://n8n.internal/webhook/callback"
        os.environ["N8N_WEBHOOK_CRM_GET_TASKS"] = "https://n8n.modern-expo.com/webhook/crm/tasks"

        mock_webhook_client, mock_callback_client = _make_httpx_mocks([{"id": "T-A1", "title": "Alpha Task"}])

        mock_llm = _setup_mock_llm(mock_chat_class, [
            AIMessage(content="Fetching...", tool_calls=[{
                "name": "get_tasks",
                "args": {"access_token": "tok-a", "deal_id": "DEAL-100"},
                "id": "call_a"
            }]),
            AIMessage(content="Alpha session: found task T-A1 for DEAL-100."),
            AIMessage(content="COMPLETE"),
        ])

        with patch("mcp_server.shared.webhook_helper._client", mock_webhook_client), \
             patch("httpx.AsyncClient", return_value=mock_callback_client):

            graph = await _build_test_graph(mcp_client)

            # Session A
            await _process_and_callback(graph, "Tasks for DEAL-100?", "tok-a", CALLBACK, "req-a", SESSION_A)

            # Reset for Session B
            mock_webhook_client2, mock_callback_client2 = _make_httpx_mocks([{"id": "T-B1", "title": "Beta Task"}])
            mock_llm.ainvoke = AsyncMock(side_effect=[
                AIMessage(content="Fetching...", tool_calls=[{
                    "name": "get_tasks",
                    "args": {"access_token": "tok-b", "deal_id": "DEAL-200"},
                    "id": "call_b"
                }]),
                AIMessage(content="Beta session: found task T-B1 for DEAL-200."),
                AIMessage(content="COMPLETE"),
            ])

        with patch("mcp_server.shared.webhook_helper._client", mock_webhook_client2), \
             patch("httpx.AsyncClient", return_value=mock_callback_client2):

            # Session B
            await _process_and_callback(graph, "Tasks for DEAL-200?", "tok-b", CALLBACK, "req-b", SESSION_B)

        # Verify Isolation
        history_a = session_store.load_history(SESSION_A)
        history_b = session_store.load_history(SESSION_B)

        a_content = " ".join(m.content for m in history_a)
        b_content = " ".join(m.content for m in history_b)

        self.assertIn("DEAL-100", a_content, "Session A should contain DEAL-100 context")
        self.assertNotIn("DEAL-200", a_content, "Session A should NOT contain DEAL-200 context")

        self.assertIn("DEAL-200", b_content, "Session B should contain DEAL-200 context")
        self.assertNotIn("DEAL-100", b_content, "Session B should NOT contain DEAL-100 context")

        logger.info("✅ Multi-session isolation verified")

        session_store.clear_session(SESSION_A)
        session_store.clear_session(SESSION_B)


# ══════════════════════════════════════════════════════════════════════════
# Test 3: Session Security — cross-session access blocked, TTL expiry
# ══════════════════════════════════════════════════════════════════════════

class TestSessionSecurity(unittest.TestCase):

    def test_cross_session_access_blocked(self):
        from app.session_store import SessionStore
        store = SessionStore(ttl=60)
        store.save_turn("private-session", "What is my salary?", "Your salary is $120k.", ["rag_search"])
        store.save_turn("other-session", "Company policy?", "Here is the policy.", ["rag_search"])

        private_text = " ".join(m.content for m in store.load_history("private-session"))
        other_text = " ".join(m.content for m in store.load_history("other-session"))

        self.assertIn("salary", private_text.lower())
        self.assertNotIn("salary", other_text.lower())
        self.assertIn("policy", other_text.lower())
        self.assertNotIn("policy", private_text.lower())
        self.assertEqual(len(store.load_history("nonexistent-session")), 0)
        logger.info("✅ Cross-session access is fully blocked")

    def test_session_ttl_expiry(self):
        from app.session_store import SessionStore
        store = SessionStore(ttl=1)
        store.save_turn("expiring-session", "Hello", "Hi there!", [])
        self.assertTrue(len(store.load_history("expiring-session")) > 0)
        time.sleep(1.5)
        self.assertEqual(len(store.load_history("expiring-session")), 0)
        self.assertEqual(store.active_sessions(), 0)
        logger.info("✅ Session TTL expiry verified (1s TTL)")

    def test_clear_session_removes_data(self):
        from app.session_store import SessionStore
        store = SessionStore(ttl=60)
        store.save_turn("clearable", "Secret question", "Secret answer", ["rag_search"])
        self.assertTrue(len(store.load_history("clearable")) > 0)
        store.clear_session("clearable")
        self.assertEqual(len(store.load_history("clearable")), 0)
        logger.info("✅ Manual session clear verified")

    def test_empty_session_id_returns_nothing(self):
        from app.session_store import SessionStore
        store = SessionStore(ttl=60)
        store.save_turn("real-session", "Hello", "Hi", [])
        self.assertEqual(len(store.load_history("")), 0)
        self.assertEqual(len(store.load_history(None)), 0)
        logger.info("✅ Empty/None session_id safety verified")

    def test_max_turns_trimming(self):
        from app.session_store import SessionStore
        store = SessionStore(ttl=60)
        for i in range(15):
            store.save_turn("trimming-test", f"Question {i}", f"Answer {i}", [])
        history = store.load_history("trimming-test")
        human_msgs = [m for m in history if isinstance(m, HumanMessage)]
        self.assertLessEqual(len(human_msgs), 10)
        all_text = " ".join(m.content for m in history)
        self.assertNotIn("Question 0", all_text)
        self.assertIn("Question 14", all_text)
        logger.info("✅ Turn trimming verified (15 saved, 10 kept)")


if __name__ == "__main__":
    unittest.main()
