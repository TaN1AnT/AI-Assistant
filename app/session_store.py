"""
Session Store — In-memory conversation history with TTL.

Stores condensed conversation turns per session_id so that
follow-up messages within the same session retain context.

Sessions auto-expire after SESSION_TTL seconds of inactivity.
Message history is capped at MAX_TURNS to control token usage.
Tool results are truncated before storage to keep costs low.

Usage:
    from app.session_store import session_store

    history = session_store.load_history(session_id)   # → List[BaseMessage]
    session_store.save_turn(session_id, query, answer, tools_used)
"""
import logging
from typing import List, Dict, Any
from cachetools import TTLCache
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

logger = logging.getLogger("app.session_store")

# ── Configuration ──────────────────────────────────────────────────────────

SESSION_TTL = 300       # 5 minutes
MAX_SESSIONS = 200      # Max concurrent sessions in cache
MAX_TURNS = 10          # Keep last 10 turns (10 human + 10 AI = 20 messages)
TOOL_RESULT_MAX = 200   # Truncate tool result summaries to this many chars


class SessionStore:
    """
    Lightweight conversation memory using TTLCache.

    Each session stores a list of condensed (human, AI) turn pairs.
    Sessions auto-expire after SESSION_TTL seconds of inactivity.
    Any access (load or save) refreshes the TTL.
    """

    def __init__(self, ttl: int = SESSION_TTL, maxsize: int = MAX_SESSIONS):
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)

    # ── Public API ─────────────────────────────────────────────────────

    def load_history(self, session_id: str) -> List[BaseMessage]:
        """
        Returns the stored conversation messages for a session.
        If the session doesn't exist or has expired, returns [].

        The returned list is ready to prepend to a new LangGraph state.
        """
        if not session_id:
            return []

        turns = self._cache.get(session_id)
        if not turns:
            return []

        messages: List[BaseMessage] = []
        for turn in turns:
            messages.append(HumanMessage(content=turn["human"]))
            messages.append(AIMessage(content=turn["ai"]))
            # Add a compact summary of tools used (if any)
            if turn.get("tools_summary"):
                messages.append(SystemMessage(
                    content=f"[Previous tools used: {turn['tools_summary']}]"
                ))

        logger.info(f"Session '{session_id}': loaded {len(turns)} previous turns")
        return messages

    def save_turn(
        self,
        session_id: str,
        query: str,
        answer: str,
        tools_used: List[str] = None,
    ):
        """
        Saves a new conversation turn to the session.

        Args:
            session_id: The session identifier.
            query:      The user's message.
            answer:     The AI's final response.
            tools_used: List of tool names called during this turn.
        """
        if not session_id:
            return

        # Load existing turns (or start fresh)
        turns = list(self._cache.get(session_id, []))

        # Build the new turn
        turn: Dict[str, Any] = {
            "human": query[:500],           # Cap human message
            "ai": answer[:1000],            # Cap AI answer
            "tools_summary": "",
        }

        if tools_used:
            turn["tools_summary"] = ", ".join(tools_used)[:TOOL_RESULT_MAX]

        turns.append(turn)

        # Trim to MAX_TURNS (keep most recent)
        if len(turns) > MAX_TURNS:
            turns = turns[-MAX_TURNS:]

        # Write back (this refreshes TTL)
        self._cache[session_id] = turns

        logger.info(
            f"Session '{session_id}': saved turn {len(turns)}/{MAX_TURNS} "
            f"(tools: {tools_used or 'none'})"
        )

    def active_sessions(self) -> int:
        """Returns the number of active (non-expired) sessions."""
        return len(self._cache)

    def clear_session(self, session_id: str):
        """Manually expire a session."""
        self._cache.pop(session_id, None)


# ── Singleton ──────────────────────────────────────────────────────────────

session_store = SessionStore()
