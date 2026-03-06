"""
LLM Helper for MCP Server Tools

Provides a shared Gemini LLM instance and helper functions
so that each MCP tool can generate polished, human-friendly answers.

Uses ChatVertexAI with service account authentication
via GOOGLE_APPLICATION_CREDENTIALS.
"""
import os
import logging
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from google.oauth2 import service_account

logger = logging.getLogger("mcp_server.shared.llm_helper")

_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
_PROJECT = os.getenv("GOOGLE_PROJECT_ID", "")
_LOCATION = os.getenv("GOOGLE_LOCATION", "europe-west4")

# Load credentials from service account file
_credentials = None
_sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
if _sa_path and os.path.exists(_sa_path):
    _credentials = service_account.Credentials.from_service_account_file(
        _sa_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    logger.info(f"LLM Helper: loaded credentials from {_sa_path}")

# Allow ~180 requests per minute (3 per second)
_rate_limiter = InMemoryRateLimiter(requests_per_second=3, check_every_n_seconds=0.1, max_bucket_size=10)


def get_llm(temperature: float = 0.2) -> ChatVertexAI:
    """Returns a configured Gemini LLM for use inside MCP tools via ChatVertexAI."""
    return ChatVertexAI(
        model=_MODEL,
        temperature=temperature,
        max_retries=3,
        rate_limiter=_rate_limiter,
        project=_PROJECT,
        location=_LOCATION,
        credentials=_credentials,
    )




_llm = None


def _get_shared_llm() -> ChatVertexAI:
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def generate_answer(system_prompt: str, raw_data: str, user_question: str = "") -> str:
    """
    Takes raw data and generates a human-friendly answer using Gemini.
    """
    llm = _get_shared_llm()

    context = f"DATA:\n{raw_data}"
    if user_question:
        context = f"USER QUESTION: {user_question}\n\n{context}"

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context),
        ])
        return response.content
    except Exception as e:
        logger.error(f"LLM answer generation failed: {e}", exc_info=True)
        truncated = raw_data[:2000] if len(raw_data) > 2000 else raw_data
        return f"⚠️ Could not generate a summary. Raw data:\n{truncated}"