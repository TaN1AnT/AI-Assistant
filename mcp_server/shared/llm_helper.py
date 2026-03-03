"""
LLM Helper for MCP Server Tools

Provides a shared Gemini LLM instance and helper functions
so that each MCP tool can generate polished, human-friendly answers.

Authentication: Uses service account credentials from
GOOGLE_APPLICATION_CREDENTIALS with cloud-platform scope.
"""
import os
import logging
from google.oauth2 import service_account
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter

logger = logging.getLogger("mcp_server.shared.llm_helper")

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
_PROJECT = os.getenv("GOOGLE_PROJECT_ID", "")
_LOCATION = os.getenv("GOOGLE_LOCATION", "europe-west4")
_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Allow ~60 requests per minute (1 per second)
_rate_limiter = InMemoryRateLimiter(requests_per_minute=180, check_every_n_seconds=0.1, max_bucket_size=10)

# Load service account credentials
_credentials = None
if _CREDENTIALS_PATH and os.path.isfile(_CREDENTIALS_PATH):
    try:
        _credentials = service_account.Credentials.from_service_account_file(
            _CREDENTIALS_PATH,
            scopes=SCOPES,
        )
        logger.info(f"LLM Helper: loaded SA {_credentials.service_account_email}")
    except Exception as e:
        logger.error(f"LLM Helper: failed to load SA: {e}")


def get_llm(temperature: float = 0.2) -> ChatVertexAI:
    """Returns a configured Gemini LLM for use inside MCP tools."""
    kwargs = dict(
        model=_MODEL,
        project=_PROJECT,
        location=_LOCATION,
        temperature=temperature,
        max_retries=3,
        rate_limiter=_rate_limiter,
    )
    if _credentials:
        kwargs["credentials"] = _credentials
    return ChatVertexAI(**kwargs)


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
