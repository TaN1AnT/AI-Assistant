"""
Application Settings — Environment-based configuration.

Loads from .env file via pydantic-settings.
Initializes Google Cloud credentials from service-account.json with
the cloud-platform scope, then passes them to vertexai.init().
"""
import os
import logging
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class Settings(BaseSettings):
    # ── Google Cloud ──
    GOOGLE_APPLICATION_CREDENTIALS: str = ""  # Path to service-account.json
    GOOGLE_PROJECT_ID: str
    GOOGLE_LOCATION: str = "europe-west4"
    VERTEX_RAG_CORPUS_ID: str

    # ── LLM Model ──
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"

    # ── n8n ──
    N8N_WEBHOOK_SECRET: str = ""

    # ── MCP Server URLs ──
    MCP_KNOWLEDGE_URL: str = "http://127.0.0.1:8081/sse"
    MCP_CRM_URL: str = "http://127.0.0.1:8082/sse"
    MCP_AUTOMATION_URL: str = "http://127.0.0.1:8083/sse"

    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()

# ── Load service account credentials with cloud-platform scope ──
credentials = None
if settings.GOOGLE_APPLICATION_CREDENTIALS:
    try:
        credentials = service_account.Credentials.from_service_account_file(
            settings.GOOGLE_APPLICATION_CREDENTIALS,
            scopes=SCOPES,
        )
        logger.info(f"Loaded service account: {credentials.service_account_email}")
    except Exception as e:
        logger.error(f"Failed to load service account: {e}")

# ── Initialize Vertex AI with explicit credentials ──
import vertexai
vertexai.init(
    project=settings.GOOGLE_PROJECT_ID,
    location=settings.GOOGLE_LOCATION,
    credentials=credentials,
)
