"""
Knowledge Tools — RAG search using Vertex AI Python SDK.

Uses the vertexai library with service account credentials
to query the RAG corpus and generate grounded answers.
The vertexai import and init are deferred to first use to
avoid import crashes at server startup.

API Reference:
    https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/rag-api-v1
"""
import os
import logging
from mcp_server.shared.llm_helper import generate_answer

logger = logging.getLogger("mcp_server.knowledge.tools")

RAG_SYSTEM_PROMPT = (
    "Ти — корпоративний помічник знань, заснований на документації компанії.\n\n"
    "ПРАВИЛА:\n"
    "1. Відповідай ТІЛЬКИ на основі наданого контексту.\n"
    "2. Якщо відповіді немає в контексті, чітко скажи про це.\n"
    "3. Завжди цитуй джерело: [Джерело: назва_документа]\n"
    "4. Давай детальні, структуровані відповіді. Використовуй маркований список для переліку характеристик та пунктів.\n"
    "5. Відповідай мовою запиту (якщо запит українською — відповідай українською)."
)


# ── Vertex AI RAG Search ──────────────────────────────────────────────────

_vertexai_initialized = False


def _ensure_vertexai():
    """Lazily initialize Vertex AI SDK on first use."""
    global _vertexai_initialized
    if _vertexai_initialized:
        return

    from google.oauth2 import service_account
    import vertexai

    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    project_id = os.getenv("GOOGLE_PROJECT_ID", "")
    location = os.getenv("GOOGLE_LOCATION", "europe-west4")

    credentials = None
    if creds_path and os.path.isfile(creds_path):
        try:
            credentials = service_account.Credentials.from_service_account_file(
                creds_path, scopes=SCOPES
            )
            logger.info(f"Knowledge: loaded SA {credentials.service_account_email}")
        except Exception as e:
            logger.error(f"Knowledge: failed to load SA: {e}")

    vertexai.init(project=project_id, location=location, credentials=credentials)
    _vertexai_initialized = True
    logger.info("Knowledge: Vertex AI initialized")


def _rag_retrieve(query: str) -> str:
    """
    Queries the Vertex AI RAG corpus and returns formatted context.
    """
    _ensure_vertexai()

    corpus_id = os.getenv("VERTEX_RAG_CORPUS_ID", "")
    if not corpus_id:
        return "Error: VERTEX_RAG_CORPUS_ID is not configured."

    try:
        from vertexai import rag

        response = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=corpus_id)],
            text=query,
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=10,
                filter=rag.utils.resources.Filter(vector_distance_threshold=0.5),
            ),
        )

        contexts = []
        if response.contexts and response.contexts.contexts:
            for ctx in response.contexts.contexts:
                source = getattr(ctx, "source_uri", "") or ""
                text = getattr(ctx, "text", "") or ""
                if text:
                    contexts.append({"text": text, "source": source})

        if not contexts:
            return "No relevant documents found in the knowledge base."

        # Format contexts for the LLM
        parts = []
        for chunk in contexts:
            if chunk["source"]:
                parts.append(f"[Джерело: {chunk['source']}]\n{chunk['text']}")
            else:
                parts.append(chunk["text"])

        return "\n\n---\n\n".join(parts)

    except Exception as e:
        logger.error(f"RAG retrieval error: {e}", exc_info=True)
        return f"Error: RAG search failed: {type(e).__name__}: {e}"


# ── Tool registration ──────────────────────────────────────────────────────

def register_tools(mcp):
    """Register knowledge-domain tools."""

    @mcp.tool()
    def rag_search(query: str) -> str:
        """
        Search the company knowledge base (policies, docs, guides)
        and return a grounded answer.

        Use when the user asks about policies, product docs, or guides.
        """
        raw = _rag_retrieve(query)

        if raw.startswith("Error:"):
            return raw

        return generate_answer(RAG_SYSTEM_PROMPT, raw, query)
