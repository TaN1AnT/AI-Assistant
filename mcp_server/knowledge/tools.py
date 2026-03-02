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
from google.oauth2 import service_account

logger = logging.getLogger("mcp_server.knowledge.tools")

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

RAG_SYSTEM_PROMPT = (
    "Ти — корпоративний помічник знань, заснований на документації компанії.\n\n"
    "ПРАВИЛА:\n"
    "1. Відповідай ТІЛЬКИ на основі наданого контексту.\n"
    "2. Якщо відповіді немає в контексті, чітко скажи про це.\n"
    "3. Завжди цитуй джерело: [Джерело: назва_документа]\n"
    "4. Давай детальні, структуровані відповіді. Використовуй маркований список для переліку характеристик та пунктів.\n"
    "5. Відповідай мовою запиту (якщо запит українською — відповідай українською)."
)


_vertex_initialized = False


def _ensure_vertex_init():
    """Lazy-init Vertex AI — only called on first RAG query."""
    global _vertex_initialized
    if _vertex_initialized:
        return

    import vertexai

    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    project_id = os.getenv("GOOGLE_PROJECT_ID", "")
    location = os.getenv("GOOGLE_LOCATION", "europe-west4")

    credentials = None
    if creds_path and os.path.isfile(creds_path):
        credentials = service_account.Credentials.from_service_account_file(
            creds_path, scopes=SCOPES
        )
        logger.info(f"Knowledge: SA {credentials.service_account_email}")

    vertexai.init(
        project=project_id,
        location=location,
        credentials=credentials,
    )
    _vertex_initialized = True
    logger.info("Vertex AI initialized for RAG")


def register_tools(mcp):
    """Register knowledge-domain tools."""

    @mcp.tool()
    def rag_search(query: str) -> str:
        """
        Search the company knowledge base (policies, docs, guides)
        and return a grounded answer.

        Use when the user asks about:
        - Company policies, procedures, rules
        - Product documentation
        - Internal how-to guides

        Supports queries in any language including Ukrainian.

        Args:
            query: The question to search for (any language).

        Returns:
            A sourced answer from internal documents.
        """
        corpus_id = os.getenv("VERTEX_RAG_CORPUS_ID", "")
        if not corpus_id:
            return "Error: RAG corpus not configured (VERTEX_RAG_CORPUS_ID missing)."

        try:
            # Lazy-init Vertex AI on first call
            _ensure_vertex_init()

            from vertexai import rag

            # Vertex AI RAG Engine API v1
            response = rag.retrieval_query(
                rag_resources=[
                    rag.RagResource(
                        rag_corpus=corpus_id,
                    )
                ],
                text=query,
                rag_retrieval_config=rag.RagRetrievalConfig(
                    top_k=10,
                    filter=rag.utils.resources.Filter(
                        vector_distance_threshold=0.5,
                    ),
                ),
            )

            # Extract text contexts
            contexts = []
            if response.contexts and response.contexts.contexts:
                for ctx in response.contexts.contexts:
                    source = getattr(ctx, "source_uri", "") or ""
                    text = getattr(ctx, "text", "") or ""
                    if text:
                        if source:
                            contexts.append(f"[Джерело: {source}]\n{text}")
                        else:
                            contexts.append(text)

            if not contexts:
                return f"Не знайдено релевантних документів за запитом: '{query}'"

            docs_text = "\n\n---\n\n".join(contexts)

            from mcp_server.shared.llm_helper import generate_answer
            return generate_answer(RAG_SYSTEM_PROMPT, docs_text, query)

        except Exception as e:
            logger.error(f"RAG search failed: {e}", exc_info=True)
            return f"Error: {type(e).__name__}: {e}"
