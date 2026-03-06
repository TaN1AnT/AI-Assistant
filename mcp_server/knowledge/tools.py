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

RAG_SYSTEM_PROMPT = """You are a documentation assistant.
Rules:
1. Answer strictly based on the provided context. If no info, say you don't know.
2. CITATIONS: Every claim MUST end with a citation marker: [Source: filename (URL)].
3. Use the exact filename and 'Signed URL' from the context.
4. CLEAN ANSWER: DO NOT create a separate 'Sources' or 'Джерела' section at the end.
5. Your inline markers like [Source: Name (URL)] will be extracted and moved to a dedicated list by the system."""


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


def _generate_signed_url(gcs_uri: str) -> str:
    """
    Generates a signed URL for a GCS object.
    gcs_uri format: gs://bucket/path/to/object
    """
    if not gcs_uri.startswith("gs://"):
        return gcs_uri

    try:
        from google.cloud import storage
        import datetime

        # Extract bucket and blob names
        path = gcs_uri.replace("gs://", "")
        bucket_name = path.split("/")[0]
        blob_name = "/".join(path.split("/")[1:])

        # Use the same credentials as Vertex AI
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        if creds_path and os.path.isfile(creds_path):
            client = storage.Client.from_service_account_json(creds_path)
        else:
            client = storage.Client()

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(hours=1),
            method="GET",
        )
        return url
    except Exception as e:
        logger.warning(f"Failed to generate signed URL for {gcs_uri}: {e}")
        return gcs_uri


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
        source_urls = {} # Cache for signed URLs to avoid redundant calls

        for chunk in contexts:
            source = chunk["source"]
            if source and source not in source_urls:
                source_urls[source] = _generate_signed_url(source)

            display_source = source_urls.get(source, source) if source else ""
            
            if source:
                parts.append(f"[Джерело: {source} ({display_source})]\n{chunk['text']}")
            else:
                parts.append(chunk['text'])

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
