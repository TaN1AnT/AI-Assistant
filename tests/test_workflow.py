"""
Test Workflow — Tests RAG retrieval using Vertex AI Python SDK.

Test 1: Direct RAG search via vertexai.rag (API v1)
Test 2: RAG search via MCP Knowledge Server

Usage:
    cd d:/AI_Orchestration
    python tests/test_workflow.py
"""
import os
import sys
import json
import logging
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_workflow")


class C:
    OK    = "\033[92m"
    WARN  = "\033[93m"
    FAIL  = "\033[91m"
    INFO  = "\033[96m"
    BOLD  = "\033[1m"
    RESET = "\033[0m"

def header(text):
    print(f"\n{C.BOLD}{'═' * 60}\n  {text}\n{'═' * 60}{C.RESET}\n")

def step(num, text):
    print(f"  {C.INFO}[Крок {num}]{C.RESET} {text}")

def ok(text):
    print(f"  {C.OK}  ✅ {text}{C.RESET}")

def fail(text):
    print(f"  {C.FAIL}  ❌ {text}{C.RESET}")

def warn(text):
    print(f"  {C.WARN}  ⚠️  {text}{C.RESET}")


TEST_QUERY = "Які технічні характеристики Conveniq?"


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 1: Direct RAG via Vertex AI SDK
# ═══════════════════════════════════════════════════════════════════════════

def test_01_rag_direct():
    """Tests RAG retrieval via vertexai.rag SDK with service account credentials."""
    header("ТЕСТ 1: RAG-пошук через vertexai SDK (API v1)")

    from google.oauth2 import service_account
    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

    step(1, "Перевірка змінних середовища...")
    project_id = os.getenv("GOOGLE_PROJECT_ID", "")
    location = os.getenv("GOOGLE_LOCATION", "")
    corpus_id = os.getenv("VERTEX_RAG_CORPUS_ID", "")
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

    for name, val in [("GOOGLE_PROJECT_ID", project_id),
                      ("GOOGLE_LOCATION", location),
                      ("VERTEX_RAG_CORPUS_ID", corpus_id),
                      ("GOOGLE_APPLICATION_CREDENTIALS", creds_path)]:
        if not val:
            fail(f"{name} не встановлено")
            return False
        ok(f"{name} = {val[:60]}{'...' if len(val) > 60 else ''}")

    if not os.path.isfile(creds_path):
        fail(f"Файл не знайдено: {creds_path}")
        return False

    step(2, "Ініціалізація Vertex AI з service account...")
    try:
        import vertexai
        credentials = service_account.Credentials.from_service_account_file(
            creds_path, scopes=SCOPES
        )
        ok(f"SA email: {credentials.service_account_email}")
        vertexai.init(project=project_id, location=location, credentials=credentials)
        ok("vertexai.init() виконано")
    except Exception as e:
        fail(f"Помилка ініціалізації: {e}")
        return False

    step(3, f"Запит: \"{TEST_QUERY}\"")
    try:
        from vertexai import rag
        response = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=corpus_id)],
            text=TEST_QUERY,
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
            warn("Документів не знайдено")
            return True

        ok(f"Знайдено {len(contexts)} фрагментів")
        for i, chunk in enumerate(contexts[:3], 1):
            preview = chunk["text"][:120].replace("\n", " ")
            print(f"       [{i}] {preview}...")
    except Exception as e:
        fail(f"RAG помилка: {e}")
        logger.error("RAG error", exc_info=True)
        return False

    step(4, "Генерація відповіді через Gemini...")
    try:
        from mcp_server.shared.llm_helper import generate_answer
        docs_parts = []
        for chunk in contexts:
            if chunk["source"]:
                docs_parts.append(f"[Джерело: {chunk['source']}]\n{chunk['text']}")
            else:
                docs_parts.append(chunk["text"])

        answer = generate_answer(
            "Відповідай українською на основі контексту.",
            "\n\n---\n\n".join(docs_parts),
            TEST_QUERY,
        )
        ok("Відповідь:")
        print(f"\n{C.BOLD}       ┌{'─' * 50}")
        for line in answer.split("\n"):
            print(f"       │ {line}")
        print(f"       └{'─' * 50}{C.RESET}\n")
        return True
    except Exception as e:
        fail(f"Gemini помилка: {e}")
        logger.error("LLM error", exc_info=True)
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 2: RAG via MCP
# ═══════════════════════════════════════════════════════════════════════════

def test_02_rag_via_mcp():
    header("ТЕСТ 2: RAG через MCP Knowledge Server")

    import requests as http_requests
    step(1, "Перевірка з'єднання...")

    knowledge_url = os.getenv("MCP_KNOWLEDGE_URL", "http://127.0.0.1:8081/sse")
    base_url = knowledge_url.replace("/sse", "")

    try:
        resp = http_requests.get(base_url, timeout=5)
        ok(f"Сервер: {base_url} ({resp.status_code})")
    except http_requests.ConnectionError:
        warn(f"Не запущено: {base_url}")
        return None

    step(2, "Виклик rag_search...")
    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        async def call_rag():
            async with sse_client(knowledge_url) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    names = [t.name for t in tools.tools]
                    ok(f"Інструменти: {names}")
                    if "rag_search" not in names:
                        fail("rag_search не знайдено!")
                        return None
                    return await session.call_tool("rag_search", arguments={"query": TEST_QUERY})

        result = asyncio.run(call_rag())
        if result is None:
            return False

        answer = "".join(item.text for item in result.content if hasattr(item, "text"))
        if answer:
            ok("Відповідь:")
            print(f"\n{C.BOLD}       ┌{'─' * 50}")
            for line in answer.split("\n"):
                print(f"       │ {line}")
            print(f"       └{'─' * 50}{C.RESET}\n")
            return True
        fail("Порожня відповідь")
        return False
    except Exception as e:
        fail(f"Помилка: {e}")
        return False


if __name__ == "__main__":
    header("AI Orchestration — Тести")
    results = {
        "Прямий RAG (vertexai SDK)": test_01_rag_direct(),
        "RAG через MCP": test_02_rag_via_mcp(),
    }

    header("ПІДСУМОК")
    for name, r in results.items():
        if r is True:    print(f"  {C.OK}✅ ПРОЙДЕНО{C.RESET}  {name}")
        elif r is False: print(f"  {C.FAIL}❌ ПОМИЛКА{C.RESET}   {name}")
        elif r is None:  print(f"  {C.WARN}⏭️  ПРОПУЩЕНО{C.RESET} {name}")

    sys.exit(1 if any(r is False for r in results.values()) else 0)
