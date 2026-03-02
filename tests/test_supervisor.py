"""
Test Supervisor — Full AI assistant pipeline tests.

Test 1: Supervisor Graph (direct invocation — builds graph, connects MCP, invokes)
Test 2: Webhook endpoint (POST to /v1/webhook/n8n on localhost:8000)

Prerequisites:
    Terminal 1: python -m mcp_server.start_all
    Terminal 2: python main.py
    Terminal 3 (optional): python start_tunnel.py

Usage:
    python tests/test_supervisor.py
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
logger = logging.getLogger("test_supervisor")


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

def print_answer(answer):
    print(f"\n{C.BOLD}       ┌{'─' * 50}")
    for line in answer.split("\n"):
        print(f"       │ {line}")
    print(f"       └{'─' * 50}{C.RESET}\n")


TEST_QUERY = "Які технічні характеристики Conveniq?"


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 1: Supervisor Graph — direct invocation
# ═══════════════════════════════════════════════════════════════════════════

def test_01_supervisor_direct():
    """
    Builds the full supervisor graph (with MCP tools),
    invokes it directly with a query, and shows the result.
    Requires: MCP servers running (python -m mcp_server.start_all)
    """
    header("ТЕСТ 1: Supervisor Graph (прямий виклик)")

    async def run():
        # 1. Connect MCP client
        step(1, "Підключення MCP клієнта...")
        from app.mcp_client import mcp_client

        try:
            await mcp_client.connect()
            tools = await mcp_client.get_tools()
            ok(f"MCP: {len(tools)} інструментів")
            for t in tools:
                print(f"       • {t.name}")
        except Exception as e:
            warn(f"MCP недоступний: {e}")
            warn("Запустіть: python -m mcp_server.start_all")
            return None

        # 2. Build the supervisor graph
        step(2, "Побудова графа supervisor...")
        from app.graphs.supervisor import build_agent_graph
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.messages import HumanMessage, AIMessage

        agent_workflow = await build_agent_graph()
        graph = agent_workflow.compile(checkpointer=MemorySaver())
        ok("Граф скомпільовано")

        # 3. Invoke with test query
        step(3, f"Запит до supervisor: \"{TEST_QUERY}\"")
        config = {"configurable": {"thread_id": "test_supervisor_001"}}
        initial_state = {
            "messages": [HumanMessage(content=TEST_QUERY)],
            "user_id": "test-user",
            "user_email": "test@test.com",
            "user_role": "admin",
            "permissions": ["full_access"],
            "access_token": "",
            "_iteration": 0,
            "_validation": "",
        }

        result = await graph.ainvoke(initial_state, config=config)

        # 4. Parse result
        step(4, "Результат...")
        answer = result["messages"][-1].content

        tools_used = []
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                tools_used.extend([tc["name"] for tc in msg.tool_calls])

        ok(f"Кроків: {len(result['messages'])}")
        ok(f"Інструменти: {list(set(tools_used)) if tools_used else '(жоден)'}")
        ok("Відповідь:")
        print_answer(answer)

        # Cleanup
        await mcp_client.disconnect()
        return True

    try:
        return asyncio.run(run())
    except Exception as e:
        fail(f"Supervisor помилка: {e}")
        logger.error("Supervisor error", exc_info=True)
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 2: Webhook endpoint (/v1/webhook/n8n)
# ═══════════════════════════════════════════════════════════════════════════

def test_02_webhook():
    """
    Sends a POST to /v1/webhook/n8n on localhost:8000 (or ngrok URL).
    Requires: FastAPI running (python main.py) + MCP servers
    """
    header("ТЕСТ 2: Webhook POST → /v1/webhook/n8n")

    import requests as http_requests

    local_url = "http://localhost:8000"
    secret = os.getenv("N8N_WEBHOOK_SECRET", "password")

    # 1. Check FastAPI is running
    step(1, "Перевірка FastAPI...")
    try:
        resp = http_requests.get(f"{local_url}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            ok(f"FastAPI: {data.get('status')} | model: {data.get('model')} | v{data.get('version')}")
        else:
            fail(f"FastAPI: HTTP {resp.status_code}")
            return False
    except http_requests.ConnectionError:
        warn(f"FastAPI не запущено: {local_url}")
        warn("Запустіть: python main.py")
        return None

    # 2. Check MCP tools are loaded
    step(2, "Перевірка MCP інструментів...")
    try:
        resp = http_requests.get(f"{local_url}/v1/mcp/status", timeout=5)
        if resp.status_code == 200:
            mcp_data = resp.json()
            ok(f"MCP: {mcp_data.get('tool_count', 0)} інструментів")
            for t in mcp_data.get("tools", []):
                print(f"       • {t}")
        else:
            warn("MCP status not available")
    except Exception:
        warn("MCP status check failed")

    # 3. Send webhook request
    step(3, f"POST /v1/webhook/n8n: \"{TEST_QUERY}\"")
    payload = {"message": TEST_QUERY}
    headers = {
        "Content-Type": "application/json",
        "X-Webhook-Secret": secret,
    }

    print(f"       URL:     {local_url}/v1/webhook/n8n")
    print(f"       Secret:  {secret}")
    print(f"       Payload: {json.dumps(payload, ensure_ascii=False)}")
    print(f"       Чекаємо відповідь (timeout 120s)...")

    try:
        resp = http_requests.post(
            f"{local_url}/v1/webhook/n8n",
            json=payload,
            headers=headers,
            timeout=120,
        )
        print(f"       HTTP:    {resp.status_code}")

        if resp.status_code == 403:
            fail("Невірний webhook secret")
            return False
        if resp.status_code == 503:
            fail("Agent not ready — FastAPI ще ініціалізується")
            return False
        if resp.status_code != 200:
            fail(f"HTTP {resp.status_code}: {resp.text[:300]}")
            return False

        data = resp.json()
        ok("Відповідь отримана!")

    except http_requests.Timeout:
        fail("Timeout (120s) — supervisor працював занадто довго")
        return False
    except Exception as e:
        fail(f"HTTP помилка: {e}")
        return False

    # 4. Parse and display
    step(4, "Результат...")
    answer = data.get("answer", "")
    tools_used = data.get("tools_used", [])
    status = data.get("status", "")

    ok(f"Статус:      {status}")
    ok(f"Інструменти: {tools_used}")
    ok("Відповідь:")
    print_answer(answer)

    if not answer or len(answer) < 20:
        fail("Відповідь занадто коротка або порожня")
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    header("AI Orchestration — Тести Supervisor & Webhook")

    results = {}

    # Test 1 — may fail due to asyncio conflicts with SSE
    try:
        results["Supervisor (прямий виклик)"] = test_01_supervisor_direct()
    except Exception as e:
        warn(f"Тест 1 не вдався: {e}")
        results["Supervisor (прямий виклик)"] = False

    # Test 2 — HTTP-based, always works if FastAPI is running
    try:
        results["Webhook (/v1/webhook/n8n)"] = test_02_webhook()
    except Exception as e:
        warn(f"Тест 2 не вдався: {e}")
        results["Webhook (/v1/webhook/n8n)"] = False

    header("ПІДСУМОК")
    for name, r in results.items():
        if r is True:    print(f"  {C.OK}✅ ПРОЙДЕНО{C.RESET}  {name}")
        elif r is False: print(f"  {C.FAIL}❌ ПОМИЛКА{C.RESET}   {name}")
        elif r is None:  print(f"  {C.WARN}⏭️  ПРОПУЩЕНО{C.RESET} {name}")

    passed = sum(1 for r in results.values() if r is True)
    print(f"\n  {passed}/{len(results)} тестів пройдено")
    sys.exit(1 if any(r is False for r in results.values()) else 0)
