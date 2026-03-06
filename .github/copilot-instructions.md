# AI-Assistant — Project Context (для агента написання коду)

## 🎯 Що за проект

**Enterprise AI Assistant** — мультиагентна AI-платформа для автоматизації бізнес-процесів, пошуку по внутрішній документації (RAG) та інтеграції з CRM-платформою Suppa. Розроблено для компанії **Modern Expo**.

Платформа приймає текстові запити від користувачів (через REST API або n8n webhooks), інтелектуально розбиває їх на підзадачі, виконує потрібні інструменти (пошук документів, CRUD операції в CRM, автоматизація) і повертає структуровану відповідь.

---

## 🏗 Архітектура (3 шари)

```
┌─────────────────────────────────────────────────────────┐
│               FastAPI Gateway (main.py)                  │
│   REST API: /v1/chat, /v1/chat/stream, /v1/webhook/n8n  │
│   Auth: X-Auth-Token → SecurityService (mock users)      │
│   Rate limiting: 60 req/min per user                     │
│   Session memory: TTLCache (5 хв, max 10 turns)          │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│          LangGraph Supervisor Agent                      │
│   ReAct loop: supervisor → tools → supervisor → ...      │
│   Validator loop: validator перевіряє повноту відповіді  │
│   Max 6 ітерацій, Gemini 2.5 Flash Lite (Vertex AI)     │
│   Checkpointer: MemorySaver (in-memory)                  │
└──────────────┬──────────────────────────────────────────┘
               │ MCP Protocol
┌──────────────▼──────────────────────────────────────────┐
│              MCP Tool Servers (2 сервери)                 │
│                                                          │
│  1. Knowledge MCP (Python, SSE, port 8081)               │
│     → rag_search: Vertex AI RAG corpus                   │
│                                                          │
│  2. Suppa MCP Server (Node.js, stdio)                    │
│     → suppa_list_entities, suppa_search_instances,        │
│       suppa_get_instance, suppa_create_instance,          │
│       suppa_update_instance, suppa_get_comments, ...      │
│     → Пряме API Suppa (https://sp.modern-expo.com)       │
│                                                          │
│  (CRM MCP на port 8082 — legacy, n8n webhooks)           │
│  (Automation MCP на port 8083 — create_task,             │
│   send_notification через n8n webhooks)                   │
└─────────────────────────────────────────────────────────┘
```

### Потік даних (Chat Request)

1. Клієнт → `POST /v1/chat` з `X-Auth-Token` + `{message, session_id, access_token}`
2. FastAPI: auth → rate limit → sanitize input → load session history
3. LangGraph: `supervisor_node` (LLM вирішує які tools викликати)
4. `should_act` router: якщо є tool_calls → `safe_tool_node` → назад до supervisor
5. Якщо tool_calls немає → `validator_node` (LLM перевіряє повноту)
6. `should_loop` router: INCOMPLETE → назад до supervisor; COMPLETE → END
7. Фінальна відповідь → save to session → return ChatResponse

---

## 📁 Структура файлів

### API шар (`main.py`)
- **FastAPI** app з lifespan (startup: connect MCP → build graph)
- Endpoints:
  - `POST /v1/chat` — синхронний чат із session memory
  - `POST /v1/chat/stream` — SSE стрімінг
  - `POST /v1/action/resume` — Human-in-the-Loop resume
  - `POST /v1/webhook/n8n` — синхронний webhook для n8n
  - `POST /v1/webhook/n8n/async` — асинхронний webhook з callback
  - `GET /health` — health check
  - `GET /v1/mcp/status` — статус MCP з'єднань
  - `GET /v1/sessions/status` — активні сесії
- JSON structured logging (JSONFormatter)
- Pydantic models для request/response

### Конфігурація (`app/config.py`)
- `pydantic-settings` BaseSettings, читає `.env`
- Ключові змінні: `GOOGLE_PROJECT_ID`, `GOOGLE_APPLICATION_CREDENTIALS`, `VERTEX_RAG_CORPUS_ID`, `N8N_WEBHOOK_SECRET`
- MCP URLs: Knowledge (8081/sse), CRM (8082/sse), Automation (8083/sse)
- LLM модель: `gemini-2.5-flash-lite`
- Автоініціалізація `vertexai.init()` з service account credentials

### Граф агента (`app/graphs/`)
- **`state.py`** — `AgentState(TypedDict)`: messages, user_id, user_email, user_role, permissions, access_token, request_id, _iteration, _validation
- **`supervisor.py`** — ReAct agent з validation loop:
  - `build_agent_graph()` — async, будує StateGraph з 3 nodes + 2 conditional edges
  - `supervisor_node` — LLM (Gemini + tools) вирішує що робити
  - `safe_tool_node` — паралельне виконання tools з error isolation
  - `validator_node` — LLM без tools перевіряє повноту відповіді
  - `_sanitize_messages()` — фікс для Vertex AI (empty content → placeholder)
  - `_fallback_workflow` — базовий граф без tools (fallback)
  - System prompt з детальним описом всіх tools і SUPPA WORKFLOW

### MCP клієнт (`app/mcp_client.py`)
- `UnifiedMCPClient` — MultiServerMCPClient wrapper
- 2 серверних з'єднання:
  - `knowledge` — SSE transport (http://127.0.0.1:8081/sse)
  - `crm` — stdio transport (Node.js suppa-mcp-server)
- Auto-routing: `_tool_route_map` маршрутизує виклики до правильного сервера
- Кешує tools після першого connect()

### Безпека (`app/security.py`)
- `SecurityService` — mock token auth (3 тестових юзери: admin, sales_rep, guest)
- `RateLimiter` — in-memory, 60 req/min per user
- `sanitize_input()` — захист від prompt injection (regex фільтри)

### Сесії (`app/session_store.py`)
- `SessionStore` — TTLCache (5 хв TTL, max 200 сесій, max 10 turns)
- Зберігає condensed conversation pairs (human + ai + tools_summary)
- Автоматичне expire при неактивності

### LLM Helper (`app/utils.py`)
- `get_gemini_llm()` — ChatVertexAI з rate limiter (3 req/sec)

### MCP Servers (`mcp_server/`)
- **`start_all.py`** — лаунчер: запускає Knowledge MCP, auto-restart при crash
- **`knowledge/`** — FastMCP сервер (port 8081):
  - `rag_search` tool → Vertex AI RAG corpus query → LLM summarization
- **`crm/`** — FastMCP сервер (port 8082, legacy):
  - Tools: `get_task_comments`, `get_subtasks`, `get_checklists`, `get_approvals`, `get_time_tracking`
  - Проксує запити через n8n webhooks
- **`automation/`** — FastMCP сервер (port 8083):
  - `create_task`, `send_notification` — через n8n webhooks
- **`shared/`**:
  - `webhook_helper.py` — async httpx client для n8n (retry, connection pooling)
  - `llm_helper.py` — shared Gemini instance для MCP tools

### Suppa MCP Server (`suppa-mcp-server/`)
- **Node.js** (TypeScript) пакет, транспорт: stdio
- Використовує `@modelcontextprotocol/sdk`
- Інструменти для роботи з Suppa API:
  - CRUD entities/instances
  - Comments, mentions, child instances
  - Custom enum values
- API endpoint: `https://sp.modern-expo.com`

---

## 🛠 Технологічний стек

| Категорія | Технологія |
|-----------|-----------|
| **Мова** | Python 3.12+, Node.js 18+ (suppa-mcp-server) |
| **Web Framework** | FastAPI + Uvicorn |
| **AI/LLM** | Google Gemini 2.5 Flash Lite (через Vertex AI) |
| **Agent Framework** | LangGraph (StateGraph, ReAct pattern) |
| **LLM Integration** | LangChain Core, LangChain Google Vertex AI |
| **Tool Protocol** | MCP (Model Context Protocol) — FastMCP (Python) + MCP SDK (Node.js) |
| **RAG** | Vertex AI RAG API (corpus-based retrieval) |
| **CRM** | Suppa Platform (REST API) |
| **Automation** | n8n (webhook-based) |
| **Auth** | Custom token-based (mock, готове до OAuth2/JWT) |
| **Config** | pydantic-settings + .env |
| **HTTP Client** | httpx (async, connection pooling) |
| **Caching** | cachetools TTLCache (sessions) |
| **Logging** | Structured JSON logging |
| **Validation** | Pydantic v2 |

---

## 🔑 Ключові паттерни та принципи

1. **ReAct + Validation Loop** — LLM-агент ітеративно викликає tools, потім validator перевіряє повноту. Max 6 ітерацій.
2. **MCP (Model Context Protocol)** — стандартизований протокол для tool discovery та виклику. Підтримує SSE та stdio транспорти.
3. **Error Isolation** — `safe_tool_node` виконує кожен tool виклик незалежно; якщо один tool падає, інші продовжують працювати.
4. **Graceful Degradation** — якщо MCP не підключився, система працює в fallback режимі (без tools).
5. **Session Memory** — conversation history з TTL для підтримки контексту між запитами.
6. **Dual Webhook Mode** — sync (n8n чекає відповідь) та async (202 + callback) для інтеграції з n8n.
7. **Input Sanitization** — regex-based захист від prompt injection.
8. **Message Sanitization** — фікс для Vertex AI API (пусті content → placeholder), щоб уникнути 400 помилок.

---

## 📋 Env-змінні (основні)

```env
GOOGLE_PROJECT_ID=...
GOOGLE_APPLICATION_CREDENTIALS=service-account.json
GOOGLE_LOCATION=europe-west4
VERTEX_RAG_CORPUS_ID=...
GEMINI_MODEL=gemini-2.5-flash-lite
N8N_WEBHOOK_SECRET=...
MCP_KNOWLEDGE_URL=http://127.0.0.1:8081/sse
MCP_CRM_URL=http://127.0.0.1:8082/sse
MCP_AUTOMATION_URL=http://127.0.0.1:8083/sse
SUPPA_API_KEY=...
SUPPA_API_URL=https://sp.modern-expo.com
N8N_WEBHOOK_CRM_COMMENTS=...
N8N_WEBHOOK_CRM_SUBTASKS=...
N8N_WEBHOOK_CRM_CHECKLISTS=...
N8N_WEBHOOK_CRM_APPROVALS=...
N8N_WEBHOOK_CRM_TIME_TRACKING=...
N8N_WEBHOOK_AUTOMATION_CREATE_TASK=...
N8N_WEBHOOK_AUTOMATION_SEND_NOTIFICATION=...
```

---

## 🚀 Як запустити

```bash
# 1. MCP сервери (Knowledge)
python -m mcp_server.start_all

# 2. API Gateway (в іншому терміналі)
uvicorn main:api --host 0.0.0.0 --port 8000 --reload
```

Suppa MCP Server (Node.js) запускається автоматично через stdio з `mcp_client.py`.

---

## 📝 Важливі нюанси для розробки

- **CRM MCP (port 8082)** — це legacy сервер через n8n webhooks. Основна CRM інтеграція тепер через **suppa-mcp-server** (Node.js, stdio).
- **System prompt** в `supervisor.py` — це центральне місце конфігурації поведінки агента. Тут перелік всіх tools та правила їх використання.
- **`_sanitize_messages()`** — критична функція; без неї Vertex AI повертає 400 помилки на пусті AIMessage content.
- **Rate limiter** у двох місцях: FastAPI рівень (60 req/min per user) + LLM рівень (3 req/sec через InMemoryRateLimiter).
- **Тести** знаходяться в `tests/` — e2e тести для chat sessions, CRM, webhooks, supervisor, session memory.

