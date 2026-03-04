# AI Orchestration Platform

Enterprise-grade AI agent system for CRM automation, document knowledge retrieval, and business process orchestration.

## 🚀 Core Features

-   **Multi-Agent Supervisor**: LangGraph-based orchestrator that breaks complex queries into parallel tool calls.
-   **Knowledge Base (RAG)**: Integrated with Google Vertex AI for searching internal documents.
-   **CRM Integration**: Connects to CRM data via secure n8n webhooks.
-   **Automation Tools**: Capability to create tasks and send multi-channel notifications.
-   **Session Persistence**: Conversation history with TTL-based memory.
-   **Enterprise Logging**: Structured JSON logging and request tracing.

---

## 🛠 Prerequisites

-   **Python 3.12+**
-   **Google Cloud Project**: With Vertex AI API enabled.
-   **Service Account**: JSON key with `Vertex AI User` permissions (save as `service-account.json`).
-   **n8n Instance**: With configured webhooks for CRM/Automation actions.

---

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd AI_Orchestration
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**:
    Copy `.env.example` to `.env` and fill in your credentials.
    ```bash
    # Essential GCP config
    GOOGLE_PROJECT_ID=your-project-id
    GOOGLE_APPLICATION_CREDENTIALS=service-account.json
    VERTEX_RAG_CORPUS_ID=your-corpus-id

    # Webhook Secret (Shared with n8n)
    N8N_WEBHOOK_SECRET=your-secret-key
    ```

---

## 🏃 How to Start

You need to run the **MCP Cluster** and the **FastAPI Gateway** simultaneously.

### Step 1: Start MCP Servers
Starts 3 servers (Knowledge, CRM, Automation) with auto-restart capability.
```bash
python -m mcp_server.start_all
```

### Step 2: Start API Gateway
In a new terminal:
```bash
uvicorn main:api --host 0.0.0.0 --port 8000 --reload
```

---

## 🧪 Testing the Setup

### Health Check
Check if all MCP servers are connected and the graph is ready:
```bash
curl http://localhost:8000/health
```

### Chat Request (Sync)
```bash
curl -X POST http://localhost:8000/v1/webhook/n8n \
  -H "X-Webhook-Secret: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What subtasks are in T-501?",
    "access_token": "your-user-token"
  }'
```

---

## 🛡 Security

-   **API Security**: Protected via `X-Webhook-Secret` and `X-Auth-Token` headers.
-   **Rate Limiting**: Sliding window protection (60 requests/min/user).
-   **Input Sanitization**: Built-in protection against prompt injection attacks.
-   **Tool Isolation**: Each tool call runs in a safe sandbox; failure of 1 tool does not break the entire workflow.

---

## 📄 Documentation

For deep technical details on the request pipeline, tool inventory, and architecture diagrams, see:
[System Documentation (docs/system_documentation.md)](docs/system_documentation.md)
