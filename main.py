import uvicorn
import time
import uuid
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Header, Depends, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Imports
from app.graphs.supervisor import build_agent_graph
from app.config import settings
from app.mcp_client import mcp_client
from app.security import SecurityService, RateLimiter

# Global State
app_state: Dict[str, Any] = {}
rate_limiter = RateLimiter(max_requests=60, window_seconds=60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager:
    1. Connects the MCP client to the FastMCP server
    2. Builds the agent graph with dynamically discovered tools
    3. Compiles the graph with a checkpointer
    """
    # 1. Connect MCP client
    try:
        await mcp_client.connect()
        logger.info("✅ MCP Client connected")
    except Exception as e:
        logger.warning(f"⚠️ MCP Client failed to connect: {e}. Running in fallback mode.")

    # 2. Build agent graph with MCP tools
    try:
        agent_workflow = await build_agent_graph()
        checkpointer = MemorySaver()
        app_state["graph"] = agent_workflow.compile(checkpointer=checkpointer)
        logger.info("✅ Agent graph compiled with MCP tools")
    except Exception as e:
        logger.error(f"❌ Failed to build agent graph: {e}", exc_info=True)
        # Fallback: import the basic workflow
        from app.graphs.supervisor import _fallback_workflow as fallback_workflow
        checkpointer = MemorySaver()
        app_state["graph"] = fallback_workflow.compile(checkpointer=checkpointer)
        logger.warning("⚠️ Using fallback graph without MCP tools")

    yield

    # Shutdown
    await mcp_client.disconnect()
    logger.info("🛑 Server shutting down")


# Initialize FastAPI
api = FastAPI(title="Enterprise CRM AI", version="3.0.0", lifespan=lifespan)

# CORS
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Security Dependency ──

def get_current_user(token: str = Header(..., alias="X-Auth-Token")):
    """Validates the token and returns the user profile."""
    user = SecurityService.verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid Authentication Token")
    return user


def check_rate_limit(request: Request, user: Dict[str, Any] = Depends(get_current_user)):
    """Checks rate limit for the authenticated user."""
    user_id = user["email"]
    if not rate_limiter.is_allowed(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait before making more requests.")
    return user


# ── Data Models ──

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    access_token: str = ""  # API token forwarded to n8n webhooks via MCP tools


class ActionResumeRequest(BaseModel):
    thread_id: str
    action: str


class LogEntry(BaseModel):
    step: int
    type: str
    source: str
    content: str
    details: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    status: str
    thread_id: str
    request_id: str
    execution_logs: List[LogEntry] = []


class N8NWebhookRequest(BaseModel):
    """Incoming request from n8n webhook."""
    message: str                              # User's chat message
    session_id: str = ""                      # n8n session ID for conversation tracking
    access_token: str = ""                    # CRM API token (if needed)
    callback_url: str = ""                    # n8n $execution.resumeUrl
    context: Optional[Dict[str, Any]] = None  # Extra context


class N8NWebhookResponse(BaseModel):
    """Response sent back to n8n."""
    answer: str
    sources: List[str] = []
    tools_used: List[str] = []
    status: str = "completed"


# ── Endpoints ──

@api.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, user: Dict[str, Any] = Depends(check_rate_limit)):
    """Standard Chat Endpoint."""
    graph: Any = app_state.get("graph")
    if not graph:
        raise HTTPException(status_code=503, detail="Agent not ready")

    request_id = str(uuid.uuid4())[:8]
    thread_id = req.thread_id or f"session_{user['email']}"
    config = {"configurable": {"thread_id": thread_id}}

    # Sanitize input
    sanitized_message = SecurityService.sanitize_input(req.message)

    initial_state = {
        "messages": [HumanMessage(content=sanitized_message)],
        "user_id": user["email"],
        "user_email": user["email"],
        "user_role": user["role"],
        "permissions": user["permissions"],
        "access_token": req.access_token,
        "_iteration": 0,
        "_validation": "",
    }

    try:
        result = await graph.ainvoke(initial_state, config=config)
        last_msg = result["messages"][-1].content
        logs = _extract_logs(result["messages"])

        return ChatResponse(
            response=last_msg,
            status="completed",
            thread_id=thread_id,
            request_id=request_id,
            execution_logs=logs,
        )

    except Exception as e:
        logger.error(f"[{request_id}] Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/v1/chat/stream")
async def stream_chat_endpoint(req: ChatRequest, user: Dict[str, Any] = Depends(check_rate_limit)):
    """Streaming Endpoint (SSE)."""
    graph: Any = app_state.get("graph")
    if not graph:
        raise HTTPException(status_code=503, detail="Agent not ready")

    thread_id = req.thread_id or f"session_{user['email']}"
    config = {"configurable": {"thread_id": thread_id}}
    sanitized_message = SecurityService.sanitize_input(req.message)

    async def event_generator():
        try:
            async for event in graph.astream_events(
                {
                    "messages": [HumanMessage(content=sanitized_message)],
                    "user_id": user["email"],
                    "user_email": user["email"],
                    "user_role": user["role"],
                    "permissions": user["permissions"],
                    "access_token": req.access_token,
                    "_iteration": 0,
                    "_validation": "",
                },
                config=config,
                version="v2",
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        yield f"data: {content}\n\n"

                elif kind == "on_tool_start":
                    tool_name = event["name"]
                    yield f"data: [TOOL: {tool_name}...]\n\n"

                elif kind == "on_tool_end":
                    yield f"data: [TOOL_DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: [ERROR: {str(e)}]\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@api.post("/v1/action/resume", dependencies=[Depends(get_current_user)])
async def resume_action(req: ActionResumeRequest):
    """Human-in-the-Loop Resume Endpoint."""
    graph: Any = app_state.get("graph")
    if not graph:
        raise HTTPException(status_code=503, detail="Agent not ready")

    config = {"configurable": {"thread_id": req.thread_id}}

    try:
        state_snapshot = await graph.aget_state(config)
        if not state_snapshot.next:
            raise HTTPException(status_code=400, detail="Thread is not paused.")

        from langgraph.types import Command

        result = await graph.ainvoke(
            Command(resume={"action": req.action}),
            config=config,
        )

        last_msg = result["messages"][-1].content
        return {"response": last_msg, "status": "resumed"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to resume: {str(e)}")


@api.get("/v1/mcp/status")
async def mcp_status():
    """Check MCP server connection and list available tools."""
    try:
        tools = await mcp_client.get_tools()
        tool_names = [t.name for t in tools]
        return {
            "connected": True,
            "tools": tool_names,
            "tool_count": len(tool_names),
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


@api.get("/health")
async def health_check():
    return {"status": "healthy", "model": settings.GEMINI_MODEL, "version": "3.0.0"}


# ── n8n Webhook Endpoints ──

async def _run_ai_query(graph, query: str, access_token: str, request_id: str, session_id: str = "") -> N8NWebhookResponse:
    """Shared logic: runs a query through the supervisor graph."""
    thread_id = session_id or f"n8n_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "user_id": "n8n-webhook",
        "user_email": "n8n@system",
        "user_role": "admin",
        "permissions": ["full_access"],
        "access_token": access_token,
        "_iteration": 0,
        "_validation": "",
    }

    result = await graph.ainvoke(initial_state, config=config)
    answer = result["messages"][-1].content

    tools_used = []
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            tools_used.extend([tc["name"] for tc in msg.tool_calls])

    logger.info(f"[{request_id}] Done. Tools: {list(set(tools_used))}")

    return N8NWebhookResponse(
        answer=answer,
        tools_used=list(set(tools_used)),
        status="completed",
    )


def _verify_webhook_secret(x_webhook_secret: str):
    """Check the webhook secret header."""
    expected = os.getenv("N8N_WEBHOOK_SECRET", "")
    if expected and x_webhook_secret != expected:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")


# ── SYNC: n8n sends request → waits → gets response ──

@api.post("/v1/webhook/n8n", response_model=N8NWebhookResponse)
async def n8n_webhook_sync(
    req: N8NWebhookRequest,
    x_webhook_secret: str = Header("", alias="X-Webhook-Secret"),
):
    """
    Synchronous webhook: n8n sends query, waits for AI response.

    Use this with n8n's HTTP Request node:
      POST → {server}/v1/webhook/n8n
      Header: X-Webhook-Secret
      Body: {"query": "...", "access_token": "..."}
    """
    _verify_webhook_secret(x_webhook_secret)

    graph = app_state.get("graph")
    if not graph:
        raise HTTPException(status_code=503, detail="Agent not ready")

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] n8n SYNC: '{req.message[:80]}'")

    try:
        response = await _run_ai_query(graph, req.message, req.access_token, request_id, req.session_id)
        return response
    except Exception as e:
        logger.error(f"[{request_id}] n8n webhook error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── ASYNC: n8n sends request → gets 202 → server calls back later ──

@api.post("/v1/webhook/n8n/async")
async def n8n_webhook_async(
    req: N8NWebhookRequest,
    background_tasks: BackgroundTasks,
    x_webhook_secret: str = Header("", alias="X-Webhook-Secret"),
):
    """
    Async webhook: returns 202 immediately, processes in background,
    sends result to callback_url when done.

    n8n flow:
      1. Chat Trigger → user sends message
      2. HTTP Request → POST {server}/v1/webhook/n8n/async
         Body: {"query": "...", "callback_url": "https://n8n.../webhook/ai-response"}
      3. Server returns 202 immediately
      4. Server processes query (RAG, tools, etc.)
      5. Server POSTs result → callback_url
      6. n8n Webhook node receives result → displays in chat
    """
    _verify_webhook_secret(x_webhook_secret)

    if not req.callback_url:
        # n8n's $execution.resumeUrl is always provided in async mode
        raise HTTPException(
            status_code=400,
            detail="callback_url is required for async mode. "
                   "Use /v1/webhook/n8n for synchronous mode.",
        )

    graph = app_state.get("graph")
    if not graph:
        raise HTTPException(status_code=503, detail="Agent not ready")

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] n8n ASYNC: '{req.message[:80]}' → callback: {req.callback_url}")

    # Schedule background processing
    background_tasks.add_task(
        _process_and_callback,
        graph, req.message, req.access_token, req.callback_url, request_id, req.session_id,
    )

    return {
        "status": "accepted",
        "request_id": request_id,
        "message": "Query accepted. Result will be sent to callback_url.",
    }


async def _process_and_callback(
    graph, query: str, access_token: str, callback_url: str, request_id: str, session_id: str = ""
):
    """Background task: run AI query and POST result to n8n callback."""
    import requests as http_requests

    try:
        response = await _run_ai_query(graph, query, access_token, request_id, session_id)
        payload = response.model_dump()

        logger.info(f"[{request_id}] Sending callback → {callback_url}")
        resp = http_requests.post(
            callback_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        logger.info(f"[{request_id}] Callback sent: HTTP {resp.status_code}")

    except Exception as e:
        logger.error(f"[{request_id}] Async processing failed: {e}", exc_info=True)
        # Try to notify n8n about the error
        try:
            http_requests.post(
                callback_url,
                json={"answer": f"Error: {e}", "status": "error", "tools_used": []},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
        except Exception:
            pass


# ── Helpers ──

def _extract_logs(messages: list) -> List[LogEntry]:
    """Extract structured execution logs from the message history."""
    logs = []
    step_count = 1

    # Find the last HumanMessage
    last_human_idx = -1
    for i, m in enumerate(reversed(messages)):
        if isinstance(m, HumanMessage):
            last_human_idx = len(messages) - 1 - i
            break

    new_messages = messages[last_human_idx + 1:] if last_human_idx != -1 else []

    for msg in new_messages:
        if isinstance(msg, AIMessage):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    logs.append(LogEntry(
                        step=step_count, type="tool_call",
                        source="ai_assistant",
                        content=f"Calling tool: {tc['name']}",
                        details=tc,
                    ))
                    step_count += 1
            elif msg.content:
                logs.append(LogEntry(
                    step=step_count, type="thought",
                    source="ai_assistant", content=str(msg.content)[:500],
                ))
                step_count += 1

        elif isinstance(msg, ToolMessage):
            logs.append(LogEntry(
                step=step_count, type="tool_result",
                source=getattr(msg, "name", "tool"),
                content=str(msg.content)[:500],
            ))
            step_count += 1

        elif isinstance(msg, SystemMessage):
            logs.append(LogEntry(
                step=step_count, type="system_update",
                source="system", content=str(msg.content)[:500],
            ))
            step_count += 1

    return logs


if __name__ == "__main__":
    uvicorn.run("main:api", host="0.0.0.0", port=8000, reload=True)