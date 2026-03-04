"""
Supervisor Agent — ReAct agent with validation loop.

Architecture:
    supervisor ──→ should_act ──→ tools ──→ supervisor ──→ ...
                              └──→ validator ──→ should_loop ──→ supervisor (loop)
                                                            └──→ END

Flow:
1. supervisor_node: LLM decides what tool(s) to call next.
2. should_act: If tool_calls → go to tools. If no tool_calls → go to validator.
3. tool_node: Executes the MCP tool(s), returns results.
4. Loop back to supervisor (LLM sees results, may call more tools).
5. When LLM stops calling tools → validator_node checks completeness.
6. should_loop: If validator says INCOMPLETE → back to supervisor. If COMPLETE → END.
7. Max 10 iterations to prevent infinite loops.
"""
import asyncio
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
import logging

from app.utils import get_gemini_llm
from app.mcp_client import mcp_client
from app.graphs.state import AgentState

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6

# ── System Prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are an Enterprise AI Assistant. Gather raw data via tools, then synthesize one clear answer.

TOOLS (CRM tools return raw JSON — interpret it, never show raw JSON):
- rag_search(query) — search internal docs
- get_task_comments(access_token, task_id) — comments on a task
- get_checklists(access_token, task_id) — checklist items
- get_subtasks(access_token, task_id) — child tasks
- get_approvals(access_token, task_id) — approval statuses
- get_time_tracking(access_token, task_id) — time entries
- create_task(access_token, title, description) — create task
- send_notification(access_token, recipient, message, channel) — notify

RULES:
1. access_token for all CRM/Automation calls: `{access_token}`
2. Call multiple tools in parallel when possible.
3. For summaries/reports: call ALL relevant tools at once.
4. Generate ONE structured answer with bullet points and icons (✅/❌/⏳).
5. Never expose tokens or URLs.

DECOMPOSITION TEMPLATE:
- Simple (1 data point): call 1 tool → answer.
- Status/progress: get_subtasks + get_checklists (parallel).
- Blocked/ready check: get_subtasks + get_approvals + get_checklists (parallel).
- Full summary/report: get_subtasks + get_checklists + get_task_comments + get_approvals + get_time_tracking (all parallel).
- Multiple tasks: call the same tool for each task_id in parallel."""


# ── Validation Prompt ──────────────────────────────────────────────────────

VALIDATION_PROMPT = """Review the conversation above. The user's original question is in the first HumanMessage.

You have gathered raw data via tool calls. Evaluate completeness:

1. Does the collected data FULLY answer the user's question?
2. Is there MISSING information that requires additional tool calls?

Common patterns that indicate INCOMPLETENESS:
- User asked for a "summary" or "overview" of a task but only 1-2 data types were fetched.
  A proper task summary needs at minimum: subtasks + checklists + comments + approvals.
- User asked about "status" or "progress" but subtasks or checklists were not fetched.
- User mentioned time/hours but get_time_tracking was not called.
- User asked about approvals/sign-offs but get_approvals was not called.
- User asked about multiple tasks but data for some tasks was not fetched.
- User asked "is task ready?" but approvals + subtasks + checklists were not all checked.

Respond with EXACTLY one of:
- "COMPLETE" — if all necessary information has been gathered.
- "INCOMPLETE: <reason>" — if more tool calls are needed. List the SPECIFIC tools to call.

Do NOT generate a final answer. Only evaluate completeness."""


# ── Graph Construction ─────────────────────────────────────────────────────

async def build_agent_graph():
    """
    Builds the ReAct agent graph with a validation loop.
    Called once at FastAPI startup after MCP client connects.

    Graph structure:
        START → supervisor → [should_act]
                               ├─ "tools" → tool_node → supervisor (loop)
                               └─ "validate" → validator → [should_loop]
                                                              ├─ "loop" → supervisor
                                                              └─ END
    """
    tools = await mcp_client.get_tools()

    if not tools:
        logger.warning("No MCP tools available — agent will run without tools")

    llm = get_gemini_llm(temperature=0)
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    llm_validator = get_gemini_llm(temperature=0)  # Separate instance, no tools bound

    # ── Node: Supervisor ───────────────────────────────────────────────

    # ── Build tool lookup for safe_tool_node ─────────────────────────
    tool_map = {t.name: t for t in tools} if tools else {}

    async def supervisor_node(state: AgentState):
        """
        The agent brain. Uses LLM to decide which tool(s) to call next.
        Injects the access_token into the system prompt.
        Tracks iteration count to prevent infinite loops.
        """
        messages = state["messages"]
        token = state.get("access_token", "NOT_PROVIDED")
        req_id = state.get("request_id", "")
        iteration = state.get("_iteration", 0) + 1

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(access_token=token)

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages

        # Add a cooling sleep to mitigate rate-limiting if we are deep in the loop
        if iteration > 2:
            await asyncio.sleep(1)

        response = await llm_with_tools.ainvoke(messages)

        if response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            logger.info(f"[{req_id}] Supervisor (iter {iteration}): calling {tool_names}")
        else:
            logger.info(f"[{req_id}] Supervisor (iter {iteration}): no tools, moving to validation")

        return {"messages": [response], "_iteration": iteration}

    # ── Node: Safe Tool Node (error-isolated) ──────────────────────────

    async def safe_tool_node(state: AgentState):
        """
        Custom tool node that executes each tool call independently.
        If one tool fails, others still return their results.
        Failed tools return a ToolMessage with the error so the
        supervisor can continue with partial data.
        """
        last_msg = state["messages"][-1]
        req_id = state.get("request_id", "")

        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {"messages": []}

        async def _run_one(tc):
            name = tc["name"]
            call_id = tc["id"]
            try:
                tool = tool_map.get(name)
                if not tool:
                    return ToolMessage(content=f"Error: Unknown tool '{name}'", tool_call_id=call_id)
                result = await tool.ainvoke(tc["args"])
                return ToolMessage(content=str(result), tool_call_id=call_id)
            except Exception as e:
                logger.error(f"[{req_id}] Tool '{name}' failed: {e}")
                return ToolMessage(content=f"Error: {type(e).__name__}: {e}", tool_call_id=call_id)

        # Run ALL tool calls in parallel
        results = await asyncio.gather(
            *[_run_one(tc) for tc in last_msg.tool_calls]
        )

        logger.info(f"[{req_id}] Tools completed: {len(results)} results")
        return {"messages": list(results)}

    # ── Node: Validator ────────────────────────────────────────────────

    async def validator_node(state: AgentState):
        """
        Checks if the supervisor gathered all the information needed
        to fully answer the user's question.

        If INCOMPLETE, adds a SystemMessage nudging the supervisor
        to call more tools. The graph then loops back to supervisor.
        """
        messages = state["messages"]
        iteration = state.get("_iteration", 0)
        last_reason = state.get("_validation", "")

        # Skip validation if we've hit the iteration limit
        if iteration >= MAX_ITERATIONS:
            logger.warning(f"Max iterations ({MAX_ITERATIONS}) reached — forcing completion")
            return {"_validation": "COMPLETE"}

        # Build a validation request: full conversation + validation prompt
        validation_messages = list(messages) + [
            HumanMessage(content=VALIDATION_PROMPT)
        ]

        response = await llm_validator.ainvoke(validation_messages)
        verdict = response.content.strip()

        req_id = state.get("request_id", "")
        logger.info(f"[{req_id}] Validator (iter {iteration}): {verdict[:80]}")

        if verdict.startswith("COMPLETE"):
            return {"_validation": "COMPLETE"}
        else:
            reason = verdict.replace("INCOMPLETE:", "").strip()
            
            # Loop protection: if the reason is repeating (contained in previous validation state), stop looping
            if reason in last_reason:
                logger.warning(f"Validator reason repeating: '{reason}'. Breaking loop.")
                return {"_validation": "COMPLETE"}

            # Add a nudge message so the supervisor knows what to fetch next
            nudge = SystemMessage(
                content=f"[VALIDATION] The answer is incomplete. Missing: {reason}. "
                        f"Call the appropriate tools to get the missing information."
            )
            return {"messages": [nudge], "_validation": f"INCOMPLETE: {reason}"}

    # ── Router: should_act ─────────────────────────────────────────────

    def should_act(state: AgentState):
        """
        After supervisor: if it wants to call tools → 'tools'.
        If no tool calls (supervisor thinks it's done) → 'validate'.
        """
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "validate"

    # ── Router: should_loop ────────────────────────────────────────────

    def should_loop(state: AgentState):
        """
        After validator: if INCOMPLETE → loop back to supervisor.
        If COMPLETE → END the graph.
        """
        validation = state.get("_validation", "COMPLETE")
        if validation.startswith("INCOMPLETE"):
            return "loop"
        return END

    # ── Build the graph ────────────────────────────────────────────────

    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("tools", safe_tool_node)
    workflow.add_node("validator", validator_node)

    workflow.add_edge(START, "supervisor")

    # After supervisor: call tools or validate
    workflow.add_conditional_edges("supervisor", should_act, {
        "tools": "tools",
        "validate": "validator",
    })



    # After tools: always go back to supervisor
    workflow.add_edge("tools", "supervisor")


    # After validator: loop back or end
    workflow.add_conditional_edges("validator", should_loop, {
        "loop": "supervisor",
        END: END,
    })

    return workflow


# ── Fallback graph ─────────────────────────────────────────────────────────

_fallback_workflow = StateGraph(AgentState)


async def _fallback_node(state: AgentState):
    """Fallback when MCP tools are not loaded."""
    llm = get_gemini_llm(temperature=0)
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content="You are a helpful assistant. No tools are available.")] + messages
    response = await llm.ainvoke(messages)
    return {"messages": [response]}


_fallback_workflow.add_node("supervisor", _fallback_node)
_fallback_workflow.add_edge(START, "supervisor")
_fallback_workflow.add_edge("supervisor", END)

workflow = _fallback_workflow