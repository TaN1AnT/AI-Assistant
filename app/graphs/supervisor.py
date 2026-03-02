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
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
import logging

from app.utils import get_gemini_llm
from app.mcp_client import mcp_client
from app.graphs.state import AgentState

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 2

# ── System Prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are an Enterprise AI Assistant with access to specialized tools.

AVAILABLE TOOLS:

**Knowledge Server** (internal documents):
- `rag_search(query)` — Search policies, docs, how-to guides.

**CRM Server** (business data via n8n):
- `get_deals(access_token, status_filter)` — List deals. Filters: 'open', 'won', 'lost', 'all'.
- `get_tasks(access_token, deal_id)` — List tasks for a deal.
- `get_task_comments(access_token, task_id)` — Get comments on a task.
- `get_checklists(access_token, task_id)` — Get checklist items for a task.
- `get_subtasks(access_token, task_id)` — Get subtasks (child tasks) for a task.
- `get_approvals(access_token, task_id)` — Get approval statuses for a task.
- `get_time_tracking(access_token, task_id)` — Get time entries logged on a task.

**Automation Server** (actions via n8n):
- `create_task(access_token, deal_id, title, description)` — Create a new task.
- `send_notification(access_token, recipient, message, channel)` — Send email/Slack/SMS.

CRITICAL RULES:
1. For ALL CRM and Automation tools, you MUST pass this access_token: `{access_token}`
2. Always use the appropriate tool. NEVER guess answers — always call a tool first.
3. For multi-step questions, call tools sequentially until you have ALL the info needed.
4. If the user asks about a deal AND its tasks, call BOTH get_deals AND get_tasks.
5. If you need related data (e.g. comments on a task, subtasks), fetch it proactively.
6. Present tool results clearly and professionally.
7. If a tool returns an error, explain it and suggest alternatives.
8. Never expose API keys, tokens, or webhook URLs in your responses."""


# ── Validation Prompt ──────────────────────────────────────────────────────

VALIDATION_PROMPT = """Review the conversation above. The user's original question is in the first HumanMessage.

You have gathered some data via tool calls. Now evaluate:

1. Does the data you collected FULLY answer the user's question?
2. Is there any MISSING information that requires additional tool calls?

Examples of incomplete answers:
- User asked about a deal's tasks AND comments, but you only fetched tasks.
- User asked for a full report on a deal, but you didn't fetch subtasks or checklists.
- User asked about time spent, but you didn't call get_time_tracking.

Respond with EXACTLY one of:
- "COMPLETE" — if all necessary information has been gathered.
- "INCOMPLETE: <reason>" — if more tool calls are needed. Explain what's missing.

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

    async def supervisor_node(state: AgentState):
        """
        The agent brain. Uses LLM to decide which tool(s) to call next.
        Injects the access_token into the system prompt.
        Tracks iteration count to prevent infinite loops.
        """
        messages = state["messages"]
        token = state.get("access_token", "NOT_PROVIDED")
        iteration = state.get("_iteration", 0) + 1

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(access_token=token)

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages

        # Add a cooling sleep to mitigate rate-limiting if we are deep in the loop
        if iteration > 2:
            import asyncio
            await asyncio.sleep(1)

        response = await llm_with_tools.ainvoke(messages)

        return {"messages": [response], "_iteration": iteration}

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

        logger.info(f"Validator (iter {iteration}): {verdict[:80]}")

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
        if validation == "INCOMPLETE":
            return "loop"
        return END

    # ── Build the graph ────────────────────────────────────────────────

    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("tools", ToolNode(tools))
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