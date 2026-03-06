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
logger.setLevel(logging.DEBUG)

MAX_ITERATIONS = 6

# ── System Prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are an Enterprise AI Assistant connected to the Suppa platform and a knowledge base.

IMPORTANT RULE: You MUST use tools for every question. The only exception is simple greetings ("hello", "hi", "thanks", "goodbye") — respond to those directly.

For ALL other questions — even general ones — use the appropriate tool:
- Company docs, policies, products, guides → use rag_search(query)
- Tasks, deals, projects, records, users, CRM data → use Suppa tools (see workflow below)
- If unsure which tool to use → try rag_search first, then Suppa if no results

Available TOOLS:
- rag_search(query) — search internal company docs, policies, product specs
- suppa_list_entities(query?) — discover available entities (tables) in Suppa
- suppa_get_entity_props(entity_id) — get field names, types, comparators for an entity
- suppa_search_instances(entity_id, fields, filter?, order?, limit?, offset?) — search records
- suppa_get_instance(entity_id, instance_id, fields?) — get a single record
- suppa_get_child_instances(entity_id, instance_id) — get sub-records (subtasks)
- suppa_get_comments(entity_id, instance_id) — get comments/discussion thread
- suppa_get_mentions() — get unread @mentions for the user
- suppa_get_custom_enum_values(entity_id, prop_name) — get dropdown/enum options
- suppa_create_instance(entity_id, data) — create a new record
- suppa_update_instance(entity_id, instance_id, data) — update an existing record
- suppa_create_comment(entity_id, instance_id, text) — add a comment to a record

TOOL RULES:
1. For company docs/policies/products → use rag_search.
2. For Suppa data (tasks, deals, projects) → ALWAYS discover first: suppa_list_entities → suppa_get_entity_props → then read/write.
3. Never guess field names or enum values — call suppa_get_entity_props first.
4. RELATIONS: To fetch linked data (e.g. assignee name), use nested objects in `fields`: `{"responsible": {"id": {}, "name": {}}, "status": {"name": {}}}`.
5. TIMESTAMPS: Suppa uses Unix timestamps in MILLISECONDS.
6. Call multiple tools in parallel when possible.
7. Generate ONE structured answer with bullet points and icons (✅/❌/⏳).
8. Never expose tokens or plain internal URLs unless cited as follows.
9. METADATA FILTERING: If the user asks for "my" data (tasks, projects, etc.), use the provided `user_id` in the `filter` argument of Suppa tools. Example: `filter={"created_by_id": "provided_user_id"}` or `{"responsible_id": "provided_user_id"}`.

SUPPA WORKFLOW (follow this order):
1. User asks about records → identify which entity (table) is involved.
2. suppa_list_entities(query="task") → get entity_id.
3. suppa_get_entity_props(entity_id) → learn field names and types.
4. Read data: suppa_search_instances / suppa_get_instance / suppa_get_child_instances / suppa_get_comments.
5. Synthesize ONE clear, structured answer from all gathered data.

IMPORTANT: When using 'rag_search', you MUST follow these citation rules:
1. Every claim from a document must end with an inline citation marker: [Source: filename (URL)].
2. Use the exact 'filename' and clickable 'URL' (Signed URL) provided by the tool.
3. CLEAN ANSWER: Do NOT provide a 'Sources:' or 'Джерела:' section at the end. 
4. Just provide the answer with inline markers like [Source: Name (URL)]. These will be extracted and moved to a list by the system."""


# ── Validation Prompt ──────────────────────────────────────────────────────

VALIDATION_PROMPT = """Review the conversation. The user's question is in the first HumanMessage.

Rules:
- If the message is ONLY a greeting (hello, hi, hey, thanks, bye) → say COMPLETE.
- For ALL other questions: at least one tool MUST have been called. If no tools were called, say INCOMPLETE.
- If the user asked about company docs/policies but rag_search was not called → say INCOMPLETE.
- If the user asked about Suppa records but entity discovery (suppa_list_entities → suppa_get_entity_props) was skipped → say INCOMPLETE.
- If the user asked about comments but suppa_get_comments was not called → say INCOMPLETE.
- If data was fetched and answers the question → say COMPLETE.

Respond with EXACTLY:
- "COMPLETE" — if the answer is sufficient.
- "INCOMPLETE: [reason + specific tools to call]" — if critical data is missing.

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

    # ── Helper: sanitize messages for Vertex AI ─────────────────────
    def _sanitize_messages(msgs):
        """
        Vertex AI rejects messages with empty 'content' (400: must include parts).
        This ensures every message has some text content while preserving
        important nudges from the validator.
        """
        clean = []
        for m in msgs:
            # Copy to avoid mutating original state if possible
            content = getattr(m, 'content', "")
            
            if not content:
                if isinstance(m, AIMessage) and getattr(m, 'tool_calls', None):
                    # AI message with tool calls often has empty content
                    m = AIMessage(content="(calling tools)", tool_calls=m.tool_calls, id=m.id)
                elif isinstance(m, ToolMessage):
                    m = ToolMessage(content="(tool result)", tool_call_id=m.tool_call_id)
                elif isinstance(m, SystemMessage):
                    m = SystemMessage(content="(system prompt)")
                else:
                    m.content = "(empty)"
            
            clean.append(m)
        return clean

    # ── Node: Supervisor ───────────────────────────────────────────────

    # ── Build tool lookup for safe_tool_node ─────────────────────────
    tool_map = {t.name: t for t in tools} if tools else {}

    async def supervisor_node(state: AgentState):
        """
        The agent brain. Uses LLM to decide which tool(s) to call next.
        Tracks iteration count to prevent infinite loops.
        """
        messages = state["messages"]
        req_id = state.get("request_id", "")
        iteration = state.get("_iteration", 0) + 1

        user_id = state.get('user_id', 'Unknown')
        user_name = state.get('user_name', 'Unknown')
        user_email = state.get('user_email', 'no-email')
        
        context = f"\n\n[USER CONTEXT]\n- Name: {user_name}\n- ID: {user_id}\n- Email: {user_email}\n"
        system_prompt = SYSTEM_PROMPT_TEMPLATE + context

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        else:
            # Update existing system prompt if it's the first message
            messages[0] = SystemMessage(content=system_prompt)

        # Sanitize messages to prevent Vertex AI 400 errors
        logger.info(f"[{req_id}] Supervisor (iter {iteration}): sanitizing {len(messages)} messages")
        messages = _sanitize_messages(messages)

        # Add a cooling sleep to mitigate rate-limiting if we are deep in the loop
        if iteration > 2:
            logger.info(f"[{req_id}] Supervisor (iter {iteration}): cooling sleep...")
            await asyncio.sleep(1)

        logger.info(f"[{req_id}] Supervisor (iter {iteration}): invoking LLM with {len(messages)} messages...")
        for i, m in enumerate(messages):
            logger.debug(f"[{req_id}] Message {i} ({type(m).__name__}): {str(m.content)[:100]}...")

        response = await llm_with_tools.ainvoke(messages)
        logger.info(f"[{req_id}] Supervisor (iter {iteration}): LLM responded")

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
        req_id = state.get("request_id", "")
        iteration = state.get("_iteration", 0)
        last_reason = state.get("_validation", "")

        # Skip validation if we've hit the iteration limit
        if iteration >= MAX_ITERATIONS:
            logger.warning(f"[{req_id}] Max iterations ({MAX_ITERATIONS}) reached — forcing completion")
            return {"_validation": "COMPLETE"}

        # Build a validation request: full conversation + validation prompt
        logger.info(f"[{req_id}] Validator (iter {iteration}): sanitizing and building request")
        validation_messages = _sanitize_messages(list(messages)) + [
            HumanMessage(content=VALIDATION_PROMPT)
        ]

        logger.info(f"[{req_id}] Validator (iter {iteration}): invoking LLM...")
        response = await llm_validator.ainvoke(validation_messages)
        logger.info(f"[{req_id}] Validator (iter {iteration}): LLM responded")
        verdict = response.content.strip()

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