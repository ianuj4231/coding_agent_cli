store is the permanent memory box.

    checkpointer, seee why ths was init in main
does ltm need checkpointer at any point ?
 Baby step 3: Give the store to the agent
After opening the store, your code passes it to build_agent() in [main.py (line 305)](C:/Users/hp/Downloads/claude1/claude-rag-agent/ia_claude/main.py:305):
await initialize(
    checkpointer,
    user_id,
    long_term_store,
)
Then the store is attached to the agent in [factory.py (line 171)](C:/Users/hp/Downloads/claude1/claude-rag-agent/ia_claude/agent/factory.py:171):
agent = create_agent(
    ...
    store=long_term_store,
)
Think of it like this:
Open memory box → Give access to the agent
The agent now has access to the store through:
runtime.store
It still has not read or saved anything. It only has permission to use the memory box.

////
Baby step 6: Pass agent_context into the agent request
After creating:
agent_context = AgentContext(
    user_id=user_id,
    memory_enabled=memory_enabled,
    long_term_memories=memory_texts,  ## long tyerm memories of taht namespace.user id 
)
main.py passes it to handle_query() in [main.py (line 419)](C:/Users/hp/Downloads/claude1/claude-rag-agent/ia_claude/main.py:419):
response = await handle_query(
    agent,
    question,
    session_id,
    ...
    agent_context=agent_context,
)
Then handle_query() passes it into the agent in [orchestrator.py (line 179)](C:/Users/hp/Downloads/claude1/claude-rag-agent/ia_claude/agent/orchestrator.py:179):
response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": question}]},
    agent_config,
    context=agent_context,
)
The key part is:
context=agent_context
Simple intuition:
agent_context = a small delivery bag
It carries these three things into this request:
user_id
memory permission
Alice’s loaded memories
Now the information is available inside the agent as:
request.runtime.context
It has reached the agent, but the memories are not yet inside the model’s prompt. That is the next baby step.

///////

Before each LLM call, LangChain runs the function. Its returned string becomes the system prompt for that call.
Your function returns:
base_prompt + long_term_memory_text
So the flow is:
Original system prompt
        ↓
Dynamic middleware runs
        ↓
Memory text is added
        ↓
Combined prompt is sent to the LLM
Important nuance: LangChain technically uses the middleware’s returned prompt for that model call. Your code preserves the original prompt because it includes:
base_prompt = request.system_prompt or ""
If you returned only:
return "Alice prefers simple English"
then that would become the prompt, and the original system prompt would be missing.

///////


Baby step 8: The model decides to save the information
User says:
My long-term goal is to become a software architect.
The model receives:
System prompt + existing memories + user’s new message
Your system prompt tells the model:
Use the remember_user_information tool when the user shares
a long-term goal.
This instruction is in [factory.py (line 48)](C:/Users/hp/Downloads/claude1/claude-rag-agent/ia_claude/agent/factory.py:48).
Therefore, the model decides to call:
remember_user_information(
    memory="The user's long-term goal is to become a software architect."
)
This tool is registered with the agent in [factory.py (line 130)](C:/Users/hp/Downloads/claude1/claude-rag-agent/ia_claude/agent/factory.py:130).
At this exact step:
The model recognizes that this is durable information.
The model creates a tool call.
Nothing has been saved yet.
The next baby step is how this tool call uses runtime.store and runtime.context.user_id to save the memory.

///


Baby step 10: Send the memory to the save function
The tool calls this code in [memory_tools.py (line 35)](C:/Users/hp/Downloads/claude1/claude-rag-agent/ia_claude/agent/memory_tools.py:35):
memory_id = await save_long_term_memory(
    runtime.store,
    runtime.context.user_id,
    memory,
)
It sends three things:
runtime.store           → which database to use
runtime.context.user_id → which user owns the memory
memory                  → what should be saved
Example:
store   = long_term_memory.sqlite
user_id = "alice"
memory  = "The user's long-term goal is to become a software architect."
Notice how store and AgentContext work together:
store says: “Here is the memory box.”
AgentContext says: “Use Alice’s section.”
memory says: “Save this sentence.”
The information has now reached save_long_term_memory(). The actual SQLite write happens in the next baby step.

////////

Two different systems provide them.
Who sends memory?
The LLM creates it in its tool call.
Example:
User: My long-term goal is to become a software architect.
The LLM produces:
remember_user_information(
    memory="The user's long-term goal is to become a software architect."
)
So:
memory → provided by the LLM
Who sends runtime?
LangChain/LangGraph automatically injects it when executing the tool.
You do not include it in the tool call manually.
It contains:
runtime.store
runtime.context
These come from the earlier agent call:
agent.ainvoke(
    ...,
    context=agent_context,
)
and agent creation:
create_agent(
    store=long_term_store,
    context_schema=AgentContext,
)
Flow:
LLM supplies memory
        +
LangChain injects runtime
        ↓
remember_user_information(memory, runtime)
The LLM sees and controls the memory argument, but it does not see or control the internal runtime argument.


1:10 P