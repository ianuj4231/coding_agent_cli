LangChain and LangGraph are related, but they solve different layers of your application.

## Simple difference

| Technology | What it provides | Why your project uses it |
|---|---|---|
| LangChain | LLMs, tools, prompts, documents, retrieval, middleware | Defines what your coding agent can do |
| LangGraph | Nodes, execution flow, state, memory, pause/resume | Controls how the agent repeatedly performs those actions |

Your understanding is close, but LangChain is not limited to simple chatbots or FAQs. LangChain supplies the building blocks; LangGraph coordinates multi-step, stateful execution.

## LangChain in your project

In [factory.py (line 171)](C:/Users/hp/Downloads/claude1/claude-rag-agent/ia_claude/agent/factory.py:171), you create the agent:

```python
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=full_prompt,
    ...
)
```

LangChain is being used for:

- Connecting your LLM.
- Defining the system prompt.
- Converting Python functions into LLM tools.
- Giving those tools to the agent.
- Summarizing long conversations.
- Adding human approval middleware.

For example, [tools.py (line 22)](C:/Users/hp/Downloads/claude1/claude-rag-agent/ia_claude/agent/tools.py:22) uses:

```python
@tool
def search_codebase(query: str) -> str:
```

`@tool` tells LangChain:

“The LLM is allowed to call this Python function.”

So LangChain defines the agent’s capabilities.

## LangGraph in your project

You do not manually create a graph using `StateGraph` or `add_node`.

Instead, LangChain’s `create_agent()` builds and compiles a LangGraph agent internally.

Conceptually, the generated graph behaves like this:

```text
User question
     ↓
Model node
     ↓
Does the model need a tool?
   ↙          ↘
 Yes           No
  ↓             ↓
Tool node     Final answer
  ↓
Model node
```

The graph can loop several times:

```text
Model → search_codebase → Model → read_file → Model → final answer
```

That looping, state management, pausing, and resuming are handled by LangGraph.

## What is a node?

A node is one processing step in a graph.

Think of it as a station through which the request passes.

Your primary conceptual nodes are:

- Model node: The LLM examines the question and decides what to do.
- Tool execution node: Runs tools such as `search_codebase`, `read_file`, or `run_command`.
- Middleware steps: Safety checks, memory injection, summarization, and human approval.

Important distinction: `search_codebase` is a LangChain tool. It is executed through the LangGraph tool-execution part of the workflow. It is not a manually declared node in your code.

## Example from your project

Suppose the user asks:

“Find where short-term memory is implemented and explain it.”

The flow is approximately:

1. User question enters the graph
2. Model decides it needs codebase information
3. Graph runs `search_codebase`
4. Search results return to the model
5. Model may decide to run `read_file`
6. Graph runs `read_file`
7. Model produces the final explanation

One user request can therefore cause multiple model and tool steps.

The request starts the graph here in [orchestrator.py (line 179)](C:/Users/hp/Downloads/claude1/claude-rag-agent/ia_claude/agent/orchestrator.py:179):

```python
response = await agent.ainvoke(...)
```

## Other direct LangGraph usage

You also use LangGraph for:

- Short-term conversation state: `AsyncSqliteSaver` in [short_term.py (line 2)](C:/Users/hp/Downloads/claude1/claude-rag-agent/ia_claude/memory/short_term.py:2).
- Long-term memory storage: `AsyncSqliteStore` in [long_term.py (line 10)](C:/Users/hp/Downloads/claude1/claude-rag-agent/ia_claude/memory/long_term.py:10).
- Pause and resume: `Command(resume=...)` after human approval in [orchestrator.py (line 205)](C:/Users/hp/Downloads/claude1/claude-rag-agent/ia_claude/agent/orchestrator.py:205).

The easiest way to remember it is:

LangChain gives your agent its brain and tools. LangGraph manages the workflow, loops, state, memory, and interruptions.




LangChain provides the building blocks:
LLM connection
Prompts
Tools
Documents and retrieval
Middleware

LangGraph manages how those blocks work together:
Execution flow
Repeated tool calls
State and memory
Pause and resume
In short:
LangChain provides the components. LangGraph controls their workflow.


