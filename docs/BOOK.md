# The OpenAI Agents SDK — A Comprehensive Guide

> *Kind but concise. Intelligent but forgiving.*

---

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [The Loop](#the-loop)
4. [Human in the Loop](#human-in-the-loop)
5. [Tools & MCP](#tools--mcp)
6. [Tracing & Observability](#tracing--observability)
7. [Developer Guide](#developer-guide)
8. [User Guide](#user-guide)

---

# Introduction

## Welcome

There's a moment that many developers experience the first time they get a language model to call a function, receive the result, and loop back to produce a final answer. It feels like watching something come alive. The model isn't just generating text anymore — it's *doing things*.

That moment is the beginning of agentic AI. And this book is about building it well.

---

## What Is the OpenAI Agents SDK?

The OpenAI Agents SDK is a lightweight Python framework for building multi-agent applications. It gives you a small set of powerful primitives — agents, tools, handoffs, guardrails, and sessions — and then gets out of your way.

At its core, the SDK is the production-ready evolution of [Swarm](https://github.com/openai/swarm), OpenAI's earlier research exploration into multi-agent coordination. Where Swarm was an experiment, the Agents SDK is an answer: stable, extensible, observable, and built for real applications.

It's also deliberately modest in scope. The SDK doesn't try to reinvent Python's control flow or wrap every possible LLM interaction in an abstraction. Instead, it handles the hard parts — the agent loop, tool invocation, output validation, conversation persistence, and observability — while letting you write ordinary Python for everything else.

---

## The Problem It Solves

Language models are powerful but stateless. They respond to a prompt, return text, and stop. Turning that capability into an application means solving a cluster of problems that are tedious to handle yourself:

- **Looping**: When the model needs to call a tool, act on the result, and continue reasoning, you need a loop. That loop needs to handle tool execution, error recovery, and a stopping condition.
- **Tool integration**: Functions need to be described in a format the model understands, called safely, and have their outputs returned in a way the model can use.
- **Multi-agent coordination**: Some tasks are better handled by specialized agents working together — a triage agent routing to a billing agent, or a researcher handing off to a writer. Coordinating that transfer of control cleanly is its own challenge.
- **Safety**: Before acting on user input, you often want to validate it. After generating output, you want to check it. Running those checks efficiently, in parallel with the main agent flow, requires plumbing.
- **Persistence**: Conversations span multiple turns. Remembering what was said — without manually stitching together message histories — is the kind of bookkeeping that gets tedious fast.
- **Observability**: When something goes wrong (and it will), you need to see exactly what the model was given, what tools were called, what was returned, and where the flow diverged from expectations.

The Agents SDK solves all of these, with enough flexibility to handle the edge cases your specific application will inevitably surface.

---

## The Core Philosophy

The SDK is built around two principles stated plainly in its documentation:

> Enough features to be worth using, but few enough primitives to make it quick to learn.
> Works great out of the box, but you can customize exactly what happens.

That's not marketing language — it reflects real design choices. The entire public API is built around four concepts:

**Agents** are LLMs configured with a name, instructions, a set of tools, optional guardrails, and optional handoffs to other agents. They're the unit of capability in the system.

**Tools** are how agents act on the world. The SDK turns any Python function into a tool with a single decorator, automatically generating the schema the model needs and validating inputs with Pydantic. Built-in tools for web search, file search, code execution, computer use, and MCP servers are available without additional setup.

**Handoffs** are how agents collaborate. When one agent's job is done or out of scope, it passes control — along with context — to another agent. This is the mechanism that makes multi-agent architectures practical rather than theoretical.

**Guardrails** are validation layers that run in parallel with agent execution. Input guardrails check what comes in; output guardrails check what goes out. They can halt execution early, before wasted work or unsafe outputs reach users.

Beyond these four, the SDK adds **sessions** for persistent conversation memory, **tracing** for built-in observability, and support for **realtime voice agents** — but the core mental model stays small.

---

## Who This Book Is For

This book is for Python developers who want to build real applications with language models.

You might be new to agentic AI — curious about what it means for a model to "use tools" or "hand off to another agent," and looking for a solid foundation before writing production code. You'll find clear explanations of the concepts alongside working code.

You might be experienced with LLMs but frustrated by how much glue code typical applications require. You'll find that the Agents SDK removes the boilerplate while keeping you in control.

You might be evaluating frameworks and want to understand the tradeoffs this one makes. You'll find an honest account of what the SDK is, what it isn't, and when you'd choose something else.

You do not need prior experience with agent frameworks. You do need to be comfortable with Python — async/await in particular will appear throughout, since agent loops are naturally concurrent.

---

## What You'll Build

By the end of this book, you'll understand how to:

- Configure agents with instructions, tools, and output schemas
- Build multi-agent pipelines using handoffs and orchestration patterns
- Add input and output validation with guardrails
- Manage conversation state with sessions
- Stream agent responses and handle real-time interactions
- Trace, debug, and evaluate your agent workflows
- Build voice-capable agents using the realtime pipeline

Each chapter pairs explanation with working code. The examples are drawn from the SDK's own `examples/` directory and extended to illustrate practical patterns.

---

## A Note on Simplicity

One thing you'll notice as you work through this book is how little ceremony the SDK requires. The hello-world example — an agent that writes a haiku — is five lines:

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")
result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
```

That simplicity isn't a limitation. It's the result of careful design choices about what belongs in the framework and what belongs in your application. As the complexity of your use case grows, the SDK grows with it — but never by demanding you learn more abstractions than you need.

Let's get started.

---

# Architecture

If you squint at the openai-agents-python codebase from across the room, a clean shape emerges: a handful of composable abstractions, an async runtime that orchestrates them, and a model layer that's swappable by design. Everything else — tracing, MCP, sessions, extensions — plugs in around that core without disturbing it. This chapter walks you through that shape, one layer at a time.

---

## Repository Layout

```
openai-agents-python/
├── src/agents/           # The library itself
│   ├── agent.py          # Agent dataclass
│   ├── run.py            # Runner entry point
│   ├── run_config.py     # RunConfig + per-run options
│   ├── run_context.py    # RunContextWrapper (shared mutable state)
│   ├── run_internal/     # Runtime internals (loop, tool exec, session, etc.)
│   ├── models/           # Model interface + OpenAI implementations
│   ├── extensions/       # Optional add-ons (LiteLLM, memory backends, codex)
│   ├── guardrail.py      # Input/output guardrails
│   ├── handoffs/         # Handoff dataclass and history helpers
│   ├── tool.py           # Tool protocol and built-in tool types
│   ├── items.py          # RunItem types and ModelResponse
│   ├── memory/           # Session persistence
│   ├── mcp/              # MCP server integration
│   ├── realtime/         # Realtime (voice/WebSocket) agent subsystem
│   └── tracing/          # Spans and trace exporters
├── tests/                # Unit and snapshot tests
├── examples/             # Working SDK usage patterns
└── docs/                 # MkDocs source (english; ja/ko/zh auto-generated)
```

The `src/agents/` directory is what users import. `run_internal/` holds implementation details that are deliberately kept out of `run.py` to keep the public entry point readable.

---

## The Six Core Abstractions

### Agent

`Agent` is a Python dataclass (defined in `agent.py`). It holds *configuration*, not runtime state. Think of it as a recipe card:

```python
@dataclass
class Agent(AgentBase, Generic[TContext]):
    name: str
    instructions: str | Callable | None = None
    model: str | Model | None = None
    tools: list[Tool] = field(default_factory=list)
    handoffs: list[Agent | Handoff] = field(default_factory=list)
    input_guardrails: list[InputGuardrail] = field(default_factory=list)
    output_guardrails: list[OutputGuardrail] = field(default_factory=list)
    output_type: type | AgentOutputSchemaBase | None = None
    ...
```

Because it's a dataclass, `agent.clone(instructions="new prompt")` gives you a shallow copy with one field changed — handy for per-request customization without mutation. An `Agent` is generic over a context type `TContext`, which is a user-supplied object that flows through every tool call and guardrail in the run.

An agent can also call `as_tool()` on itself, which wraps its entire execution into a `FunctionTool` that another agent can invoke — enabling hierarchical agent graphs without needing a formal handoff.

### Runner

`Runner` (in `run.py`) is the orchestrator. It has two main entry points:

- `Runner.run(agent, input, ...)` — returns a `RunResult` once the agent is done.
- `Runner.run_streamed(agent, input, ...)` — returns a `RunResultStreaming` that you iterate for real-time events.

Both delegate almost immediately into `run_internal/`, which contains the actual turn loop (`run_loop.py`), tool execution (`tool_execution.py`), and session management (`session_persistence.py`). The division is intentional: `run.py` is the wiring; `run_internal/` is the machinery.

The loop runs until one of four outcomes arrives from the model: a final output, a handoff to another agent, an interruption (approval required), or `MaxTurnsExceeded`.

### Model

The `Model` abstract base class (`models/interface.py`) defines exactly two methods:

```python
class Model(abc.ABC):
    async def get_response(...) -> ModelResponse: ...
    def stream_response(...) -> AsyncIterator[TResponseStreamEvent]: ...
```

That's the entire contract. Every implementation — `OpenAIResponsesModel`, `OpenAIChatCompletionsModel`, the LiteLLM extension — implements these two methods and nothing else. The `ModelProvider` companion interface resolves a model name string into a `Model` instance.

This separation means you can point an agent at a completely different LLM backend by providing a custom `ModelProvider` in `RunConfig`, without touching any other part of the code.

### Tool

`Tool` is a protocol/union type (`tool.py`). The SDK ships several concrete implementations:

| Type | What it does |
|---|---|
| `FunctionTool` | Wraps a Python function; schema auto-derived from type hints |
| `FileSearchTool` | OpenAI hosted file search |
| `WebSearchTool` | OpenAI hosted web search |
| `ComputerTool` | Screen/keyboard automation |
| `CodeInterpreterTool` | OpenAI code interpreter |
| `MCPTool` | Tool fetched from an MCP server at runtime |

The `@function_tool` decorator is the common entry point — it inspects a function's signature and docstring to build the JSON schema the LLM sees, then wraps it in a `FunctionTool`. Tools can be sync or async; the runtime handles both.

### Handoff

A `Handoff` (`handoffs/__init__.py`) is how one agent hands control to another while passing conversation history. Mechanically it looks like a tool call to the LLM — the model calls `transfer_to_<agent_name>` — but the runtime treats it specially: rather than returning a result to the same agent, it switches the active agent.

```python
@dataclass
class Handoff(Generic[TContext, TAgent]):
    tool_name: str
    tool_description: str
    input_json_schema: dict
    on_invoke_handoff: Callable[..., Awaitable[TAgent]]
    input_filter: HandoffInputFilter | None = None
    ...
```

The `input_filter` is notable: it lets you reshape or trim the conversation history before the receiving agent sees it — useful for staying within token limits or omitting implementation details irrelevant to the next agent.

### Guardrail

Guardrails are safety or validation checks that run around agent execution. Input guardrails run at the start of the first turn; output guardrails run when the agent produces a final answer.

```python
@dataclass
class InputGuardrail(Generic[TContext]):
    guardrail_function: Callable[..., MaybeAwaitable[GuardrailFunctionOutput]]
    run_in_parallel: bool = True  # concurrent with model call by default
```

If a guardrail's `tripwire_triggered` is `True`, the run halts immediately and raises `InputGuardrailTripwireTriggered` or `OutputGuardrailTripwireTriggered`. Guardrails are decorated with `@input_guardrail` or `@output_guardrail` — both sync and async functions work.

---

## The Model Provider Abstraction

The model layer has three tiers:

```
RunConfig.model_provider (ModelProvider)
    └── .get_model(name) -> Model
            ├── OpenAIResponsesModel      (Responses API — default for OpenAI)
            ├── OpenAIChatCompletionsModel (Chat Completions API)
            └── LiteLLMModel              (extensions/models/litellm_model.py)
```

`OpenAIProvider` is the default `ModelProvider`. By default it uses the Responses API (`OpenAIResponsesModel`), which supports stateful conversation management via `previous_response_id`. Switching to the Chat Completions API is a one-line config change. The `MultiProvider` (used internally) lets you route different model names to different providers in the same run.

The LiteLLM extension (`extensions/models/litellm_model.py`) implements `Model` using the `litellm` package as a shim, giving access to Anthropic, Cohere, Azure, Bedrock, and hundreds of other providers without changing any application code. It's opt-in via `pip install 'openai-agents[litellm]'`.

---

## The Extensions System

The `extensions/` package is the SDK's "batteries included but removable" layer:

- `extensions/models/` — LiteLLM model and provider
- `extensions/memory/` — SQLite, Redis, SQLAlchemy, Dapr, and encrypted session backends
- `extensions/handoff_filters.py` — ready-made input filters (e.g., remove all tool calls from history)
- `extensions/handoff_prompt.py` — utilities for injecting handoff context into prompts
- `extensions/visualization.py` — render agent graphs
- `extensions/experimental/codex/` — Codex agent (file-editing, code execution) in early preview

Extensions import from `agents.*` but the core never imports from `extensions.*`, keeping the dependency direction clean.

---

## How the Pieces Connect

```
User code
    │
    ▼
Runner.run(agent, input, run_config)
    │
    ├── RunContextWrapper  ◄──────────────────────────────┐
    │   (shared mutable state, usage, approvals)          │
    │                                                     │
    ├── Session (memory)                                  │
    │   load prior history → prepend to input             │
    │                                                     │
    ▼                                                     │
[Turn loop]  run_internal/run_loop.py                     │
    │                                                     │
    ├── InputGuardrails (parallel with model call)        │
    │                                                     │
    ├── Agent.get_all_tools()                             │
    │   ├── Agent.tools (FunctionTool, etc.)              │
    │   └── MCP servers → MCPTool list                    │
    │                                                     │
    ├── Model.get_response() / .stream_response()         │
    │   └── ModelProvider.get_model(name)                 │
    │       ├── OpenAIResponsesModel                      │
    │       ├── OpenAIChatCompletionsModel                │
    │       └── LiteLLMModel (extension)                  │
    │                                                     │
    ▼                                                     │
ProcessedResponse (run_steps.py)                          │
    │                                                     │
    ├── NextStepFinalOutput ──► OutputGuardrails          │
    │                           └── RunResult             │
    │                                                     │
    ├── NextStepHandoff ──► switch active agent ──────────┘
    │                       (new turn, same loop)
    │
    ├── NextStepRunAgain ──► execute tools, loop again ───┘
    │
    └── NextStepInterruption ──► RunResult (interrupted)
                                 caller must resume
```

Every item the model generates — text messages, tool calls, handoff calls, reasoning traces — becomes a typed `RunItem` (from `items.py`). These accumulate in `RunResult.new_items` and are also persisted to the session if one is configured.

---

## Key Design Decisions

**Dataclasses + Pydantic, not class hierarchies.** `Agent`, `Handoff`, `Guardrail`, and `RunConfig` are all `@dataclass`. This keeps them inspectable, serializable, and easy to copy with `dataclasses.replace()`. Pydantic is used where validation and JSON schema generation are needed (`ModelResponse`, function tool schemas, output types).

**Async-first, sync-optional.** The runtime is entirely async. `Runner.run()` is a coroutine; `Runner.run_sync()` exists as a thin wrapper for callers who are not in an async context. This avoids the overhead and complexity of running two separate codebases for sync and async paths.

**Streaming and non-streaming are behaviorally identical.** `run_single_turn` and `run_single_turn_streamed` are kept in lockstep — if you add a new item type or change tool behavior, both paths need to reflect the change. This is enforced by convention and tested by snapshot tests.

**The context is yours.** `TContext` is a generic type parameter for user-supplied state. The SDK never inspects or mutates it — it just passes it to your tools, guardrails, and hooks. You choose the shape; the runtime is the carrier.

**Open for extension, closed for surprise.** The `Model`/`ModelProvider` interfaces are abstract base classes with exactly the methods the runtime needs. Adding a new backend means implementing two methods; nothing else in the codebase changes.

---

The architecture is deliberately narrow at the core and wide at the edges. Once you understand the `Agent → Runner → Model → Tool` flow, the rest of the codebase is variations on that theme. The next chapter follows a single request through the turn loop in detail.

---

# The Loop

There's a moment, when you first call `Runner.run()`, where the SDK quietly takes over. You hand it an agent and some input, and it does the rest — calling the model, running tools, switching agents if needed, and eventually handing you back a finished result. What happens in between is the loop.

Understanding the loop means understanding how the SDK thinks about time. Not wall-clock time, but turns.

## What Is a Turn?

A **turn** is one round trip to the model. Each turn begins when the SDK calls the model with the current conversation state and ends when it has processed the model's response. A run may have many turns; it's bounded by `max_turns` (default: `10`), which prevents runaway loops.

The turn counter only advances when the model is actually called. Resuming from an interruption — say, after a human-in-the-loop approval gate — doesn't count as a new turn. The SDK is careful about this: only real model calls advance the clock.

## Runner.run() at a High Level

The public entry point lives in `src/agents/run.py`. You call either `Runner.run()` (for a final result) or `Runner.run_streamed()` (for token-by-token streaming). Both methods converge on the same internal machinery:

```python
result = await Runner.run(agent, "What's the weather in Paris?")
```

After setup — building a `RunContextWrapper`, resolving tracing settings, loading session history — the non-streaming path calls `run_single_turn()` in a loop, while the streaming path launches `start_streaming()` as a background task that feeds a queue of events.

The outer loop looks roughly like this (simplified from `start_streaming` in `run_loop.py`):

```python
while True:
    current_turn += 1
    if current_turn > max_turns:
        raise MaxTurnsExceeded(...)

    turn_result = await run_single_turn(...)

    if isinstance(turn_result.next_step, NextStepFinalOutput):
        break  # We're done
    elif isinstance(turn_result.next_step, NextStepHandoff):
        current_agent = turn_result.next_step.new_agent  # Switch agents, loop again
    elif isinstance(turn_result.next_step, NextStepRunAgain):
        continue  # Tools ran, loop again
    elif isinstance(turn_result.next_step, NextStepInterruption):
        break  # Paused for approval
```

The loop keeps going until it knows what to do next. The `next_step` field on each `SingleStepResult` is the SDK's internal signal — a tagged value that says "keep going," "hand off," "we're finished," or "wait for a human."

## Inside a Single Turn

Each call to `run_single_turn()` (or `run_single_turn_streamed()`) runs through a predictable sequence. Let's walk it.

**1. Prepare the turn**

Before the model is called, the SDK gathers everything it needs: the agent's system prompt, all available tools (including MCP tools), possible handoff targets, and the output schema if the agent produces structured output. If this is turn 1, input guardrails run here — they check the user's message before anything reaches the model.

**2. Call the model**

The SDK calls `get_new_response()`, which invokes the model with the current input items and configuration. If a `call_model_input_filter` is set on `RunConfig`, it runs immediately before the API call, giving you a chance to trim the context window or inject additional instructions.

In the streaming path, tokens flow in as they arrive. In the non-streaming path, the SDK waits for the full response. Both paths then hand the completed response to the same processing logic.

**3. Process the response**

`process_model_response()` in `turn_resolution.py` reads the model's output and classifies each item. A text message becomes a `MessageOutputItem`. A function call becomes a `ToolRunFunction`. A handoff becomes a `ToolRunHandoff`. Computer actions, shell calls, MCP approval requests — each becomes a typed struct. The result is a `ProcessedResponse`: a structured view of everything the model asked for.

**4. Execute tools**

`execute_tools_and_side_effects()` runs all the tool calls the model requested. Function tools run concurrently (via `asyncio.gather`). Tool-level guardrails run too — input guardrails before execution, output guardrails after. If any tool needs human approval, the affected calls are held back and an interruption is signaled.

Tool results are appended to the conversation as new input items, forming the growing context the next turn will see.

**5. Decide the next step**

After tools finish, the SDK asks: are we done? This is where `check_for_final_output_from_tools()` runs. If a tool is configured to produce the final output directly, the run ends here. Otherwise, if there are no more tools to run and the model produced a message (or structured output), that becomes the final output via `NextStepFinalOutput`. If there's a handoff, `NextStepHandoff` names the new agent. If tools ran but nothing concluded, `NextStepRunAgain` tells the outer loop to go again.

## Handoffs: Switching Agents Mid-Run

A handoff is just a tool call with a special meaning. When the model invokes a handoff tool, `execute_handoffs()` runs the handoff's `on_handoff` callback, applies any input filters (which can reshape the conversation history), and wraps the result in `NextStepHandoff` pointing to the new agent.

Back in the outer loop, the runner updates `current_agent`, resets the agent start hooks, and emits an `AgentUpdatedStreamEvent` so that streaming consumers know who is talking now. The turn counter continues from where it left off. From the loop's perspective, a handoff is just an agent swap between turns — the machinery doesn't change.

## Streaming and Non-Streaming: Two Paths, One Model

One of the clearest design decisions in this SDK is the commitment to keeping streaming and non-streaming behaviorally identical. The AGENTS.md contributor guide even calls this out explicitly: changes to `run_single_turn` must be mirrored in `run_single_turn_streamed`.

The non-streaming path calls the model, waits, processes, returns. The streaming path does the same work, but it emits events as it goes — raw response deltas via `RawResponsesStreamEvent`, completed items via `RunItemStreamEvent`, agent changes via `AgentUpdatedStreamEvent`. A `QueueCompleteSentinel` signals the end of the stream.

The streaming path runs as a background task; `RunResultStreaming` exposes an async iterator that drains the queue. Your code can iterate events live, but the final `RunResult`-equivalent data is assembled the same way it would be in the blocking path.

Both paths produce the same `SingleStepResult` from each turn. The difference is only in how results are surfaced, not how they are computed.

## max_turns and RunConfig

`max_turns` is the loop's safety valve. Once `current_turn > max_turns`, the SDK raises `MaxTurnsExceeded`. By default this is `10`, enough for most workflows. You can raise or lower it via `RunConfig`:

```python
config = RunConfig(model="gpt-4o", max_turns=25)
result = await Runner.run(agent, input, run_config=config)
```

`RunConfig` is the broader knob panel for a run. Beyond `max_turns`, it lets you override the model globally (useful when you want all agents in a multi-agent run to use the same model), set `ModelSettings` like temperature, install input and output guardrails, configure tracing, attach a session for conversation persistence, and intercept model input via `call_model_input_filter`.

Think of `RunConfig` as the run's constitution — the rules that govern every agent and every turn in that run, regardless of how many agents get involved or how long the loop runs.

## Reading the Code

If you want to follow the loop yourself, start at `Runner.run()` in `src/agents/run.py` and trace into `start_streaming()` (for both paths, since the non-streaming path also drives through the same state machine). Then look at `run_single_turn()` in `src/agents/run_internal/run_loop.py` to see one turn's anatomy. `process_model_response()` in `turn_resolution.py` shows how model output becomes typed items, and `execute_tools_and_side_effects()` in the same file shows how those items turn into real side effects.

The data structures in `run_steps.py` are worth reading in full — they're small, well-documented, and they form the vocabulary the loop uses to talk to itself. `NextStepRunAgain`, `NextStepFinalOutput`, `NextStepHandoff`, `NextStepInterruption`: four cases, and one of them is always true at the end of a turn.

The loop is not magic. It's a `while True` with a turn counter and a four-way branch. Once you see that, the rest is just details.

---

# Human in the Loop

Agents are powerful precisely because they can take action autonomously. They call tools, make decisions, and carry tasks through to completion without needing constant guidance. But "autonomous" doesn't have to mean "unsupervised." The openai-agents-python SDK is built around a clear philosophy: agents should be capable of working independently, while humans retain meaningful control at every layer of the system.

This section explores the mechanisms the SDK provides to keep you in the loop — guardrails that validate what goes in and what comes out, tool-level checks that intercept individual calls, and a full pause-resume system that lets you halt an agent mid-run, collect a human decision, and continue exactly where you left off.

---

## Guardrails: Your Safety Net

Guardrails are validation functions that wrap the edges of an agent run. They are not part of the agent's reasoning; they are checks you write, running alongside or around the agent, that can trip a wire and halt execution the moment something looks wrong.

### Input Guardrails

Input guardrails run on the message the user sends before — or in parallel with — the agent processing it. They receive the raw input and return a `GuardrailFunctionOutput`. If `tripwire_triggered` is `True`, an `InputGuardrailTripwireTriggered` exception is raised and the agent halts immediately.

```python
from agents import Agent, GuardrailFunctionOutput, input_guardrail, RunContextWrapper

@input_guardrail
async def topic_check(ctx: RunContextWrapper, agent, input) -> GuardrailFunctionOutput:
    is_off_topic = "homework" in str(input).lower()
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=is_off_topic)

agent = Agent(
    name="Support agent",
    instructions="Help customers with product questions.",
    input_guardrails=[topic_check],
)
```

By default, input guardrails run **in parallel** with the agent (`run_in_parallel=True`). This keeps latency low — both the guardrail and agent start together. If the guardrail fires, the agent is cancelled. If you prefer to guarantee the agent never starts at all for flagged inputs, set `run_in_parallel=False` for a blocking mode that evaluates the guardrail first.

Input guardrails only apply to the **first** agent in a run. This is intentional: guardrails live on the agent definition, and they guard the entry point, not every internal handoff.

### Output Guardrails

Output guardrails run after the agent produces its final response. They receive the agent's structured output and can reject it by triggering their own tripwire, raising `OutputGuardrailTripwireTriggered`.

```python
from agents import output_guardrail, GuardrailFunctionOutput

@output_guardrail
async def no_math_output(ctx, agent, output) -> GuardrailFunctionOutput:
    contains_math = "=" in str(output)
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=contains_math)
```

Like input guardrails, output guardrails are attached to the agent and only run when that agent is the **last** agent in the run — the one producing the final answer.

### Tool Guardrails

The SDK also lets you guard individual tool calls. Tool guardrails sit between the model's decision to call a tool and the tool's actual execution.

- `ToolInputGuardrail` runs **before** the tool executes. You can inspect the arguments and choose to allow, reject with a message, or raise an exception entirely.
- `ToolOutputGuardrail` runs **after** the tool returns. You can inspect the result and sanitize or reject it.

```python
from agents import ToolGuardrailFunctionOutput, tool_input_guardrail, function_tool

@tool_input_guardrail
def block_secrets(data):
    if "sk-" in str(data.context.tool_arguments):
        return ToolGuardrailFunctionOutput.reject_content("Remove secrets before calling this tool.")
    return ToolGuardrailFunctionOutput.allow()

@function_tool(tool_input_guardrails=[block_secrets])
def classify_text(text: str) -> str:
    return f"length:{len(text)}"
```

The three possible behaviors — `allow`, `reject_content`, and `raise_exception` — give you precise control over what happens when a guardrail fires. `reject_content` is especially useful because it keeps the run going: the model gets an error message instead of the tool result and can try something else. `raise_exception` stops everything.

---

## Interruptions: Pausing for Human Approval

Sometimes a guardrail isn't what you need. Instead of blocking unconditionally, you want to pause, ask a human, and then continue. That's where the interruption mechanism comes in.

Any tool can declare that it `needs_approval`. When the model tries to call it, execution pauses and the result surfaces a list of `ToolApprovalItem` entries in `RunResult.interruptions` instead of calling the tool.

```python
from agents import Agent, Runner, function_tool

@function_tool(needs_approval=True)
async def cancel_order(order_id: int) -> str:
    return f"Cancelled order {order_id}"

agent = Agent(name="Support agent", tools=[cancel_order])
result = await Runner.run(agent, "Cancel order 42")

if result.interruptions:
    for item in result.interruptions:
        print(f"Waiting for approval: {item.name}({item.arguments})")
```

`needs_approval` can also be a callable that evaluates each invocation dynamically:

```python
async def requires_review(ctx, params, call_id) -> bool:
    return params.get("amount", 0) > 100

@function_tool(needs_approval=requires_review)
async def process_refund(order_id: int, amount: float) -> str:
    ...
```

This way, small refunds go through automatically while large ones pause for review.

---

## RunState: Serializing a Paused Run

When a run is interrupted, you need a way to record what happened, collect a human decision — possibly hours or days later — and then continue. `RunState` is the serializable snapshot of a paused run.

```python
state = result.to_state()
serialized = state.to_string()  # or state.to_json()

# Store to disk, database, or message queue...
# Later, in a different process:
state = await RunState.from_string(agent, serialized)

for interruption in result.interruptions:
    approved = ask_human(interruption.name, interruption.arguments)
    if approved:
        state.approve(interruption)
    else:
        state.reject(interruption)

result = await Runner.run(agent, state)
```

`RunState` captures everything needed to resume: model responses, generated items, approval decisions, conversation identifiers, and trace metadata. A resumed run picks up exactly where it left off, without re-running earlier turns or re-evaluating guardrails that already passed.

Approval decisions can also be scoped:

- `state.approve(interruption)` approves just this one call.
- `state.approve(interruption, always_approve=True)` caches the decision so future calls to the same tool are automatically approved without prompting.
- `state.reject(interruption, always_reject=True)` works the same way in the other direction.

`RunState` is versioned with a schema version (`CURRENT_SCHEMA_VERSION = "1.4"` at the time of writing). Older schema versions remain readable for backward compatibility, while unknown newer versions fail fast so you don't silently process malformed state.

---

## Long-Running Approvals

Some decisions don't happen in seconds. An approval might sit in a queue for hours while a compliance team reviews it. `RunState` is designed for this. Serialize it to a database, push it to a message queue, or store it in a file. When the decision comes back, deserialize and resume:

```python
# Serialize with custom context handling
serialized = state.to_json(context_serializer=my_serializer)

# Restore later
state = await RunState.from_json(
    agent,
    json_data,
    context_deserializer=my_deserializer,
)
```

If your approval flow involves multiple services — an HTTP webhook, an async database update, a Slack message — `RunState` gives you a durable contract: the paused work waits patiently, and the agent resumes cleanly when the human is ready.

---

## The Philosophy

Giving an agent access to tools that affect the real world — cancelling orders, sending emails, running shell commands — is a significant responsibility. The SDK doesn't pretend otherwise.

The layered approach here is intentional. Guardrails let you filter at the edges: bad inputs never reach the model, bad outputs never reach the user. Tool guardrails let you inspect at the call site: secrets don't leak, dangerous arguments don't execute. The interruption and `RunState` mechanism lets you pause in the middle of a run: consequential decisions wait for a human, and the agent resumes only after that decision is made.

None of this requires you to distrust your agent. Most of the time, agents work exactly as intended, and all these layers sit quietly in the background. But when something unexpected happens — an unusual input, a surprising tool call, a high-stakes action — you have the hooks to catch it, review it, and respond.

You built the agent. You stay in control.

---

# Tools & MCP

If the agent loop is the heartbeat of an AI system, tools are its hands. Without them, an agent can only think and respond; with them, it can search the web, read files, call APIs, run code, and invoke entire other agents. This chapter covers everything you need to equip your agents with the right tools — from dead-simple Python functions to distributed MCP servers.

---

## What Is a Tool?

In the SDK, a **tool** is anything the model can choose to call during a run. The framework handles the plumbing: when the model decides to invoke a tool, the SDK parses the arguments, calls your function (or connects to a server), and feeds the result back into the conversation. You just write the logic.

There are several categories of tools:

| Category | Examples |
|---|---|
| **Hosted tools** | `WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool` |
| **Function tools** | Any Python function, wrapped with `@function_tool` |
| **MCP servers** | Stdio, SSE, or HTTP servers exposing tool collections |
| **Agents as tools** | `agent.as_tool(...)` — one agent calling another |

This chapter focuses on function tools, MCP, and the agent-as-tool pattern. Hosted tools follow the same interface but run entirely within OpenAI's infrastructure.

---

## Function Tools: Wrapping Python Callables

The most common and flexible tool type is the **function tool**. Any Python function — sync or async — can become a tool with a single decorator.

```python
from agents import function_tool, Agent

@function_tool
async def fetch_weather(location: str) -> str:
    """Return the current weather for a location.

    Args:
        location: The city and country to look up.
    """
    # In production, call a real weather API
    return f"Sunny and 22°C in {location}"

agent = Agent(name="Weather Bot", tools=[fetch_weather])
```

The decorator automatically:
- Names the tool after the function (`fetch_weather`)
- Extracts the description from the docstring
- Builds a JSON schema from the type annotations
- Parses argument descriptions from the docstring

The SDK uses Python's `inspect` module for signature analysis, `griffe` for docstring parsing (supporting Google, NumPy, and Sphinx styles), and Pydantic for schema generation. You get sensible defaults with almost no configuration.

### Overriding Defaults

Need a different name or description? Pass arguments to the decorator:

```python
@function_tool(name_override="get_weather", use_docstring_info=False)
def fetch_weather(location: str) -> str:
    return f"Sunny in {location}"
```

### Accessing Run Context

If your tool needs access to the run context — for example, to read shared state or call a database — declare it as the first parameter with type `RunContextWrapper`:

```python
from agents import function_tool, RunContextWrapper
from typing import Any

@function_tool
def get_user_data(ctx: RunContextWrapper[Any], user_id: str) -> str:
    """Look up user data from the request context."""
    db = ctx.context["db"]
    return db.get(user_id)
```

The context parameter is never exposed in the tool schema — the model never sees it.

### Strict Mode and Schema Constraints

For models that support structured outputs, you can add Pydantic `Field` constraints directly to your function arguments:

```python
from pydantic import Field
from typing import Annotated
from agents import function_tool

@function_tool
def set_rating(score: Annotated[int, Field(ge=0, le=100, description="Score 0-100")]) -> str:
    return f"Rating set to {score}"
```

These constraints flow directly into the generated JSON schema and are validated before your function is called.

### The `FunctionTool` Class

When you need full control — custom schema, manual argument parsing, or dynamic tool definitions — use `FunctionTool` directly:

```python
from pydantic import BaseModel
from agents import FunctionTool, RunContextWrapper
from typing import Any

class Args(BaseModel):
    username: str
    age: int

async def run_function(ctx: RunContextWrapper[Any], args: str) -> str:
    parsed = Args.model_validate_json(args)
    return f"Hello, {parsed.username}! You are {parsed.age} years old."

tool = FunctionTool(
    name="greet_user",
    description="Greet a user by name and age",
    params_json_schema=Args.model_json_schema(),
    on_invoke_tool=run_function,
)
```

---

## Handling Tool Errors

Tools fail. Networks drop, APIs return errors, code throws exceptions. The SDK gives you clear, composable control over what happens next.

When a tool raises an exception, the **`failure_error_function`** determines what the model sees. By default, the SDK sends a generic error message so the model can recover gracefully. You can customize this:

```python
from agents import function_tool, RunContextWrapper
from typing import Any

def my_error_handler(ctx: RunContextWrapper[Any], error: Exception) -> str:
    print(f"Tool error: {error}")
    return "Something went wrong. Please try a different query."

@function_tool(failure_error_function=my_error_handler)
def risky_operation(query: str) -> str:
    """Perform a potentially unstable operation."""
    raise ValueError("Connection refused")
```

Three modes:
- **Default** — sends a generic error string to the model
- **Custom function** — your function formats the error message
- **`None`** — the exception propagates; you handle it yourself in your run loop

### Tool Timeouts

For async tools, you can set a per-call timeout so a slow external API can't stall an entire run:

```python
@function_tool(timeout=3.0, timeout_behavior="error_as_result")
async def slow_api_call(query: str) -> str:
    ...  # If this takes more than 3 seconds, the model gets an error message
```

Set `timeout_behavior="raise_exception"` if you prefer to hard-fail the run instead.

---

## MCP: The Model Context Protocol

Function tools are powerful, but they live inside your codebase. What if you want to reuse a tool server across multiple agents or teams? Or connect to an existing ecosystem of pre-built integrations?

That's where **MCP** comes in.

The Model Context Protocol is an open standard that defines how applications expose tools and context to language models. Think of it as a USB-C port for AI: a common connector that works across tools, agents, and infrastructure. Instead of writing custom integration code for every data source, you connect an MCP server and the agent discovers its tools automatically.

The SDK supports four MCP integration modes:

| Mode | When to Use |
|---|---|
| `HostedMCPTool` | OpenAI manages everything; zero infrastructure |
| `MCPServerStreamableHttp` | You control an HTTP server; modern transport |
| `MCPServerSse` | Legacy SSE transport; prefer Streamable HTTP for new work |
| `MCPServerStdio` | Local subprocess; great for CLI tools and quick experiments |

### Stdio: Launch a Local Process

The simplest MCP integration: the SDK spawns a subprocess, communicates over stdin/stdout, and cleans up when done.

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

async with MCPServerStdio(
    name="Filesystem Server",
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"],
    },
) as server:
    agent = Agent(
        name="File Assistant",
        instructions="Help users read and manage files.",
        mcp_servers=[server],
    )
    result = await Runner.run(agent, "List the files available.")
    print(result.final_output)
```

### Streamable HTTP: Connect to a Remote Server

For production deployments where tool servers are standalone services:

```python
from agents.mcp import MCPServerStreamableHttp

async with MCPServerStreamableHttp(
    name="My Tool Server",
    params={
        "url": "http://localhost:8000/mcp",
        "headers": {"Authorization": f"Bearer {token}"},
    },
    cache_tools_list=True,
    max_retry_attempts=3,
) as server:
    agent = Agent(name="Assistant", mcp_servers=[server])
    ...
```

### Hosted MCP: Zero-Infrastructure Tools

If you're using OpenAI's Responses API, `HostedMCPTool` lets OpenAI manage the MCP connection entirely:

```python
from agents import Agent, HostedMCPTool

agent = Agent(
    name="Research Assistant",
    tools=[
        HostedMCPTool(
            tool_config={
                "type": "mcp",
                "server_label": "gitmcp",
                "server_url": "https://gitmcp.io/openai/codex",
                "require_approval": "never",
            }
        )
    ],
)
```

### Tool Filtering

MCP servers sometimes expose dozens of tools. You can restrict which ones the agent sees:

```python
from agents.mcp import MCPServerStdio, create_static_tool_filter

server = MCPServerStdio(
    params={"command": "npx", "args": ["...", "/data"]},
    tool_filter=create_static_tool_filter(allowed_tool_names=["read_file", "list_files"]),
)
```

For dynamic logic, pass a callable that receives a `ToolFilterContext` with the current agent, run context, and server name.

---

## Agents as Tools

One of the SDK's most expressive patterns is treating an entire agent as a tool. Instead of handing off control (which ends the current agent's turn), you call a specialized sub-agent and get its result back — all without leaving the orchestrating agent's context.

```python
from agents import Agent

translator = Agent(
    name="Translator",
    instructions="You translate text into the requested language.",
)

orchestrator = Agent(
    name="Orchestrator",
    instructions="Use the translation tool when users need translations.",
    tools=[
        translator.as_tool(
            tool_name="translate_text",
            tool_description="Translate text into a target language.",
        )
    ],
)
```

The `.as_tool()` method converts the agent into a `FunctionTool`. When the orchestrator calls it, the SDK runs the translator as a nested `Runner.run()` — fully isolated, with its own turn loop — and returns the final output as the tool result.

---

## Putting It All Together

Tools transform agents from chatbots into systems that get things done. Function tools handle local logic and API calls with minimal boilerplate. MCP servers bring a growing ecosystem of pre-built integrations — filesystems, databases, external services — under a common protocol. And the agent-as-tool pattern lets you compose complex multi-agent workflows without giving up the clarity of a single orchestrating loop.

The best agents are well-equipped ones. Give them the right tools, handle failures gracefully, and the loop does the rest.

---

# Tracing & Observability

The question is not whether your agent will do something surprising. It will. The question is whether you will be able to see what happened.

Observability is how you answer that question. The OpenAI Agents SDK builds tracing in from the start — not as an afterthought bolted on after launch, but as a first-class feature that activates the moment you call `Runner.run()`. You get structured, hierarchical records of every decision your agent made, every tool it called, and every token it spent. Then you can look at those records, understand what went wrong (or right), and improve.

## The Tracing System at a Glance

Tracing is **on by default**. Every call to `Runner.run()`, `Runner.run_sync()`, or `Runner.run_streamed()` is automatically wrapped in a trace and sent to OpenAI's [Traces dashboard](https://platform.openai.com/traces). You don't have to do anything to get started.

If you need to turn it off — for example, if your organization operates under a Zero Data Retention policy — set the environment variable `OPENAI_AGENTS_DISABLE_TRACING=1`, or pass `tracing_disabled=True` in a `RunConfig`:

```python
from agents import Runner, RunConfig

result = await Runner.run(agent, "Hello", run_config=RunConfig(tracing_disabled=True))
```

Under the hood, the SDK initializes a global `TraceProvider` (lazily, on first use) that routes trace data through a `BatchTraceProcessor` and a `BackendSpanExporter`. You rarely need to touch these directly, but they're there when you do.

## Traces and Spans

The data model has two layers:

- A **Trace** represents an entire end-to-end workflow — one call to `Runner.run()`, or a group of calls you manually wrap together. A trace has a `workflow_name`, a unique `trace_id`, and optional `group_id` metadata (useful for linking multiple traces from the same conversation thread).

- **Spans** are the operations within a trace. Each span has a start and end time, a reference to its parent span, and a typed `span_data` payload that describes what happened.

The SDK automatically creates spans for everything meaningful:

| Span type | What it captures |
|-----------|-----------------|
| `agent_span` | Agent name, tools available, handoffs, output type |
| `generation_span` | LLM inputs, outputs, model name, config, token usage |
| `function_span` | Tool name, input arguments, return value |
| `guardrail_span` | Guardrail name, whether it triggered |
| `handoff_span` | Source agent, destination agent |
| `response_span` | The raw API response ID |
| `transcription_span` / `speech_span` | Audio I/O data for voice pipelines |

Spans nest naturally. An agent span contains the generation spans it caused, which in turn contain function spans for any tools called. This hierarchy lets you trace through multi-agent workflows and see exactly which agent made which decision.

### Sensitive Data

`generation_span` and `function_span` include the actual content of LLM inputs/outputs and tool arguments by default. If that's too much exposure, set `trace_include_sensitive_data=False` in your `RunConfig` or export `OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA=false` before running.

## Grouping Multiple Runs into One Trace

By default, each `Runner.run()` creates its own trace. If you're running several agents as part of a single workflow, wrap them in a `trace()` context manager:

```python
from agents import Agent, Runner, trace

async def run_joke_workflow():
    agent = Agent(name="Joke generator", instructions="Tell funny jokes.")

    with trace("Joke workflow"):
        joke = await Runner.run(agent, "Tell me a joke")
        rating = await Runner.run(agent, f"Rate this joke: {joke.final_output}")

    print(joke.final_output)
    print(rating.final_output)
```

The two runs now appear as a single trace in the dashboard, with their spans nested under one roof.

## Custom Trace Processors

The default setup sends traces to OpenAI's backend. If you want to send them somewhere else — or somewhere else _in addition_ — you have two paths:

**Add a processor** (keeps the default OpenAI export, adds yours):

```python
from agents.tracing import add_trace_processor

add_trace_processor(my_processor)
```

**Replace all processors** (you're now responsible for everything):

```python
from agents.tracing import set_trace_processors

set_trace_processors([my_processor])
```

A processor implements the `TracingProcessor` interface:

```python
from agents.tracing import TracingProcessor

class MyProcessor(TracingProcessor):
    def on_trace_start(self, trace): ...
    def on_trace_end(self, trace): ...
    def on_span_start(self, span): ...
    def on_span_end(self, span): ...
    def shutdown(self): ...
    def force_flush(self): ...
```

Each method is called synchronously, so keep them fast. Errors are caught and logged — a misbehaving processor won't crash your agent.

The ecosystem has you covered if you'd rather not write your own: Weights & Biases, Langfuse, Braintrust, Pydantic Logfire, AgentOps, and a dozen more observability platforms all support the SDK's tracing surface through ready-made integrations.

## Token Usage Tracking

Alongside traces, the SDK maintains a running tally of every token spent during a run. This lives in `result.context_wrapper.usage` after the run completes.

```python
result = await Runner.run(agent, "Summarize this document...")
usage = result.context_wrapper.usage

print(f"Requests made: {usage.requests}")
print(f"Input tokens:  {usage.input_tokens}")
print(f"Output tokens: {usage.output_tokens}")
print(f"Total tokens:  {usage.total_tokens}")
print(f"Cached tokens: {usage.input_tokens_details.cached_tokens}")
print(f"Reasoning:     {usage.output_tokens_details.reasoning_tokens}")
```

Usage is aggregated across _all_ model calls in the run — including tool calls and agent handoffs.

### Per-Request Breakdown with `RequestUsage`

The aggregate numbers are convenient, but `request_usage_entries` gives you the fine-grained breakdown. Each entry is a `RequestUsage` object tracking one API call:

```python
for i, req in enumerate(usage.request_usage_entries):
    print(
        f"Call {i+1}: {req.input_tokens} in / {req.output_tokens} out"
        f" | model={req.model_name}"
        f" | agent={req.agent_name}"
        f" | response_id={req.response_id}"
    )
```

The three fields `model_name`, `agent_name`, and `response_id` were added recently and make per-call attribution practical. In a multi-agent run where several models may be in play, `agent_name` tells you which agent made the call, `model_name` tells you which model answered it, and `response_id` links back to the raw API response for auditing or replay.

## What Good Observability Looks Like in Practice

A well-instrumented agent application does a few things consistently:

1. **Names its workflows** — pass a `workflow_name` in `RunConfig` or use a named `trace()` context. Anonymous traces are hard to filter later.
2. **Groups related runs** — use a `group_id` (a session or conversation ID works well) to stitch together multi-turn interactions.
3. **Tracks usage per run** — log `total_tokens` and `requests` after every `Runner.run()`. Small numbers add up quickly in production.
4. **Checks `request_usage_entries` when costs spike** — the per-call breakdown will show you which agent and model is responsible.

Tracing isn't just a debugging tool. It's a production feedback loop: the records you collect during development become the ground truth you need to optimize, audit, and explain your system's behavior over time.

---

# Developer Guide

Welcome. If you are reading this, you are thinking about contributing to the OpenAI Agents Python SDK — and that is worth celebrating. Whether you are fixing a typo, squashing a bug, or building an entirely new capability, you are helping shape the tools that developers around the world use to build intelligent systems. This chapter will help you get productive quickly and feel at home in the codebase.

---

## Prerequisites

Before anything else, make sure your machine has:

- **Python 3.10 or newer.** The SDK supports 3.10 through 3.14, but your development environment should use at least 3.10.
- **[`uv`](https://github.com/astral-sh/uv)**, the fast Python package manager this project uses for dependency management and running commands. If you do not have it, install it before proceeding.
- **`make`**, for running the repository's task shortcuts. It comes pre-installed on macOS and most Linux distributions.

---

## Setting Up Your Development Environment

Clone the repository, then install all dependencies including development tools:

```bash
make sync
```

This runs `uv sync --all-extras --all-packages --group dev` under the hood, pulling in everything you need: pytest, mypy, ruff, mkdocs, coverage, inline-snapshot, and more. Run this again any time dependencies change or you pull new commits that update `pyproject.toml` or `uv.lock`.

For your day-to-day Python commands, always use `uv run` rather than calling `python` directly:

```bash
uv run python my_script.py
```

This ensures you are always running inside the managed environment.

---

## Running Tests

The test suite is your first line of defense. To run everything:

```bash
make tests
```

This runs two passes: a parallelized pass (using `pytest-xdist`) for the bulk of the suite, followed by a serial pass for tests marked `@pytest.mark.serial`. Both must pass.

To run a focused subset of tests during development — much faster when you are iterating:

```bash
uv run pytest -s -k <pattern>
```

Replace `<pattern>` with any substring of a test name, file, or class. For example, `-k test_guardrail` will only run tests whose names contain `test_guardrail`.

### Snapshot Tests

Some tests use [inline-snapshot](https://15r10nk.github.io/inline-snapshot/latest/) to assert against recorded output. If your change alters behavior covered by a snapshot, the test will fail with a diff showing what changed.

To update existing snapshots to match your new output:

```bash
make snapshots-fix
```

To generate snapshots for brand-new snapshot tests:

```bash
make snapshots-create
```

After either operation, run `make tests` again to confirm everything passes cleanly.

---

## Type Checking

The codebase uses strict mypy type checking. Run it with:

```bash
make mypy
```

All new code you write must pass. If you see a type error in code you did not touch, it may be a pre-existing issue — but try not to make it worse. When in doubt, prefer explicit type annotations over `Any`.

---

## Linting and Formatting

This project uses [`ruff`](https://github.com/astral-sh/ruff) for both formatting and linting. It is fast and opinionated, which means less time debating style and more time writing code.

To auto-format your code and apply lint fixes:

```bash
make format
```

To check for lint issues without making changes:

```bash
make lint
```

A few things to keep in mind:

- The line length limit is 100 characters.
- Imports are sorted and managed by ruff — do not organize them by hand.
- Write comments as complete sentences ending with a period.

---

## The Full Verification Workflow

Before you mark your work done, run the complete verification sequence in order:

```bash
make format
make lint
make mypy
make tests
```

This is the same sequence CI runs. If all four pass locally, you can submit with confidence. Running them in this order matters: formatting first, then linting (which may catch issues formatting cannot fix), then type checking, then the full test suite.

---

## Code Coverage

To check coverage and ensure you have not let it fall below the 85% threshold:

```bash
make coverage
```

This runs the full test suite under `coverage`, generates a report, and fails if coverage drops below the minimum. If your new code is not covered, add tests.

---

## Documentation

Source documentation lives in `docs/`. The site is built with MkDocs.

> **Note:** Do not edit the translated docs under `docs/ja`, `docs/ko`, or `docs/zh` — they are generated automatically.

To build the docs after making changes:

```bash
make build-docs
```

To preview the docs locally in your browser (with live reload):

```bash
make serve-docs
```

If you are working on translations or need to do a full build including translated pages:

```bash
make build-full-docs
```

Rebuild the docs any time you make user-facing changes to behavior, APIs, or examples.

---

## Pull Request and Commit Guidelines

### Commits

Keep commits small and focused. A good commit does one logical thing and has a message that says what that thing is — written in the **imperative mood**:

```
Add retry logic to the tool execution loop
Fix type error in run_state deserialization
Update snapshots for guardrail output format change
```

Not:
```
Fixed stuff
Changes
Updated code
```

Small, focused commits make reviews faster and make the history easier to understand later.

### Opening a Pull Request

Use the PR template at `.github/PULL_REQUEST_TEMPLATE/pull_request_template.md`. It asks for:

- **Summary:** What does this change do and what problem does it solve?
- **Test plan:** How did you verify this works?
- **Issue number:** If this closes an open issue, note it (e.g., `Closes #1234`).
- **Checklist:** New tests added, docs updated, `make lint` and `make format` run, tests passing.

Fill this out thoughtfully. A clear PR description is the single biggest thing you can do to speed up review.

---

## What Reviewers Look For

When your PR comes up for review, the team will be checking for:

- All automated checks pass (`make format`, `make lint`, `make mypy`, `make tests`).
- New behavior is covered by tests, including edge cases.
- Code is readable and consistent with the surrounding style.
- Public APIs and user-facing behavior are documented.
- Examples are updated if behavior changes.
- A clean commit history and a clear PR description.

Reviewers are not looking for perfection — they are looking for care and thoroughness. If you are uncertain about an approach, say so in the PR description. A question is better than a guess.

---

## A Note to First-Time Contributors

Every expert in this codebase was once a newcomer. If something is confusing, that is a documentation bug, not a reflection on you. If you are stuck, open an issue or ask in the PR — the community is here to help.

The best first contribution is often a small one: a clearer error message, a missing test case, a documentation clarification. These matter. Start there, learn the workflow, and go from there.

Thank you for being here.

---

# User Guide

> This chapter is about getting things done. We'll move from zero to a working agent in minutes, then systematically cover the features you'll use every day: configuration, multi-agent coordination, structured output, sessions, and streaming.

---

## Installation

Install the core SDK with pip:

```bash
pip install openai-agents
```

The base install covers everything you need for text-based agents. Optional extras unlock additional capabilities:

| Extra | What it adds | Command |
|---|---|---|
| `litellm` | Route to 100+ model providers (Anthropic, Gemini, Mistral, ...) | `pip install "openai-agents[litellm]"` |
| `voice` | Voice pipeline components (numpy + websockets) | `pip install "openai-agents[voice]"` |
| `realtime` | Realtime API (voice WebSocket) | `pip install "openai-agents[realtime]"` |
| `sqlalchemy` | SQLAlchemy-backed session memory | `pip install "openai-agents[sqlalchemy]"` |
| `redis` | Redis-backed session memory | `pip install "openai-agents[redis]"` |
| `encrypt` | Encrypted session wrapper | `pip install "openai-agents[encrypt]"` |

Set your API key once (the SDK reads it automatically):

```bash
export OPENAI_API_KEY=sk-...
```

---

## Quickstart: Your First Agent

Three concepts to know before you type anything:

- **Agent** — an LLM configured with a name, instructions, and optional tools.
- **Runner** — executes an agent and returns a `RunResult`.
- **`result.final_output`** — the string (or structured object) the agent produced.

```python
import asyncio
from agents import Agent, Runner

agent = Agent(
    name="History Tutor",
    instructions="You answer history questions clearly and concisely.",
)

async def main():
    result = await Runner.run(agent, "When did the Roman Empire fall?")
    print(result.final_output)

asyncio.run(main())
```

That's the whole thing. The SDK handles prompt formatting, API calls, retries, and tracing in the background.

### Adding a Tool

Decorate any function with `@function_tool` and pass it to the agent. The SDK automatically generates the JSON schema from type hints and docstrings.

```python
from agents import Agent, Runner, function_tool

@function_tool
def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """Return the current exchange rate between two currencies."""
    # call your real API here
    return 1.08

agent = Agent(
    name="Finance Helper",
    instructions="Help with currency questions. Use get_exchange_rate when needed.",
    tools=[get_exchange_rate],
)
```

### Synchronous Code

If you're not in an async context, `Runner.run_sync` is available:

```python
result = Runner.run_sync(agent, "What's the capital of Japan?")
print(result.final_output)
```

---

## Configuration: RunConfig and ModelSettings

Most agent behavior is controlled through two dataclasses.

### `ModelSettings`

Attached directly to an agent (or provided globally via `RunConfig`). Controls the model call itself:

```python
from agents import Agent, ModelSettings

agent = Agent(
    name="Creative Writer",
    instructions="Write vivid prose.",
    model="gpt-4o",
    model_settings=ModelSettings(
        temperature=0.9,
        max_tokens=1000,
        tool_choice="auto",
    ),
)
```

### `RunConfig`

Passed to `Runner.run(...)` to control a single execution. Think of it as the supervisor's memo for one run:

```python
from agents import Runner, RunConfig

result = await Runner.run(
    agent,
    "Summarize this document.",
    run_config=RunConfig(
        model="gpt-4o-mini",
        max_turns=5,
        workflow_name="Doc Summary",
        tracing_disabled=False,
    ),
)
```

Key `RunConfig` fields:

| Field | Purpose |
|---|---|
| `model` | Override the model for every agent in this run |
| `model_settings` | Override model tuning params globally |
| `max_turns` | Hard limit on agentic loop iterations (default: 10) |
| `workflow_name` | Label shown in the trace viewer |
| `input_guardrails` / `output_guardrails` | Attach safety checks to the run |
| `session` | Wire in persistent conversation memory |
| `trace_include_sensitive_data` | Exclude inputs/outputs from trace payloads |

---

## Multi-Agent Patterns: Handoffs

Handoffs let one agent delegate control to another. The new agent receives the conversation history and takes over — the original agent is done.

```python
from agents import Agent, Runner

billing_agent = Agent(
    name="Billing Agent",
    handoff_description="Handles billing, invoices, and payment questions.",
    instructions="You are a billing specialist. Answer billing questions concisely.",
)

support_agent = Agent(
    name="Support Agent",
    handoff_description="Handles general technical support.",
    instructions="You are a support specialist. Diagnose and resolve issues.",
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="Route each customer question to the right specialist.",
    handoffs=[billing_agent, support_agent],
)

result = await Runner.run(triage_agent, "I was charged twice for my subscription.")
print(f"Answered by: {result.last_agent.name}")
print(result.final_output)
```

**Rule of thumb:** Use handoffs when the specialist should own the conversation. Use agents-as-tools when a coordinator should own the conversation and call specialists for subtasks.

---

## Structured Output

By default, agents return plain text. Pass `output_type` to receive a validated, typed object instead:

```python
from pydantic import BaseModel
from agents import Agent, Runner

class TravelPlan(BaseModel):
    destination: str
    duration_days: int
    highlights: list[str]
    estimated_budget_usd: float

agent = Agent(
    name="Travel Planner",
    instructions="Create detailed travel plans.",
    output_type=TravelPlan,
)

result = await Runner.run(agent, "Plan a 5-day trip to Japan on a $3000 budget.")
plan = result.final_output  # type: TravelPlan
print(f"Destination: {plan.destination}")
print(f"Budget: ${plan.estimated_budget_usd}")
```

---

## Sessions and Memory

Without a session, every `Runner.run` call starts a blank conversation. To maintain context across multiple turns, attach a session.

```python
from agents import Agent, Runner, SQLiteSession

agent = Agent(name="Assistant", instructions="Reply concisely.")
session = SQLiteSession("user_42")

# Turn 1
result = await Runner.run(agent, "My favorite color is blue.", session=session)

# Turn 2 — agent remembers turn 1
result = await Runner.run(agent, "What's my favorite color?", session=session)
print(result.final_output)  # "Blue."
```

### Choosing a Backend

| Session | Best for |
|---|---|
| `SQLiteSession("id")` | Development, simple single-process apps |
| `SQLiteSession("id", "path/to/db")` | Persistent file-backed storage |
| `RedisSession.from_url("id", url=...)` | Distributed apps, multiple workers |
| `SQLAlchemySession.from_url("id", url=...)` | Production, existing SQL database |

---

## Streaming

Streaming lets you display output token-by-token as it arrives:

```python
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent

agent = Agent(name="Storyteller", instructions="Tell vivid short stories.")

result = Runner.run_streamed(agent, "Tell me a story about a lighthouse keeper.")
async for event in result.stream_events():
    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
        print(event.data.delta, end="", flush=True)
print()
```

---

## Common Patterns

A tour through `examples/` reveals idioms that come up constantly:

- **Deterministic pipelines**: chain agents sequentially, passing output from one to the next.
- **Routing/triage**: one lightweight agent reads intent and hands off to a specialist.
- **LLM-as-a-judge**: run a task agent, then a separate evaluator that scores the output.
- **Guardrails**: attach `InputGuardrail` and `OutputGuardrail` to screen content in parallel.
- **Human in the loop**: tools can require approval before executing; the runner pauses and you resume after collecting a decision.

---

## What's Next

- **Tools** — function tools, hosted tools, MCP servers, and agents-as-tools.
- **Handoffs** — input filters, handoff callbacks, nested history.
- **Tracing** — the trace viewer, custom exporters, and sensitive-data controls.
- **Sessions** — advanced backends, encryption, compaction, and branching.
