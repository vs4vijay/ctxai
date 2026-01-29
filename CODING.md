# ctxai AI Coding Agent - Architecture & Implementation Plan

> **Goal**: Transform ctxai into a comprehensive AI coding agent similar to Claude Code, GitHub Copilot CLI, Codex CLI, Amp, and Goose.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Research Findings](#research-findings)
3. [Current State Analysis](#current-state-analysis)
4. [Architectural Design](#architectural-design)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Technical Specifications](#technical-specifications)
7. [References](#references)

---

## Executive Summary

Based on deep research into modern AI coding agents in 2026, ctxai will be enhanced from a semantic code search tool into a full-featured autonomous coding agent. The agent will follow industry-standard agentic workflow patterns while maintaining ctxai's unique strength in semantic code understanding.

**Key Transformations:**
- **From**: Semantic code search engine with basic agent capabilities
- **To**: Full autonomous coding agent with multi-agent orchestration, specialized tools, and agentic workflows

**Core Capabilities:**
- Autonomous code generation and modification
- Multi-agent orchestration (orchestrator, explorer, coder)
- Advanced tool system (file ops, bash, git, web, testing)
- Reflection and planning patterns
- Model Context Protocol (MCP) integration
- Interactive CLI with conversation management
- Session persistence and memory

---

## Research Findings

### 1. Industry Analysis: Leading AI Coding Agents

#### Goose (Block - 2025)

**Architecture**: Built in **Rust** (59.4% of codebase) with TypeScript UI components, deployed as both desktop app and CLI.

**Core Technical Details**:
- **Modular Crate System**: Organized into specialized crates (goose-core, goose-cli, goose-server, goose-mcp)
- **Agent Entry Point**: Primary agent logic in `crates/goose/src/agents/agent.rs`
- **Provider Trait Pattern**: Extensibility through trait implementation (`providers/base.rs`)
- **MCP Integration**: Dedicated crate for Model Context Protocol extensions
- **Local Execution**: Runs entirely on-machine for privacy and control
- **Multi-LLM Support**: Abstraction layer decouples model providers from core logic

**Key Features**:
- Autonomous task completion: code writing, execution, debugging, workflow orchestration
- Docker support for containerized deployment
- Cross-platform build infrastructure (Cross.toml)
- Extensive testing framework (goose-self-test.yaml)

**Sources**: [GitHub - block/goose](https://github.com/block/goose), [Goose AGENTS.md](https://github.com/block/goose/blob/main/AGENTS.md), [Block Open Source](https://block.xyz/inside/block-open-source-introduces-codename-goose)

---

#### Amp (Sourcegraph)

**The Simplicity Thesis**: Amp demonstrates that sophisticated coding agents require surprisingly minimal infrastructure: **"It's an LLM, a loop, and enough tokens."**

**Core Loop Implementation** (~300-400 lines of code):
```
1. Maintain conversation history slice
2. Send entire history to Claude (stateless server)
3. Check response for tool invocations
4. Execute matching tools locally
5. Return results to Claude
6. Repeat until no tool calls requested
```

**Tool Definition Structure**:
- **Name**: Identifier for the tool
- **Description**: Human-readable text that guides model behavior
- **JSON Schema**: Input validation structure
- **Executable Function**: Actual operation handler

**Three Core Tools**:
- `read_file`: Access file contents
- `list_files`: Enumerate directory structures
- `edit_file`: Modify files via string replacement

**Critical Design Insight**: The model autonomously chains multiple tools to solve complex problems without explicit instruction—emergent multi-step reasoning from simple tool interfaces.

**Team Features**: Threads, context, and workflows shared by default. Track adoption and improve together.

**Sources**: [Amp - How to Build an Agent](https://ampcode.com/how-to-build-an-agent), [Amp by Sourcegraph](https://sourcegraph.com/amp), [Staying in the Loop with Amp](https://medium.com/@jonathanaraney/staying-in-the-loop-how-amp-keeps-programmers-engaged-in-the-sdlc-with-agentic-ai-fe1a9d49eedc)

---

#### Cursor

**Composer Model** (Cursor 2.0): A mixture-of-experts model trained through **reinforcement learning inside real codebases**.

**Training Approach**:
- Placed model inside actual development environments
- Learned to use real development tools: semantic search, file editors, terminal commands
- Picked up practical behaviors: running tests, fixing linter errors, navigating large projects
- **Performance**: Most tasks completed in <30 seconds at ~250 tokens/second (4x faster than comparable models)

**Agent Architecture**:
- **Instructions**: System prompts and contextual rules
- **Tools**: Comprehensive toolkit with no limit on tool calls per task
- **User Messages**: Directs agent work

**Tool Capabilities**:
- Semantic search within indexed codebases
- File operations (read, search, edit)
- Terminal execution with output monitoring
- Web integration (search, information retrieval)
- Browser control (screenshots, navigation, visual verification)
- Image generation (UI mockups, diagrams)
- Rules retrieval (context-specific guidelines)
- Clarification queries (ask users during tasks)

**Operational Features**:
- **Checkpoints**: Automatic snapshots enable reverting changes (locally stored, separate from VCS)
- **Message Management**: Auto-summarization as conversations extend
- **Queued Processing**: Queue follow-up messages while agent works
- **Codebase Embedding**: Deep understanding and recall via custom embedding model

**Sources**: [Cursor Features](https://cursor.com/features), [Cursor Docs - Agent](https://cursor.com/docs/agent/overview), [Cursor 2.0 Explained](https://www.codecademy.com/article/cursor-2-0-new-ai-model-explained)

---

#### Aider (Paul Gauthier)

**Architect/Editor Pattern**: Separation of concerns between planning and execution.

**Two-Model System**:
- **Architect Model**: Focuses on problem-solving and solution design (e.g., o3-high, o1-preview)
- **Editor Model**: Translates architect's solution into specific code edits (e.g., GPT-4.1, DeepSeek, o1-mini)

**Performance**:
- Achieved **85% on polyglot coding benchmark** (current SOTA)
- o3-high + gpt-4.1: **78.2%** with **100% correctly formatted** edits
- Substantially reduced costs compared to using o3-high alone

**Repository Mapping (Repomap)**:
- Creates "concise map of whole git repository"
- Catalogs important classes, functions, and type signatures
- Uses **graph-ranking algorithm** for optimization:
  - Creates dependency graph (files as nodes, dependencies as edges)
  - Identifies most referenced/important identifiers
  - Dynamically adjusts map size based on token budget (~1,000 tokens default)
  - Expands when comprehensive repo understanding needed

**Context Provision**:
- Sends repository map to LLM alongside requests
- Presents "key symbols defined in each file" + critical code lines
- Avoids overwhelming context window while maintaining architectural awareness

**Git Integration**:
- Automatic commits after each AI change with descriptive messages
- Full version control integration for easy review/rollback
- Developers use familiar Git tools to manage AI suggestions

**Agentic Loop**:
```
1. Accept user requests and code context
2. LLM generates edit proposals
3. Apply changes with automatic linting/testing feedback
4. Refine implementations based on test results/lint errors
5. Iterate until success
```

**Model Support**: Optimized for Claude 3.7 Sonnet, DeepSeek R1/Chat V3, OpenAI o1/o3-mini, GPT-4o. Compatible with nearly any LLM including local models.

**Sources**: [Aider GitHub](https://github.com/paul-gauthier/aider), [Aider Repomap](https://aider.chat/docs/repomap.html), [Aider Leaderboards](https://aider.chat/docs/leaderboards/), [Architect/Editor Approach](https://generaitelabs.com/aider-implements-new-architect-editor-approach-for-ai-assisted-coding/)

---

#### Continue.dev

**Three-Tier Mode Structure**:
1. **Chat Mode**: Pure conversation without tool access
2. **Plan Mode**: Read-only tools for safe exploration and planning
3. **Agent Mode**: Full tool access for implementing features, fixing bugs, running tests

**Agent Architecture**:
- **Cloud Agents (Headless Mode)**: Async agents running in cloud, triggered by events (PR opens, schedules, webhooks)
- **CLI Agents (TUI Mode)**: Terminal-based synchronous agents with step-by-step user approval
- **IDE Agents**: Integrated into VS Code and JetBrains

**Implementation Layers** (TypeScript 84.1%):
- **Core**: Foundational agent logic and execution engine
- **CLI**: Both headless and terminal UI modes
- **Extensions**: IDE integrations
- **GUI**: Web-based interfaces including Mission Control dashboard
- **Packages**: Modular components (tool system, context management)

**Tool System (MCP Tools)**:
- **Permission Framework**: User approval before tool execution by default
- **Response Handling**: "Any data returned from a tool call is automatically fed back into the model as a context item"
- **Error Recovery**: Most errors caught and returned for agent decision-making
- **Context Integration**: Compatible with `@` context providers and highlighted code

**Contractor Execution Model**: AI autonomously determines which tools to deploy based on natural language instructions. Example: "Set the @typescript-eslint/naming-convention rule to 'off' for all eslint configurations" — agent independently selects tools.

**Context Store Innovation**: Shared knowledge base enabling compound intelligence through knowledge accumulation and reuse during task execution.

**Multi-Provider Support**: Claude, GPT, Gemini, Qwen, and local models.

**Sources**: [Continue GitHub](https://github.com/continuedev/continue), [Continue Agent Docs](https://docs.continue.dev/ide-extensions/agent/quick-start), [Continue Analysis](https://atoms.dev/insights/continuedev-an-in-depth-analysis-of-an-open-source-ai-powered-coding-assistant-for-enhanced-developer-workflows/6de278ae9d7e4858beaa8e53780b2773)

---

#### Codeium/Windsurf

**Cascade Agent**: Collaborative agent flow where AI works within IDE in real-time, explaining steps as it goes.

**Departure from Traditional Approach**: Instead of users defining a task and waiting passively for results, Windsurf enables active collaboration with the agent during execution.

**Agent Capabilities**:
- **Planning**: Can plan multi-step solutions
- **File Retrieval**: Retrieves relevant files across codebase
- **Multi-File Context**: Operates with holistic context spanning multiple files

**Assistance Modalities**:
- **Autocomplete**: In-line suggestions
- **Chat**: Ask questions about codebase
- **Command**: Turn natural language into code edits
- **Supercomplete & Cascade**: Context-rich assistance across multiple files

**Context Architecture**:
- Uses current file, local program libraries, public/private repos
- Learns programming standards and styles of teams/organizations
- Privacy: Processes context locally or on-premise for enterprise

**Sources**: [Windsurf](https://windsurf.com/), [Codeium Review](https://skywork.ai/blog/codeium-definition-ai-code-assistant/), [Scaling with Llama](https://ai.meta.com/blog/codeium-ai-coding-assistant-llama/)

---

### 2. AI Coding Agent Architectures (2026)

#### Multi-Agent Systems
Modern coding agents use **specialized sub-agents** rather than monolithic agents:
- **Orchestrator Agent**: Strategy, task classification, routing
- **Explorer Agent**: Investigation, codebase analysis, context gathering
- **Coder Agent**: Implementation, code generation, modification
- **Reviewer Agent**: Code review, quality gates, testing

**Source**: [Building AI Agents in 2026](https://levelup.gitconnected.com/the-2026-roadmap-to-ai-agent-mastery-5e43756c0f26)

#### Agentic Workflow Patterns

**1. Reflection Pattern**
- Self-evaluation before finalizing responses
- Feedback loops through code execution in sandbox
- Continuous self-improvement
- **Use Case**: Code review, bug detection, optimization

**2. Planning Pattern**
- Decomposing goals into sequential steps
- Agentic workflows for complex tasks
- Task classification and routing
- **Use Case**: Feature implementation, refactoring, architecture changes

**3. Tool Use Pattern**
- External tool integration (APIs, databases, file systems)
- Model Context Protocol (MCP) for standardized tool access
- Function calling with LLMs
- **Use Case**: File operations, bash execution, web search, testing

**4. ReAct Pattern (Reasoning + Acting)**
- Cyclical process: Reason → Act → Observe → Reflect
- Combines analysis with external tool use
- Error recovery and retry logic
- **Use Case**: Debugging, exploration, problem-solving

**Sources**:
- [Top AI Agentic Workflow Patterns](https://medium.com/@Deep-concept/top-ai-agentic-workflow-patterns-that-will-lead-in-2026-0e4755fdc6f6)
- [4 Agentic AI Design Patterns](https://research.aimultiple.com/agentic-ai-design-patterns/)

### 2. Key Insights from Industry Leaders

After analyzing Goose, Amp, Cursor, Aider, Continue, and Codeium/Windsurf, several critical patterns emerge:

#### The Simplicity Principle (Amp)
**"It's an LLM, a loop, and enough tokens."** The core agentic loop can be implemented in ~300-400 lines:
```
while not done:
    response = llm.chat(conversation_history, tools)
    if response.has_tool_calls:
        results = execute_tools(response.tool_calls)
        conversation_history.append(results)
    else:
        done = True
        return response.content
```

This demonstrates that **sophisticated behavior emerges from simple patterns**—not from complex architectures.

#### The Specialization Strategy (Aider)
**Architect/Editor Pattern** significantly improves both quality and cost:
- **Architect Model** (expensive, reasoning-focused): Plans and designs solutions
- **Editor Model** (cheaper, fast): Implements the architect's instructions
- **Result**: 78.2% accuracy + 100% correctly formatted edits + substantial cost reduction

This pattern is more effective than using a single model for both tasks.

#### The Context Challenge (Aider + Cursor)
All agents struggle with providing the right context to LLMs. Solutions:
- **Repository Mapping**: Graph-ranking algorithms to identify relevant code (~1,000 tokens)
- **Codebase Embedding**: Custom embeddings for semantic search and understanding
- **Adaptive Context**: Dynamically expand/contract based on task needs

#### The Tool Philosophy (Cursor vs Amp)
Two approaches to tools:
1. **Comprehensive Toolkit** (Cursor): "No limit on tool calls," extensive capabilities (semantic search, terminal, web, browser, image generation)
2. **Minimal Toolkit** (Amp): Just 3 core tools (read, list, edit) that chain together

Both work—extensive tools enable more capabilities, minimal tools keep the system simple and maintainable.

#### The Deployment Model (Continue + Goose)
Modern agents support multiple deployment modes:
- **Cloud Agents**: Async execution triggered by events (PR opens, schedules)
- **CLI Agents**: Terminal-based with step-by-step approval
- **IDE Agents**: Real-time collaboration within editor
- **Desktop Apps**: Electron-based standalone applications

This flexibility enables adoption across different workflows and preferences.

#### The Privacy-Performance Tradeoff (Goose + Codeium)
- **Local Execution** (Goose): Runs entirely on-machine, full privacy, resource intensive
- **Cloud Processing** (most others): Better performance, requires sending code to cloud
- **Hybrid** (Codeium): Local processing for enterprise, cloud for individuals

Privacy concerns drive architecture decisions.

#### The Training Innovation (Cursor)
Cursor 2.0's Composer was **trained inside real codebases** using reinforcement learning:
- Learned to use actual development tools
- Picked up practical behaviors (running tests, fixing linter errors)
- **4x faster** than comparable models (~250 tokens/second)

This training approach produces more practical agents than pure pre-training.

#### The Rust Advantage (Goose)
Block chose **Rust** (59.4% of codebase) for Goose, providing:
- Type safety for complex agent logic
- Performance for local execution
- Memory safety for handling sensitive code
- Cross-platform compilation

Consider Rust for performance-critical agent components.

---

### 3. Claude Code & Copilot CLI Patterns

#### Specialized Sub-agents
- Task() function invokes specialized agents with subagent_type parameter
- Common roles: database-architect, frontend-specialist, security-auditor
- Composable agent configurations in .github/agents/

#### Agent Skills (Open Standard)
- Skills stored in .github/skills/ (project) or ~/.copilot/skills/ (personal)
- SKILL.md file defines skill metadata
- Works across multiple AI agents (Copilot, Claude Code)

#### Session Protocol Requirements
- Five blocking gates using RFC 2119 keywords (MUST, SHOULD, MAY)
- MCP integration for server support
- Orchestrator pattern for non-trivial tasks

#### Parallel Agent Workflows
- Multiple terminal windows with different agents
- Architect agent iterates on plan
- Fresh instances implement the plan
- YOLO mode for trusted tasks

**Sources**:
- [Claude Code and Copilot CLI](https://jgandrews.com/posts/claude-and-copilot/)
- [Claude Code CLI Documentation](https://deepwiki.com/rjmurillo/ai-agents/5.3-claude-code-cli)

### 4. LLM Tool Calling System Design

#### Function/Tool Schema Design
- Detailed function definitions with name, purpose, parameters
- JSON Schema for parameter validation
- Expected output formats
- Supports both Anthropic and OpenAI formats

#### Execution Logic
- Planner modules for task decomposition
- Routing mechanisms for specialized agents
- Conditional flows for complex logic
- Iteration limits to prevent infinite loops

#### Popular Frameworks
- **LangGraph**: Persistence, streaming, debugging support
- **AutoGen**: Multi-agent conversations
- **Langroid**: Multi-agent programming with message passing
- **LiteLLM**: Lightweight agent in ~140 lines of Python

**Sources**:
- [Agent System Design Patterns - Databricks](https://docs.databricks.com/aws/en/generative-ai/guide/agent-system-design-patterns)
- [LLM Agents - Prompt Engineering Guide](https://www.promptingguide.ai/research/llm-agents)
- [LLM Coding Agent in 6 Steps](https://kanaka.github.io/blog/litellm-agent-in-six-steps/)

### 5. Best Practices (2026)

1. **Specification-Driven Development**
   - Define problem and plan solution before coding
   - Brainstorm detailed specification with AI
   - Outline step-by-step plan

2. **Agentic Amplification**
   - Agents amplify existing technical disciplines
   - Strong foundations required: GitOps, CI/CD, test automation
   - Architecture oversight for predictable productivity

3. **Tool Standardization**
   - Model Context Protocol (MCP) as standard for function calling
   - Query SQL databases, search web, execute Python in sandbox
   - Standardized tool interfaces across providers

4. **Flow-Based Architecture**
   - Discrete nodes for specific functions
   - Decision-making, file manipulation, code analysis
   - Support for parallel execution

**Sources**:
- [My LLM Coding Workflow](https://addyosmani.com/blog/ai-coding-workflow/)
- [Optimizing Agentic Coding](https://research.aimultiple.com/agentic-coding/)

---

---

## Comparative Analysis: Agent Implementations

| Feature | Goose | Amp | Cursor | Aider | Continue | Codeium |
|---------|-------|-----|--------|-------|----------|---------|
| **Language** | Rust (59%) | Go | TypeScript | Python | TypeScript (84%) | C++/Python |
| **Architecture** | Modular crates | Simple loop | Composer MoE | Architect/Editor | Three-tier modes | Cascade agent |
| **Tool Count** | Extensive (MCP) | 3 core tools | Unlimited | Git-focused | MCP-based | Multi-file aware |
| **Deployment** | Desktop + CLI | CLI + VS Code | Desktop IDE | CLI | Cloud + CLI + IDE | IDE (Windsurf) |
| **LLM Support** | Multi-model | Claude-focused | Multiple | Multiple | Multiple | Multiple |
| **Local Execution** | Yes | No | No | Yes | Optional | Optional |
| **Key Innovation** | MCP + Rust | Simplicity | RL training | Repo mapping | Context store | Collaborative flow |
| **Cost Model** | Local (free) | Cloud ($) | Cloud ($$$) | Local (free) | Hybrid | Cloud |
| **Context Method** | MCP | Full history | Embeddings | Graph ranking | Context store | Multi-file |
| **Checkpoints** | Git | Git | Built-in | Git | Git | Built-in |
| **Team Features** | ❌ | ✅ Shared threads | ❌ | ❌ | ✅ Hub | ✅ Team styles |
| **Open Source** | ✅ Full | ⚠️ Partial | ❌ Closed | ✅ Full | ✅ Full | ❌ Closed |

**Key Takeaways**:
1. **No single "best" approach**—different tradeoffs for different use cases
2. **Local execution** (Goose, Aider) vs **cloud performance** (most others)
3. **Simple core loop** (Amp: 300 lines) vs **complex architectures** (Cursor, Continue)
4. **Open source** (Goose, Aider, Continue) vs **proprietary** (Cursor, Codeium)
5. **Tool philosophy**: Minimal vs comprehensive both work

---

## Current State Analysis

### Existing Capabilities

**Strengths:**
1. ✅ **Semantic Code Search**: Tree-sitter parsing, embedding generation, vector search
2. ✅ **Basic Agent Core**: Agent loop with tool calling, conversation context
3. ✅ **Tool System**: Read, Write, Edit, List, Glob, Grep, Bash, SemanticSearch
4. ✅ **LLM Integration**: Anthropic provider with tool calling
5. ✅ **MCP Server**: Basic MCP server for code indexing/querying
6. ✅ **CLI**: Typer-based CLI with multiple commands
7. ✅ **Configuration**: Flexible config system for embeddings, indexing

**Current Architecture:**
```
ctxai/
├── src/ctxai/
│   ├── agent/
│   │   ├── core.py              # Agent loop with tool calling
│   │   ├── context.py           # Conversation management
│   │   ├── config.py            # Agent configuration
│   │   ├── prompts.py           # System prompts
│   │   ├── llm/
│   │   │   ├── base.py          # LLM provider interface
│   │   │   └── anthropic_provider.py
│   │   └── tools/
│   │       ├── base.py          # Tool interface
│   │       ├── registry.py      # Tool registration
│   │       ├── file_ops.py      # File tools
│   │       ├── bash_tool.py     # Bash execution
│   │       └── code_search.py   # Semantic search
│   ├── chunking.py              # Code chunking
│   ├── embeddings.py            # Embedding generation
│   ├── vector_store.py          # ChromaDB management
│   └── app.py                   # CLI entry point
```

### Gaps to Address

**Missing Features:**
1. ❌ **Interactive CLI Mode**: No conversation loop, no session management
2. ❌ **Multi-Agent Orchestration**: No specialized sub-agents
3. ❌ **Planning & Reflection**: No explicit planning or reflection patterns
4. ❌ **Git Integration**: No git tools (commit, diff, branch, PR)
5. ❌ **Testing Tools**: No test execution, coverage, assertion tools
6. ❌ **Web Tools**: No web search, fetch capabilities
7. ❌ **Session Persistence**: No conversation history saving
8. ❌ **Agent Skills**: No skills system like Copilot
9. ❌ **Memory System**: No long-term memory or knowledge graph
10. ❌ **Multiple LLM Support**: Only Anthropic, no OpenAI/local models
11. ❌ **Safety & Sandboxing**: No sandboxed execution environment
12. ❌ **User Approval Flow**: No confirmation for dangerous operations

---

## Architectural Design

### 1. System Architecture (Informed by Industry Analysis)

**Design Philosophy**: Combine the **simplicity of Amp's core loop** with **Aider's architect/editor pattern**, **Goose's MCP extensibility**, and **ctxai's unique semantic search capabilities**.

```
┌─────────────────────────────────────────────────────────────┐
│                   CLI Interface (Continue-inspired)         │
│  • Interactive REPL with Rich formatting                    │
│  • Three modes: Chat, Plan, Agent                           │
│  • Session Management & Checkpoints (Cursor-style)          │
│  • Desktop App (Electron) + CLI (Goose-inspired)            │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Core Agentic Loop (Amp-inspired)               │
│                                                             │
│  while not done:                                            │
│    response = llm.chat(conversation_history, tools)         │
│    if response.has_tool_calls:                              │
│      results = execute_tools(response.tool_calls)           │
│      conversation_history.append(results)                   │
│    else:                                                    │
│      done = True                                            │
│                                                             │
│  Key: Stateless, simple, ~300 lines core logic             │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│           Architect/Editor Layer (Aider-inspired)           │
│                                                             │
│  ┌──────────────┐                    ┌──────────────┐      │
│  │  Architect   │────── Plan ──────▶ │   Editor     │      │
│  │   (o3/o1)    │                    │  (GPT-4o/    │      │
│  │              │                    │   Claude)    │      │
│  │ • Analyze    │                    │ • Apply edits│      │
│  │ • Plan       │◀──── Feedback ──── │ • Format     │      │
│  │ • Design     │                    │ • Validate   │      │
│  └──────────────┘                    └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│            Context Management (Multi-strategy)              │
│                                                             │
│  Repository Map │ Semantic Search  │  MCP Context          │
│  (Aider-style)  │ (ctxai strength) │ (Goose-style)         │
│  ─────────────  │  ──────────────  │  ──────────           │
│  • Graph rank   │  • Vector search │  • MCP servers        │
│  • Dependencies │  • Tree-sitter   │  • External tools     │
│  • ~1K tokens   │  • Embeddings    │  • Live data          │
└─────────────────────────────────────────────────────────────┘
                         │
┌───────────────────────▼─────────────────────────────────────┐
│         Tool System (Hybrid: Minimal + Extensions)          │
│                                                             │
│  Core Tools (Amp-inspired)  │  Extended Tools (Cursor)      │
│  ─────────────────────────  │  ───────────────────────      │
│  • read_file                │  • semantic_search (ctxai)    │
│  • write_file               │  • git_* (6 tools)            │
│  • edit_file                │  • test_runner                │
│  • list_files               │  • web_search                 │
│  • bash_exec                │  • browser_control            │
│                             │  • + MCP extensions (Goose)   │
└─────────────────────────────────────────────────────────────┘
                         │
┌───────────────────────▼─────────────────────────────────────┐
│         LLM Provider Layer (Multi-model like all)           │
│  Anthropic │ OpenAI │ Gemini │ DeepSeek │ Local (Ollama)   │
│                                                             │
│  Strategy: Architect (expensive) + Editor (cheap)           │
└─────────────────────────────────────────────────────────────┘
                         │
┌───────────────────────▼─────────────────────────────────────┐
│              Storage & Memory Layer                         │
│  • Vector Store (ChromaDB) - semantic search                │
│  • Session History (SQLite) - checkpoints                   │
│  • Repository Map (graph) - context                         │
│  • Configuration (JSON) - settings                          │
│  • MCP Registry - extensions                                │
└─────────────────────────────────────────────────────────────┘
```

**Key Architectural Decisions**:

1. **Core Loop**: Use Amp's simple pattern (~300 lines) rather than complex orchestration
2. **Two-Model Strategy**: Implement Aider's architect/editor for quality + cost optimization
3. **Context**: Triple strategy—repository mapping (Aider) + semantic search (ctxai) + MCP (Goose)
4. **Tools**: Start minimal (5 core tools), extend via MCP like Goose
5. **Deployment**: CLI-first (like Aider), add desktop app later (like Goose)
6. **Language**: Python for rapid development, consider Rust later for performance components
7. **Open Source**: Fully open like Goose/Aider/Continue (competitive advantage)

### 2. Agentic Workflow Patterns

#### Pattern 1: ReAct Loop (Primary Pattern)
```
User Request
     │
     ▼
┌────────────────┐
│   REASON       │  ← Analyze task, plan approach
└────────┬───────┘
         │
         ▼
┌────────────────┐
│   ACT          │  ← Execute tool(s)
└────────┬───────┘
         │
         ▼
┌────────────────┐
│   OBSERVE      │  ← Review tool results
└────────┬───────┘
         │
         ▼
┌────────────────┐
│   REFLECT      │  ← Self-critique, decide next action
└────────┬───────┘
         │
         ├─── Continue? ───→ Loop back to REASON
         │
         └─── Done? ───→ Respond to user
```

#### Pattern 2: Planning Pattern
```
Complex Task
     │
     ▼
┌─────────────────────┐
│  Decompose Task     │  ← Break into subtasks
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Create Plan        │  ← Sequential steps with dependencies
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Execute Each Step  │  ← Use ReAct loop per step
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Validate Result    │  ← Check against plan objectives
└─────────────────────┘
```

#### Pattern 3: Multi-Agent Orchestration
```
Orchestrator
     │
     ├──→ Explorer Agent (Gather context)
     │         │
     │         └──→ Returns: Relevant code, architecture insights
     │
     ├──→ Coder Agent (Implement changes)
     │         │
     │         └──→ Returns: Modified code, new files
     │
     └──→ Reviewer Agent (Validate changes)
               │
               └──→ Returns: Test results, review feedback
```

### 3. Enhanced Tool System

#### Tool Categories

**1. File Operations** (Existing + Enhanced)
- `read_file`: Read file contents with line ranges
- `write_file`: Write content to file (new/overwrite)
- `edit_file`: Edit existing file (find/replace, line-based)
- `delete_file`: Delete file with confirmation
- `move_file`: Move/rename file
- `list_files`: List files in directory with filters
- `glob`: Pattern-based file search
- `grep`: Content search with context
- `tree`: Directory tree visualization

**2. Bash Operations** (Enhanced)
- `bash_exec`: Execute bash command with timeout
- `bash_shell`: Interactive shell session
- `bash_pipe`: Pipe commands together
- `set_env`: Set environment variable
- `get_env`: Get environment variable
- `watch_command`: Watch command output
- `kill_process`: Kill running process

**3. Git Operations** (New)
- `git_status`: Get repository status
- `git_diff`: Show file differences
- `git_commit`: Create commit with message
- `git_branch`: List/create/delete branches
- `git_checkout`: Switch branches
- `git_log`: View commit history
- `git_pull`: Pull from remote
- `git_push`: Push to remote
- `create_pr`: Create pull request (GitHub/GitLab)
- `git_stash`: Stash changes

**4. Testing Tools** (New)
- `run_tests`: Execute test suite
- `run_test_file`: Run specific test file
- `run_test_case`: Run specific test case
- `get_coverage`: Get code coverage report
- `create_test`: Generate test from code
- `mock_function`: Create mock for function
- `assert_output`: Assert expected output
- `debug_test`: Debug failing test
- `benchmark`: Run performance benchmarks

**5. Web Tools** (New)
- `web_search`: Search the web for information
- `web_fetch`: Fetch URL content
- `web_parse`: Parse HTML/JSON from web
- `api_call`: Make API request
- `web_cache`: Cache web content locally
- `api_auth`: Authenticate with API

**6. Code Analysis Tools** (New)
- `semantic_search`: Search code by meaning (existing)
- `find_references`: Find all references to symbol
- `find_definition`: Find symbol definition
- `get_call_graph`: Get function call graph
- `analyze_complexity`: Analyze code complexity
- `detect_patterns`: Detect code patterns/smells
- `suggest_refactor`: Suggest refactoring opportunities

**7. Planning & Memory Tools** (New)
- `create_plan`: Create execution plan
- `update_plan`: Update existing plan
- `get_plan`: Retrieve current plan
- `save_memory`: Save to long-term memory
- `recall_memory`: Retrieve from memory
- `add_to_todo`: Add item to TODO list
- `mark_done`: Mark TODO item complete

**8. Session Management Tools** (New)
- `save_session`: Save conversation session
- `load_session`: Load previous session
- `list_sessions`: List all sessions
- `delete_session`: Delete session
- `export_session`: Export session to file

### 4. LLM Provider Layer

#### Multi-Provider Support

```python
class BaseLLMProvider(ABC):
    """Abstract interface for LLM providers"""

    @abstractmethod
    def chat(self, messages, tools=None) -> LLMResponse:
        pass

    @abstractmethod
    def stream_chat(self, messages, tools=None) -> Iterator[LLMChunk]:
        pass

# Implementations:
- AnthropicProvider (existing)
- OpenAIProvider (new)
- GeminiProvider (new)
- OllamaProvider (new - local models)
- AzureOpenAIProvider (new)
```

#### Provider Selection Strategy
1. Check config for default provider
2. Support provider override per request
3. Fallback chain: Primary → Secondary → Local
4. Cost tracking per provider

### 5. Interactive CLI Design

#### Command Structure
```
ctxai                           # Interactive mode (new default)
ctxai chat                      # Interactive chat (alias)
ctxai code                      # Code generation mode
ctxai index <path> <name>       # Index codebase (existing)
ctxai query <index> <query>     # Query index (existing)
ctxai dashboard                 # Web dashboard (existing)
ctxai server                    # MCP server (existing)
ctxai session list              # List sessions
ctxai session load <id>         # Load session
ctxai config show               # Show configuration
ctxai config set <key> <value>  # Set config value
```

#### Interactive Mode Features
- **REPL Interface**: Read-Eval-Print Loop with rich formatting
- **Auto-completion**: Tab completion for commands, files, symbols
- **Syntax Highlighting**: Code highlighting in responses
- **Progress Indicators**: Spinners, progress bars for long operations
- **History Navigation**: Up/down arrow for command history
- **Multi-line Input**: Support for multi-line code blocks
- **File Watching**: Live reload on file changes
- **Inline Editing**: Edit agent responses before applying

#### Session Management
```python
class Session:
    session_id: str
    created_at: datetime
    updated_at: datetime
    conversation_history: List[Message]
    working_directory: Path
    active_plan: Optional[Plan]
    metadata: Dict[str, Any]

    def save(self) -> None:
        """Persist session to disk"""

    def load(session_id: str) -> Session:
        """Load session from disk"""
```

### 6. Planning & Reflection System

#### Plan Structure
```python
@dataclass
class Plan:
    """Represents an execution plan"""
    plan_id: str
    goal: str
    steps: List[PlanStep]
    created_at: datetime
    status: PlanStatus  # draft, active, completed, failed

@dataclass
class PlanStep:
    """Single step in plan"""
    step_id: str
    description: str
    agent_type: str  # explorer, coder, reviewer
    tools_needed: List[str]
    dependencies: List[str]  # step_ids
    status: StepStatus  # pending, in_progress, done, failed
    result: Optional[str]
```

#### Planning Flow
1. **User Request** → Orchestrator analyzes complexity
2. **High Complexity** → Create multi-step plan
3. **Present Plan** → Show user for approval
4. **Execute Plan** → Run each step with appropriate agent
5. **Reflect** → Self-critique after each step
6. **Adapt** → Modify plan if needed
7. **Complete** → Validate against original goal

### 7. Safety & Sandboxing

#### Safety Layers
1. **User Confirmation**: Prompt for dangerous operations
   - File deletion
   - Git push to remote
   - System command execution
   - Network requests

2. **Sandboxed Execution**: Isolated environments
   - Docker containers for code execution
   - chroot jails for file operations
   - Network isolation for web tools

3. **Rate Limiting**: Prevent abuse
   - Max requests per minute
   - Max tokens per hour
   - Max file operations per session

4. **Audit Logging**: Track all operations
   - Tool calls with parameters
   - File modifications
   - Network requests
   - LLM interactions

### 8. Agent Skills System

#### Skill Definition
```yaml
# .ctxai/skills/python-test-writer/SKILL.md
name: python-test-writer
description: Write pytest tests for Python functions
version: 1.0.0

inputs:
  - name: function_code
    type: string
    required: true
  - name: test_framework
    type: enum
    values: [pytest, unittest]
    default: pytest

instructions: |
  1. Analyze the function signature and behavior
  2. Generate comprehensive test cases covering:
     - Happy path
     - Edge cases
     - Error handling
  3. Use appropriate fixtures and mocks
  4. Follow project testing conventions

examples:
  - input: "def add(a: int, b: int) -> int: return a + b"
    output: |
      def test_add():
          assert add(1, 2) == 3
          assert add(-1, 1) == 0
```

#### Skill Loading
- Project skills: `.ctxai/skills/`
- User skills: `~/.ctxai/skills/`
- Built-in skills: Package resources

#### Skill Invocation
```
User: @python-test-writer write tests for add_user function
Agent: [Loads skill] → [Follows instructions] → [Generates tests]
```

### 9. Memory & Knowledge System

#### Short-Term Memory (Existing)
- Conversation context (last N messages)
- Current plan and status
- Recent tool results

#### Long-Term Memory (New)
Using **Graphiti** or similar knowledge graph:

```python
class MemorySystem:
    """Long-term memory with knowledge graph"""

    def add_fact(self, fact: str, context: Dict) -> None:
        """Store fact with context"""

    def query_memory(self, query: str) -> List[Memory]:
        """Retrieve relevant memories"""

    def build_context(self, task: str) -> Dict:
        """Build context from related memories"""
```

**Memory Types:**
- **Episodic**: Past interactions and outcomes
- **Semantic**: Code patterns, best practices, project conventions
- **Procedural**: Successful workflows and strategies

### 10. Model Context Protocol (MCP) Integration

#### Enhanced MCP Server
```python
# Expose ctxai capabilities via MCP
mcp_server.add_tool("index_codebase")
mcp_server.add_tool("search_code")
mcp_server.add_tool("analyze_code")
mcp_server.add_tool("generate_code")
mcp_server.add_tool("run_tests")
mcp_server.add_tool("create_pr")

# Allow other agents to use ctxai as a tool
```

#### MCP Client
```python
# Use external MCP servers as tools
mcp_client.connect("github-mcp-server")
mcp_client.connect("aws-mcp-server")
mcp_client.connect("postgres-mcp-server")

# Agent can now use external MCP tools
```

---

## Implementation Roadmap

### Phase 1: Foundation Enhancement (Weeks 1-2)

**Goals**: Strengthen core agent capabilities

#### Tasks:
1. **Multi-Provider LLM Support**
   - [ ] Implement OpenAIProvider
   - [ ] Implement GeminiProvider
   - [ ] Implement OllamaProvider (local)
   - [ ] Add provider selection in config
   - [ ] Add fallback chain

2. **Enhanced Tool System**
   - [ ] Add Git tools (status, diff, commit, branch, push, PR)
   - [ ] Add Web tools (search, fetch, parse)
   - [ ] Add Testing tools (run_tests, coverage, mock)
   - [ ] Add Code analysis tools (find_references, call_graph)
   - [ ] Improve error handling and retries

3. **Planning System**
   - [ ] Implement Plan and PlanStep classes
   - [ ] Add planning prompt templates
   - [ ] Create plan generation logic
   - [ ] Add plan approval flow
   - [ ] Implement plan execution tracking

**Deliverables**:
- Multi-provider LLM support
- 30+ tools across 6 categories
- Basic planning system

### Phase 2: Multi-Agent Architecture (Weeks 3-4)

**Goals**: Implement specialized agents and orchestration

#### Tasks:
1. **Specialized Agents**
   - [ ] Implement ExplorerAgent (code search, analysis)
   - [ ] Implement CoderAgent (code generation, editing)
   - [ ] Implement ReviewerAgent (testing, validation)
   - [ ] Implement OrchestratorAgent (routing, coordination)

2. **Agent Orchestration**
   - [ ] Task classification logic
   - [ ] Agent routing based on task type
   - [ ] Inter-agent communication
   - [ ] Result aggregation

3. **Reflection Pattern**
   - [ ] Self-critique prompts
   - [ ] Result validation logic
   - [ ] Feedback loop implementation
   - [ ] Error recovery strategies

**Deliverables**:
- 4 specialized agents
- Orchestrator with routing
- Reflection capabilities

### Phase 3: Interactive CLI (Weeks 5-6)

**Goals**: Build rich interactive experience

#### Tasks:
1. **REPL Interface**
   - [ ] Implement rich REPL with prompt_toolkit
   - [ ] Add syntax highlighting with pygments
   - [ ] Add auto-completion for commands/files
   - [ ] Add multi-line input support
   - [ ] Add history navigation

2. **Session Management**
   - [ ] Implement Session class
   - [ ] Add session persistence (SQLite)
   - [ ] Add session list/load/delete commands
   - [ ] Add session export/import

3. **Progress & Feedback**
   - [ ] Add progress indicators (spinners)
   - [ ] Add progress bars for long operations
   - [ ] Add streaming output for LLM responses
   - [ ] Add formatted tables for results

**Deliverables**:
- Interactive REPL mode
- Session persistence
- Rich CLI experience

### Phase 4: Skills & Memory (Weeks 7-8)

**Goals**: Add skills system and long-term memory

#### Tasks:
1. **Agent Skills**
   - [ ] Define SKILL.md format
   - [ ] Implement skill loader
   - [ ] Add skill directory scanning
   - [ ] Create 5-10 built-in skills
   - [ ] Add skill invocation syntax (@skill-name)

2. **Memory System**
   - [ ] Integrate Graphiti or similar
   - [ ] Implement fact storage
   - [ ] Implement memory retrieval
   - [ ] Add context building from memories
   - [ ] Add memory pruning/cleanup

3. **Built-in Skills**
   - [ ] python-test-writer
   - [ ] code-reviewer
   - [ ] documentation-generator
   - [ ] refactoring-assistant
   - [ ] bug-finder

**Deliverables**:
- Skills system with 10+ built-in skills
- Long-term memory with knowledge graph
- Context-aware agent

### Phase 5: Safety & Polish (Weeks 9-10)

**Goals**: Production-ready features

#### Tasks:
1. **Safety Features**
   - [ ] User confirmation for dangerous operations
   - [ ] Sandboxed execution (Docker integration)
   - [ ] Rate limiting
   - [ ] Audit logging

2. **Enhanced MCP**
   - [ ] Expand MCP server capabilities
   - [ ] Add MCP client for external servers
   - [ ] Document MCP integration

3. **Documentation**
   - [ ] Comprehensive README updates
   - [ ] API documentation
   - [ ] Tutorial videos/guides
   - [ ] Example projects

4. **Testing & Quality**
   - [ ] Unit tests for all components
   - [ ] Integration tests for workflows
   - [ ] Performance benchmarks
   - [ ] Security audit

**Deliverables**:
- Production-ready agent
- Comprehensive documentation
- Full test coverage

### Phase 6: Advanced Features (Weeks 11-12)

**Goals**: Differentiation and advanced capabilities

#### Tasks:
1. **Parallel Agent Execution**
   - [ ] Run multiple agents in parallel
   - [ ] Task queue management
   - [ ] Resource allocation
   - [ ] Result merging

2. **Advanced Analysis**
   - [ ] Code quality metrics
   - [ ] Architecture visualization
   - [ ] Dependency analysis
   - [ ] Security scanning

3. **Cloud Integration**
   - [ ] GitHub Actions integration
   - [ ] GitLab CI integration
   - [ ] Cloud deployment tools
   - [ ] Container orchestration

**Deliverables**:
- Parallel agent execution
- Cloud CI/CD integration
- Advanced code analysis

---

## Technical Specifications

### Technology Stack

**Core:**
- Python 3.10+
- Tree-sitter (code parsing)
- ChromaDB (vector storage)
- SQLite (session/memory storage)
- Pydantic (data validation)

**CLI:**
- Typer (command framework)
- prompt_toolkit (interactive REPL)
- Rich (formatting, tables, progress)
- Pygments (syntax highlighting)

**LLM Providers:**
- anthropic-sdk (Claude)
- openai-sdk (GPT-4, GPT-4o)
- google-generativeai (Gemini)
- ollama-python (local models)

**Tools:**
- GitPython (git operations)
- pytest (testing tools)
- requests (web tools)
- beautifulsoup4 (web parsing)

**Memory:**
- graphiti or neo4j (knowledge graph)
- redis (optional caching)

**MCP:**
- mcp-sdk (Model Context Protocol)

### Configuration Schema

```json
{
  "version": "2.0",
  "llm": {
    "default_provider": "anthropic",
    "providers": {
      "anthropic": {
        "model": "claude-3-5-sonnet-20241022",
        "api_key": "${ANTHROPIC_API_KEY}",
        "temperature": 0.7,
        "max_tokens": 4096
      },
      "openai": {
        "model": "gpt-4o",
        "api_key": "${OPENAI_API_KEY}",
        "temperature": 0.7,
        "max_tokens": 4096
      },
      "ollama": {
        "model": "codellama:13b",
        "base_url": "http://localhost:11434",
        "temperature": 0.7
      }
    },
    "fallback_chain": ["anthropic", "openai", "ollama"]
  },
  "agent": {
    "max_iterations": 10,
    "require_approval": true,
    "planning_enabled": true,
    "reflection_enabled": true,
    "verbose": false
  },
  "tools": {
    "bash": {
      "timeout": 30,
      "require_approval": true,
      "allowed_commands": []
    },
    "git": {
      "require_approval_for_push": true,
      "auto_commit": false
    },
    "web": {
      "max_requests_per_minute": 10,
      "timeout": 10
    }
  },
  "skills": {
    "project_skills_dir": ".ctxai/skills",
    "user_skills_dir": "~/.ctxai/skills",
    "auto_load": true
  },
  "memory": {
    "enabled": true,
    "backend": "graphiti",
    "max_memories": 10000
  },
  "session": {
    "auto_save": true,
    "save_interval": 60
  },
  "safety": {
    "sandbox_enabled": false,
    "rate_limit_enabled": true,
    "audit_log_enabled": true
  }
}
```

### API Interfaces

#### Agent Interface
```python
class IAgent(ABC):
    @abstractmethod
    async def process_message(self, message: str) -> str:
        """Process user message and return response"""

    @abstractmethod
    async def execute_plan(self, plan: Plan) -> PlanResult:
        """Execute a plan"""

    @abstractmethod
    def create_plan(self, task: str) -> Plan:
        """Create execution plan for task"""
```

#### Tool Interface
```python
class ITool(ABC):
    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Get tool schema"""

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute tool"""

    @abstractmethod
    async def validate(self, **kwargs) -> Tuple[bool, Optional[str]]:
        """Validate parameters"""
```

### Performance Targets

- **Agent Response Time**: < 5s for simple queries, < 30s for complex tasks
- **Tool Execution**: < 1s for file ops, < 5s for bash, < 10s for web
- **Session Load Time**: < 1s for recent sessions
- **Memory Retrieval**: < 500ms for semantic search
- **Token Efficiency**: < 10k tokens per simple task

### Security Considerations

1. **Input Validation**: Sanitize all user inputs
2. **Command Injection**: Prevent shell injection in bash tools
3. **Path Traversal**: Validate file paths
4. **API Key Safety**: Never log or expose API keys
5. **Rate Limiting**: Prevent abuse and cost overruns
6. **Sandboxing**: Isolate code execution
7. **Audit Logging**: Track all sensitive operations

---

## References

### Research Sources

1. [Building AI Agents in 2026](https://levelup.gitconnected.com/the-2026-roadmap-to-ai-agent-mastery-5e43756c0f26)
2. [How to Build AI Agents in 2026](https://vocal.media/journal/how-to-build-ai-agents-in-2026-stop-coding-like-it-s-2024)
3. [My LLM Coding Workflow](https://addyosmani.com/blog/ai-coding-workflow/)
4. [The 2026 Architect's Dilemma](https://dev.to/ridwan_sassman_3d07/the-2026-architects-dilemma-orchestrating-ai-agents-not-writing-code-the-paradigm-shift-from-219c)
5. [A Deep Dive into Deep Agent Architecture](https://dev.to/apssouza22/a-deep-dive-into-deep-agent-architecture-for-ai-coding-assistants-3c8b)
6. [Claude Code and Copilot CLI](https://jgandrews.com/posts/claude-and-copilot/)
7. [Claude Code CLI Documentation](https://deepwiki.com/rjmurillo/ai-agents/5.3-claude-code-cli)
8. [Agent Skills](https://claude-plugins.dev/skills/@githubnext/gh-aw/copilot-cli)
9. [Agent System Design Patterns - Databricks](https://docs.databricks.com/aws/en/generative-ai/guide/agent-system-design-patterns)
10. [LLM Agents - Prompt Engineering Guide](https://www.promptingguide.ai/research/llm-agents)
11. [Function Calling with LLMs](https://www.promptingguide.ai/applications/function_calling)
12. [LLM Coding Agent in 6 Steps](https://kanaka.github.io/blog/litellm-agent-in-six-steps/)
13. [Top AI Agentic Workflow Patterns](https://medium.com/@Deep-concept/top-ai-agentic-workflow-patterns-that-will-lead-in-2026-0e4755fdc6f6)
14. [4 Agentic AI Design Patterns](https://research.aimultiple.com/agentic-ai-design-patterns/)
15. [What are Agentic Workflows?](https://www.wrike.com/blog/what-are-agentic-workflows/)
16. [5 Key Trends Shaping Agentic Development](https://thenewstack.io/5-key-trends-shaping-agentic-development-in-2026/)
17. [Top AI Agentic Workflow Patterns - ByteByteGo](https://blog.bytebytego.com/p/top-ai-agentic-workflow-patterns)
18. [Optimizing Agentic Coding](https://research.aimultiple.com/agentic-coding/)

### Agent Implementation References

#### Goose (Block)
19. [GitHub - block/goose](https://github.com/block/goose)
20. [Goose AGENTS.md](https://github.com/block/goose/blob/main/AGENTS.md)
21. [Block Open Source - Goose](https://block.xyz/inside/block-open-source-introduces-codename-goose)
22. [Goose Official Site](https://block.github.io/goose/)
23. [What Makes Goose Different](https://dev.to/nickytonline/what-makes-goose-different-from-other-ai-coding-agents-2edc)
24. [Meet Goose - Data + AI Summit](https://www.databricks.com/dataaisummit/session/meet-goose-open-source-ai-agent)

#### Amp (Sourcegraph)
25. [Amp - How to Build an Agent](https://ampcode.com/how-to-build-an-agent)
26. [Amp by Sourcegraph](https://sourcegraph.com/amp)
27. [Amp Code Official Site](https://ampcode.com/)
28. [Staying in the Loop with Amp](https://medium.com/@jonathanaraney/staying-in-the-loop-how-amp-keeps-programmers-engaged-in-the-sdlc-with-agentic-ai-fe1a9d49eedc)
29. [GitHub - sourcegraph/amp-examples-and-guides](https://github.com/sourcegraph/amp-examples-and-guides)
30. [GitHub - sourcegraph/amp-contrib](https://github.com/sourcegraph/amp-contrib)
31. [Amp Owner's Manual](https://ampcode.com/manual)
32. [Amp Agentic Code Review](https://ampcode.com/news/agentic-code-review)

#### Cursor
33. [Cursor Features](https://cursor.com/features)
34. [Cursor Agent Overview](https://cursor.com/docs/agent/overview)
35. [Cursor Agent Modes](https://cursor.com/docs/agent/modes)
36. [Cursor Changelog 2.0](https://cursor.com/changelog/2-0)
37. [Cursor 2.0 Explained - Codecademy](https://www.codecademy.com/article/cursor-2-0-new-ai-model-explained)
38. [Cursor AI Review 2026](https://prismic.io/blog/cursor-ai)
39. [License to Kill: Coding with Cursor AI Agents](https://levelup.gitconnected.com/license-to-kill-coding-with-cursor-ai-agents-1df3d6a0bfe8)
40. [How to Use Cursor Agent Mode](https://apidog.com/blog/how-to-use-cursor-agent-mode/)

#### Aider
41. [GitHub - Aider-AI/aider](https://github.com/paul-gauthier/aider)
42. [Aider Repository Mapping](https://aider.chat/docs/repomap.html)
43. [Aider Leaderboards](https://aider.chat/docs/leaderboards/)
44. [Aider Architect/Editor Approach](https://generaitelabs.com/aider-implements-new-architect-editor-approach-for-ai-assisted-coding/)
45. [Getting Started with Aider](https://blog.openreplay.com/getting-started-aider-ai-coding-terminal/)
46. [Level Up Your Coding with Aider](https://medium.com/@honeyricky1m3/level-up-your-coding-with-aider-the-open-source-ai-coding-assistant-thats-changing-the-game-43ef98f82612)
47. [Aider o3 + gpt-4.1 Performance](https://x.com/paulgauthier/status/1912892114310160392)

#### Continue.dev
48. [GitHub - continuedev/continue](https://github.com/continuedev/continue)
49. [Continue Agent Quick Start](https://docs.continue.dev/ide-extensions/agent/quick-start)
50. [Continue Official Site](https://www.continue.dev/)
51. [Continue In-Depth Analysis](https://atoms.dev/insights/continuedev-an-in-depth-analysis-of-an-open-source-ai-powered-coding-assistant-for-enhanced-developer-workflows/6de278ae9d7e4858beaa8e53780b2773)
52. [Continue - Open-source AI Code Agent](https://marketplace.visualstudio.com/items?itemName=Continue.continue)
53. [Continue.dev In-Depth Guide](https://skywork.ai/skypage/ko/Continue.dev-In-Depth:-My-Guide-to-the-Future-of-AI-Assisted-Development/1972847152152506368)
54. [Ollama Blog - Continue Code Assistant](https://ollama.com/blog/continue-code-assistant)

#### Codeium/Windsurf
55. [Windsurf - The best AI for Coding](https://windsurf.com/)
56. [Codeium AI Coding Assistant](https://skywork.ai/blog/codeium-ai-coding-assistant/)
57. [What Is Codeium?](https://skywork.ai/blog/codeium-definition-ai-code-assistant/)
58. [Scaling with Llama - Codeium](https://ai.meta.com/blog/codeium-ai-coding-assistant-llama/)
59. [GitHub - Exafunction/codeium](https://github.com/Exafunction/codeium)
60. [Codeium for AI Coding Review](https://codeium.en.softonic.com/web-apps)
61. [AI-assisted Coding with Codeium](https://carpentries-incubator.github.io/gen-ai-coding/)

### General Resources
62. [How to Build a Coding Agent Workshop](https://ghuntley.com/agent/)

---

## Conclusion

This comprehensive plan transforms ctxai from a semantic code search tool into a full-featured autonomous AI coding agent. The implementation is informed by deep analysis of the leading agents in 2026: **Goose, Amp, Cursor, Aider, Continue, and Codeium/Windsurf**.

### Key Learnings from Industry Leaders

1. **Simplicity Works** (Amp): Core agentic loop can be ~300 lines—sophisticated behavior emerges from simple patterns

2. **Specialization Wins** (Aider): Architect/Editor pattern achieves 78.2% accuracy + 100% formatting + cost reduction

3. **Context is Critical** (Aider, Cursor): Repository mapping, semantic search, and embeddings solve the context challenge

4. **Tools Can Be Minimal** (Amp): Just 3 core tools that chain together can be as effective as comprehensive toolkits

5. **Multi-Modal Deployment** (Continue, Goose): Support cloud, CLI, and IDE modes for different workflows

6. **Training Matters** (Cursor): RL training inside real codebases produces 4x faster, more practical agents

7. **Rust for Performance** (Goose): When local execution matters, Rust provides safety + speed

8. **Open Source Advantage** (Goose, Aider, Continue): Full transparency attracts contributors and trust

### ctxai's Unique Strengths

**What ctxai brings that others don't:**
1. **Semantic Search Foundation**: Existing tree-sitter + vector search infrastructure
2. **MCP Integration Ready**: Already has MCP server, can add client
3. **Embedding Expertise**: Multiple provider support (local, OpenAI, HuggingFace)
4. **Repository Indexing**: Already indexes codebases—extend to repository mapping
5. **Python Stack**: Rapid development, extensive ecosystem

**Strategic Positioning:**
- **Simple** like Amp (core loop)
- **Smart** like Aider (architect/editor + repo mapping)
- **Extensible** like Goose (MCP + open source)
- **Fast** like Cursor (efficient token usage)
- **Unique** (semantic search + triple context strategy)

### Success Metrics

The agent will be considered production-ready when it can:
- ✅ Autonomously implement features from natural language descriptions
- ✅ Debug and fix bugs with minimal human guidance
- ✅ Refactor code intelligently while maintaining tests
- ✅ Write comprehensive tests for existing code
- ✅ Integrate with CI/CD pipelines and Git workflows
- ✅ Work across multiple programming languages
- ✅ Learn from past interactions via memory system
- ✅ Match or exceed Aider's 78% benchmark performance
- ✅ Complete simple tasks in <5s, complex tasks in <30s

### Implementation Strategy

**Phase 1 Focus** (Immediate):
1. Implement Amp's simple agentic loop (~300 lines core)
2. Add Aider's repository mapping (graph-ranking)
3. Keep existing tools, add 6 git tools
4. Support OpenAI + Anthropic (architect/editor pattern)

**Avoid Overengineering**:
- Don't build complex orchestration initially
- Don't add 50 tools—start with 10-15 core ones
- Don't build custom embeddings—use existing ctxai strength
- Don't build desktop app yet—CLI first like Aider

**Measure and Iterate**:
- Benchmark against Aider's polyglot coding test (target: >70%)
- Track token efficiency (target: <10k tokens per simple task)
- Monitor cost (architect/editor should reduce costs 40-60%)
- Gather user feedback early and often

The roadmap spans 12 weeks with clear deliverables at each phase, ensuring steady progress toward a production-ready AI coding agent that combines the best patterns from industry leaders with ctxai's unique semantic search capabilities.

**Start simple. Iterate fast. Learn from the best. Build something better.**
