# ctxai Agent System Documentation

## Overview

The ctxai agent system is an autonomous AI coding assistant that combines LLM reasoning with tool-calling capabilities to perform complex software development tasks. The agent can read/write files, execute bash commands, and perform semantic code searches - all orchestrated through an intelligent agent loop with error recovery.

---

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────┐
│                    Agent Core                       │
│  ┌──────────────────────────────────────────────┐  │
│  │           Conversation Context                │  │
│  │  (Message History + Token Management)         │  │
│  └──────────────────────────────────────────────┘  │
│                        ↕                            │
│  ┌──────────────────────────────────────────────┐  │
│  │             Agent Loop                        │  │
│  │  - Process user messages                      │  │
│  │  - Call LLM with tool schemas                 │  │
│  │  - Execute tool calls                         │  │
│  │  - Feed results back to LLM                   │  │
│  │  - Iterate until task complete                │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
           ↓                               ↓
    ┌──────────┐                    ┌──────────┐
    │   LLM    │                    │  Tools   │
    │ Provider │                    │ Registry │
    └──────────┘                    └──────────┘
         ↓                                ↓
    ┌──────────┐              ┌─────────────────────┐
    │ Claude   │              │ • File Operations   │
    │ OpenAI   │              │ • Bash Execution    │
    │ Ollama   │              │ • Semantic Search   │
    └──────────┘              └─────────────────────┘
```

---

## Agent Loop Flow

```
1. User Message
   ↓
2. Add to Conversation Context
   ↓
3. Generate LLM Request (with tool schemas)
   ↓
4. LLM Response
   ├─→ Text Only → Return to User (Done)
   └─→ Tool Calls
       ↓
5. Execute Tools (async)
   ↓
6. Add Tool Results to Context
   ↓
7. Loop back to Step 3 (until done or max iterations)
```

**Key Features**:
- **Multi-turn conversations**: Maintains context across interactions
- **Tool calling orchestration**: LLM decides which tools to use
- **Error recovery**: Automatic retry with recovery prompts on failures
- **Iteration limits**: Prevents infinite loops (default: 10 iterations)
- **Token management**: Truncates old messages to stay within limits

---

## Configuration

### 1. AgentLLMConfig

Controls the language model provider and behavior.

```python
@dataclass
class AgentLLMConfig:
    provider: str = "anthropic"  # "anthropic", "openai", "ollama"
    model: Optional[str] = None  # Provider-specific default if None
    api_key: Optional[str] = None  # Falls back to env vars
    fallback_providers: List[str] = ["openai", "ollama"]
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
```

**Supported Providers**:
- **Anthropic**: claude-3-5-sonnet-20241022 (default), claude-3-opus, claude-3-haiku
- **OpenAI**: gpt-4, gpt-3.5-turbo (future)
- **Ollama**: Local models (future)

**API Key Resolution**:
1. Check `api_key` field
2. Check environment variables:
   - `ANTHROPIC_API_KEY` for Anthropic
   - `OPENAI_API_KEY` for OpenAI

### 2. AgentToolsConfig

Controls which tools are available and their safety limits.

```python
@dataclass
class AgentToolsConfig:
    enabled_tools: Optional[List[str]] = None  # None = all enabled
    bash_allowed_commands: Optional[List[str]] = None  # Whitelist
    bash_blocked_commands: List[str] = [
        "rm -rf /",
        "dd if=",
        "mkfs",
        ":(){ :|:& };:",  # Fork bomb
        "chmod -R 777",
        "> /dev/sda",
        "mv / /dev/null",
    ]
    bash_timeout: int = 30
    max_file_size_mb: int = 10
    allow_outside_project: bool = False
```

**Safety Features**:
- **Bash command filtering**: Blacklist for dangerous commands
- **File size limits**: Prevents reading huge files
- **Project sandboxing**: Restrict file operations to project directory
- **Command timeout**: Kills long-running commands

### 3. AgentBehaviorConfig

Controls agent execution behavior.

```python
@dataclass
class AgentBehaviorConfig:
    planning_enabled: bool = True
    require_user_approval: bool = True
    max_iterations: int = 10
    auto_save_context: bool = True
    verbose: bool = False
    stream_responses: bool = True
```

---

## Available Tools

### 1. File Operations (6 tools)

#### read_file
Read file contents with optional line ranges.

```python
Parameters:
  - file_path (str): Absolute or relative path
  - start_line (int, optional): Starting line (1-indexed)
  - end_line (int, optional): Ending line (inclusive)

Returns:
  - File contents with line numbers
  - Error if file not found or too large
```

#### write_file
Create or overwrite a file.

```python
Parameters:
  - file_path (str): Target file path
  - content (str): File contents

Returns:
  - Success confirmation with path
```

#### edit_file
Perform string replacements in files.

```python
Parameters:
  - file_path (str): Target file path
  - old_text (str): Text to replace
  - new_text (str): Replacement text

Returns:
  - Success confirmation with changes made
```

#### list_directory
List contents of a directory.

```python
Parameters:
  - directory_path (str): Directory to list
  - recursive (bool, optional): Recursive listing

Returns:
  - List of files and directories
```

#### glob_search
Search for files using glob patterns.

```python
Parameters:
  - pattern (str): Glob pattern (e.g., "**/*.py")
  - base_path (str, optional): Base directory

Returns:
  - List of matching file paths
```

#### grep_search
Search file contents using regex.

```python
Parameters:
  - pattern (str): Regex pattern
  - file_pattern (str, optional): Filter files (e.g., "*.py")
  - case_sensitive (bool, optional): Case sensitivity

Returns:
  - Matching lines with file paths and line numbers
```

### 2. Bash Execution

#### bash
Execute bash commands with safety filtering.

```python
Parameters:
  - command (str): Bash command to execute
  - working_directory (str, optional): Working directory

Returns:
  - STDOUT and STDERR output
  - Exit code and metadata

Safety:
  - Commands are checked against blacklist/whitelist
  - Timeout enforced (default: 30 seconds)
  - Captures both stdout and stderr
```

**Use Cases**:
- Git operations: `git status`, `git diff`, `git commit`
- Package management: `npm install`, `pip install`
- Running tests: `pytest`, `npm test`
- Building projects: `npm run build`, `cargo build`

### 3. Semantic Code Search

#### semantic_search
Search indexed codebases using natural language.

```python
Parameters:
  - query (str): Natural language search query
  - index_name (str, optional): Index to search
  - n_results (int, optional): Number of results (max: 20)

Returns:
  - Relevant code chunks with:
    - File path and line numbers
    - Chunk type (function, class, etc.)
    - Similarity score
    - Code preview

Example Queries:
  - "authentication functions"
  - "error handling logic"
  - "database connection setup"
  - "API endpoint definitions"
```

---

## System Prompts

### Main System Prompt

The agent receives a comprehensive system prompt that includes:

1. **Capability Overview**: What the agent can do
2. **Current Context**: Working directory, available indexes
3. **Tool Descriptions**: All available tools with parameters
4. **Guidelines**:
   - Planning for complex tasks
   - Using semantic search before making changes
   - Reading files before editing
   - Testing changes when possible
   - Clear explanations and reasoning
5. **Best Practices**:
   - Use semantic search to understand existing patterns
   - Read complete files before editing
   - Test changes appropriately
   - Be precise with paths and syntax
   - Explain complex changes

### Error Recovery Prompt

When tools fail, the agent receives a recovery prompt:

```
The {tool_name} tool failed with: {error}

Original goal: {original_goal}

Please:
1. Try an alternative approach
2. Use different tools
3. Explain if unrecoverable

Focus on:
- Understanding why error occurred
- Finding workarounds
- Being creative with available tools
```

---

## Usage Examples

### Basic Setup

```python
from pathlib import Path
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.config import AgentConfig
from ctxai.agent.llm.anthropic_provider import AnthropicProvider
from ctxai.agent.tools.registry import ToolRegistry
from ctxai.agent.tools.file_ops import ReadFileTool, WriteFileTool, EditFileTool
from ctxai.agent.tools.bash_tool import BashTool
from ctxai.agent.tools.code_search import SemanticSearchTool

# Initialize configuration
agent_config = AgentConfig.get_default()

# Initialize LLM provider
llm_provider = AnthropicProvider(
    api_key=agent_config.llm.get_api_key_for_provider("anthropic"),
    model="claude-3-5-sonnet-20241022",
    temperature=0.7
)

# Initialize tool registry
tool_registry = ToolRegistry()
tool_registry.register_tool(ReadFileTool(max_file_size_mb=10))
tool_registry.register_tool(WriteFileTool())
tool_registry.register_tool(EditFileTool())
tool_registry.register_tool(BashTool(agent_config.tools))
tool_registry.register_tool(SemanticSearchTool(project_path=Path.cwd()))

# Create agent loop config
loop_config = AgentLoopConfig(
    llm_provider=llm_provider,
    tool_registry=tool_registry,
    agent_config=agent_config,
    working_directory=Path.cwd(),
    available_indexes=["my-project"],
    max_iterations=10,
    verbose=True
)

# Initialize agent
agent = Agent(loop_config)

# Process a message
response = await agent.process_message("Find authentication functions and add error handling")
print(response)
```

### Example Interactions

#### 1. Code Search and Modification

```
User: "Find authentication functions and add logging"

Agent Process:
1. Uses semantic_search tool with query "authentication functions"
2. Finds relevant code in src/auth.py:45-67
3. Uses read_file to read the function
4. Uses edit_file to add logging statements
5. Uses bash to run tests
6. Returns summary of changes
```

#### 2. Bug Fix with Testing

```
User: "Fix the bug in the checkout flow and verify with tests"

Agent Process:
1. Uses semantic_search with query "checkout flow bug"
2. Uses read_file to understand the code
3. Identifies the issue (missing validation)
4. Uses edit_file to add validation
5. Uses bash to run: pytest tests/test_checkout.py
6. Confirms tests pass and explains fix
```

#### 3. Feature Implementation

```
User: "Add a rate limiting middleware to the API"

Agent Process:
1. Uses semantic_search to find existing middleware patterns
2. Uses read_file to study the pattern
3. Uses write_file to create new middleware file
4. Uses edit_file to integrate into app
5. Uses bash to run the app and test
6. Returns implementation summary
```

---

## Tool Registry

The `ToolRegistry` manages all available tools and handles execution.

**Key Methods**:

```python
class ToolRegistry:
    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool for use by the agent."""

    def get_all_schemas(self, format: str = "anthropic") -> list:
        """Get schemas for all registered tools in specified format."""

    def get_tool_descriptions(self) -> str:
        """Get human-readable descriptions of all tools."""

    async def execute_tool(self, tool_name: str, **kwargs) -> dict:
        """Execute a tool by name with given parameters."""
```

**Supported Schema Formats**:
- `anthropic`: Anthropic's tool use format
- `openai`: OpenAI's function calling format
- `generic`: Generic JSON schema format

---

## LLM Providers

### Base Provider Interface

All LLM providers implement the `BaseLLMProvider` abstract class:

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: List[Message], tools: Optional[List[dict]] = None) -> LLMResponse:
        """Send messages and get response with optional tool calls."""

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get provider and model information."""
```

### Anthropic Provider

Full implementation with Claude models:

```python
from ctxai.agent.llm.anthropic_provider import AnthropicProvider

provider = AnthropicProvider(
    api_key="sk-ant-...",
    model="claude-3-5-sonnet-20241022",  # or opus, haiku
    temperature=0.7,
    max_tokens=4096
)

response = provider.chat(messages, tools=tool_schemas)
```

**Features**:
- Native tool use support
- Streaming responses (future)
- Full context window support (200K tokens for Sonnet)
- Proper handling of tool calls and results

---

## Message Types

The agent uses structured message types:

```python
@dataclass
class Message:
    role: MessageRole  # SYSTEM, USER, ASSISTANT, TOOL_RESULT
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None

@dataclass
class ToolCall:
    id: str  # Unique ID for tracking
    name: str  # Tool name
    parameters: dict  # Tool parameters

@dataclass
class LLMResponse:
    content: str
    tool_calls: List[ToolCall]
    model: str
    usage: Optional[Dict[str, int]] = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
```

---

## Conversation Context

Manages message history and token limits:

```python
class ConversationContext:
    def add_system_message(self, content: str) -> None
    def add_user_message(self, content: str) -> None
    def add_assistant_message(self, content: str, tool_calls: Optional[List[ToolCall]] = None) -> None
    def add_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> None

    def get_messages_for_llm(self) -> List[Message]
    def truncate_old_messages(self, max_messages: int = 50) -> None
    def get_message_count(self) -> int
    def get_token_count_estimate(self) -> int
    def clear(self) -> None
```

**Features**:
- Automatic message truncation to prevent context overflow
- Preserves system message always
- Token count estimation
- Proper tool call and result tracking

---

## Error Handling

The agent has robust error handling:

1. **Tool Execution Errors**:
   - Caught and returned as failed results
   - Fed back to LLM with error details
   - LLM can try alternative approaches

2. **LLM API Errors**:
   - Caught at agent loop level
   - Recovery prompt injected
   - Agent attempts to continue

3. **Timeout Protection**:
   - Max iterations limit (default: 10)
   - Bash command timeout (default: 30s)
   - LLM API timeout (default: 60s)

4. **Safety Checks**:
   - Bash command filtering
   - File size limits
   - Project directory restrictions

---

## Best Practices

### For Agent Users

1. **Be Specific**: Provide clear, detailed requests
2. **Provide Context**: Reference relevant files or locations
3. **Break Down Complex Tasks**: Split large tasks into steps
4. **Review Changes**: Always review agent modifications
5. **Use Verbose Mode**: Enable for debugging

### For Agent Developers

1. **Tool Design**: Make tools atomic and focused
2. **Error Messages**: Provide clear, actionable error messages
3. **Safety First**: Always validate inputs and paths
4. **Documentation**: Document tool parameters thoroughly
5. **Testing**: Test tools independently before integration

---

## Future Enhancements

### Planned Features

1. **Planning Phase**:
   - Multi-step planning before execution
   - User approval of plans
   - Plan visualization

2. **Additional LLM Providers**:
   - OpenAI GPT-4 support
   - Ollama for local models
   - Custom endpoint support

3. **More Tools**:
   - HTTP request tool for API testing
   - Database query tool
   - Docker operations tool
   - Web browsing tool

4. **Context Enhancements**:
   - Automatic context summarization
   - Better token management
   - Multi-project context

5. **Collaboration**:
   - Multi-agent coordination
   - Human-in-the-loop approval
   - Team workflows

---

## Performance Considerations

### Token Usage

- **System Prompt**: ~500 tokens
- **Tool Schemas**: ~200 tokens per tool
- **Message History**: Grows with conversation
- **Tool Results**: Variable (file contents can be large)

**Optimization Tips**:
- Use line ranges when reading large files
- Limit semantic search results (n_results parameter)
- Truncate context periodically
- Use concise tool descriptions

### Execution Speed

- **File Operations**: Fast (<100ms)
- **Bash Commands**: Variable (depends on command)
- **Semantic Search**: ~500ms (with local embeddings)
- **LLM Calls**: 2-5 seconds per call

**Optimization Tips**:
- Use faster models for simple tasks (haiku vs sonnet)
- Cache semantic search results when possible
- Batch file operations when appropriate
- Use async execution for independent tasks

---

## Troubleshooting

### Common Issues

**1. "Command blocked for safety"**
- Solution: Check `bash_blocked_commands` in config
- Add command to whitelist if needed
- Use alternative approach

**2. "Max iterations reached"**
- Solution: Increase `max_iterations` in config
- Break task into smaller steps
- Check for circular logic in agent behavior

**3. "File too large"**
- Solution: Increase `max_file_size_mb` in config
- Use line ranges to read portions
- Use grep/glob instead of reading full file

**4. "No indexes found"**
- Solution: Index your codebase first with `ctxai index`
- Verify index exists in `.ctxai/indexes/`
- Specify index_name explicitly in semantic_search

**5. "API key not found"**
- Solution: Set environment variable (ANTHROPIC_API_KEY, etc.)
- Pass api_key explicitly in config
- Check .env file is loaded

---

## Summary

The ctxai agent system provides a powerful foundation for building autonomous coding assistants. It combines:

- **Flexible LLM integration** with support for multiple providers
- **Rich tool ecosystem** for file operations, bash, and code search
- **Robust error handling** with automatic recovery
- **Safety features** to prevent destructive operations
- **Extensible architecture** for adding new tools and providers

The agent is designed to be both powerful for complex tasks and safe for production use, with comprehensive configuration options and safety checks throughout.

---

## File Locations

```
src/ctxai/agent/
├── core.py                      # Agent loop implementation
├── config.py                    # Configuration classes
├── context.py                   # Conversation context
├── prompts.py                   # System prompts
├── llm/
│   ├── base.py                  # Abstract LLM provider
│   └── anthropic_provider.py    # Claude integration
└── tools/
    ├── base.py                  # Tool base class
    ├── registry.py              # Tool registry
    ├── file_ops.py              # File operation tools
    ├── bash_tool.py             # Bash execution
    └── code_search.py           # Semantic search
```
