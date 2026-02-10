# ctxai Quick Start Guide

Get started with ctxai AI coding agent in 5 minutes!

## 🚀 Installation

```bash
# Clone the repo (if not already done)
git clone https://github.com/your-username/ctxai.git
cd ctxai

# Install with all dependencies
pip install -e ".[all]"

# Or use uv (faster)
uv pip install -e ".[all]"
```

## 🔑 Setup (Choose One)

### Option 1: OpenRouter (Recommended)

Access 100+ models through one API:

```bash
# Get API key at: https://openrouter.ai/keys
export OPENROUTER_API_KEY=your-key-here

# Test it
python scripts/setup_providers.py --test openrouter
```

### Option 2: Ollama (Local & Free)

Run models locally (no API key needed):

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh
# Or download from: https://ollama.ai

# Start Ollama
ollama serve

# Pull a model
ollama pull codellama:13b

# Test it
python scripts/setup_providers.py --test ollama
```

### Option 3: Direct Provider

Use Anthropic or OpenAI directly:

```bash
# Anthropic
export ANTHROPIC_API_KEY=your-key-here

# OpenAI
export OPENAI_API_KEY=your-key-here
```

## ✅ Verify Setup

```bash
# Check all providers
python scripts/setup_providers.py

# You should see:
#   ✓ OpenRouter: configured  (if you set the key)
#   ✓ Ollama: running        (if installed)
#   etc.
```

## 🎯 First Commands

### 1. Interactive Chat

```bash
# With OpenRouter (100+ models)
ctxai chat --provider openrouter

# With local Ollama (free!)
ctxai chat --provider ollama --model codellama:13b

# Specify exact model
ctxai chat --provider openrouter --model anthropic/claude-3.5-sonnet
```

### 2. Architect/Editor Pattern (40-60% cost savings!)

```bash
# Use default preset (o1-mini + Claude Sonnet)
ctxai chat --architect-editor

# Budget preset (GPT-4o + GPT-4o-mini)
ctxai chat --architect-editor --preset budget

# Local preset (fully free!)
ctxai chat --architect-editor --preset local

# Mixed preset (cloud planning + local implementation)
ctxai chat --architect-editor --preset mixed
```

### 3. One-Shot Tasks

```bash
# Execute a coding task
ctxai code "Create a Python function to validate email addresses"

# With specific provider
ctxai code "Add error handling to main.py" --provider openrouter
```

## 📝 Example Session

```bash
$ ctxai chat --architect-editor --preset default

╔═══════════════════════════════════════════════════════╗
║              ctxai - AI Coding Agent                 ║
║                                                       ║
║  Your autonomous coding assistant powered by AI       ║
╚═══════════════════════════════════════════════════════╝

Provider Status:
  ✓ OpenRouter: configured
  ✓ Ollama: running

Using preset: default
  Description: Best quality + cost balance
  Cost: $$
  Architect: openai/o1-mini
  Editor: anthropic/claude-3.5-sonnet

🤖 Agent ready with 8 tools
✓ Working directory: /home/user/project
✓ Repository map created

You: List all Python files in the src directory

Agent: [Uses list_files tool]

I found 15 Python files in the src directory:
- src/__init__.py
- src/agent/core.py
- src/agent/tools/base.py
...

You: Create a function to validate email addresses with regex

Agent: [Uses write_file tool]

I've created a new file src/validators.py with an email validation function...
```

## 🎓 Available Presets

| Preset | Architect | Editor | Cost | Description |
|--------|-----------|--------|------|-------------|
| **default** | o1-mini | Claude Sonnet | $$ | Best balance |
| **premium** | o1 | Claude Opus | $$$$$ | Best quality |
| **budget** | GPT-4o | GPT-4o-mini | $ | Lower cost |
| **cheap** | DeepSeek R1 | DeepSeek Chat | ¢ | Cheapest |
| **local** | CodeLlama 34B | CodeLlama 13B | Free | Fully local |
| **mixed** | o1-mini | CodeLlama 13B | $ | Cloud + local |

## 🛠️ Common Tasks

### Code Generation

```bash
You: Write a FastAPI endpoint to handle user authentication
```

### Code Review

```bash
You: Review the code in src/auth.py and suggest improvements
```

### Debugging

```bash
You: There's a bug in the login function, help me debug it
```

### Refactoring

```bash
You: Refactor the UserService class to use dependency injection
```

### Documentation

```bash
You: Add docstrings to all functions in src/utils.py
```

### Testing

```bash
You: Write pytest tests for the User class
```

## 💡 Tips

1. **Use Architect/Editor**: Save 40-60% on costs with same or better quality
2. **Try Local First**: Use Ollama for privacy and zero cost
3. **Mixed Preset**: Best of both worlds (cloud planning + local execution)
4. **Repository Map**: Enabled by default, gives agent context about your codebase
5. **Commands**: Type `/help` in chat to see available commands

## 🔧 Advanced Usage

### Custom Architect/Editor

```bash
ctxai chat --architect-editor \
  --architect-model openai/o1-mini \
  --editor-model anthropic/claude-3.5-sonnet
```

### Disable Repository Mapping

```bash
ctxai chat --no-repomap
```

### Verbose Mode

```bash
ctxai chat --verbose
```

### Different Working Directory

```bash
ctxai chat --working-directory /path/to/project
```

## 📚 Next Steps

1. ✅ Run interactive chat
2. ✅ Try architect/editor pattern
3. 🔜 Read [AI_AGENT.md](AI_AGENT.md) for detailed documentation
4. 🔜 Read [CODING.md](CODING.md) for architecture details
5. 🔜 Run examples: `python examples/architect_editor_example.py`

## ❓ Troubleshooting

### "OPENROUTER_API_KEY not set"

```bash
export OPENROUTER_API_KEY=your-key-here
```

### "Ollama not running"

```bash
# Start Ollama in a separate terminal
ollama serve

# Or run in background
ollama serve &
```

### "Model not found"

```bash
# Pull the model first
ollama pull codellama:13b
```

### "No such provider"

Check the provider name:
- ✅ `openrouter` (not `open_router`)
- ✅ `ollama` (not `olama`)
- ✅ `anthropic` (not `claude`)
- ✅ `openai` (not `gpt`)

## 🆘 Get Help

```bash
# Check provider status
python scripts/setup_providers.py

# Get help with command
ctxai chat --help

# In chat, type:
/help
```

---

**Ready to code with AI?** 🚀

```bash
ctxai chat --architect-editor
```
