# Updates Summary - Multiple Embedding Providers, Size Limits & CTXAI_HOME

## Major Changes

### 1. Multiple Embedding Provider Support ✅

**Previously:** Only OpenAI embeddings (required API key)

**Now:** Multiple providers with local as default (no API key needed!)

#### Implemented Providers

- **Local (Default)** - sentence-transformers, works offline, free
- **OpenAI** - High quality, requires API key
- **HuggingFace** - Flexible, many models available

#### Architecture

Created a factory pattern with base class and provider implementations:

```python
# Base class
class BaseEmbeddingProvider(ABC):
    def generate_embeddings(texts: List[str]) -> List[List[float]]
    def generate_embedding(text: str) -> List[float]
    def get_dimension() -> int

# Factory
EmbeddingsFactory.create(config) -> BaseEmbeddingProvider
```

### 2. Configuration System ✅

**New file:** `src/ctxai/config.py`

Manages `.ctxai/config.json` for:
- Embedding provider selection
- Model configuration
- API keys (optional)
- Project size limits
- Chunking parameters

**Default config:**
```json
{
  "version": "1.0",
  "embedding": {
    "provider": "local",
    "model": null,
    "batch_size": 100
  },
  "indexing": {
    "max_files": 10000,
    "max_total_size_mb": 500,
    "max_file_size_mb": 5,
    "chunk_size": 1000,
    "chunk_overlap": 100
  }
}
```

### 3. Project Size Validation ✅

**New file:** `src/ctxai/size_validator.py`

Prevents indexing overly large projects with:
- File count limits
- Total size limits
- Individual file size limits
- Warnings at 80% of limits
- Hard errors when limits exceeded
- Detailed statistics and summaries

**Features:**
- ⚠️ Warnings at 80% of limit
- ❌ Hard stop when limit exceeded
- 📊 Project statistics (file count, size, largest files)
- 🔍 Identifies oversized files

### 4. Updated Index Command

**Modified:** `src/ctxai/commands/index_command.py`

Now includes:
1. Configuration loading from `.ctxai/config.json`
2. Dynamic embedding provider initialization
3. Size validation before indexing
4. Better error messages with solutions
5. Shows which provider is being used
6. Filters out oversized files automatically
7. **CTXAI_HOME support** - Uses global or project-specific .ctxai

### 5. CTXAI_HOME Environment Variable ✅

**New feature:** Control where `.ctxai` directory is located

**Usage:**
```bash
# Global .ctxai for all projects
export CTXAI_HOME=~/.ctxai

# Or custom location
export CTXAI_HOME=/path/to/.ctxai

# Default (no env var): uses project/.ctxai
```

**Benefits:**
- ✅ Share configuration across projects
- ✅ Centralize all indexes
- ✅ Easier backup and management
- ✅ Team collaboration

**Implementation:**
- New `utils.py` module with path resolution
- Priority: CTXAI_HOME → project path → current directory
- All modules updated to respect CTXAI_HOME
- Backwards compatible (works without env var)

## Files Modified

### Core Modules

1. **`src/ctxai/embeddings.py`** - Complete rewrite
   - Base class for all providers
   - Local, OpenAI, HuggingFace implementations
   - Factory pattern for provider creation
   - Extensible for custom providers

2. **`src/ctxai/commands/index_command.py`** - Major updates
   - Load config before indexing
   - Create embedding provider from config
   - Validate project size
   - Show provider information
   - Better error handling

3. **`src/ctxai/__init__.py`** - Updated exports
   - Export new classes
   - Updated documentation

### New Modules

4. **`src/ctxai/config.py`** - NEW
   - Configuration data classes
   - ConfigManager for loading/saving
   - Helper methods for updates
   - CTXAI_HOME support

5. **`src/ctxai/size_validator.py`** - NEW
   - ProjectStats dataclass
   - ProjectSizeValidator class
   - ProjectSizeLimitError exception
   - Human-readable size formatting

6. **`src/ctxai/utils.py`** - NEW
   - CTXAI_HOME environment variable handling
   - Path resolution utilities
   - get_ctxai_home(), get_indexes_dir(), etc.
   - Helper functions for config location

### Documentation

7. **`README.md`** - Major updates
   - Updated Quick Start (no API key needed!)
   - Added configuration section
   - Multiple provider examples
   - Project size limit documentation
   - **CTXAI_HOME documentation**
   - Updated troubleshooting

8. **`CONFIGURATION.md`** - NEW
   - Complete configuration guide
   - All provider options
   - Size limit explanations
   - **CTXAI_HOME usage and benefits**
   - Best practices
   - Cost estimation

9. **`.env.example`** - Updated
   - Added CTXAI_HOME example
   - Added HuggingFace API key
   - Clarified what's optional
   - Better organization

10. **`examples/example_usage.py`** - Updated
    - Local embedding examples
    - Config-based examples
    - Multiple provider demos
    - **CTXAI_HOME usage example**

### Dependencies

10. **`pyproject.toml`** - Updated
    - Added `sentence-transformers` (local embeddings)
    - Made `openai` optional: `pip install ctxai[openai]`
    - Made `requests` optional: `pip install ctxai[huggingface]`
    - Added `[all]` option for everything

## Migration Guide

### For Existing Users

If you were using OpenAI embeddings before:

**Option 1: Switch to local (recommended for testing)**
```bash
# Just run - will use local embeddings by default
ctxai index ./project "my-index"
```

**Option 2: Continue with OpenAI**
```bash
# Edit .ctxai/config.json:
{
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-small"
  }
}

# Keep your OPENAI_API_KEY set
export OPENAI_API_KEY=your-key
```

### For New Users

Just install and run - no API key needed!

```bash
pip install ctxai
ctxai index ./project "my-index"

# Optional: Use global .ctxai directory
export CTXAI_HOME=~/.ctxai
```

## Backward Compatibility

- ✅ Existing indexes work fine
- ✅ Config auto-created if missing
- ✅ Environment variables still work
- ✅ Command-line interface unchanged
- ✅ Works without CTXAI_HOME (uses project/.ctxai)
- ⚠️ Must install provider dependencies:
  - Local: Auto-installed
  - OpenAI: `pip install ctxai[openai]`
  - HuggingFace: `pip install ctxai[huggingface]`

## Benefits

### 1. **No Barrier to Entry**
- Works out of the box
- No API key registration required
- No costs for exploration/testing

### 2. **Privacy**
- Local embeddings = code stays local
- No data sent to third parties
- Perfect for sensitive codebases

### 3. **Flexibility**
- Choose provider based on needs
- Switch providers without re-indexing
- Add custom providers easily

### 4. **Cost Control**
- Local embeddings are free
- Size limits prevent surprises
- Clear cost estimation in docs

### 5. **Better UX**
- Clear errors with solutions
- Progress indicators
- Warnings before failures
- Configuration discovery
- Shows CTXAI_HOME location

### 6. **Team Collaboration**
- Share configuration via CTXAI_HOME
- Centralized index management
- Consistent settings across projects
- Easier onboarding

## Future Enhancements (TODO)

As marked in the code, we still need to design strategies for:

1. **Large codebase handling**
   - Incremental indexing
   - Smart sampling
   - Parallel processing
   - Streaming for huge files

2. **Additional providers**
   - Azure OpenAI
   - Cohere
   - Anthropic (when available)
   - Local LLMs (Ollama, etc.)

3. **Advanced features**
   - Multiple indexes per project
   - Index merging
   - Selective re-indexing
   - Cache layer for embeddings

## Testing Checklist

- [x] Local embeddings work without API key
- [x] OpenAI embeddings work with API key
- [x] Config file auto-creation
- [x] Size validation warnings
- [x] Size validation errors
- [x] Oversized file filtering
- [x] Provider switching
- [x] Error messages are helpful
- [x] Documentation is complete
- [x] Examples are updated

## Summary

This update transforms ctxai from an OpenAI-only tool to a flexible, multi-provider semantic search engine with:

✅ **Local-first** - Works without API keys
✅ **Configurable** - Easy provider switching
✅ **Safe** - Size limits prevent accidents  
✅ **Flexible** - Extensible provider system
✅ **User-friendly** - Better errors and guidance

The tool is now more accessible to everyone while maintaining the power and flexibility needed for production use!
