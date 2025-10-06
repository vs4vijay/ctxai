# Configuration Guide

## Overview

ctxai uses a configuration file `.ctxai/config.json` to manage embedding providers, project size limits, and other settings. The location of `.ctxai` can be customized using the `CTXAI_HOME` environment variable.

## CTXAI_HOME Environment Variable

### What is CTXAI_HOME?

`CTXAI_HOME` is an environment variable that specifies where ctxai stores its configuration and indexes. If not set, ctxai uses a `.ctxai` directory in your project.

### Usage

```bash
# Use a global .ctxai directory (shared across all projects)
export CTXAI_HOME=~/.ctxai

# Or use a custom location
export CTXAI_HOME=/path/to/my/.ctxai

# Windows
set CTXAI_HOME=C:\Users\YourName\.ctxai

# Windows PowerShell
$env:CTXAI_HOME="C:\Users\YourName\.ctxai"
```

### Priority

ctxai determines the `.ctxai` location in this order:

1. **CTXAI_HOME environment variable** (if set)
2. **Project directory** (default): `project/.ctxai`

### Benefits

**Global Configuration:**
- ✅ Share settings across all projects
- ✅ One place to configure embedding providers
- ✅ Centralized API key management

**Centralized Indexes:**
- ✅ All indexes in one location
- ✅ Easier backup and management
- ✅ Better disk space control

**Use Cases:**

1. **Individual Developer** - Global config, project-specific indexes:
   ```bash
   export CTXAI_HOME=~/.ctxai
   # Config shared, but each project has its own index
   ```

2. **Team Environment** - Shared config location:
   ```bash
   export CTXAI_HOME=/shared/team/.ctxai
   # Team shares configuration
   ```

3. **Per-Project** - Different config per project (default):
   ```bash
   # Don't set CTXAI_HOME
   # Each project has its own .ctxai directory
   ```

### Directory Structure

**With CTXAI_HOME:**
```
~/.ctxai/                    # Global location
├── config.json              # Shared configuration
└── indexes/
    ├── project1-index/      # Index for project 1
    └── project2-index/      # Index for project 2
```

**Without CTXAI_HOME (default):**
```
project1/
├── .ctxai/
│   ├── config.json          # Project 1 config
│   └── indexes/
│       └── project1-index/

project2/
├── .ctxai/
│   ├── config.json          # Project 2 config
│   └── indexes/
│       └── project2-index/
```

## Default Configuration

On first run, ctxai creates a default configuration:

```json
{
  "version": "1.0",
  "embedding": {
    "provider": "local",
    "model": null,
    "api_key": null,
    "batch_size": 100,
    "max_tokens": null
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

## Embedding Configuration

### Local Embeddings (Default)

**No API key required!** Uses sentence-transformers locally.

```json
{
  "embedding": {
    "provider": "local",
    "model": "all-MiniLM-L6-v2",
    "batch_size": 100
  }
}
```

**Available models:**
- `all-MiniLM-L6-v2` - Fast, good quality (default)
- `all-mpnet-base-v2` - Better quality, slower
- `paraphrase-MiniLM-L6-v2` - Good for paraphrasing

**Pros:**
- ✅ Free and private
- ✅ Works offline after first download
- ✅ No rate limits

**Cons:**
- ❌ Slower than cloud APIs
- ❌ Downloads model on first use (~80MB)

### OpenAI Embeddings

High-quality embeddings from OpenAI's API.

```json
{
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "api_key": "sk-...",
    "batch_size": 100
  }
}
```

**Available models:**
- `text-embedding-3-small` - Cheaper, 1536 dimensions
- `text-embedding-3-large` - Best quality, 3072 dimensions
- `text-embedding-ada-002` - Legacy, 1536 dimensions

**Environment variable:**
```bash
export OPENAI_API_KEY=your-key-here
```

**Pros:**
- ✅ High quality embeddings
- ✅ Fast API
- ✅ Optimized for code

**Cons:**
- ❌ Costs money ($0.02-0.13 per 1M tokens)
- ❌ Requires internet
- ❌ Data sent to OpenAI

### HuggingFace Embeddings

Use HuggingFace's Inference API.

```json
{
  "embedding": {
    "provider": "huggingface",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "api_key": "hf_...",
    "batch_size": 50
  }
}
```

**Environment variable:**
```bash
export HUGGINGFACE_API_KEY=your-key-here
```

**Pros:**
- ✅ Many models available
- ✅ Free tier available
- ✅ Open source models

**Cons:**
- ❌ Can be slower
- ❌ Rate limits on free tier

## Indexing Configuration

### File Limits

Control how many files can be indexed:

```json
{
  "indexing": {
    "max_files": 10000,
    "max_total_size_mb": 500,
    "max_file_size_mb": 5
  }
}
```

**Options:**
- `max_files` - Maximum number of files to index
- `max_total_size_mb` - Maximum total project size in MB
- `max_file_size_mb` - Maximum individual file size in MB

**Why limits?**
- Prevent accidentally indexing huge projects
- Control embedding costs (for cloud providers)
- Ensure reasonable performance
- Files exceeding `max_file_size_mb` are skipped with a warning

### Chunking Configuration

Control how code is split into chunks:

```json
{
  "indexing": {
    "chunk_size": 1000,
    "chunk_overlap": 100
  }
}
```

**Options:**
- `chunk_size` - Maximum characters per chunk
- `chunk_overlap` - Characters to overlap between chunks

**Guidelines:**
- Smaller chunks: More precise search, more chunks to store
- Larger chunks: More context, fewer chunks
- Overlap: Ensures important code at boundaries isn't lost
- Default (1000/100) works well for most projects

## Warnings and Errors

### 80% Warning

When approaching limits (80%+), you'll see warnings:

```
⚠️  Approaching file limit: 8500 files (limit: 10000)
```

You can continue, but consider:
- Using `--include` patterns to filter
- Increasing limits in config
- Splitting into multiple indexes

### Hard Limits

Indexing stops if limits are exceeded:

```
❌ Too many files: 12000 files (limit: 10000)
   Consider using --include patterns to filter files,
   or increase max_files in .ctxai/config.json
```

**To fix:**
1. Filter files: `ctxai index . "name" --include "*.py"`
2. Edit `.ctxai/config.json` and increase limits
3. Exclude directories: `--exclude "node_modules/*"`

### Oversized Files

Files larger than `max_file_size_mb` are skipped:

```
⚠️  Found 3 file(s) exceeding 5 MB (these will be skipped):
   • large_data.json: 12.5 MB
   • bundle.min.js: 8.2 MB
```

These files won't be indexed but won't block the process.

## Configuration Location

Configuration is stored based on CTXAI_HOME:

**With CTXAI_HOME set:**
```bash
export CTXAI_HOME=~/.ctxai
# Configuration: ~/.ctxai/config.json
# Indexes: ~/.ctxai/indexes/<name>/
```

**Without CTXAI_HOME (default):**
```
your-project/
├── .ctxai/
│   ├── config.json          # Configuration
│   └── indexes/
│       └── my-index/        # Vector database
├── your-code.py
└── ...
```

### Checking CTXAI_HOME

From Python:
```python
from ctxai.utils import get_ctxai_home_info

info = get_ctxai_home_info()
print(f"Using global CTXAI_HOME: {info['is_global']}")
print(f"Location: {info['resolved_path']}")
```

From CLI (coming soon):
```bash
ctxai config show
```

## Programmatic Configuration

You can also configure ctxai programmatically:

```python
from pathlib import Path
from ctxai.config import ConfigManager, EmbeddingConfig

# Load configuration
config_manager = ConfigManager(Path(".ctxai"))
config = config_manager.load()

# Change provider
config.embedding.provider = "openai"
config.embedding.model = "text-embedding-3-small"

# Change limits
config.indexing.max_files = 20000
config.indexing.max_total_size_mb = 1000

# Save
config_manager.save(config)
```

## Best Practices

1. **Start with local embeddings** - Free and works offline
2. **Use OpenAI for production** - Better quality results
3. **Set appropriate limits** - Prevent accidents and control costs
4. **Use include patterns** - Index only what you need
5. **Monitor warnings** - Approaching limits? Time to review
6. **Keep API keys in environment** - Don't commit them to git
7. **Use CTXAI_HOME for teams** - Share configuration across developers
8. **Project-specific for experiments** - Don't use CTXAI_HOME when testing

## Environment Variable Summary

```bash
# CTXAI configuration
export CTXAI_HOME=~/.ctxai              # Optional: global .ctxai location

# Embedding providers (optional - based on config)
export OPENAI_API_KEY=sk-...            # For OpenAI provider
export HUGGINGFACE_API_KEY=hf_...       # For HuggingFace provider

# Application settings (future use)
export PORT=8000                         # MCP server port
```

## Cost Estimation (OpenAI)

For a typical project:
- 1000 files × 500 lines = 500K lines
- ~100 characters per line = 50M characters
- ~12.5M tokens
- Cost: ~$0.25 (with text-embedding-3-small)

Adjust limits to control costs!
