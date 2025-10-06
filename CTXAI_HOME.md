# CTXAI_HOME Feature Summary

## Overview

Added `CTXAI_HOME` environment variable to allow users to control where the `.ctxai` directory is located. This enables both global and project-specific configurations.

## Implementation

### New Module: `utils.py`

Created utilities module with path resolution functions:

```python
def get_ctxai_home(project_path: Optional[Path] = None) -> Path
def get_indexes_dir(project_path: Optional[Path] = None) -> Path
def get_config_path(project_path: Optional[Path] = None) -> Path
def ensure_ctxai_home(project_path: Optional[Path] = None) -> Path
def is_using_global_home() -> bool
def get_ctxai_home_info() -> dict
```

### Path Resolution Priority

1. **CTXAI_HOME environment variable** (if set) - Highest priority
2. **project_path/.ctxai** (if project_path provided)
3. **current_directory/.ctxai** (default fallback)

### Updated Modules

All modules that reference `.ctxai` now use the utility functions:

- `config.py` - ConfigManager uses `get_ctxai_home()`
- `commands/index_command.py` - Uses `get_ctxai_home()` and `get_indexes_dir()`
- `examples/example_usage.py` - Updated all examples

## Usage

### Global .ctxai Directory

Share configuration across all projects:

```bash
# Set globally in your shell profile
export CTXAI_HOME=~/.ctxai

# Now all projects use the same configuration
ctxai index /project1 "project1"
ctxai index /project2 "project2"

# Both use: ~/.ctxai/config.json
# Indexes: ~/.ctxai/indexes/project1/ and ~/.ctxai/indexes/project2/
```

### Custom Location

Use any custom directory:

```bash
export CTXAI_HOME=/shared/team/.ctxai

# Team shares the same configuration
```

### Project-Specific (Default)

Don't set CTXAI_HOME for project-specific configs:

```bash
# No CTXAI_HOME set
ctxai index /project "my-index"

# Uses: /project/.ctxai/config.json
# Index: /project/.ctxai/indexes/my-index/
```

## Benefits

### 1. Centralized Configuration

One configuration file for all projects:
- Embedding provider settings
- Project size limits
- Chunking parameters
- API keys (via environment or config)

### 2. Centralized Index Management

All indexes in one location:
- Easy backup: just backup `~/.ctxai/indexes/`
- Better disk space control
- Simplified maintenance

### 3. Team Collaboration

Share configuration across team:
```bash
# In team's shared directory
export CTXAI_HOME=/shared/ctxai
```

Everyone uses the same:
- Embedding provider
- Size limits
- Best practices

### 4. Flexibility

Choose the right approach per use case:
- **Individual dev**: Global config, centralized indexes
- **Team**: Shared config location
- **Per-project**: Independent configs (default)
- **Testing**: Don't set CTXAI_HOME

## Use Cases

### Use Case 1: Individual Developer

```bash
# Set once in ~/.bashrc or ~/.zshrc
export CTXAI_HOME=~/.ctxai

# Benefits:
# - One configuration for all projects
# - All indexes in one place
# - Easy to backup ~/.ctxai
```

### Use Case 2: Team Development

```bash
# Team shares a directory
export CTXAI_HOME=/shared/team/.ctxai

# Benefits:
# - Consistent embedding provider across team
# - Shared configuration and limits
# - Easier onboarding (just set env var)
```

### Use Case 3: CI/CD Pipeline

```bash
# Use project-specific (don't set CTXAI_HOME)
# Each project build is isolated

# Or use a build cache location
export CTXAI_HOME=/build/cache/.ctxai
```

### Use Case 4: Experimentation

```bash
# Don't set CTXAI_HOME
# Each experimental project is independent
```

## Directory Structure Examples

### With CTXAI_HOME

```
~/.ctxai/                     # Global location
├── config.json               # Shared configuration
└── indexes/
    ├── project1-backend/     # Index for project 1
    ├── project2-frontend/    # Index for project 2
    └── project3-ml/          # Index for project 3

/code/project1/               # Project 1
└── src/

/code/project2/               # Project 2  
└── src/
```

### Without CTXAI_HOME (Default)

```
/code/project1/
├── .ctxai/
│   ├── config.json           # Project 1 specific
│   └── indexes/
│       └── project1-index/
└── src/

/code/project2/
├── .ctxai/
│   ├── config.json           # Project 2 specific
│   └── indexes/
│       └── project2-index/
└── src/
```

## Checking CTXAI_HOME

### From Python

```python
from ctxai.utils import get_ctxai_home_info

info = get_ctxai_home_info()
print(f"Has CTXAI_HOME env var: {info['has_env_var']}")
print(f"Environment value: {info['env_value']}")
print(f"Resolved path: {info['resolved_path']}")
print(f"Using global: {info['is_global']}")
```

### From CLI

The index command shows the location:

```bash
ctxai index ./project "my-index"

# Output includes:
# Using global CTXAI_HOME: /home/user/.ctxai
# or
# Using project .ctxai: /home/user/project/.ctxai
```

## Migration

### Migrating to Global CTXAI_HOME

If you have existing project-specific `.ctxai` directories:

```bash
# 1. Set CTXAI_HOME
export CTXAI_HOME=~/.ctxai

# 2. (Optional) Move existing config
cp /old/project/.ctxai/config.json ~/.ctxai/config.json

# 3. (Optional) Move indexes
mv /old/project/.ctxai/indexes/* ~/.ctxai/indexes/

# 4. Re-index if needed
ctxai index /old/project "project-name"
```

### Migrating to Project-Specific

From global to project-specific:

```bash
# 1. Unset CTXAI_HOME
unset CTXAI_HOME

# 2. Re-index (will use project/.ctxai)
ctxai index ./project "project-name"
```

## API Reference

### get_ctxai_home(project_path)

Get the .ctxai directory path with priority resolution.

**Parameters:**
- `project_path` (Optional[Path]): Project root path

**Returns:**
- Path: Resolved .ctxai directory

**Examples:**
```python
# With CTXAI_HOME set
os.environ["CTXAI_HOME"] = "/global/.ctxai"
get_ctxai_home()  # Returns: Path('/global/.ctxai')

# With project path
get_ctxai_home(Path("/project"))  # Returns: Path('/project/.ctxai')
# Unless CTXAI_HOME is set, then returns that

# Default
get_ctxai_home()  # Returns: Path('./ctxai')
```

### get_indexes_dir(project_path)

Get the indexes directory.

**Returns:**
- Path: Indexes directory (CTXAI_HOME/indexes or project/.ctxai/indexes)

### get_config_path(project_path)

Get the config file path.

**Returns:**
- Path: Config file path (CTXAI_HOME/config.json or project/.ctxai/config.json)

### is_using_global_home()

Check if using global CTXAI_HOME.

**Returns:**
- bool: True if CTXAI_HOME environment variable is set

### get_ctxai_home_info()

Get detailed information about CTXAI_HOME configuration.

**Returns:**
- dict: Information including env var status, value, resolved path, etc.

## Best Practices

1. **Set CTXAI_HOME in shell profile** - Persistent configuration
   ```bash
   # In ~/.bashrc or ~/.zshrc
   export CTXAI_HOME=~/.ctxai
   ```

2. **Use global for personal projects** - Easier management

3. **Use project-specific for shared repos** - Avoid confusion

4. **Document team setup** - If using shared CTXAI_HOME

5. **Backup global .ctxai** - All your indexes in one place

6. **Use .gitignore** - Exclude `.ctxai` from version control
   ```
   .ctxai/
   ```

## Environment Variables Summary

```bash
# CTXAI configuration
export CTXAI_HOME=~/.ctxai              # Optional: global .ctxai location

# Embedding providers (based on config)
export OPENAI_API_KEY=sk-...            # For OpenAI provider
export HUGGINGFACE_API_KEY=hf_...       # For HuggingFace provider
```

## Backward Compatibility

✅ **Fully backward compatible**
- Works without CTXAI_HOME (uses project/.ctxai)
- Existing projects continue to work
- No breaking changes
- Optional feature

## Summary

The CTXAI_HOME feature provides flexibility in managing ctxai configuration and indexes:

- ✅ **Global configuration** - One config for all projects
- ✅ **Centralized indexes** - All indexes in one location
- ✅ **Team collaboration** - Shared configuration
- ✅ **Flexible** - Choose global or project-specific
- ✅ **Backward compatible** - Works without CTXAI_HOME
- ✅ **Easy migration** - Simple to switch between modes

Perfect for both individual developers and teams! 🎉
