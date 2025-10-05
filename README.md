# codecontext

Intelligent semantic search across your entire codebase

Transform your code into searchable embeddings with advanced chunking and vector database indexing

An advanced code search engine leveraging large language models to comprehend the context and intent of your queries. Quickly locate relevant code snippets, documentation, and examples within large codebases using intelligent semantic understanding.


## Usage

- Indexing of the code
```bash
python -m codecontext.index /path/to/codebase "index_name"
```

- Usage
Create mcp.json file with as below:
```json
{
    "inputs": [],
    "servers": {	
        "calculator": {
            "command": "python",
            "args": [
                "-m",
                "code_context.start",
                "--index",
                "index_name"
            ],
        }
    }
}
```

- Use Github Copilot's Agent mode to use this as below:
```
/code_context "Find the code related to Profile Image update"
```

---

## Installation

Pre-requisites:

- Python 3.10+

```bash
uvx codecontext
```

## Running

- copy `.env.example` to `.env` and fill in the values

```bash
codepilot
```

### Development Notes

```bash

codecontext server
codecontext playgruond
codecontext shell






perfect

autogen

pydentic ai

memory

spec kit

agent-framework


```

## Contributing

We welcome all contributions to the project! Before submitting your pull request, please ensure you have run the tests and linters locally. This helps us maintain the quality of the project and makes the review process faster for everyone.

All contributions should adhere to the project's code of conduct. Let's work together to create a welcoming and inclusive environment for everyone.