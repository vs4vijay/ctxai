# Provider compatibility

Generated from `ctxai.agent.llm.contract.PROVIDER_SPECS`.

| Provider | Boundary | Transport | Tools | Streaming | Models |
|---|---|---|---:|---:|---|
| anthropic | cloud | Anthropic Messages | yes | yes | API/static |
| openai | cloud | OpenAI Chat Completions | yes | yes | API/static |
| openrouter | cloud | OpenAI-compatible | yes | yes | API/cached |
| github-copilot | cloud | Copilot Chat | yes | no | API/cached |
| ollama | local | Ollama Chat | yes | no | dynamic/local |
| custom | cloud | OpenAI-compatible | yes | no | endpoint-defined |
| nvidia | cloud | OpenAI-compatible | yes | no | endpoint-defined |

Fallback is disabled by default. Crossing the local/cloud boundary requires the explicit `allow_fallback_boundary_crossing` setting.
