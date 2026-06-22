# Claude Code Proxy

Routes Claude Code requests to any OpenAI-compatible endpoint via LiteLLM.

## Quick Start

```bash
# Clone
git clone https://github.com/adrianR84/claude-code-proxy.git
cd claude-code-proxy

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Configure
cp config.example.json config.json
# Edit config.json with your API keys and provider settings

# Run
./run.sh        # dev (with --reload)
./run.sh prod   # production

# Or manually:
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload  # dev
uv run uvicorn server:app --host 0.0.0.0 --port 8082            # prod

# Unix users can also use:
make run        # dev
make run-prod   # prod
```

## Configure Claude Code

**Configuration is done in `config.json` — see examples below.**

```bash
# Standard
ANTHROPIC_BASE_URL=http://localhost:8082 claude

# If not authorized (use placeholder token):
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_AUTH_TOKEN="sk-ant-api03-placeholder" claude
```

## Configuration

All settings are in `config.json`. Copy `config.example.json` to `config.json` and edit.

### Structure

```json
{
  "preferred_provider": "openai",
  "big_model": "gpt-5.5",
  "small_model": "gpt-5.4-mini",
  "providers": {
    "anthropic": { "api_key": null },
    "openai": { "api_key": null, "base_url": null },
    "google": { "api_key": null },
    "azure": { "api_key": null, "base_url": null, "api_version": "2024-06-01" },
    "vertex": { "project": null, "location": null, "use_auth": false },
    "custom_1": { "model": null, "api_key": null, "base_url": null },
    "custom_2": { "model": null, "api_key": null, "base_url": null },
    "custom_3": { "model": null, "api_key": null, "base_url": null }
  }
}
```

### Switching Providers

Set `preferred_provider` to `openai`, `google`, `anthropic`, `azure`, `vertex`, `custom_1`, `custom_2`, or `custom_3`.

| Provider | Description |
|----------|-------------|
| `openai` | Default. Routes to OpenAI. haiku/sonnet/opus map to `small_model`/`big_model` |
| `google` | Routes to Gemini. haiku/sonnet/opus map to Gemini models |
| `anthropic` | Pass-through to Anthropic. No remapping |
| `azure` | Routes to Azure OpenAI |
| `vertex` | Routes to Google Vertex AI (uses GCP credentials, not API key) |
| `custom_1/2/3` | OpenAI-compatible custom endpoint |

### Custom Providers

Custom providers (`custom_1`, `custom_2`, `custom_3`) are OpenAI-compatible endpoints:

- **`model` set**: Every request (haiku, sonnet, opus) routes to the specified model
- **`model` null/empty**: Routes via `big_model`/`small_model` mapping (haiku → `small_model`, sonnet/opus → `big_model`)

```json
"custom_1": { "model": "qwen/qwen3-235b-a22b", "api_key": "key", "base_url": "https://..." }
```

```json
"custom_2": { "model": null, "api_key": "key", "base_url": "https://..." }
```

### Full Example

```json
{
  "preferred_provider": "custom_1",
  "big_model": "gpt-5.5",
  "small_model": "gpt-5.4-mini",
  "providers": {
    "openai": { "api_key": "sk-..." },
    "custom_1": { "model": "qwen/qwen3-235b-a22b", "api_key": "auton_sk_...", "base_url": "https://api.autonaisol.xyz/api/v1/gateway/v1" }
  }
}
```

## Model Mapping

| Claude Model | Maps To |
|--------------|---------|
| haiku | `small_model` (gpt-5.4-mini default) |
| sonnet | `big_model` (gpt-5.5 default) |
| opus | `big_model` (gpt-5.5 default) |

---

Original creator: [1rgs](https://github.com/1rgs/claude-code-proxy) — simplified by [adrianR84](https://github.com/adrianR84)
