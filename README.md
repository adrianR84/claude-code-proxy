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
cp .env.example .env
# Edit .env with your API keys

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

**Configuration (API keys, providers) is done in `.env` — see examples below.**

```bash
# Standard
ANTHROPIC_BASE_URL=http://localhost:8082 claude

# If not authorized (use placeholder token):
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_AUTH_TOKEN="sk-ant-api03-placeholder" claude
```

## Configuration

### Custom Provider (override ALL models)

```env
CUSTOM_MODEL=qwen/qwen3-235b-a22b
CUSTOM_API_KEY=your-api-key
CUSTOM_BASE_URL=https://your-endpoint.com/api/v1/gateway/v1
```

Every request (haiku, sonnet, opus) routes to `CUSTOM_MODEL`.

### Custom Endpoint + Model Mapping

```env
CUSTOM_API_KEY=your-api-key
CUSTOM_BASE_URL=https://your-endpoint.com/api/v1/gateway/v1
PREFERRED_PROVIDER=custom
BIG_MODEL=gpt-5.5
SMALL_MODEL=gpt-5.4-mini
```

Routes through custom endpoint but respects mapping: `haiku → SMALL`, `sonnet/opus → BIG`.

### Standard Providers

| Provider | Description |
|----------|-------------|
| `openai` | Default. Routes to OpenAI. haiku/sonnet/opus map to SMALL_MODEL/BIG_MODEL |
| `google` | Routes to Gemini. haiku/sonnet/opus map to Gemini models |
| `anthropic` | Pass-through to Anthropic. No remapping |

### Full Example

```env
ANTHROPIC_API_KEY=
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=

CUSTOM_MODEL=qwen/qwen3-235b-a22b
CUSTOM_API_KEY=auton_sk_...
CUSTOM_BASE_URL=https://api.autonaisol.xyz/api/v1/gateway/v1

PREFERRED_PROVIDER=openai
OPENAI_BASE_URL=
BIG_MODEL=gpt-5.5
SMALL_MODEL=gpt-5.4-mini
```

## Model Mapping

| Claude Model | Maps To |
|--------------|---------|
| haiku | SMALL_MODEL (gpt-5.4-mini) |
| sonnet | BIG_MODEL (gpt-5.5) |
| opus | BIG_MODEL (gpt-5.5) |

## Docker

```bash
curl -O https://raw.githubusercontent.com/adrianR84/claude-code-proxy/main/.env.example
# Edit .env, then:
docker run -d --env-file .env -p 8082:8082 ghcr.io/adrianR84/claude-code-proxy:latest
```

Or with compose:

```yaml
services:
  proxy:
    image: ghcr.io/adrianR84/claude-code-proxy:latest
    restart: unless-stopped
    env_file: .env
    ports:
      - 8082:8082
```

---

Original creator: [1rgs](https://github.com/1rgs/claude-code-proxy) — simplified by [adrianR84](https://github.com/adrianR84)
