<div align="center">

# Antigravity Proxy

**Gemini, Claude and GPT-OSS through a local OpenAI-compatible proxy**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-green.svg)](#requirements)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Features](#features) · [Quick Start](#quick-start) · [Models](#available-models) · [Configuration](#configuration) · [Docs](#how-it-works)

</div>

---

## What is this?

A lightweight **OpenAI-compatible proxy** that bridges AI tools to **Google Antigravity's** internal Cloud Code Assist API. Authenticate once with the [Antigravity CLI](https://antigravity.google/), then use the Gemini, Claude, and GPT-OSS models available to your account through an OpenAI-compatible client — **no separate API key**.

```
Your Tool (OpenAI format)  →  Antigravity Proxy (localhost:8877)  →  Google Antigravity models
```

### Why use this?

- **One login** — uses the Antigravity CLI's OAuth token; model access and quotas depend on your Google account
- **OpenAI-compatible** — works with clients such as Aider, Continue, Hermes, the OpenAI SDK, and LangChain
- **No dependencies** — pure Python stdlib, no `pip install` needed
- **Live model catalog and quotas** — `GET /v1/models` and `GET /quota`
- **Tool/function calling** — full support, including round-trip through conversation history
- **Streaming** — SSE streaming support for real-time token generation

---

## Features

- **OpenAI-compatible** `POST /v1/chat/completions` — streaming + non-streaming
- **Native Gemini passthrough** `POST /v1beta/models/{model}:generateContent` — text-only structured JSON output
- **Tool/function calling** — OpenAI tools format → Gemini functionDeclarations, with round-trip encoding
- **Model discovery** — lists current upstream text models; model IDs are sent upstream unchanged
- **OAuth token auto-refresh** — token refreshed automatically when expired, no manual intervention
- **Zero pip dependencies** — Python 3.10+ standard library only
- **systemd service** included — run as a background service on Linux

---

## Available Models

Run `curl http://127.0.0.1:8877/v1/models` and use a returned `id` in your client. `/models` is an alias. The proxy uses the `agy` CLI User-Agent for catalog, generation, and quota requests, caches the catalog for five minutes, and excludes internal and image-only entries. The CLI catalog can include subscription-only models such as `gemini-4-argon-high`, `gemini-4-argon-medium`, and `gemini-4-argon-low` that the Electron catalog omits. If Google does not return a usable catalog, the endpoint reports an error instead of showing an outdated list. Access and quotas can still vary per model; a listed ID does not prove generation works. GPT-OSS is an open-weight model, not a claim that proprietary OpenAI GPT models are available.

Use the exact model ID returned by `/v1/models`. The proxy does not silently remap old names such as `gemini-3.8-flash` to another model; an unsupported ID may be rejected by Google.

---

## Quick Start

### Prerequisites

1. **Python 3.10+** — check with `python3 --version`
2. **Antigravity CLI** — install and authenticate (one-time):

```bash
# Install agy (macOS/Linux)
curl -fsSL https://storage.googleapis.com/antigravity-releases/install.sh | bash

# Authenticate (opens browser, sign in with Google)
agy
```

> Windows: download from [antigravity.google](https://antigravity.google/)

### Install & Run

```bash
git clone https://github.com/usamashehab/antigravity-proxy.git
cd antigravity-proxy

# Start the proxy
python3 antigravity_proxy.py

# Verify it's running
curl http://127.0.0.1:8877/health
# → {"status": "ok", "service": "antigravity-proxy"}
```

That's it. The proxy is now listening on `http://127.0.0.1:8877`.

### Connect Your Tool

Set an OpenAI-compatible client's base URL to `http://127.0.0.1:8877/v1` and choose an ID from `GET /v1/models`. The API key can be any string (the proxy doesn't check it).

<details>
<summary><b>OpenAI Python SDK</b></summary>

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8877/v1",
    api_key="antigravity",
)

response = client.chat.completions.create(
    model="gemini-3-flash",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```
</details>

<details>
<summary><b>curl</b></summary>

```bash
curl http://127.0.0.1:8877/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3-flash",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```
</details>

<details>
<summary><b>Aider</b></summary>

```bash
export OPENAI_API_BASE=http://127.0.0.1:8877/v1
export OPENAI_API_KEY=antigravity
aider --model gemini-3-flash
```
</details>

<details>
<summary><b>Continue.dev (VS Code)</b></summary>

In `~/.continue/config.json`:
```json
{
  "models": [{
    "title": "Antigravity",
    "provider": "openai",
    "model": "gemini-3-flash",
    "apiBase": "http://127.0.0.1:8877/v1",
    "apiKey": "antigravity"
  }]
}
```
</details>

<details>
<summary><b>Hermes Agent</b></summary>

```bash
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8877/v1
hermes config set model.api_key antigravity
hermes config set model.default gemini-3-flash
```
</details>

<details>
<summary><b>Claude Code</b></summary>

This proxy does not expose the Anthropic Messages API expected by Claude Code. Use an OpenAI-compatible client to access Claude models through it.
</details>

<details>
<summary><b>LangChain</b></summary>

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8877/v1",
    api_key="antigravity",
    model="gemini-3-flash",
)
```
</details>

---

## AI-Assisted Setup 🤖

**Using an AI assistant (Claude, GPT, Gemini, etc.) to set this up?**

This repo includes [`AGENTS.md`](AGENTS.md) — a complete instruction file for AI agents. Just tell your AI assistant:

> "Help me install and configure the antigravity-proxy from https://github.com/usamashehab/antigravity-proxy"

The AI will read `AGENTS.md` and guide you through every step: prerequisites check, installation, tool configuration, and troubleshooting.

---

## Configuration

### CLI Arguments

```
python3 antigravity_proxy.py [--port PORT] [--host HOST]

  --port PORT    Port to listen on (default: 8877)
  --host HOST    Host to bind to (default: 127.0.0.1)
```

### Run as a systemd Service (Linux)

```bash
# Clone to permanent location
git clone https://github.com/usamashehab/antigravity-proxy.git ~/antigravity-proxy

# Install service
mkdir -p ~/.config/systemd/user
cp ~/antigravity-proxy/antigravity-proxy.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable antigravity-proxy
systemctl --user start antigravity-proxy

# Check status
systemctl --user status antigravity-proxy
```

### Environment Variables

| Variable                     | Default | Description                        |
|------------------------------|---------|------------------------------------|
| `ANTIGRAVITY_CLIENT_ID`      | built-in| Override OAuth client ID           |
| `ANTIGRAVITY_CLIENT_SECRET`  | built-in| Override OAuth client secret       |
| `ANTIGRAVITY_CAPTURE_UPSTREAM_ERRORS` | unset | Set to `1` temporarily to retain upstream model-generation HTTP error bodies for debugging |

The default client credentials are the same values embedded in every Antigravity CLI binary — you only need these env vars if you're using a custom build.

### Temporary upstream error capture

Set `ANTIGRAVITY_CAPTURE_UPSTREAM_ERRORS=1` in the proxy service environment and restart **the proxy only**. On an upstream HTTP error during model generation, the proxy saves up to 8 KiB of the response body, the HTTP status, a bounded model ID, timestamp, and truncation flag under `~/.hermes/cache/proxy-upstream-errors/`. The directory is private (`0700`) and each file is `0600`. The proxy does not copy its request or OAuth headers into the diagnostic, and the upstream body is never written to `proxy.log`, the journal, or the client error. The journal contains only a filename to locate the diagnostic.

Upstream error messages **may quote private user content or credentials**, even though the proxy never adds them to the file itself. Do not attach these files to bug reports or commit them. A single proxy process retains at most 50 diagnostics and prunes files older than 24 hours **when the next error is captured**. Remove the files and unset the environment variable when debugging is finished.

---

## Three API Endpoints

### 1. OpenAI Chat Completions

```
POST /v1/chat/completions
```

Standard OpenAI format with messages, tools, and streaming. Use this for any OpenAI-compatible client.

### 2. Native Gemini Passthrough

```
POST /v1beta/models/{model}:generateContent
```

Accepts text-only Gemini requests. Image and media parts are rejected with HTTP 400. Use this for:

- **Structured output** (`responseSchema` for enforced JSON schema)
- **Thinking control** (`thinkingConfig`)

<details>
<summary><b>Structured output example</b></summary>

```bash
curl http://127.0.0.1:8877/v1beta/models/gemini-3-flash:generateContent \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"parts": [{"text": "Extract: Ahmed is 30 years old"}]}],
    "generationConfig": {
      "temperature": 0,
      "responseMimeType": "application/json",
      "responseSchema": {
        "type": "OBJECT",
        "properties": {
          "name": {"type": "STRING"},
          "age": {"type": "NUMBER"}
        },
        "required": ["name", "age"]
      }
    }
  }'
```
</details>

### 3. Model Quotas

```
GET /quota
GET /v1/quota
```

Both paths use the configured OAuth account to retrieve **model buckets** (`retrieveUserQuota`) and separate **subscription quota groups** (`retrieveUserQuotaSummary`) directly from Google, without running `agy`. The JSON response contains `source`, `fetched_at`, `buckets`, `groups_source`, and `groups`. Model buckets have `model`, `token_type`, `remaining_fraction` (0 to 1, or `null` when unavailable), and `reset_time` (or `null`). Each group has `name` and `buckets`; group buckets have `id`, `window`, a measured `remaining_fraction` (0 to 1, including a real zero), and `reset_time` (or `null`). Groups can include a separate weekly Gemini 4 Argon limit; they are not interchangeable with model buckets. No OAuth token or project ID is returned. Empty, malformed, or inconsistent group data and upstream errors return an error rather than a misleading partial success. This is a private Google API and its response can change. The proxy has no authentication layer, so keep it bound to loopback.

---

## Management

```bash
# Health check
curl http://127.0.0.1:8877/health

# List models
curl http://127.0.0.1:8877/v1/models

# View logs (systemd)
journalctl --user -u antigravity-proxy -f

# Restart / stop
systemctl --user restart antigravity-proxy
systemctl --user stop antigravity-proxy

# Re-authenticate when token expires
agy && systemctl --user restart antigravity-proxy
```

---

## How It Works

```
Your Client (OpenAI format)
    │
    ▼
Antigravity Proxy (localhost:8877)
    │  ┌─ Translates OpenAI messages → Gemini contents
    │  ├─ Converts OpenAI tools → Gemini functionDeclarations
    │  ├─ Manages OAuth token (auto-refresh)
    │  ├─ Forwards live model IDs unchanged (no legacy aliases)
    │  └─ Wraps in Antigravity envelope {project, model, request}
    │
    ▼
daily-cloudcode-pa.googleapis.com
    │
    ▼
Gemini / Claude / GPT-OSS models → Response unwrapped → OpenAI format → Your client
```

<details>
<summary><b>Technical details</b></summary>

1. Reads OAuth token from `~/.gemini/antigravity-cli/antigravity-oauth-token`
2. Refreshes expired access tokens via `oauth2.googleapis.com/token`
3. Discovers Cloud Code Assist project ID via `v1internal:loadCodeAssist`
4. Wraps each request in the Antigravity envelope: `{project, model, request, requestType, userAgent, requestId}`
5. Forwards to `v1internal:streamGenerateContent?alt=sse` and adapts the result to streaming or non-streaming OpenAI format
6. Unwraps the `{response: {...}}` envelope
7. Converts Gemini response → OpenAI format (choices, message, usage)
8. For tool calls, preserves Gemini's `functionCall.id` and `thoughtSignature` for round-tripping through OpenAI messages
</details>

---

## Adding or Updating Models

You do not need to edit code when Google changes its model catalog: `GET /v1/models` fetches live IDs. Configure clients with one of those exact IDs; the proxy forwards it unchanged.

---

## Requirements

- Python 3.10+
- Antigravity CLI (`agy`) installed and authenticated
- No pip dependencies

---

## FAQ

<details>
<summary><b>Is this legal to use?</b></summary>

The proxy uses the same OAuth token and API endpoints that the Antigravity CLI itself uses. It does not bypass authentication — you need a valid Google account with Antigravity access. Multiple similar projects exist publicly on GitHub. However, Google's terms of service may restrict programmatic API access. Use at your own discretion.
</details>

<details>
<summary><b>Does it support streaming?</b></summary>

Yes. SSE streaming (`"stream": true`) on the OpenAI chat endpoint. The text-only Gemini passthrough exposes `:generateContent`, not a separate streaming endpoint.
</details>

<details>
<summary><b>Can I send images?</b></summary>

No. Both endpoints reject image/media input with HTTP 400; this proxy handles text only.
</details>

<details>
<summary><b>Is there a rate limit?</b></summary>

Rate limits are determined by Google's Antigravity quota, not the proxy. The proxy itself is multithreaded with no concurrency limit.
</details>

<details>
<summary><b>Does the token expire?</b></summary>

The OAuth access token expires (~1 hour). The proxy auto-refreshes it. The refresh token can expire if unused for extended periods — re-run `agy` if that happens.
</details>

<details>
<summary><b>Can I bind to 0.0.0.0?</b></summary>

Yes, but **the proxy has no authentication layer**. Anyone who can reach it can use your Antigravity quota. Only bind to a non-localhost address if you understand the risk.
</details>

---

## License

[MIT](LICENSE)
