# AGENTS.md — LLM Agent Instructions

> **If you are an AI assistant (Claude, GPT, Gemini, etc.) helping a user set up this proxy, read this file first. It contains everything you need to guide the user through installation, configuration, and troubleshooting.**

---

## What This Project Is

A local proxy server that translates OpenAI-compatible API calls into Google Antigravity's internal Cloud Code Assist API format. It lets OpenAI-compatible tools use available Gemini, Claude, and GPT-OSS models, authenticated via the Antigravity CLI's OAuth token.

The proxy listens on `http://127.0.0.1:8877` and exposes these API endpoints:
- `POST /v1/chat/completions` — OpenAI chat format (messages, tools, streaming)
- `POST /v1beta/models/{model}:generateContent` — text-only Gemini passthrough (structured output; images rejected)
- `GET /v1/models` (also `/models`) — live text-model catalog from Google under the agy CLI User-Agent, cached for five minutes; excludes internal and image-only entries; listing a model does not prove generation works
- `GET /quota` (also `/v1/quota`) — live read-only model buckets and subscription groups (including separate Argon quotas when reported) directly from Google; no `agy` subprocess, token, or project ID in the response

---

## How to Guide a User Through Setup

### Step 1 — Check Prerequisites

Ask the user:
1. **Do you have Python 3.10+?** Check with `python3 --version`
2. **Do you have the Antigravity CLI (`agy`) installed and authenticated?** Check with `ls ~/.gemini/antigravity-cli/antigravity-oauth-token`

If they don't have `agy`, tell them to install it:
- **macOS/Linux:** `curl -fsSL https://storage.googleapis.com/antigravity-releases/install.sh | bash`
- **Windows:** Download from [antigravity.google](https://antigravity.google/)

Then authenticate: run `agy` once, sign in with Google account.

### Step 2 — Clone and Start

```bash
git clone https://github.com/usamashehab/antigravity-proxy.git
cd antigravity-proxy
python3 antigravity_proxy.py
```

Verify it's running:
```bash
curl http://127.0.0.1:8877/health
# Should return: {"status": "ok", "service": "antigravity-proxy"}
```

### Step 3 — Configure Their Tool

Ask what tool they want to connect. Apply these settings:

| Tool Type         | Base URL                              | API Key         | Model             |
|-------------------|---------------------------------------|-----------------|-------------------|
| OpenAI SDK        | `http://127.0.0.1:8877/v1`            | Any string      | See `/v1/models` |
| Continue.dev      | `http://127.0.0.1:8877/v1`            | Any string      |                   |
| Aider             | `--openai-api-base http://127.0.0.1:8877/v1` | Any string |                   |
| Hermes Agent      | `hermes config set ...`               | Any string      |                   |
| Claude Code       | Not supported (Anthropic Messages API) | N/A          | N/A               |

### Step 4 — Verify It Works

Have them run a test request:
```bash
curl http://127.0.0.1:8877/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemini-3-flash", "messages": [{"role": "user", "content": "Say hello"}]}'
```

---

## Available Models

Fetch `GET http://127.0.0.1:8877/v1/models` and use a returned `id`. The catalog depends on the authenticated account and Google's current response. A listed model may still have exhausted quota; GPT-OSS is open-weight and does not mean proprietary OpenAI GPT models are available. The proxy sends model IDs upstream unchanged; it does not silently map old aliases to different models.

---

## Common Issues and Solutions

### "Token file not found"
The user hasn't authenticated with `agy` yet. Run `agy` and sign in.

### "No refresh_token in token file"
The token file is corrupted or incomplete. Delete it and re-run `agy`:
```bash
rm ~/.gemini/antigravity-cli/antigravity-oauth-token
agy  # re-authenticate
```

### 401 / auth errors after working for a while
The OAuth refresh token has expired. Re-run `agy` and restart the proxy.

### Model returns empty response or 404
The model name may not exist on the Antigravity backend. Choose a current ID from `GET /v1/models`; `gemini-3-flash` is a common fallback when listed. If the catalog endpoint fails, check the OAuth account and upstream availability instead of assuming an old alias still works.

### Proxy is slow on first request
The first request discovers the project ID (extra network call). Subsequent requests are faster.

### Requests timeout
The Antigravity backend can be slow, especially for thinking models. This is not a proxy issue — it's upstream latency.

---

## Architecture (for context)

```
Client (OpenAI format)
  → Proxy translates to Gemini format
  → Wraps in Antigravity envelope
  → Forwards to cloudcode-pa.googleapis.com
  → Response unwrapped, converted to OpenAI format
  → Returned to client
```

The proxy auto-refreshes the OAuth token when it expires. It uses Python stdlib only — no dependencies to install.

---

## File Reference

| File                        | Purpose                                          |
|-----------------------------|--------------------------------------------------|
| `antigravity_proxy.py`      | Main proxy server (single file, stdlib only)     |
| `antigravity-proxy.service` | systemd user service template                    |
| `README.md`                 | Full human-readable documentation                |
| `AGENTS.md`                 | This file — instructions for AI assistants       |

---

## Security Notes

- The proxy binds to `127.0.0.1` by default — only accessible locally
- No API key or authentication layer (relies on localhost binding)
- OAuth token stays on the user's machine, never sent anywhere except Google
- The `CLIENT_ID` and `CLIENT_SECRET` constants are the same values embedded in every Antigravity CLI binary — they are not personal secrets
