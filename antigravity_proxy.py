#!/usr/bin/env python3
"""
Antigravity OpenAI-Compatible Proxy (Cloud Code Assist API)

Bridges Hermes Agent (OpenAI-compatible client) to Google Antigravity's Cloud
Code Assist API. Uses the OAuth token managed by the Antigravity CLI (agy) and
talks directly to https://cloudcode-pa.googleapis.com — no `agy` subprocess
needed, giving us native streaming + tool/function-call support.

Architecture:
  1. OAuth token read from ~/.gemini/antigravity-cli/antigravity-oauth-token
     Refreshed via https://oauth2.googleapis.com/token when expired.
  2. Project ID discovered via v1internal:loadCodeAssist (cached).
  3. Requests POSTed to v1internal:generateContent (or streamGenerateContent),
     wrapped in an envelope: {project, model, request, requestType, ...}
  4. Responses unwrapped from {"response": {<Gemini response>}} and transformed
     to OpenAI chat.completion format.

Usage:
  python3 antigravity_proxy.py [--port 8877] [--host 127.0.0.1]

Then configure Hermes:
  hermes config set model.provider custom
  hermes config set model.base_url http://127.0.0.1:8877/v1
  hermes config set model.api_key antigravity
  hermes config set model.default gemini-3-flash
"""

import argparse
import json
import itertools
import os
import re
import stat
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# OAuth client credentials for Antigravity CLI (reverse-engineered from agy).
# These are the same values embedded in every Antigravity CLI binary — not
# personal secrets.  Set them via env vars if you prefer, or see README.md
# for the default values to put in a .env file.
CLIENT_ID = os.environ.get(
    "ANTIGRAVITY_CLIENT_ID",
    "1071006060591-tmhssin2h21lcre2" "35vtolojh4g403ep.apps.googleusercontent.com",
)
CLIENT_SECRET = os.environ.get(
    "ANTIGRAVITY_CLIENT_SECRET",
    "GOCSPX-K58FWR486LdL" "J1mLB8sXC4z6qDAf",
)

TOKEN_FILE = os.path.expanduser("~/.gemini/antigravity-cli/antigravity-oauth-token")
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
CLOUDCODE_BASE = "https://daily-cloudcode-pa.googleapis.com"
LOAD_CODEASSIST_URL = f"{CLOUDCODE_BASE}/v1internal:loadCodeAssist"
RETRIEVE_USER_QUOTA_URL = f"{CLOUDCODE_BASE}/v1internal:retrieveUserQuota"
RETRIEVE_USER_QUOTA_SUMMARY_URL = f"{CLOUDCODE_BASE}/v1internal:retrieveUserQuotaSummary"
FETCH_AVAILABLE_MODELS_URL = f"{CLOUDCODE_BASE}/v1internal:fetchAvailableModels"
MODEL_CATALOG_TTL = 300
_models_cache_lock = threading.Lock()
_models_cache_ids = None
_models_cache_until = 0.0
GENERATE_CONTENT_URL = f"{CLOUDCODE_BASE}/v1internal:streamGenerateContent?alt=sse"
STREAM_GENERATE_CONTENT_URL = f"{CLOUDCODE_BASE}/v1internal:streamGenerateContent?alt=sse"

ANTIGRAVITY_USER_AGENT = "antigravity/cli/1.4.7"
CLIENT_METADATA = json.dumps(
    {"ideType": "ANTIGRAVITY", "platform": "MACOS", "pluginType": "GEMINI"},
    separators=(",", ":"),
)

# Minimal system instruction — just enough for the Code Assist API to accept
# the request. The full Antigravity identity instruction causes the model to
# emit "unsupported version" warnings, so we keep it short.
DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful AI assistant."

UPSTREAM_TIMEOUT = 300  # 5 minutes
TOKEN_REFRESH_SKEW = 120  # refresh this many seconds before actual expiry

DEFAULT_MODEL = "gemini-3-flash"

# ──────────────────────────────────────────────────────────────────────────────
# Token / OAuth management (thread-safe)
# ──────────────────────────────────────────────────────────────────────────────

_token_lock = threading.Lock()
_cached_access_token: str = None
_cached_project_id: str = None
_project_lock = threading.Lock()


LOG_FILE = "./proxy.log"
UPSTREAM_ERROR_DIR = Path.home() / ".hermes" / "cache" / "proxy-upstream-errors"
_upstream_error_lock = threading.Lock()
_UPSTREAM_ERROR_BODY_LIMIT = 8192
_UPSTREAM_ERROR_FILE_LIMIT = 50
_UPSTREAM_ERROR_MAX_AGE = 24 * 60 * 60

def _log(msg: str):
    entry = f"[antigravity-proxy] {msg}"
    print(entry, file=sys.stderr, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(entry + "\n")
    except Exception:
        pass


def _capture_upstream_error(error: HTTPError, envelope: dict) -> None:
    """Opt-in private response-body capture; never persist request content or headers."""
    if os.environ.get("ANTIGRAVITY_CAPTURE_UPSTREAM_ERRORS") != "1":
        return
    try:
        body = error.read(_UPSTREAM_ERROR_BODY_LIMIT + 1)
        directory = UPSTREAM_ERROR_DIR
        with _upstream_error_lock:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            info = directory.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                return
            now = time.time()
            files = sorted(directory.glob("upstream-*.json"), key=lambda p: p.stat().st_mtime)
            for old in files:
                if old.stat().st_mtime < now - _UPSTREAM_ERROR_MAX_AGE:
                    old.unlink()
            files = [old for old in files if old.exists()]
            for old in files[:-(_UPSTREAM_ERROR_FILE_LIMIT - 1)]:
                old.unlink()
            text = body[:_UPSTREAM_ERROR_BODY_LIMIT].decode("utf-8", errors="replace")
            encoded = text.encode("utf-8")
            expanded = len(encoded) > _UPSTREAM_ERROR_BODY_LIMIT
            if expanded:
                text = encoded[:_UPSTREAM_ERROR_BODY_LIMIT].decode("utf-8", errors="ignore")
            record = {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "model": str(envelope.get("model") or "")[:128],
                "status": error.code,
                "body": text,
                "truncated": len(body) > _UPSTREAM_ERROR_BODY_LIMIT or expanded,
            }
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                             prefix="upstream-", suffix=".json", delete=False) as output:
                json.dump(record, output, ensure_ascii=False)
                filename = Path(output.name).name
        _log(f"Saved upstream HTTP {error.code} diagnostic: {filename}")
    except Exception:
        _log(f"Could not save upstream HTTP {error.code} diagnostic")


def _parse_expiry(expiry_raw):
    """Parse the 'expiry' field from the token file into a unix timestamp.

    The Antigravity CLI stores expiry as an RFC3339 string like
    "2026-06-30T12:55:03.123456789Z". Python stdlib datetime.fromisoformat
    can't handle nanosecond precision, so we truncate to 6 digits.
    """
    if not expiry_raw:
        return 0.0
    if isinstance(expiry_raw, (int, float)):
        return float(expiry_raw)
    s = str(expiry_raw).strip()
    # Truncate sub-second precision to microseconds for fromisoformat.
    try:
        if "." in s:
            head, tail = s.split(".", 1)
            # Keep only the part up to 'Z' or timezone offset, truncate ns.
            tz_part = ""
            for i, ch in enumerate(tail):
                if ch in "Z+-":
                    tz_part = tail[i:]
                    tail = tail[:i]
                    break
            tail = tail[:6]  # microseconds max
            s = f"{head}.{tail}{tz_part}"
        # fromisoformat in 3.12 handles 'Z' suffix.
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0


def _read_token_from_disk():
    """Read and parse the token file. Returns the raw dict or None."""
    try:
        with open(TOKEN_FILE, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        _log(f"WARNING: could not read token file {TOKEN_FILE}: {e}")
        return None


def _write_token_to_disk(data: dict):
    """Persist updated token data back to disk (atomic-ish)."""
    try:
        tmp = TOKEN_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, TOKEN_FILE)
        try:
            os.chmod(TOKEN_FILE, 0o600)
        except OSError:
            pass
    except OSError as e:
        _log(f"WARNING: could not write token file: {e}")


def _refresh_access_token(refresh_token: str) -> dict:
    """Refresh the access token via Google's OAuth endpoint.

    Returns the new token dict {access_token, refresh_token?, expiry, token_type}.
    Raises RuntimeError on failure.
    """
    body = json.dumps({
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }).encode()
    req = Request(OAUTH_TOKEN_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
    except HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()
        except Exception:
            pass
        raise RuntimeError(f"OAuth refresh failed (HTTP {e.code}): {detail}")
    except URLError as e:
        raise RuntimeError(f"OAuth refresh network error: {e}")

    new_tok = {
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token", refresh_token),
        "token_type": payload.get("token_type", "Bearer"),
        "expiry": payload.get("expires_in", 3600),
    }
    # Convert expires_in (seconds) to an RFC3339 expiry timestamp ~now.
    expires_in = int(payload.get("expires_in", 3600))
    exp_dt = datetime.now(timezone.utc)
    from datetime import timedelta
    exp_dt = exp_dt + timedelta(seconds=expires_in)
    new_tok["expiry"] = exp_dt.strftime("%Y-%m-%dT%H:%M:%S.000000Z")
    return new_tok


def get_access_token() -> str:
    """Return a valid access token, refreshing if necessary. Thread-safe."""
    global _cached_access_token
    with _token_lock:
        data = _read_token_from_disk()
        if not data:
            raise RuntimeError(
                f"No OAuth token found at {TOKEN_FILE}. "
                "Run `agy` to authenticate first."
            )
        token_obj = data.get("token") or {}
        access_token = token_obj.get("access_token")
        expiry_ts = _parse_expiry(token_obj.get("expiry"))
        now = time.time()

        needs_refresh = (
            not access_token
            or expiry_ts == 0.0
            or (expiry_ts - now) < TOKEN_REFRESH_SKEW
        )

        if needs_refresh:
            refresh_token = token_obj.get("refresh_token")
            if not refresh_token:
                raise RuntimeError("No refresh_token available; re-run `agy` to authenticate.")
            _log("Access token expired (or missing) — refreshing...")
            new_tok = _refresh_access_token(refresh_token)
            data["token"] = new_tok
            _write_token_to_disk(data)
            access_token = new_tok["access_token"]

        _cached_access_token = access_token
        return access_token


# ──────────────────────────────────────────────────────────────────────────────
# Project discovery (cached)
# ──────────────────────────────────────────────────────────────────────────────

def _load_code_assist(access_token: str) -> str:
    """Discover the cloudaicompanion project ID. Returns project id string.

    NOTE: The metadata field (ideType/platform/pluginType) is rejected by the
    API with INVALID_ARGUMENT for all known enum string values. Sending an
    empty body {} works and returns the cloudaicompanionProject. The Client-Metadata
    header still carries the ide/plugin info.
    """
    body = json.dumps({}).encode()
    req = Request(LOAD_CODEASSIST_URL, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", ANTIGRAVITY_USER_AGENT)
    req.add_header("X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1")
    req.add_header("Client-Metadata", CLIENT_METADATA)
    try:
        with urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
    except HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()
        except Exception:
            pass
        raise RuntimeError(f"loadCodeAssist failed (HTTP {e.code}): {detail}")
    except URLError as e:
        raise RuntimeError(f"loadCodeAssist network error: {e}")

    project_id = (
        payload.get("cloudaicompanionProject")
        or payload.get("cloudaicompanion_project")
    )
    if not project_id:
        # Some responses nest it differently; do a recursive search.
        project_id = _deep_find(payload, "cloudaicompanionProject")
    if not project_id:
        raise RuntimeError(
            f"loadCodeAssist did not return a project ID. Response: {payload}"
        )
    return project_id


def _deep_find(obj, key):
    """Recursively search for a key in nested dicts/lists. Returns first match."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _deep_find(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = _deep_find(item, key)
            if r is not None:
                return r
    return None


def get_project_id() -> str:
    """Return project id. Defaults to aicode-consumers."""
    return "aicode-consumers"


def fetch_user_quota(access_token: str) -> dict:
    """Retrieve read-only quota buckets without returning the upstream envelope."""
    req = Request(RETRIEVE_USER_QUOTA_URL,
                  data=json.dumps({"project": get_project_id()}).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", ANTIGRAVITY_USER_AGENT)
    req.add_header("X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1")
    req.add_header("Client-Metadata", CLIENT_METADATA)
    with urlopen(req, timeout=15) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Quota response too large")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("buckets"), list):
        raise ValueError("Invalid quota response")

    buckets = []
    for item in payload["buckets"]:
        if not isinstance(item, dict):
            continue
        model = item.get("modelId")
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,127}", model):
            continue
        token_type = item.get("tokenType")
        if not isinstance(token_type, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", token_type):
            token_type = None
        remaining = item.get("remainingFraction")
        if (isinstance(remaining, bool) or not isinstance(remaining, (int, float))
                or not 0 <= remaining <= 1):
            remaining = None
        reset = item.get("resetTime")
        try:
            valid_reset = isinstance(reset, str) and len(reset) <= 64 and bool(
                datetime.fromisoformat(reset.replace("Z", "+00:00")).tzinfo)
        except ValueError:
            valid_reset = False
        if not valid_reset:
            reset = None
        buckets.append({"model": model, "token_type": token_type,
                        "remaining_fraction": remaining, "reset_time": reset})
    if payload["buckets"] and not buckets:
        raise ValueError("Invalid quota buckets")
    return {"source": "retrieveUserQuota",
            "fetched_at": datetime.now(timezone.utc).isoformat(), "buckets": buckets}


def fetch_quota_groups(access_token: str) -> list[dict]:
    """Read subscription quota groups directly, without spawning the CLI."""
    req = Request(RETRIEVE_USER_QUOTA_SUMMARY_URL,
                  data=json.dumps({"project": get_project_id()}).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", ANTIGRAVITY_USER_AGENT)
    with urlopen(req, timeout=15) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Quota summary response too large")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("groups"), list):
        raise ValueError("Invalid quota summary response")

    groups = []
    for item in payload["groups"]:
        if not isinstance(item, dict):
            raise ValueError("Invalid quota summary group")
        name = item.get("displayName")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 /()&._-]{0,79}", name):
            raise ValueError("Invalid quota summary group name")
        source_buckets = item.get("buckets")
        if not isinstance(source_buckets, list) or not source_buckets:
            raise ValueError("Missing quota summary buckets")
        buckets = []
        for bucket in source_buckets:
            if not isinstance(bucket, dict):
                raise ValueError("Invalid quota summary bucket")
            bucket_id = bucket.get("bucketId")
            window = bucket.get("window")
            fraction = bucket.get("remainingFraction")
            if (not isinstance(bucket_id, str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", bucket_id)
                    or not isinstance(window, str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", window)
                    or isinstance(fraction, bool)
                    or not isinstance(fraction, (int, float))
                    or not 0 <= fraction <= 1):
                raise ValueError("Invalid quota summary bucket fields")
            reset = bucket.get("resetTime")
            if reset is not None:
                try:
                    valid_reset = isinstance(reset, str) and len(reset) <= 64 and bool(
                        datetime.fromisoformat(reset.replace("Z", "+00:00")).tzinfo)
                except ValueError:
                    valid_reset = False
                if not valid_reset:
                    raise ValueError("Invalid quota summary reset time")
            buckets.append({"id": bucket_id, "window": window,
                            "remaining_fraction": fraction,
                            "reset_time": reset})
        groups.append({"name": name, "buckets": buckets})
    if not groups:
        raise ValueError("Quota summary has no measured buckets")
    return groups

def fetch_available_models(access_token: str) -> list[str]:
    """List text-capable models from the same backend used for generation."""
    req = Request(FETCH_AVAILABLE_MODELS_URL,
                  data=json.dumps({"project": get_project_id()}).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", ANTIGRAVITY_USER_AGENT)
    req.add_header("X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1")
    req.add_header("Client-Metadata", CLIENT_METADATA)
    with urlopen(req, timeout=15) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Model catalog response too large")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise ValueError("Invalid model catalog response")
    excluded_ids = set()
    for category in ("imageGenerationModelIds", "tabModelIds"):
        entries = payload.get(category)
        if isinstance(entries, list):
            excluded_ids.update(entry for entry in entries if isinstance(entry, str))
    agent_ids = set()
    remaining = [payload.get("agentModelSorts")]
    while remaining:
        item = remaining.pop()
        if isinstance(item, str) and item in payload["models"]:
            agent_ids.add(item)
        elif isinstance(item, dict):
            remaining.extend(item.values())
        elif isinstance(item, list):
            remaining.extend(item)
    model_ids = []
    for model_id, info in payload["models"].items():
        if (not isinstance(model_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,127}", model_id)
                or not isinstance(info, dict) or info.get("isInternal") is True
                or model_id in excluded_ids
                or (model_id not in agent_ids
                    and not model_id.startswith(("gemini-", "claude-", "gpt-")))
                or "-image" in model_id):
            continue
        model_ids.append(model_id)
    if not model_ids:
        raise ValueError("Model catalog has no supported text models")
    return sorted(model_ids)

def get_available_models(access_token: str) -> list[str]:
    """Share one short-lived catalog between concurrent model picker requests."""
    global _models_cache_ids, _models_cache_until
    with _models_cache_lock:
        if _models_cache_ids is not None and time.monotonic() < _models_cache_until:
            return list(_models_cache_ids)
        model_ids = fetch_available_models(access_token)
        _models_cache_ids = model_ids
        _models_cache_until = time.monotonic() + MODEL_CATALOG_TTL
        return list(model_ids)

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI -> Gemini message transformation
# ──────────────────────────────────────────────────────────────────────────────

def _reject_image_data(value):
    """Reject image/media payloads rather than fetching or forwarding them."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("responseSchema", "response_schema"):
                continue  # Output field names are not input media payloads.
            media_part = key in {"inlineData", "inline_data", "fileData", "file_data", "image_url", "input_image"}
            image_payload = key == "image" and isinstance(item, dict) and bool(
                {"data", "url", "mimeType", "mime_type", "fileUri"}.intersection(item)
            )
            if media_part or image_payload:
                raise ValueError("Images are not supported by this proxy")
            _reject_image_data(item)
    elif isinstance(value, list):
        for item in value:
            _reject_image_data(item)


def _content_to_parts(content) -> list:
    """Normalize an OpenAI message 'content' field to a list of Gemini parts."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                s = str(block)
                if s:
                    parts.append({"text": s})
                continue
            btype = block.get("type", "text")
            if btype == "text":
                text = block.get("text", "")
                if text:
                    parts.append({"text": text})
            elif btype in ("image_url", "input_image", "image"):
                raise ValueError("Images are not supported by this proxy")
            elif btype == "input_text":
                text = block.get("text", "")
                if text:
                    parts.append({"text": text})
            else:
                txt = json.dumps(block.get('text', ''))[:100]
                parts.append({"text": f"[{btype}: {txt}]"})
        return parts
    s = str(content)
    return [{"text": s}] if s else []


_SCHEMA_MAP_KEYWORDS = ("properties", "patternProperties", "$defs", "definitions",
                        "dependentSchemas", "dependencies")
_SCHEMA_CHILD_KEYWORDS = ("items", "additionalProperties", "additionalItems", "contains",
                          "propertyNames", "unevaluatedProperties", "unevaluatedItems", "contentSchema")
_SCHEMA_LIST_KEYWORDS = ("allOf", "prefixItems")


def _schema_contains_anyof(schema):
    """Search schema positions only, never literal enum/default values."""
    if isinstance(schema, list):
        return any(_schema_contains_anyof(item) for item in schema)
    if not isinstance(schema, dict):
        return False
    if "anyOf" in schema:
        return True
    for keyword in _SCHEMA_MAP_KEYWORDS:
        mapping = schema.get(keyword)
        if isinstance(mapping, dict) and any(_schema_contains_anyof(value)
                                             for value in mapping.values()):
            return True
    return any(_schema_contains_anyof(schema[keyword]) for keyword in
               (*_SCHEMA_CHILD_KEYWORDS, *_SCHEMA_LIST_KEYWORDS, "not", "oneOf", "if", "then", "else")
               if keyword in schema)


def _claude_tool_schema(schema):
    """Remove anyOf from tool schemas for Google's Claude adapter.

    Its Claude endpoint rejects a valid Draft 2020-12 object union in
    skill_manage.operations.items. Relax unions rather than dropping valid
    arguments; the receiving tool still validates its own input.
    """
    if not isinstance(schema, dict):
        return schema

    result = dict(schema)
    for keyword in _SCHEMA_MAP_KEYWORDS:
        mapping = result.get(keyword)
        if isinstance(mapping, dict):
            result[keyword] = {name: _claude_tool_schema(value)
                               for name, value in mapping.items()}
    for keyword in _SCHEMA_CHILD_KEYWORDS:
        if keyword in result:
            result[keyword] = _claude_tool_schema(result[keyword])
    for keyword in _SCHEMA_LIST_KEYWORDS:
        if isinstance(result.get(keyword), list):
            result[keyword] = [_claude_tool_schema(value) for value in result[keyword]]

    # Widening inside negation, exclusive alternatives or an if-condition
    # could make a formerly valid argument invalid. Drop the whole constraint.
    if "not" in result:
        if _schema_contains_anyof(result["not"]):
            del result["not"]
        else:
            result["not"] = _claude_tool_schema(result["not"])
    if "oneOf" in result:
        if _schema_contains_anyof(result["oneOf"]):
            del result["oneOf"]
        elif isinstance(result["oneOf"], list):
            result["oneOf"] = [_claude_tool_schema(value) for value in result["oneOf"]]
    if _schema_contains_anyof(result.get("if")):
        for keyword in ("if", "then", "else"):
            result.pop(keyword, None)
    else:
        for keyword in ("if", "then", "else"):
            if keyword in result:
                result[keyword] = _claude_tool_schema(result[keyword])

    options = result.pop("anyOf", None)
    if not isinstance(options, list) or not options:
        return result

    variants = [_claude_tool_schema(option) for option in options]
    if not all(isinstance(option, dict) and option.get("type") == "object"
               and isinstance(option.get("properties", {}), dict)
               for option in variants):
        return result  # An unconstrained schema accepts all branches.

    props = dict(result.get("properties") or {})
    by_name = {}
    for option in variants:
        for name, prop in option.get("properties", {}).items():
            by_name.setdefault(name, []).append(prop)
    for name, definitions in by_name.items():
        if name in props:
            definitions.append(props[name])
        if all(definition == definitions[0] for definition in definitions):
            props[name] = definitions[0]
            continue
        # Keep only common constraints; a stricter schema could reject a
        # valid argument for one of the alternative operations.
        types = [definition.get("type") for definition in definitions
                 if isinstance(definition, dict)]
        merged = ({"type": types[0]} if len(types) == len(definitions)
                  and all(value == types[0] for value in types) else {})
        enums = [definition.get("enum") if isinstance(definition, dict) else None
                 for definition in definitions]
        if all(isinstance(values, list) for values in enums):
            values = []
            for alternatives in enums:
                if not isinstance(alternatives, list):
                    continue
                for value in alternatives:
                    if value not in values:
                        values.append(value)
            merged["enum"] = values
        props[name] = merged

    result["type"] = "object"
    result["properties"] = props
    common_required = set(variants[0].get("required", []))
    for option in variants[1:]:
        common_required.intersection_update(option.get("required", []))
    required = list(result.get("required", []))
    required.extend(name for name in variants[0].get("required", [])
                    if name in common_required and name not in required)
    if required:
        result["required"] = required
    if all(option.get("additionalProperties") is False
           and "patternProperties" not in option for option in variants):
        result["additionalProperties"] = False
    return result


def _openai_tools_to_gemini(tools: list, *, claude_model: bool = False) -> list:
    """Convert OpenAI tools array to Gemini functionDeclarations format.

    OpenAI:  [{"type":"function","function":{"name","description","parameters"}}]
    Gemini:  [{"functionDeclarations":[{"name","description","parameters"}]}]
    """
    if not tools:
        return []
    decls = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if not name:
            continue
        decl = {
            "name": name,
            "description": func.get("description", ""),
        }
        params = func.get("parameters")
        if isinstance(params, dict) and params:
            decl["parameters"] = _claude_tool_schema(params) if claude_model else params
        decls.append(decl)
    if not decls:
        return []
    return [{"functionDeclarations": decls}]


def _tool_choice_to_gemini(tool_choice) -> str | None:
    """Convert OpenAI tool_choice to Gemini toolConfig.functionCallingConfig.mode.

    Returns None if no transformation applies.
    """
    if tool_choice is None:
        return None
    if tool_choice == "auto":
        return "AUTO"
    if tool_choice == "none":
        return "NONE"
    if tool_choice == "required":
        return "ANY"
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        # Specific function forced — Gemini uses ANY + allowed_function_names.
        return "ANY"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Gemini-native passthrough (for invoice_reader.py / any Gemini SDK client)
# ──────────────────────────────────────────────────────────────────────────────

# Matches /v1beta/models/{model}:generateContent
_GEMINI_PASSTHROUGH_RE = re.compile(
    r"^/v1beta/models/(.+):generateContent$"
)


def _build_passthrough_envelope(model: str, gemini_body: dict) -> dict:
    """Wrap a native Gemini generateContent request in the Antigravity
    envelope. The caller's text-only body is used as ``request`` without
    message-format conversion; media parts are rejected.

    The model ID is sent to Antigravity unchanged.
    Safety settings are injected only when the caller did not supply its own.
    """
    _reject_image_data(gemini_body)
    inner = dict(gemini_body)  # shallow copy — don't mutate caller's dict

    if "safetySettings" not in inner:
        inner["safetySettings"] = [
            {"category": cat, "threshold": "BLOCK_NONE"}
            for cat in (
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            )
        ]

    project_id = get_project_id()
    return {
        "project": project_id,
        "model": model,
        "request": inner,
        "requestType": "agent",
        "userAgent": "antigravity",
        "requestId": f"agent-{uuid.uuid4().hex}",
    }


def _build_gemini_request(body: dict, backend_model: str, thinking_level = None):
    """Transform an OpenAI chat completion body into a Gemini generateContent
    request envelope. The model ID is sent upstream unchanged.
    """
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        messages = []

    system_texts = []
    contents = []

    # Map tool_call_id -> function name from assistant messages so tool results
    # always carry the correct function name even if OpenAI client omitted 'name'.
    tool_call_names = {}
    for msg in messages:
        if isinstance(msg, dict):
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    tcid = tc.get("id")
                    tname = (tc.get("function") or {}).get("name")
                    if tcid and tname:
                        tool_call_names[tcid] = tname

    # Collect system instruction pieces and convert messages.
    for msg in messages:
        role = msg.get("role", "user")

        if role == "system":
            text = "\n".join(p["text"] for p in _content_to_parts(msg.get("content")) if "text" in p)
            if text:
                system_texts.append(text)
            continue

        if role in ("tool", "function"):
            # OpenAI tool result message -> Gemini functionResponse part with role: 'user'.
            tool_call_id = msg.get("tool_call_id", "")
            # tool content handled as text
            content_text = ""
            for p in _content_to_parts(msg.get("content")):
                if "text" in p:
                    content_text += p["text"]
            # Try to parse JSON from the tool result; Gemini wants an object.
            response_obj = {}
            if content_text:
                try:
                    parsed = json.loads(content_text)
                    response_obj = parsed if isinstance(parsed, dict) else {"result": parsed}
                except (json.JSONDecodeError, TypeError):
                    response_obj = {"result": content_text}
            # Look up function name from preceding assistant tool_calls.
            fn_name = msg.get("name") or tool_call_names.get(tool_call_id)
            if not fn_name:
                # If only one tool call was registered, fall back to it.
                if len(tool_call_names) == 1:
                    fn_name = next(iter(tool_call_names.values()))
                else:
                    fn_name = "tool_result"
            # Decode the fc_id from the encoded tool_call_id (format: call_<fc_id>|<sig>).
            fc_response_id = ""
            if tool_call_id.startswith("call_"):
                raw = tool_call_id[5:]  # strip "call_"
                if "|" in raw:
                    fc_response_id = raw.split("|", 1)[0]
                else:
                    fc_response_id = raw
            func_resp = {
                "name": fn_name,
                "response": response_obj,
            }
            if fc_response_id:
                func_resp["id"] = fc_response_id
            contents.append({
                "role": "user",
                "parts": [{"functionResponse": func_resp}],
            })
            continue

        if role == "assistant":
            parts = _content_to_parts(msg.get("content"))
            # Prior tool calls from the assistant -> functionCall parts.
            # Hermes persists thought signatures in extra_content, not in tool-call IDs.
            tool_calls = msg.get("tool_calls") or []
            first_function_call = True
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                fn_name = fn.get("name", "")
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": str(args_str)}
                if fn_name:
                    tc_id = tc.get("id", "") or ""
                    fc_id = ""
                    thought_sig = ""
                    # Also accept legacy IDs from clients that kept the encoded suffix.
                    if tc_id.startswith("call_"):
                        raw = tc_id[5:]
                        if "|" in raw:
                            fc_id, thought_sig = raw.split("|", 1)
                        else:
                            fc_id = raw
                    extra = tc.get("extra_content")
                    if isinstance(extra, dict):
                        google = extra.get("google")
                        candidate = google.get("thought_signature") if isinstance(google, dict) else None
                        if candidate is None:
                            candidate = extra.get("thought_signature")
                        if isinstance(candidate, str) and candidate:
                            thought_sig = candidate
                    fc_obj = {"name": fn_name, "args": args}
                    if fc_id:
                        fc_obj["id"] = fc_id
                    part_obj: dict = {"functionCall": fc_obj}
                    # thoughtSignature goes as a sibling key on the part, not inside functionCall.
                    # For parallel calls Google signs only the first functionCall.
                    # A missing signature on later calls must remain absent.
                    if thought_sig:
                        part_obj["thoughtSignature"] = thought_sig
                    elif first_function_call:
                        part_obj["thoughtSignature"] = "skip_thought_signature_validator"
                    parts.append(part_obj)
                    first_function_call = False
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        if role == "developer":
            # Treat developer role like system.
            text = "\n".join(p["text"] for p in _content_to_parts(msg.get("content")) if "text" in p)
            if text:
                system_texts.append(text)
            continue

        # Default: user role.
        parts = _content_to_parts(msg.get("content"))
        # Some clients send tool_calls on user messages; handle gracefully.
        for tc in (msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            fn_name = fn.get("name", "")
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except (json.JSONDecodeError, TypeError):
                args = {"raw": str(args_str)}
            if fn_name:
                parts.append({"functionCall": {"name": fn_name, "args": args}})
        if parts:
            contents.append({"role": "user", "parts": parts})

    # Build system instruction: use user's system messages if provided,
    # otherwise fall back to the default.
    if system_texts:
        system_instruction = {"parts": [{"text": "\n\n".join(s for s in system_texts if s)}]}
    else:
        system_instruction = {"parts": [{"text": DEFAULT_SYSTEM_INSTRUCTION}]}

    # Build the inner Gemini request.
    inner = {
        "contents": contents,
        "systemInstruction": system_instruction,
    }

    # Tools.
    tools = _openai_tools_to_gemini(body.get("tools"),
                                   claude_model=backend_model.startswith("claude-"))
    if tools:
        inner["tools"] = tools

    # Tool choice -> toolConfig.
    tool_choice = body.get("tool_choice")
    mode = _tool_choice_to_gemini(tool_choice)
    if mode:
        tc_config = {"functionCallingConfig": {"mode": mode}}
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            fn_name = (tool_choice.get("function") or {}).get("name")
            if fn_name:
                tc_config["functionCallingConfig"]["allowedFunctionNames"] = [fn_name]
        inner["toolConfig"] = tc_config

    # Generation config.
    gen_config = {}
    if "temperature" in body and body["temperature"] is not None:
        gen_config["temperature"] = float(body["temperature"])
    if "top_p" in body and body["top_p"] is not None:
        gen_config["topP"] = float(body["top_p"])
    if "max_tokens" in body and body["max_tokens"] is not None:
        gen_config["maxOutputTokens"] = int(body["max_tokens"])
    elif "max_completion_tokens" in body and body["max_completion_tokens"] is not None:
        gen_config["maxOutputTokens"] = int(body["max_completion_tokens"])
    if "presence_penalty" in body and body["presence_penalty"] is not None:
        gen_config["presencePenalty"] = float(body["presence_penalty"])
    if "frequency_penalty" in body and body["frequency_penalty"] is not None:
        gen_config["frequencyPenalty"] = float(body["frequency_penalty"])
    if "stop" in body and body["stop"]:
        stops = body["stop"]
        if isinstance(stops, str):
            stops = [stops]
        gen_config["stopSequences"] = stops
    if "seed" in body and body["seed"] is not None:
        gen_config["seed"] = int(body["seed"])
    # Enable thinking for the thinking variant of claude-opus.
    if backend_model == "claude-opus-4-6-thinking":
        thinking_cfg = gen_config.get("thinkingConfig", {})
        thinking_cfg.setdefault("includeThoughts", False)
        gen_config["thinkingConfig"] = thinking_cfg
    # Apply an explicitly supplied thinking level to OpenAI-format requests.
    if thinking_level:
        gen_config.setdefault("thinkingConfig", {})["thinkingLevel"] = thinking_level
    if gen_config:
        inner["generationConfig"] = gen_config

    # Safety settings: relax standard categories so coding tasks aren't blocked.
    inner["safetySettings"] = []

    # Build the wrapped envelope.
    project_id = get_project_id()
    envelope = {
        "project": project_id,
        "model": backend_model,
        "request": inner,
        "requestType": "agent",
        "userAgent": "antigravity",
        "requestId": f"agent-{uuid.uuid4().hex}",
    }
    return envelope


# ──────────────────────────────────────────────────────────────────────────────
# Gemini -> OpenAI response transformation
# ──────────────────────────────────────────────────────────────────────────────

def _declared_tool_names(envelope: dict) -> set:
    """Only translate textual calls for tools the client actually offered."""
    request = envelope.get("request", {})
    config = request.get("toolConfig", {}).get("functionCallingConfig", {})
    if config.get("mode") == "NONE":
        return set()
    names = {
        declaration["name"]
        for group in request.get("tools", [])
        for declaration in group.get("functionDeclarations", [])
        if declaration.get("name")
    }
    restricted = config.get("allowedFunctionNames")
    return names.intersection(restricted) if restricted is not None else names


def _parse_text_tool_call(text: str, allowed_tool_names: set):
    """Recognize a complete model-emitted call, never prose containing one."""
    if not allowed_tool_names or not text or len(text) > 32768:
        return None
    match = re.fullmatch(
        r"\s*call:(?:default_api:)?([A-Za-z_][A-Za-z0-9_-]*)\s*(\{.*\})\s*",
        text, re.DOTALL,
    )
    if not match or match.group(1) not in allowed_tool_names:
        return None
    name, payload = match.groups()
    try:
        args = json.loads(payload)
    except json.JSONDecodeError:
        # Some Gemini responses use call:default_api:terminal{command:pwd}
        # instead of JSON. Limit this shorthand to terminal's sole command field.
        if name != "terminal":
            return None
        command_match = re.fullmatch(r"\{\s*command\s*:\s*(.*?)\s*\}", payload, re.DOTALL)
        if not command_match or not command_match.group(1):
            return None
        command = command_match.group(1)
        if command.startswith('"') and command.endswith('"'):
            try:
                command = json.loads(command)
            except json.JSONDecodeError:
                return None
        args = {"command": command}
    if not isinstance(args, dict):
        return None
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _extract_parts(parts: list, allowed_tool_names=None) -> tuple[str, list, str]:
    """Extract (text, tool_calls, finish_reason_hint) from a Gemini parts list.

    Keep the legacy ID suffix for other clients, and use extra_content for Hermes replay.
    """
    text_chunks = []
    tool_calls = []
    finish_hint = None

    for part in parts or []:
        if not isinstance(part, dict):
            continue
        if "text" in part and part["text"] and not part.get("thought"):
            text_chunks.append(part["text"])
        if "functionCall" in part:
            fc = part["functionCall"] or {}
            name = fc.get("name", "")
            args = fc.get("args", {})
            fc_id = fc.get("id", "")
            # thoughtSignature is a sibling of functionCall in the part, not inside it.
            thought_sig = part.get("thoughtSignature") or part.get("thought_signature") or ""
            if name and (allowed_tool_names is None or name in allowed_tool_names):
                call = {
                    "id": f"call_{fc_id}|{thought_sig}" if (fc_id or thought_sig) else "",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args) if isinstance(args, (dict, list)) else str(args),
                    },
                }
                if isinstance(thought_sig, str) and thought_sig:
                    call["extra_content"] = {"google": {"thought_signature": thought_sig}}
                tool_calls.append(call)
        if "executableCode" in part:
            ec = part["executableCode"] or {}
            code = ec.get("code", "")
            if code:
                text_chunks.append(f"```python\n{code}\n```")
        if "codeExecutionResult" in part:
            cer = part["codeExecutionResult"] or {}
            out = cer.get("output", "")
            if out:
                text_chunks.append(f"```\n{out}\n```")
    text = "".join(text_chunks)
    if not tool_calls:
        textual_call = _parse_text_tool_call(text, allowed_tool_names)
        if textual_call:
            tool_calls.append(textual_call)
            text = ""
    if tool_calls:
        finish_hint = "tool_calls"
    return text, tool_calls, finish_hint


def _finish_reason_from_gemini(candidate: dict, tool_calls: list) -> str:
    """Map Gemini finishReason / stopReason to OpenAI finish_reason."""
    fr = (candidate.get("finishReason") or candidate.get("stopReason") or "").upper()
    if fr == "MALFORMED_FUNCTION_CALL":
        raise UpstreamError("Upstream generated an invalid function call", status=502)
    if fr == "MAX_TOKENS":
        return "length"
    if tool_calls:
        return "tool_calls"
    mapping = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "BLOCKLIST": "content_filter",
        "PROHIBITED_CONTENT": "content_filter",
        "SPII": "content_filter",
        "IMAGE_SAFETY": "content_filter",
        "LANGUAGE": "content_filter",
        "OTHER": "stop",
    }
    return mapping.get(fr, "stop")


def _extract_usage(response: dict) -> dict:
    """Extract OpenAI-format usage from a Gemini response."""
    usage = response.get("usageMetadata") or {}
    if not usage:
        cands = response.get("candidates") or []
        if cands:
            usage = cands[0].get("usageMetadata") or {}
    pt = usage.get("promptTokenCount", 0) or 0
    ct = usage.get("candidatesTokenCount", 0) or usage.get("completionTokenCount", 0) or 0
    tt = usage.get("totalTokenCount", (pt + ct)) or (pt + ct)
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
    }


def gemini_response_to_openai(response: dict, model: str, allowed_tool_names=None) -> dict:
    """Transform an unwrapped Gemini generateContent response to OpenAI
    chat.completion format.
    """
    candidates = response.get("candidates") or []
    text = ""
    tool_calls = []
    finish_reason = "stop"

    if candidates:
        cand = candidates[0]
        finish_reason = _finish_reason_from_gemini(cand, [])
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        text, tool_calls, _ = _extract_parts(parts, allowed_tool_names)
        if tool_calls:
            finish_reason = _finish_reason_from_gemini(cand, tool_calls)
        # Assign tool call ids.
        for i, tc in enumerate(tool_calls):
            if not tc["id"]:
                tc["id"] = f"call_{uuid.uuid4().hex[:24]}"
    else:
        # No candidates — possibly a promptFeedback block.
        pf = response.get("promptFeedback") or {}
        br = (pf.get("blockReason") or "").upper()
        if br:
            finish_reason = "content_filter"
            text = f"[Content blocked: {br}]"

    message = {"role": "assistant", "content": text if text else None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": _extract_usage(response),
    }


def _extract_delta_from_parts(parts: list, prev_tool_calls: dict, allowed_tool_names=None) -> tuple[str, list, dict]:
    """Extract text and tool calls, assigning one index per distinct call."""
    text_chunks = []
    tool_call_deltas = []
    current_calls = dict(prev_tool_calls)

    for part in parts or []:
        if not isinstance(part, dict):
            continue
        if "text" in part and part["text"] and not part.get("thought"):
            text_chunks.append(part["text"])
        if "functionCall" in part:
            fc = part["functionCall"] or {}
            name = fc.get("name", "")
            args = fc.get("args", {})
            fc_id = fc.get("id", "")
            thought_sig = part.get("thoughtSignature") or part.get("thought_signature") or ""
            if name and (allowed_tool_names is None or name in allowed_tool_names):
                key = ("id", fc_id) if fc_id else ("generated", uuid.uuid4().hex)
                if key in current_calls:
                    continue  # repeated complete functionCall in a later snapshot
                index = len(current_calls)
                current_calls[key] = index
                call = {
                    "index": index,
                    "id": f"call_{fc_id}|{thought_sig}" if (fc_id or thought_sig) else f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args) if isinstance(args, (dict, list)) else str(args),
                    },
                }
                if isinstance(thought_sig, str) and thought_sig:
                    call["extra_content"] = {"google": {"thought_signature": thought_sig}}
                tool_call_deltas.append(call)

    return "".join(text_chunks), tool_call_deltas, current_calls


def _merge_function_snapshot(part: dict, accumulated: list, indices: dict,
                             no_id_slot: int | None = None) -> None:
    """Merge a repeated call by ID or by its position when Google omits IDs."""
    fc = part.get("functionCall") if isinstance(part, dict) else None
    if not isinstance(fc, dict):
        accumulated.append(part)
        return
    fc_id = fc.get("id")
    key = fc_id or (("no_id", no_id_slot) if no_id_slot is not None else None)
    index = indices.get(key) if key is not None else None
    if index is not None and not fc_id:
        previous = accumulated[index]
        previous_sig = previous.get("thoughtSignature") or previous.get("thought_signature")
        latest_sig = part.get("thoughtSignature") or part.get("thought_signature")
        if previous_sig and not latest_sig and previous["functionCall"].get("args") != fc.get("args"):
            index = None  # a new unsigned parallel call after a signed first call
    if index is None or (not fc_id and accumulated[index]["functionCall"].get("name") != fc.get("name")):
        if key is not None:
            indices[key] = len(accumulated)
        accumulated.append(part)
        return
    previous = accumulated[index]
    latest = dict(part)
    latest["functionCall"] = {**previous["functionCall"], **fc}
    if not (latest.get("thoughtSignature") or latest.get("thought_signature")):
        sig = previous.get("thoughtSignature") or previous.get("thought_signature")
        if sig:
            latest["thoughtSignature"] = sig
    accumulated[index] = latest


# ──────────────────────────────────────────────────────────────────────────────
# Upstream API calls
# ──────────────────────────────────────────────────────────────────────────────

class ShapeTrace:
    """Bounded, content-free response metadata for correlating proxy and Hermes turns."""

    _FINISH_REASONS = {
        "STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "BLOCKLIST",
        "PROHIBITED_CONTENT", "SPII", "MALFORMED_FUNCTION_CALL",
        "IMAGE_SAFETY", "LANGUAGE", "OTHER",
    }

    def __init__(self, envelope: dict, stream: bool):
        request = envelope.get("request") or {}
        groups = request.get("tools") or []
        declarations = [decl for group in groups if isinstance(group, dict)
                        for decl in group.get("functionDeclarations", []) if isinstance(decl, dict)]
        mode = (request.get("toolConfig") or {}).get("functionCallingConfig", {}).get("mode")
        self.allowed = _declared_tool_names(envelope)
        self.data = {
            "stream": stream,
            "offered_tools": len(declarations),
            "mode": mode if mode in ("AUTO", "ANY", "NONE") else "UNSET",
            "input_contents": len(request.get("contents") or []),
            "system_chars": sum(len(part.get("text", "")) for part in
                                (request.get("systemInstruction") or {}).get("parts", [])
                                if isinstance(part, dict) and isinstance(part.get("text"), str)),
            "events": 0,
            "upstream_candidates": 0,
            "upstream_candidate_indices": [],
            "finish_reasons": [],
            "upstream_visible_text_parts": 0,
            "upstream_thought_parts": 0,
            "upstream_signature_parts": 0,
            "upstream_function_parts": 0,
            "upstream_allowed_function_parts": 0,
            "upstream_nonfirst_function_parts": 0,
            "prompt_blocked": False,
        }

    def observe(self, event: dict):
        try:
            self._observe(event)
        except Exception:
            pass  # Malformed metadata must not interrupt a valid first candidate.

    def _observe(self, event: dict):
        if not isinstance(event, dict):
            return
        response = event.get("response", event)
        if not isinstance(response, dict):
            return
        self.data["events"] += 1
        self.data["prompt_blocked"] |= bool((response.get("promptFeedback") or {}).get("blockReason"))
        for index, candidate in enumerate(response.get("candidates") or []):
            if not isinstance(candidate, dict):
                continue
            self.data["upstream_candidates"] += 1
            candidate_index = candidate.get("index", index)
            indices = self.data["upstream_candidate_indices"]
            if type(candidate_index) is int and 0 <= candidate_index <= 32 and candidate_index not in indices:
                indices.append(candidate_index)
            finish = candidate.get("finishReason") or candidate.get("stopReason")
            if finish:
                sanitized = finish if isinstance(finish, str) and finish in self._FINISH_REASONS else "UNKNOWN"
                reasons = self.data["finish_reasons"]
                if sanitized not in reasons:
                    reasons.append(sanitized)
            content = candidate.get("content") or {}
            if not isinstance(content, dict):
                continue
            parts = content.get("parts") or []
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                self.data["upstream_thought_parts"] += bool(part.get("thought"))
                self.data["upstream_visible_text_parts"] += bool(part.get("text") and not part.get("thought"))
                self.data["upstream_signature_parts"] += bool(part.get("thoughtSignature") or part.get("thought_signature"))
                if "functionCall" in part:
                    self.data["upstream_function_parts"] += 1
                    self.data["upstream_nonfirst_function_parts"] += index != 0
                    call = part.get("functionCall") or {}
                    if isinstance(call, dict):
                        self.data["upstream_allowed_function_parts"] += call.get("name") in self.allowed

    def emit(self, completion_id: str, finish_reason: str, output_tool_calls: int):
        try:
            data = dict(self.data, completion_id=completion_id,
                        finish_reason=finish_reason if finish_reason in
                        ("stop", "tool_calls", "length", "content_filter", "upstream_empty_response",
                         "upstream_malformed_function_call") else "unknown",
                        output_tool_calls=output_tool_calls)
            _log("shape " + json.dumps(data, sort_keys=True, separators=(",", ":")))
        except Exception:
            pass  # Diagnostic logging must never change a model response.

def _make_upstream_request(url: str, envelope: dict, streaming: bool = False):
    """Build and return a urllib Request object for the upstream call."""
    body = json.dumps(envelope).encode()
    req = Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {get_access_token()}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", ANTIGRAVITY_USER_AGENT)
    req.add_header("X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1")
    req.add_header("Client-Metadata", CLIENT_METADATA)
    if streaming:
        req.add_header("Accept", "text/event-stream")
    return req


def call_generate_content(envelope: dict, trace: ShapeTrace = None) -> dict:
    """Non-streaming call. Collects SSE chunks and returns combined response dict."""
    chunks = list(stream_generate_content(envelope))
    if trace is not None:
        for chunk in chunks:
            trace.observe(chunk)
    if not chunks:
        raise UpstreamError("Empty response from upstream stream", status=502)

    combined_parts = []
    function_parts_by_id = {}
    last_finish_reason = None
    total_prompt_tokens = 0
    total_candidates_tokens = 0
    model_version = ""
    prompt_feedback = None

    for chunk in chunks:
        # Each SSE chunk contains {"response": {...}} or direct {...}
        resp = chunk.get("response") if isinstance(chunk, dict) and "response" in chunk else chunk
        if not isinstance(resp, dict):
            continue
        if resp.get("promptFeedback"):
            prompt_feedback = resp["promptFeedback"]
        candidates = resp.get("candidates") or []
        for cand in candidates:
            content = cand.get("content") or {}
            parts = content.get("parts") or []
            no_id_slot = 0
            for part in parts:
                fc = part.get("functionCall") if isinstance(part, dict) else None
                slot = None
                if isinstance(fc, dict) and not fc.get("id"):
                    slot = no_id_slot
                    no_id_slot += 1
                _merge_function_snapshot(part, combined_parts, function_parts_by_id, slot)
            reason = cand.get("finishReason") or cand.get("stopReason")
            if reason and (str(reason).upper() == "MALFORMED_FUNCTION_CALL"
                           or str(last_finish_reason).upper() != "MALFORMED_FUNCTION_CALL"):
                last_finish_reason = reason
        if resp.get("modelVersion"):
            model_version = resp.get("modelVersion")
        um = resp.get("usageMetadata") or {}
        if um.get("promptTokenCount"):
            total_prompt_tokens = um.get("promptTokenCount")
        if um.get("candidatesTokenCount"):
            total_candidates_tokens = um.get("candidatesTokenCount")

    if not last_finish_reason and not (prompt_feedback or {}).get("blockReason"):
        raise UpstreamError("Upstream stream ended without finishReason", status=502)
    total_prompt = total_prompt_tokens or 0
    total_candidates = total_candidates_tokens or 0
    resp_dict = {
        "candidates": ([{
            "content": {"role": "model", "parts": combined_parts},
            "finishReason": last_finish_reason,
        }] if last_finish_reason else []),
        "promptFeedback": prompt_feedback or {},
        "modelVersion": model_version,
        "usageMetadata": {
            "promptTokenCount": total_prompt,
            "candidatesTokenCount": total_candidates,
            "totalTokenCount": total_prompt + total_candidates,
        }
    }
    return resp_dict


def stream_generate_content(envelope: dict):
    """Streaming call. Yields parsed JSON event objects from the SSE stream."""
    req = _make_upstream_request(STREAM_GENERATE_CONTENT_URL, envelope, streaming=True)
    try:
        resp = urlopen(req, timeout=UPSTREAM_TIMEOUT)
    except HTTPError as e:
        _capture_upstream_error(e, envelope)
        _log(f"Upstream HTTP {e.code} on streamGenerateContent")
        e.close()
        raise UpstreamError(f"Upstream stream HTTP {e.code}", status=e.code)
    except URLError as e:
        raise UpstreamError(f"Upstream stream network error: {e}", status=502)

    def parse_event(lines):
        if not lines:
            return None
        data = "\n".join(lines)
        if data.strip() == "[DONE]":
            return {"__done__": True}
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            raise UpstreamError("Invalid upstream SSE event", status=502)
        if isinstance(event, dict) and "error" in event:
            error = event["error"]
            code = error.get("code") if isinstance(error, dict) else None
            status = code if isinstance(code, int) and 400 <= code <= 599 else 502
            raise UpstreamError(f"Upstream SSE HTTP {status}", status=status)
        return event

    data_lines = []
    try:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
            line = line.rstrip("\r\n")
            if line == "":
                event = parse_event(data_lines)
                data_lines = []
                if event is not None:
                    yield event
                    if event.get("__done__"):
                        break
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
        else:
            event = parse_event(data_lines)
            if event is not None:
                yield event
    finally:
        resp.close()


def _try_parse_one_event(buffer: str):
    """Attempt to parse one SSE event or JSON array element from the buffer.

    Returns (event_obj_or_None, bytes_consumed) or None if incomplete.
    The event_obj may be None when it's a comment/heartbeat line.
    """
    # SSE "data:" lines.
    if "data:" in buffer[:10] or buffer.startswith("\n") or buffer.startswith(":"):
        # Find end of this SSE block (double newline).
        for sep in ("\n\n", "\r\n\r\n"):
            idx = buffer.find(sep)
            if idx != -1:
                block = buffer[:idx]
                consumed = idx + len(sep)
                obj = _parse_sse_block(block)
                return (obj, consumed)
        # Maybe single newline-terminated line (some servers use \n not \n\n).
        nl = buffer.find("\n")
        if nl != -1 and buffer.strip().startswith("data:"):
            # Only consume if we have a complete data line.
            line = buffer[:nl]
            consumed = nl + 1
            obj = _parse_sse_block(line)
            return (obj, consumed)
        return None  # incomplete

    # Bare JSON array stream: [ {...}, {...} ]
    if buffer.lstrip().startswith("["):
        obj, consumed = _try_parse_json_element_in_array(buffer)
        if obj is not None or consumed > 0:
            return (obj, consumed)
        return None

    # Bare JSON object stream: {...}{...}
    if buffer.lstrip().startswith("{"):
        obj, consumed = _try_parse_one_json_object(buffer)
        if obj is not None:
            return (obj, consumed)
        return None

    # Unknown prefix; skip a line to resync.
    nl = buffer.find("\n")
    if nl != -1:
        return (None, nl + 1)
    return None


def _parse_sse_block(block: str):
    """Parse an SSE block (one or more lines) into a JSON object or None."""
    data_lines = []
    for line in block.split("\n"):
        line = line.rstrip("\r")
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif line.startswith(":"):
            continue  # comment / heartbeat
        elif line.startswith("event:") or line.startswith("id:") or line.startswith("retry:"):
            continue
        elif line.strip() == "":
            continue
        else:
            # Unexpected line; ignore.
            pass
    if not data_lines:
        return None
    data_str = "\n".join(data_lines).strip()
    if data_str == "[DONE]":
        return {"__done__": True}
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        return None


def _try_parse_json_element_in_array(buffer: str):
    """For bare JSON array streams like [{...},{...}]. Parse one element."""
    s = buffer.lstrip()
    if not s.startswith("["):
        # Maybe we already consumed the opening bracket.
        pass
    # We need to find a complete top-level object. Use brace counting.
    obj, consumed = _try_parse_one_json_object(buffer)
    return (obj, consumed)


def _try_parse_one_json_object(buffer: str):
    """Parse one complete top-level JSON object from buffer using brace counting.

    Returns (obj, consumed_chars) or (None, 0) if incomplete.
    """
    # Find first '{'.
    start = buffer.find("{")
    if start == -1:
        return (None, 0)
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(buffer)):
        ch = buffer[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = buffer[start:i + 1]
                try:
                    obj = json.loads(candidate)
                    return (obj, i + 1)
                except json.JSONDecodeError:
                    # Keep scanning.
                    continue
    return (None, 0)


def _parse_remaining(buffer: str):
    """Parse any remaining JSON objects in the buffer at stream end."""
    s = buffer.strip()
    if not s:
        return
    # SSE.
    for line in s.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                yield {"__done__": True}
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                pass
    # Bare JSON.
    if s.startswith("[") or s.startswith("{"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                for item in obj:
                    yield item
            else:
                yield obj
        except json.JSONDecodeError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Error helper
# ──────────────────────────────────────────────────────────────────────────────

class UpstreamError(Exception):
    def __init__(self, message, status=502, body=""):
        super().__init__(message)
        self.status = status
        self.body = body


def _openai_error(message: str, err_type: str = "api_error", code=None, status: int = 500):
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": None,
            "code": code,
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
# HTTP handler
# ──────────────────────────────────────────────────────────────────────────────

class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "AntigravityProxy/2.0"

    def log_message(self, fmt, *args):
        # We do our own logging.
        pass

    def _send_json(self, code, data):
        self._response_status = code
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error_json(self, code, message, err_type="api_error", http_code=None):
        self._send_json(http_code or code, _openai_error(message, err_type, code=code))

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None  # signal parse failure

    # ── GET routes ────────────────────────────────────────────────────────────

    def do_GET(self):
        t0 = time.time()
        path = urlparse(self.path).path
        try:
            if path == "/v1/models" or path == "/models":
                self._handle_models()
                self._log_req("GET", path, getattr(self, "_response_status", 500), time.time() - t0)
            elif path in ("/v1/quota", "/quota"):
                self._handle_quota()
                self._log_req("GET", path, getattr(self, "_response_status", 500), time.time() - t0)
            elif path in ("/health", "/"):
                self._send_json(200, {"status": "ok", "service": "antigravity-proxy"})
                self._log_req("GET", path, 200, time.time() - t0)
            else:
                self._send_error_json("not_found", "Not found", err_type="invalid_request_error", http_code=404)
                self._log_req("GET", path, 404, time.time() - t0)
        except Exception as e:
            _log(f"GET {path} error: {e}\n{traceback.format_exc()}")
            try:
                self._send_error_json("internal_error", str(e), http_code=500)
            except Exception:
                pass
            self._log_req("GET", path, 500, time.time() - t0)

    def _handle_models(self):
        try:
            access_token = get_access_token()
        except RuntimeError:
            self._send_error_json("auth_error", "OAuth token unavailable", http_code=401)
            return
        try:
            model_ids = get_available_models(access_token)
        except HTTPError as error:
            status = error.code if 400 <= error.code <= 599 else 502
            error.close()
            self._send_error_json("models_upstream_error", f"Model service returned HTTP {status}", http_code=status)
            return
        except Exception:
            self._send_error_json("models_upstream_error", "Model service unavailable", http_code=502)
            return
        now = int(time.time())
        models = []
        for name in model_ids:
            models.append({
                "id": name,
                "object": "model",
                "created": now,
                "owned_by": "google",
            })
        self._send_json(200, {"object": "list", "data": models})

    def _handle_quota(self):
        try:
            access_token = get_access_token()
        except RuntimeError:
            self._send_error_json("auth_error", "OAuth token unavailable", http_code=401)
            return
        try:
            result = fetch_user_quota(access_token)
            groups = fetch_quota_groups(access_token)
            if (any(b["model"].startswith("gemini-") for b in result["buckets"])
                    and not any(g["name"].lower().startswith("gemini") for g in groups)):
                raise ValueError("Gemini quota groups missing from summary")
            result["groups"] = groups
            result["groups_source"] = "retrieveUserQuotaSummary"
        except HTTPError as error:
            status = error.code if 400 <= error.code <= 599 else 502
            error.close()
            self._send_error_json("quota_upstream_error", f"Quota service returned HTTP {status}", http_code=status)
            return
        except Exception:
            self._send_error_json("quota_upstream_error", "Quota service unavailable", http_code=502)
            return
        self._send_json(200, result)

    # ── POST routes ───────────────────────────────────────────────────────────

    def do_POST(self):
        t0 = time.time()
        path = urlparse(self.path).path
        try:
            if path in ("/v1/chat/completions", "/chat/completions"):
                self._handle_chat()
                self._log_req("POST", path, getattr(self, "_response_status", 200), time.time() - t0)
            elif _GEMINI_PASSTHROUGH_RE.match(path):
                self._handle_gemini_passthrough(path)
                self._log_req("POST", path, getattr(self, "_response_status", 200), time.time() - t0)
            else:
                self._send_error_json("not_found", "Not found", err_type="invalid_request_error", http_code=404)
                self._log_req("POST", path, 404, time.time() - t0)
        except Exception as e:
            _log(f"POST {path} error: {e}\n{traceback.format_exc()}")
            try:
                self._send_error_json("internal_error", str(e), http_code=500)
            except Exception:
                pass
            self._log_req("POST", path, 500, time.time() - t0)

    def _log_req(self, method, path, status, duration):
        _log(f"{method} {path} {status} {duration:.3f}s")

    # ── Chat completions ──────────────────────────────────────────────────────

    def _handle_gemini_passthrough(self, path: str):
        """Handle a native Gemini generateContent request, wrapping it in the
        Antigravity envelope and returning the raw Gemini response (unwrapped).
        Text-only requests can use responseSchema through the OAuth proxy.
        Media parts are rejected before calling the upstream API."""
        body = self._read_body()
        if body is None or not isinstance(body, dict):
            self._send_error_json("invalid_request", "Invalid JSON body", err_type="invalid_request_error", http_code=400)
            return

        model = _GEMINI_PASSTHROUGH_RE.match(path).group(1)

        try:
            envelope = _build_passthrough_envelope(model, body)
        except UpstreamError as e:
            self._send_error_json("upstream_error", str(e), err_type="api_error", http_code=e.status)
            return
        except RuntimeError as e:
            self._send_error_json("auth_error", str(e), err_type="authentication_error", http_code=401)
            return
        except Exception as e:
            self._send_error_json("request_error", f"Failed to build request: {e}", err_type="invalid_request_error", http_code=400)
            return

        try:
            gemini_resp = call_generate_content(envelope)
        except UpstreamError as e:
            self._send_error_json("upstream_error", str(e), err_type="api_error", http_code=e.status)
            return
        except Exception as e:
            self._send_error_json("upstream_error", f"Upstream call failed: {e}", err_type="api_error", http_code=502)
            return

        self._send_json(200, gemini_resp)

    def _handle_chat(self):
        body = self._read_body()
        if body is None:
            self._send_error_json("invalid_request", "Invalid JSON body", err_type="invalid_request_error", http_code=400)
            return
        if not isinstance(body, dict):
            self._send_error_json("invalid_request", "Request body must be a JSON object", err_type="invalid_request_error", http_code=400)
            return

        openai_model = body.get("model") or DEFAULT_MODEL

        try:
            envelope = _build_gemini_request(body, openai_model)
        except UpstreamError as e:
            self._send_error_json("upstream_error", str(e), err_type="api_error", http_code=e.status)
            return
        except RuntimeError as e:
            self._send_error_json("auth_error", str(e), err_type="authentication_error", http_code=401)
            return
        except Exception as e:
            self._send_error_json("request_error", f"Failed to build request: {e}", err_type="invalid_request_error", http_code=400)
            return

        inner = envelope.get("request", {})
        if not inner.get("contents"):
            self._send_error_json("invalid_request", "No messages provided", err_type="invalid_request_error", http_code=400)
            return

        stream = bool(body.get("stream", False))
        if stream:
            self._handle_stream(envelope, openai_model)
        else:
            self._handle_nonstream(envelope, openai_model)

    def _handle_nonstream(self, envelope, openai_model):
        trace = ShapeTrace(envelope, stream=False)
        try:
            gemini_resp = call_generate_content(envelope, trace=trace)
        except UpstreamError as e:
            self._send_error_json("upstream_error", str(e), err_type="api_error", http_code=e.status)
            return
        except Exception as e:
            self._send_error_json("upstream_error", f"Upstream call failed: {e}", err_type="api_error", http_code=502)
            return

        try:
            openai_resp = gemini_response_to_openai(
                gemini_resp, openai_model, _declared_tool_names(envelope)
            )
        except UpstreamError as e:
            trace.emit(f"chatcmpl-{uuid.uuid4().hex[:24]}", "upstream_malformed_function_call", 0)
            self._send_error_json("upstream_malformed_function_call", str(e), http_code=e.status)
            return
        except Exception as e:
            _log(f"Response transform error: {e}\n{traceback.format_exc()}")
            # Fallback: return raw text if we can find any.
            try:
                text = (gemini_resp.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", ""))
            except Exception:
                text = ""
            openai_resp = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": openai_model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text or "[error: could not parse response]"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        choice = openai_resp["choices"][0]
        message = choice["message"]
        text = (message.get("content") or "").strip()
        if choice["finish_reason"] == "stop" and not message.get("tool_calls") and not text:
            trace.emit(openai_resp["id"], "upstream_empty_response", 0)
            self._send_error_json("upstream_empty_response", "Upstream returned no visible content", err_type="api_error", http_code=502)
            return
        trace.emit(openai_resp["id"], choice["finish_reason"], len(message.get("tool_calls") or []))
        self._send_json(200, openai_resp)

    def _handle_stream(self, envelope, openai_model):
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        trace = ShapeTrace(envelope, stream=True)
        created = int(time.time())

        # Resolve the lazy upstream generator before committing HTTP 200.
        allowed_tool_names = _declared_tool_names(envelope)
        try:
            gen = iter(stream_generate_content(envelope))
            first_event = next(gen)
        except StopIteration:
            self._send_error_json("upstream_error", "Empty upstream response", http_code=502)
            return
        except UpstreamError as e:
            self._send_error_json("upstream_error", str(e), http_code=e.status)
            return
        except Exception as e:
            _log(f"Upstream start failed: {type(e).__name__}")
            self._send_error_json("upstream_error", "Upstream request failed", http_code=502)
            return

        # Send SSE headers only after the upstream yielded its first event.
        try:
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
        except Exception:
            gen.close()
            return

        def send_sse(obj, event=None):
            prefix = f"event: {event}\n" if event else ""
            data = f"{prefix}data: {json.dumps(obj)}\n\n"
            try:
                self.wfile.write(data.encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return False
            return True

        # Initial role chunk.
        first_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": openai_model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }],
        }
        if not send_sse(first_chunk):
            return

        finish_reason = "stop"
        upstream_finished = False
        blocked_reason = None
        seen_tool_calls = {}
        pending_tool_parts = []
        pending_tool_by_id = {}
        textual_calls_emitted = 0
        pending_text = ""
        checking_text_call = True
        total_usage = None
        events_processed = 0
        emitted_visible = False

        try:
            for event in itertools.chain((first_event,), gen):
                if not isinstance(event, dict):
                    continue
                if event.get("__done__"):
                    break

                trace.observe(event)
                events_processed += 1

                # Unwrap envelope if present.
                resp = event.get("response", event)

                # Check for usage metadata (often on the last chunk).
                usage_meta = resp.get("usageMetadata")
                if usage_meta:
                    total_usage = usage_meta

                feedback = resp.get("promptFeedback") or {}
                if feedback.get("blockReason"):
                    blocked_reason = feedback["blockReason"]
                candidates = resp.get("candidates") or []
                if not candidates:
                    continue

                if any(isinstance(item, dict) and str(
                    item.get("finishReason") or item.get("stopReason") or ""
                ).upper() == "MALFORMED_FUNCTION_CALL" for item in candidates):
                    trace.emit(completion_id, "upstream_malformed_function_call", len(seen_tool_calls))
                    send_sse(_openai_error("Upstream generated an invalid function call",
                                           code="upstream_malformed_function_call"), event="error")
                    return
                cand = candidates[0]
                fr = cand.get("finishReason") or cand.get("stopReason")
                content = cand.get("content") or {}
                parts = content.get("parts") or []

                # Function calls are complete snapshots, not argument deltas. Wait
                # for finishReason so later snapshots can supply the final args or
                # a thought signature without duplicating the call in OpenAI SSE.
                text_delta, _, _ = _extract_delta_from_parts(parts, {}, set())
                no_id_slot = 0
                for part in parts:
                    if not isinstance(part, dict) or not isinstance(part.get("functionCall"), dict):
                        continue
                    slot = None
                    if not part["functionCall"].get("id"):
                        slot = no_id_slot
                        no_id_slot += 1
                    _merge_function_snapshot(part, pending_tool_parts, pending_tool_by_id, slot)
                tool_call_deltas = []
                if fr:
                    _, tool_call_deltas, seen_tool_calls = _extract_delta_from_parts(
                        pending_tool_parts, seen_tool_calls, allowed_tool_names
                    )
                    pending_tool_parts.clear()
                    pending_tool_by_id.clear()

                # Hold only a possible call: prefix until complete. Normal prose
                # continues streaming as soon as it cannot match that prefix.
                if text_delta and checking_text_call:
                    pending_text += text_delta
                    stripped = pending_text.lstrip()
                    if len(pending_text) > 32768 or not (
                        stripped.startswith("call:") or "call:".startswith(stripped)
                    ):
                        text_delta = pending_text
                        pending_text = ""
                        checking_text_call = False
                    else:
                        text_delta = ""

                # Determine finish reason.
                if fr:
                    upstream_finished = True
                    if str(fr).upper() == "MAX_TOKENS":
                        finish_reason = "length"
                    elif tool_call_deltas or finish_reason == "tool_calls":
                        finish_reason = "tool_calls"
                    else:
                        fr_up = str(fr).upper()
                        mapping = {
                            "STOP": "stop",
                            "MAX_TOKENS": "length",
                            "SAFETY": "content_filter",
                            "RECITATION": "content_filter",
                            "BLOCKLIST": "content_filter",
                            "PROHIBITED_CONTENT": "content_filter",
                            "SPII": "content_filter",
                            "IMAGE_SAFETY": "content_filter",
                            "LANGUAGE": "content_filter",
                            "OTHER": "stop",
                        }
                        finish_reason = mapping.get(fr_up, "stop")

                # Build delta.
                delta = {}
                if text_delta:
                    delta["content"] = text_delta
                if tool_call_deltas:
                    delta["tool_calls"] = tool_call_deltas
                    if finish_reason != "length":
                        finish_reason = "tool_calls"

                if not delta and not fr:
                    # Empty chunk with nothing — skip.
                    continue

                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": openai_model,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }],
                }
                if not send_sse(chunk):
                    return
                if (delta.get("content") or "").strip() or delta.get("tool_calls"):
                    emitted_visible = True

            if blocked_reason:
                finish_reason = "content_filter"
                notice = f"[Content blocked: {blocked_reason}]"
                send_sse({"id": completion_id, "object": "chat.completion.chunk", "created": created,
                          "model": openai_model, "choices": [{"index": 0, "delta": {"content": notice}, "finish_reason": None}]})
            elif not upstream_finished:
                finish_reason = "length"
                send_sse({"id": completion_id, "object": "chat.completion.chunk", "created": created,
                          "model": openai_model, "choices": [{"index": 0, "delta": {"content": "[Upstream stream ended without finishReason]"}, "finish_reason": None}]})
            if pending_text:
                textual_call = _parse_text_tool_call(pending_text, allowed_tool_names) if upstream_finished and not blocked_reason else None
                if textual_call and finish_reason == "tool_calls":
                    # Prefer the structured call when the model emitted both.
                    delta = None
                elif textual_call:
                    delta = {"tool_calls": [{"index": len(seen_tool_calls), **textual_call}]}
                    if finish_reason != "length":
                        finish_reason = "tool_calls"
                else:
                    delta = {"content": pending_text}
                pending_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": openai_model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                if delta is not None:
                    if not send_sse(pending_chunk):
                        return
                    textual_calls_emitted += len(delta.get("tool_calls") or [])
                    if (delta.get("content") or "").strip() or delta.get("tool_calls"):
                        emitted_visible = True

            if upstream_finished and finish_reason == "stop" and not emitted_visible and not blocked_reason:
                finish_reason = "length"
                send_sse({"id": completion_id, "object": "chat.completion.chunk", "created": created,
                          "model": openai_model, "choices": [{"index": 0,
                          "delta": {"content": "[Upstream returned empty response]"}, "finish_reason": None}]})

            # Final chunk with finish_reason.
            trace.emit(completion_id, finish_reason, len(seen_tool_calls) + textual_calls_emitted)
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": openai_model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }],
            }
            # Include usage in the final chunk if the client requested
            # stream_options.include_usage (OpenAI convention).
            if total_usage:
                pt = total_usage.get("promptTokenCount", 0) or 0
                ct = total_usage.get("candidatesTokenCount", 0) or total_usage.get("completionTokenCount", 0) or 0
                tt = total_usage.get("totalTokenCount", (pt + ct)) or (pt + ct)
                final_chunk["usage"] = {
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": tt,
                }
            send_sse(final_chunk)

            # Termination sentinel.
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass

        except Exception as e:
            # HTTP status cannot change once SSE headers have been sent. Do not
            # present an upstream failure as a normal stop or leak its body.
            _log(f"Streaming interrupted: {type(e).__name__}")
            error_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": openai_model,
                "choices": [{"index": 0, "delta": {"content": "[Upstream stream interrupted]"},
                             "finish_reason": "length"}],
            }
            if send_sse(error_chunk):
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass


# ──────────────────────────────────────────────────────────────────────────────
# Server entry point
# ──────────────────────────────────────────────────────────────────────────────

def _preflight_check():
    """Validate the token file exists and is readable before starting."""
    if not os.path.isfile(TOKEN_FILE):
        _log(f"ERROR: Token file not found at {TOKEN_FILE}")
        _log("Run the Antigravity CLI (`agy`) to authenticate first.")
        return False
    data = _read_token_from_disk()
    if not data:
        _log("ERROR: Token file exists but could not be parsed.")
        return False
    token_obj = data.get("token") or {}
    if not token_obj.get("refresh_token"):
        _log("ERROR: No refresh_token in token file. Re-run `agy` to authenticate.")
        return False
    _log(f"Token file OK (auth_method={data.get('auth_method', 'unknown')})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Antigravity OpenAI-compatible proxy (Cloud Code Assist API)"
    )
    parser.add_argument("--port", type=int, default=8877, help="Port to listen on (default 8877)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default 127.0.0.1)")
    args = parser.parse_args()

    if not _preflight_check():
        sys.exit(1)

    # Pre-discover the project ID so the first request is fast.
    try:
        get_project_id()
    except Exception as e:
        _log(f"WARNING: Could not pre-discover project ID: {e}")
        _log("It will be discovered on the first request.")

    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    server.daemon_threads = True
    _log(f"Antigravity proxy listening on http://{args.host}:{args.port}/v1")
    _log("  Models: GET /v1/models")
    _log(f"  Token:  {TOKEN_FILE}")
    _log(f"  Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()