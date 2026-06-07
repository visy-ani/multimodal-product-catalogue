"""Thin OpenAI SDK wrapper with retry/backoff and a deterministic offline stub.

Every agent depends ONLY on this interface:

    client.embed(text)            -> list[float]            (or list[list[float]])
    client.chat(messages)         -> str
    client.vision(prompt, b64)    -> str
    client.structured(messages, Model, image_b64=None) -> Model instance

Model strings come from config.py. `client.total_calls` counts real API calls.

Set SMOKE_OFFLINE=1 to get `StubOpenAIClient`, which honours the same interface
with deterministic fake data — for fast structural checks WITHOUT network/keys.
"""
from __future__ import annotations

import base64
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Type, TypeVar, Union

from pydantic import BaseModel

from . import config

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def encode_image(path: Union[str, Path]) -> str:
    """Read an image file and return base64 (no data: prefix)."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _data_url(image_b64: str) -> str:
    return f"data:image/jpeg;base64,{image_b64}"


def vision_messages(prompt: str, image_b64: str) -> list[dict]:
    """Build a chat `messages` list with one text part + one image part."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _data_url(image_b64)}},
            ],
        }
    ]


# --------------------------------------------------------------------------- #
# Real client                                                                  #
# --------------------------------------------------------------------------- #
class OpenAIClient:
    """Single OpenAI client with retry on 429/5xx (PRD §2 shared services)."""

    def __init__(self) -> None:
        config.require_api_key()  # fail loudly if missing
        from openai import OpenAI

        self._client = OpenAI()
        self.total_calls = 0
        self._last_request = 0.0  # monotonic timestamp of the last request start

    # -- proactive pacing (for tiny RPM quotas) -------------------------- #
    def _pace(self) -> None:
        if config.MIN_REQUEST_INTERVAL <= 0:
            return
        wait = self._last_request + config.MIN_REQUEST_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    # -- retry wrapper ---------------------------------------------------- #
    def _with_retry(self, fn, *args, **kwargs):
        import openai

        last_exc = None
        for attempt in range(config.MAX_RETRIES):
            try:
                self._pace()  # respect the rate limit before every attempt
                self.total_calls += 1
                return fn(*args, **kwargs)
            except (openai.RateLimitError, openai.APIConnectionError) as exc:
                last_exc = exc
            except openai.APIStatusError as exc:
                if exc.status_code and 500 <= exc.status_code < 600:
                    last_exc = exc
                else:
                    raise
            sleep = min(config.BACKOFF_CAP, config.BACKOFF_BASE * (2 ** attempt))
            time.sleep(sleep)
        raise RuntimeError(f"OpenAI call failed after {config.MAX_RETRIES} retries") from last_exc

    # -- embeddings ------------------------------------------------------- #
    def embed(self, text: Union[str, list[str]]) -> Union[list[float], list[list[float]]]:
        single = isinstance(text, str)
        inp = [text] if single else text
        resp = self._with_retry(
            self._client.embeddings.create, model=config.EMBED_MODEL, input=inp
        )
        vecs = [d.embedding for d in resp.data]
        return vecs[0] if single else vecs

    # -- chat (text) ------------------------------------------------------ #
    def chat(self, messages: list[dict], **kw) -> str:
        resp = self._with_retry(
            self._client.chat.completions.create,
            model=config.CHAT_MODEL,
            messages=messages,
            **kw,
        )
        return resp.choices[0].message.content or ""

    # -- vision (image + prompt) ----------------------------------------- #
    def vision(self, prompt: str, image_b64: str, **kw) -> str:
        return self.chat(vision_messages(prompt, image_b64), **kw)

    # -- structured output (JSON schema via parse) ----------------------- #
    def structured(
        self, messages: list[dict], schema: Type[T], image_b64: str | None = None
    ) -> T:
        msgs = [dict(m) for m in messages]
        if image_b64 is not None:
            # merge the image INTO the last user message (avoids an empty-text part)
            last = dict(msgs[-1])
            content = last.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}] if content else []
            else:
                content = list(content)
            content.append({"type": "image_url", "image_url": {"url": _data_url(image_b64)}})
            last["content"] = content
            msgs[-1] = last
        resp = self._with_retry(
            self._client.chat.completions.parse,
            model=config.CHAT_MODEL,
            messages=msgs,
            response_format=schema,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("Structured parse returned no content")
        return parsed


# --------------------------------------------------------------------------- #
# Offline stub — deterministic, no network, no key                            #
# --------------------------------------------------------------------------- #
_CATEGORY_KEYWORDS = ["chair", "lamp", "sofa", "table", "couch", "desk", "stool"]
_COLOURS = ["blue", "red", "green", "black", "white", "brown", "grey", "yellow"]
_MATERIALS = ["leather", "wood", "metal", "fabric", "plastic", "glass"]


def _hash_floats(text: str, n: int) -> list[float]:
    """Deterministic pseudo-vector from text (stable across runs)."""
    out: list[float] = []
    i = 0
    while len(out) < n:
        h = hashlib.sha256(f"{text}|{i}".encode()).digest()
        for b in h:
            out.append((b / 255.0) * 2 - 1)
            if len(out) >= n:
                break
        i += 1
    return out


class StubOpenAIClient:
    """Same interface as OpenAIClient, deterministic fake data (PRD §7 offline).

    Embeddings are biased by detected category/colour keywords so that offline
    semantic search is *roughly* meaningful — enough for structural smoke checks.
    """

    def __init__(self) -> None:
        self.total_calls = 0

    # bias the hash vector toward category & colour signal so cosine is meaningful
    def embed(self, text: Union[str, list[str]]):
        single = isinstance(text, str)
        items = [text] if single else text
        self.total_calls += 1  # one request, regardless of batch size (matches real client)
        out = []
        for t in items:
            v = _hash_floats(t, config.EMBED_DIM)
            low = t.lower()
            for idx, kw in enumerate(_CATEGORY_KEYWORDS):
                if kw in low:
                    v[idx] += 6.0  # strong category signal in fixed dims
            for idx, c in enumerate(_COLOURS):
                if c in low:
                    v[32 + idx] += 3.0
            out.append(v)
        return out[0] if single else out

    def chat(self, messages: list[dict], **kw) -> str:
        self.total_calls += 1
        return "A simple, well-made product suitable for everyday use."

    def vision(self, prompt: str, image_b64: str, **kw) -> str:
        self.total_calls += 1
        # deterministic caption derived from the image bytes
        digest = hashlib.sha256(image_b64.encode()).hexdigest()
        cat = _CATEGORY_KEYWORDS[int(digest[:2], 16) % 4]
        col = _COLOURS[int(digest[2:4], 16) % len(_COLOURS)]
        return f"A {col} {cat} with a simple modern design made of wood."

    def structured(self, messages, schema, image_b64=None):
        self.total_calls += 1
        seed = (str(messages) + (image_b64 or ""))[:200]
        digest = hashlib.sha256(seed.encode()).hexdigest()
        fields = set(schema.model_fields)
        col = _COLOURS[int(digest[0:2], 16) % len(_COLOURS)]
        mat = _MATERIALS[int(digest[2:4], 16) % len(_MATERIALS)]

        # EnrichmentOut: flat {colour, style, material, shape, description}
        if {"colour", "style", "material", "shape", "description"} <= fields:
            return schema(
                colour=col, style="modern", material=mat, shape="rectangular",
                description=f"A {col} {mat} item with a clean modern look and "
                            f"rectangular shape.",
            )

        # AttributesOut: flat {colour, style, material, shape}
        if {"colour", "style", "material", "shape"} <= fields:
            return schema(colour=col, style="modern", material=mat, shape="rectangular")

        # ParsedQueryOut: flat {semantic_query, category, colour, material, style, price_range}
        if "semantic_query" in fields:
            txt = " ".join(
                m["content"] for m in messages
                if isinstance(m.get("content"), str)
            ).strip()
            low = txt.lower()
            return schema(
                semantic_query=txt or "product",
                category="",
                colour=next((c for c in _COLOURS if c in low), ""),
                material=next((mt for mt in _MATERIALS if mt in low), ""),
                style="",
                price_range="",
            )

        # Fallback: build with defaults (may fail for required-field schemas)
        return schema()


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #
def get_client():
    """Return a real or stub client depending on SMOKE_OFFLINE."""
    if os.environ.get("SMOKE_OFFLINE") == "1":
        return StubOpenAIClient()
    return OpenAIClient()
