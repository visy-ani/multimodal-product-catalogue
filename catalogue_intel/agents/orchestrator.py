"""OrchestratorAgent — classifies intent, routes to a specialist, logs analytics,
and keeps per-session conversational state for refinements (PRD §2.1).

Routing is rule-based on modality (deterministic, free, and exactly what smoke
test #8 asserts). Refinement is the one place gpt-4o-mini is used for intent:
when a text-only follow-up arrives mid-session, the orchestrator rewrites it into
a standalone query that folds in the previous turn ("now show me cheaper ones in
blue" -> "cheap blue office chair").
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from time import perf_counter
from typing import Optional

log = logging.getLogger(__name__)

from ..models import Intent, ProductMatch, SearchRequest, SearchResponse
from ..openai_client import encode_image
from .analytics import AnalyticsAgent
from .image_search import ImageSearchAgent
from .multimodal_search import MultimodalSearchAgent
from .text_search import TextSearchAgent

# Cheap, deterministic cues that a text-only turn is refining the previous one.
# Prefix cues only fire when the user's text *starts* with them; phrase cues fire
# when the phrase appears as a word boundary anywhere in the text. This avoids
# false positives like "more comfortable chair" being treated as a refinement
# (the bare substring "more " would otherwise match).
_REFINEMENT_PREFIX_CUES = (
    "now ", "instead", "but ", "also ", "make it", "show me", "what about",
)
_REFINEMENT_PHRASE_CUES = (
    "cheaper", "in blue", "in red", "in black", "in white", "in green",
    "smaller", "bigger", "darker", "lighter",
)
_REFINEMENT_PHRASE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _REFINEMENT_PHRASE_CUES) + r")\b"
)


def _to_b64(image: str) -> str:
    """Accept either a file path or raw base64; return base64."""
    try:
        p = Path(image)
        if p.exists():
            return encode_image(p)
    except (OSError, ValueError):
        # OSError: path too long for the filesystem (long base64 strings).
        # ValueError: embedded null bytes that Path rejects outright.
        pass
    return image


class OrchestratorAgent:
    def __init__(self, client, index, products: dict, analytics: AnalyticsAgent) -> None:
        self.client = client
        self.analytics = analytics
        self.text_agent = TextSearchAgent(client, index, products)
        self.image_agent = ImageSearchAgent(client, index, products)
        self.multimodal_agent = MultimodalSearchAgent(client, index, products)
        # session_id -> {"last_query_text": str, "last_qid": str}
        self._sessions: dict[str, dict] = {}

    # -- intent classification ------------------------------------------- #
    def classify_intent(self, req: SearchRequest) -> Intent:
        has_text = bool(req.text and req.text.strip())
        has_image = bool(req.image)
        if has_text and has_image:
            return Intent.COMBINED
        if has_image:
            return Intent.IMAGE_ONLY
        if has_text:
            state = self._sessions.get(req.session_id)
            if state and state.get("last_query_text") and self._looks_like_refinement(req.text):
                return Intent.REFINEMENT
            return Intent.TEXT_ONLY
        raise ValueError("SearchRequest has neither text nor image")

    # maps each intent to the specialist that will handle it
    _AGENT_FOR_INTENT = {
        Intent.COMBINED: "MultimodalSearchAgent",
        Intent.IMAGE_ONLY: "ImageSearchAgent",
        Intent.TEXT_ONLY: "TextSearchAgent",
        Intent.REFINEMENT: "TextSearchAgent",
    }

    def route(self, req: SearchRequest) -> tuple[Intent, str]:
        """Pure routing decision — no API calls (used by smoke test #8)."""
        intent = self.classify_intent(req)
        return intent, self._AGENT_FOR_INTENT[intent]

    def _looks_like_refinement(self, text: str) -> bool:
        low = text.lower().strip()
        if any(low.startswith(c) for c in _REFINEMENT_PREFIX_CUES):
            return True
        return bool(_REFINEMENT_PHRASE_RE.search(low))

    def _merge_refinement(self, session_id: str, new_text: str) -> str:
        """Use gpt-4o-mini to fold the previous query into the follow-up."""
        prev = self._sessions[session_id]["last_query_text"]
        messages = [
            {"role": "system", "content": "Rewrite the user's follow-up into a single "
             "standalone product search query that keeps the relevant constraints from "
             "their previous query. Reply with ONLY the rewritten query."},
            {"role": "user", "content": f"Previous query: {prev}\nFollow-up: {new_text}"},
        ]
        try:
            merged = self.client.chat(messages).strip()
            return merged or f"{prev} {new_text}"
        except Exception as exc:
            # Never let refinement break a search, but record why we fell back so
            # silent failures don't hide a broken chat path.
            log.warning("refinement rewrite failed (%s); using concatenation fallback", exc)
            return f"{prev} {new_text}"

    # -- main entry ------------------------------------------------------- #
    def run(self, req: SearchRequest, top_k: int = 5) -> SearchResponse:
        intent = self.classify_intent(req)
        t0 = perf_counter()

        if intent == Intent.COMBINED:
            b64 = _to_b64(req.image)
            matches = self.multimodal_agent.run(b64, req.text, top_k=top_k)
            dispatched, modality = "MultimodalSearchAgent", "both"
            query_text = f"{req.text} [+image]"
        elif intent == Intent.IMAGE_ONLY:
            b64 = _to_b64(req.image)
            matches = self.image_agent.run(b64, top_k=top_k)
            dispatched, modality = "ImageSearchAgent", "image"
            query_text = "[image query]"
        else:  # TEXT_ONLY or REFINEMENT
            text = req.text
            if intent == Intent.REFINEMENT:
                text = self._merge_refinement(req.session_id, req.text)
            matches = self.text_agent.run(text, top_k=top_k)
            dispatched, modality = "TextSearchAgent", "text"
            query_text = text

        latency_ms = (perf_counter() - t0) * 1000.0

        # log every request to analytics (PRD §2.1)
        qid = self.analytics.log_query(
            session_id=req.session_id, modality=modality,
            query_text=query_text, result_count=len(matches), latency_ms=latency_ms,
        )
        self._sessions[req.session_id] = {"last_query_text": query_text, "last_qid": qid}

        return SearchResponse(
            intent=intent, dispatched_agent=dispatched, query_text=query_text,
            matches=matches, latency_ms=latency_ms,
        )

    # -- click feedback --------------------------------------------------- #
    def register_click(self, session_id: str, query_id: Optional[str] = None) -> None:
        """Record a click for a query (defaults to the session's most recent)."""
        qid = query_id or self._sessions.get(session_id, {}).get("last_qid")
        if qid:
            self.analytics.log_click(qid)
