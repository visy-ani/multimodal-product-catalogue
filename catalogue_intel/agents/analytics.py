"""AnalyticsAgent — query logging + catalogue-gap reporting (PRD §2.7).

Responsibilities:
    1. Persist every search event (one row per query) to a SQLite store at
       `config.ANALYTICS_DB`, including modality, the effective text, the result
       count, latency, and whether the query was later clicked.
    2. Record clicks against a previously-logged query.
    3. Produce a structured `AnalyticsReport` summarising query health:
       zero-result rate, click-through rate, abandonment rate, and a clustered
       view of "catalogue gaps" — the failed queries the catalogue could not
       satisfy, grouped into themes the merchandising team can act on.

The agent depends ONLY on the shared contracts: the `config` paths, the
`QueryLog` / `AnalyticsReport` pydantic models, and (optionally) the OpenAIClient
wrapper for semantic gap clustering. No model-name literals live here — embedding
goes through `client.embed`, whose model is fixed in config.py.
"""
from __future__ import annotations

import math
import re
import sqlite3
import uuid
from typing import TYPE_CHECKING, Optional

from .. import config
from ..models import AnalyticsReport, QueryLog

if TYPE_CHECKING:  # keep runtime deps minimal / avoid import cycles
    from ..openai_client import OpenAIClient


# Cosine-similarity threshold above which two failed queries are considered to
# belong to the same catalogue-gap cluster (only used when a client is supplied).
_GAP_SIMILARITY_THRESHOLD = 0.8


# --------------------------------------------------------------------------- #
# Small pure helpers                                                          #
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    """Lower-case, collapse whitespace — the key used for exact-text grouping."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors (0 if either is null)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# Agent                                                                       #
# --------------------------------------------------------------------------- #
class AnalyticsAgent:
    """Logs query events and reports catalogue gaps (PRD §2.7)."""

    def __init__(
        self,
        db_path=config.ANALYTICS_DB,
        client: Optional["OpenAIClient"] = None,
    ) -> None:
        # `db_path` may be a str or Path; sqlite3 accepts str, so normalise.
        self.db_path = str(db_path)
        # `client` is used ONLY by report() to embed failed queries for semantic
        # gap clustering. When None we fall back to exact normalized-text grouping.
        self.client = client
        self._ensure_schema()

    # -- connection / schema --------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create the query-log table if it does not already exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_logs (
                    query_id     TEXT PRIMARY KEY,
                    session_id   TEXT NOT NULL,
                    modality     TEXT NOT NULL,
                    query_text   TEXT NOT NULL DEFAULT '',
                    result_count INTEGER NOT NULL DEFAULT 0,
                    latency_ms   REAL NOT NULL DEFAULT 0.0,
                    clicked      INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.commit()

    # -- writes ----------------------------------------------------------- #
    def log_query(
        self,
        session_id: str,
        modality: str,
        query_text: str,
        result_count: int,
        latency_ms: float,
    ) -> str:
        """Insert one query event; return the generated query_id (uuid4 hex)."""
        query_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO query_logs
                    (query_id, session_id, modality, query_text,
                     result_count, latency_ms, clicked)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    query_id,
                    session_id,
                    modality,
                    query_text or "",
                    int(result_count),
                    float(latency_ms),
                ),
            )
            conn.commit()
        return query_id

    def log_click(self, query_id: str) -> None:
        """Mark the given query's row as clicked (idempotent)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE query_logs SET clicked = 1 WHERE query_id = ?",
                (query_id,),
            )
            conn.commit()

    def reset(self) -> None:
        """Clear all logged rows (useful for tests / smoke)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM query_logs")
            conn.commit()

    # -- reads ------------------------------------------------------------ #
    def _all_logs(self) -> list[QueryLog]:
        """Load every logged row as a typed QueryLog."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT query_id, session_id, modality, query_text,
                       result_count, latency_ms, clicked
                FROM query_logs
                """
            ).fetchall()
        return [
            QueryLog(
                query_id=r["query_id"],
                session_id=r["session_id"],
                modality=r["modality"],
                query_text=r["query_text"],
                result_count=r["result_count"],
                latency_ms=r["latency_ms"],
                clicked=bool(r["clicked"]),
            )
            for r in rows
        ]

    # -- reporting -------------------------------------------------------- #
    def report(self) -> AnalyticsReport:
        """Compute aggregate analytics over all logged queries (PRD §2.7)."""
        logs = self._all_logs()
        total = len(logs)

        if total == 0:
            # Nothing logged yet: return a well-formed, all-zero report.
            return AnalyticsReport(
                total_queries=0,
                zero_result_count=0,
                zero_result_rate=0.0,
                abandonment_rate=0.0,
                click_through_rate=0.0,
                catalogue_gaps=[],
            )

        zero_result_count = sum(1 for q in logs if q.result_count == 0)
        clicked_count = sum(1 for q in logs if q.clicked)

        # Abandonment definition (simple and defensible): a query is "abandoned"
        # when it received no click (clicked == 0). abandonment_rate is the
        # fraction of all queries that were abandoned. This is the per-query
        # complement of the click-through rate.
        abandoned_count = sum(1 for q in logs if not q.clicked)

        zero_result_rate = zero_result_count / total
        click_through_rate = clicked_count / total
        abandonment_rate = abandoned_count / total

        # Failed queries feed the catalogue-gap clustering: a query failed if it
        # returned zero results OR was abandoned (no click). Only those with
        # non-empty text can be themed.
        failed = [
            q
            for q in logs
            if (q.result_count == 0 or not q.clicked) and q.query_text.strip()
        ]
        catalogue_gaps = self._cluster_gaps(failed)

        return AnalyticsReport(
            total_queries=total,
            zero_result_count=zero_result_count,
            zero_result_rate=zero_result_rate,
            abandonment_rate=abandonment_rate,
            click_through_rate=click_through_rate,
            catalogue_gaps=catalogue_gaps,
        )

    # -- gap clustering --------------------------------------------------- #
    def _cluster_gaps(self, failed: list[QueryLog]) -> list[dict]:
        """Group failed queries into catalogue-gap themes.

        With a client: embed each failed query and greedily assign it to the
        first existing cluster whose representative is within
        `_GAP_SIMILARITY_THRESHOLD` cosine similarity; otherwise start a new
        cluster. Without a client: group by exact normalized text.

        Each gap entry: {"theme": <representative text>, "count": n,
        "examples": [<up to a few original query texts>]}.
        """
        if not failed:
            return []

        if self.client is not None:
            return self._cluster_gaps_semantic(failed)
        return self._cluster_gaps_exact(failed)

    def _cluster_gaps_exact(self, failed: list[QueryLog]) -> list[dict]:
        """Fallback clustering: bucket by normalized query text."""
        buckets: dict[str, list[str]] = {}
        for q in failed:
            buckets.setdefault(_normalize(q.query_text), []).append(q.query_text)
        gaps: list[dict] = []
        for texts in buckets.values():
            gaps.append(
                {
                    "theme": texts[0],          # representative (first seen)
                    "count": len(texts),
                    "examples": texts[:5],
                }
            )
        # Most impactful gaps first.
        gaps.sort(key=lambda g: g["count"], reverse=True)
        return gaps

    def _cluster_gaps_semantic(self, failed: list[QueryLog]) -> list[dict]:
        """Greedy cosine clustering over embedded failed-query texts."""
        texts = [q.query_text for q in failed]
        # Batch-embed all failed query texts in one call (client.embed accepts a
        # list and returns a list of vectors).
        vectors = self.client.embed(texts)  # type: ignore[union-attr]

        # Each cluster keeps the representative vector + text and its examples.
        clusters: list[dict] = []
        for text, vec in zip(texts, vectors):
            placed = False
            for cluster in clusters:
                if _cosine(vec, cluster["_vec"]) > _GAP_SIMILARITY_THRESHOLD:
                    cluster["_examples"].append(text)
                    placed = True
                    break
            if not placed:
                clusters.append(
                    {"theme": text, "_vec": vec, "_examples": [text]}
                )

        gaps: list[dict] = []
        for cluster in clusters:
            examples = cluster["_examples"]
            gaps.append(
                {
                    "theme": cluster["theme"],
                    "count": len(examples),
                    "examples": examples[:5],
                }
            )
        gaps.sort(key=lambda g: g["count"], reverse=True)
        return gaps


# --------------------------------------------------------------------------- #
# Self-check (offline-friendly)                                               #
# --------------------------------------------------------------------------- #
def _selfcheck() -> None:
    """Exercise the full log -> click -> report cycle against a TEMP db."""
    import tempfile
    from pathlib import Path

    from ..openai_client import get_client

    # Use a throwaway db file so we never touch the real ANALYTICS_DB.
    tmp_dir = tempfile.mkdtemp(prefix="analytics_selfcheck_")
    tmp_db = str(Path(tmp_dir) / "analytics_selfcheck.db")

    agent = AnalyticsAgent(db_path=tmp_db, client=get_client())
    agent.reset()  # ensure a clean slate even if the temp file was reused

    # 1) A normal query that returned results and got clicked.
    qid_ok = agent.log_query(
        session_id="s1",
        modality="text",
        query_text="blue leather sofa",
        result_count=5,
        latency_ms=42.0,
    )
    agent.log_click(qid_ok)

    # 2) A deliberate zero-result query (failed) — never clicked.
    agent.log_query(
        session_id="s1",
        modality="text",
        query_text="purple velvet hammock chair",
        result_count=0,
        latency_ms=37.5,
    )

    report = agent.report()

    assert report.total_queries == 2, report.total_queries
    assert report.zero_result_count == 1, report.zero_result_count
    assert report.click_through_rate == 0.5, report.click_through_rate
    assert report.abandonment_rate == 0.5, report.abandonment_rate
    assert len(report.catalogue_gaps) >= 1, report.catalogue_gaps

    print("analytics OK")


if __name__ == "__main__":
    _selfcheck()
