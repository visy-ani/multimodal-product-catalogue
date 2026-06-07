"""End-to-end smoke suite (PRD §7). One command, REAL OpenAI calls, real numbers.

    make smoke                    # hits the real API (the actual E2E test)
    SMOKE_OFFLINE=1 make smoke    # stubbed client, fast STRUCTURAL checks only

Designed for a TINY RPM quota: the real work is done ONCE, with combined
attribute+description vision calls and BATCHED embeddings, then the 8 required
assertions are evaluated against the captured outputs. Every printed number is
measured, never hardcoded (PRD §1 honesty). Total real API calls ≈ 16. Set
OPENAI_RPM=3 to pace requests and avoid 429s. Exit non-zero on any failure.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from rich.console import Console
from rich.table import Table

from .. import config
from ..agents.analytics import AnalyticsAgent
from ..agents.filters import rank_matches
from ..agents.image_search import ImageSearchAgent
from ..agents.orchestrator import OrchestratorAgent
from ..agents.text_search import TextSearchAgent
from ..ingest import build_catalogue, enrich
from ..models import SearchRequest
from ..openai_client import encode_image, get_client

OFFLINE = os.environ.get("SMOKE_OFFLINE") == "1"


@dataclass
class Result:
    name: str
    ok: bool
    detail: str
    latency_ms: float = 0.0
    note: str = ""


def _sentences(text: str) -> list[str]:
    norm = text.replace("!", ".").replace("?", ".")
    return [s for s in (p.strip() for p in norm.split(".")) if s]


def _reflect_score(matches) -> float:
    """AVERAGE red/leather reflection PER returned result (PRD test #4).

    Averaging (not summing) is what makes the combined-vs-image-only comparison
    meaningful: the combined run narrows to red-leather items so its per-result
    reflection is high, while the image-only run returns a mix and is diluted —
    independent of catalogue size.
    """
    if not matches:
        return 0.0
    total = 0
    for m in matches:
        text = f"{m.title} {m.description} {m.attributes.as_text()}".lower()
        total += int("red" in text) + int("leather" in text)
    return total / len(matches)


def main() -> int:
    console = Console()
    manifest = json.loads(config.MANIFEST_PATH.read_text())
    client = get_client()

    console.rule("[bold]Multimodal Product Catalogue — E2E Smoke Suite")
    console.print(
        f"[bold]Mode:[/] {'OFFLINE (stub, structural only)' if OFFLINE else 'REAL OpenAI API'}   "
        f"[bold]chat/vision:[/] {config.CHAT_MODEL}   "
        f"[bold]embeddings:[/] {config.EMBED_MODEL} ({config.EMBED_DIM}-dim)   "
        f"[bold]RPM cap:[/] {config.OPENAI_RPM or 'none'}"
    )

    wall_start = perf_counter()
    results: list[Result] = []
    tmp = tempfile.mkdtemp(prefix="smoke_")

    # ===================================================================== #
    # Real work, executed once (batched + combined to minimise API calls)   #
    # ===================================================================== #

    # --- Ingest (Test 1) -------------------------------------------------- #
    t0 = perf_counter()
    index, products = build_catalogue(client, verbose=False)
    ingest_ms = (perf_counter() - t0) * 1000

    text_agent = TextSearchAgent(client, index, products)
    image_agent = ImageSearchAgent(client, index, products)

    # held-out query images
    q_by_cat = {q["expected_category"]: q for q in manifest["queries"]}
    chair_q = manifest["combined_test"]["query_image"]
    chair_b64 = encode_image(config.QUERIES_DIR / chair_q)

    # --- Query-time vision (combined enrich on the chair; captions for rest) --
    t0 = perf_counter()
    chair_enr = enrich(client, chair_b64)               # 1 call: attrs + desc (tests 5,6 + chair caption)
    enrich_ms = (perf_counter() - t0) * 1000

    captions: dict[str, str] = {chair_q: chair_enr.description}
    t0 = perf_counter()
    for q in manifest["queries"]:
        if q["image"] != chair_q:
            captions[q["image"]] = image_agent.caption(encode_image(config.QUERIES_DIR / q["image"]))
    caption_ms = (perf_counter() - t0) * 1000

    # --- Query parsing (structured) -------------------------------------- #
    text_query = "comfortable blue office chair"
    combined_text = manifest["combined_test"]["text"]   # "in red leather"
    t0 = perf_counter()
    parsed_text = text_agent.parse_query(text_query)
    parse_text_ms = (perf_counter() - t0) * 1000
    t0 = perf_counter()
    parsed_comb = text_agent.parse_query(combined_text)
    parse_comb_ms = (perf_counter() - t0) * 1000

    # --- ONE batched embedding call for every query-time text ------------- #
    embed_labels = ["textq", "combq"] + [q["image"] for q in manifest["queries"]]
    embed_texts = [parsed_text.semantic_query, parsed_comb.semantic_query] + [
        captions[q["image"]] for q in manifest["queries"]
    ]
    t0 = perf_counter()
    embed_vecs = client.embed(embed_texts)
    embed_ms = (perf_counter() - t0) * 1000
    emap = dict(zip(embed_labels, embed_vecs))

    # ===================================================================== #
    # Assemble results locally (no further API calls)                       #
    # ===================================================================== #

    # Test 1: Ingest
    bad = [
        p.id for p in products.values()
        if not p.attributes.as_text() or not p.description
        or not p.vector or len(p.vector) != config.EMBED_DIM
    ]
    results.append(Result(
        "1. Ingest", not bad,
        f"{len(products)} products enriched; all vectors {config.EMBED_DIM}-dim, "
        f"attrs+desc non-empty" + (f"; OFFENDERS={bad}" if bad else ""),
        ingest_ms,
    ))

    # Test 2: Text search
    text_matches = rank_matches(index, products, emap["textq"], top_k=3, parsed=parsed_text)
    top3 = [(m.id, round(m.score, 3)) for m in text_matches]
    top_cat_text = text_matches[0].category if text_matches else "none"
    ok2 = bool(text_matches) if OFFLINE else (bool(text_matches) and top_cat_text == "chairs")
    results.append(Result(
        "2. Text search", ok2,
        f"q={text_query!r} → top-3={top3}; top category={top_cat_text}",
        parse_text_ms + embed_ms, "structural (offline)" if OFFLINE else "",
    ))

    # Test 3: Image search (each held-out query image)
    image_results = {
        q["image"]: rank_matches(index, products, emap[q["image"]], top_k=3)
        for q in manifest["queries"]
    }
    per_image, hits = [], 0
    for q in manifest["queries"]:
        ms = image_results[q["image"]]
        top1 = ms[0].category if ms else "none"
        hit = top1 == q["expected_category"]
        hits += int(hit)
        per_image.append(f"{q['image']}→{top1}({'✓' if hit else '✗'}/{q['expected_category']})")
    accuracy = hits / len(manifest["queries"])
    ok3 = all(image_results[q["image"]] for q in manifest["queries"]) if OFFLINE else accuracy >= 0.66
    results.append(Result(
        "3. Image search", ok3,
        f"top-1 category accuracy={accuracy:.0%} ({hits}/{len(manifest['queries'])}); "
        + " | ".join(per_image),
        caption_ms + embed_ms, "structural (offline)" if OFFLINE else "",
    ))

    # Test 4: Combined (image + text) vs image-only baseline on the same chair
    w = config.W_TEXT
    fused = (w * np.asarray(emap["combq"]) + (1 - w) * np.asarray(emap[chair_q])).tolist()
    comb_matches = rank_matches(index, products, fused, top_k=3, parsed=parsed_comb)
    img_only = image_results[chair_q]
    comb_score, img_score = _reflect_score(comb_matches), _reflect_score(img_only)
    delta = comb_score - img_score
    top_cat_comb = comb_matches[0].category if comb_matches else "none"
    ok4 = bool(comb_matches) if OFFLINE else (
        bool(comb_matches) and top_cat_comb == "chairs" and delta > 1e-9
    )
    results.append(Result(
        "4. Combined", ok4,
        f"chair image + {combined_text!r} → top category={top_cat_comb}; "
        f"avg red/leather reflection combined={comb_score:.2f} vs "
        f"image-only={img_score:.2f} (Δ={delta:+.2f})",
        parse_comb_ms, "structural (offline)" if OFFLINE else "",
    ))

    # Test 5: Attribute extraction (from the combined enrich on the chair image)
    attrs = chair_enr.attributes
    ok5 = attrs.is_complete()
    results.append(Result(
        "5. Attribute extraction", ok5,
        f"colour={attrs.colour!r} style={attrs.style!r} "
        f"material={attrs.material!r} shape={attrs.shape!r}",
        enrich_ms,
    ))

    # Test 6: Description (same enrich call)
    desc = chair_enr.description
    n_sent = len(_sentences(desc))
    tokens = [v.lower() for v in (attrs.colour, attrs.style, attrs.material, attrs.shape) if v]
    mentions = any(tok in desc.lower() for tok in tokens)
    ok6 = (bool(desc.strip()) and 1 <= n_sent <= 3) if OFFLINE else (
        bool(desc.strip()) and 1 <= n_sent <= 3 and mentions
    )
    results.append(Result(
        "6. Description", ok6,
        f"{n_sent} sentence(s), mentions-attr={mentions}: {desc!r}",
        0.0, "structural (offline)" if OFFLINE else "",
    ))

    # Test 7: Analytics (isolated scripted session)
    an = AnalyticsAgent(db_path=os.path.join(tmp, "test7.db"), client=client)
    an.reset()
    t0 = perf_counter()
    qid_hit = an.log_query("sess", "text", "blue office chair", result_count=5, latency_ms=10.0)
    an.log_click(qid_hit)
    an.log_query("sess", "text", "holographic floating massage chair", result_count=0, latency_ms=8.0)
    rep = an.report()
    analytics_ms = (perf_counter() - t0) * 1000
    ok7 = (
        rep.total_queries == 2 and rep.zero_result_count == 1
        and abs(rep.click_through_rate - 0.5) < 1e-9
        and abs(rep.abandonment_rate - 0.5) < 1e-9
        and len(rep.catalogue_gaps) >= 1
    )
    gap0 = rep.catalogue_gaps[0]["theme"] if rep.catalogue_gaps else "—"
    results.append(Result(
        "7. Analytics", ok7,
        f"total={rep.total_queries} zero={rep.zero_result_count} "
        f"CTR={rep.click_through_rate:.0%} abandon={rep.abandonment_rate:.0%} "
        f"gaps={len(rep.catalogue_gaps)} ({gap0!r})",
        analytics_ms,
    ))

    # Test 8: Routing (pure decision, no API calls)
    orch = OrchestratorAgent(client, index, products, an)
    _, a_text = orch.route(SearchRequest(text="a chair", session_id="r1"))
    _, a_img = orch.route(SearchRequest(image=str(config.QUERIES_DIR / chair_q), session_id="r2"))
    _, a_both = orch.route(SearchRequest(text="in red", image=chair_b64, session_id="r3"))
    ok8 = (a_text == "TextSearchAgent" and a_img == "ImageSearchAgent"
           and a_both == "MultimodalSearchAgent")
    results.append(Result(
        "8. Routing", ok8,
        f"text→{a_text}, image→{a_img}, both→{a_both}", 0.0,
    ))

    # ===================================================================== #
    wall_ms = (perf_counter() - wall_start) * 1000
    table = Table(title="Smoke Results", show_lines=True)
    table.add_column("Test", style="bold")
    table.add_column("Status")
    table.add_column("Latency", justify="right")
    table.add_column("Measured value")
    for r in results:
        status = "[green]PASS[/]" if r.ok else "[red]FAIL[/]"
        if r.note:
            status += f"\n[dim]{r.note}[/]"
        table.add_row(r.name, status, f"{r.latency_ms:.0f} ms", r.detail)
    console.print(table)

    passed = sum(r.ok for r in results)
    console.print(
        f"\n[bold]{passed}/{len(results)} passed[/]   "
        f"[bold]OpenAI calls:[/] {client.total_calls}   "
        f"[bold]wall:[/] {wall_ms/1000:.1f}s   "
        f"[bold]models:[/] {config.CHAT_MODEL} + {config.EMBED_MODEL}"
    )
    if OFFLINE:
        console.print("[yellow]OFFLINE mode: semantic assertions relaxed to structural checks.[/]")

    all_ok = all(r.ok for r in results)
    console.print("[bold green]ALL REQUIRED TESTS PASSED[/]" if all_ok
                  else "[bold red]SMOKE SUITE FAILED[/]")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
