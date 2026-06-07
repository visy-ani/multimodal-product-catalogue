"""Shared hard-filter + category-navigation helpers for the search agents.

Search agents rank semantically, then call `apply_filters` to enforce the hard
constraints parsed from the query (colour/material/style) and category nav.
Filtering is soft-fail: if nothing survives a hard filter we keep the semantic
ranking rather than returning an empty list (better UX than a dead end).
"""
from __future__ import annotations

from ..models import ParsedQuery, Product, ProductMatch


def _matches_value(haystack: str, needle: str | None) -> bool:
    if not needle:
        return True
    return needle.lower().strip() in haystack.lower()


def apply_filters(
    matches: list[ProductMatch], parsed: ParsedQuery
) -> list[ProductMatch]:
    """Apply category navigation + hard attribute filters to ranked matches.

    Order preserved (already sorted by score). Returns the filtered list, or the
    original ranking if filtering would empty it.
    """
    result = matches

    # category navigation
    if parsed.category:
        narrowed = [m for m in result if _matches_value(m.category, parsed.category)]
        if narrowed:
            result = narrowed

    # hard attribute filters
    f = parsed.filters
    for attr_value, get in (
        (f.colour, lambda m: m.attributes.colour + " " + m.title + " " + m.description),
        (f.material, lambda m: m.attributes.material + " " + m.description),
        (f.style, lambda m: m.attributes.style + " " + m.description),
    ):
        if attr_value:
            narrowed = [m for m in result if _matches_value(get(m), attr_value)]
            if narrowed:
                result = narrowed

    return result


def rank_matches(
    index,
    products: dict[str, Product],
    vector: list[float],
    top_k: int = 5,
    parsed: ParsedQuery | None = None,
) -> list[ProductMatch]:
    """Cosine-rank `vector` against the index, hydrate, optionally filter, trim.

    This is the single ranking implementation shared by the search agents' run()
    methods AND the smoke runner's batched path, so rankings never diverge.
    """
    hits = index.search(vector, top_k=max(top_k * 3, top_k))
    matches = [
        ProductMatch.from_product(products[pid], score)
        for pid, score in hits
        if pid in products
    ]
    if parsed is not None:
        matches = apply_filters(matches, parsed)
    return matches[:top_k]
