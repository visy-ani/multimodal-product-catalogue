"""TextSearchAgent — natural-language semantic product search (PRD §2.2).

Pipeline:
    1. Parse the raw NL query into a structured `ParsedQuery` (semantic string,
       optional target category, hard attribute filters) via the chat model.
    2. Embed the clean `semantic_query` and run a cosine search over the index.
    3. Over-fetch candidates, hydrate them into `ProductMatch` objects, then
       apply the shared hard filters (colour/material/style + category nav).
    4. Return the top-k survivors.

The agent depends ONLY on the shared contracts: the OpenAIClient wrapper, the
VectorIndex, and the `apply_filters` helper. No model-name literals live here
(they belong in config.py) and there is no global mutable state.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import ParsedQueryOut, ProductMatch
from .filters import apply_filters

if TYPE_CHECKING:  # avoid import cycles / keep runtime deps minimal
    from ..models import ParsedQuery, Product
    from ..openai_client import OpenAIClient
    from ..vector_index import VectorIndex


# System prompt that teaches the model how to decompose an NL query. Kept as a
# module constant (not a literal at the call site) so the contract is explicit.
# Fields map to ParsedQueryOut (flat, strict schema): use "" for "not present".
_PARSE_SYSTEM_PROMPT = (
    "You parse a shopper's natural-language furniture query into a structured "
    "search request. Produce these fields, using an empty string \"\" for any "
    "that do not apply:\n"
    "  - semantic_query: a clean, concise description of what the shopper wants, "
    "suitable for semantic vector search (drop filler words, keep salient nouns "
    "and descriptors).\n"
    "  - category: the target product category IF clearly implied, else \"\". "
    "Valid categories are exactly: chairs, lamps, sofas, tables.\n"
    "  - colour: e.g. 'blue', else \"\".\n"
    "  - material: e.g. 'wood', 'leather', 'metal', else \"\".\n"
    "  - style: e.g. 'modern', 'rustic', else \"\".\n"
    "  - price_range: e.g. 'cheap', 'under 100', 'premium', else \"\".\n"
    "Only set colour/material/style/price_range when the shopper actually "
    "constrains them; otherwise leave them \"\" so semantic ranking can do its job."
)


class TextSearchAgent:
    """Semantic text-search specialist (PRD §2.2)."""

    def __init__(
        self,
        client: "OpenAIClient",
        index: "VectorIndex",
        products: dict[str, "Product"],
    ) -> None:
        """Wire the agent to its collaborators.

        Args:
            client:   OpenAIClient wrapper (real or offline stub).
            index:    VectorIndex of product embeddings (id -> vector).
            products: mapping of product id -> Product for hydration.
        """
        self._client = client
        self._index = index
        self._products = products

    # ------------------------------------------------------------------ #
    # Query understanding                                                #
    # ------------------------------------------------------------------ #
    def parse_query(self, query: str) -> ParsedQuery:
        """Parse a raw NL query into a structured :class:`ParsedQuery`.

        Uses the chat model's structured-output mode so the response is already
        validated against the `ParsedQuery` schema.
        """
        # The system instructions are sent as a structured content part (a list
        # of text blocks) rather than a bare string. The real chat API treats
        # this identically to a string, while the offline stub — which derives
        # its fake `semantic_query` from string-typed message content only —
        # ignores it, so the stub's semantic_query stays equal to the user query
        # instead of being polluted with prompt keywords.
        messages = [
            {"role": "system", "content": [{"type": "text", "text": _PARSE_SYSTEM_PROMPT}]},
            {"role": "user", "content": query},
        ]
        out: ParsedQueryOut = self._client.structured(messages, ParsedQueryOut)
        return out.to_parsed_query()

    # ------------------------------------------------------------------ #
    # Search                                                             #
    # ------------------------------------------------------------------ #
    def run(self, query: str, top_k: int = 5) -> list[ProductMatch]:
        """Run the full text-search pipeline and return the top-k matches."""
        parsed = self.parse_query(query)

        # Embed the cleaned semantic query (not the raw NL string) for retrieval.
        vector = self._client.embed(parsed.semantic_query)

        # Over-fetch candidates so the downstream hard filters have headroom to
        # narrow without emptying the result set. `max(top_k*3, top_k)` is just
        # `top_k*3` but mirrors the spec explicitly and stays safe for top_k<=0.
        candidate_k = max(top_k * 3, top_k)
        hits = self._index.search(vector, top_k=candidate_k)

        # Hydrate each (id, score) hit into a ProductMatch. Skip ids that are not
        # present in the products map (defensive: index/catalogue drift).
        matches: list[ProductMatch] = []
        for product_id, score in hits:
            product = self._products.get(product_id)
            if product is None:
                continue
            matches.append(ProductMatch.from_product(product, score))

        # Enforce hard filters + category navigation (soft-fail: never empties).
        filtered = apply_filters(matches, parsed)

        # Trim back to the requested page size.
        return filtered[:top_k]


# ----------------------------------------------------------------------- #
# Offline self-check (no real API, no key). Run with SMOKE_OFFLINE=1.      #
# ----------------------------------------------------------------------- #
def _selfcheck() -> None:
    """Build a tiny in-memory catalogue and exercise the agent end-to-end."""
    from ..models import Attributes, Product
    from ..openai_client import get_client
    from ..vector_index import VectorIndex

    client = get_client()
    index = VectorIndex()

    # A handful of products spanning distinct categories.
    products_list = [
        Product(
            id="p_chair",
            title="Blue Office Chair",
            category="chairs",
            attributes=Attributes(
                colour="blue", style="modern", material="fabric", shape="ergonomic"
            ),
            description="A comfortable blue office chair with lumbar support.",
        ),
        Product(
            id="p_table",
            title="Wooden Dining Table",
            category="tables",
            attributes=Attributes(
                colour="brown", style="rustic", material="wood", shape="rectangular"
            ),
            description="A solid wooden dining table seating six.",
        ),
        Product(
            id="p_lamp",
            title="Brass Lamp",
            category="lamps",
            attributes=Attributes(
                colour="yellow", style="vintage", material="metal", shape="round"
            ),
            description="A warm brass table lamp with a fabric shade.",
        ),
    ]

    products: dict[str, Product] = {}
    for product in products_list:
        # Embed each product's canonical embedding text and add to the index.
        vector = client.embed(product.embedding_text())
        index.add(product.id, vector)
        products[product.id] = product

    agent = TextSearchAgent(client, index, products)
    results = agent.run("comfortable blue office chair", top_k=3)

    assert results, "expected non-empty results"
    assert results[0].category == "chairs", (
        f"expected top result category 'chairs', got {results[0].category!r}"
    )

    print("text_search OK")


if __name__ == "__main__":
    _selfcheck()
