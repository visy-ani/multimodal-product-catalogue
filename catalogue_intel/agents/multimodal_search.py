"""MultimodalSearchAgent — fuse an image AND text into one search (PRD §2.4).

A shopper supplies both a query image (the visual *style* they like) and a text
query (the concrete *attributes* they want). Neither signal alone is enough: the
image pins the look, the text pins constraints like colour/material. This agent
reflects BOTH.

Pipeline:
    1. Caption the image into rich attribute-bearing text (reusing the
       ImageSearchAgent's `caption()`).
    2. Embed the text query and the caption *separately* in the shared
       text-embedding space.
    3. DEFAULT FUSION = weighted mean of the two embeddings:
           fused = w_text * vec_text + (1 - w_text) * vec_caption
       This keeps both signals live in a single query vector. (A concatenation
       "enrichment string" is the documented alternative and is kept as a
       fallback for degenerate inputs — e.g. empty text.)
    4. Cosine-search the shared VectorIndex, hydrate ProductMatch objects.
    5. Parse hard attribute filters from the TEXT (via TextSearchAgent.parse_query)
       and overlay them with the shared `apply_filters` (soft-fail).
    6. Return the top_k survivors.

The agent owns no global state and references no model strings — model selection
lives in config.py and is handled entirely inside the client wrapper.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import config
from ..models import ProductMatch
from .filters import apply_filters
from .image_search import ImageSearchAgent
from .text_search import TextSearchAgent

if TYPE_CHECKING:  # type-only imports keep runtime deps minimal / cycle-free
    from ..models import Product
    from ..openai_client import OpenAIClient, StubOpenAIClient
    from ..vector_index import VectorIndex


class MultimodalSearchAgent:
    """Search the catalogue from an image + text by fusing both signals (PRD §2.4)."""

    def __init__(
        self,
        client: "OpenAIClient | StubOpenAIClient",
        index: "VectorIndex",
        products: dict[str, "Product"],
    ) -> None:
        """Wire the agent to its collaborators and the sub-agents it reuses.

        Args:
            client:   OpenAIClient wrapper (real or offline stub).
            index:    shared cosine VectorIndex keyed by product id.
            products: mapping of product id -> Product for hydration.
        """
        self._client = client
        self._index = index
        self._products = products

        # Reuse the sibling specialists rather than re-implementing their logic:
        #   - ImageSearchAgent.caption() : image -> rich text caption.
        #   - TextSearchAgent.parse_query() : text -> structured hard filters.
        # They are wired to the SAME client/index/products, so there is no extra
        # state to keep in sync.
        self._image_agent = ImageSearchAgent(client, index, products)
        self._text_agent = TextSearchAgent(client, index, products)

    # ------------------------------------------------------------------ #
    # Fusion helpers                                                     #
    # ------------------------------------------------------------------ #
    def _fuse(
        self, vec_text: list[float], vec_caption: list[float], w_text: float
    ) -> np.ndarray:
        """Weighted mean of the text and caption embeddings (primary fusion path).

        `w_text` in [0, 1] trades off the described attributes (text) against the
        visual style (image caption). The shared VectorIndex re-normalises the
        query vector at search time, so we do not normalise here.
        """
        t = np.asarray(vec_text, dtype=np.float32)
        c = np.asarray(vec_caption, dtype=np.float32)
        return w_text * t + (1.0 - w_text) * c

    # ------------------------------------------------------------------ #
    # Search                                                             #
    # ------------------------------------------------------------------ #
    def run(
        self,
        image_b64: str,
        text: str,
        top_k: int = 5,
        w_text: float | None = None,
    ) -> list[ProductMatch]:
        """Run the full multimodal pipeline and return the top_k matches."""
        # Default the fusion weight from config so the 0.5/0.5 policy lives in one
        # place (PRD §2.4) and callers can still override per-request.
        if w_text is None:
            w_text = config.W_TEXT

        # 1. Image -> rich text caption (reuses the image specialist).
        caption = self._image_agent.caption(image_b64)

        # 2. Embed both signals separately in the shared text-embedding space.
        if text and text.strip():
            vec_text = self._client.embed(text)
            vec_caption = self._client.embed(caption)
            # 3a. PRIMARY: weighted mean of the two embeddings keeps both live.
            fused = self._fuse(vec_text, vec_caption, w_text)
        else:
            # 3b. FALLBACK: with no usable text, fuse via the documented
            # concatenation-enrichment string and embed that single string. This
            # also covers the degenerate w_text such that one signal vanishes.
            enriched = f"user wants: {text}. Image shows: {caption}"
            fused = np.asarray(self._client.embed(enriched), dtype=np.float32)

        # 4. Over-fetch candidates so the downstream hard filters have headroom to
        # narrow without emptying the result set (mirrors TextSearchAgent).
        candidate_k = max(top_k * 3, top_k)
        hits = self._index.search(fused.tolist(), top_k=candidate_k)

        # Hydrate each (id, score) hit into a ProductMatch; skip unknown ids
        # (defensive against index/catalogue drift).
        matches: list[ProductMatch] = []
        for product_id, score in hits:
            product = self._products.get(product_id)
            if product is None:
                continue
            matches.append(ProductMatch.from_product(product, score))

        # 5. Parse hard attribute filters from the TEXT only (the image is style,
        # not a constraint) and overlay them (soft-fail: never empties).
        parsed = self._text_agent.parse_query(text)
        filtered = apply_filters(matches, parsed)

        # 6. Trim back to the requested page size.
        return filtered[:top_k]


# --------------------------------------------------------------------------- #
# Offline self-check (no real API, no key). Run with SMOKE_OFFLINE=1.          #
# --------------------------------------------------------------------------- #
def _selfcheck() -> None:
    """Build a tiny real-ish catalogue offline, run multimodal search, assert hits."""
    from ..config import EMBED_DIM, QUERIES_DIR
    from ..models import Attributes, Product
    from ..openai_client import encode_image, get_client
    from ..vector_index import VectorIndex

    client = get_client()

    # A small but varied catalogue spanning chairs/lamps/sofas/tables, with at
    # least two chairs so the index has genuine spread to rank against.
    products_list = [
        Product(
            id="chair_red_leather",
            title="Red Leather Armchair",
            category="chairs",
            attributes=Attributes(
                colour="red", style="classic", material="leather", shape="rounded"
            ),
            description="A plush red leather armchair with rolled arms.",
        ),
        Product(
            id="chair_blue_office",
            title="Blue Office Chair",
            category="chairs",
            attributes=Attributes(
                colour="blue", style="modern", material="fabric", shape="ergonomic"
            ),
            description="A comfortable blue office chair with lumbar support.",
        ),
        Product(
            id="lamp_yellow",
            title="Yellow Desk Lamp",
            category="lamps",
            attributes=Attributes(
                colour="yellow", style="industrial", material="metal", shape="angular"
            ),
            description="A bright yellow desk lamp with an adjustable metal arm.",
        ),
        Product(
            id="sofa_blue",
            title="Blue Three-Seat Sofa",
            category="sofas",
            attributes=Attributes(
                colour="blue", style="contemporary", material="fabric", shape="rectangular"
            ),
            description="A spacious blue three-seat sofa upholstered in soft fabric.",
        ),
        Product(
            id="table_black",
            title="Black Coffee Table",
            category="tables",
            attributes=Attributes(
                colour="black", style="minimalist", material="glass", shape="rectangular"
            ),
            description="A sleek black coffee table with a glass top.",
        ),
    ]

    # Embed each product's canonical text and load the cosine index.
    index = VectorIndex(dim=EMBED_DIM)
    products: dict[str, Product] = {}
    for p in products_list:
        vector = client.embed(p.embedding_text())
        p.vector = vector
        index.add(p.id, vector)
        products[p.id] = p

    # Encode the held-out query image and fuse it with a text constraint.
    image_b64 = encode_image(QUERIES_DIR / "query_chair_green.jpg")
    agent = MultimodalSearchAgent(client, index, products)
    matches = agent.run(image_b64, "in red leather", top_k=3)

    # Offline the stub caption/embeddings are deterministic but not semantically
    # rich, so only assert well-formed, non-empty results.
    assert matches, "expected non-empty multimodal-search results"
    assert len(matches) <= 3, f"expected <= 3 matches, got {len(matches)}"
    assert all(isinstance(m, ProductMatch) for m in matches), "non-ProductMatch in results"
    assert all(m.id in products for m in matches), "result id not in catalogue"

    print("multimodal_search OK")


if __name__ == "__main__":
    _selfcheck()
