"""ImageSearchAgent — "describe-then-embed" image search (PRD §2.3).

Catalogue products are embedded from TEXT (`Product.embedding_text()`), so an
image query cannot be embedded directly against that vector space. Instead this
agent first turns the query image into a rich TEXT caption (colour, material,
style, shape, category) via the vision model, then embeds that caption with the
same `text-embedding-3-small` space and searches the shared `VectorIndex`.

The agent owns no global state and references no model strings — model selection
lives in config.py and is handled entirely inside the client wrapper.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import ProductMatch

if TYPE_CHECKING:  # avoid runtime import cycles; these are type hints only
    from ..models import Product
    from ..openai_client import OpenAIClient, StubOpenAIClient
    from ..vector_index import VectorIndex

# Instruction prompt for the captioner. The image itself is attached by the
# client wrapper via `vision(prompt, image_b64)`, so this is purely the text
# guidance. We ask for a dense, attribute-rich description because that text is
# what gets embedded and matched against the catalogue's text embeddings.
_CAPTION_PROMPT = (
    "You are a product cataloguing assistant. Describe the product in the "
    "attached image in a single rich paragraph optimised for semantic search. "
    "Explicitly mention every visible attribute you can identify:\n"
    "  - category: what kind of product it is (e.g. 'chair', 'lamp', 'sofa', 'table')\n"
    "  - colour: the primary colour(s)\n"
    "  - material: the dominant material(s)\n"
    "  - style: the design style (e.g. 'modern', 'rustic')\n"
    "  - shape: the overall shape\n"
    "Be concrete and specific; do not speculate about price or brand. Return "
    "plain text only."
)


class ImageSearchAgent:
    """Search the catalogue from a query image via describe-then-embed (PRD §2.3)."""

    def __init__(
        self,
        client: "OpenAIClient | StubOpenAIClient",
        index: "VectorIndex",
        products: dict[str, "Product"],
    ) -> None:
        # `client` is an OpenAIClient / StubOpenAIClient (see openai_client.py).
        self.client = client
        # Shared cosine `VectorIndex` keyed by product id (see vector_index.py).
        self.index = index
        # id -> Product, used to hydrate `ProductMatch` from a search hit.
        self.products = products

    def caption(self, image_b64: str) -> str:
        """Turn the query image into a rich, attribute-bearing text caption."""
        # The wrapper attaches the image to the prompt and returns plain text.
        return self.client.vision(_CAPTION_PROMPT, image_b64)

    def run(self, image_b64: str, top_k: int = 5) -> list[ProductMatch]:
        """Caption the image, embed the caption, and return the top_k matches."""
        # 1. Image -> text. 2. Text -> vector in the catalogue's text space.
        caption = self.caption(image_b64)
        vec = self.client.embed(caption)

        # 3. Cosine search against the shared index -> [(id, score)].
        hits = self.index.search(vec, top_k)

        # 4. Hydrate each hit into a stable `ProductMatch` contract.
        return [
            ProductMatch.from_product(self.products[pid], score)
            for pid, score in hits
        ]


# --------------------------------------------------------------------------- #
# Self-check                                                                   #
# --------------------------------------------------------------------------- #
def _selfcheck() -> None:
    """Build a tiny real-ish catalogue offline, run image search, assert hits."""
    from ..config import EMBED_DIM, QUERIES_DIR
    from ..models import Attributes, Product
    from ..openai_client import encode_image, get_client
    from ..vector_index import VectorIndex

    client = get_client()

    # A small but varied catalogue covering chairs / lamps / sofas / tables, so
    # the index has genuine category spread to rank against.
    products_list = [
        Product(
            id="chair_green",
            title="Green Accent Chair",
            category="chair",
            attributes=Attributes(colour="green", style="modern", material="fabric", shape="rounded"),
            description="A comfortable green accent chair with a modern silhouette.",
        ),
        Product(
            id="chair_blue",
            title="Blue Dining Chair",
            category="chair",
            attributes=Attributes(colour="blue", style="modern", material="wood", shape="straight"),
            description="A sturdy blue dining chair made of wood.",
        ),
        Product(
            id="lamp_yellow",
            title="Yellow Desk Lamp",
            category="lamp",
            attributes=Attributes(colour="yellow", style="industrial", material="metal", shape="angular"),
            description="A bright yellow desk lamp with an adjustable metal arm.",
        ),
        Product(
            id="sofa_blue",
            title="Blue Three-Seat Sofa",
            category="sofa",
            attributes=Attributes(colour="blue", style="contemporary", material="fabric", shape="rectangular"),
            description="A spacious blue three-seat sofa upholstered in soft fabric.",
        ),
        Product(
            id="table_black",
            title="Black Coffee Table",
            category="table",
            attributes=Attributes(colour="black", style="minimalist", material="glass", shape="rectangular"),
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

    # Encode the held-out query fixture and run describe-then-embed search.
    image_b64 = encode_image(QUERIES_DIR / "query_chair_green.jpg")
    agent = ImageSearchAgent(client, index, products)
    matches = agent.run(image_b64, top_k=3)

    # Offline the stub caption derives a category from image bytes, so a category
    # match is not guaranteed — only assert we got well-formed, non-empty results.
    assert matches, "expected non-empty image-search results"
    assert len(matches) <= 3, f"expected <= 3 matches, got {len(matches)}"
    assert all(isinstance(m, ProductMatch) for m in matches), "non-ProductMatch in results"
    print("image_search OK")


if __name__ == "__main__":
    _selfcheck()
