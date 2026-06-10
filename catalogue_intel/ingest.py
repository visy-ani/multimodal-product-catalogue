"""Ingest pipeline: enrich every fixture product with attributes + a description,
embed it, and persist the catalogue + vector index (PRD §5 step 3, §6).

describe-then-embed: each product image -> attributes (AttributeExtractionAgent)
+ description (DescriptionAgent) -> embed `title + category + attributes +
description` with text-embedding-3-small -> one vector per product.

Run:  python -m catalogue_intel.ingest
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config
from .models import EnrichmentOut, Product, ProductEnrichment
from .openai_client import encode_image, get_client
from .vector_index import load_index, make_index

_ENRICH_PROMPT = (
    "You are a product cataloguer. Look at this product image and return strict "
    "JSON with two fields. `attributes`: ALWAYS fill colour, style, material, and "
    "shape with your best single-word judgement (never leave them blank — infer a "
    "plausible value if unsure), plus any salient extras. `description`: 1 to 3 "
    "short sentences (never more than 3) that explicitly mention at least two of "
    "the attributes. Be concise and accurate to what is visible."
)


def enrich(client, image_b64: str, title: str = "") -> ProductEnrichment:
    """One structured vision call -> attributes + description (PRD §2.5/§2.6).

    Combining the two halves the per-image vision calls vs running the
    AttributeExtractionAgent and DescriptionAgent separately — needed under a
    tiny RPM quota.
    """
    if not image_b64 or not image_b64.strip():
        raise ValueError("enrich() requires a non-empty base64 image string")
    prompt = _ENRICH_PROMPT + (f"\nKnown title: {title}" if title else "")
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    out: EnrichmentOut = client.structured(messages, EnrichmentOut, image_b64=image_b64)
    return out.to_enrichment()


def _load_manifest() -> dict:
    return json.loads(config.MANIFEST_PATH.read_text())


def build_catalogue(client=None, verbose: bool = True):
    """Enrich + embed + persist the fixture catalogue. Returns (index, products).

    Vision calls: one per product (combined enrich). Embeddings: ONE batched
    call for the whole catalogue. Index backend (chroma/numpy) comes from config.
    """
    client = client or get_client()
    manifest = _load_manifest()

    index = make_index(reset=True)  # fresh index each ingest
    products: dict[str, Product] = {}

    # 1) one combined vision call per product
    for item in manifest["catalogue"]:
        image_path = config.CATALOGUE_DIR / item["image"]
        enriched = enrich(client, encode_image(image_path), title=item["title"])
        products[item["id"]] = Product(
            id=item["id"], title=item["title"], category=item["category"],
            attributes=enriched.attributes, description=enriched.description,
            image_path=str(image_path),
        )

    # 2) ONE batched embedding call for the whole catalogue
    ordered = list(products.values())
    vectors = client.embed([p.embedding_text() for p in ordered])
    for product, vector in zip(ordered, vectors):
        product.vector = vector
        if verbose:
            print(f"  [{product.id}] {product.title}  "
                  f"({product.attributes.colour}/{product.attributes.material}) "
                  f"— {len(vector)}-dim vector")
    # single batched insert into the index backend
    index.add_many([(p.id, p.vector) for p in ordered])

    _persist(index, products)
    if verbose:
        print(f"Ingested {len(products)} products into '{config.VECTOR_BACKEND}' index; "
              f"metadata -> {config.CATALOGUE_JSON.name}")
    return index, products


def _persist(index, products: dict[str, Product]) -> None:
    index.save(config.INDEX_PATH)  # numpy writes .npz; chroma is already persisted
    # store metadata WITHOUT the bulky vector (vectors live in the index backend)
    payload = [p.model_dump(exclude={"vector"}) for p in products.values()]
    config.CATALOGUE_JSON.write_text(json.dumps(payload, indent=2))


def load_persisted():
    """Load a previously-ingested catalogue (index + metadata) from disk."""
    if not config.CATALOGUE_JSON.exists():
        raise FileNotFoundError("No persisted catalogue. Run `make ingest` first.")
    index = load_index()
    raw = json.loads(config.CATALOGUE_JSON.read_text())
    products = {p["id"]: Product(**p) for p in raw}
    return index, products


if __name__ == "__main__":
    build_catalogue()
