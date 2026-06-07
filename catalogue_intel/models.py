"""Pydantic contracts shared across every agent (PRD §2).

These models are the *stable interface*. Agents accept and return only these.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Product attributes                                                          #
# --------------------------------------------------------------------------- #
class Attributes(BaseModel):
    """Salient visual attributes extracted from a product image (PRD §2.5)."""

    colour: str = Field(default="", description="Primary colour, e.g. 'blue'")
    style: str = Field(default="", description="Design style, e.g. 'modern'")
    material: str = Field(default="", description="Material, e.g. 'leather'")
    shape: str = Field(default="", description="Overall shape, e.g. 'rectangular'")
    extras: dict[str, str] = Field(
        default_factory=dict,
        description="Any additional salient attributes (key -> value).",
    )

    def as_text(self) -> str:
        """Flatten attributes to a single searchable string."""
        parts = [self.colour, self.style, self.material, self.shape]
        parts += [f"{k} {v}" for k, v in self.extras.items()]
        return " ".join(p for p in parts if p).strip()

    def is_complete(self) -> bool:
        """True when the four core attributes are all non-empty (smoke test #5)."""
        return all([self.colour, self.style, self.material, self.shape])


# --------------------------------------------------------------------------- #
# Catalogue product                                                           #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Structured-output DTOs                                                       #
#                                                                             #
# OpenAI's strict json_schema mode (used by client.chat.completions.parse)     #
# requires EVERY field to be required and forbids open-ended dict/optional      #
# fields. So the models we hand to `structured()` are FLAT, all-required, and   #
# use "" as the "not present" sentinel. We map them to the rich domain models   #
# (Attributes / ParsedQuery) afterwards.                                        #
# --------------------------------------------------------------------------- #
class AttributesOut(BaseModel):
    """Strict-schema attribute extraction output."""

    model_config = ConfigDict(extra="forbid")
    colour: str
    style: str
    material: str
    shape: str

    def to_attributes(self) -> "Attributes":
        return Attributes(colour=self.colour, style=self.style,
                          material=self.material, shape=self.shape)


class EnrichmentOut(BaseModel):
    """Strict-schema combined attribute + description output."""

    model_config = ConfigDict(extra="forbid")
    colour: str
    style: str
    material: str
    shape: str
    description: str

    def to_enrichment(self) -> "ProductEnrichment":
        return ProductEnrichment(
            attributes=Attributes(colour=self.colour, style=self.style,
                                  material=self.material, shape=self.shape),
            description=self.description,
        )


class ParsedQueryOut(BaseModel):
    """Strict-schema NL-query parse output ("" = field not present)."""

    model_config = ConfigDict(extra="forbid")
    semantic_query: str
    category: str
    colour: str
    material: str
    style: str
    price_range: str

    def to_parsed_query(self) -> "ParsedQuery":
        return ParsedQuery(
            semantic_query=self.semantic_query or "",
            category=self.category or None,
            filters=QueryFilters(
                colour=self.colour or None,
                material=self.material or None,
                style=self.style or None,
                price_range=self.price_range or None,
            ),
        )


class ProductEnrichment(BaseModel):
    """Combined attribute extraction + description in ONE structured vision call.

    Used at ingest (and for query images) to halve vision calls vs running the
    AttributeExtractionAgent and DescriptionAgent separately — important under a
    tiny RPM quota. The standalone agents still exist for modular use.
    """

    attributes: Attributes
    description: str = Field(description="1-3 sentence description mentioning key attributes.")


class Product(BaseModel):
    """A catalogue item (PRD §2 shared services)."""

    id: str
    title: str
    category: str
    attributes: Attributes = Field(default_factory=Attributes)
    description: str = ""
    image_path: str = ""
    vector: Optional[list[float]] = None  # text-embedding-3-small, EMBED_DIM

    def embedding_text(self) -> str:
        """The text that gets embedded: title + category + attributes + description."""
        return (
            f"{self.title}. Category: {self.category}. "
            f"Attributes: {self.attributes.as_text()}. "
            f"{self.description}"
        ).strip()


class ProductMatch(BaseModel):
    """A ranked search hit."""

    id: str
    title: str
    category: str
    score: float
    attributes: Attributes = Field(default_factory=Attributes)
    description: str = ""
    image_path: str = ""

    @classmethod
    def from_product(cls, product: Product, score: float) -> "ProductMatch":
        return cls(
            id=product.id,
            title=product.title,
            category=product.category,
            score=float(score),
            attributes=product.attributes,
            description=product.description,
            image_path=product.image_path,
        )


# --------------------------------------------------------------------------- #
# Query parsing (TextSearchAgent structured output, PRD §2.2)                  #
# --------------------------------------------------------------------------- #
class QueryFilters(BaseModel):
    """Hard filters parsed from a natural-language query."""

    colour: Optional[str] = None
    material: Optional[str] = None
    style: Optional[str] = None
    price_range: Optional[str] = None  # e.g. "cheap", "under 100", "premium"
    extras: dict[str, str] = Field(default_factory=dict)


class ParsedQuery(BaseModel):
    """Structured form of a user's NL query."""

    semantic_query: str = Field(description="The clean semantic search string.")
    category: Optional[str] = Field(default=None, description="Target category if implied.")
    filters: QueryFilters = Field(default_factory=QueryFilters)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
class Intent(str, Enum):
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    COMBINED = "combined"
    REFINEMENT = "refinement"


class SearchRequest(BaseModel):
    """Input to the OrchestratorAgent (PRD §2.1)."""

    text: Optional[str] = None
    image: Optional[str] = None  # path to an image file OR base64 data
    session_id: str = "default"


class SearchResponse(BaseModel):
    """What the orchestrator returns."""

    intent: Intent
    dispatched_agent: str  # class name of the specialist that handled it
    query_text: str = ""   # the effective text query used (for analytics/debug)
    matches: list[ProductMatch] = Field(default_factory=list)
    latency_ms: float = 0.0


# --------------------------------------------------------------------------- #
# Analytics (PRD §2.7)                                                         #
# --------------------------------------------------------------------------- #
class QueryLog(BaseModel):
    """A single logged query event."""

    query_id: str
    session_id: str
    modality: str           # "text" | "image" | "both"
    query_text: str = ""    # effective text (or image caption) for gap clustering
    result_count: int = 0
    latency_ms: float = 0.0
    clicked: bool = False    # set true once a click is logged for this query


class AnalyticsReport(BaseModel):
    """Structured analytics summary (PRD §2.7, smoke test #7)."""

    total_queries: int
    zero_result_count: int
    zero_result_rate: float
    abandonment_rate: float       # queries with no click within session
    click_through_rate: float
    catalogue_gaps: list[dict] = Field(default_factory=list)  # clustered failed queries
