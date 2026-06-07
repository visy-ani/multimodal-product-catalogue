"""AttributeExtractionAgent — pull salient visual attributes from an image (PRD §2.5).

Given a product image, this agent asks the vision-capable model (via the client
wrapper) to return a STRICT, schema-validated `Attributes` object: the four core
fields (colour, style, material, shape) plus any salient `extras`.

The agent owns no global state and references no model strings — model selection
lives in config.py and is handled entirely inside the client wrapper.
"""
from __future__ import annotations

from ..models import Attributes, AttributesOut

# Instruction prompt. The image itself is attached by the client wrapper via the
# `image_b64` argument, so this is purely the text guidance for the model.
_PROMPT = (
    "You are a product cataloguing assistant. Look at the attached product image "
    "and extract its salient visual attributes. Identify:\n"
    "  - colour: the primary colour (e.g. 'blue')\n"
    "  - style: the design style (e.g. 'modern', 'rustic')\n"
    "  - material: the dominant material (e.g. 'leather', 'wood', 'metal')\n"
    "  - shape: the overall shape (e.g. 'rectangular', 'round')\n"
    "Always fill all four fields with your best estimate; do not leave them "
    "blank. Return strict JSON matching the requested schema."
)


class AttributeExtractionAgent:
    """Extract structured `Attributes` from a single product image."""

    def __init__(self, client) -> None:
        # `client` is an OpenAIClient / StubOpenAIClient (see openai_client.py).
        self.client = client

    def run(self, image_b64: str) -> Attributes:
        """Return schema-validated `Attributes` for the given base64 image."""
        messages = [{"role": "user", "content": _PROMPT}]
        # Use a flat, strict-schema-friendly DTO, then map to the domain model.
        out: AttributesOut = self.client.structured(messages, AttributesOut, image_b64=image_b64)
        return out.to_attributes()


# --------------------------------------------------------------------------- #
# Self-check                                                                   #
# --------------------------------------------------------------------------- #
def _selfcheck() -> None:
    """Run the agent against a fixture image and assert the core fields are filled."""
    from ..config import CATALOGUE_DIR
    from ..openai_client import encode_image, get_client

    client = get_client()
    agent = AttributeExtractionAgent(client)

    image_b64 = encode_image(CATALOGUE_DIR / "chair_blue.jpg")
    attrs = agent.run(image_b64)

    assert isinstance(attrs, Attributes), f"expected Attributes, got {type(attrs)}"
    assert attrs.is_complete(), f"core attributes incomplete: {attrs!r}"
    print("attribute_extraction OK")


if __name__ == "__main__":
    _selfcheck()
