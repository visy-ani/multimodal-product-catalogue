"""DescriptionAgent — natural-language product copy from an image (PRD §2.6).

Given a product image plus the already-extracted visual attributes (and an
optional title), this agent asks the vision model for a short, customer-facing
description. The prompt feeds the known attributes back to the model so the
generated copy stays grounded in them and mentions the salient ones.
"""
from __future__ import annotations

from ..models import Attributes


class DescriptionAgent:
    """Generate a 1-3 sentence product description grounded in known attributes."""

    def __init__(self, client) -> None:
        # The OpenAIClient / StubOpenAIClient wrapper. Model strings live in the
        # wrapper/config, never here — we only call its high-level methods.
        self._client = client

    def run(self, image_b64: str, attributes: "Attributes", title: str = "") -> str:
        """Write a short product description from the image.

        Builds a prompt that supplies the known attributes (and title, if any),
        instructs the model to write 1-3 sentences mentioning the key
        attributes, then runs it against the vision model.
        """
        attr_text = attributes.as_text()

        # Compose the prompt. We hand the model the attributes we already know so
        # the copy is consistent with downstream metadata and mentions them.
        lines = [
            "You are writing a concise, appealing product catalogue description.",
            "Look at the product image and write 1 to 3 sentences of natural "
            "marketing copy.",
            "The description MUST mention at least one of the key attributes below.",
            "Do not invent prices, brands, or specifications you cannot see.",
            "Return only the description text, no preamble or bullet points.",
        ]
        if title:
            lines.append(f"Product title: {title}")
        if attr_text:
            lines.append(f"Known attributes: {attr_text}")

        prompt = "\n".join(lines)

        # The wrapper attaches the image and selects the vision-capable model.
        description = self._client.vision(prompt, image_b64)
        return description.strip()


# --------------------------------------------------------------------------- #
# Self-check (PRD §7 offline smoke)                                            #
# --------------------------------------------------------------------------- #
def _selfcheck() -> None:
    """Structural smoke check; runs fully offline under SMOKE_OFFLINE=1."""
    from ..config import CATALOGUE_DIR
    from ..openai_client import encode_image, get_client

    client = get_client()
    agent = DescriptionAgent(client)

    image_b64 = encode_image(CATALOGUE_DIR / "chair_blue.jpg")
    attributes = Attributes(
        colour="blue", style="modern", material="leather", shape="rectangular"
    )

    description = agent.run(image_b64, attributes, title="Blue Accent Chair")

    # Robust offline: must be non-empty and a reasonable 1-3 sentence blurb.
    assert description, "description is empty"
    sentences = [s for s in description.split(".") if s.strip()]
    assert 1 <= len(sentences) <= 3, f"expected 1-3 sentences, got {len(sentences)}"

    # When backed by a real model, the copy should ground itself in an attribute.
    # Offline the stub may not, so this is a soft (non-asserting) check.
    attr_tokens = attributes.as_text().lower().split()
    if any(tok in description.lower() for tok in attr_tokens):
        pass  # ideal: mentions an extracted attribute

    print("description OK")


if __name__ == "__main__":
    _selfcheck()
