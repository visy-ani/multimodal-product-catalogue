"""Generate a small, reproducible fixture catalogue (PRD §4).

Draws category-distinct schematic furniture (chair / lamp / sofa / table) in a
product's real colour, so gpt-4o-mini vision has something honest to read.
Writes images + a synchronized manifest.json. Run once:  python -m
catalogue_intel.fixtures.generate_fixtures
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
CATALOGUE = HERE / "catalogue"
QUERIES = HERE / "queries"
SIZE = 256

COLOURS = {
    "blue": (40, 90, 200), "red": (200, 50, 50), "green": (50, 150, 80),
    "black": (40, 40, 40), "white": (235, 235, 235), "brown": (120, 80, 50),
    "grey": (130, 130, 130), "brass": (181, 166, 66), "yellow": (220, 200, 60),
}


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (SIZE, SIZE), (250, 250, 248))
    return img, ImageDraw.Draw(img)


def _outline(c):
    return (max(c[0] - 40, 0), max(c[1] - 40, 0), max(c[2] - 40, 0))


def draw_chair(d, c):
    d.rectangle([90, 60, 120, 160], fill=c, outline=_outline(c), width=3)   # backrest
    d.rectangle([90, 150, 180, 175], fill=c, outline=_outline(c), width=3)  # seat
    for x in (95, 168):                                                     # legs
        d.rectangle([x, 175, x + 10, 220], fill=c, outline=_outline(c), width=2)


def draw_table(d, c):
    d.rectangle([50, 90, 206, 115], fill=c, outline=_outline(c), width=3)   # top
    for x in (60, 186):
        d.rectangle([x, 115, x + 12, 210], fill=c, outline=_outline(c), width=2)


def draw_lamp(d, c):
    d.polygon([(98, 60), (158, 60), (175, 110), (81, 110)], fill=c, outline=_outline(c))  # shade
    d.rectangle([124, 110, 132, 200], fill=_outline(c), width=2)            # pole
    d.ellipse([96, 198, 160, 218], fill=c, outline=_outline(c), width=2)    # base


def draw_sofa(d, c):
    d.rectangle([40, 110, 216, 170], fill=c, outline=_outline(c), width=3)  # body
    d.rectangle([40, 80, 216, 120], fill=c, outline=_outline(c), width=3)   # back
    d.rectangle([34, 110, 56, 195], fill=c, outline=_outline(c), width=2)   # arm L
    d.rectangle([200, 110, 222, 195], fill=c, outline=_outline(c), width=2) # arm R
    for x in (60, 200):
        d.rectangle([x, 170, x + 12, 205], fill=_outline(c), width=2)       # feet


DRAW = {"chairs": draw_chair, "tables": draw_table, "lamps": draw_lamp, "sofas": draw_sofa}


def render(category: str, colour: str, path: Path) -> None:
    img, d = _canvas()
    DRAW[category](d, COLOURS[colour])
    img.save(path, "JPEG", quality=85)


# (id, title, category, colour, filename)
# Kept deliberately small (2 per category) — the account has a 3 RPM quota, so
# ingest cost (one vision call per product) is the dominant constraint.
CATALOGUE_ITEMS = [
    ("c1", "Ergonomic Blue Office Chair", "chairs", "blue", "chair_blue.jpg"),
    ("c2", "Red Leather Armchair", "chairs", "red", "chair_red.jpg"),
    ("c3", "Brown Wooden Dining Chair", "chairs", "brown", "chair_brown.jpg"),
    ("c4", "Black Gaming Chair", "chairs", "black", "chair_black.jpg"),
    ("l1", "Black Adjustable Desk Lamp", "lamps", "black", "lamp_black.jpg"),
    ("l2", "Brass Table Lamp", "lamps", "brass", "lamp_brass.jpg"),
    ("s1", "Grey Fabric Three-Seater Sofa", "sofas", "grey", "sofa_grey.jpg"),
    ("s2", "Brown Leather Sofa", "sofas", "brown", "sofa_brown.jpg"),
    ("t1", "Wooden Coffee Table", "tables", "brown", "table_brown.jpg"),
    ("t2", "Black Metal Side Table", "tables", "black", "table_black.jpg"),
]

# Held-out query images — NOT in the catalogue (PRD §4). 3 of the 4 categories;
# the chair is reused for the combined text+image test.
QUERY_ITEMS = [
    ("q_chair", "chairs", "green", "query_chair_green.jpg"),
    ("q_lamp", "lamps", "yellow", "query_lamp_yellow.jpg"),
    ("q_table", "tables", "white", "query_table_white.jpg"),
]


def main() -> None:
    CATALOGUE.mkdir(parents=True, exist_ok=True)
    QUERIES.mkdir(parents=True, exist_ok=True)

    catalogue = []
    for id, title, category, colour, fname in CATALOGUE_ITEMS:
        render(category, colour, CATALOGUE / fname)
        catalogue.append(
            {"id": id, "title": title, "category": category,
             "colour": colour, "image": fname}
        )

    queries = []
    for id, category, colour, fname in QUERY_ITEMS:
        render(category, colour, QUERIES / fname)
        queries.append(
            {"id": id, "expected_category": category, "colour": colour, "image": fname}
        )

    manifest = {
        "categories": sorted({c["category"] for c in catalogue}),
        "catalogue": catalogue,
        "queries": queries,
        # query image designated for the combined text+image test (#4)
        "combined_test": {"query_image": "query_chair_green.jpg",
                          "text": "in red leather", "expected_category": "chairs"},
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(catalogue)} catalogue + {len(queries)} query images + manifest.json")


if __name__ == "__main__":
    main()
