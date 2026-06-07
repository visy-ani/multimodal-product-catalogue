# Multimodal Product Catalogue Intelligence

A multimodal product search system built as a set of cooperating agents. A
customer can search by **text**, by **image**, or **both**; the system extracts
product attributes from images, auto-writes product descriptions, and tracks
search analytics to surface catalogue gaps.

Everything runs on cheap models — `gpt-4o-mini` (chat + vision) and
`text-embedding-3-small` (embeddings) — by design (college-project scope, cost
discipline). Model strings live in exactly one place: `catalogue_intel/config.py`.

## Architecture (multi-agent)

### System overview

```
   user request                ┌──────────────────────────────────────────────┐
   {text?, image?,   ────────▶ │              OrchestratorAgent                │
    session_id}                │  classify intent (text/image/combined/refine) │  gpt-4o-mini
                               │  route • per-session state • log every query  │  (intent + query rewrite)
                               └────┬──────────────┬──────────────┬────────────┘
                  text-only ────────┘   image-only ─┘     both ────┘
                       ▼                   ▼                  ▼
               TextSearchAgent      ImageSearchAgent    MultimodalSearchAgent          AnalyticsAgent
               parse→embed→rank     caption→embed→rank  caption+text→FUSE→rank          ┌───────────────┐
                       │                   │                  │                         │ SQLite store: │
                       └───────────────────┼──────────────────┘   ── every query ────▶ │ CTR, zero-res,│
                                           ▼                          logged here       │ abandonment,  │
                            ╔══════════════════════════════╗                            │ catalogue gaps│
                            ║   shared VectorIndex          ║ ◀──────────────────────── │ (embed+cluster│
                            ║   Chroma (default) │ numpy    ║                            │  failed qs)   │
                            ║   ONE text vector / product   ║                            └───────────────┘
                            ╚══════════════════════════════╝
                                           ▲
                            ┌──────────────┴───────────────┐    INGEST  (run once, offline)
                            │ AttributeExtraction + Desc.   │    gpt-4o-mini vision  (enrich)
                            │        →  embed text          │    text-embedding-3-small (batched)
                            └───────────────────────────────┘
```

Each agent is a class with a single typed `run(...)` method, Pydantic models on
every boundary, and no hidden global state — so the orchestrator wires them
together mechanically.

- **OrchestratorAgent** — classifies intent (text / image / combined /
  refinement), routes to the right specialist, holds per-session conversational
  state (so "now show me cheaper ones in blue" refines the previous query), and
  logs every request to analytics.
- **TextSearchAgent** — parses the NL query into `{semantic_query, category?,
  filters}` with structured output, embeds the semantic query, cosine-ranks, then
  applies hard filters + category navigation.
- **ImageSearchAgent** — captions the query image with vision, embeds the
  caption, cosine-ranks (describe-then-embed; see below).
- **MultimodalSearchAgent** — captions the image, fuses it with the text query
  (weighted mean of the two embeddings, `W_TEXT` configurable, default 0.5),
  ranks, and overlays hard attribute filters parsed from the text.
- **AttributeExtractionAgent** — vision → strict JSON `{colour, style, material,
  shape, extras}` via structured output. Used at ingest and for query images.
- **DescriptionAgent** — vision → a 1–3 sentence description that mentions the
  key attributes. Used at ingest.
- **AnalyticsAgent** — logs every query (modality, result count, latency, click)
  to SQLite and computes total queries, zero-result rate, abandonment rate, CTR,
  and **catalogue gaps** (clusters of failed queries).

### Ingest pipeline (describe-then-embed)

Run once with `make ingest`. Each product image is turned into text, then a
single vector. Vision calls are one-per-product; embeddings are **batched** into
a single call for the whole catalogue.

```
  product image (.jpg)
        │
        ▼
  ┌──────────────────────────────┐   ONE structured vision call (strict JSON)
  │ enrich()   gpt-4o-mini vision │ ─────────────────────────────────────────────┐
  └──────────────┬───────────────┘                                               │
                 ▼                                                                ▼
        Attributes{colour,style,material,shape}              description (1–3 sentences)
                 │                                                                │
                 └──────────────┬─────────────────────────────────────────────────┘
                                ▼
        embedding_text = "title · category · attributes · description"
                                │
                                ▼
                 ┌──────────────────────────────┐   batched: ALL products → ONE call
                 │ text-embedding-3-small (1536) │
                 └──────────────┬───────────────┘
                                ▼
                 VectorIndex (Chroma/numpy)  →  one vector per product, persisted
```

### Query-time routing

The orchestrator picks a flow from what's present in the request:

```
  has text & image ─▶ COMBINED   ─▶ MultimodalSearchAgent
  has image only   ─▶ IMAGE_ONLY ─▶ ImageSearchAgent
  has text only    ─▶ TEXT_ONLY  ─▶ TextSearchAgent
       └ + prior session context & refinement cues ─▶ REFINEMENT
         (gpt-4o-mini rewrites "now cheaper in blue" into a standalone query,
          still handled by TextSearchAgent)
```

### How multimodal (image + text) is processed

This is the heart of the system. Because embeddings are **text-only**, an image
is never embedded as pixels — it is **described, then embedded** (a caption), and
fused with the text query in the *same* 1536-dim text space. Two signals go in,
one ranked result list comes out, and hard attribute filters parsed from the text
are overlaid on top.

```
                 ┌───────────────────────────────────────────────────────────────┐
  query IMAGE ──▶│ gpt-4o-mini vision → caption ──▶ text-embedding-3-small → v_img │
                 └───────────────────────────────────────────────────────────────┘
                                                                       │
  query TEXT ──▶ gpt-4o-mini parse ──▶ semantic_query ─▶ embed ─▶ v_txt │
   "in red          │  (structured)        + filters{colour,material,…} │
    leather"        │                              │                    │
                    │                              │                    ▼
                    │                              │   fused = W_TEXT·v_txt + (1−W_TEXT)·v_img
                    │                              │            (weighted mean, W_TEXT=0.5)
                    │                              │                    │
                    │                              │                    ▼
                    │                              │      cosine search over VectorIndex
                    │                              │                    │
                    │                              ▼                    ▼
                    └──────────────────────▶ hard-filter overlay  →  ranked ProductMatch[]
                                          (keep red + leather items;   (reflects BOTH the
                                           soft-fail if it empties)     visual style and the
                                                                        described attributes)
```

- **`v_img`** carries the *visual* signal (what the photo looks like, via its caption).
- **`v_txt`** carries the *described* signal (what the shopper asked for).
- **Weighted-mean fusion** (`W_TEXT` in `config.py`, default `0.5/0.5`) blends them
  before the search, so neither dominates.
- **Hard filters** parsed from the text (`colour=red, material=leather`) are applied
  *after* ranking to enforce explicit constraints — this is why the combined run
  surfaces the red-leather chair where an image-only run of a green chair does not.

### Vector store: Chroma (default) or numpy

The `VectorIndex` interface has two interchangeable backends, picked via
`VECTOR_BACKEND`:

- **`chroma`** (default) — a persistent [Chroma](https://www.trychroma.com/) DB
  via `langchain-chroma`. We embed once (batched) with the official OpenAI SDK
  and hand Chroma the **precomputed** vectors, so Chroma never re-embeds and the
  API budget is unaffected.
- **`numpy`** — in-memory cosine (L2-normalise once, dot product) persisted to
  `.npz`. Zero extra dependencies.

```bash
VECTOR_BACKEND=numpy make ingest   # opt out of Chroma if you prefer
```

## The embedding approach: describe-then-embed

OpenAI embeddings are **text-only**, so there is exactly **one vector space —
text**. Every product gets a single vector built from
`title + category + attributes + description`. A query *image* is turned into
text first (caption + attributes via `gpt-4o-mini` vision), then embedded and
matched against the same product vectors.

**Honest limitation:** image search here is **semantic / caption-based**
matching, *not* pixel-level visual similarity. Two visually different chairs that
caption similarly will match; a subtle pattern the captioner misses will not be
searchable. That is the correct, cheap choice for this project. Pixel-level
visual similarity (CLIP / SigLIP) is noted as a `# REVISIT`, not built.

## Models we use — and why

All model strings live in one place (`catalogue_intel/config.py`). Both are the
small, cheap tier on purpose: a college-project scope rewards cost discipline,
and these two models cover every task in the system.

| Model | Used for | Why this model |
|---|---|---|
| **`gpt-4o-mini`** | attribute extraction, description writing, image captioning, NL-query parsing, intent routing / refinement rewrite | The cheapest **vision-capable** chat model that also supports **strict structured outputs** (json_schema). One model handles every chat *and* vision task, so there's nothing else to provision. Flagship models (gpt-5.x, o-series) are deliberately avoided — they cost far more for no benefit at this scale. |
| **`text-embedding-3-small`** | embedding products and queries into the shared vector space (1536-dim) | Cheap, fast, and 1536-dim is plenty for a small catalogue. It's **text-only**, which is exactly what drives the *describe-then-embed* design. `text-embedding-3-large` (3072-dim) would cost more and add nothing here. |

> **One vector space.** Because the embedding model is text-only, images and text
> queries all become **text** before embedding — so there is a single 1536-dim
> space and no separate image-vector path to maintain.

## Tech stack — and why

| Tool | Role in the project | Why it's here |
|---|---|---|
| **`openai`** (official SDK) | all model calls — `chat.completions.parse`, `embeddings.create`, vision content parts | PRD-mandated; gives first-class **structured outputs**, vision, and embeddings without a framework in the way. |
| **`langchain-chroma` + `chromadb`** | default vector store (`VECTOR_BACKEND=chroma`) — persistent, cosine, on-disk | A real, industry-standard **vector database**. We hand it **precomputed** vectors (we embed once, batched, with the OpenAI SDK), so it never re-embeds and the API budget is untouched. Persists across runs. |
| **`numpy`** | fallback vector index (`VECTOR_BACKEND=numpy`) + the fusion math (`W_TEXT·v_txt + (1−W_TEXT)·v_img`) | Zero-dependency cosine index for when you don't want Chroma, and the array math behind multimodal fusion and cosine ranking. |
| **`pydantic`** | typed contracts on every agent boundary (`Product`, `ProductMatch`, …) + the flat DTOs that drive strict JSON-schema structured output | Turns "parse the model's JSON" into a validated, typed object — **no regex parsing**, ever. |
| **`pillow`** | generates the reproducible fixture furniture images | Lets the repo ship a self-contained, network-free test catalogue (`make fixtures`). |
| **`python-dotenv`** | loads `OPENAI_API_KEY` from a local `.env` | Keeps secrets out of the code; the app fails loudly if the key is missing. |
| **`rich`** | the `make smoke` PASS/FAIL report table + headers | A clear, scannable end-to-end report with per-test latency and measured values. |
| **`pytest`** | dev test runner (each agent also ships a `python -m … ` self-check) | Fast structural checks during development. |
| **SQLite** (stdlib `sqlite3`) | the AnalyticsAgent store (queries, clicks, latency) | Zero-setup, file-based persistence for analytics — no server to run. |

> **Not used (and why):** `unstructured.io` — it parses *documents* (PDF/DOCX/HTML)
> for RAG, but this pipeline ingests **product images** read by a vision model, so
> there's nothing for it to do. Full LangChain LLM wrappers were skipped too: the
> PRD mandates the official OpenAI SDK directly, and we kept that. We adopted only
> the `langchain-chroma` piece, which genuinely fits.

## Setup

```bash
make venv                     # create .venv and install deps into it
source .venv/bin/activate
cp .env.example .env          # then put your key in .env:  OPENAI_API_KEY=sk-...
make ingest                   # enrich + embed + persist the fixture catalogue
make smoke                    # END-TO-END test against the REAL OpenAI API
```

(`make install` installs into the active interpreter instead of a venv.)

Requirements: Python 3.11+, an `OPENAI_API_KEY`. The key is read from the
environment or a local `.env` (never hardcoded; the app fails loudly if missing).

**Low rate-limit quotas:** set `OPENAI_RPM` to pace requests client-side and
avoid 429s — e.g. `OPENAI_RPM=3 make smoke` spaces calls ~21s apart. The smoke
suite is tuned to make only **~18 real API calls** total (combined
attribute+description vision calls, fully batched embeddings).

`make smoke-offline` runs the same suite with a stubbed client (no key, no
network) for fast **structural** checks — semantic-accuracy assertions are
relaxed in that mode. The default `make smoke` is the real end-to-end test.

## Fixtures

`catalogue_intel/fixtures/` ships a small, reproducible catalogue — 10 product
images across 4 categories (4 chairs, 2 lamps, 2 sofas, 2 tables) plus 3 held-out
query images that are **not** in the catalogue — all recorded in `manifest.json`.
The set is kept deliberately small because the dominant cost under a tiny RPM
quota is one vision call per product at ingest. The
images are simple generated schematic furniture (regenerate with
`make fixtures`). The smoke suite reads only from fixtures; no network image
fetching at test time.

## Smoke suite (`make smoke`)

Ingests the fixtures, exercises every feature against the real API, and prints a
`rich` PASS/FAIL table with **measured** numbers (latency, real top-k hits) and a
header showing the model strings, total OpenAI calls, and wall-clock. Exits
non-zero if any required test fails. Numbers are never hardcoded — a truthful
FAIL beats a green lie.

| # | Test | What it checks |
|---|------|----------------|
| 1 | Ingest | every product has non-empty attributes + description + a 1536-dim vector |
| 2 | Text search | "comfortable blue office chair" → non-empty ranked hits, top category = chairs |
| 3 | Image search | each held-out query image → top-1 category accuracy across the set |
| 4 | Combined | chair image + "in red leather" → chair on top AND more red/leather than image-only (reports Δ) |
| 5 | Attribute extraction | query image → non-empty colour/style/material/shape |
| 6 | Description | 1–3 sentences, mentions ≥1 extracted attribute |
| 7 | Analytics | scripted session → correct zero-result count, abandonment, CTR, ≥1 catalogue gap |
| 8 | Routing | text→TextSearchAgent, image→ImageSearchAgent, both→MultimodalSearchAgent |

### Latest smoke run

**8/8 PASS on real `gpt-4o-mini` + `text-embedding-3-small` calls** — 18 API
calls, 366s wall (paced at `OPENAI_RPM=3`), Chroma backend, 10-product catalogue.

| Test | Measured result |
|---|---|
| 1 Ingest | 10 products enriched, all vectors 1536-dim, attrs+desc non-empty |
| 2 Text search | `comfortable blue office chair` → top = chairs (`c1`, 0.758) |
| 3 Image search | top-1 category accuracy **100% (3/3)** |
| 4 Combined | "in red leather" → top = chairs; avg red/leather reflection combined **2.00** vs image-only **0.00** (**Δ=+2.00**) |
| 5 Attribute extraction | colour=Green, style=Modern, material=Wood, shape=Rectangular |
| 6 Description | 2 sentences, mentions attributes ✅ |
| 7 Analytics | zero-result=1, CTR=50%, abandon=50%, 1 catalogue gap |
| 8 Routing | text→Text, image→Image, both→Multimodal ✅ |

Re-run anytime with `OPENAI_RPM=3 make smoke`.

## Layout

```
catalogue_intel/
  config.py            model strings, weights, paths, embed dim
  openai_client.py     SDK wrapper (retry/backoff) + deterministic offline stub
  models.py            pydantic contracts (Product, ProductMatch, ...)
  vector_index.py      Chroma (langchain-chroma) + numpy backends, one interface
  agents/              orchestrator + 6 specialists + shared filter helpers
  ingest.py            build catalogue: enrich + embed + persist
  fixtures/            tiny product/query images + manifest.json
  smoke/run_smoke.py   one-command E2E runner with a rich report
```
