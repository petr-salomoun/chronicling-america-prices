# Chronicling America → Social Graph + Historical Prices: End-to-End Pipeline Design

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 0: DATA ACQUISITION                          │
│  download_chronicling_america.py                                           │
│  ┌─────────┐    ┌──────────┐    ┌─────────────────────────────────────┐   │
│  │ LOC API  │───▶│ Per-year │───▶│ data/raw/{year}/{lccn}/{date}/     │   │
│  │ crawl    │    │ batching │    │   {seq}.txt  (OCR plain text)      │   │
│  └─────────┘    └──────────┘    │   _meta.json (title, date, page#)  │   │
│                                  └─────────────────────────────────────┘   │
│  • Resumable: tracks state in data/raw/_progress.json                     │
│  • Rate-limited: respects LOC's 20 req/min                                │
│  • Year-by-year: separate directory per year (1770–1963)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 1: LLM EXTRACTION (Condensation)                 │
│  extract_pass1.py (NOT delivered now — design only)                        │
│                                                                            │
│  For each OCR page text file:                                              │
│  ┌──────────┐   ┌────────────┐   ┌──────────────────────────────────────┐ │
│  │ Raw OCR  │──▶│ LLM Pass 1 │──▶│ data/pass1/{year}/{lccn}_{date}.jsonl│ │
│  │ .txt     │   │ (extract)  │   │                                      │ │
│  └──────────┘   └────────────┘   │ Each line = one extraction:          │ │
│                                   │  {"type":"person_mention",           │ │
│                                   │   "names":["J. Smith","R. Brown"],   │ │
│                                   │   "context":"business partners...",  │ │
│                                   │   "ref":"sun_18840312_p3"}          │ │
│                                   │  {"type":"price",                    │ │
│                                   │   "item":"flour per barrel",         │ │
│                                   │   "price":"$5.25",                   │ │
│                                   │   "context":"ad: wholesale...",      │ │
│                                   │   "ref":"sun_18840312_p3"}          │ │
│                                   └──────────────────────────────────────┘ │
│  • Maximal condensation: throws away everything except people co-mentions │
│    and pricing information                                                 │
│  • Keeps minimal provenance: newspaper LCCN + date + page sequence         │
│  • Runs in streaming batches — one JSONL per issue (all pages merged)      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 PHASE 2: LLM NORMALIZATION (Structuring)                   │
│  normalize_pass2.py (NOT delivered now — design only)                      │
│                                                                            │
│  ┌────────────────────┐      ┌────────────────────────────────────┐       │
│  │ Pass1 JSONL        │─────▶│ data/pass2/                        │       │
│  │ (condensed extracts)│      │   social_graph/                    │       │
│  └────────────────────┘      │     nodes.parquet    (person_id,   │       │
│                               │       canonical_name, aliases,     │       │
│         ┌──────────────┐      │       first_seen, last_seen,      │       │
│         │ Reference    │      │       occupation, location)        │       │
│         │ dictionaries │─────▶│     edges.parquet    (person_a_id, │       │
│         │ (built incr.)│      │       person_b_id, co_mention_ct, │       │
│         └──────────────┘      │       first_date, last_date,      │       │
│                               │       rel_types[], refs[])        │       │
│                               │   prices/                          │       │
│                               │     prices.parquet  (date,         │       │
│                               │       commodity_id, commodity_name,│       │
│                               │       category_l1, category_l2,   │       │
│                               │       price_numeric, currency,    │       │
│                               │       unit, location, ref)        │       │
│                               │     commodities.parquet (id, name, │       │
│                               │       category hierarchy,          │       │
│                               │       synonyms[])                  │       │
│                               └────────────────────────────────────┘       │
│  • Entity resolution: deduplicate people across articles                   │
│  • Commodity standardization: hierarchical categorization                  │
│  • Price normalization: numeric extraction, unit standardization            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 3: ANALYSIS & EXPLORATION                        │
│  (Future — not designed here)                                              │
│                                                                            │
│  Social graph: NetworkX / Neo4j → community detection, centrality,         │
│                temporal evolution of social networks                        │
│  Prices:       DuckDB / Pandas → commodity price indices, inflation,       │
│                regional price differentials, arbitrage detection            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Data Acquisition — Detailed Design

### Chronicling America API Structure

The LOC API provides access to OCR text at the **page** level:

```
https://chroniclingamerica.loc.gov/lccn/{lccn}/{date}/ed-{edition}/seq-{sequence}/ocr.txt
```

**Discovery flow:**
1. `GET /newspapers.json` → list of all newspaper titles (LCCN codes)
2. `GET /lccn/{lccn}.json` → list of all issues for that title (with dates)
3. `GET /lccn/{lccn}/{date}/ed-1.json` → list of pages for that issue
4. `GET /lccn/{lccn}/{date}/ed-1/seq-{n}/ocr.txt` → raw OCR text

**Rate limit:** LOC allows ~20 requests/minute. Exceeding triggers 429 or CAPTCHA for ~1 hour. The downloader enforces 3.1-second spacing.

### Directory Structure

```
data/
├── raw/
│   ├── _progress.json              # Global resume state
│   ├── _newspapers.json            # Cached newspaper list
│   ├── 1770/
│   │   └── sn83045462/             # LCCN directory
│   │       ├── _title_meta.json    # Title metadata (cached)
│   │       ├── 1770-01-05/
│   │       │   ├── _issue_meta.json
│   │       │   ├── seq-001.txt
│   │       │   ├── seq-002.txt
│   │       │   └── ...
│   │       └── 1770-01-12/
│   │           └── ...
│   ├── 1771/
│   │   └── ...
│   └── ...
├── pass1/                          # Phase 1 output (future)
└── pass2/                          # Phase 2 output (future)
```

### Resume Strategy

`_progress.json` tracks:
```json
{
  "newspapers_fetched": true,
  "years_completed": [1770, 1771, 1772],
  "current_year": 1773,
  "current_year_titles_completed": ["sn83045462", "sn84026749"],
  "current_title": "sn85038115",
  "current_title_issues_completed": ["1773-01-02", "1773-01-09"],
  "stats": {
    "total_pages_downloaded": 1423567,
    "total_bytes": 8734523456,
    "errors": 42,
    "last_updated": "2026-03-27T14:30:00Z"
  }
}
```

On restart, the script reads this file and skips all completed work. Each successfully downloaded page is committed before advancing state. If the process crashes mid-page, only that one page is re-downloaded.

### Estimated Scale

| Metric | Estimate |
|--------|----------|
| Newspaper titles | ~3,000 |
| Total issues | ~3,000,000 |
| Total pages | ~20,000,000 |
| OCR text per page | ~2–8 KB |
| Total raw text | ~60–120 GB |
| Download time at 20 req/min | ~2 years for pages alone |

**Optimization strategy for realistic timelines:**
- Download the newspaper + issue index first (much smaller: ~30,000 API calls)
- Then download OCR text page-by-page, year-by-year
- Support `--year-start` and `--year-end` to parallelize across machines
- Support `--state` to filter by US state for faster targeted downloads
- Consider LOC bulk data if available (see below)

### LOC Bulk Data Alternative

LOC provides bulk access via their [data.gov dumps](https://www.loc.gov/collections/chronicling-america/) and AWS S3 open data:
- `s3://chronicling-america-bulk/` (OCR XML and ALTO files)
- These are faster than API calls but require ALTO XML parsing

The downloader in this design uses the API approach (simpler, more resumable). A future version could add an `--bulk-s3` mode.

---

## Phase 1: LLM Extraction — Detailed Design

### Input/Output Contract

**Input:** One OCR text file (`seq-NNN.txt`), typically 2–8 KB of noisy 19th/20th-century OCR.

**Output:** JSONL with two record types, appended to `data/pass1/{year}/{lccn}_{date}.jsonl`:

```jsonl
{"t":"P","names":["John Smith","Robert Brown","Mary Wilson"],"rel":"business partners in railroad venture","ref":"sn83045462/1884-03-12/p3"}
{"t":"P","names":["President Cleveland","Senator Sherman"],"rel":"Cleveland criticized Sherman's tariff bill","ref":"sn83045462/1884-03-12/p3"}
{"t":"$","item":"flour, superfine, per barrel","price":"$5.25","ref":"sn83045462/1884-03-12/p3"}
{"t":"$","item":"house, 3-story brick, Elm Street","price":"$4,500","note":"for sale ad","ref":"sn83045462/1884-03-12/p3"}
{"t":"$","item":"domestic servant, weekly wage","price":"$3.50","ref":"sn83045462/1884-03-12/p3"}
```

**Key principles:**
- `t` field: `"P"` for people co-mention, `"$"` for price/value
- `ref`: compact provenance string: `{lccn}/{date}/p{page}` — enough to retrieve original
- `rel` / `note`: free-text context from the article (1 sentence max) — not normalized yet
- `names`: array of 2+ people mentioned in the same article context (co-occurrence = graph edge)
- **No full article text preserved** — this is maximal condensation
- Single-person mentions (no co-occurrence) are DROPPED in pass 1 (no graph edge possible)
- Prices without a discernible item/commodity are DROPPED

### LLM Prompt — Pass 1 (Extraction & Condensation)

```
SYSTEM:
You are a precise data extraction engine processing OCR text from historical
American newspapers (1770–1963). The OCR is noisy — expect misspellings,
broken words, and garbled characters. Use context to infer correct readings.

Your task: extract ONLY two types of information, discard everything else.

TYPE 1 — PEOPLE CO-MENTIONS (output tag: "P")
Find groups of 2 or more named people mentioned in the SAME article or
paragraph context. For each group, output:
- "names": array of full names as they appear (fix obvious OCR errors)
- "rel": one short phrase describing their relationship or why they appear
  together (e.g. "married", "plaintiff and defendant", "board members",
  "mentioned in same crime report"). Keep under 15 words.

Rules:
- A "person" must be a specific named individual, not a generic role
- Include titles if present (Dr., Gen., Mrs., Rev., Sen., etc.)
- If the SAME person appears with different people in different contexts
  within the same page, output SEPARATE records for each group
- Do NOT output single-person mentions — minimum 2 people per record
- Advertisements listing multiple business owners count
- Obituaries mentioning surviving family members count

TYPE 2 — PRICES AND MONETARY VALUES (output tag: "$")
Find any mention of a specific price, wage, cost, rent, or monetary value
attached to an identifiable item, commodity, service, property, or wage.
- "item": what is being priced, as specifically as stated (include
  quantity/unit if given, e.g. "wheat per bushel", "board per week")
- "price": the price as stated, including currency symbol and original format
- "note": optional, 1–5 words of context if helpful (e.g. "auction",
  "wholesale", "for sale ad", "government contract")

Rules:
- Include prices from: articles, advertisements, market reports, auction
  notices, real estate listings, help wanted ads, legal notices
- Include wages, rents, fares, tolls, fees, fines, bounties, rewards
- Do NOT include numbers that are not prices (vote counts, populations, etc.)
- If a market report lists many commodities, output one record per commodity
- Preserve the original price format — do NOT convert or normalize

OUTPUT FORMAT:
Return a JSON array. Each element is one of:
  {"t":"P","names":[...],"rel":"..."}
  {"t":"$","item":"...","price":"...","note":"..."}

If the page contains NO extractable people co-mentions and NO prices,
return exactly: []

Do NOT include any text outside the JSON array.
```

```
USER:
Newspaper: {title} ({lccn})
Date: {date}
Page: {sequence}

--- OCR TEXT START ---
{ocr_text}
--- OCR TEXT END ---
```

### Batching & Cost Strategy

| Parameter | Value |
|-----------|-------|
| Context window needed | ~4K tokens input (OCR page) + ~1K output |
| Recommended model | GPT-4o-mini or Claude 3.5 Haiku (cost-optimized) |
| Cost per page (estimate) | ~$0.001–0.003 |
| Cost for 20M pages | ~$20,000–60,000 |
| Throughput at 10K RPM | ~7 days for full corpus |

**Cost reduction strategies:**
- **Pre-filter:** Skip pages that are clearly ads-only (detect via keyword density) or are blank/garbled (< 100 readable characters). Estimate: eliminates ~30% of pages.
- **Chunk large pages:** Pages over 6K tokens get split at paragraph boundaries.
- **Batch API:** Use OpenAI Batch API (50% discount) or Anthropic batch mode.
- **Yearly prioritization:** Start with high-value decades (1850–1910 for social graph density; 1840–1930 for price data richness).

---

## Phase 2: LLM Normalization — Detailed Design

### 2A: Social Graph Normalization

**Goal:** Resolve the raw `names` arrays from Pass 1 into a unified person entity graph.

**Strategy (incremental, LLM-assisted):**

1. **Build raw name frequency table** from all Pass 1 `"P"` records
2. **Cluster by string similarity** (Jaro-Winkler > 0.92) within same decade + state → candidate merge groups
3. **LLM disambiguation** for ambiguous clusters:

#### LLM Prompt — Pass 2A (Person Entity Resolution)

```
SYSTEM:
You are a historical person entity resolution engine. You receive a cluster
of name variants extracted from American newspapers, with metadata about
when and where each variant appeared and who they co-occurred with.

Your task: determine which variants refer to the SAME real person, and
which are DIFFERENT people who happen to have similar names.

Output a JSON object:
{
  "entities": [
    {
      "canonical_name": "John Adams Smith",
      "aliases": ["J. A. Smith", "Jno. A. Smith", "John A. Smith, Esq."],
      "occupation": "lawyer" | null,
      "location": "Springfield, IL" | null,
      "confidence": 0.95,
      "notes": "Consistently mentioned with Judge Brown in legal contexts"
    },
    {
      "canonical_name": "John B. Smith",
      "aliases": ["John Smith (of Chicago)", "J. B. Smith"],
      ...
    }
  ]
}

Reasoning guidelines:
- Same name + same co-occurring people + same location → likely same person
- Same name + different decades + different locations → likely different people
- Titles (Dr., Rev., Gen.) are strong disambiguators
- "Mrs. John Smith" and "Mary Smith (wife of John)" may be the same person
- Default to SPLITTING (creating separate entities) when uncertain
```

```
USER:
Name cluster to resolve: "John Smith" (and variants)
Appearances:
- "J. Smith" — co-occurred with "Robert Brown", "Mary Wilson" — Springfield IL newspapers, 1882–1889 (23 times)
- "John A. Smith, Esq." — co-occurred with "Judge Brown", "Sen. Douglas" — Springfield IL, 1884–1891 (15 times)
- "John Smith" — co-occurred with "Captain Jones" — Chicago Tribune, 1885 (2 times)
- "Jno. Smith" — co-occurred with "Robert Brown" — Springfield IL, 1883 (4 times)
```

4. **Assign persistent IDs:** `P{8-digit-hash}` derived from canonical name + location + decade
5. **Build edge list:** Every Pass 1 `"P"` record with N names generates N×(N-1)/2 edges

**Output schema — `nodes.parquet`:**

| Column | Type | Description |
|--------|------|-------------|
| person_id | string | `P00000001` — stable unique ID |
| canonical_name | string | Best canonical form |
| aliases | string[] | All observed name variants |
| title | string | Dr., Gen., Mrs., etc. (if any) |
| occupation | string | If extractable |
| locations | string[] | Associated locations |
| first_seen_date | date | Earliest mention |
| last_seen_date | date | Latest mention |
| mention_count | int | Total co-mention appearances |

**Output schema — `edges.parquet`:**

| Column | Type | Description |
|--------|------|-------------|
| person_a_id | string | FK to nodes |
| person_b_id | string | FK to nodes |
| co_mention_count | int | Times mentioned together |
| first_date | date | Earliest co-mention |
| last_date | date | Latest co-mention |
| relationship_types | string[] | Aggregated `rel` descriptions |
| sample_refs | string[] | Up to 5 source references |

### 2B: Price Normalization

**Goal:** Convert free-text price records into structured, queryable commodity price time series.

**Strategy:**

1. **Build raw item frequency table** from all Pass 1 `"$"` records
2. **LLM-assisted commodity taxonomy mapping:**

#### LLM Prompt — Pass 2B (Commodity Standardization)

```
SYSTEM:
You are a historical commodity classification engine. You receive raw item
descriptions extracted from American newspaper prices (1770–1963) and must
map each to a standardized commodity hierarchy.

The hierarchy has 3 levels:
  L1: broad category (e.g., "Food & Agriculture", "Real Estate", "Labor",
      "Manufactured Goods", "Transportation", "Financial", "Services")
  L2: sub-category (e.g., "Grains", "Livestock", "Residential Property",
      "Domestic Labor", "Textiles")
  L3: specific commodity (e.g., "Wheat", "Beef Cattle", "Cotton Cloth")

Output JSON:
{
  "commodity_id": "FA-GR-WHEAT",
  "commodity_name": "Wheat",
  "category_l1": "Food & Agriculture",
  "category_l2": "Grains",
  "category_l3": "Wheat",
  "unit_standard": "bushel",
  "notes": null
}

Rules:
- Map to the MOST SPECIFIC level possible
- If the item is a wage, classify under "Labor" with appropriate sub-category
- If the item is real estate, classify under "Real Estate"
- For compound items ("board and lodging per week"), split if possible
- If you cannot determine the commodity, set commodity_id to "UNKNOWN"
  and preserve the original item text in notes
- Use consistent IDs: {L1_abbrev}-{L2_abbrev}-{L3_abbrev}
```

```
USER:
Item descriptions to classify (batch):
1. "flour, superfine, per barrel"
2. "house, 3-story brick, Elm Street"
3. "domestic servant, weekly wage"
4. "cotton, middling, per pound"
5. "passage to Liverpool, steerage"
```

3. **Price parsing:** Extract numeric value, currency, and unit from the `price` field using regex + LLM fallback for unusual formats (e.g., "two dollars and six bits", "5s 6d")

4. **Currency normalization table:**

| Era | Currencies encountered | Normalization |
|-----|----------------------|---------------|
| 1770–1790 | British pounds, Continental dollars, Spanish dollars | Keep original + provide USD equivalent where possible |
| 1790–1963 | USD dominant | Normalize to decimal USD |
| Throughout | Regional/slang ("bits", "picayune", "levy") | Map to USD fraction |

**Output schema — `prices.parquet`:**

| Column | Type | Description |
|--------|------|-------------|
| date | date | Publication date |
| commodity_id | string | FK to commodities table |
| price_original | string | As stated in newspaper |
| price_usd | float | Normalized to USD (null if impossible) |
| unit | string | Standardized unit |
| quantity | float | If stated (default 1.0) |
| price_per_unit | float | Computed: price_usd / quantity |
| location_paper | string | Newspaper's city/state |
| context | string | "ad" / "market_report" / "article" / "auction" / "legal" |
| ref | string | Source reference |

**Output schema — `commodities.parquet`:**

| Column | Type | Description |
|--------|------|-------------|
| commodity_id | string | `FA-GR-WHEAT` |
| commodity_name | string | Canonical name |
| category_l1 | string | Broad category |
| category_l2 | string | Sub-category |
| category_l3 | string | Specific commodity |
| unit_standard | string | Default unit |
| synonyms | string[] | All observed item descriptions that map here |

---

## Phase 3: Strategy for Exploration & Use

### Social Graph Analysis Roadmap

| Analysis | Method | Expected Discovery |
|----------|--------|--------------------|
| Community detection | Louvain/Leiden on co-mention graph | Identify social circles, political factions, business networks invisible to historians |
| Temporal evolution | Sliding-window graph snapshots (decade) | Watch communities form, merge, dissolve over 200 years |
| Bridge detection | Betweenness centrality | Find people who connected otherwise separate social worlds |
| Geographic network | Bipartite projection (person × newspaper-city) | Map social connectivity between cities before telecommunications |
| Role classification | Node2Vec + clustering | Automatically classify nodes as politicians, businessmen, clergy, criminals, etc. |
| Influence propagation | Temporal motif mining | How news about a person spreads from local to national press |

### Price Analysis Roadmap

| Analysis | Method | Expected Discovery |
|----------|--------|--------------------|
| Commodity price indices | Median price per commodity per month | First comprehensive US price index from primary sources (not government statistics) |
| Regional price differentials | Price comparison across newspaper locations | Map market integration: when did prices converge between regions? |
| Arbitrage detection | Price spread analysis across simultaneous markets | Identify historical market inefficiencies and their resolution |
| Inflation measurement | Basket-of-goods tracking over decades | Independent inflation measure to cross-validate official CPI (which starts 1913) |
| War/crisis price shocks | Event study around known crises | Quantify economic impact of Civil War, panics, WWI/WWII at commodity level |
| Seasonal patterns | Fourier analysis of price time series | Reveal agricultural calendar effects on prices |
| Wage-price dynamics | Joint analysis of wages and goods prices | Real wage calculation at city level over 200 years |

### Technical Stack Recommendation

| Component | Tool | Why |
|-----------|------|-----|
| Storage (raw) | Local filesystem / S3 | Simple, cheap, scales to 100GB+ |
| Storage (structured) | Parquet files + DuckDB | Column-oriented, fast analytics, no server needed |
| Graph store | NetworkX (analysis) + Neo4j (exploration) | NetworkX for computation, Neo4j for interactive exploration |
| LLM processing | OpenAI Batch API / Anthropic batch | 50% cost reduction, async processing |
| Orchestration | Simple Python + progress JSON | No Airflow/Prefect needed — the pipeline is linear |
| Visualization | Gephi (graph), Plotly/Matplotlib (prices) | Standard, well-documented |

### Reference Dictionaries to Build/Source

| Dictionary | Source | Purpose |
|------------|--------|---------|
| US newspaper title → city/state | LOC API metadata | Geolocate every price and person mention |
| Historical given name variants | IPUMS name crosswalk | "Jno." = "John", "Wm." = "William", "Eliz." = "Elizabeth" |
| Historical occupations | HISCO classification | Standardize occupation mentions |
| Commodity synonyms | Historical commodity dictionaries + iterative LLM extraction | "superfine flour" = "flour, superfine grade" |
| Historical currency/unit table | Officer & Williamson "MeasuringWorth" | Convert historical monetary units |
| US city gazetteer (historical) | GNIS + Atlas of Historical County Boundaries | Resolve historical place names |

---

## Estimated Total Project Costs

| Phase | Compute | LLM API | Storage | Time |
|-------|---------|---------|---------|------|
| Phase 0 (download) | Minimal (network I/O) | $0 | 60–120 GB | 3–12 months (API) or days (bulk S3) |
| Phase 1 (extraction) | Minimal | $20K–60K (full corpus) | ~5 GB (JSONL) | 1–2 weeks |
| Phase 2 (normalization) | Moderate (entity resolution) | $2K–5K | ~2 GB (Parquet) | 1 week |
| **Total** | **Low** | **$22K–65K** | **~130 GB** | **4–14 months** |

### Budget Optimization: Start Small

**Recommended pilot:** Pick **one state** (e.g., New York) for **one decade** (e.g., 1880–1890).
- ~50K–100K pages → ~$100–300 in LLM costs
- Validates entire pipeline end-to-end
- Produces a publishable micro-dataset
- Informs cost/quality tradeoffs before scaling

---

## File Manifest

| File | Status | Description |
|------|--------|-------------|
| `chronicling_america_pipeline_design.md` | ✅ This file | Full architecture and prompt design |
| `download_chronicling_america.py` | ✅ Delivered | Robust, resumable downloader |
| `extract_pass1.py` | 🔜 Future | LLM extraction (Phase 1) |
| `normalize_pass2.py` | 🔜 Future | LLM normalization (Phase 2) |
| `analyze_graph.py` | 🔜 Future | Social graph analysis |
| `analyze_prices.py` | 🔜 Future | Price time-series analysis |