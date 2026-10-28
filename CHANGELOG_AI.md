# AI assistance log (Member 1) — for the AI Use Statement

Tool: Claude (Claude Code, model Sonnet 5.5), used 25–27 Oct 2026.

The starter code (`pokeprice/`) did not exist on the machine, so Claude wrote all
four scripts **from scratch** from Member 1's written specification. Nothing was
adapted from earlier code.

## What Claude wrote
* `pokeprice/fetch_cards.py`, `build_ebay.py`, `make_splits.py`, `baselines.py`
* `README.md`, `requirements.txt`, `.gitignore`, `data/ebay_sold_TEMPLATE.csv`
* `report/member1_sections_draft.md` (draft text of dataset / baseline / setup sections)

## Problems found while running the code, and fixes
1. **API outages**: the Pokémon TCG API returned HTTP 500/502 for long stretches. Added
   exponential-backoff retries (up to 20) and page caching so the download resumes.
2. **Missing vintage prices**: ~840 vintage cards were dropped because the API labels
   their standard print `unlimited`/`unlimitedHolofoil`. Added these keys
   (priced cards 18,649 → 19,486).
3. **eBay filter false positive**: the word "collection" in the lot filter rejected
   a legitimate set name ("McDonald's Collection"). Removed it.
4. Removed a no-op assertion and an unused import.

## Verification done
* `fetch_cards.py` run against the live API (20,670 cards downloaded).
* `make_splits.py` run; its leakage assertions passed.
* `baselines.py` run on both splits (3 seeds).
* `build_ebay.py` tested only on **synthetic** listings built from real card names
  (checks filters and matcher mechanically; the 100 % matcher agreement is circular
  and says nothing about real eBay titles). **Not yet run on real eBay data.**

## Not done / human decisions
* eBay sold listings must be collected manually by the team; none exist yet.
* The team chose the research question, data sources, split design and metrics.

## 27–28 Oct 2026: data-checking dashboard (Claude Code, Opus 5.5)
* `pokeprice/dashboard_server.py` + `pokeprice/dashboard.html`: local page to browse cards
  (`data/cards.csv`) and check the hand-collected eBay data. Shows **sold** prices from
  `build_ebay.py`'s outputs (`ebay_cards.csv` median, `ebay_clean.csv` sales), the rejected /
  unmatched rows for hand review, and **active** eBay listings (asking prices) from the eBay
  Browse API, shown separately. TCGplayer prices are not shown.
* eBay's sold-item API (Marketplace Insights) was tried first and abandoned: eBay refused
  access on 27 Oct 2026 (reserved for approved partners), so sold prices stay hand-collected.
* Browse API keys come from the git-ignored `.env` (see `.env.example`); responses are
  cached in `data/ebay_api_cache/` (git-ignored) for 6 h.
* The active-listing title check adds two rules beyond `build_ebay.py`: a `4/102`-style set
  total must match the card's printed total, and reprint-set titles (Celebrations / Classic
  Collection / Base Set 2) are rejected for the original card.
* Verified: Browse API live with production keys (e.g. base1-4, sv3pt5-199, cel25c-4_A);
  sold-data views tested on **synthetic** listings run through the real `build_ebay.py`
  (2 cards, 6 sales), then deleted. No real eBay sold data exists yet.
* Known gap: "TAG" grading is not caught by the `graded` filter (adding it naively would
  reject "Tag Team" cards).

## 28 Oct 2026: eBay card selection (Claude Code, Sonnet 5.5)
* Wrote `pokeprice/select_ebay_cards.py`: seeded stratified (rarity group x era) worklist
  of cards to look up by hand on eBay Sold Items, so card choice is fixed before seeing
  listings. Default seed 42 -> 189 candidates, 15 strata, 8 wanted per stratum (120 cards,
  >= 360 listings at >= 3 sales each). Two strata (common/uncommon, 2020-2025) have only
  3-4 cards priced >= $1, so they are small.
* Design choices that bias the eBay set (to be stated in the report): price floor $1, no promos.
* Not yet verified against real eBay listings; no real sold data exists yet.
