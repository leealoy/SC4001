# Does the artwork matter? Pokémon card price prediction across unseen sets

SC4001 group project. This repository currently holds the **data, split and
metadata-baseline** part (Member 1). Image encoders / fusion models and the
robustness experiments are added by Members 2 and 3.

## Setup

Python 3.11. Dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing the data and baselines

Run everything from the repository root.

### 1. Download cards and prices (Pokémon TCG API)

```bash
python pokeprice/fetch_cards.py            # metadata + prices -> data/cards.csv
python pokeprice/fetch_cards.py --images   # optional: also download card images
```

* No API key is needed. A free key from pokemontcg.io raises the rate limit:
  `export POKEMONTCG_API_KEY=...` (never commit it).
* The public API often returns HTTP 500/502; the script retries with backoff and is
  **resumable** (pages are cached in `data/raw/`). Re-run it if it gives up.
* Target = `log1p(TCGplayer market price, USD)` of the card's **standard printing**
  (`normal`, else `holofoil`; vintage sets use `unlimited` / `unlimitedHolofoil`).
  Reverse-holo and 1st-edition prices are ignored.
* Writes `data/download_date.txt`, the date used for the "set age" feature.
* The API had no TCGplayer prices for sets released after 2025-11-14 when we
  downloaded (Oct 2026), so those cards are excluded.

### 2. eBay sold listings (real-world test set)

We do **not** scrape eBay (it breaches eBay's User Agreement) and eBay's sold-item
API (Marketplace Insights) is restricted to approved developers. Sold listings are
collected **by hand** from ebay.com → Sold Items into `data/ebay_sold.csv`
(columns: see `data/ebay_sold_TEMPLATE.csv`). Protocol: cards chosen in advance
across rarities and eras; raw English singles; USD; fixed date window; ≥3 sales
per card; item price only (shipping excluded).

```bash
python pokeprice/build_ebay.py --start 2026-09-01 --end 2026-10-15
```

Filters out graded, lots, fakes, sealed product, non-English, damaged and variant
listings; matches titles to card ids by card number + name + set; removes
per-card price outliers; aggregates to a median per card. Rejected and unmatched
listings are written to `data/ebay_rejected.csv` / `data/ebay_unmatched.csv` and
**must be reviewed by hand**. Output: `data/ebay_cards.csv`.

### Checking the eBay data (dashboard)

```bash
python pokeprice/dashboard_server.py     # then open http://127.0.0.1:8765/
```

Local page for checking the hand-collected data: per card, the clean sales and median from
`build_ebay.py`, plus the rejected / unmatched rows to review by hand ("Review rejected").
It also shows **active** eBay listings (asking prices, eBay Browse API; keys in the
git-ignored `.env`, see `.env.example`). Asking prices are not sale prices and are
**not** used in the experiments.

### 3. Splits

```bash
python pokeprice/make_splits.py --test-months 12 --val-months 12
```

Writes `data/splits.csv` with two columns:

* `split_temporal` (main experiment): whole sets by release date. Test = sets
  released in the last 12 months of priced data, val = the 12 months before, train =
  everything earlier. No set spans two splits (asserted).
* `split_random`: random cards, same split sizes (for comparison only).

eBay cards are labelled `ebay` in both and never used for training or tuning.

### 4. Baselines (metadata only)

```bash
python pokeprice/baselines.py --split split_temporal
python pokeprice/baselines.py --split split_random
```

Models: rarity-median, HistGradientBoosting, MLP; 3 seeds each (rarity-median is
deterministic). Outputs in `results/`: `baseline_summary_*.csv` (mean/std),
`baseline_runs_*.csv` (per seed) and `baseline_preds_*.csv` (per-card predictions
for failure analysis). Metrics: MAE on log price, Spearman, median absolute % error,
share of predictions within 2×.

## Rules we follow

* Encoders, scalers and medians are fitted on train rows only.
* Hyperparameters were fixed in advance, not tuned on the test split or eBay set.
* Test and eBay hold-out sets are scored, never used to choose settings.

## Layout

```
pokeprice/            scripts (fetch_cards, build_ebay, make_splits, baselines, dashboard_server)
data/                 cards.csv, sets.csv, splits.csv, (ebay_*.csv)  [data/raw, images ignored by git]
results/              baseline metrics and predictions
report/               draft report sections
CHANGELOG_AI.md       what the AI assistant wrote/changed (for the AI Use Statement)
```
