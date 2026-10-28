# Draft report sections — Member 1 (dataset, baseline, experimental setup)

> DRAFT. Every number below comes from a run in this repository (`results/`).
> Items in [BRACKETS] are not done yet and must be filled in or deleted before
> submission. Rewrite in your own words and check every claim.

## 3. Dataset

**Source.** Card metadata, official card images and TCGplayer market prices come
from the Pokémon TCG API (pokemontcg.io). We downloaded 20,670 cards from 176 sets
(released 1999–2026) in October 2026; the price snapshot is dated 15 Oct 2026.

**Target.** `y = log1p(price)`, where price is the TCGplayer market price in USD of the
card's standard printing (normal, else holofoil; vintage "unlimited" prints). Reverse
holo and 1st-edition prices are ignored so each card has exactly one price. 19,486 cards
have such a price. Prices are heavily right-skewed (median $0.95 across all priced cards, up
to $9,200), which motivates the log transform.

**Coverage gap.** The API has no TCGplayer prices for sets released after 14 Nov 2025
(six sets, 2026), so these cards are excluded. The temporal test window therefore ends
14 Nov 2025.

**Metadata features.** Rarity, supertype, subtypes, energy type, HP, artist, set id,
series, regulation mark, card number relative to set size, printed set size, number of
attacks and abilities, retreat cost and set age (years since release).

**Real-world test set (eBay).** [Not yet collected.] Sold listings are collected by
hand from ebay.com's Sold Items filter; we do not scrape eBay because that breaches its
User Agreement, and eBay's Marketplace Insights API is restricted to approved developers.
Cards are chosen in advance across rarities and eras, raw English singles only, USD, a
fixed date window, ≥3 sales per card; the item price (without shipping) is used.
Listings are filtered (graded, lots, fakes, sealed, non-English, damaged, variants),
matched to card ids by number + name + set, per-card outliers removed (a sale more than
2.5× from the card's median), and aggregated to a median. Rejected and unmatched listings
are reviewed by hand. [Report n cards / n sales and the match audit here.]

## 4. Experimental setup

**Splits.** *Temporal (main):* whole sets are assigned by release date, so test sets are
never seen in training and are newer than all training data. Train: 16,995 cards from 147
sets (1999 – Nov 2023); validation: 1,214 cards from 6 sets (Jan – Nov 2024); test: 1,277
cards from 7 sets (Jan – Nov 2025). No set spans two splits (checked by assertion).
*Random (comparison):* cards are shuffled with the same split sizes, ignoring sets.
eBay cards are excluded from training, validation and tuning in both splits.

**Leakage control.** Encoders, scalers and medians are fitted on training rows only;
unseen categories (e.g. every test set id) map to an "unknown" code. Hyperparameters were
fixed in advance and not tuned; the test split and eBay set are only scored.

**Metrics.** MAE on log price (primary), Spearman rank correlation, median absolute
percentage error in USD, and the share of predictions within a factor of 2 of the true
price. Neural/boosted models use 3 seeds (0–2); we report mean ± std over seeds.

## 5. Baselines

Three metadata-only models: (i) rarity-median — the training median log price of the
card's rarity; (ii) HistGradientBoosting (500 iterations max, lr 0.05, early stopping on
10 % of the training rows); (iii) an MLP (128-64, one-hot + standardised inputs, early
stopping). Test results (3 seeds):

| Split | Model | MAE (log) | Spearman | Within 2× |
|---|---|---|---|---|
| Random | Rarity-median | 0.725 | 0.666 | 0.413 |
| Random | HistGB | 0.349 ± 0.004 | 0.936 | 0.643 |
| Random | MLP | 0.364 ± 0.007 | 0.928 | 0.631 |
| Temporal | Rarity-median | 0.412 | 0.784 | 0.410 |
| Temporal | HistGB | 0.415 ± 0.025 | 0.779 | 0.412 |
| Temporal | MLP | 0.523 ± 0.073 | 0.773 | 0.362 |

**Observations (descriptive, not yet explained).**
* On a random split, the boosted model halves the error of the rarity-median
  (0.349 vs 0.725); on the temporal split it does **not** beat it (0.415 vs 0.412).
  Metadata that includes set identity and set age helps a lot when the sets are seen in
  training and gives no advantage on unseen future sets. This is consistent with
  Hypothesis 2 (partial memorisation of set-specific pricing), but we have not run the
  ablation that would show it (e.g. removing set id / age). [Member 3]
* Any gain from images must be measured against the *temporal* rarity-median (0.412), a
  much harder bar than the random-split numbers suggest.
* Price levels shift: the median test-set price is $0.21 against $1.13 in training, so
  absolute-percentage metrics (median error ≈ 100 %) are dominated by cheap cards.
* The MLP is best on validation (MAE 0.385) but worst on test (0.523) with high seed
  variance (std 0.073); with only 7 test sets the temporal test is noisy. We report
  both and do not select on test.

**Limitations.** One snapshot of TCGplayer prices; market price is itself an estimate;
the temporal test covers 7 sets; eBay results pending.

## For Member 2 (handoff)
* Use `data/splits.csv` (`split_temporal`) and `data/cards.csv`; image URLs are in
  `image_url` and images download with `fetch_cards.py --images` to `data/images/<card_id>.png`.
* Baselines to beat on temporal test: rarity-median 0.412, HistGB 0.415 (MAE log).
* Run your models with seeds 0–2 and report with `baselines.metrics()`-style metrics.
