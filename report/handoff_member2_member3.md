# Handoff from Member 1 → Members 2 and 3

Everything here was produced by `pokeprice/` and is reproducible from the README.

## Data (do not regenerate unless told)
* `data/cards.csv` — 20,670 cards, 19,486 with a price. Join key `card_id`.
  Target = `log1p(price_market)` (TCGplayer market price, standard printing).
  Metadata columns: rarity, supertype, subtypes, types, hp, artist, set_id, series,
  number, printed_total, n_attacks, n_abilities, retreat_cost, regulation_mark, release_date.
* `data/splits.csv` — `split_temporal` (MAIN) and `split_random`, values
  `train | val | test | ebay | none`. Use ONLY these columns to pick rows.
* Images: `python pokeprice/fetch_cards.py --images` -> `data/images/<card_id>.png`
  (small official scans; `--image-size large` for large). ~19k files, not in git.
* eBay hold-out: `data/ebay_cards.csv` (when collected). Rows labelled `ebay`.

## Rules
1. `test` and `ebay` rows are never used for training, early stopping, model selection,
   feature selection or hyperparameter tuning. Use `val` (or CV inside `train`).
2. Fit every scaler / encoder / PCA on `train` only.
3. Run 3 seeds (0, 1, 2); report mean ± std.
4. Use the same metrics: `from baselines import metrics` (add `pokeprice/` to `sys.path`)
   -> MAE on log price, Spearman, median abs % error, share within 2x.
5. Report negative results too.

## Numbers to beat (test split, mean of 3 seeds, MAE on log price)

| Split | Rarity-median | HistGB | MLP |
|---|---|---|---|
| Temporal (main) | 0.412 | 0.415 ± 0.025 | 0.523 ± 0.073 |
| Random | 0.725 | 0.349 ± 0.004 | 0.364 ± 0.007 |

Full table: `results/baseline_summary_split_*.csv`; per-card predictions for failure
analysis: `results/baseline_preds_split_*.csv`.

## Things to know
* On the temporal split metadata-only models do NOT beat rarity-median; the gap to beat is small.
  The image model must be compared against the temporal baseline, not the random one.
* Unseen set ids map to an "unknown" code in the baselines. Hypothesis 2 (memorisation of
  set-specific pricing) is untested: ablate `set_id`, `series`, `age_years`, `printed_total`
  (Member 3).
* Test sets (2025) have median price $0.21 vs $1.13 in training: newer prices may not have
  settled. Discuss as a confound.
* Prices only exist up to 14 Nov 2025; 2026 sets have none in the API.
* The MLP baseline is best on val but worst on test; 7 test sets is a noisy sample.
* Credit pretrained weights (CLIP / ResNet), scikit-learn and the Pokémon TCG API in the report.
