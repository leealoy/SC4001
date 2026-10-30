# Handoff to Member 3: final analysis and reporting

## Project status

Data preparation, identity-controlled baselines, controlled image views,
multimodal training, temporal final evaluation and random-split diagnostics are
complete. The temporal test set has been opened and must not be used for any
further model selection, tuning or architectural changes.

The report-ready Member 1 and Member 2 drafts are:

- `report/member1_sections_draft.md`
- `report/member2_sections_draft.md`

Member 3 now owns the consolidated results, figures, error analysis,
limitations and final discussion.

## Research questions and final answers

### 1. Do images improve prediction beyond identity and metadata?

Only under a restricted condition. On the later-set temporal test, full-card
residual gated fusion achieved mean log-price MAE 0.0770 ± 0.0099 compared with
0.0852 ± 0.0143 for metadata. After averaging seed predictions per card, its
paired MAE difference from metadata was -0.0086 with a whole-set bootstrap 95%
interval of [-0.0183, -0.0026].

Artwork-only fusion did not significantly improve on metadata. Image-only
models were worse than metadata for every image condition.

### 2. Does visual value persist on later unseen sets?

The full-card gated improvement occurs on seven complete test sets released
after every training set, so it supports later-set transfer for full-card cues.
The result is not evidence that artwork itself transfers. Temporal validation
and test differed, and only seven temporal test cards have identities unseen in
training, so both set-level and unseen-identity claims must remain cautious.

### 3. Artwork or shortcuts?

Evidence favours layout, text or generation-specific design shortcuts.
Full-card concatenation beat artwork-only concatenation by -0.0040 paired MAE
with a 95% interval of [-0.0054, -0.0025]. Full-card and layout-only image
models also substantially outperformed artwork-only image models, while
artwork-only fusion failed to beat metadata. Layout-only fusion was competitive
but not significantly better than metadata.

### 4. Does residual gated fusion with modality dropout beat concatenation?

Not in the primary temporal setting. Full-card residual gating produced the
clearest gain against metadata, but its direct advantage over ordinary
concatenation was not conclusive. Adding modality dropout made the full-card
gated model worse by +0.0046 MAE, 95% interval [0.0003, 0.0104].

On the random split, modality dropout significantly improved on concatenation
for all three image conditions, but it did not significantly beat metadata.
Thus modality dropout helps within-set interpolation but does not demonstrate
better chronological domain generalisation.

## Important split interpretation

The temporal visual cohort has 10,253 training, 604 validation and 652 test
cards. Sets are disjoint and chronologically ordered.

The random visual cohort has 10,034 training, 719 validation and 756 test cards.
All 121 random validation sets and all 120 random test sets also occur in random
training; only two random-test cards have unseen identities. Its Spearman
correlations near 0.93 therefore reflect interpolation across familiar sets and
must not replace the temporal result. Absolute MAE should not be compared
directly across splits because their target-price distributions differ.

## Canonical result files

Keep these raw prediction files because they reproduce every final evaluation:

```text
results/multimodal_temporal_artwork_only.csv
results/multimodal_temporal_full_card.csv
results/multimodal_temporal_layout_only.csv
results/multimodal_random_artwork_only.csv
results/multimodal_random_full_card.csv
results/multimodal_random_layout_only.csv
```

Use these temporal evaluation files:

```text
results/multimodal_temporal_final_summary.csv
results/multimodal_temporal_final_runs.csv
results/multimodal_temporal_final_paired.csv
results/multimodal_temporal_final_per_set.csv
results/multimodal_temporal_final_by_identity.csv
```

Use these random diagnostic files:

```text
results/multimodal_random_final_summary.csv
results/multimodal_random_final_runs.csv
results/multimodal_random_final_paired.csv
results/multimodal_random_final_per_set.csv
results/multimodal_random_final_by_identity.csv
```

The retained top-level baseline prediction and evaluation files are Member 1's
canonical structured-data results. Redundant direct baseline summaries and
superseded validation-only multimodal evaluations are stored under
`results/archive/`; they must not be presented as final results.

## Evaluation conventions

- Summary tables report mean ± standard deviation across seeds 0, 1 and 2.
- Paired files first average seed predictions per card, then bootstrap complete
  sets. A negative `mae_delta` means the candidate is better than the named
  reference.
- The primary metric is MAE on `log1p(price_market)`.
- Spearman, median absolute percentage error in dollars and share within a
  factor of two are secondary metrics.
- Compare models only on identical card IDs.

## Member 3 remaining work

1. Produce a compact temporal table with metadata, image-only, concatenation,
   gated and gated-plus-dropout results for all three image conditions.
2. Plot paired temporal MAE differences and 95% whole-set bootstrap intervals.
3. Produce a smaller temporal-versus-random diagnostic figure; explain set
   overlap rather than comparing raw MAE as though cohorts were identical.
4. Analyse temporal errors by set, identity, rarity and target-price range.
5. Select representative over- and under-predicted cards without using them to
   alter the model.
6. Audit repeated artwork using `difference_hash` in
   `data/image_views/manifest.csv` and discuss any sensitivity.
7. Report the small unseen-identity sample as a limitation rather than a strong
   subgroup conclusion.
8. Integrate the Member 1 and Member 2 drafts, remove duplication, and keep the
   main report within the page limit.

## Reproduction commands

No model retraining is required for report production. Recreate the canonical
evaluation tables from the saved predictions with:

```powershell
.\.venv\Scripts\python.exe pokeprice/evaluate_predictions.py results/multimodal_temporal_artwork_only.csv results/multimodal_temporal_full_card.csv results/multimodal_temporal_layout_only.csv --reference torch_metadata_identity_metadata --out results/multimodal_temporal_final
```

```powershell
.\.venv\Scripts\python.exe pokeprice/evaluate_predictions.py results/multimodal_random_artwork_only.csv results/multimodal_random_full_card.csv results/multimodal_random_layout_only.csv --reference torch_metadata_identity_metadata --out results/multimodal_random_final
```
