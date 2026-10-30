# Results directory

The top level contains only canonical training predictions and final evaluation
outputs.

## Structured-data baselines

- `baseline_preds_*`: card-level predictions needed to reproduce evaluation.
- `baseline_evaluation_*`: canonical common-evaluator outputs.

Redundant direct `baselines.py` run and summary tables are archived under
`archive/baseline_direct_summaries/`.

## Multimodal raw predictions

The six `multimodal_{temporal,random}_{artwork_only,full_card,layout_only}.csv`
files contain the card-level predictions for all seeds. Keep them because the
final evaluation outputs can be regenerated from these files.

## Multimodal final evaluations

- `multimodal_temporal_final_*`: primary later-set evaluation.
- `multimodal_random_final_*`: random within-set diagnostic.

For each prefix, `summary` reports mean and standard deviation across seeds,
`runs` contains individual seed metrics, `paired` contains whole-set bootstrap
MAE comparisons against metadata, `per_set` contains set-level metrics, and
`by_identity` contains seen/unseen-identity metrics.

## Archive

Superseded validation-only evaluations are retained under
`archive/validation_intermediate/` for provenance. They are not final report
inputs.
