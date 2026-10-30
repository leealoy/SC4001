# Archived result files

`validation_intermediate/` contains superseded validation-only multimodal
evaluations produced before the canonical temporal final evaluation.

The archived groups are:

- `multimodal_temporal_artwork_validation_*`
- `multimodal_temporal_artwork_vs_concat_*`
- `multimodal_temporal_validation_*`

They are retained for provenance but should not be used in the final report.
The canonical temporal and random files remain directly under `results/` with
the prefixes `multimodal_temporal_final_*` and `multimodal_random_final_*`.
Raw condition-level prediction files also remain at the top level because the
canonical evaluation tables can be regenerated from them.

`baseline_direct_summaries/` contains the four direct `baselines.py` run and
summary tables. They are redundant with the retained `baseline_preds_*` files
and canonical `baseline_evaluation_*` outputs, so they are kept only for
provenance.
