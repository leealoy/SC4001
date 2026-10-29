# Project handoff: data, modelling and analysis

## Research questions

1. After controlling for Pokémon identity, do visual features improve price
   prediction beyond structured metadata?
2. Does the visual contribution persist on sets released after every training
   set?
3. Are improvements produced by artwork, or by shortcuts such as card layout,
   text and set-specific design?
4. Can residual gated fusion with modality dropout generalise better than
   ordinary concatenation?

The supervised target is `log1p(price_market)`, where `price_market` is the
TCGplayer market price for one standard printing of a card on 29 October 2026.
This is cross-sectional price prediction, not historical price forecasting.

## Dataset and cohorts

The Pokémon TCG API snapshot contains 20,670 cards, of which 19,486 have a
usable standard-printing market price. The primary structured-data cohort has
16,423 priced Pokémon cards with a National Pokédex identity across 160 sets.
Ninety-eight priced Pokémon cards without an API identity are excluded.

The controlled visual cohort contains 11,509 standard framed cards across 12
reviewed layout families. Special rarities, irregular mechanics, promotional or
unsupported layouts, non-Pokémon cards, cards without usable prices or identity,
and cards without a local image are recorded in the exclusions file rather than
force-fitted to an unreliable artwork crop.

The temporal visual cohort contains 10,253 training cards, 604 validation cards
and 652 test cards. Every visual model and its corresponding metadata baseline
must use this same cohort for a paired comparison.

## Data contract

- `data/cards.csv`: card metadata, target price, National Pokédex identity and
  the safe local `image_file` name; join key `card_id`.
- `data/splits.csv`: fixed temporal and random partitions plus seen/unseen
  identity flags.
- `data/images/<image_file>`: official full-card scans.
- `data/artwork_layout_templates.csv`: reviewed normalised artwork box for each
  layout family.
- `data/artwork_layout_assignments.csv`: eligible visual-cohort cards and their
  assigned families.
- `data/artwork_layout_exclusions.csv`: excluded cards and explicit reasons.
- `data/artwork_layout_audit.csv`: seeded manual-audit sample of 12 cards per
  family.
- `data/image_views/manifest.csv`: 34,527 matched rows covering `full_card`,
  `artwork_only` and `layout_only`, including split, family, crop coordinates,
  image paths and hashes.

The family template is used unless a future audited exception file is supplied
to `prepare_image_views.py` through `--overrides`. No per-card override file is
currently required.

## Member 2: model contract

Compare the following models on identical cards:

1. Identity-plus-metadata baseline.
2. Frozen pretrained image encoder with a regression head.
3. Ordinary image-plus-metadata concatenation.
4. Residual gated fusion without modality dropout.
5. Residual gated fusion with modality dropout.

Run the image and fusion comparisons separately for `full_card`,
`artwork_only` and `layout_only`. Keep the encoder, metadata representation,
training budget and seeds matched when comparing fusion mechanisms. Tune only
on validation data. If a residual method uses a pretrained metadata prediction,
its training predictions must be generated out of fold.

Each model writes prediction rows with:

```text
card_id,set_id,model,seed,eval,y_true_log,y_pred_log
```

Include `pokemon_identity`, `identity_seen_in_train`, `feature_set` and
`image_condition` where applicable. Use seeds 0, 1 and 2.

## Member 3: analysis contract

- Treat the temporal split as the main experiment and the random split as a
  diagnostic.
- Report seen-identity and unseen-identity results.
- Compare full-card, artwork-only and layout-only predictions on identical card
  IDs.
- Run image and metadata modality ablations.
- Report per-set metrics and paired confidence intervals from a whole-set
  bootstrap.
- Audit duplicate or reprinted art using manifest hashes.
- Analyse errors by identity, rarity, set and price range.

Use `pokeprice/evaluate_predictions.py` so all models use the same metrics and
pairing rules.

## Leakage controls

1. Test data cannot be used for tuning, early stopping, crop selection or image
   transformations.
2. Fit encoders, scalers, target encoders and group statistics using training
   rows only.
3. Use the supplied splits; do not create model-specific partitions.
4. Compare models on identical card IDs.
5. Keep active eBay asking prices outside supervised training and evaluation.
