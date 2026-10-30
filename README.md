# Artwork or shortcut? Identity-controlled Pokémon card price prediction across unseen sets

SC4001 group project. The supervised target is a dated snapshot of TCGplayer
market prices. Live eBay listings, when collected, are unsold **asking prices**
and are kept as a separate exploratory dataset.

## Research questions

1. After controlling for Pokémon identity and structured metadata, do visual
   features improve price prediction?
2. Does the visual contribution persist on sets released after every training
   set?
3. Is any improvement produced by artwork, or by card layout, text and
   set-specific design shortcuts?
4. Does residual gated fusion with modality dropout generalise better than
   ordinary concatenation?

Questions 1–3 are the empirical questions. Question 4 is the proposed-method
hypothesis and should be evaluated under the same splits and tuning budget.

## Setup

Python 3.11:

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Using the project virtual environment avoids conflicts with system-wide machine
learning packages. Run later commands with `.\.venv\Scripts\python.exe` in
place of `python` if the environment has not been activated. The NumPy and
PyTorch versions are intentionally bounded because PyTorch 2.3 Windows wheels
are not compatible with NumPy 2.x.

## 1. Download the card data

```bash
python pokeprice/fetch_cards.py
python pokeprice/fetch_cards.py --images --image-size large
```

The Pokémon TCG API supplies metadata, official scans and TCGplayer prices.
`cards.csv` includes both the printed name and `national_pokedex_numbers` /
`pokemon_identity`. Its `image_file` column gives the cross-platform-safe local
filename; reserved characters in unusual card IDs are percent-encoded. National
Pokédex identity is used because printed names such
as Charizard, Charizard ex and Charizard VMAX should not be treated as unrelated
characters when controlling for stable popularity.

The target is `log1p(price_market)` for one standard printing per card. Prices
are a single dated snapshot, not a price history or forecasting dataset.

The downloader caches raw API pages and is resumable. Set
`POKEMONTCG_API_KEY` to use an optional free API key. Do not commit credentials.

## 2. Create the fixed splits

```bash
python pokeprice/make_splits.py --test-months 12 --val-months 12
```

`data/splits.csv` contains:

- `split_temporal`: main experiment; entire later-released sets are held out.
- `split_random`: diagnostic comparison with the same split sizes.
- `identity_seen_in_temporal_train` and `identity_seen_in_random_train`: used
  for predeclared seen-versus-unseen identity analysis.

The script asserts that no set crosses temporal partitions and that release
dates are ordered. This is a later-set generalisation test, not prospective
future-price forecasting: all labels come from the same price snapshot.

## 3. Run identity-controlled metadata baselines

```bash
python pokeprice/baselines.py --split split_temporal
python pokeprice/baselines.py --split split_random
```

The primary cohort is Pokémon cards with a National Pokédex identity. Use
`--cohort all` only as a sensitivity analysis. The baseline feature sets are:

- `identity`: Pokémon identity only.
- `metadata`: structured metadata without identity.
- `identity_metadata`: identity plus structured metadata; this is the baseline
  that image and fusion models must beat for research question 1.

Models are rarity median, identity median, HistGradientBoosting and an MLP.
Tree categorical features use cross-fitted target encoding; neural categorical
features use training-fitted one-hot encoding. Unknown validation/test identities
fall back without using validation/test targets.

Outputs include `set_id`, `pokemon_identity` and `identity_seen_in_train` for
paired and stratified analysis. Canonical files have a cohort suffix, for
example `results/baseline_preds_split_temporal_pokemon.csv`.

## 4. Prepare controlled image views

A full card scan contains artwork, borders, text, rarity symbols and layout.
Consequently, a full-card model alone cannot establish that artwork helped.

First assign ordinary framed cards to generation-aware layout families:

```bash
python pokeprice/make_artwork_layouts.py
```

This writes the eligible assignments, explicit exclusions and a seeded audit
sample. Edit `data/artwork_layout_templates.csv`: inspect the listed example
card for each family, enter its normalised artwork box, and change `reviewed` to
`true`. The already reviewed XY template came from `xy5-1`. Full-art, special
mechanic, promotional and unsupported layouts are excluded rather than assigned
an unreliable crop.

No per-card override file is needed for the current reviewed templates. If an
audited exception is found later, pass an override CSV to
`prepare_image_views.py` with `--overrides PATH`.

Pilot the seeded audit cards before generating the complete cohort:

```bash
python pokeprice/prepare_image_views.py --card-list data/artwork_layout_audit.csv
python pokeprice/prepare_image_views.py --require-all-templates
```

The second command should be run only after every assigned family template has
been reviewed. It produces a manifest and three matched conditions:

```text
full_card
artwork_only
layout_only
```

The reusable family boxes make the experiment scalable while retaining manual
quality control through `data/artwork_layout_audit.csv`. The manifest records
the family, template example, optional override, coordinates, split and image
hashes. The full-card condition references the existing official image instead
of copying it; only artwork-only and masked-layout images are materialised.
Models must use the manifest rather than scanning output directories.

The relevant files are:

```text
data/artwork_layout_templates.csv
data/artwork_layout_assignments.csv
data/artwork_layout_exclusions.csv
data/artwork_layout_audit.csv
```

To regenerate after changing a template:

```bash
python pokeprice/prepare_image_views.py
```

## 5. Extract frozen image embeddings

Install the PyTorch dependencies in `requirements.txt`, then extract one frozen
ImageNet ResNet-50 embedding for each matched image condition:

```bash
python pokeprice/extract_image_embeddings.py --encoder resnet50
```

The first run may download the official torchvision weights. The extractor
pads portrait cards to a square before resizing, keeps the full card visible,
and writes float16 feature files under `data/features/resnet50/`. Existing
feature files are not replaced unless `--overwrite` is supplied.

## 6. Train Member 2 models

During model selection, run validation only (the default). Start with the
artwork condition and all five model families:

```bash
python pokeprice/train_multimodal.py --condition artwork_only
```

This trains the cohort-matched identity-plus-metadata model, image-only model,
ordinary concatenation, residual gated fusion, and residual gated fusion with
modality dropout for seeds 0, 1 and 2. Checkpoints selected by validation MAE
are written after every improvement.

Run the shortcut conditions without duplicating the metadata prediction rows:

```bash
python pokeprice/train_multimodal.py --condition full_card --models image concat gated gated_dropout
python pokeprice/train_multimodal.py --condition layout_only --models image concat gated gated_dropout
```

Tune architecture and optimisation settings using validation outputs only. Once
the settings and hypotheses are frozen, score the untouched test rows from the
validation-selected checkpoints with
`--load-checkpoints --include-test --overwrite`. The random split is a secondary
diagnostic and is selected with `--split split_random`.

The residual models learn the metadata base prediction and gated image
correction jointly. They do not consume in-sample predictions from a separately
pretrained metadata model. With modality dropout, dropped-image examples reduce
exactly to the metadata branch.

## 7. Shared evaluation contract

Every metadata, image and fusion model should write:

```text
card_id,set_id,model,seed,eval,y_true_log,y_pred_log
```

Recommended additional fields are `pokemon_identity`,
`identity_seen_in_train`, `image_condition` and `feature_set`.

Evaluate one or more prediction files with:

```bash
python pokeprice/evaluate_predictions.py results/predictions.csv --reference hgb_identity_metadata --out results/main_evaluation
```

The evaluator produces overall, per-set and seen/unseen-identity metrics. With a
reference model, it also produces paired MAE differences and confidence intervals
by bootstrapping whole sets. The shared metrics are log-price MAE, Spearman rank
correlation, median absolute percentage error and the share within 2×.

For the three final temporal prediction files, use the cohort-matched metadata
model as the paired reference:

```bash
python pokeprice/evaluate_predictions.py results/multimodal_temporal_artwork_only.csv results/multimodal_temporal_full_card.csv results/multimodal_temporal_layout_only.csv --reference torch_metadata_identity_metadata --out results/multimodal_temporal_evaluation
```

## 8. Optional live eBay asking-price snapshots

The eBay Browse API does not provide realised sale prices. Active listings must
not be mixed with TCGplayer labels or described as sold prices. To capture a
timestamped exploratory snapshot for preselected card IDs:

```bash
python pokeprice/select_ebay_cards.py
python pokeprice/collect_ebay_live.py
```

Credentials belong in the git-ignored `.env`; see `.env.example`. Each output
records retrieval time, listing ID, query, price, shipping, sale format,
condition, match/rejection decision and whether a cached response was used.

The older `build_ebay.py` and `parse_ebay_paste.py` scripts are retained only as
legacy work. They are not part of the revised reproducible experiment unless the
group later obtains an authorised, auditable sold-price source.

## Experimental rules

- Test rows are never used for tuning, feature selection, early stopping or
  choosing image transformations.
- Fit encoders, scalers, target encoders and identity medians on training data.
- Use validation data for model and hyperparameter selection.
- Run seeds 0, 1 and 2 for stochastic models.
- Compare models on identical cards and report per-set results.
- Report exclusions, missing identities, missing images and negative results.
- Do not claim that identity measures time-varying hype; it controls stable
  between-character differences.

## Layout

```text
pokeprice/   collection, splitting, baselines, image views and evaluation
data/        source tables, fixed splits and optional ignored image/cache data
results/     prediction-level and aggregate results
report/      report drafts and member handoff notes
```
