# Member 1 report sections

## Research framing

This project investigates whether card images contribute predictive information
about Pokémon-card prices beyond structured metadata and Pokémon identity. It
also tests whether visual gains persist on chronologically later sets and
whether a model learns artwork or shortcuts from text, borders and set-specific
layout. The methodological comparison evaluates ordinary feature concatenation
against residual gated fusion with modality dropout.

## Dataset

### Source and target

Card metadata, official scans and TCGplayer market prices were collected from
the Pokémon TCG API on 29 October 2026. The snapshot contains 20,670 cards;
19,486 have a usable market price for a standard printing. Where available, the
standard-price selection covers normal, holofoil, unlimited and unlimited
holofoil variants. The regression target is

```text
y = log(1 + price_market).
```

The logarithm reduces the influence of the strongly right-skewed price
distribution. Because the target is a single dated snapshot, the task is
cross-sectional price prediction rather than price forecasting.

### Primary cohort and identity control

The primary structured-data cohort contains 16,423 priced Pokémon cards from
160 sets with a usable National Pokédex identity. Trainer and Energy cards are
outside the primary cohort because Pokémon identity is undefined for them.
Ninety-eight priced Pokémon cards without an API Pokédex identity are also
excluded.

Identity is derived from sorted National Pokédex numbers rather than the printed
card name. Variants such as Charizard, Charizard ex and Charizard VMAX therefore
share a species identity while mechanics and subtypes remain structured card
attributes. Multi-Pokémon cards retain their tuple of Pokédex numbers. This
controls stable between-character differences, but it is not a direct measure
of time-varying popularity or temporary hype.

### Structured metadata

The structured predictors are rarity, supertype, subtypes, energy type, HP,
artist, set ID, series, regulation mark, card number relative to printed set
size, printed set size, number of attacks and abilities, retreat cost, and age
at the snapshot date. Categorical encoders, numerical imputers and scalers are
fit on training rows only.

### Controlled visual cohort

A complete card scan contains artwork as well as text, borders, rarity symbols
and set-specific design. To separate these signals, standard framed Pokémon
cards are assigned to 12 generation-aware layout families. Each family has a
manually reviewed normalised artwork box, and a seeded sample of 12 cards per
family is retained for the recorded quality audit. Special rarities, irregular
mechanics, promotions and unsupported layouts are explicitly excluded from this
rectangular-crop experiment.

The resulting visual cohort has 11,509 cards. Each card supplies three matched
conditions:

- `full_card`: the complete official scan;
- `artwork_only`: the cropped illustration resized to 224 × 224 pixels; and
- `layout_only`: the complete card with the artwork region replaced by grey.

The visual manifest contains 34,527 rows and records the card, layout family,
split, crop coordinates, image path and hashes. Layout-family coverage is:

| Layout family | Cards |
|---|---:|
| Black & White standard | 1,060 |
| Diamond & Pearl / Platinum standard | 1,124 |
| E-Card standard | 411 |
| EX-era standard | 1,282 |
| HeartGold & SoulSilver standard | 390 |
| Mega Evolution standard | 176 |
| Sun & Moon standard | 1,596 |
| Scarlet & Violet standard | 1,827 |
| Sword & Shield standard | 1,720 |
| WotC Base / Gym standard | 541 |
| WotC Neo standard | 301 |
| XY standard | 1,081 |

## Experimental setup

### Splits

The main temporal split assigns complete sets by release date. Training contains
sets released from 9 January 1999 through 3 November 2023, validation contains
sets from 26 January through 8 November 2024, and test contains sets from
17 January through 14 November 2025. Thus every test set is later than every
training set.

| Cohort | Train | Validation | Test |
|---|---:|---:|---:|
| Primary identity-controlled cohort | 14,402 | 953 | 1,068 |
| Matched visual cohort | 10,253 | 604 | 652 |

The primary temporal partitions contain 147 training sets, six validation sets
and seven test sets. Five validation cards and nine test cards have identities
not observed in temporal training. A seeded random split with comparable sizes
is used only to diagnose the effect of chronological domain shift; it is not the
main generalisation claim.

### Baselines

The structured-data baselines are:

1. training-set rarity median;
2. training-set identity median with a global-median fallback;
3. HistGradientBoosting and multilayer perceptron models using identity only;
4. the same model families using metadata without identity; and
5. the same model families using identity plus metadata.

The boosted models use five-fold cross-fitted target encoding for categorical
training features. The MLP uses an exact one-hot identity representation,
frequency-aware one-hot encoding for other categorical variables, median
imputation and standardisation for numerical variables. Stochastic models use
seeds 0, 1 and 2.

On the primary temporal test cohort, the mean log-price MAE was 0.659 ± 0.014
for HistGradientBoosting with identity and metadata, and 0.522 ± 0.030 for the
identity-plus-metadata MLP. The rarity-median reference achieved 0.417. These
figures describe the full primary cohort; the multimodal experiment must train
and compare its metadata baseline on the same 11,509-card visual cohort.

### Evaluation

The primary metric is MAE on log price. Secondary metrics are Spearman rank
correlation, median absolute percentage error after conversion to dollars, and
the proportion of predictions within a factor of two of the target price.
Results are reported overall, by set and by whether the Pokémon identity was
seen during training. Paired model differences use identical cards and
confidence intervals from bootstrapping complete sets.

All preprocessing is fitted on training data. Validation data are used for
model selection and early stopping. Test rows are not used to select image
transformations, layout templates, features or hyperparameters.

## Limitations

- The target is one dated TCGplayer price snapshot and does not describe price
  dynamics.
- National Pokédex identity controls stable character differences but does not
  directly measure contemporary popularity.
- Chronologically later sets test later-set generalisation, not future prices
  for existing cards.
- The rectangular artwork analysis represents standard framed layouts and does
  not cover every promotional, full-art or special-mechanic design.
- Active eBay listings are asking prices that may never result in sales; they
  remain outside supervised modelling and evaluation.
