# AI assistance log

This file records material AI assistance for the project AI Use Statement. It
focuses on research design and implementation contributions rather than routine
debugging history.

## Claude Code, 25-28 October 2026

Claude Code (Sonnet 5.5 and Opus 5.5) created the initial repository from the
team's written specification. Its main contributions were:

- Pokémon TCG API collection, local caching and standard-printing price
  selection;
- temporal and random dataset splits;
- structured-metadata price baselines;
- the shared README, dependencies and draft Member 1 report material;
- utilities for selecting and reviewing possible eBay comparison cards; and
- a local dashboard for inspecting card and listing data.

The team determined the topic, data sources, target, experimental requirements
and division of work. The team also ran the collection and baseline programs and
checked their outputs.

## OpenAI Codex, 28-29 October 2026

The final project asks whether visual information improves Pokémon-card price
prediction after controlling for Pokémon identity, whether any gain persists on
later unseen sets, and whether apparent visual gains come from artwork or from
layout and text shortcuts. A fourth methodological comparison tests residual
gated fusion with modality dropout against ordinary concatenation.

Codex adapted the repository to this experimental design by:

- representing Pokémon identity with National Pokédex numbers and recording
  whether each validation or test identity occurred in training;
- defining identity-only, metadata-only and identity-plus-metadata baselines;
- treating the set-chronological split as the main later-set generalisation
  experiment and the random split as a diagnostic;
- adding a common evaluator for overall, per-set, identity-stratified and paired
  whole-set-bootstrap comparisons;
- separating active eBay asking prices from the supervised TCGplayer target;
- defining generation-aware layout families for standard framed cards;
- generating matched `full_card`, `artwork_only` and `layout_only` conditions
  from manually reviewed family templates; and
- documenting the model-output contract and leakage controls for the image and
  fusion experiments.

The team manually reviewed the layout templates and executed the full image-view
pipeline. It produced 11,509 matched cards across 12 layout families, or 34,527
manifest rows across the three image conditions.

## Data-source boundary

The eBay Browse API exposes active listings rather than realised historical sale
prices. Active listing prices are therefore optional exploratory data only and
are not labels, training data or substitutes for TCGplayer market prices. The
supervised target is the dated TCGplayer market-price snapshot.
