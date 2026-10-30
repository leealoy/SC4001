# Member 2 report sections

## Introduction and motivation

Pokémon-card prices may depend on structured properties such as rarity, set,
artist and Pokémon identity, but collectors also respond to visual presentation.
A full card image combines several possible signals: the illustration itself,
the surrounding layout and text, rarity symbols, borders, and generation- or
set-specific design. Consequently, an improvement from a full-card model would
not by itself show that artwork matters.

This project asks four connected questions:

1. After controlling for Pokémon identity, do visual features improve price
   prediction beyond structured metadata?
2. Does any visual contribution persist when all test sets were released after
   the training sets?
3. Is apparent visual value attributable to artwork, or to shortcuts such as
   layout, text and set design?
4. Does residual gated fusion with modality dropout generalise better than
   ordinary feature concatenation?

The primary contribution is therefore a controlled multimodal experiment rather
than a higher prediction score alone. The same cards are presented under three
matched visual conditions, and all models are evaluated on both a chronological
whole-set split and a random diagnostic split.

## Related work

### Pretrained visual representation

Residual networks learn residual functions around identity mappings, making
deep convolutional networks easier to optimise. ResNet-50 provides a mature
ImageNet-pretrained representation and is used here as a frozen encoder rather
than fine-tuned on price labels. Freezing the encoder reduces computational
cost and limits the number of trainable parameters relative to the 11,509-card
visual cohort [He et al., 2016].

### Multimodal fusion and gating

Simple concatenation treats metadata and image embeddings as equally available
inputs to a downstream predictor. Gated multimodal models instead learn
input-dependent multiplicative weights that control how much each modality
influences the fused representation [Arevalo et al., 2017]. Our method adapts
this idea to regression by retaining a metadata prediction as a base estimate
and learning a gated visual residual.

### Modality dropout

ModDrop randomly removes complete modalities during training to discourage
co-adaptation and improve robustness when a modality is missing or unreliable
[Neverova et al., 2016]. We apply dropout to the complete image modality rather
than independently dropping image-embedding dimensions. A dropped-image sample
reduces exactly to the metadata branch.

### Shortcut learning

Neural networks can exploit predictive cues that work within a benchmark but
fail under distribution shift. This behaviour is commonly described as
shortcut learning [Geirhos et al., 2020]. Card borders, text and generation-
specific templates are plausible shortcuts because they correlate with set and
release period. The artwork-only and layout-only conditions make this concern a
testable hypothesis.

## Proposed method

### Frozen image encoder

Official card scans are read through the controlled image manifest. Portrait
images are padded to a square with neutral grey before resizing to 224 × 224;
this preserves the complete card instead of discarding its top and bottom with
a centre crop. A ResNet-50 pretrained on ImageNet is frozen and its final
classification layer is replaced by the identity function, producing a
2,048-dimensional embedding. Embeddings are extracted once and stored as
float16 arrays. Before regression, every embedding dimension is standardised
using the temporal or random training partition only.

The three matched conditions are:

- **Full card:** the complete official scan.
- **Artwork only:** the illustration crop defined by the card's reviewed layout
  family.
- **Layout only:** the complete scan with the illustration region replaced by
  grey.

### Metadata branch

The metadata branch includes National Pokédex identity, rarity, supertype,
subtypes, energy types, artist, set ID, series and regulation mark as
categorical inputs. Numerical inputs are HP, card number relative to printed set
size, printed set size, number of attacks and abilities, retreat cost and age at
the price snapshot.

Categorical vocabularies, numeric medians, means and standard deviations are
fitted on training rows only. Index zero is reserved for unseen categories and
has a fixed zero embedding. This is essential for chronological testing because
validation and test set identifiers do not occur in training. The concatenated
categorical embeddings and numerical features pass through two 128-unit ReLU
layers with LayerNorm and dropout. A linear head produces the metadata estimate
\(\hat{y}_m\).

### Image-only and concatenation models

The image-only model passes the standardised ResNet vector through two 128-unit
ReLU layers and a linear regression head. Ordinary fusion concatenates the
128-dimensional metadata and image representations and applies another
128-unit hidden layer before regression:

\[
\hat{y}_{concat}=F([h_m;h_i]).
\]

### Residual gated fusion

The proposed fusion model treats the metadata estimate as a stable base and
lets the image branch learn a correction. For metadata representation \(h_m\),
image representation \(h_i\), and image-availability indicator \(a\), define

\[
z=[h_m;h_i;a], \qquad
g=\sigma(G(z)), \qquad
\Delta=D(z),
\]

where \(G\) and \(D\) are learned multilayer perceptrons and \(g\) is a scalar
gate. The prediction is

\[
\hat{y}=\hat{y}_m+a\,g\,\Delta.
\]

The residual model is trained jointly; it does not use an in-sample prediction
from a separately pretrained metadata regressor. This avoids constructing a
leaky residual target.

For the modality-dropout variant, the complete image embedding is set to zero
with probability 0.25 during training and \(a=0\). The correction is then
forced to zero, so the sample reduces exactly to the metadata prediction. At
evaluation, the image is always available and \(a=1\).

## Training protocol

All model families use the same controlled visual cohort and supplied splits.
The target is \(\log(1+\text{TCGplayer market price})\). Models minimise Huber
loss with delta 0.5 using AdamW, learning rate \(10^{-3}\), weight decay
\(10^{-4}\), batch size 256 and network dropout 0.2. Training runs for at most
200 epochs with validation-MAE early stopping after 20 epochs without
improvement. Seeds 0, 1 and 2 quantify optimisation variability.

The temporal split is primary: 10,253 training cards, 604 validation cards and
652 test cards. Its test partition contains seven complete sets released after
every training set. Hyperparameters and checkpoints were selected using
validation only; final temporal test predictions were generated once from the
saved validation-selected checkpoints.

The random split is diagnostic and contains 10,034 training, 719 validation and
756 test cards. Every random validation and test set is also represented in
training, so it measures within-set interpolation rather than later-set
generalisation.

Primary evaluation uses MAE on log price. Secondary metrics are Spearman rank
correlation, median absolute percentage error in dollars and the fraction within
a factor of two. Paired MAE differences average the three seed predictions for
each card and bootstrap complete sets, preserving the cluster structure of the
temporal experiment.

## Results relevant to the proposed method

### Temporal later-set test

| Model and image condition | MAE, mean ± SD | Spearman, mean ± SD |
|---|---:|---:|
| Metadata only | 0.0852 ± 0.0143 | 0.2157 ± 0.0541 |
| Artwork-only concatenation | 0.0826 ± 0.0199 | 0.1777 ± 0.0243 |
| Full-card concatenation | 0.0770 ± 0.0143 | 0.2062 ± 0.0190 |
| Layout-only concatenation | 0.0751 ± 0.0025 | 0.3190 ± 0.0324 |
| Full-card residual gated | 0.0770 ± 0.0099 | 0.2085 ± 0.0096 |
| Full-card gated + modality dropout | 0.0814 ± 0.0146 | 0.2356 ± 0.0147 |

Against the metadata ensemble, full-card residual gated fusion reduced MAE by
0.0086 with a whole-set bootstrap 95% interval of [-0.0183, -0.0026]. This was
the only clearly supported temporal improvement. Full-card concatenation and
layout-only concatenation had favourable mean scores, but their paired
intervals against metadata crossed zero.

Artwork-only fusion did not significantly improve on metadata. Full-card
concatenation performed significantly better than artwork-only concatenation
(paired MAE difference -0.0040, 95% interval [-0.0054, -0.0025]). Image-only
full-card and layout-only models also substantially outperformed the artwork-
only image model, although all image-only models remained worse than metadata.
These comparisons are more consistent with layout, text or set-design cues than
with a distinct artwork-price signal.

Modality dropout did not improve unseen-set generalisation. For the full-card
residual model it was worse than residual gating without modality dropout by
0.0046 MAE, with a 95% interval of [0.0003, 0.0104]. Residual gating achieved
the clearest improvement against metadata, but its direct advantage over
ordinary full-card concatenation was not statistically conclusive.

### Random-split diagnostic

The random split produced Spearman correlations near 0.93 for metadata and
fusion models because cards from every validation and test set were also present
in training. Layout-only gated fusion with modality dropout had the lowest
per-seed mean MAE (0.1911 ± 0.0017 versus 0.2019 ± 0.0080 for metadata), but its
paired ensemble difference from metadata was not significant: -0.0021 with a
95% interval of [-0.0108, 0.0068].

Modality dropout did outperform ordinary concatenation within the random split
for artwork-only, full-card and layout-only inputs. These gains did not persist
under chronological whole-set shift. The methodological hypothesis is therefore
not supported for the primary generalisation setting: modality dropout helped
within-distribution fusion but did not improve unseen-set transfer.

## Interpretation and limitations

The experiment does not support the claim that artwork independently improves
price prediction after identity and metadata control. The strongest temporal
result requires the complete card, while removing artwork often retains much of
the useful signal. The likely explanation is that the encoder exploits layout,
text and generation-specific design. This conclusion should remain cautious:
only seven temporal test sets were available, and validation and test periods
did not behave identically.

Only seven temporal visual-cohort test cards have Pokémon identities unseen in
training, so the study cannot make a strong identity-level extrapolation claim.
The image encoder is frozen and pretrained on natural images; fine-tuning or an
OCR-aware encoder could behave differently. Finally, the target is a single
dated market-price snapshot and does not measure price dynamics.

## References

- Arevalo, J., Solorio, T., Montes-y-Gómez, M., and González, F. A. (2017).
  *Gated Multimodal Units for Information Fusion*.
  https://arxiv.org/abs/1702.01992
- Geirhos, R., Jacobsen, J.-H., Michaelis, C., et al. (2020). *Shortcut
  learning in deep neural networks*. Nature Machine Intelligence, 2, 665-673.
  https://doi.org/10.1038/s42256-020-00257-z
- He, K., Zhang, X., Ren, S., and Sun, J. (2016). *Deep Residual Learning for
  Image Recognition*. CVPR, 770-778.
  https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
- Neverova, N., Wolf, C., Taylor, G. W., and Nebout, F. (2016). *ModDrop:
  Adaptive Multi-Modal Gesture Recognition*. IEEE Transactions on Pattern
  Analysis and Machine Intelligence, 38(8), 1692-1706.
  https://doi.org/10.1109/TPAMI.2015.2461544
