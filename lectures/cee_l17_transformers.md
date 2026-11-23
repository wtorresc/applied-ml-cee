---
title: Tokens, Attention, and Transformers
subtitle: "Lecture 17 · Can a transformer screen the crack tiles better?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 2–15

On your own, no notes, two minutes:

1. Lecture 2: 5-NN predicts a mix's strength from its 5 nearest tested mixes. What weight does
   each of them get, and what weight does every other mix get?
2. Lecture 6: softmax turns scores (2, 1, 0) into probabilities. Roughly what are they?
3. Lecture 15: a 3 × 3 filter sees 9 pixels. How many pixels could one layer see at most?

**Today:** every token looks at every other, with learned weights.

::: notes
Give the answers before moving on.
1. One fifth each; every other mix gets exactly zero. Today's attention gives every tested mix a
   weight, larger for the more similar, and the weights still sum to one.
2. e² : e : 1 over their sum: 0.67, 0.24, 0.09. Attention uses the same softmax on similarity
   scores.
3. All of them. A convolution sees a window and needs many layers to see the whole tile
   (Lecture 15's receptive fields); one attention layer lets every patch draw on every other.
:::

## The screen again: a better reader?

:::::: {.columns}
::: {.column}
- Lecture 15: frozen ResNet-18 + a probe.
- It flags **86%** of cracks; the inspector reviews **51%** of tiles.
- Today: the same probe on a transformer.
:::
::: {.column}
| What changes | Cost |
|:-------------|:-----|
| A crack missed | unrecorded until the next inspection |
| A tile reviewed | seconds of the inspector's time |
| A bigger model | slower screening, more energy |
:::
::::::

::: notes
The question from Lecture 15, asked of a newer architecture: on the same 230 photographs, the
same photo split, the same probe (logistic regression, C chosen on validation), and the same
threshold rule (flag 90% of the validation photos' cracked tiles), does a transformer's view
of a tile leave the inspector fewer tiles to review? Before images, the lecture builds the
machinery on text and on Lecture 2's mixes, as CS 441 does: tokens, attention, transformers.
:::

## Your turn: which two reports are alike?

- **A.** An employee was working from a ladder while installing roof trusses on a residential structure when they fell 9 feet off the ladder, resulting in neck, back, and leg injuries.
- **B.** An employee was dismantling scaffolding. As he knelt down to lower the frame, he was pulled to the ground by the weight of the frame, resulting in a laceration to the side of his head.
- **C.** An employee was setting up an extension ladder to access the roof of a house when the ladder contacted a power line, resulting in an electric shock and hospitalization.

::: notes
Three reports from OSHA's Severe Injury Reports, construction, 2023-2025, chosen by a rule in
the figure script (A: a fall from a ladder; B: a fall from a scaffold sharing the fewest words
with A; C: not a fall, sharing the most words with A). Ask: which pair is most alike, and why?
Write it down. A safety engineer coding the event says A and B: both are falls to a lower
level, the event a report is coded by. Counting shared words says A and C. The answer comes
back in the break-it step.
:::

## Text becomes tokens

| One report, median of 2,000 (2023–2025) | Tokens |
|:-----------------------------------------|-------:|
| Characters | 196.5 |
| Words and punctuation marks | 38 |
| Subwords learned from the reports (8,000) | 39 |
| BERT's subwords (30,522, books and Wikipedia) | 41 |

::: notes
A model reads integers, so text is cut into tokens first. Characters: a tiny vocabulary, every
text covered, but long sequences and little meaning per token. Words: short sequences, but the
vocabulary grows without end and a new word has no integer (15,041 different words in the
training years' 79,307 reports; in 2023–2025, 0.2% of words are new to it). Subwords sit in
between: frequent words stay whole, rare ones split into pieces that exist. Data: OSHA Severe
Injury Reports, 105,996 reports from January 2015 to November 2025 (3,683,085 words), federal
jurisdiction, public U.S. government data; training years 2015–2022, test 2023–2025.
:::

## Byte pairs: merge the most frequent

| Merges 1–4 | Merges 5–8 | Merges 9–12 |
|:-----------|:-----------|:------------|
| 1. h + e | 5. in + g | 9. m + p |
| 2. i + n | 6. l + o | 10. e + d |
| 3. a + n | 7. e + r | 11. e + mp |
| 4. t + he | 8. e + e | 12. a + s |

::: notes
Byte-pair encoding (Sennrich et al., 2016): start from characters; count every adjacent pair
of symbols over the corpus (never across a space); fuse the most frequent pair into a new
symbol; repeat until the vocabulary is the size you want. "the" becomes one token after two
merges; "ing" by merge 5. Written out in 30 lines in the figure script, our first
50 merges are identical to Hugging Face's `tokenizers` trained on the same 79,307 reports. WordPiece (BERT's
tokenizer; Wu et al., 2016) picks merges by likelihood instead of count; the idea is the same.
:::

## Your turn: two merges by hand

"the worker held the ladder there, then the ladder fell"

Count the adjacent letter pairs inside the words. Which pair merges first? Then which?

::: notes
Answer: h + e first (6 times; next t + h, 5); then t + he (5 times, ahead of e + r
with 3). After the first merge, t + h no longer exists where "he" was fused, which is why
the counts change. The same arithmetic, over 8 years of reports, built the table before.
:::

## BERT's vocabulary splits the job site

| Word | BERT (books, Wikipedia) | Learned from the reports |
|:-----|:------------------------|:-------------------------|
| scaffold | sc, ##af, ##fold | scaffold |
| excavator | ex, ##ca, ##vat, ##or | excavator |
| amputated | amp, ##uta, ##ted | amputated |
| lacerated | lace, ##rated | lacerated |
| rebar | re, ##bar | rebar |
| forklift | fork, ##lift | forklift |

::: notes
"##" marks a piece that continues a word. A vocabulary reflects the text it was learned from:
BERT's (Devlin et al., 2019; 30,522 pieces) from books and Wikipedia, ours from injury reports
(8,000). "lace + rated" and "re + bar" are not wrong: the model still sees the whole sequence.
But every split spends tokens and asks the model to rebuild a word from pieces whose other
uses mean something else. A model for a field is often pretrained, or at least tokenized, on
that field's text.
:::

## Words become vectors

$$\max \sum_{(w, c)} \Big[\log \sigma(u_w \cdot v_c) + \sum_{k=1}^{5} \log \sigma(-u_w \cdot v_{n_k})\Big]$$

| Word | Its nearest words in the reports |
|:-----|:---------------------------------|
| ladder | rung, rungs, climbing |
| trench | excavation, caved, soil |
| forklift | standup, backing, forklifts |
| scaffold | baker, plastering, stucco |

::: notes
Word2vec's skip-gram (Mikolov et al., 2013): each word w gets a vector u_w that should predict
the words c within 5 positions of it (σ of the dot product high), and not 5 random words n_k
(drawn by frequency^0.75). Words used in the same contexts end up with similar vectors.
Trained here from scratch on the training years: 6,783 words seen at least 5 times, 100
numbers each, 3 passes, 5.3 min on a CPU shared with other jobs. "baker" is a Baker scaffold, a rolling platform.
BERT starts the same way, with a learned table of one vector per token (30,522 rows, learned
with the rest of the model); ViT instead projects each patch's pixels to a vector.
:::

## Assume: similar keys, similar values

![Left: the tested mixes as grey dots by water-binder ratio and age; the proposed mix, a star at a w/b near 0.5 and 28 days; the mixes with the largest weights, circled, sit beside it. Middle and right are still empty.](cee_l17_transformers/attention_1.png)

::: notes
Lecture 2's question again: predict a proposed mix's 28-day strength from the tested ones. The
query is the proposed mix (row 638, Lectures 5 and 10); the keys are the 1,028 tests of other
designs (427 mixes, some tested at several ages), described by Lecture 2's six mix groups, standardized; the values are their
strengths. The assumption is Lecture 2's: mixes with similar keys have similar values.
:::

## Weights from similarity, by softmax

![The same, with the middle panel filled: the eight largest softmax weights as bars, each labelled with its test's strength (37.4 MPa for the three largest); the largest weights are about 0.1 each.](cee_l17_transformers/attention_2.png)

::: notes
Score every key by its similarity to the query, s_i = −|q − k_i|² / τ, and turn the scores into
weights with Lecture 6's softmax: w_i = e^(s_i) / Σ_j e^(s_j). Every tested mix gets a weight; the
eight largest carry 62%. τ, the temperature (0.2), is chosen on whole held-out designs: small τ
gives all the weight to the nearest mix (1-NN), large τ spreads it evenly (the mean).
:::

## The prediction: a weighted average

![The full picture: right panel, attention predicts 35.3 MPa, Lecture 2's 5-NN 33.2 MPa; the mix tested at 38.2 MPa.](cee_l17_transformers/attention_3.png)

::: notes
Attention's output is Σ w_i v_i: 35.3 MPa; 5-NN, 33.2; the test, 38.2. On Lecture 2's 30 splits by
design, attention's median RMSE is 6.56 MPa against 5-NN's 6.71, lower in 25 of 30: a soft
nearest-neighbour average. This is cross-attention: the query comes from one set (new
mixes), keys and values from another (tested mixes). CS 441 calls it instance-based regression.
:::

## Self-attention: tokens query each other

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$

$$\text{Attention}(X) = \text{softmax}\!\Big(\frac{QK^\top}{\sqrt{d}}\Big)\,V$$

::: notes
Self-attention: every token is a query, a key, and a value at once, each by its own learned
matrix (W_Q, W_K, W_V, learned by backpropagation, Lecture 14). Similarity is a scaled dot
product instead of a distance; √d keeps the scores from growing with the dimension d, so the
softmax does not saturate. Each output token is a weighted average of all tokens' values,
weighted by how well its query matches their keys. Multi-head: do this h times with smaller
matrices, side by side, so different heads can match on different things (ViT-B/16: 12 heads
of 64 numbers). Vaswani et al. (2017).
:::

## Order needs a position

![Four 14 by 14 maps of a vision transformer's learned position embeddings: for a patch in a corner, an edge, the centre, and the bottom row, how similar every other patch's position embedding is to its own; each map lights up the patch's own row and column, brightest near the patch.](cee_l17_transformers/positions.png)

::: notes
Attention is a weighted sum over all tokens: shuffle the tokens and the outputs shuffle with
them; nothing in it knows order. So a position embedding is added to each token: fixed sines
and cosines in the original transformer, learned numbers in ViT. ViT-B/16 learned these from
ImageNet alone: neighbouring patches' position embeddings have cosine similarity 0.73 on average,
patches 8 or more apart -0.10, and the row-and-column pattern of the image grid appears on its own
(Dosovitskiy et al., 2021, showed the same).
:::

## One block: attend, then think

![A vision transformer reading a crack tile: the 224-pixel tile is cut into 16-pixel patches, giving 197 tokens of 768 numbers with a class token and position added. One block normalizes, applies self-attention with 12 heads, adds the input back, normalizes again, applies an MLP from 768 to 3072 to 768 numbers, and adds again: 7.09 million weights, the same shape 12 times. The class token goes to Lecture 15's probe; 85.8 million weights in all.](cee_l17_transformers/vit_block.png)

::: notes
A transformer alternates two steps: attention, where tokens exchange information, and an MLP,
applied to each token alone, where each token processes what it gathered (Lecture 14's network,
768 → 3,072 → 768, two-thirds of the block's weights). Each step sits inside a residual path
(Lecture 15's skips) after a layer norm. ViT-B/16 stacks 12 identical blocks of 7,087,872
weights; the class token, a learned extra token, gathers the tile's summary, and Lecture 15's
probe reads it. Weights: 85,798,656 against ResNet-18's 11,176,512.
:::

## Your turn: predict the check

```python
def block(x, blk):                     # x: (1, 197, 768) tokens
    a, _ = self_attention(norm(x, blk.ln_1), blk.self_attention)
    x = x + a                          # attend, then add back
    y = norm(x, blk.ln_2)
    h = gelu(y @ blk.mlp[0].weight.T + blk.mlp[0].bias)
    return x + h @ blk.mlp[3].weight.T + blk.mlp[3].bias
```

Against torchvision's first ViT block, on a real crack tile, in float64: how close?

::: notes
`self_attention` is the slide's two equations in eight lines: one matrix product makes Q, K,
and V for all heads, reshape to 12 heads of 64, softmax(QKᵀ/√64)V, heads side by side again,
one output projection. The tile is Lecture 15's filter tile, cut and embedded by ViT-B/16's
own patch layer. Write a number.
:::

## Agrees with torchvision to round-off

| Quantity, one tile | Value |
|:-------------------|------:|
| Tokens × numbers | 197 × 768 |
| Heads | 12 |
| Largest difference, all outputs | below 10⁻¹⁵ |

::: notes
Float64 round-off: the order of additions differs between our code and PyTorch's fused kernel,
so the last bits can differ, and they differ between runs with different thread counts too
(the notebook asserts below 10⁻¹⁴, not a digit). The block is the whole architecture: ViT-B/16 is
this function applied 12 times with different weights.
:::

## BERT: learn by filling blanks

![Bars for each word that completed "fell from a ___" in the 2023-2025 reports, beside how often BERT's first guess was that word: ladder 238 and 231, roof 73 and 124, trailer 49 and 60, truck 43 and 57, top 28 and 64, forklift 27 and 0, lift 24 and 26, back 23 and 15, side 17 and 38, and building, which the reports say once and BERT guesses 41 times.](cee_l17_transformers/masked.png)

::: notes
BERT (Devlin et al., 2019): a stack of transformer blocks over text tokens, pretrained by hiding
15% of the tokens of books and Wikipedia and predicting them from both sides. No labels: the
text is its own answer key. Test on reports it never saw: every 2023–2025 report with "fell
from/off a/an/the ___" (1,102 of them), the word replaced by [MASK]. BERT's first guess is right
46% of the time, in its top five 62%; always guessing ladder (the most common answer in 2015–2022)
is right 22% (top five 39%). Of the answers, 956 are one BERT token; the others cannot be
guessed in one step and count as misses. BERT-base, bert-base-uncased (Apache-2.0).
:::

## Patches are tokens: the screen

![Three pairs of bars on Lecture 15's test photographs, ResNet-18 against ViT-B/16 under the same probe and threshold rule: average precision 0.67 against 0.77; cracks flagged 86% against 88%; tiles to review 51% against 41%.](cee_l17_transformers/decide.png)

::: notes
The tile becomes 197 tokens: 196 patches of 16 × 16 pixels, each flattened and projected to 768
numbers, plus the class token (Dosovitskiy et al., 2021). ViT-B/16's ImageNet weights, frozen;
the same probe, the same split, the same threshold rule as Lecture 15. Test photos: average
precision 0.77 against 0.67; at the threshold, 88% of cracks flagged against 86%, and 41% of tiles to review
against 51%. By surface (review / cracks flagged): decks 50% / 84% against 66% / 91%, walls
56% / 92% against 66% / 92%, pavements 26% / 86% against 31% / 77%. One threshold, set to flag 90% of all cracks, flags different shares on
each surface (ResNet-18: decks 91%, pavements 77%): the target is set overall, as in Lecture 15.
:::

## What the class token draws on

![Four test tiles with cracks and, below each, where the class token's attention goes, computed by attention rollout: on the wall tile it runs along the crack; on the deck tile it sits on the crack's darkest patch; on the pavement tile the screen catches, mostly beside the crack rather than on it; on the pavement tile it misses, on the dark band at the tile's right edge.](cee_l17_transformers/attention_maps.png)

::: notes
Attention rollout (Abnar & Zuidema, 2020): average each block's attention over its heads, add
the residual path (half and half with the identity), and multiply the 12 blocks; the class
token's row shows which patches its summary draws on. Computed with our block, not a library.
The first three tiles are the screen's surest catches on each surface (p near 1); the fourth
is its surest miss (p = 0.0008 against a threshold of 0.050). A map is evidence of what the model
used, not of why; rollout ignores the MLPs and the values (Chefer et al., 2021).
:::

## CNNs and transformers, side by side

| Aspect | ResNet-18 (Lecture 15) | ViT-B/16 (today) |
|:-------------------|:--------------------------------|:--------------------------------|
| Position | the index in the feature map | a learned embedding per patch |
| Locality | built in: small filters | learned: any patch, any layer |
| Weights | 11,176,512 | 85,798,656 |
| Time per tile | 1.25 ms | 5.37 ms |
| Data needs | ImageNet suffices | far more, or ImageNet + care |

::: notes
CS 441's comparison. A CNN assumes locality (Lecture 15's filters) and gets it for free; a
transformer must learn which patches matter, which takes more data: ViT trained on ImageNet
alone trails ResNets, and beats them after pretraining on 300 million images (Dosovitskiy et
al., 2021); the weights used here were trained on ImageNet with strong augmentation
(torchvision's recipe). Time per
tile: reading, resizing, and the network on this GPU, averaged over 224,368 passes each, while
other jobs shared the machine (timings move from run to run).
:::

## Your turn: predict four failures

1. Cut each tile into 32-pixel blocks, shuffle them, refit both probes.
2. Which of reports A, B, C does each representation call alike?
3. BERT, given only "The employee fell from a [MASK] and fractured his leg."
4. A tokenizer from books, on injury reports.

Write what happens to each, and why.

::: notes
Two minutes alone. The next slides reveal one each (the fourth was the vocabulary slide; ask
what it costs).
:::

## Shuffled tiles: both readers lose

| Test photos, probe refitted | Average precision | Tiles to review |
|:----------------------------|------------------:|----------------:|
| ResNet-18, tiles as taken | 0.66 | 50% |
| ResNet-18, blocks shuffled | 0.53–0.55 | 57–60% |
| ViT-B/16, tiles as taken | 0.77 | 41% |
| ViT-B/16, blocks shuffled | 0.70–0.71 | 49–51% |

::: notes
Each tile's 224-pixel input is cut into 49 blocks of 32 pixels and put back in a random order,
the same for every tile; three permutations; both probes refitted on the shuffled tiles. A block
is 2 × 2 of ViT's patches and one cell of ResNet-18's last 7 × 7 feature map, so the cut is
aligned to both grids, but the seams differ for the two: ViT's patch embedding never reads
across a seam between blocks, while ResNet-18's filters cross the seams at every layer. Ranges are over
the three permutations. Both lose; neither is order-blind. ResNet-18's average precision falls
by 0.12 to 0.13, the ViT's by 0.06 to 0.07: ResNet-18 loses about twice as much under each
permutation (one photo split), part of which may be the seams only its filters see (a test
that isolates the seams was not run). Shuffled, the ViT
still ranks above the intact ResNet-18 (0.70–0.71 against 0.66). These are the measured drops;
which features each network loses was not measured. "Tiles as taken" re-embeds every tile: its
features agree with the cache to 2.4e-5, yet ResNet-18's refitted probe gives 0.66 and 50%
against 0.67 and 51% from the cache, so read the third decimal of any probe with care.
:::

## Similar to what?

| Pair | Shared words | Word vectors | BERT |
|:---------------------------------------|------------:|------------:|-----:|
| A, B: both falls | 0.16 | 0.94 | 0.87 |
| A, C: both ladders at a house | 0.28 | 0.97 | 0.94 |
| B, C | 0.23 | 0.95 | 0.87 |
| All test reports: same event first (AUC) | 0.64 | 0.79 | 0.66 |

::: notes
The attempt revealed: every representation calls A and C the closest pair; a safety engineer
coding the event calls A and B alike (both falls to a lower level; C is an electric shock).
Word vectors and BERT: the cosine of each report's mean vector. Three reports are an example,
not a measurement. The last row is the measurement: all 26,599
test reports with a classifiable event, 199,991 random pairs (32% of them of the same event
division: falls, contact with objects, ...). AUC: how often a same-division pair scores above a
pair of different divisions (0.5 is a coin, 1 always). All three track the event, none
closely: the mean word vectors best (0.79), BERT's mean token about as well as shared words
(0.66 against 0.64). Averaged vectors are also all alike: two random reports have a median
cosine of 0.93 with word vectors and 0.86 with BERT (shared words: 0.16), and the three
reports differ only in the second decimal. To get your meaning, train for it: a probe on frozen features with
your labels (Lecture 15, and today's screen), or a model trained to embed sentences
(Lecture 19).
:::

## Context fills the blank

| BERT's input | First guess |
|:-------------|:------------|
| "The employee fell from a [MASK] and fractured his leg." | building (ladder 8th) |
| The full report, 2023–2025 | right 46% of 1,102 |

::: notes
A one-line sentence gives BERT only general English: people fall from buildings, windows,
roofs. A full report gives it the job: the task, the tools, the injury; self-attention lets the
blank draw on every one of those tokens. The same model, with more context, beats the
frequency baseline (22%) by a factor of two. Its first guesses still include words the reports
rarely end the phrase with: "building" 41 times against the reports' once, "side" 38 times
against 17.
:::

## Swap the screen's backbone?

| On the test photos | ResNet-18 | ViT-B/16 |
|:-------------------|----------:|---------:|
| Cracks flagged | 86% | 88% |
| Tiles to review | 51% | 41% |
| Average precision | 0.67 | 0.77 |
| Weights | 11,176,512 | 85,798,656 |
| Time per tile | 1.25 ms | 5.37 ms |

::: notes
Overall the ViT screen opens fewer tiles at a similar share of cracks flagged, but not on every
surface: on decks it reviews 50% of tiles against 66% and flags 84% of cracks against 91%; on
pavements it flags more (86% against 77%) and reviews fewer. One threshold set overall moves the misses
between surfaces; a deck owner sets it on decks. The price: about 8 times the weights and
about 4 times the time per tile; all 54 deck photos of SDNET2018 (13,620 tiles) take about 73 s with the
ViT and about 17 s with ResNet-18 on this GPU, shared with other jobs (timings move from run to run). Lecture 15's caution holds unchanged: both thresholds were set on SDNET2018's validation
photos; a new owner's camera needs its own check.
:::

## Who answers for a borrowed model?

| | Lecture 15's screen | Today's screen |
|:--|:--------------------|:---------------|
| Trained on | ImageNet, then our labels | ImageNet, then our labels |
| Size, energy | 11 M weights | 86 M weights, about 4× the time |
| What we can check | our test photos | our test photos |
| What we cannot | ImageNet's contents | ImageNet's contents |

::: notes
The social and environmental note. Both screens rest on a pretrained network whose training
data (ImageNet; for BERT, books and Wikipedia) no one in this room chose or audited; our
evidence is our own test set and a random audit of unflagged tiles in service (Lecture 15).
A larger model costs more energy for every tile screened; at a few milliseconds a tile it is
small here, and large when an agency screens every photo of every bridge every cycle. The
inspector who signs the report answers for both.
:::

## Vote: what screens next season's decks?

- **A.** Lecture 15's ResNet-18 screen, as is.
- **B.** The ViT-B/16 screen, threshold re-set on the owner's photos, unflagged tiles audited.
- **C.** Fine-tune a vision transformer on SDNET2018 first.
- **D.** No screen: review every tile.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
B. On these photos the ViT screen reviews 41% of tiles against 51%, flagging 88% of cracks against 86%; the extra
cost is milliseconds per tile; on decks alone it flags fewer, which is why the threshold must
be re-set on the owner's own deck photos. The re-set threshold and the audit are Lecture 15's conditions,
unchanged. C may gain more, but needs a GPU budget, a validation plan, and a reason to expect
gains from 137 training photos (CS 441: with moderate data, fine-tune; with little, probe). A
is defensible if the owner already validated it; D is the baseline.
:::

## Where transformers live today

| Today | What it is | Lecture |
|:------|:-----------|:--------|
| ChatGPT, Claude, Gemini | decoder-only transformers on text tokens | 19 |
| CLIP, SAM, Prithvi | transformers on image patches and text | 18 |
| Chronos-2, TimesFM | transformers on time-series patches | 20 |
| Whisper | a transformer on spectrogram patches | 20 |

::: notes
The block on today's slides is the block inside all of them. Decoder-only models (GPT; Lecture
19) mask attention so each token sees only earlier ones and predict the next token; encoders
(BERT, ViT) see both sides. Lecture 18 uses CLIP and SAM on construction images; Lecture 20
uses Chronos-2 on rivers.
:::

## Exit question

A vendor's ViT screen reports 95% recall on "held-out tiles". Name two things you would ask
before swapping your ResNet screen for it, and the number you would want instead.

**In the notebook** (`cee_l17_transformers.ipynb`): byte pairs, word vectors, attention on
concrete, a transformer block from scratch, BERT's blanks, and the ViT screen.

::: notes
Answer: held out how? Tiles of the same photographs leak (Lecture 15's siblings): ask for
whole photos, or whole structures, held out. At what threshold, set on what? Ask for recall and
the share of tiles to review at a threshold fixed on validation, by surface, and on your own
camera's photos. The number to want: tiles to review at your recall target, on your photos.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- Text and images become tokens.
- Attention: a softmax-weighted average.
- A block: attend, then an MLP, with skips.
- Position must be added.
- Similar to what? Train for your meaning.

**Next:** CLIP and SAM, foundation models for images.
:::
::: {.column}
![Four thumbnails: the proposed mix attending to the tested mixes, with its weights and prediction; BERT's guesses against what the reports said; a vision transformer's learned position embeddings; ResNet-18 against ViT-B/16 on the inspector's numbers.](cee_l17_transformers/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Abnar & Zuidema (2020). *ACL*.
- Chefer et al. (2021). *CVPR*.
- Devlin et al. (2019). *NAACL*.
- Dosovitskiy et al. (2021). *ICLR*.
- He et al. (2016). *CVPR*.
- Maguire et al. (2018). SDNET2018.
- Mikolov et al. (2013). *NeurIPS*.
:::
::: {.column}
- OSHA (2025). Severe Injury Reports.
- Sennrich et al. (2016). *ACL*.
- Vaswani et al. (2017). *NeurIPS*.
- Wu et al. (2016). arXiv.
- Yeh (1998). *Cem. Concr. Res.* 28.
:::
::::::
