---
title: Vision Foundation Models
subtitle: "Lecture 18 · Can the screen start before anyone labels a tile?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 6–17

On your own, no notes, two minutes:

1. Lecture 6: cross-entropy for one example with 3 classes, when the right class gets p = 0.5?
2. Lecture 15: what did the probe on frozen ResNet-18 features need that a network trained
   from scratch needed far more of?
3. Lecture 17: BERT learned from books with no labels. Where did its "answers" come from?

**Today:** a model that learned from captions, asked about cracks it has never been taught.

::: notes
Give the answers before moving on.
1. −log 0.5 ≈ 0.69. Today CLIP is trained with this loss, over a batch: each image must pick
   its own caption out of all the batch's captions.
2. Labelled photos: with 16 training photos the probe beat the network from scratch. Today: how
   many labels does a newer model need, and can it start with none?
3. From the text itself: hide a word, predict it. CLIP's labels are free in the same way:
   400 million images that came with captions on the web.
:::

## A new owner, no labels yet

:::::: {.columns}
::: {.column}
- Lecture 15: the screen learned from 137 labelled photos.
- A new owner has photos, and no labels.
- Screen now, or label first? How many?
:::
::: {.column}
| Option | Labels needed |
|:-------|:--------------|
| Zero-shot: describe the classes in words | none |
| Probe on a pretrained model | a few photos |
| Lecture 15's screen | 137 photos, labelled |
:::
::::::

::: notes
Labelling is the cost an owner sees first: an inspector marking every tile of a photo as
cracked or not takes a while, and someone must check the marks. A foundation model trained on
image-caption pairs from the web can be asked, in words, which tiles show cracks. Today's
question: how well does that work on our tiles, and how few labels beat it?
:::

## Your turn: which caption fits each tile?

![Six test tiles numbered 1 to 6: a bridge deck, a concrete wall, and a pavement, each once without and once with a crack; next to them a grid of the six captions (bridge deck, cracked bridge deck, concrete wall, cracked concrete wall, pavement, cracked pavement), still empty.](cee_l18_vision_foundation/picture_1.png)

::: notes
Two minutes: for each tile, pick the caption you think fits best. Tiles 1 and 2 are a deck
without and with a crack, 3 and 4 a wall, 5 and 6 a pavement (tell them after they commit).
Keep the answers; the grid is filled in with CLIP's answers on the next slides.
:::

## Foundation models: train once, adapt

| Way to use it | What it needs | Lecture |
|:--------------|:--------------|:--------|
| Zero-shot | the task in words, no examples | today |
| Linear probe | a few labelled examples, a frozen network | 15, 17, today |
| Nearest neighbours | labelled examples, a frozen network | 2, today |
| Fine-tune | many examples, a GPU budget | 15 |

::: notes
CS 441's definition: models trained on exorbitant data and compute on a broad task, for
general use or for adapting to other tasks. ImageNet networks (Lecture 15) and BERT (Lecture 17)
are adapted by probing or fine-tuning; CLIP and GPT (Lecture 19) can also be used zero-shot,
with no examples of the new task. The four ways are CS 441's for CLIP; today runs three of
them on the crack tiles.
:::

## A text encoder as the classifier

$$p(c \mid x) = \frac{\exp\!\big(s \cos(f(x), g(t_c))\big)}{\sum_{c'} \exp\!\big(s \cos(f(x), g(t_{c'}))\big)}$$

f: image encoder (ViT-L/14). g: text encoder. $t_c$: a caption per class. s = 100.

::: notes
CLIP (Radford et al., 2021). The classifier's weights are not learned for the task: they are
the text embeddings of one caption per class ("a photo of a cracked bridge deck"). Lecture 6's
softmax regression with weights written in words. The image and the captions are embedded in
one 768-number space; cosine similarity scores each pair; s, the inverse temperature, was
learned in training (CLIP clips it at 100). A new task needs new captions, not new training.
:::

## Trained to match images to captions

:::::: {.columns}
::: {.column}
- 400 million image-caption pairs from the web.
- Batches of 32,768: every image against every caption.
- Loss: pick your own caption, and your own image.
:::
::: {.column}
$$\mathcal{L} = \tfrac{1}{2}\big[\text{CE}_{\text{rows}}(sZ_IZ_T^\top) + \text{CE}_{\text{cols}}(sZ_IZ_T^\top)\big]$$
:::
::::::

::: notes
The contrastive objective: in a batch of N pairs, the N × N similarities have the true pairs on
the diagonal; cross-entropy along each row picks each image's caption among N, along each
column each caption's image among N. Lecture 6's loss, twice. Predicting the exact caption is
hopeless; picking it out of a batch is learnable, and every batch is a new 32,768-way
classification task. Cost, from the paper: the largest ResNet took 18 days on 592 V100 GPUs,
the largest ViT 12 days on 256 (CS 441: about $300K and $91K); the model used here is that
ViT-L/14, given one more epoch at 336 pixels.
:::

## Your turn: predict the check

```python
zi = img / img.norm(dim=1, keepdim=True)     # 16 tiles
zt = txt / txt.norm(dim=1, keepdim=True)     # their 16 captions
logits = scale * zi @ zt.T                   # (16, 16)
target = torch.arange(16)                    # the diagonal
loss = 0.5 * (cross_entropy(logits, target)
              + cross_entropy(logits.T, target))
```

Against the loss in Hugging Face's CLIP, in float64: how close?

::: notes
The batch: 16 random SDNET2018 tiles, each with a caption from its labels ("a photo of a
cracked concrete wall"). Captions repeat in such a batch (several walls without cracks); the
loss still counts only the diagonal as right, which is why CLIP needs captions as varied as the
web's. Write a number.
:::

## The same loss, to the last bit

| Batch of 16 tiles and captions | Value |
|:-------------------------------|------:|
| Loss, ours | 3.8438 |
| Difference from Hugging Face's | 0 |
| Chance level, log 16 | 2.7726 |

::: notes
Identical in float64: the same operations in the same order. The loss is above log 16 (the
loss of guessing uniformly): with repeated captions, the right caption is often not the most
similar, and CLIP is confidently wrong on some of these tiles. The next slide shows how.
:::

## Six tiles: 2 of 6 captions right

![The six tiles and CLIP's probabilities over the six captions, as a grid. Tile 1, a clean deck: concrete wall, 0.76. Tile 2, a cracked deck: pavement, 0.52. Tile 3, a clean wall: concrete wall, 0.79. Tile 4, a cracked wall: concrete wall, 0.87. Tile 5, a clean pavement: pavement, 0.62. Tile 6, a cracked pavement: cracked concrete wall, 0.63.](cee_l18_vision_foundation/picture_2.png)

::: notes
The attempt revealed. Each row: one tile, CLIP's softmax over the six captions (s = 100). The right
surface gets the most weight for 3 of 6 tiles; cracked or not, for 4 of 6; the whole caption for
2 of 6. No bridge deck is called a deck. Compare with the class's answers. Six tiles are an
anecdote: before saying what CLIP "knows", the next slide asks all the test tiles.
:::

## Over all tiles: cracks, not surfaces

| The six captions, 11,208 test tiles | Value |
|:-----------------------------------------|------:|
| Surface named right | 45% |
| … always answering "pavement" | 44% |
| Deck tiles called "bridge deck" | 4 of 2,772 |
| Cracked or not, as a ranking (AUC) | 0.78 |
| Wall or pavement, as a ranking (AUC) | 0.62 |

::: notes
The six tiles suggested CLIP tells surfaces apart better than cracks; all the test tiles say
the opposite. CLIP names the surface barely more often than always answering "pavement" would, and deck tiles are
nearly all called something else (2,497 of them walls): an SDNET2018 tile is a 256 × 256 crop of
a close-up photo of concrete, on a deck as on a wall. Cracked against clean tiles rank well (AUC
0.78, where 0.5 is a coin), walls against pavements barely (0.62). The lesson is the course's: a
claim about what a model knows needs the test set, not a handful of examples (an earlier draft
of this slide got it backwards from the six tiles).
:::

## Zero-shot: prompts, then a threshold

| Prompt pair (clean / cracked) | Validation AP |
|:------------------------------|-------------:|
| a photo of concrete / … of cracked concrete | 0.51 |
| a photo of a {surface} / … of a cracked {surface} | 0.43 |
| … without cracks / … with a crack | 0.48 |
| a close-up photo of intact concrete / … of a crack in concrete | 0.52 |

::: notes
p(cracked) for every tile from each pair of captions; the pair is chosen on Lecture 15's
validation photos (close-up), and the test photos scored once: average precision 0.52
(AUC 0.76), against Lecture 15's ResNet-18 probe 0.67. With no labels the natural threshold is p = 0.5: it
flags 55% of cracked tiles and sends 22% of tiles to review. Set to flag 90% on labelled validation photos (so
not label-free anymore), it flags 84% and sends 63% to review, against ResNet-18's 51%.
:::

## A few labels beat no labels

![Test average precision against labelled training photos per surface, 1 to 16 and all: CLIP's probe, the median of ten draws with their range as a band, climbs from 0.63 with one photo per surface to 0.76 with 16 and 0.78 with all; zero-shot CLIP is a flat dashed line at 0.52, below every draw; Lecture 17's ViT-B/16 probe, 0.77, and Lecture 15's ResNet-18 probe, 0.67, are marks at "all".](cee_l18_vision_foundation/shots.png)

::: notes
Probes (Lecture 15's logistic regression) on CLIP's image embeddings, as an owner with k
labelled photos per surface would fit them: ten random draws of k training photos per surface;
C fixed at 0.01, since k photos leave no labels to tune it on; the 90% threshold set on k more
labelled photos per surface, from the validation photos. Two choices borrow labels this owner
would not have: 0.01 is the C the probe on all photos chose on all 47 validation photos, and
zero-shot's prompt pair was chosen on the same 47. Median test AP 0.63 with 1,
0.68 with 2, 0.71 with 4, 0.73 with 8, 0.76 with 16, and 0.78 with all (32, 43, 62 photos of
decks, walls, pavements, C and threshold on all 47 validation photos). One labelled photo per
surface beats zero-shot in every draw (lowest 0.56). Two per surface (0.68, range 0.67–0.69
over the draws) come close to Lecture 15's ResNet-18 trained on all 137 (0.67), one split.
Checked after the fact on the test photos, C does not decide it: at 0.003 or 0.03 the
two-photo median is 0.69 or 0.67. A "photo" is many
labelled tiles: one photo per surface was 738 tiles in the first draw (93 cracked). On everyday
photos, CLIP's paper finds zero-shot matching a probe on CLIP's own features at about four
labelled examples per class (Radford et al., 2021); on cracks, one photo per surface already
wins.
:::

## Three ways to adapt, on the test photos

| Adaptation of CLIP | Labels | AP | Tiles to review |
|:-------------------|:-------|---:|----------------:|
| Zero-shot, threshold 0.5 | none | 0.52 | 22% |
| Nearest neighbours (k = 50) | all training photos | 0.65 | 55% |
| Linear probe | all training photos | 0.78 | 39% |
| Linear probe | 2 + 2 photos per surface | 0.68 | 52% |

::: notes
Review shares at the 90% target, the threshold set on the validation photos (all 47, or two
per surface for the last row: the median of ten draws), except zero-shot at 0.5, which flags
only 55% of cracks. Nearest neighbours (Lecture 2, cosine distance on CLIP's embeddings, each
neighbour's vote weighted by one over its distance, k = 50 chosen on validation from 5 to 100)
flags 88% and asks for 55% of tiles; with weighted votes no test tile scores exactly at the
threshold. The fourth way, fine-tuning CLIP's encoder, is not run here: it needs a GPU budget,
and the model loses its general use (CS 441). The linear probe on all photos, 39% of tiles at
88% of cracks, is close to Lecture 17's ViT-B/16 probe (41%) on this one split.
:::

## Segment anything, from a click

:::::: {.columns}
::: {.column}
- SAM: a click, a box, or a mask in; masks out.
- Trained on 11 million images, 1.1 billion masks.
- Three candidate masks, each with a predicted IoU.
:::
::: {.column}
| Lecture 16 | SAM |
|:-----------|:----|
| U-Net trained on 3DCrack | no training on cracks |
| a class per pixel | the object at the click |
| IoU 0.70 on the test images | ? |
:::
::::::

::: notes
Segment Anything (Kirillov et al., 2023; SA-1B: 11 million images, 1.1 billion masks) and SAM
2 (Ravi et al., 2024): an image encoder run once, a prompt encoder for clicks or boxes, and a
light decoder that returns three nested candidate masks (a part, the object, the whole) with
its own estimate of each one's IoU. Used here: SAM 2.1 Hiera base-plus (Apache-2.0). The
question: can a click on a crack replace Lecture 16's trained U-Net?
:::

## Your turn: predict four failures

1. Reword the zero-shot prompts.
2. Screen with zero-shot and no labels at all.
3. Click 1, 3, or 5 times on a crack's centreline.
4. Click once inside a building (SpaceNet 8).

Write what happens to each, and why.

::: notes
Two minutes alone. The next three slides reveal them: the prompts, the threshold, the clicks.
:::

## Prompts move the answer

| Prompt pair (clean / cracked), test photos | Test AP |
|:-------------------------------------------|--------:|
| a photo of a {surface} / … of a cracked {surface} | 0.42 |
| … without cracks / … with a crack | 0.46 |
| a photo of concrete / … of cracked concrete | 0.47 |
| a close-up photo of intact concrete / … of a crack in concrete | 0.52 |

::: notes
First reveal. The same images, the same model; only the words change, and average precision
moves from 0.42 to 0.52. Naming the surface ("a cracked bridge deck") does not hurt the ranking within a
surface; it ranks a little better there: test AUC 0.78, 0.79, 0.80 on decks, walls, pavements, against 0.75, 0.78, 0.75 for the close-up
pair. It lifts every deck tile instead (median P(cracked) 0.59 on decks, 0.04 on walls,
0.04 on pavements), so clean decks outrank cracked walls once the surfaces are pooled: one
threshold for all surfaces needs scores on one scale. Prompt wording is a hyperparameter: choose
it on validation photos, as here, never on the test photos (CLIP's paper averages 80 prompts
per class, prompt ensembling). Here the validation choice was also the test's best; it need
not be.
:::

## No labels, no threshold

| Zero-shot CLIP, test photos | Cracks flagged | Tiles to review |
|:----------------------------|---------------:|----------------:|
| Threshold 0.5 (no labels at all) | 55% | 22% |
| Threshold set on 47 labelled validation photos | 84% | 63% |
| … on 2 labelled photos per surface (10 draws) | 71–92% | 39–78% |
| By surface, the 47-photo threshold: decks / walls / pavements | 82% / 86% / 85% | 62% / 62% / 64% |

::: notes
Second reveal. Without labels, the only threshold is the softmax's 0.5, and it misses nearly
half the cracks: nothing ties 0.5 to the inspector's 90%. Setting the threshold for 90% needs
labelled photos, so zero-shot is not label-free once a target is set. With all 47 validation
photos it sends 63% of tiles to review; with two labelled photos per surface, what it flags
depends on which two (71% to 92% of the cracks over ten draws, median 86%; a median 65% of
tiles to review), against a median 52% for a probe fitted on two photos per surface with the
same threshold photos. A zero-shot score ranks tiles; it does not say how many to look at.
:::

## SAM finds buildings, not cracks

![Left: a 3DCrack range image with the labelled crack in green, three clicks on it, and SAM's mask, a blob that covers far more than the crack. Middle: a SpaceNet 8 house with one click and SAM's mask, IoU 0.50 with the footprint (1,655 pixels against 1,142). Right: the IoU of SAM's own pick against the labels, as box plots: cracks with 1, 3, and 5 clicks around 0.04 to 0.12 at the median; buildings with one click around 0.5.](cee_l18_vision_foundation/sam.png)

::: notes
3DCrack's test images with a crack (197): clicks evenly along the labelled centreline, SAM's
own choice of its three masks. Median IoU 0.04 with 1 click, 0.10 with 3, 0.12 with 5; taking the best
of the three candidates by the label (cheating) only reaches 0.27. SpaceNet 8's test buildings
(300 of 750 over 400 pixels, one click at the most interior pixel): median IoU 0.50, 50% above 0.5. SAM learned what
people outline as objects; a crack is a line through a surface. Lecture 16's U-Net, trained
on cracks: 0.70.
:::

## Few labels rank; thresholds need more

| Test photos, 90% target | AP | Cracks flagged | Tiles to review |
|:------------------------|---:|---------------:|----------------:|
| Zero-shot, 2 photos per surface | 0.52 | 71–92% | 39–78% |
| CLIP probe, 2 + 2 per surface | 0.68 | 79–95% | 32–74% |
| CLIP probe, 4 + 4 per surface | 0.71 | 83–93% | 36–59% |
| ResNet-18 probe, all (Lecture 15) | 0.67 | 86% | 51% |
| CLIP probe, all photos | 0.78 | 88% | 39% |

::: notes
The decision for the new owner. "2 + 2": two labelled photos per surface fit the probe, two
more set the threshold; ranges over ten draws of the photos, AP the median. Two per surface
rank about as well as Lecture 15's screen on 137 photos (AP 0.68 against 0.67: close, one
split), but the threshold from two photos per surface is a gamble: depending on which photos,
it flags 79% to 95% of the cracks and sends 32% to 74% of the tiles to review. With four per
surface the ranges narrow (83–93%, 36–59%); with all 137 and 47 photos the CLIP probe gives
0.78, 88%, 39%. Zero-shot can rank photos for a first look, not set a threshold the inspector
can sign. Label a few photos to fit, more to set the threshold, on the owner's own camera, and
refit as labels accumulate in service.
:::

## The model card says: not for deployment

| | What we know |
|:--|:-------------|
| Training data | 400 million web pairs, not released |
| Its card | "any deployed use case … out of scope" |
| What we checked | our test photos, three surfaces |
| What we did not | other cameras, other owners, bias in the web data |

::: notes
OpenAI's model card for CLIP states that any deployed use case of the model, commercial or not,
is currently out of scope, and that the model was released for research. An agency screening
bridge photos is a deployment: before using CLIP's features in service, read the card, the
licence, and test on the owner's data; or use a model released for deployment (SigLIP,
Apache-2.0, is one). The training data are web captions no one in this room chose: whatever
they over- and under-represent comes along. The environmental note: one embedding pass over
56,092 tiles took about 20 minutes on one GPU; training CLIP took thousands of GPU-days.
:::

## Vote: what screens the new decks?

- **A.** Zero-shot CLIP, threshold 0.5: no labels, start today.
- **B.** Label 2 photos per surface, probe CLIP, set the threshold on a few more.
- **C.** Lecture 15's ResNet-18 screen, trained on SDNET2018.
- **D.** Wait until 137 photos are labelled.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
B. A flags only 55% of cracks, with no way to set a threshold an inspector can sign. C was
trained on another owner's photos; its threshold still needs the new owner's. D ends with the
best screen (AP 0.78, 39% to review) but screens nothing until 137 photos are labelled. B ranks
about as well as Lecture 15's screen from six photos (AP 0.68 against 0.67), but its threshold
needs more than two photos per surface: with two, the share of cracks flagged ranged from 79%
to 95% over ten draws; with four, 83% to 93%. B refits as labels accumulate toward D's. B also
respects CLIP's card if the owner chooses a deployable model for the features, and the audit of
unflagged tiles (Lecture 15) applies unchanged.
:::

## Where vision foundation models live

| Today | What it is | Lecture |
|:------|:-----------|:--------|
| Stable Diffusion, DALL-E 2 | CLIP's text side steers image generation | today |
| LLaVA, GPT-4o, Claude | an image encoder feeding a language model | 19 |
| Annotation tools (CVAT, Label Studio) | SAM proposes masks, people correct them | today |
| Prithvi (NASA, IBM) | a foundation model for satellite images | not yet |

::: notes
CLIP's encoders sit inside text-to-image generators and early vision-language models; SAM is
now a standard first draft in labelling tools, which is how it pays in practice: a person
clicks, SAM drafts, the person fixes. Geospatial foundation models (Prithvi, trained on NASA's
Harmonized Landsat-Sentinel data) map floods and burn scars from a few labelled scenes.
:::

## Exit question

A vendor says its screen "needs no training data: describe the defect in words." Name two
things you would ask before using it, and the number you would want.

**In the notebook** (`cee_l18_vision_foundation.ipynb`): CLIP zero-shot with your own prompts,
few-shot probes, the contrastive loss, and SAM on cracks and buildings.

::: notes
Answer: on whose photos was it tested, and at what threshold? Without labels there is no
threshold an inspector can sign. Ask for recall and the share of tiles to review at your
target, on your photos, and what its model card allows. The number: tiles to review at 90% of
cracks flagged, on a few labelled photos of yours.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- CLIP: captions as the classifier.
- Contrastive: pick your pair from a batch.
- A few labels rank; thresholds need more.
- SAM outlines objects; cracks are lines.
- Read the model card before deploying.

**Next:** language models, GPT, and retrieval.
:::
::: {.column}
![Four thumbnails: CLIP's probabilities for six tiles against six captions; test AP against labelled photos per surface, with zero-shot as a flat line; SAM's masks on a crack and a building with their IoU; and the caption grid again.](cee_l18_vision_foundation/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Dosovitskiy et al. (2021). *ICLR*.
- Kirillov et al. (2023). *ICCV*.
- Maguire et al. (2018). SDNET2018.
- Radford et al. (2021). *ICML*.
:::
::: {.column}
- Ravi et al. (2024). SAM 2. arXiv.
- Hänsch et al. (2022). *CVPR Workshops*.
- Zhang et al. (2025). 3DCrack. arXiv.
:::
::::::
