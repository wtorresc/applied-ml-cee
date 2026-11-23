---
title: Convolutions and Transfer Learning
subtitle: "Lecture 15 · Can a network flag the cracked tiles?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 6–14

On your own, no notes, two minutes:

1. Lecture 14's network took 2 inputs. A colour tile of 224 × 224 pixels has 150,528. How
   many weights does a first layer of 64 units need?
2. Lecture 8's forest scored R² +0.16 on random folds and 0.00 with whole sites held out. Why?
3. Lecture 6: missing a crack costs 20 times a needless review. Where does the threshold on
   p go?

**Today:** one layer that covers the whole tile with 1,792 weights.

::: notes
Give the answers before moving on.
1. 150,528 x 64 = 9,633,792 weights, plus 64 biases: 9,633,856. Every unit looks at every
   pixel, and a crack in the corner and the same crack in the centre need different weights.
2. Rows from one site share its climate, traffic, and construction; with sites in the training
   folds, the forest recognized the site rather than learning the deterioration. Today's tiles
   come from 230 photographs, the same question one level down.
3. Flag when p (cost of a miss) > (1 - p) (cost of a review): p > 1/21, about 0.048. Today the
   inspector states the target as a share of cracks to catch instead, and the threshold follows.
:::

## Decks, walls, pavements: 56,092 tiles

![Three photographs rebuilt from their 256-pixel tiles, with the tiles labelled cracked outlined. A bridge deck, 83 of 252 tiles cracked, with an X-shaped pattern of cracks; a wall, 27 of 252, one long diagonal crack past a tie hole; a pavement, 23 of 234, one crack crossing from the shaded half into the sunlit half.](cee_l15_convolutional_networks/photos.png)

::: notes
SDNET2018 (Maguire, Dorafshan & Thomas, 2018; CC BY 4.0): 230 photographs from a 16-megapixel
camera at Utah State University: 54 bridge decks built in the SMASH laboratory, 72 walls of a
campus building, 104 campus pavements and sidewalks. Each photo was cut into 256-pixel tiles, 18
across, and each tile labelled cracked or not by the authors: 56,092 tiles, 15.1% cracked (decks
14.9%, walls 21.2%, pavements 10.7%). The cracks run from 0.06 to 25 mm wide; the tiles include
shadows, scaling, edges, holes, and debris. A photo is the unit of collection: one camera
position, one light, one surface. Ask: which tiles would you call hard?
:::

## A miss waits for the next inspection

:::::: {.columns}
::: {.column}
- The network flags tiles; the inspector reviews only those.
- Target: flag **9 of 10** cracked tiles.
- Then: review as few tiles as possible.
:::
::: {.column}
| The network | The tile is cracked | The tile is clean |
|:------------|:--------------------|:------------------|
| Flags it | reviewed, recorded | seconds of review |
| Passes it | unrecorded | nothing |
:::
::::::

::: notes
A missed crack is not recorded until the next inspection, routinely 24 months later under the
National Bridge Inspection Standards; a false alarm costs seconds of the inspector's time. So
the inspector sets the share of cracks the network must catch, here 90%, and the threshold is
chosen on validation photos to meet it. What is left to optimize is the review load: the share
of tiles flagged. Reviewing everything catches 100% and saves nothing; that is the baseline.
:::

## Your turn: crack or not?

![Eight tiles from test photographs, labelled A to H: decks, walls, and pavements, some with clear dark cracks, some with faint lines, joints, stains, or coarse texture.](cee_l15_convolutional_networks/attempt.png)

::: notes
Two minutes, alone. For each tile write cracked or not, and how sure you are, from 1 to 3.
Then compare with a neighbour: where do you disagree, and why? Keep the sheet; the network's
answers and the labels come back later. Ask: what did you look for? Most students name a thin
dark line that wanders, which is the first filter on the next slide.
:::

## Nine pixels, nine weights, one number

![Four panels. A deck tile with a crack and a small box on it; the box zoomed to 16 by 16 pixels, with a 3 by 3 window on the crack; the window's nine grey levels, 96, 104, 147, 115, 89, 115, 133, 97, 107, beside nine weights, plus 1 around a minus 8 centre; their weighted sum divided by 8 is 202 divided by 8, 25.25. The fourth panel, the response everywhere, is still a question mark.](cee_l15_convolutional_networks/filter_1.png)

::: notes
The filter says: a pixel darker than its eight neighbours. Multiply each of the nine grey
levels by its weight and add: 96 + 104 + 147 + 115 + 115 + 133 + 97 + 107 - 8 x 89 = 914 - 712
= 202; divided by 8, 25.25. On flat concrete the eight neighbours match the centre and the sum
is near 0. Ask: what will the map look like if we do this at every pixel?
:::

## Slide it everywhere: the crack lights up

![The same four panels, now with the fourth filled in: the filter's response at every pixel of the tile, dark almost everywhere, bright along the crack; the window's pixel is circled on the bright line.](cee_l15_convolutional_networks/filter_2.png)

::: notes
Sliding the same nine weights over all 65,536 pixels gives a map that is bright along the
crack and dark elsewhere: the crack's pixel scores 25.25; the brightest pixel, 29.1. That
operation, the same small set of weights applied at every position, is a convolution (strictly a
cross-correlation; the build-it slide comes back to this). A convolutional layer learns 64 such
filters at once, from the data, instead of having them designed.
:::

## The same filter drowns in texture

![Two 3D surfaces of the hand-made filter's response, capped at 40. Left, the deck tile: a flat dark plain with one ridge running across it, the crack; 1% of pixels respond above 10. Right, a pavement tile with a crack in coarse aggregate: spikes everywhere, a forest in which the crack cannot be told apart; 11% of pixels respond above 10.](cee_l15_convolutional_networks/ridges.png)

::: notes
Heights: the response, max-pooled over 4 x 4 pixel blocks so one-pixel ridges stay visible, and
capped at 40. On the deck, 0.8% of pixels respond above 10, along the crack, as the map shows. On the
pavement, 10.6% do: every dark gap between aggregate stones is "darker than its neighbours". A
designed filter encodes one idea of a crack. Learning many filters, and combining them over
larger areas, is how a network tells a crack from texture. Logistic regression on this filter's
summaries is today's naive baseline.
:::

## Assume: a crack looks alike anywhere

:::::: {.columns}
::: {.column}
One filter $w$, every position $(i, j)$, every channel $c$:

$$y_{ij} = b + \sum_{c} \sum_{u,v=0}^{2} w_{c,u,v}\, x_{c,\,i+u,\,j+v}$$

Shift the tile: the map shifts with it.
:::
::: {.column}
| First layer, 64 units, on 3 × 224 × 224 | Weights |
|:----------------------------------------|--------:|
| Dense: every unit sees every pixel | 9,633,856 |
| Convolution: 64 filters, 3 × 3 × 3 | 1,792 |
| ResNet-18's first layer: 64 filters, 7 × 7 × 3 | 9,408 |
:::
::::::

::: notes
The assumption has two parts. Locality: whether a pixel belongs to a crack depends on its
neighbourhood. Stationarity: the same pattern means the same thing anywhere in the tile. Both
give weight sharing: one small filter, applied at every position, so a crack learned in one
corner is recognized in every other; shifting the input shifts the output (translation
equivariance, checked exactly in the notebook away from the borders). Dense layer: 150,528 x
64 + 64. Convolution: 3 x 3 x 3 x 64 + 64 = 1,792. ResNet-18 uses 7 x 7 filters with no bias
in its first layer (a batch normalization follows). Where the assumption fails: a photo's
corners can be blurrier and darker than its centre, and a joint looks like a crack anywhere.
:::

## Stacks of small filters see farther

![Four panels over the deck tile, one per stage of ResNet-18. A dashed square marks each stage's theoretical receptive field: 43, 99, 211, and 435 pixels, the last larger than the tile. A glow shows where the gradient of one central unit falls, and a solid square holds 90% of it: 25, 50, 115, and 208 pixels.](cee_l15_convolutional_networks/receptive.png)

::: notes
Each 3 x 3 layer widens what one unit can see by 2 steps of the layer below; each stride-2 layer
doubles the step. Theory, from the layer sizes: 43, 99, 211, and 435 pixels after the four
stages. In effect, the gradient of a central unit with respect to the input pixels, averaged over
32 random tiles, concentrates in about half of that: 90% falls within 25, 50, 115, and 208
pixels (Luo et al., 2016). By the last stage one unit sees almost the whole 224-pixel tile, which
is why a crack anywhere can switch it on.
:::

## Halve the map, double the channels

![The deck tile passing through ResNet-18. Under each stage, the channel that best follows the crack: input 3 by 224 squared; first layer 64 by 112 squared, r 0.79; layer1 64 by 56 squared, r 0.73; layer2 128 by 28 squared, r 0.88; layer3 256 by 14 squared, r 0.87; layer4 512 by 7 squared, r 0.76. The crack stays visible as the maps get coarser.](cee_l15_convolutional_networks/stages.png)

::: notes
Each stage halves the map's width and, from layer2 on, doubles the channels: 64 x 112 x 112
after the first layer (a 7 x 7 convolution with stride 2), 64 x 56 x 56 after max pooling and
layer1, down to 512 x 7 x 7. Global average pooling then
gives 512 numbers per tile: the features today's logistic regression uses. Under each stage, the
channel whose map correlates best with the hand-made filter's map, at that resolution (r from 0.73
to 0.88). ResNet-18 has 11,176,512 weights; 8,393,728 of them (75%) are in layer4.
:::

## Skips keep the gradient in range

![Left, two blocks: a plain block, x through conv, batch norm, ReLU, conv, batch norm, then ReLU to y; a residual block, the same with an identity path from x added before the ReLU. Right, a log plot of the gradient's size at each of 32 blocks, divided by the last block's, at initialization: the plain stack grows to 36,483 times at the first block; the residual stack to 31.](cee_l15_convolutional_networks/residual.png)

::: notes
A residual block outputs relu(x + F(x)) (He et al., 2016), so its Jacobian is I + dF/dx: the
gradient reaches earlier blocks along the identity, whatever F does. Measured here at
initialization on 64 real tiles (a toy network: 16 channels, 128-pixel tiles, PyTorch's default
start; median of five seeds): the gradient at the first block over the last grows 11, 175, and
36,483 times through 8, 16, and 32 plain blocks, and 8, 15, and 31 times through residual ones.
Deep plain stacks with batch normalization explode at the start (Yang et al., 2019); a skip turns
products of Jacobians into sums (Balduzzi et al., 2017). He et al. trace the trouble of deep
plain networks to optimization, not vanishing gradients. ResNet-18's eight blocks are residual.
:::

## ImageNet's first filters find edges

![Left, the 64 first-layer filters of ResNet-18 trained on ImageNet, each 7 by 7 by 3, shown as small colour patches: oriented edges, stripes, colour blobs. Two are outlined. Right, their responses on the deck tile: filter 49 traces the crack as a thin bright line, r 0.76 with the hand-made filter; filter 45, a colour blob, as a broader line, r 0.63.](cee_l15_convolutional_networks/conv1.png)

::: notes
These filters were learned from 1.28 million photographs of everyday objects (ImageNet; He et
al., 2016; torchvision's weights). Nobody showed them a crack, yet several are oriented edge and
line detectors that respond to one: the two outlined are those whose response correlates best
with the hand-made filter's on the deck tile. Edges, textures, and shapes are common to all
photographs, one reason the features transfer. ImageNet's terms allow research and education.
:::

## Your turn: predict the check

```python
def conv2d(x, w, b):     # x (N,C,H,W), w (F,C,k,k), b (F,)
    k = w.shape[-1]
    win = sliding_window_view(x, (k, k), axis=(2, 3))
    # win (N, C, H-k+1, W-k+1, k, k): every k x k window
    y = np.einsum("nchwij,fcij->nfhw", win, w)
    return y + b[None, :, None, None], win

def conv2d_back(g, w, win):   # g = dL/dy, (N, F, H', W')
    dw = np.einsum("nchwij,nfhw->fcij", win, g)
    db = g.sum(axis=(0, 2, 3))
    dx = ...  # each pixel collects every window that read it
    return dx, dw, db
```

**Predict:** against PyTorch's conv2d, how many digits agree? And with the filter flipped?

::: notes
sliding_window_view lists every k x k window without copying the data; one einsum multiplies
each window by each filter and sums over channel and position in the window. The weight gradient
is the same windows multiplied by the output gradient and summed over images and positions; the
input gradient (in the notebook) sends each output's gradient back through the nine taps that
read it. The textbook convolution flips the filter first, w[::-1, ::-1]; machine learning drops
the flip (Forsyth §17.1.1 too). Take a show of hands for each question.
:::

## Agrees with PyTorch, unflipped

| Check (float64, 4 × 3 × 32 × 32, eight 3 × 3 filters) | Largest difference |
|:-------------------------------------------------------|-------------------:|
| Output vs torch.nn.functional.conv2d | 1 × 10⁻¹⁴ |
| Gradient for the input | 1 × 10⁻¹⁴ |
| Gradient for the filters | 1 × 10⁻¹³ |
| Gradient for the biases | 2 × 10⁻¹⁴ |
| Output with the filter flipped | 37 |

::: notes
Same products in a different order: round-off only. Flipped, the result is a different operation
(true convolution) and misses by 36.7, against outputs with a standard deviation of
5.5. For a
learned filter the flip does not matter, since training would learn the flipped weights; for
loading someone else's weights, it does. Translation equivariance holds exactly: shifting the
deck tile 32 pixels shifts the filter's map by 32 pixels, difference 0 away from
the borders.
:::

## A line separates cracks in features

![Two scatter plots of 600 test tiles, half cracked, placed by a logistic regression's score, left to right, and the first principal component of the representation, up. Left, pixels, 64 by 64 grey levels: cracked and uncracked tiles overlap completely, AUC 0.54. Right, ResNet-18's 512 features, the same regression on the same training tiles: the cracked tiles spread to the right of the uncracked cluster, AUC 0.83.](cee_l15_convolutional_networks/space.png)

::: notes
Transfer learning, the simplest kind: freeze ImageNet's ResNet-18, take each tile's 512 pooled
features, and fit Lecture 6's logistic regression with an L2 penalty, its strength chosen on the
validation photos (C = 0.03 with every training tile). Here both regressions are fitted on the
same 6,000 training tiles, C chosen on the same 2,000 validation tiles (0.001 for pixels, 0.01
for features, each inside the values tried), and scored on every test tile. On raw pixels it
barely beats chance: AUC 0.54. On the features, 0.83. The network bent the space so that a
plane does most of the work. Up: the first principal component, the direction of largest
spread. For pixels it is the tile's brightness (r 1.00 with the mean grey level) and follows the
label at AUC 0.56; for features it follows the label a little more (AUC 0.62), and surfaces
explain 14% of its variance.
:::

## Trained long enough, scratch catches up

![Two panels of validation average precision. Left, all 137 training photos: the small CNN trained from scratch climbs from 0.34 after one pass to 0.74 after 12, still rising, reaching the line of ImageNet's ResNet-18 with a logistic regression, 0.74. Right, against the number of labelled training photos, 9 to 137: the probe rises from 0.55 to 0.74, and on 128-pixel tiles from 0.56 to 0.74. With 16 photos the network from scratch scores 0.51 after 33 passes (a hollow square) and 0.64 after 132 passes, the length chosen on validation, its three seeds 0.62 to 0.64, just above the probe's 0.61; with all 137, 0.74.](cee_l15_convolutional_networks/transfer.png)

::: notes
From scratch: 97,809 weights learned from a random start, on 128-pixel tiles, flips drawn per
batch. With all 137 training photos (33,418 tiles) 12 passes, about 27 minutes on 14 CPU cores,
reach the probe's 0.74, and validation AP is still rising (0.735, then 0.740; 12 passes were set
after a 4-pass pilot, and longer was not tried; one seed). With 16 photos (3,907 tiles) the first
recipe stopped at 2,013 steps (33 passes): 0.50 to 0.54 over three seeds, against the probe's
0.61. Had it finished? We chose the schedule's length on validation, the median of three seeds:
16 passes 0.46, 33 passes 0.51, 66 passes 0.61, 132 passes 0.64, 264 passes 0.62. At 132 passes
(8,052 steps, about 29 minutes a seed) it reaches 0.62 to 0.64, above the probe. The probe
on tiles shrunk to 128 pixels (box filter; the network's tiles are shrunk bilinearly), the
network's resolution, scores 0.61 too. Both pick their settings on the same 47 validation photos.
With 16 photos the frozen features gave their score after seconds of fitting; the network needed
a search over five lengths, 5.7 hours on 14 cores. Curves on validation photos; the test photos
are scored once, on the next slide.
:::

## Tested once: the gap closes

| Test photos, average precision | 16 training photos | 137 training photos |
|:-------------------------------|------------------:|-------------------:|
| Flag at random | 0.14 | 0.14 |
| Hand filter + logistic | — | 0.27 |
| Small CNN, first recipe | 0.42–0.46 | 0.64 |
| Small CNN, length chosen on validation | 0.58–0.60 | — |
| ImageNet ResNet-18 + logistic | 0.56 | 0.67 |
| Same, tiles shrunk to 128 px | 0.54 | 0.66 |

::: notes
Average precision summarizes the precision-recall curve; flagging at random scores the share of
cracked tiles. The hand filter's three summaries (share of line-like pixels, mean, 99th
percentile) under the same logistic regression: the designed feature, the naive baseline. Each
method's threshold was set on the validation photos to flag 90% of their cracks; on the test
photos the probe flags 86% of cracks and 51% of tiles, the network from scratch 88% and
47%, the 128-pixel probe 86% and 48%, the hand filter 91% and 84%. So with all 137 photos the
network trained from scratch is as good at the inspector's threshold (one seed, 12 passes). With
16 photos, three seeds each: after the first recipe's 2,013 steps the network's test AP runs 0.42
to 0.46; at the length chosen on validation, 0.58 to 0.60, above the probe's 0.56 in every
seed, and it flags 84% to 87% of cracks for 51% to 55% of tiles. Train each model to its best
before comparing: at 2,013 steps this table said transfer wins. The probe costs one pass of
ResNet-18 over every tile (about 11 minutes on 14 CPU cores, longer on a laptop) and seconds of
fitting. With a GPU, fine-tuning ResNet-18 itself is the usual next step (the notebook's
exercise).
:::

## Accuracy hides the misses

:::::: {.columns}
::: {.column}
- Flag nothing: 86% accurate.
- The probe at p ≥ 0.5: 91%.
- Ask for recall and the review share.
:::
::: {.column}
![Precision against recall on the test photos of each surface, decks, walls, and pavements, each falling from near 1 toward the share of cracked tiles, shown as dotted lines. A marker on each curve shows the operating point at the threshold set on validation photos.](cee_l15_convolutional_networks/pr.png)
:::
::::::

::: notes
14.4% of the test tiles are cracked, so a classifier that never flags scores 85.6% accuracy.
At p >= 0.5 the probe scores 90.6% and flags half of the cracks (50%). Precision is the share of flags
that are cracked; recall, the share of cracks flagged. The dotted lines are each surface's share
of cracked tiles, the precision of flagging at random. The markers: the threshold the inspector's
target set (0.057), far below 0.5.
:::

## Predict before we test

1. **Siblings.** Train also on half the tiles of every test photo, and score the other half.
   How much does average precision rise?
2. **Another surface.** Train on decks only. What AP on pavements?
3. **Fly higher.** Half, then a quarter, of the pixels. What share of cracks is still flagged?

::: notes
One minute; students commit in writing. The probe scores AP 0.65 on the held-out half of the test
tiles with no test photo in training, and 0.65, 0.74, 0.59 on decks, walls, and pavements.
Most students predict a large jump for (1): that is the critique of random tile splits. The next
slides reveal (1), then (2), the decisive test, then (3).
:::

## Siblings barely help a frozen probe

| Same 5,580 held-out test tiles | Training photos | + siblings | + other tiles |
|:-------------------------------|----------:|----------:|----------:|
| Probe (ResNet-18 + logistic), AP | 0.655 | 0.676 | 0.657–0.660 |
| 5 nearest neighbours, AP | 0.464 | 0.510 | 0.473 |
| Neighbouring tiles' labels, AUC | — | 0.86 | — |

::: notes
Siblings: the other half of each test photo's tiles (5,628), added to the training tiles. The
control adds as many tiles of the validation photos instead (three draws): more data alone. The
probe gains 0.022 AP from siblings and 0.003 to 0.005 from other tiles, so the leak proper is
about 0.018; 5 nearest neighbours gain 0.046 against a median 0.009, a leak of about 0.037. The neighbours' labels,
though, predict a tile with AUC 0.86, because cracks run across tiles (look at the photos slide).
Anything that can see a tile's neighbours, a segmentation network (Lecture 16) or a model
fine-tuned on tiles cut with overlap, would gain from a random split. Measure the leak for every
model; never assume its size. Even photo splits leak a little: for 14 of the 46 test photos,
the most similar training photo (mean features) is the one taken just before or after, possibly
of the same slab. Holding out whole structures would be stricter; SDNET2018 does not record them.
:::

## Decks don't teach pavements

:::::: {.columns}
::: {.column}
- Trained on decks: AP 0.37 on pavements.
- Trained on pavements: 0.62.
- New surface, new camera: re-test.
:::
::: {.column}
![A 3 by 3 grid of average precision on test photos, rows the surface trained on, columns the surface tested on. Diagonal: deck 0.65, wall 0.75, pavement 0.62. Trained on decks: walls 0.65, pavements 0.37. Trained on walls: decks 0.59, pavements 0.38. Trained on pavements: decks 0.52, walls 0.61.](cee_l15_convolutional_networks/cross_surface.png)
:::
::::::

::: notes
The lecture's decisive figure. Each row: a probe trained on one surface's training photos only,
C chosen on that surface's validation photos, then tested on each surface's test photos. Decks
and walls are smooth concrete, and partly transfer to each other; pavements are coarse
aggregate in sun and shadow. At its own deck threshold, the deck-trained probe flags
15% of the clean pavement tiles and 43% of the cracked ones; the pavement-trained probe, 47% and
88%. Compare the share of cracked pavement tiles, 0.12, the AP of flagging at random. Every
new bridge, camera, or light is a new test.
:::

## Fly higher, and cracks slip through

![Left, one cracked test tile, a pavement with a wide dark crack, at 256, 128, and 64 pixels across, its probability of a crack falling from 1.00 to 0.97 to 0.01. Right, at the same threshold, the share of cracked test tiles flagged falls from 86% at 256 pixels to 84% at 128 and 77% at 64; the share of clean tiles flagged falls from 44% to 39% and 31%.](cee_l15_convolutional_networks/resolution.png)

::: notes
A drone flying twice as high puts half as many pixels on each millimetre. Here every cracked test
tile (1,611) and as many clean ones were shrunk to 128 and 64 pixels, then scored by the
unchanged probe at the unchanged threshold. At 64 pixels, 211 cracked tiles flagged at full
resolution are missed. The tile shown is the one whose probability falls most, and its crack is
still plain to us at 64 pixels. The ranking barely moves (AUC 0.84, 0.85, 0.82 at 256, 128, 64
pixels) while both shares fall: mostly every score shifts below a threshold set at another
scale. Fly at the altitude the training photos were taken from, or train on tiles at the scale
you will fly: the probe fitted on 128-pixel tiles scores test AP 0.66, against 0.67 at 256.
Lecture 16 measures widths, and needs the camera's millimetres per pixel.
:::

## The attempt, revealed

![The eight tiles of the attempt with their labels and the network's probability of a crack. Solid frames: the network's call at p 0.5 agrees with the label; dashed: it does not. A, a pavement edge in shadow: uncracked, p 0.00. B, a wall with a crack: cracked, 1.00. C, pavement with faint lines: cracked, 0.01, missed. D, a plain deck tile: uncracked, 0.01. E, a wall with small pits: uncracked, 0.68, a false alarm. F, a deck with a branching crack: cracked, 1.00. G, a plain-looking deck tile: cracked, 0.04, missed. H, coarse pavement with debris: uncracked, 0.94, a false alarm.](cee_l15_convolutional_networks/attempt_reveal.png)

::: notes
How did you do? Compare with the network, and with the label. The eight were picked by rank, not by hand:
two sure catches, two sure passes, two false alarms, and two misses, each pair from different
surfaces (the probe's p at fixed percentiles of the test tiles of that surface and label), then
shuffled. G is labelled cracked: look closely,
is there a crack? The network says no. H shows why pavements are hard. Where the
network and the label disagree, either can be wrong; the next slides come back to that.
:::

## Where the network looks

![The four attempt tiles the network flags, with the drop in the log-odds of a crack when each 48-pixel square is greyed out, shaded orange. B, a wall crack: the orange follows the crack; largest drop 4.2. E, a wall flagged at 0.68: faint, scattered patches; largest drop 0.9. F, a deck crack: orange on the crack's upper branch and right end; largest drop 5.8. H, coarse pavement flagged at 0.94: orange on a dark elongated pit near the top; largest drop 5.1.](cee_l15_convolutional_networks/occlusion.png)

::: notes
Occlusion (Zeiler & Fergus, 2014): grey out one 48-pixel square at a time, every 16 pixels, and
record how much the log-odds of "cracked" fall; orange marks the squares that mattered. Log-odds,
because near p = 1 a probability cannot show a fall. On the catches, B and F, the squares that
matter lie on the crack. On the false alarm H, they lie on a dark elongated pit: a crack-like
shape. On E, no single square carries the call (largest drop 0.9); the network is unsure. Occlusion
shows where, not why, and a grey square is an image the network never saw in training.
:::

## Your turn: same pixels, two labels

:::::: {.columns}
::: {.column}
- These tiles appear twice, pixel for pixel.
- Each copy is filed under a different label.
- What does that do to a 90% target?
:::
::: {.column}
![Two tiles from SDNET2018, D7039-112, a deck, and W7074-105, a wall, each stored twice with identical pixels, once labelled cracked and once uncracked. The network gives them 0.14 and 0.34.](cee_l15_convolutional_networks/twins.png)
:::
::::::

::: notes
Two minutes with a neighbour: is there a crack? Hashing every tile's decoded pixels finds 13 tiles
stored twice as identical files, each pair within one photo; 2 of them carry both labels.
Labels are judgments, made tile by tile. Thirteen repeats are too few to estimate how often the
labels are wrong, but they show that it happens: no classifier can agree with both copies. When
the network and the label disagree on a faint line, the label is not always right, and a
reviewer's call is the reference that matters.
:::

## One deck: review 160 tiles, miss 17

![Test deck photo D7049 rebuilt from its 252 tiles. Orange squares: the 160 tiles the network flags at the inspector's threshold. Blue dots: the 73 tiles labelled cracked; white crosses: the 17 of them the network misses, most in one band across the middle of the deck.](cee_l15_convolutional_networks/decide.png)

::: notes
The threshold, set on the validation photos to flag 90% of their cracked tiles, applied to one
test photo, the test deck with the most cracked tiles: 160 of 252 tiles to review (63%), 56 of
73 cracked tiles caught (77%), short of the 90% target. Without the screen, the inspector
looks at all 252. Ask: is a third of the review saved worth 17 cracks unrecorded until
the next inspection? On this deck, the honest answer may be no.
:::

## What the threshold buys, by surface

| Test photos | Cracked tiles | Cracks flagged | Tiles to review |
|:------------|--------------:|---------------:|----------------:|
| Decks | 17% | 91% | 66% |
| Walls | 16% | 92% | 66% |
| Pavements | 12% | 77% | 31% |
| Five photo splits, all surfaces | — | 86–91% | 43–55% |

::: notes
One threshold (0.057) for all surfaces, set on the validation photos. The promise was 90%; the
test photos give 86% overall (resampling the 46 test photos: 83% to 89%, and under 1% of
resamples reach 90%). Across five random photo splits (the lecture's is the first), recall at
each split's own validation threshold ranges 86–91% and the review share 43–55%; test AP ranges
0.67–0.75; the lecture's split is the hardest of the five. With 46 test photos, the split moves
test AP by 0.08, more than the 0.03 between the probe and the network trained from scratch on all
137 photos. Re-set the threshold on the owner's photos, and audit a random sample of unflagged tiles.
:::

## When a tile classifier is the wrong tool

| Situation | Better choice | Seen today |
|:----------|:--------------|:-----------|
| The crack's width or length matters | segmentation, calibrated camera (L16) | tiles say yes or no |
| A new surface, camera, or altitude | re-test, re-set the threshold | decks → pavements |
| Few labelled photos of your own | frozen features first | 0.56 AP in seconds; CNN 0.58–0.60 in 5.7 h |
| A miss is safety-critical | inspect all, screen to prioritize | targets slip off-sample |

::: notes
The tile classifier answers "is there a crack in these 256 pixels", which is not what a load
rating or a maintenance contract asks: they need width, length, and location, in millimetres
(Lecture 16). It is tied to its training photos: today's threshold promised 90% and delivered
86% on the test photos, and much less on a surface it had not seen.
:::

## Who signs for a missed crack?

| | Today | With a screen |
|:----------------|:-------------------------|:------------------------------|
| Who looks at a tile | the inspector, every tile | the inspector, flagged tiles |
| A missed crack | the inspector's miss | the screen's, or the inspector's? |
| What is recorded | the inspection report | also: the tiles never reviewed |
| Skills that grow | spotting | auditing the screen, re-setting it |

::: notes
The course's Unit 3 question: automating inspection changes the inspector's role. A screen that
lets an inspector skip a third of a deck shifts attention, and a crack in the unflagged third was
never looked at by anyone. Whose miss is it: the inspector's, who signed the report, the
owner's, who bought the screen, or the vendor's? Write down which tiles a person reviewed, and
audit a random sample of the rest. Ask: who pays for the audit, and who is told when the camera
changes? The skills that grow: judging a screen's evidence, re-setting its threshold, auditing it.
:::

## Vote: what goes on the next inspection?

The owner wants a screen on next season's deck photos.

- **A.** Deploy today's probe as is: it promised 90% of cracks.
- **B.** Deploy it, with the threshold re-set on the owner's photos and a random sample of unflagged tiles reviewed.
- **C.** First train a network from scratch on SDNET2018.
- **D.** Don't deploy: inspect every tile.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
B. A's promise was made on SDNET2018's validation photos; on its test photos it delivered
86%, and across five photo splits 86–91%; on another owner's camera and decks, no one knows until
someone measures. The random sample of unflagged tiles is that measurement, in service. C,
trained long enough, matched the probe even with 16 photos (test AP 0.58 to 0.60 against 0.56),
after a search over schedule lengths that took hours, and its threshold would still need the
owner's photos. D is the baseline, and the right answer if no one will audit the misses. Then ask what would change the answer: pavements instead of decks (re-test first), or a
drone flying twice as high (fewer pixels: recall 84% at 128 pixels).
:::

## CNNs live inside today's vision models

| Today's system | The convolutional part | Lecture |
|:---------------|:-----------------------|:--------|
| U-Net segmentation, crack widths | an encoder of convolutions, a decoder back to pixels | 16 |
| Vision transformers | patches replace sliding filters | 17 |
| Foundation models (CLIP, SAM) | frozen features, as ResNet-18's today | 18 |
| 1-D sensor and runoff models | the same filters, along time | 20 |

::: notes
Residual connections are in nearly every deep network trained today, transformers included
(Vaswani et al., 2017). Frozen features plus a small head, today's recipe, is how foundation models are used
on new tasks (Radford et al., 2021; Kirillov et al., 2023); Lecture 18 compares CLIP's features
with ResNet-18's on these tiles, and uses SAM. Detection networks (Faster R-CNN, YOLO; Forsyth §18.2) put boxes around
defects with a ResNet backbone.
:::

## Exit question

A vendor reports 98% accuracy for a crack classifier, tested on a random 20% of the tiles cut
from its 300 photos. Name two things you would ask before believing it, and the number you would
want instead of accuracy.

**In the notebook** (`cee_l15_convolutional_networks.ipynb`): the convolution and its gradients
from scratch, checked against PyTorch; the network from scratch against ImageNet's features; and
the break-it tests.

::: notes
Answer: accuracy first: with 15% of tiles cracked, flagging nothing scores 85%; ask for recall
and the share of tiles to review at a threshold fixed before the test. Then the split: a random
20% of tiles puts every test photo in training; ask for whole photos, or whole structures, held
out, and results per surface and camera. Collect the answers; they seed the next warm-up.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- One small filter, everywhere: few weights.
- Stacks see farther; skips keep gradients sane.
- Few labels? Frozen features are the fast start.
- Train each model fully before comparing.
- New surface, new camera: re-test.

**Next:** segmentation, and cracks in millimetres.
:::
::: {.column}
![Four thumbnails of the lecture's key figures: the 3 by 3 filter's arithmetic and its response along a crack; the receptive fields of ResNet-18's four stages; validation average precision of the network from scratch and the frozen ResNet-18, by pass and by number of labelled photos; the decision on a test deck photo, flagged tiles and missed cracks.](cee_l15_convolutional_networks/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Balduzzi et al. (2017). *ICML*.
- Deng et al. (2009). *CVPR* (ImageNet).
- Dorafshan, Thomas & Maguire (2018). *Data in Brief* 21.
- He et al. (2016). *CVPR*.
- Ioffe & Szegedy (2015). *ICML*.
- Kirillov et al. (2023). *ICCV*.
- Luo et al. (2016). *NeurIPS*.
:::
::: {.column}
- Maguire, Dorafshan & Thomas (2018). SDNET2018, Utah State University.
- Radford et al. (2021). *ICML*.
- Smith & Topin (2019). *SPIE*.
- Vaswani et al. (2017). *NeurIPS*.
- Yang et al. (2019). *ICLR*.
- Zeiler & Fergus (2014). *ECCV*.
:::
::::::
