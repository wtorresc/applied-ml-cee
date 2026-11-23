---
title: Segmentation and Measuring
subtitle: "Lecture 16 · How wide is the crack, and which roads are under water?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 6–15

On your own, no notes, two minutes:

1. Lecture 6: logistic regression gives one row a probability p of class 1. What is the row's
   loss if its label is 1?
2. Lecture 15: a convolution with 16 filters of 3 × 3 on a one-channel image. How many weights,
   biases included?
3. Lecture 15: ResNet-18's last stage sees 435 pixels but keeps a 7 × 7 map. Can it say
   *where* in the tile the crack runs?

**Today:** a label for each of 319,488 pixels, then millimetres.

::: notes
Give the answers before moving on.
1. −log p (cross-entropy). Today the same loss is averaged over every pixel of an image: a U-Net
   is Lecture 6's logistic regression at each pixel, on features a network learns.
2. 16 × 9 = 144 weights plus 16 biases: 160. Today's network has 1,942,289.
3. Only roughly: one cell of the 7 × 7 map covers 32 × 32 pixels of the network's 224-pixel
   input, about 37 × 37 of the 256-pixel tile. The classifier knew *what*; today's decoder
   recovers *where*.
:::

## The rating needs width, not a label

:::::: {.columns}
::: {.column}
| LTPP severity (asphalt) | Mean crack width |
|:------------------------|-----------------:|
| Low | ≤ 6 mm |
| Moderate | 6–19 mm |
| High | > 19 mm |

Record **metres** of cracking at each level.
:::
::: {.column}
![A range image of a pavement with its crack branches drawn in three colours by severity: low in green, moderate in blue, high in orange; the legend gives the metres of cracking at each level in this image.](cee_l16_segmentation/ltpp.png)
:::
::::::

::: notes
The LTPP Distress Identification Manual (FHWA-HRT-13-092, 2014), chapter 1: for longitudinal
(ACP 4) and transverse (ACP 6) cracking in asphalt, severity is set by the crack's mean width:
low up to 6 mm, moderate above 6 and up to 19 mm, high above 19 mm (cracks with adjacent random
cracking move up a level, left out here). The rater records the length at each severity, which
feeds condition indices and the choice between sealing, patching, and resurfacing. Lecture 15's
classifier said "this tile has a crack". The agency needs where the crack runs (its length) and
how wide it is, in millimetres. This image (sample_000609) holds 2.0 m of low,
11.3 m of moderate, and 1.3 m of high-severity cracking by
the range image's widths. Ask: which of these numbers does a pixel mask give you directly?
:::

## 3DCrack: depth images, 4 mm a pixel

![Four range images of Georgia pavements, 512 by 624 pixels, with the annotated cracks in green: a transverse crack, 0.5% of the pixels; a longitudinal crack, 0.3%; alligator cracking in the wheel path, 4.3%; and a crack-free lane edge with a dark joint, 0%.](cee_l16_segmentation/samples.png)

::: notes
3DCrack (Zhang, Wang, Hsieh, Yang, Yezzi & Tsai, Georgia Tech, 2025; CC BY 4.0): a survey van's
3D line lasers scan a 4-m lane at highway speed; each pixel is the surface height, darker = lower,
after a high-pass filter that removes the road's slope. The scanner samples every 1 mm; the
release is downsampled to about 4 mm per pixel, 512 × 624 pixels (2.0 × 2.5 m). 1,139
training, 245 validation, 250 test images from SR 275, US 80,
and I-16, split by the authors; 53 test images have no crack. Masks were
drawn by trained annotators. Crack pixels are 0.74% of the training pixels.
Each cracked image also comes with centrelines and a width at every point, from the mask and
from the range image: two rulers, which come back later.
:::

## Your turn: cracks are dark, right?

Every crack is a depression, so it is darker than the road around it.

**Rule:** call a pixel crack when it is darker than a grey level chosen on the 245 validation
images.

**Predict:** of every 100 crack pixels in the test images, how many will the rule find, and
how many false ones will it add?

::: notes
Two minutes, alone, then with a neighbour. The grey level that maximizes F1 on validation is
96 (out of 255). Ask what else is dark in a range image: joints, lane-edge
grooves, potholes, the textured aggregate. Collect a few predictions; the next slide reveals.
:::

## Darkness finds joints too

![Three panels of one test image: the range image, with a dark lane-edge joint on the left and a branching crack on the right; the pixels darker than 96, which trace the joint as clearly as the crack; and the errors: the crack's middle found in green, the joint flagged in orange, the crack's faint arms missed in blue.](cee_l16_segmentation/attempt.png)

::: notes
On the 250 test images: recall 64%, precision 41%,
F1 0.50, IoU 0.33. Lecture 15's "darker than its eight
neighbours" filter, thresholded the same way, does worse: F1 0.31. It finds a quarter of the
crack pixels (recall 0.26, against 0.64) at the same precision (0.41): it loses on finding,
not on false alarms. A crack is not "dark"; it is a thin, connected, wandering dark line, which is a
statement about a neighbourhood the size of the crack's path, not about one pixel. This is the
naive baseline for the rest of the lecture.
:::

## A label per pixel: slide or share

![Left, a range image with one 64 by 64 window drawn around a crack pixel. Right, a bar chart on a log scale of multiply-adds per image: a window per pixel through a small network, 21,294 billion; the U-Net, one pass, 15 billion.](cee_l16_segmentation/sliding.png)

::: notes
The obvious way to label every pixel: Lecture 15's classifier on a window around each pixel,
319,488 windows per image. Neighbouring windows overlap almost entirely, so the
same convolutions are recomputed thousands of times: the windows cost 1,449 times the U-Net's
whole pass, decoder included, which computes each convolution once over the image and shares
it (Long, Shelhamer & Darrell, 2015, "fully convolutional"). Counted here with the U-Net's own encoder as the window network.
:::

## Shrink to see, grow back to point

![A U-Net's maps on a test image, arranged as a U. Down the left arm, five encoder levels from 624 by 512 with 16 channels to 39 by 32 with 256 channels, each showing the channel that best matches the crack; up the right arm, four decoder levels back to 624 by 512; dashed arrows carry each encoder level across to its decoder twin; at the top right, the predicted probability of crack.](cee_l16_segmentation/unet_maps.png)

::: notes
Each encoder level: two 3 × 3 convolutions, then a 2 × 2 max-pool that halves the map. By the
bottom (39 × 32 × 256) every cell sees a large part of the image and knows *what* is there
(crack, joint, texture) but only roughly *where*. The decoder doubles the map four times with
transposed convolutions (next slides), and at each level concatenates the encoder's map of the
same size: the skip connection carries the fine location the pooling threw away (Ronneberger,
Fischer & Brox, 2015). Shown: at each level, the channel most correlated with the true mask.
1,942,289 weights in all; the last layer is a 1 × 1 convolution giving a logit per pixel.
:::

## Skips carry the where

![The same 160-pixel crop of a test image decoded two ways, with found, false, and missed pixels: on the left, an encoder-decoder without skip connections, its crack thick and blurred, many false pixels beside the line and gaps in it; on the right, the U-Net, its crack thin and continuous.](cee_l16_segmentation/skips.png)

::: notes
Same encoder, decoder, data, seed, and 40 passes; the only difference is the concatenation of
the encoder's maps. Validation F1: 0.764 without skips, 0.830
with them; test F1 0.750 and 0.822. Without skips the decoder redraws the crack from a
39 × 32 map: it still finds it (within 2 pixels, F1 0.94 against 0.95) but draws it thick
(strict precision 0.72 against 0.84).
:::

## Assume: a pixel is its neighbourhood

:::::: {.columns}
::: {.column}
Every pixel gets its own logistic regression on learned features:

$$p_{ij} = \sigma\big(f_\theta(x)_{ij}\big)$$

$$\begin{aligned} L = -\frac{1}{HW}\sum_{i,j} \big[ & y_{ij}\log p_{ij} \\ & + (1-y_{ij})\log(1-p_{ij}) \big] \end{aligned}$$
:::
::: {.column}
| Training pixels | Share |
|:----------------|------:|
| Crack | 0.74% |
| Not crack | 99.26% |
| "Never crack" scores | accuracy 99.3%, F1 0 |
:::
::::::

::: notes
The assumption: whether a pixel is crack depends on the image within the network's receptive
field around it, and the same evidence means the same thing anywhere (Lecture 15). The loss is
Lecture 6's cross-entropy averaged over all 319,488 pixels of each image. Accuracy is useless
here: a network that never says crack is right on 99.3% of pixels. So we score
with F1 (also called Dice) and IoU over the crack pixels, and choose the threshold on validation,
as in Lecture 6.
:::

## Grow a map: stamp the kernel

![Three grids. The input, 2 by 2: 2, 4, 6, 0, with the first cell outlined. The kernel, 3 by 3, a tent 1, 2, 1 over 2, 4, 2 over 1, 2, 1, divided by 4, scaled by 2. The output, 5 by 5: the scaled kernel stamped at the top-left, the rest still zero.](cee_l16_segmentation/tconv_1.png)

::: notes
A transposed convolution with stride 2: each input value multiplies the whole kernel, and the
copy is added into the output at twice the input's position. The output grows from 2 × 2 to
(2 − 1) × 2 + 3 = 5 × 5. Ask: where does the next stamp go, and what happens where two stamps
overlap?
:::

## Grow a map: stamp the kernel

![The same three grids for the second input value, 4: its scaled kernel is stamped two columns to the right; where it overlaps the first stamp, the values add.](cee_l16_segmentation/tconv_2.png)

::: notes
Overlaps add. In the shared column, 2 × 1/4 + 4 × 1/4 = 1.5 at the top, and the centre row gets
2 × 2/4 + 4 × 2/4 = 3: the average of 2 and 4.
:::

## Grow a map: stamp the kernel

![The same three grids for the third input value, 6, stamped two rows down.](cee_l16_segmentation/tconv_3.png)

::: notes
Every new value between two inputs is their average; at an input's own position, the kernel's
centre weight 4/4 copies it.
:::

## Grow a map: stamp the kernel

![The finished output, 5 by 5, after the fourth input value, 0: inside, each new pixel is the average of the inputs around it.](cee_l16_segmentation/tconv_4.png)

::: notes
With this tent kernel the transposed convolution is exactly bilinear upsampling (inside the
border). In a U-Net the kernel is learned (2 × 2 here, 16 to 256 channels), so the network can
do better than interpolate: it can sharpen an edge it knows is there.
:::

## Upsampling is convolution, run backward

![Two matrices of the same nine kernel weights. Left, C, 4 by 16: each row lays the 3 by 3 kernel along a flattened 4 by 4 image, so C maps an image to a 2 by 2 map. Right, C transposed, 16 by 4: each column spreads one map value back over the 4 by 4 image.](cee_l16_segmentation/matrix.png)

::: notes
A convolution is a linear map, so it is a (sparse) matrix C. Its transpose maps the small map
back to the image's size with the same weights: hence the name. And Cᵀ is exactly what
backpropagation multiplies the output gradient by to get the input gradient: the dx of Lecture
15's backward pass. A decoder layer runs that gradient computation forward, with its own learned
kernel. The notebook checks the identity (C transposed times a 2 × 2 map equals the stamps).
:::

## Your turn: predict the check

```python
def conv_transpose2d(x, w, stride):  # x (Cin,h,w), w (Cin,Cout,k,k)
    _, h, wd = x.shape
    k = w.shape[-1]
    out = np.zeros((w.shape[1], (h-1)*stride + k, (wd-1)*stride + k))
    for i in range(h):
        for j in range(wd):   # each input pixel stamps the kernel
            out[:, i*stride:i*stride+k, j*stride:j*stride+k] += \
                np.einsum("c,cokl->okl", x[:, i, j], w)
    return out
```

**Predict:** against PyTorch's `conv_transpose2d`, and against the gradient of `conv2d`?

::: notes
Two checks: the library's transposed convolution, and the backward pass of an ordinary strided
convolution with the same weights (autograd's gradient of ⟨conv2d(z, w), x⟩ with respect to z).
If the second one holds, "transposed convolution" and "the gradient of a convolution" are the
same operation. Show of hands: exact, about 10⁻⁶, or different?
:::

## Same numbers three ways

| Check (float64, 3 → 2 channels, 5 × 4 input, 4 × 4 kernel, stride 2) | Largest difference |
|:-------------------------------------------------------------|-------------------:|
| From scratch vs torch.nn.functional.conv_transpose2d | 3 × 10⁻¹⁵ |
| From scratch vs the gradient of conv2d (autograd) | 3 × 10⁻¹⁵ |
| Output size, (5 − 1) × 2 + 4 by (4 − 1) × 2 + 4 | 12 × 10 |

::: notes
Round-off only: the same products, summed in a different order. The output size formula,
(h − 1) s + k, is why the U-Net's input must be divisible by 16: four halvings and four doublings
must land on the same size (624 = 16 × 39, 512 = 16 × 32). The notebook also builds the U-Net
from these parts and checks every level's shape.
:::

## Dice: overlap you can differentiate

:::::: {.columns}
::: {.column}
$$\text{F1} = \text{Dice} = \frac{2\,TP}{2\,TP + FP + FN}$$

$$\text{IoU} = \frac{TP}{TP + FP + FN}$$

Soft Dice: replace counts by sums of $p_{ij}$.
:::
::: {.column}
![A crop of a test image with the U-Net's crack pixels: found in green, false in orange, missed in blue; beside it, bars of the three counts: 626 found, 45 false, 19 missed.](cee_l16_segmentation/dice.png)
:::
::::::

::: notes
On this crop: Dice = 2 × 626 / (2 × 626 + 45 + 19) =
0.95; IoU = 626 / (626 + 45 + 19) = 0.91.
They rank models identically (F1 = 2 IoU / (1 + IoU)). The soft Dice loss, 1 − 2 Σ p y / (Σ p +
Σ y), ignores the true negatives, so 99% background cannot dominate it (Milletari et al., 2016);
a common recipe for thin structures adds it to cross-entropy. Measured here, same everything
else: validation F1 0.830 with cross-entropy alone and 0.828 with
Dice added; on the test images the difference is +0.004 (95% interval −0.006 to +0.016,
resampling the same images). One seed, an interval across zero: no gain either way. With the
threshold chosen on validation, imbalance did not need a special loss here.
:::

## One pixel off costs 40% of IoU

![Scores of the true test masks against themselves moved sideways by 0 to 3 pixels. Strict IoU falls from 1 to 0.61 at one pixel and 0.38 at two; F1 counting pixels within 2 pixels as hits stays at 1 until the shift exceeds 2 pixels. A dotted line marks the IoU of two annotators on the same image, about 0.7.](cee_l16_segmentation/shift.png)

::: notes
Most of a crack mask is one or two pixels wide (54% of the centreline points by the mask
ruler). Move a perfect mask one pixel (4 mm) and strict IoU
drops to 0.61; two pixels, 0.38. For thin structures, strict
overlap measures where the annotator's pen landed as much as whether the crack was found. Crack
benchmarks therefore also report F1 with a tolerance: a predicted pixel counts if a true pixel is
within 2 pixels (8 mm), and vice versa. We report both.
:::

## The same image, annotated twice

![Two zoomed crops, one from each image that appears in two of the dataset's splits with identical pixels: sample 130 in training and validation, sample 131 in training and test. The two annotations are overlaid: where they agree in green, pixels only in one in orange or blue; the lines run side by side, a pixel apart.](cee_l16_segmentation/twins.png)

::: notes
Two images were published twice, with identical pixels and different masks: the same scan,
annotated twice (found by hashing the pixels, not from the documentation). Strict IoU between
the two annotations: 0.70 and 0.69 (F1 0.83 and
0.81); F1 within 1 pixel: 0.98 and 0.97. So on these two images, strict IoU near 0.70 is
what two annotations of one scan agree on: a scale for any model's IoU, not a hard ceiling
(two images are a small sample, and a model can match each annotator better than they match
each other). The paper's best U-Net reports IoU 0.703 (Zhang et al., 2025, Table 4). Sample 131 is also in both training and test:
a one-image leak. Leaving it out changes the U-Net's test F1 by −0.0001. Sample 130 is in
training and validation, so one validation image was seen in training.
:::

## A U-Net fits on one slide

```python
class UNet(nn.Module):          # block = two 3 × 3 convs, each + BN + ReLU
    def __init__(self, ch=(16, 32, 64, 128, 256)):
        super().__init__()
        ab = list(zip(ch, ch[1:]))[::-1]      # (128, 256), ..., (16, 32)
        self.down = ModuleList(block(a, b) for a, b in zip((1, *ch), ch))
        self.up = ModuleList(ConvTranspose2d(b, a, 2, 2) for a, b in ab)
        self.dec = ModuleList(block(2 * a, a) for a, b in ab)
        self.head = Conv2d(ch[0], 1, 1)
    def forward(self, x):
        feats = []
        for i, b in enumerate(self.down):               # halve, deepen
            x = b(x if i == 0 else max_pool2d(x, 2)); feats.append(x)
        x = feats.pop()
        for up, dec in zip(self.up, self.dec):          # double, sharpen
            x = dec(torch.cat([feats.pop(), up(x)], 1)) # skip: the where
        return self.head(x)                             # logit per pixel
```

::: notes
block = two 3 × 3 convolutions, each with batch normalization and a ReLU (Lecture 15).
`ConvTranspose2d(b, a, 2, 2)`: b channels in, a out, a 2 × 2 kernel at stride 2. The
network without skips feeds `dec` the upsampled map alone (`block(a, a)`).
1,942,289 weights. Training: AdamW, a one-cycle learning rate peaking at 2 × 10⁻³,
batches of 8 images, random flips across and along the lane, 40 passes, bf16 on the GPU. The
notebook asserts every level's shape.
:::

## Under eight minutes on one GPU

![Two panels by pass over the 1,139 training images. Left, the U-Net's training loss falling and levelling off. Right, validation F1 at the best threshold for the U-Net, the U-Net with Dice added, and the network without skips: the two U-Nets level off near 0.83, the network without skips lower, near 0.76.](cee_l16_segmentation/curves.png)

::: notes
7.6 minutes for 40 passes on an RTX 3090 Ti; on four CPU threads, roughly half a day per
network (the notebook's header gives the measurement), so the notebook's default setting trains
on crops for a few passes instead. Validation F1 rises fast for 15 passes (0.790) and levels off
as the one-cycle learning rate falls to zero at pass 40 (0.824 at pass 30; 0.830 at 40). A
longer schedule was not tried. Threshold chosen on validation after the last pass, inside the
grid tried: 0.40.
:::

## Typical test images

![Three test images at the 25th, 50th, and 75th percentile of the U-Net's per-image F1 within 2 pixels, with found pixels in green, false in orange, missed in blue: most of each crack is green; misses are faint branch ends; false pixels sit on joints and dark aggregate.](cee_l16_segmentation/overlays.png)

::: notes
Chosen by rank, not by eye: the images at the quartiles of per-image F1 within 2 pixels
(0.92, 0.96, 0.98).
Ask: which misses would change the LTPP rating? A faint branch end adds length at low severity;
a missed wide crack would drop a moderate or high rating.
:::

## Strict overlap at annotator level

| Test images (250) | F1 | F1, 95% interval | IoU | F1 within 2 px |
|:-----------------|---:|----------------:|----:|---------------:|
| Darker than 96 | 0.50 | 0.46–0.54 | 0.33 | 0.58 |
| No skips | 0.75 | 0.73–0.77 | 0.60 | 0.94 |
| U-Net + Dice | 0.82 | 0.80–0.84 | 0.69 | 0.94 |
| **U-Net** | **0.82** | **0.80–0.84** | **0.70** | **0.95** |
| Two annotators, two images | 0.82 | — | 0.69 | 0.98 |

::: notes
Bracketed: the naive threshold above, the annotators' agreement below (the mean of the two twin
images: a scale, not a ceiling). Strict F1 and IoU are at the annotators' level; within 2 pixels
the two annotations agree more closely (0.98) than the U-Net matches them (0.95). On the twin
test image itself (sample 131, also in training with the other mask), the U-Net scores strict F1 0.87 against its
test mask and 0.84 against the training copy's mask (within 2 pixels,
0.96 and 0.96); the two masks agree with each
other at 0.81. The intervals resample the 250 test images 2,000 times; they cover the choice of
test images, not of roads (the release has no road field), and not the training seed (one seed
per network). The U-Net's IoU interval is 0.67–0.72. The paper's ResUNet reports IoU 0.703 with its own recipe. Within 2 pixels the U-Net finds 93% of crack pixels with
96% of its pixels on a crack. Strict precision 0.84,
recall 0.81.
:::

## Width: twice the distance to the edge

![Width w = (2d - 1) x 4 mm, where d is the distance to the nearest non-crack pixel; a one-pixel line, d = 1, is 4 mm wide. Left, a 40-pixel square crop of the widest crack in a test image with its mask outlined. Right, the distance transform of the mask as a 3D surface: a roof over the crack whose ridge rises to 2.2 pixels at the widest point.](cee_l16_segmentation/edt.png)

::: notes
The Euclidean distance transform gives every crack pixel its distance to the nearest
background pixel. On a straight band w pixels wide the centre has d = ⌈w/2⌉, so 2d − 1 is
exact for odd widths and one pixel short for even ones: a one- and a two-pixel line both read
4 mm. At the widest point of this crop, d = 2.24, so w = 13.9 mm. Checked against the
dataset's own `width_mm` at 213,065 test centreline points: identical at
99.9% of them. Our ruler is the dataset's mask ruler.
:::

## A pixel is 4 mm: widths come in steps

![Histograms of crack width at 213,065 test centreline points. The mask ruler's widths fall on a few spikes, 4.0, 7.3, 12.0, and 13.9 mm, with most points at 4.0 mm; the range image's widths spread smoothly, 90% of them from 2.5 to 27.5 mm, centred near 10 mm. Dotted lines mark the LTPP limits at 6 and 19 mm.](cee_l16_segmentation/widths.png)

::: notes
On a 4-mm grid the mask ruler's widths come in steps: 4.0 mm (54% of points) is a line one
or two pixels wide, 12.0 mm one three or four wide, and 7.3 and
13.9 mm are diagonal distances (d = √2 and √5). LTPP's 6 mm limit falls between the
first two steps: "low" means "the annotator drew a line one or two pixels wide". The dataset's second ruler,
measured on the range image (method not documented), is continuous: median
10.5 mm against 4.0 mm; correlation between the two,
0.41.
:::

## Two rulers, two severities

:::::: {.columns}
::: {.column}
![A 3 by 3 table of metres of test cracking, severity by the mask ruler (rows) against severity by the range-image ruler (columns). Most of the length rated low by the mask is moderate by the range image.](cee_l16_segmentation/rulers.png)
:::
::: {.column}
| Metres | Low | Mod. | High |
|:-------|----:|-----:|-----:|
| Mask ruler | 427 | 542 | 45 |
| Range ruler | 63 | 819 | 131 |
| True, ridge | 411 | 556 | 47 |
| U-Net, ridge | 360 | 540 | 43 |
| True, ridge, found | 352 | 546 | 44 |
:::
::::::

::: notes
Per branch: the mean width over its points, then the LTPP band; metres summed over the 250 test
images; "True, ridge" and "U-Net, ridge" are the true and the U-Net's mask read with the ridge
ruler (below). LTPP's width rule is for longitudinal and transverse cracks; 3DCrack does not label
crack types, so alligator cracking (rated by area in LTPP) is treated as linear too. The two
rulers put the same branch in the same band for 55% of the length. A predicted mask has its
own centreline, so both masks are read with one ridge ruler: d at the mask's skeleton pixel
nearest each true centreline point. On the true mask it matches the dataset's widths at
89% of the points; the other 11% is the ruler's own disagreement. The "U-Net, ridge" row leaves
out 71 m on branches the U-Net mostly missed, so it and "True, ridge" cover different branches.
Like for like, on the branches it found (the last row), the true mask puts 352 m at low and the
U-Net 360 m: its shortfall against 411 m is the 58 m of low cracking on branches it missed, not
its widths. 90% of
the points have a predicted crack within a pixel; there the U-Net's width equals the true
mask's at 66%, is a step wider at 16% and narrower at 18%.
:::

## Millimetres need distance and lens

![Left, a pinhole camera: sensor, lens at focal length f, and the surface at distance Z; one sensor pixel of pitch p covers Z p over f on the surface. Right, millimetres per pixel on a log scale: 0.1 mm to see a 0.3 mm crack at 3 pixels; a phone at 1 m, 0.33 mm; the 3D scanner's own sampling, 1 mm; a phone at 3 m, 1.0 mm; and the 3DCrack release, 4 mm.](cee_l16_segmentation/pinhole.png)

::: notes
Ground pixel = Z × p / f. Here the survey system fixes Z and the dataset states the result (about
4 mm). For a photo from a phone or a drone, the pixel size is unknown until someone measures Z
(or puts a target of known size in the frame) and calibrates the lens's distortion; without
that, a mask gives widths in pixels only. The example phone (f = 4.2 mm, 1.4 µm pixels) is
illustrative. Hüthwohl et al. (2019): to see a 0.3 mm crack (a common limit for reinforced
concrete) at 3 pixels per feature, the surface must be sampled at 0.1 mm per pixel, 40
times finer than this release. These pavement cracks are wider; a structural engineer's are not.
:::

## Your turn: predict each failure

Write one line for each, before the reveal:

1. Read the masks with `> 0` instead of `> 127`. Train, then score. What changes?
2. Survey at 8 mm per pixel instead of 4. Which cracks go missing, and what happens to LTPP's
   "low"?
3. Same U-Net, satellite pixels of 0.4 m, before and after a flood. What new problem appears
   when you *count* buildings?
4. The tiles' pixels are in degrees of longitude and latitude. Road length = pixels × 0.42 m?

::: notes
Collect predictions; each of the next slides reveals one. Keep the sheets for the exit question.
:::

## Read with > 0, both scores drop

![Three panels of training image sample_000797: the range image with a crack along its right side; its mask read with greater than 0, 98% of the pixels white; and read with greater than 127, 0.4% white, the crack alone.](cee_l16_segmentation/faint.png)

::: notes
The documentation says masks are 0 or 255; 233 of the 1,634 masks also hold grey
values, 99% of them within a pixel of a crack, and 4 masks (samples 797–800, three in training, one in
test) hold a faint value on almost every pixel. Trained on masks read with > 0 (grey edges count
as crack, three images are almost all "crack"): validation F1 0.781 against
0.830 (both on the clean > 127 validation masks, so this is the damage from the training labels
alone), test F1 0.77 against 0.82. And score the clean
U-Net against test masks read with > 0: F1 0.66, because the crack
share of the test set jumps from 0.81% to
1.21%, almost all from one image. There the model did not change;
the ruler did. Look at your labels' values, not their documentation.
:::

## At 8 mm, cracks found but not rated

![The same test crop at 4 and at 8 mm per pixel, each with its own U-Net's found, false, and missed pixels: at 8 mm the crack is still traced, in coarser steps, every line at least one 8 mm pixel wide.](cee_l16_segmentation/coarse.png)

::: notes
A second U-Net, same recipe, trained and tested on the images averaged over 2 × 2 blocks (a
block is crack if any of its pixels is): F1 0.85 on its own grid. Finding
survives: true centreline points with a predicted crack within one pixel of each model's grid,
hairline points (4 mm by the mask ruler: one or two pixels) 84% at 4 mm and
91% at 8 mm (an 8 mm pixel is a wider tolerance, so not like for like);
wider points 97% and 98%. Rating does not: the
narrowest crack an 8 mm mask can draw is 8 mm wide, already past LTPP's
6 mm, so with the mask ruler nothing can be rated low. By the range image, 62% of the
centreline points are wider than an 8 mm pixel (25% by the mask ruler): found, but not rated.
:::

## From cracks to a flooded valley

![Three panels of one test tile from SpaceNet 8 in Germany: the town before the July 2021 flood; the same town after, streets brown with mud and water; and the truth over the after image, buildings in yellow, flooded buildings in blue, roads in amber, flooded roads in pink.](cee_l16_segmentation/sn8_pair.png)

::: notes
SpaceNet 8 (Hänsch et al., 2022; CC BY-SA 4.0; Maxar imagery under a non-commercial licence):
pansharpened satellite tiles of the Ahr valley (from the tiles' coordinates, near Altenahr and
Dernau) before and after the July 2021 flood,
with OpenStreetMap building footprints and roads, each marked flooded or not. 134
+ 31 + 37 tiles in spatial blocks along the valley
(train, validation, test; the blocks touch, so a building on a block's edge can be cut into
tiles of two blocks); the test block holds 897 buildings,
374 flooded, and 11.3 km of flooded road.
The after images have other pixel sizes (some finer, some coarser); each is resampled onto
the before image's grid with both images' georeferencing.
:::

## One encoder reads before and after

![The same tile: truth on the left, the network's map on the right, in the same colours; flooded buildings and roads in blue and pink, dry ones in yellow and amber.](cee_l16_segmentation/sn8_pred.png)

::: notes
A siamese U-Net: ImageNet's ResNet-18 (Lecture 15) reads the before image and, with the same
weights, the after image; at each of its five levels the two maps are concatenated, and a U-Net
decoder doubles back to full size, 5 classes per pixel (the design of the SpaceNet 8 baseline,
smaller). Fine-tuned on random 512-pixel crops for 60 passes with cross-entropy plus Dice
(7 min). Roads are labelled as centrelines; they are drawn 12 pixels (3
to 5 m) wide, an assumption.
Test pixel F1: building 0.44, flooded building 0.28, road
0.43, flooded road 0.30. A flooded roof called dry counts against both building classes, so
apart: buildings found, flooded or not, pixel F1 0.62; on the building
pixels found, the flooded-or-dry call is right for 62%; ranked by
flood score, flooded against dry footprints, AUC 0.61. One seed. The footprints are
known before the flood (OpenStreetMap), so the responder's question is per building. Score each
footprint by its mean P(flooded) − P(dry), and set the threshold on the validation block to
catch 80% of flooded buildings: on the test block it catches
52% and flags 41% of all
897 buildings, against 42% truly flooded
(precision 53%). The threshold promised 80% on a block upstream
(75 flooded buildings, a different village) and delivered 52%: a target
set on one place must be re-checked on the next. Roads, by the larger class: 6.1 of
11.3 km of flooded road found; 5.7 km of dry road called flooded.
:::

## Detection: one box per building

![Three panels of a crop of the test town. Left, 22 overlapping proposal boxes from three thresholds of the building probability. Centre, the boxes kept by non-maximum suppression. Right, the true buildings' boxes, green where a kept box overlaps them with IoU at least 0.5, blue where none does.](cee_l16_segmentation/nms.png)

::: notes
Counting needs instances, not pixels: a detector outputs a box and a score per object. Here
proposals are the connected blobs of the building probability at 0.3, 0.5, and 0.7: one
building gives up to three nested boxes, 2,487 in the test tiles.
Non-maximum suppression keeps the best-scoring box and drops every box overlapping it with IoU >
0.5, then repeats: 1,224 kept. Our 15-line NMS keeps the same boxes as
torchvision.ops.nms in every test tile (the script stops if it does not). Scored like a
detector: average precision at IoU ≥ 0.5 against the 897 true boxes,
0.14. Faster R-CNN and YOLO learn their proposals (anchors) instead.
:::

## Touching houses become one blob

![Left, a crop of the town with each true footprint in its own colour; buildings that the U-Net merges are outlined in white. Right, the U-Net's connected blobs in the same crop: rows of touching houses come out as single long blobs.](cee_l16_segmentation/merged.png)

::: notes
Semantic segmentation labels pixels, not objects. Count buildings as connected blobs and
touching houses merge: in this tile 172 of 235 true
buildings share a blob with a neighbour; across the test block, building recall at IoU ≥ 0.5 is
20% and precision 22%. Fixes: a
third class for building edges, or instance segmentation (Mask R-CNN; He et al., 2017), which
predicts a mask inside each detected box.
:::

## A pixel in degrees is not square

![Left, one pixel of the before image drawn to scale on the ground: 0.27 m east-west by 0.42 m north-south. Right, the error of each road's length against the dataset's reference length, by road direction: pixels times 0.42 m overestimates east-west roads by up to about 60% and gets north-south roads right; converting each axis separately is within 0.1% for 95% of the roads, in every direction.](cee_l16_segmentation/degrees.png)

::: notes
The tiles are in EPSG:4326: a pixel is 3.74 × 10⁻⁶ degrees in both directions, but at 50.5° N a
degree of longitude is only cos(50.5°) ≈ 0.64 of a degree of latitude. So a pixel is
0.266 m east-west by 0.416 m north-south. Total
road length: 163.3 km (reference); pixels × one size,
207.7 km; each axis converted, 163.2 km. East-west
roads come out 56% too long, north-south ones 1%.
Reproject to a metric grid (UTM) before measuring anything; areas carry the same factor.
:::

## Rate severity from depth, not the mask

![Bars of metres of test cracking at each LTPP severity, all cracking treated as linear, under four rulers: the true mask by the dataset's ruler puts 427 m at low, the true mask by the ridge ruler 411 m, the U-Net's mask by the same ridge ruler 360 m, and the range image 63 m; all put most length at moderate. A fourth group shows the 71 m on branches the U-Net mostly missed, which only its ruler leaves unmeasured.](cee_l16_segmentation/decide.png)

::: notes
The U-Net finds where cracks run, with strict overlap at the annotators' level (within 2 pixels
they agree more closely). Its lengths, from its mask's
skeleton, are −4% from the reference centrelines at the median image (quartiles −15% and
+1%; 180 images with at least 1 m of cracking). The same skeleton on the true mask reads
+5%: the estimator runs long, so the U-Net's lengths are, if anything, short of the
annotators'. Lengths are usable with that check; its widths are not a measurement: on a 4-mm grid a width is a count of pixels, the 6 mm limit falls
inside one pixel, and the dataset's two rulers give two answers for the same cracks. The decision: use the network for location and length, and measure
width across the crack on the depth profile (the dataset's second ruler), at the scanner's native
1 mm when the severity limit is close. When not to use a mask width at all: whenever the limit
is within a pixel or two of the pixel size, and for structural cracks of 0.3 mm, which need
0.1 mm per pixel. Impact: a missed or under-rated crack is a liability question for the agency,
and the inspector's work moves from drawing cracks to auditing a network's widths.
:::

## Vote: who rates severity next season?

The agency wants LTPP severity from next season's scans.

- **A.** The U-Net's mask, widths by the distance transform.
- **B.** The U-Net for where; widths from the depth profile.
- **C.** The U-Net for where; widths re-annotated by inspectors.
- **D.** Inspectors rate everything, as now.

Vote alone. Discuss with a neighbour for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation. B. A rates by pixel
steps: the annotators' masks put 411 m at low by the ridge ruler, the range image 63 m. B keeps
the network's strength (location and length) and measures width where the information is, in the height profile; it still needs a
check against field measurements of width, which this dataset does not have. C repeats the
convention problem unless the inspectors measure rather than draw. D is the baseline. Ask what
would change the answer: a 1 mm release of the same scans.
:::

## Where segmentation lives today

| Today's system | The segmentation part | Lecture |
|:---------------|:----------------------|:--------|
| Segment Anything (SAM) | masks from a click or a box | 18 |
| Geospatial models (Prithvi): flood maps | a decoder on pretrained features | — |
| Detectors (Faster R-CNN, YOLO, DETR) | boxes, scores, NMS | 16 |
| Image generators (Stable Diffusion) | a U-Net denoiser | — |

::: notes
Segment Anything (Kirillov et al., 2023) returns masks for a prompt and is tested on CEE images
in Lecture 18. Prithvi, a geospatial foundation model, is fine-tuned for flood mapping with a
segmentation decoder (Lecture 18 mentions it; not used in this course). DETR (Carion et al., 2020) replaces anchors and NMS with set prediction.
The U-Net's encoder-decoder with skips is also the denoising network inside latent diffusion
image generators (Rombach et al., 2022). SpaceNet 8's baseline is a pair of U-Nets of this kind (Hänsch et al., 2022). HW3's
stretch goals: crack widths in millimetres and flood mapping.
:::

## Exit question

A drone photographs a concrete bridge deck, and a U-Net reports a largest crack width of 0.4 mm,
above the owner's 0.3 mm repair limit. Name the numbers you need before believing the width, and
one check you would run on the masks.

**In the notebook** (`cee_l16_segmentation.ipynb`): transposed convolution from scratch, the
U-Net trained on crops on a laptop CPU (in full on a GPU), widths from the distance transform,
and the break-it tests.

::: notes
Answer: the ground pixel size (distance to the deck, focal length, pixel pitch, and the lens's
distortion, or a target of known size in the frame), and the pixel size relative to 0.3 mm: at
anything coarser than about 0.1 mm per pixel, a 0.4 mm crack is one pixel and its width is the
annotator's convention. Checks: the mask's values (grey edges, faint masks), and two annotators
on the same images (how far two annotations agree). Collect the answers; they seed the next warm-up.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- Shrink to see; grow back with skips.
- Upsampling is convolution run backward.
- Thin masks: score with a tolerance.
- Width needs mm per pixel, and finer pixels.
- Count objects, not pixels; metres, not degrees.

**Next:** tokens, attention, and transformers.
:::
::: {.column}
![Four thumbnails of the lecture's key figures: the U-Net's maps laid out as a U; IoU and tolerant F1 against a one-pixel shift; crack widths in steps of the pixel against the range image's widths; and building boxes before and after non-maximum suppression.](cee_l16_segmentation/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Carion et al. (2020). *ECCV* (DETR).
- FHWA (2014). *LTPP Distress Identification Manual*, FHWA-HRT-13-092.
- Hänsch et al. (2022). *CVPR Workshops* (SpaceNet 8).
- He et al. (2017). *ICCV* (Mask R-CNN).
- Hüthwohl, Lu & Brilakis (2019). *Automation in Construction*.
- Kirillov et al. (2023). *ICCV*.
:::
::: {.column}
- Long, Shelhamer & Darrell (2015). *CVPR*.
- Milletari, Navab & Ahmadi (2016). *3DV* (V-Net, Dice loss).
- Rombach et al. (2022). *CVPR*.
- Ronneberger, Fischer & Brox (2015). *MICCAI*.
- Smith & Topin (2019). *SPIE*.
- Zhang, Wang, Hsieh, Yang, Yezzi & Tsai (2025). arXiv:2508.10256; 3DCrack, doi:10.21227/2etq-xh36.
:::
::::::
