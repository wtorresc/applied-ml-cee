---
title: Point Clouds
subtitle: "Lecture 22 · Was the lab built as modelled?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 5–21

On your own, no notes, two minutes:

1. Lecture 5: which factorization solves least squares without forming the inverse?
2. Lecture 9: how does RANSAC keep 30% bad points from moving a fitted line?
3. Lecture 21: a graph layer sums its neighbours' messages. Why a sum, not a list?

**Today:** a laser scan of a lab against its building model: label the points, line them up,
and find what differs.

::: notes
Give the answers before moving on.
1. QR or the SVD (Lecture 5). Today the SVD finds the rotation that best lines up two point sets.
2. It fits to the largest group that agrees within a tolerance and ignores the rest. Today
   registration ignores pairs farther apart than a cap, for the same reason: clutter.
3. A sum does not depend on the neighbours' order. A scan's points have no order at all;
   today a maximum plays the sum's part.
:::

## 584,701,977 points, one model

| CRAS lab, University of Porto (Abreu et al., 2023) | Value |
|:---------------------------------------------------|------:|
| Laser scans (Leica BLK360 G2, March 2023) | 21 |
| Points, each with colour, intensity, and a label | 584,701,977 |
| Rooms: a water-tank lab, a workshop, two offices, a hallway | 5 |
| Labels in the file → today's classes | 34 → 6 |
| Elements of the as-built BIM model meshed (IFC) | 135 |

::: notes
The decision: a facility manager plans a renovation from the building's model. Does the model
match what was built? Today's pipeline: label each point (building or clutter), line the scan
up with the model, measure where they differ. Classes: ceiling, floor, wall, door, window, and
clutter (furniture, equipment, the tank, the stairs); "unassigned" points (17%
of today's) are not scored. We use a 2% random sample, thinned to one point per 3-cm cube:
1,577,190 points. A data check: the paper's label table lists 288,024,661
ceiling points, which would make the labels sum to 644,701,977; the file has
228,024,661, which sums to the stated total. A typo, caught by adding up.
:::

## Your turn: wall, window, or clutter?

![One 1.5-m column of the scan, 14,816 points (4,096 drawn), every point grey: from above, and from the side.](cee_l22_point_clouds/attempt.png)

::: notes
Two minutes, alone: mark the walls, the window, and the furniture; then ask what they used to
decide. The column: of room 110's columns spanning the full 1.5 m, the one holding
the most classes. The labels of the 4,096 drawn points: 55% clutter,
18% ceiling, 14% wall,
10% floor, 3% window. Collect the cues students used: height, flat
and vertical, straight edges, what sits on what. Each cue is a feature; today two models
learn them differently.
:::

## A scan is a set; a max reads it

![The same column in 3-D: 4,096 points, no order, no grid, no edges.](cee_l22_point_clouds/picture_1.png)

::: notes
The one picture, in three frames. A point cloud is a set: listing its points in another order
changes nothing. No grid (Lecture 15), no edges (Lecture 21). Ask: what kind of function gives
the same answer for every order of its inputs?
:::

## A scan is a set; a max reads it

![The same column with 576 points in orange: the points that set the largest value of at least one of the network's 1,024 features. They spread over every surface in the column.](cee_l22_point_clouds/picture_2.png)

::: notes
PointNet lifts every point by the same small network to 1,024 numbers and keeps, for each
number, the largest over the column: a maximum ignores order. Only 576
points win any maximum (Qi et al. call them the critical points); delete all the others and the
1,024 numbers do not change. On single objects Qi et al. found them sketching the outline; in
this cluttered column they spread over every surface.
:::

## A scan is a set; a max reads it

![The same column, each point coloured by the class PointNet gives it: ceiling, floor, wall, door, window, clutter; 80% of the labelled points right.](cee_l22_point_clouds/picture_3.png)

::: notes
Each point's label comes from its own features and the column's 1,024 maxima: the point says
where it is; the maxima say what the column contains. This network never saw room 110 (it
learned from the other four).
:::

## Grids, graphs, and now sets

| Data | What is shared | Order-free by | Lecture |
|:-----|:---------------|:--------------|:--------|
| Pixels on a grid | the 3 × 3 weights | (the grid fixes the order) | 15, 16 |
| Nodes and edges | message and update rules | a sum over neighbours | 21 |
| Points in space | one network for every point | a max over the points | today |

::: notes
Each structure gets the network that respects it. A set needs a symmetric function: any
symmetric function of a set can be approximated by a network applied to each point, then a
maximum, then another network (Qi et al., 2017, Theorem 1); Deep Sets (Zaheer et al., 2017)
prove the same with a sum. We omit PointNet's small alignment networks.
:::

## PointNet: lift each point, take the max

:::::: {.columns}
::: {.column}
One network for every point, then a max:

$$\text{global} = \max_{i}\, h(x_i)$$

$$\text{label}_i = g\big(\ell(x_i),\ \text{global}\big)$$
:::
::: {.column}
| Part | Size |
|:-----|:-----|
| Each point *x*ᵢ | 7 numbers: x, y in the column, height, colour (3), intensity |
| *ℓ*, then *h*: the same MLP for every point | 7 → 64 (local), → 128 → 1,024 |
| *g*: label from local and global | 64 + 1,024 → 512 → 256 → 128 → 6 |
| Training | 4 rooms, columns of 4,096 points, turned at random |
:::
::::::

::: notes
Assumption: what a point is depends on its own coordinates and colour and on what its column
contains, not on the order of the points. 872,006 weights. Trained
from scratch, 100 epochs (AdamW, cosine schedule), each column turned by a random angle
and mirrored half the time, since walls run every way in a new building. Inputs: x and y
relative to the column's centre (so where the column sits in the building does not matter),
height above the floor, colour, and intensity. A turn changes x and y: the network is not
built to ignore turns, it is trained on turned columns.
:::

## Your turn: predict the check

```python
out_p = model(x[perm])          # the same points, shuffled
crit = unique(argmax over points of h(x))
g_crit = max over crit of h(x)  # only the critical points
R = kabsch(p, R0 @ p + t0)      # SVD; against SciPy
R, t = icp(scan, R0 @ scan + t0) # recover a known motion
```

How far is `out_p` from `out[perm]`? `g_crit` from `g`? `R` from `R0`?

::: notes
Two minutes, written. The known motion: turned (1°, −2°, 10°) about x, y, z and shifted
(0.5, −0.3, 0.1) m.
:::

## Same answer, any order

| Check | Value |
|:------|------:|
| Shuffled points: largest change in any class score | 8e-15 |
| Critical points of 4,096 | 576 |
| Global feature from the critical points alone: largest change | 2e-16 |
| Our SVD rotation against SciPy's | 4e-16 |
| ICP recovering the known motion: RMS point error, m | 4e-15 |

::: notes
Shuffling permutes the outputs and changes nothing else, up to round-off (the checks run in
64-bit floats). Keeping only the critical points leaves every maximum where it was, so the
global feature is unchanged: it rests on a few hundred points (each label also uses its own
point). The SVD solution (Kabsch, 1976) matches
SciPy's to round-off. ICP, given the scan and a moved copy of it, finds the motion again
(92,568 points, 39 iterations).
:::

## Neighbour features close PointNet's gap

![Intersection over union per class on the held-out rooms, pooled over the five, for four methods; mean IoU in the legend: height lookup 0.43, covariance + trees 0.61, PointNet 0.56, PointNet + covariance 0.63.](cee_l22_point_clouds/iou.png)

::: notes
Pooled over the five held-out rooms, mean IoU: height lookup
0.43, covariance + trees 0.61, PointNet
0.56, PointNet + covariance 0.63. The
max-pooled global feature earns its place: without it PointNet gets
0.52. Plain PointNet sees each point alone plus the
column's maxima; no point knows its neighbours. The covariance features say what each point's
neighbourhood is like (flat or scattered, vertical or level, how large); given them as inputs,
PointNet reaches 0.63 against the trees'
0.61: close, with one training seed each, ahead in four rooms of
five (room 108: 0.576 against
0.582). Both learn from every labelled point of the four training rooms (the trees stop
early on a held-out tenth). The global feature and the covariance inputs each
raise PointNet in every room. One caveat: PointNet's settings were fixed on one run that held
out room 110 and validated on room 111, both test rooms here, an advantage the trees did not
get. Building neighbourhoods into the network is
what PointNet++ does (Qi et al., 2017b). Windows defeat every method (IoU
0.14, 0.13,
0.13), and the hallway is the hardest room for all
(0.51 at best).
:::

## Room 110, a room it never saw

![Room 110 from above, floor and ceiling left out. Left, the labels: walls navy, doors orange, windows green, clutter pink. Right, PointNet's labels, right in grey and wrong in orange: 18% of these points wrong.](cee_l22_point_clouds/labels.png)

::: notes
PointNet on room 110, which it never saw (mean IoU 0.70). Its
commonest mistakes there: wall points called clutter
(11% of the wall points), door points called
clutter (50%), window
points called wall
(17%). Ask: where does the orange gather, and what
would a technician check first?
:::

## Line the scan up with the model

![From above, a horizontal slice: the model's surfaces in grey and the scan's walls in blue; left, after the coarse start (bounding boxes matched); right, after ICP.](cee_l22_point_clouds/registration.png)

::: notes
Two frames, two coordinate systems: the scanner's and the model's. Coarse start: match the
centres of the bounding boxes, a shift of (-9.94, -0.29, 0.63) m. Then ICP (Besl &
McKay, 1992): pair each scan point with the nearest model point, solve for the rotation and
shift that best move one onto the other (the SVD, Lecture 5), repeat. Pairs farther apart than
a cap are ignored, as RANSAC ignores outliers (Lecture 9); the cap shrinks from 1 m to 10 cm.
Result: a shift of (-9.85, -0.39, 1.25) m and a turn of
0.78° about the vertical.
:::

## Most of the lab is where the model says

![From above, a horizontal slice: each wall, door, and window point coloured by its distance to the plane of the nearest model surface, 0 (light) to 0.10 m or more (dark).](cee_l22_point_clouds/deviation.png)

::: notes
With the scan in the model's frame, measure each scan point's distance to the plane of the
nearest model surface (the model's triangles, sampled every 2 cm). Over the building's points
(ceiling, floor, wall, door, window): median 0.006 m,
94% within 5 cm, 90% within
0.040 m. The fit was made on a 5-cm thinning of these points, so a small median is
expected; what matters is where the dark points gather. Walls, doors, and windows alone: median
0.024 m; clutter, for comparison, 0.44 m.
Ask: what could make a stretch of wall dark? (A wall built elsewhere, a model drawn wrong, or a
poor fit there; the map alone cannot say which.)
:::

## The scan saw under 10% of every column

![For each of the model's 135 elements, the share of its surface with a scan point within 0.05 m, by type, with the median.](cee_l22_point_clouds/coverage.png)

::: notes
Coverage: spread points every 5 cm over each element's surface and count the share with a scan
point within 5 cm. Medians: ceilings 82%, windows
44%, doors 41%, walls
23%, slabs 2%, columns
0%. Under 10%: 10 of
10 columns, 4 of 5 slabs (the IFC slabs are the floors and the stair's landing),
22 of 61 walls. A scanner records only the
surfaces in its line of sight. A column that is missing and a column hidden behind a finish leave
the same trace: none. But under 10% within 5 cm is not the same as unseen: a wall displaced by
more than 5 cm has little surface within 5 cm of the scan, however many points sit on it. Of
those 22 walls, 12 have 100 or more wall, door, or window
points, and 6 of them go to the tape measure (the decide slide). Ask: how else
would you check a column you cannot see?
:::

## Your turn: predict four failures

1. Score the trees on random 1.5-m columns instead of whole rooms.
2. Scan the held-out room at a tenth of the density.
3. Start ICP turned 30°, 45°, or 90° from the coarse start.
4. Run ICP with no distance cap, with and without the clutter.

Write what happens to each.

::: notes
Two minutes alone. Then four reveals.
:::

## Random columns flatter the score

| Covariance + trees, the same features | Mean IoU | Accuracy |
|:--------------------------------------|---------:|---------:|
| Whole rooms held out (five folds) | 0.61 | 85% |
| Random 1.5-m columns held out (one in five) | 0.71 | 91% |

::: notes
First reveal. The same features and trees, tested on 41
random columns of 206 after training on the rest. A held-out
column's neighbours are in training: the other half of its wall, the rest of its desk, the same
lighting. The score then measures memory of this lab, not labelling the next building. Hold out
what you will meet: whole rooms here, whole buildings in practice (counties in Lecture 3,
photographs in Lecture 15). The two rows also score different points; scored on the random
split's own test points, the trees trained with each point's room held out reach 0.63 against
0.71: the random split still flatters.
:::

## A tenth of the points: PointNet holds

| Held-out rooms, mean IoU | Full scan | A tenth of the points |
|:-------------------------|----------:|----------------------:|
| Covariance + trees | 0.61 | 0.49 |
| PointNet | 0.56 | 0.56 |
| PointNet + covariance | 0.63 | 0.52 |

::: notes
Second reveal. Each held-out room thinned to a tenth of its points (a faster scan, or one
from farther away), features recomputed, the same trained models. The two kinds of input change
by different amounts. PointNet reads each column in sets of 4,096 points: at full density the
median point sits in a column of 8,999 points, so one set already holds 46% of
them; at a tenth, its column holds 895, all in one set, filled with
repeats (which change no maximum). The points in a set fall about 4.6-fold, not tenfold. The
covariance features are computed over the 24 nearest neighbours, which now span a patch about
three times wider (the square root of 10). The trees fall, doors nearly vanish (IoU
0.0003), and so does the network fed the same features. Train at the
density you will scan, or compute features within a fixed radius rather than over a fixed
count.
:::

## ICP needs a start within 20°

![Distance of ICP's result from the reference fit against the start's turn from the coarse start: 0.001 m at 0°, 0.001 m at 20°, 1.16 m at 30°, 12.7 m at 90°; dotted line at 1 cm.](cee_l22_point_clouds/basin.png)

::: notes
Third reveal. The same scan, started turned about its centre by 0° to 180° from the coarse
start, each result measured as its distance from the reference fit (the capped ICP on the
5-cm thinning; itself a fit, not the true pose). Up to 20°, ICP lands on the reference fit
(within 0.0012 m, the difference between 10-cm and 5-cm thinning); from 30° it stops 1.16 m
away, from 90° 12.7 m. ICP stops where its pairs and its fit agree; from
a start turned 30° or more, that is a wrong pose: a local minimum. With no coarse start at
all (the two frames as given, 10 m apart) it ends 7.0 m off. The
coarse start is the engineer's job: matching bounding boxes here; in practice, picking a few
common points or the building's main axes.
:::

## Clutter drags plain ICP off

| ICP from the coarse start | Points every | From the reference fit, m |
|:--------------------------|-------------:|--------------:|
| Every point (clutter too), no cap | 10 cm | 0.24 |
| Labelled structure points only, no cap | 10 cm | 0.011 |
| Every point, cap 1 m → 10 cm | 10 cm | 0.009 |
| PointNet's structure points, cap 1 m → 10 cm | 10 cm | 0.003 |

::: notes
Fourth reveal. Plain ICP pairs every scan point with the model, desks and boxes included,
and the fit ends 0.24 m from the reference fit (RMS over the
structure points; the reference is itself a fit, the capped ICP on the labelled structure, so
these are distances from it, not errors against the truth). Two cures, each from a lecture:
leave the clutter out (with the labels: 0.011 m), or ignore pairs farther
apart than a shrinking cap, a trimmed fit (Lecture 9: 0.009 m).
Today's pipeline does both, with PointNet's labels from rooms it never saw:
0.003 m. Every row uses the same 10-cm thinning and the same start. The two cures are close
(0.0105 m without clutter, 0.0091 m with the cap), a gap about the 0.0012 m by which the two
thinnings differ; only clutter without a cap stands apart. (On the reference's own 5-cm
thinning the capped rows land closer, 0.006 and 0.002 m, partly by construction: the reference
was fitted by the same method on overlapping points.)
:::

## Measure 9 walls; label 14 more

![The model's walls and columns from above, over the scan's points (light grey), by verdict: grey, within 0.05 m of its labelled points (31 walls); orange, off by more (7); pink, two surfaces, labelled points off and unassigned points within 0.05 m (2); the 9 orange and pink walls numbered worst first; green, seen only through unassigned points (14); blue, not seen, under 100 points (7 walls and every column).](cee_l22_point_clouds/verdict.png)

::: notes
The decision, element by element. A wall's points are the scan points whose nearest model
surface is on that wall, within 0.30 m, of two kinds: its labelled points (the dataset's hand
labels wall, door, and window, not PointNet's; the ceiling and floor points at its top and foot
would measure the ceiling and the floor) and its unassigned points, which carry no label. Of the
model's 61 walls, 7 have fewer than 100 points of either kind: not seen, no
verdict; nor do the 10 columns. 14 more are seen only through unassigned points:
label those points before judging them. The 40 walls with 100 or more labelled
points are judged by those: the typical one sits 0.040 m from them (the median over walls of
each wall's median distance); 31 are within 5 cm and 9 are farther, each with at
least 1,322 labelled points. Of those 9, 7 are off, and 2 rest on two surfaces:
wall 1's 3,270 labelled points sit 0.28 m from the model, but 21,407 unassigned
points sit a median 0.023 m from it; wall 8's 1,322 sit 0.058 m off, its
10,682 unassigned points 0.017 m. Unassigned points farther than 5 cm do not overrule
labelled ones: 20 of the 31 walls within 5 cm have 100 or more unassigned points
sitting farther (furniture in front, or a surface the labels missed). Any threshold from 25 to
1,000 points sends the same 9 walls to the tape measure. Wall 9 sits at 0.0505 m, just over
the line. For a numbered wall the scan and the model disagree, and the scan cannot say which is
wrong: a tape measure on site can. Five centimetres is today's threshold; a contract's tolerance
may be tighter, and the scan's noise and the fit's error both count against it. Ask: which blue
walls stand where the scan has no points at all, and which stand inside the scanned rooms?
:::

## Vote: who labels the next scan?

- **A.** A technician, by hand, point by point.
- **B.** The covariance features and trees, trained on this lab.
- **C.** PointNet, trained on this lab.
- **D.** A model, then a technician who checks doors and windows.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: vote, discuss two minutes, revote, then explain. The evidence: A is how
today's labels were made (Abreu et al., 2023), and every score here is measured against it. B:
0.61 on rooms held out, 0.49
on a sparser scan. C: 0.56, and it holds on a sparser scan
(0.56). D: even the best network here (PointNet + covariance,
0.63) labels windows at IoU
0.13 and doors at 0.47,
the openings a building model needs. A defensible answer: D, the model doing the bulk (ceilings
0.92, floors 0.96)
and a person checking the openings, tested first on one hand-labelled room of the new
building.
:::

## Where point clouds live

| Where | What the points are |
|:------|:--------------------|
| Scan-to-BIM and progress checks | laser scans of a building, against its model |
| Self-driving cars | lidar returns, sorted into cars, people, road |
| Road and rail mapping | scans from a moving vehicle |
| Phones and tablets with depth sensors | room scans for floor plans |

::: notes
Today's lab was scanned for the first row (Abreu et al., 2023); comparing scans with the
model to track what was built goes back at least to Bosché et al. (2014). Self-driving cars
find objects in lidar point clouds with networks that grew from PointNet (PointPillars, Lang et
al., 2019). Road and rail operators scan from moving vehicles to map assets and clearances.
Phones and tablets with depth sensors now draw a room's floor plan from a walk-through.
:::

## Exit question

Your firm scans a new building with a cheaper scanner, at a tenth of the density. Which part of
today's pipeline would you check first, and how?

**In the notebook** (`cee_l22_point_clouds.ipynb`): PointNet and ICP from scratch, the checks,
leave one room out, and the four failures.

::: notes
Check the labelling first: the covariance features depend on point density (the trees fell
from 0.61 to 0.49 at a
tenth of the points); PointNet did not here (0.56 to
0.56). Label one room of the new scan by hand and score every
model on it before trusting any; compute features within a fixed radius, or retrain. Then the
comparison: a 5-cm coverage test needs points spaced well under 5 cm, so check the new scan's
spacing.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- A scan is a set: use order-free functions.
- PointNet: a shared MLP, then a max.
- Neighbourhood features lifted PointNet.
- ICP: pair, SVD, repeat; start close.
- No labelled wall points, no verdict.
- Hold out whole rooms.

**Next:** physics-informed neural networks.
:::
::: {.column}
![Thumbnails of today's figures.](cee_l22_point_clouds/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Abreu et al. (2023). *Data* 8(6), 101.
- Qi et al. (2017). PointNet. *CVPR*.
- Qi et al. (2017). PointNet++. *NeurIPS*.
- Zaheer et al. (2017). Deep Sets. *NeurIPS*.
:::
::: {.column}
- Besl & McKay (1992). *IEEE TPAMI* 14(2).
- Kabsch (1976). *Acta Cryst.* A32.
- Weinmann et al. (2015). *ISPRS J.* 105.
- Bosché et al. (2014). *J. Comput. Civ. Eng.* 28.
- Lang et al. (2019). PointPillars. *CVPR*.
:::
::::::
