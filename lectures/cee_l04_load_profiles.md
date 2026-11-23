---
title: Retrieval, Clustering, and Modes
subtitle: "Lecture 4 · Which buildings should get an after-hours visit?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 1–3

On your own, no notes, two minutes:

1. Lecture 2: a raw distance between concrete mixes barely noticed the water. Why, and what
   fixed it?
2. Lecture 3: bridge records were split into blocks of counties or years, not at random. Why?
3. Lecture 1: a model's test error is low. Name one more thing you need before you sign a
   decision that rests on it.

**Today:** which campus buildings should get an after-hours visit?

::: notes
Give the answers before moving on; recall helps most with feedback.
1. A Euclidean distance is dominated by the inputs with the largest spread, so kilograms of
   cement and aggregate drowned out water. Engineering fixed it: dimensionless groups, above all
   the water–binder ratio, decided what "similar" means. Today asks the same question of buildings,
   where size plays the part of the big units.
2. Nearby bridges share climate, traffic, materials, and inspectors, and records of one bridge in
   different years are near-copies. A random split puts near-copies on both sides and flatters the
   model; holding out whole blocks of counties or years imitates a new county or a future year.
3. Any of: the error on cases like this one; whether this case lies inside the data; what an error
   costs and who carries it; that the test set was used only once; who is accountable.
:::

## Hourly meters can choose the visits

- Find **after-hours waste:** air, lights, equipment left on.
- A few visits a month: which buildings?

![Carpet plot of building Q, an office, every hour of 2016: one row per day, January at the top, colour by kW from 293 to 994. Nights, from 0 to 4 h, are as bright as afternoons all year: over all 2016 days, 679 kW at night against 707 in the afternoon; summer rows are brightest.](cee_l04_load_profiles/carpet.png)

::: notes
Frame the problem by what an error costs before talking about any method. We return to the two
kinds of error in the Decide step, with the ranking of a real campus.
A wasted visit: an engineer's day, spent on a building that was fine. A missed building: the waste
goes on every night, in money and in carbon. Today's tools: describe each building's day as a
vector; find buildings that run alike (retrieval), sort them into typical days (k-means), and find
the main ways days differ (PCA). The carpet: building Q, today's worked example; its nights
(0-4 h, 679 kW) are almost as high as its afternoons (12-16 h, 707 kW), over all 2016 days,
weekends included; the next table uses weekdays only.
:::

## 1,159 buildings, one typical weekday

Typical weekday $\mathbf{x}\in\mathbb{R}^{24}$: mean kW each hour over complete weekdays.

| Building Data Genome 2 (Miller et al., 2020) | Buildings |
|:----------------------------------------------|----------:|
| metered hourly, 19 sites, 2016–2017 | 1,578 |
| dropped: missing data | 356 |
| dropped: one value per day, spread over 24 h | 63 |
| **kept:** 15 sites | **1,159** |
| held out: Q's campus (67 offices) | 136 |

::: notes
The held-out campus is the one with the most offices: it plays the energy manager's campus, new to
every model. It was picked as the worked example because its offices make the point clearly; the
error comparisons use 30 random campus splits, but the visit-ranking numbers in Decide are for
this one campus only. A daily meter spread over the hours looks perfectly flat, like a building that never
sleeps; it is bookkeeping, not operation. We return to it under "when not to use".
Kept: at least 80% of weekdays complete in both years. 61 of the 63 daily meters are at one site.
Models are fit on the other 14 campuses (1,023 buildings). Times are local.
:::

## Your turn: which building runs like Q?

| kW | Use | Site | 0–4 h | 4–8 h | 8–12 h | 12–16 h | 16–20 h | 20–24 h |
|:----|:-----------|:------|---------:|---------:|---------:|---------:|---------:|---------:|
| Q | Office | Hog | 690 | 689 | 708 | 722 | 709 | 674 |
| A | Office | Hog | 558 | 549 | 714 | 841 | 831 | 673 |
| B | Office | Hog | 551 | 546 | 720 | 853 | 861 | 719 |
| C | Education | Rat | 62 | 62 | 63 | 65 | 64 | 62 |
| D | Public | Fox | 56 | 56 | 59 | 60 | 58 | 57 |

::: notes
Read the table aloud: mean kW on a 2016 weekday, by four-hour block; Q, A, and B are offices on one
campus (BDG2 code name Hog); C and D are at other sites. Two minutes on your own: which of A–D runs
most like Q? Then: would you send an engineer to look at Q's nights? Write down the answer and the
reason. Collect a few answers and ask what each student compared. By kilowatts, A and B are
nearest: the same size, offices on the same campus. By schedule, C and D: their nights stay level
with their afternoons, as Q's do. Q's 0–4 h load is 96% of its 12–16 h load; A's and B's nights
drop to 66% and 65%. Do not settle it yet; the next slides trace Q.
:::

## The encoder keeps shape, sets size aside

![Diagram of the course pipeline with the encoder highlighted, inside a PCA model with r equal to 3. A day x, 24 hourly values in kW, enters the encoder, which divides by the daily mean x-bar to get the shape p and projects the centered shape on three modes, z equals V_r transpose times p minus mu. The decoder rebuilds p-hat equals mu plus V_r z; the loss compares p-hat with the target p itself. A dashed arrow says the fitting is an SVD of the 2016 shapes on 14 other campuses, 1,023 buildings. Below, building Q, an office on the held-out campus: size 699 kW, night 690 kW, afternoon 722 kW. A mini chart shows Q's shape nearly flat against the mean shape, which rises during the day. z equals minus 0.62, minus 0.38, minus 0.01; its day-night swing z1 is smaller than in 80 percent of 157 training offices. A second mini chart shows the reconstruction close to Q's shape. The error is 2.2 percent of x-bar; the nearest typical day is always on; the nearest buildings are C and D.](cee_l04_load_profiles/profile_pipeline.png)

::: notes
This is Q from your attempt. The encoder is where today's lecture lives: divide out size, then keep
three numbers. There is no label: the target is the input's own shape, as in every unsupervised
method. Q's size, 699 kW, is kept aside; it returns in the Decide step. Three numbers rebuild Q's
day to 2.2% of its mean, and both other tools agree with what you saw: its nearest typical day is
"always on", and its nearest buildings are C and D.
:::

## Close shapes, buildings run alike

Shape $\mathbf{p}=\mathbf{x}/\bar{x}$. From Q, raw: A 536, D 3,141 kW. Shape: A 0.77, D **0.07**.

![Line plot of load divided by daily mean against hour of the day for building Q and the four buildings A to D. A, 694 kW, and B, 708 kW, dip to 0.75 and 0.73 at night and rise to 1.26 in the afternoon. C, 63 kW, D, 58 kW, and Q, 699 kW, stay between 0.96 and 1.04 all day, on top of one another.](cee_l04_load_profiles/attempt_shapes.png)

::: notes
State the assumption before the method; every method rests on one, and the Break-it step tests
this one. Closeness is the Euclidean distance between shapes, and retrieval returns the k stored
buildings nearest to p: Lecture 2's search, without the average. Why divide: x = x-bar times p, so
a raw distance mixes a difference in size with a difference in shape, and across buildings from
0.2 kW to 3.5 MW the size term wins. In raw kW, A and B are nearest to Q; in shape, C and D, a
tenth of its size, are.
Check: in raw kW the doubled Q is 3,423 kW away from Q, farther than any of A–D; in shape the
distance is exactly 0. Same schedule, twice the size.
Assumption: buildings that run alike have close daily shapes. Size x-bar is the daily mean in kW.
Raw distances from Q: A 536, B 596, C 3,114, D 3,141 kW. Shape distances: A 0.77, B 0.84, C 0.07,
D 0.07. Check question: double Q's load. Near or far?
:::

## k-means: a few typical days

Choose $K$ centers $\mathbf{c}_k$ (typical days) and assignments $z_n$ to minimize
$$J=\sum_{n=1}^{N}\lVert\mathbf{p}_n-\mathbf{c}_{z_n}\rVert^2$$

1. **Assign:** centers fixed, $z_n=\arg\min_k\lVert\mathbf{p}_n-\mathbf{c}_k\rVert^2$ lowers
   every term.
2. **Update:** assignments fixed, each $\mathbf{c}_k$ becomes the mean of its buildings: the mean
   minimizes a sum of squared distances.

- Neither step raises $J$; with finitely many assignments, Lloyd's algorithm stops at a **local**
  minimum.
- **By hand:** Q and A–D, $K=2$, started at A and C: {A, B} and {Q, C, D}.

::: notes
Update step: the gradient of the sum over n in cluster k of ||p_n − c||² is −2 times the sum of
(p_n − c); setting it to zero gives the mean. Lloyd (1982). Each iteration costs O(NKD). By hand, on
the block shapes of the five buildings in the attempt table (six blocks each, divided by their
mean): J falls from 0.0048 (centers A and C) to 0.0026 after the update, and the next assignment
does not change, so it has converged. A local minimum depends
on the start: use several starts, seeded with k-means++ (Arthur & Vassilvitskii, 2007), and keep
the lowest J.
:::

## Lloyd's steps, drawn in 2D

![Three panels of the 1,023 training buildings' scores on modes 1 and 2, coloured by their nearest of four centers marked with crosses. Start: four random buildings as centers. Assign, then update: each center moves, by an arrow, to the mean of its buildings. Converged after 12 passes: the centers spread along mode 1, from a flat day on the left to a strong working day on the right.](cee_l04_load_profiles/kmeans_build.png)

::: notes
Lloyd's algorithm on two numbers per building, to see it: their scores on the first two modes.
Assign colours each building by its nearest cross; update moves each cross to the mean of its
colour. J falls from 471.9 to 356.2 in 12 passes and never rises. The lecture's typical days run
the same two steps on all 24 hours. A few buildings lie off the axes.
:::

## PCA: the line and plane of most spread

Three hours of each building's day. **Line:** mode 1. **Sticks:** what the plane drops.

![3D scatter of 400 of the 1,023 training buildings' shapes at 3, 9, and 15 h, the cloud long and thin. A thick blue line through it is mode 1, keeping 79% of the spread of all 1,023; a translucent plane through the line keeps 94%; short orange sticks join each drawn building to its projection on the plane.](cee_l04_load_profiles/plane.png)

::: notes
The picture with three of the 24 hours, to see it: the cloud of shapes is long and thin. The
direction of most spread is mode 1 (79% of the variance of these three numbers, over all 1,023
training buildings; 400 are drawn); the best plane adds mode 2 (94%). Each stick is a building's error with two modes, perpendicular to the plane:
PCA minimizes the sum of the squared stick lengths, which is the same as keeping the most spread.
The view leaves out the 2% most extreme buildings on each hour.
:::

## PCA: the best r-number summary

Centered shapes as the rows of $\tilde{P}$ ($N\times 24$); $C=\tilde{P}^{\top}\tilde{P}/(N-1)$.

- Variance along a unit $\mathbf{v}$: $\mathbf{v}^{\top}C\mathbf{v}$. Its maximum solves
  $C\mathbf{v}=\lambda\mathbf{v}$: the top eigenvector, variance $\lambda$.
- Keep $r$ modes: $\mathbf{z}=V_r^{\top}(\mathbf{p}-\boldsymbol{\mu})$,
  $\hat{\mathbf{p}}=\boldsymbol{\mu}+V_r\mathbf{z}$. The error is the variance dropped:
  $$\frac{1}{N-1}\sum_{n}\lVert\mathbf{p}_n-\hat{\mathbf{p}}_n\rVert^2=\sum_{k>r}\lambda_k$$
- Most variance kept = least error (Eckart & Young, 1936).
- **SVD:** $\tilde{P}=U\Sigma V^{\top}$ gives the modes and $\lambda_k=\sigma_k^2/(N-1)$.

::: notes
Lagrangian: v'Cv − λ(v'v − 1); its gradient is zero when Cv = λv, and then the variance is λ, so
take the largest. The next mode maximizes variance orthogonal to the first, and so on. The error
identity: the modes are orthonormal, so the squared length of a centered shape splits into its
squared scores, and dropping modes r+1 to 24 leaves exactly their variance. No rank-r summary does
better in squared error. The SVD of the data avoids forming C, which would square the condition
number. Modes of a sensor array are the same SVD applied to snapshots of the sensors; we leave
that example to the notebook's Going deeper.
:::

## PCA in seven lines of NumPy

```python
def pca_svd(P: np.ndarray, r: int) -> tuple[np.ndarray, ...]:
    """Mean shape, top-r modes, their variances, and the scores."""
    mu = P.mean(axis=0)                        # mean shape, (24,)
    _, sig, Vt = np.linalg.svd(P - mu, full_matrices=False)
    lam = sig**2 / (len(P) - 1)                # variance along each mode
    V = Vt[:r].T                               # modes as columns, (24, r)
    Z = (P - mu) @ V                           # scores, (N, r)
    assert V.shape == (P.shape[1], r) and Z.shape == (len(P), r)
    return mu, V, lam, Z
```

**Predict:** fit on the 1,023 training shapes with $r=3$. Will this match scikit-learn's `PCA`
exactly, up to rounding, or not at all? What could differ?

::: notes
Take a show of hands for the three answers, and ask what could differ. The next slide answers.
:::

## The first mode is the working day

**Mode 1, 65%:** day against night. **Mode 2, 19%:** an evening peak. Three modes: 91%.

![Two panels of load divided by daily mean against hour. Mode 1, 65 percent of the variance: plus two standard deviations is a strong working day, 0.45 at night and 1.79 at 11 h; minus two standard deviations has a night higher than its afternoon. Mode 2, 19 percent: plus two standard deviations adds an evening peak near 20 h and lowers the early morning; minus two standard deviations drops the evening. The dotted mean shape rises from 0.77 at night to 1.24 at 13 h.](cee_l04_load_profiles/modes.png)

::: notes
Verified on the 1,023 training shapes: variances, modes, and scores agree with scikit-learn's PCA
to round-off (below 1e-13) once signs are aligned. Signs are arbitrary, and the linear algebra
library picks them: v and −v are equally good modes. We fix each by a rule (mode 1: more load in
the afternoon than at night), so the scores are the same on any machine. Mode 1
rediscovers the engineer's measure, how far the night falls below the afternoon, from data alone.
Mode 3, 7%, shifts the day earlier or later. Our k-means also matches scikit-learn's Lloyd
iterations from the same four starting shapes: identical labels, J within 1e-15, 14 iterations, J
never rose.
Verified: matches scikit-learn below 1e-13; a mode may flip sign, and both signs are valid. Mode 1's
score follows the night/day ratio (r = -0.77). Mode 2 peaks near 20 h.
:::

## Mean plus three modes rebuild Q's day

![Four panels of building Q's 2016 shape, dashed, flat near 1 all day, against its reconstruction in blue. The mean shape alone, a working day, is 15.1% off. Minus 0.62 times mode 1 flattens it: 8.1%. Minus 0.38 times mode 2: 2.2%. Mode 3 adds almost nothing: 2.2%.](cee_l04_load_profiles/rebuild.png)

::: notes
Q's shape as the mean shape plus its three scores times the three modes, z = V^T (p - mu). Q's
mode-1 score is negative: a flatter day than average, the after-hours load the visit looks for.
Three numbers replace 24 with an error of 2.2% of Q's daily mean.
:::

## Four modes reach the yearly change

4 modes: **6.0%**, below the year-to-year change (6.9%) in 21 of 30 splits. 16 typical days: 12.2%.

![Line plot of 2017 error in percent of the daily mean against the number of modes kept, from 0 to 8, with error bars over 30 splits: 22.1 at 0, 13.3 at 1, 9.9 at 2, 7.5 at 3, 6.0 at 4, and 3.5 at 8. Horizontal lines: the mean shape at 22.1, 4 typical days at 16.0, 16 typical days at 12.2, and, dashed, the year-to-year change at 6.9, labelled rough.](cee_l04_load_profiles/bracket.png)

::: notes
Error: root-mean-square over buildings and hours, as a share of each building's daily mean. Each of
the 30 splits holds out a quarter of the sites; models fit on 2016 shapes at the training sites
and are scored on 2017 shapes at the held-out sites. The floor is rough: it is the year-to-year
change, a building's 2016 shape against its own 2017 shape (6.9 ± 2.4% over the splits). That
is a forecast error, while the modes' error is a reconstruction (each held-out 2017 shape
projected on the modes), so it marks where extra modes start to describe this year; it is not a
strict floor. Paired, split by split: 4 modes are below it in 21 of the 30 splits, 3 modes in 12.
Each number is one RMS per split over its held-out buildings, averaged over the 30 splits. The
typical days here keep the best of 10 k-means++ starts in each split. Four modes reach the
year-to-year change; below it, extra modes describe this
year's details rather than the building. A typical day is one integer per building,
the coarsest summary: good for talking to people, poor for measuring.
Held-out campuses' 2017 shapes, 30 splits. Above: the mean shape, 22.1%. Three modes 7.5%. Four
typical days 16.0%; sixteen, 12.2%, worse than two modes (9.9%).
:::

## Predict before we test

1. **Size.** Run PCA on raw kW, or standardize each hour first as in Lecture 2. What does mode 1
   capture, and how much of the variance?
2. **Days.** Retrieve single days, with every other building-day in the database. How often is the
   nearest day the same building?
3. **Typical days.** $k$-means with $K=4$ on the 1,023 training buildings. About how many buildings
   in each typical day?

::: notes
Give one minute; students commit in writing. Each of the next three slides reveals one answer.
Ask who predicted it, and what the others missed. All three test the assumption: close vectors,
buildings that run alike.
:::

## Unscaled, the first mode is size

Raw kW: mode 1 holds **98%**, and its score is the size ($r=0.99995$).

![Two panels. Left: the mode-1 loading against hour for raw kW, for each hour standardized, and for shapes. Raw and standardized are nearly flat lines near 0.2; the shape loading is negative at night, about minus 0.2, and peaks near 0.3 at noon. Right: the raw mode-1 score in MW against mean load in MW for 1,023 buildings, a straight line.](cee_l04_load_profiles/size_mode.png)

::: notes
This is the lecture's decisive figure. Why: x = x-bar times p, and x-bar runs from 0.2 kW to
3.5 MW, so nearly all the variation between buildings is size, and every hour moves with it. The
top eigenvector is then almost the mean profile, and its score is almost the size. Standardizing
hour h divides column h by its spread (97.9%, r = 0.99996); it cannot remove a factor that
multiplies each row. Dividing each row by its own mean does. The assumption fails completely in kW:
"close" means "the same size". The 50% (size only) and 90% (size × shape) come from the Decide step.
Each hour standardized (Lecture 2): 97.9%. Ranking on it is ranking by size: 10 visits find 50%
of the excess, against 90%.
:::

## The nearest day is the same building

Random days: nearest is the same building **56%**. Sites held out: same use 25% (chance 16%).

![Bar chart of the share of queries whose nearest day is the same building, the same site, or the same use. Random split of days: 56, 66, and 68 percent. Sites held out: 0, 0, and 25 percent. A dotted line marks chance for the same use at 16 percent.](cee_l04_load_profiles/day_leakage.png)

::: notes
Queries: 2016 weekdays of buildings on the held-out campus. With a random split of days, 66% of
nearest days come from the same site. A random split measures how well a building recognizes
itself: its other days are near-copies of today, as a bridge's other inspections were in Lecture
3. Leakage (Kaufman et al., 2012); structured validation (Roberts et al., 2017). The honest number
for a new campus is 25%, well above chance but far from 68%. Use labels are coarse ("Education"
spans classrooms and laboratories), which may be part of why 25% is low; we have not measured it.
2,000 held-out building-days; exact FAISS search over 288,999 others. Random split of days: same
building 56%, same use 68%.
:::

## Four buildings claim a typical day

$K=4$: 518, 394, 107, and **4** buildings, peaking at 20 h up to 8.6× their mean.

![Line plot of four k-means centers, load divided by daily mean against hour. Always on, 518 buildings: flat, between 0.89 and 1.09. Moderate day, 394: from 0.69 at night to 1.31 at noon. Strong day, 107: from 0.43 at night to 1.87 at noon. Evening peak, 4 buildings: low all day, 0.37 at its lowest, rising to 4.7 at 20 h.](cee_l04_load_profiles/typical_days.png)

::: notes
Three are assembly venues, with events and lighting at night; the fourth is a warehouse. The four
are worth a look in their own right, but they should not take one of four typical days. A typical day is not a building type: read the
counts, flag outliers, then cluster. Remedies: remove or cap outliers before clustering, more
clusters, or k-medoids, whose centers are actual buildings and are less pulled by extremes. The
assumption "a few typical days describe them all" also hides how unequal the groups are.
Best of 20 starts. The 4: three assembly venues and a warehouse. Squared error charges far points
heavily, so a center goes to 4 buildings.
:::

## Shape says whether; size says how much

Excess = night − 0.66 × afternoon. 10 visits find **90%** of 2017's excess; size alone 50%.

![Line plot of the share of 2017 excess night load covered against the number of visits, ranked on 2016 data, for 67 offices. Size times shape reaches 63 percent at 5 visits and 90 percent at 10; shape only 60 and 75; size only 9 and 50; random order 15 percent at 10.](cee_l04_load_profiles/decision.png)

::: notes
0.66 is the median night/day ratio of the offices on the other 14 campuses, 0.658, used as 0.66
so every excess can be checked by hand: the office yardstick. Shape decides whether a night is high; size decides how many kilowatts it is. Q is 214 kW above an
office's typical night. Excess is not proven waste. Rough scale for Q: 214 kW for the 8 hours from
midnight on the 261 weekdays of 2016 is 214 × 8 × 261 kWh, about 447 MWh a year. Rankings move: one office's
excess fell from −29 to −97 kW, another's rose from 39 to 100 kW, so re-rank every year.
67 offices on Q's campus, ranked on 2016. 10 visits: size times shape 90%, shape alone 75%, size
alone 50%, random 15%. 9 of the top 10 repeat in 2017.
One campus only: resampling its 67 offices (1,000 draws) gives 95% intervals of 73-97% for size
times shape, 56-95% for shape alone, 14-76% for size alone. Paired draw by draw, size times shape
leads size alone by 15 to 76 points; its lead over shape alone, median +8 points, runs from 0 to
+28. So on this campus size times shape is at least as good as shape alone, not shown to be
better; it clearly beats size alone.
:::

## Screen with meters; confirm on site

| Watch for | Why | Do |
|:---------------|:------------------------------|:---------------------|
| Flat can be normal | laboratories, data rooms, hospitals, residences run at night | use the right yardstick |
| Flat can be the meter | 63 meters gave one value per day | check the time resolution |
| Shapes move with the season | 17% change typical day summer to winter | compare the same season |
| Not a diagnosis | a profile says when, not by what | the visit finds out |

::: notes
When not to use the method is part of the method. Q itself uses 71 W/m², more than twice its peers
A (28) and B (31): a server room or a laboratory would explain its flat day, and only the visit can
tell. Compare buildings with the same season's typical days. The 7.4% is one RMS over the 1,144
buildings with enough summer and winter weekdays (over all 1,159 it is also 7.4%). The 6.9% floor
on "Four modes reach the yearly change" averages the same comparison differently: one RMS per
split over its held-out sites, then the mean of the 30 splits.
"Education" includes laboratories. Summer and winter shapes differ by 16.1%, against 12.8%
(summer) and 9.2% (winter) for the same season a year apart, windows of equal length (1,138
buildings). The full-year 7.4% compares 12-month windows, so it is not the yardstick.
:::

## A load profile is a fingerprint

- Code names do not anonymize a meter.
- For residences, this is personal data.

![Bar chart of re-identification among 1,159 buildings: a building's 2017 weekday finds its own 2016 weekday first 64% of the time by shape and 50% by raw kW, and within the top five 81% and 65%.](cee_l04_load_profiles/fingerprint.png)

::: notes
Retrieval used as re-identification: the same nearest-neighbor search, with ground truth. BDG2
already replaces names with code names and gives no location for some sites (Miller et al., 2020);
a profile can still be matched across releases. This is the ethical side of today's tool; Lecture
27 returns to who is measured and who decides.
Hourly profiles show when people arrive, leave, and sleep (Molina-Markham et al., 2010). For
residences, share aggregates, coarser time steps, or data with consent.
:::

## Vote: which building is next?

Q got this month's visit. The screen lists three more on Q's campus, by night load above the
office yardstick:

- **A.** Education, 3,494 kW, nights 80% of afternoon, 540 kW above; 131 W/m².
- **B.** Office, 1,307 kW, nights 74%, 123 kW above; 42 W/m².
- **C.** Residence, 362 kW, nights 94%, 103 kW above; 21 W/m².

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
B is the after-hours visit: the office yardstick fits an office, and 123 kW above a typical office
night on a 1.3 MW building is the largest real candidate. A has the most kilowatts above the
yardstick, but the yardstick is wrong for it: 131 W/m², over four times offices A (28) and B (31)
in the attempt, says something runs there day and night, by design or not. It may deserve a visit
of a different kind, with a different yardstick. C is a residence: people are home at night. Ask
what would change the answer: A's records say classrooms only; B's excess vanishing (in 2017
it fell from 123 to 71 kW, but stayed); a cheaper visit.
:::

## Same search, learned embeddings

| Today | At scale | Lectures |
|:--------------------|:----------------------------------------------|:----------|
| PCA scores: 24 numbers → 3 | learned embeddings of text, images, series, searched on GPUs (Johnson et al., 2021) | 17–19 |
| k-means | builds the search index; e-means beats 25 Lloyd passes in 1.4–2.6 (Bessa et al., 2026) | — |
| retrieval | RAG: answer from the passages nearest a question (Lewis et al., 2020) | 19 |
| load profiles | forecasters pretrained on 900,000 simulated buildings, tested on BDG2 (Emami et al., 2023) | 20 |

::: notes
Where today's methods live in current systems. Vector-search indexes (FAISS) split the vectors
into cells with k-means and search only the nearest cells. e-means (Bessa et al., 2026, FAIR at
Meta, arXiv:2611.09501, November 2026) keeps the same objective J and changes the optimizer:
each center is a running sum over a count, both updated by an exponential moving average over
mini-batches, with a cosine-decayed step, dead-cluster revival, and step and batch sizes set by
fitted scaling laws. It partitions 10 billion image embeddings into 16 million clusters in 2.5
passes, below Lloyd's loss after 25. The notebook exercise runs it on our profiles: without its
revival rule it matches Lloyd in fewer passes; with it, it never keeps a small cluster such as
the four evening-peak buildings, so it gives up training error and does better on held-out
campuses. When not
to use it: at our size Lloyd takes milliseconds, and revival drops a small cluster even when
the small cluster is the finding. Today's failures carry over: an embedding can
encode size (or any nuisance) instead of what matters, and a random split of documents or days
flatters retrieval. Lecture 19 builds question answering over public specifications with
retrieval, and evaluates the retrieval step; Lecture 20 covers pretrained forecasters for time series.
:::

## Exit question

Why does dividing each profile by its own mean fix what standardizing each hour cannot? Which split
would you use to report how well retrieval finds similar buildings on a new campus?

**In the notebook** (`cee_l04_load_profiles.ipynb`): build the shapes, `pca_svd`, and Lloyd's
algorithm; match scikit-learn and FAISS; work Q and A–D by hand; and reproduce the size, leakage,
and outlier tests.

::: notes
Answer: size multiplies every hour of a building, a factor on the row; standardizing rescales each
column and leaves that factor in place, so mode 1 stays size (97.9%). Report retrieval with whole
sites held out (25% same use, against 68% with random days), because a new campus shares no
buildings with the database. Collect the answers; they show what to revisit in the next warm-up.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- Divide by the mean: shape vs size.
- PCA: the plane of most spread.
- Hold out buildings and sites.
- k-means spends centers on outliers.

**Next:** least squares through QR and the SVD, on Abrams' law.
:::
::: {.column}
![Four thumbnails of the lecture's key figures: the plane of most spread through three hours of each building's day; Q's day rebuilt from the mean and three modes; raw kW whose first mode is only size; four typical days, one claimed by four buildings.](cee_l04_load_profiles/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Arthur & Vassilvitskii (2007). *SODA*.
- Bessa et al. (2026). arXiv:2611.09501.
- Eckart & Young (1936). *Psychometrika*.
- Emami et al. (2023). *NeurIPS* 36.
- Forsyth (2019). Springer, Ch. 5, 6, 8.
- Johnson et al. (2021). *IEEE T. Big Data*.
:::
::: {.column}
- Kaufman et al. (2012). *ACM TKDD* 6(4).
- Lewis et al. (2020). *NeurIPS* 33.
- Lloyd (1982). *IEEE Trans. Inf. Theory* 28(2).
- Miller et al. (2020). *Scientific Data* 7, 368.
- Molina-Markham et al. (2010). *BuildSys '10*, 61–66.
- Roberts et al. (2017). *Ecography* 40(8).
:::
::::::

