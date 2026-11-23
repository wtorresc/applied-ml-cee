---
title: Stochastic Gradient Descent
subtitle: "Lecture 13 · How many decks will lose a point?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 5–12

On your own, no notes, two minutes:

1. QR solves least squares in about $ND^2$ operations (Lecture 5). With $N$ = 864,669 rows
   and $D$ = 108 weights, is that seconds or hours? With a million weights?
2. What system does one Newton step for logistic regression solve (Lecture 6)?
3. L-BFGS chose the GP's lengthscale (Lecture 12). What did it need from us at each step?

**Today:** 864,669 inspections, one gradient from 256 of them at a time.

::: notes
Give the answers before moving on.
1. About 10^10 multiply-adds: seconds. With a million weights, 10^18: never, and the D x D
   matrix alone would not fit in memory. Lecture 14's networks live there.
2. (A^T S A) delta = A^T (p - y): a weighted least squares with a D x D matrix, S = diag(p(1 - p)).
3. The objective and its gradient. Today's methods need only the gradient, and only on a few
   rows at a time.
:::

## 1.3 million deck inspections

:::::: {.columns}
::: {.column}
- FHWA's inventory, 2018–2024: 476,144 bridges, 50 states, DC and Puerto Rico; 3 sentinel records dropped.
- Did the deck lose a rating point by the next inspection? 10.4% did.
- **Train:** 864,669 pairs (2018 → 20, 2020 → 22). **Test:** 434,373 (2022 → 24).
- **Decide:** decks expected to drop, per state, for the next cycle's budget.
:::
::: {.column}
![Bar chart of the share of decks that lost a rating point by the next inspection, by the rating at the first inspection, over 864,669 training pairs, the bar labels giving the pairs in thousands: rated 3 or less, 6% of 5,007 pairs; 4, 4% of 21,879; 5, 5% of 106,473; 6, 7% of 219,273; 7, 9% of 368,607; 8, 23% of 121,333; 9, 39% of 22,097.](cee_l13_stochastic_gradient/drop_by_rating.png)
:::
::::::

::: notes
Data: FHWA's National Bridge Inventory, all-states files for 2018, 2020, 2022, and 2024
(public domain). A pair joins a bridge's record in one file to its record two years later:
highway bridges with a deck rating (Item 58) in both, culverts left out, a new inspection in
between (71% of pairs are 24 months apart), and no reconstruction in between. Inputs: age and
its square, years since the last reconstruction, ADT (log), percent trucks, length, deck width,
months between inspections, and coded items: the deck rating now, deck type, wearing surface,
membrane, deck protection, structure kind, owner, and state. With the intercept, 108 weights:
the logistic regression of Lecture 6, now national. Decks rated 9 drop most often
(39%): 9 is the top of the scale.
Three 2018 records carry sentinel codes, not measurements, and are dropped: Ohio 2504235's ADT
of 999,999 vehicles a day, and deck widths of 999.9 m on Minnesota 82527 and Virginia 3802.
33 training pairs (53 in all) still
report decks wider than 200 m; they are used as published, a limitation of this lecture.
:::

## Your turn: take one step

Four Illinois decks, 2018 → 2020. Input $x$ = 1 if rated 8 or 9; $y$ = 1 if it dropped.
Model $p=\sigma(w_0+w_1x)$, from $w_0=w_1=0$, step size $\eta=1$.

| Deck | 2018 → 2020 | x | y |
|:-----|:-----------:|--:|--:|
| A | 8 → 7 | 1 | 1 |
| B | 7 → 7 | 0 | 0 |
| C | 8 → 8 | 1 | 0 |
| D | 5 → 5 | 0 | 0 |

::: notes
The bridges: Ogden Road over Little Wolf Creek; Farragut Avenue over Pettibone Creek and Crusley
Drive; West Gate Road over Grassy Creek; Sampson Street over Pettibone Creek.
Three minutes, alone or in pairs. Each row's gradient is (p - y)(1, x) (Lecture 6).
(1) Batch {A, B}: average gradient, and the weights after one step. (2) Batch {C, D}. (3) The
average of the two gradients against the gradient on all four rows.
Answers: every p is 0.5. Batch {A, B}: (0, -0.25), so w = (0, 0.25): good decks drop more.
Batch {C, D}: (0.5, 0.25), so w = (-0.5, -0.25): good decks drop less. The two batches disagree
on the sign of w1. Their average, (0.25, 0), is the gradient on all four rows exactly. Each
step is noisy; on average it is the full gradient. That property needs random batches, and
the break-it test removes it.
:::

## Shuffled batches, one step each

![Diagram of the course pipeline with the optimizer highlighted. An inspection x goes to an encoder that scales 108 inputs, giving z; the decoder computes sigma of w transpose z, giving p; a forecast sums p per state, giving the drops expected. Above, the training loop: 864,669 inspections are shuffled each pass into batches of 256; each batch makes one SGD step, w minus eta times A_B transpose times p_B minus y_B, over 256, which updates the decoder's weights; 3,377 steps make one pass. Below, step 1 of the run from w = 0 with eta = 0.3: of 256 shuffled inspections, 29 dropped; every p is 0.5, so the intercept's gradient is 0.5 minus 29 over 256, 0.387; after the step the intercept is minus 0.116, and the training loss falls from 0.693 to 0.620.](cee_l13_stochastic_gradient/sgd_loop.png)

::: notes
The decoder is Lecture 6's; only the way its weights are found changes. Newton used all
864,669 rows for every step; SGD uses 256 and steps right away. The first step alone takes the
loss from 0.693 (a coin flip) to 0.620; moving the intercept alone, toward the base
rate, gets it to 0.649, 60% of that drop.
:::

## A random batch is right on average

**Assumption:** each batch is drawn at random from the training rows.

- Loss $F(\mathbf{w})=\frac{1}{N}\sum_{i}\ell_i(\mathbf{w})$; batch gradient $\mathbf{g}_B=\frac{1}{B}\sum_{i\in B}\nabla\ell_i(\mathbf{w})$.
- $\mathbb{E}[\mathbf{g}_B]=\nabla F(\mathbf{w})$: right on average.
- Its noise shrinks like $1/B$; its cost is $BD$, not $ND$.

::: notes
For the logistic loss, the gradient of one row is (p_i - y_i) a_i (Lecture 6). The expectation
is over which rows land in the batch: exact for sampling with replacement. With a fresh
shuffle each pass (random reshuffling, the standard practice), each batch is a uniform sample
of the rows but not independent of the batches before it in the pass; it converges at least as
fast (Mishchenko, Khaled & Richtárik, 2020; Bottou, Curtis & Nocedal, 2018). Here B = 256, so
a step costs 1/3,377 of a full gradient. The break-it test streams the rows in the
order FHWA publishes them, which breaks this assumption.
:::

## Batches agree far off, scatter near

![Two contour maps of the attempt's two-weight loss over the intercept and the rated-8-or-9 weight, from all 864,669 training pairs, with the optimum as a white star at minus 2.52 and 1.44. Left, the whole bowl: from w = 0, thirty steps of size 3, each from a different random batch of 256 decks, as thin orange arrows, spread over 8 degrees around the full-data step, a thick black arrow 1.19 long. Right, a zoom near the optimum: thirty batch steps point in every direction, while the full-data step is 0.017 long; the mean of the thirty is 0.004 from it.](cee_l13_stochastic_gradient/batch_arrows.png)

::: notes
The attempt's model on every training pair: x = 1 if the deck is rated 8 or 9 (25.4% of those
dropped, against 7.5% of the rest). Each orange arrow is one SGD step, eta = 3, from one random
batch. Far from the optimum the signal dwarfs the noise: every batch agrees. Near it the true
gradient is tiny and the noise is not: batches point every which way. Their mean lands
0.004 from the full step (far off, 0.010): thirty random batches average out only so
much. That is the noise ball of the next slides. Ask: what happens to the scatter with a
batch of 16? Of 4,096?
:::

## The step must fit the curvature

- Gradient descent: $\mathbf{w}\leftarrow\mathbf{w}-\eta\nabla F(\mathbf{w})$, safe for $\eta\le 1/L$, $L$ the largest curvature.
- Near the optimum its error shrinks like $(1-\eta\lambda_{\min})^t$; $\kappa=\lambda_{\max}/\lambda_{\min}$ of the Hessian.
- Here $\kappa$ = 11,507: the flattest direction is small states' weights (DC, Delaware).
- Newton divides by the curvature: exact in 8 steps.

::: notes
For the logistic loss, sigma' is at most 1/4, so L = lambda_max(A^T A)/(4N) = 0.72 and the
safe step is 1/L = 1.38. Tuned like the stochastic methods, with steps of 1, 2, 4, 5, 6, and 8
over L, gradient descent does best at 5/L and is still 0.0021 nats above the optimum after
100 passes; at 8/L it oscillates (0.040). Larger steps than 1/L help because near the optimum
the largest curvature is 0.25, about a third of L (0.34 L): the bound L holds at w = 0,
where every p is 0.5. Near the optimum the steps stay stable up to 2 over that curvature,
5.8/L. Newton needs the D x D Hessian, N D^2 = 10^10 multiply-adds per step: about as much
arithmetic as 50 to 100 gradient passes, and affordable only because D is 108.
:::

## Raw units carve a ravine

![Two 3D loss surfaces of a two-weight model of deck drops, intercept and deck age, cut flat above 0.75 nats, with gradient descent from w = 0 in orange, each at 1.9 over its largest curvature at the start. Left, age in decades counted from 0: a long, narrow ravine, condition number 88; the first step drops off the wall onto the floor and the next 29 crawl along it, ending at w0 = minus 0.49, far from the optimum, a white star at minus 1.87. Right, age centered and scaled: a round bowl, condition number 1.3; the first step overshoots across it and the run is within 0.001 nats of the optimum in 4 steps.](cee_l13_stochastic_gradient/ravine3d.png)

::: notes
Same data, same model, two units for one input. Counted from 0, age and the intercept pull
together: the bowl stretches into a ravine. The steep walls cap the step at 1.9/L = 0.29; along
the floor that step is tiny, so gradient descent crawls: (1 - 1/kappa)^t. Centering and scaling
make the bowl round, and the safe step (7.6) is right in every direction. The national model's
kappa of 11,507 comes from the rare state codes instead. Standardizing the numeric inputs
leaves it, since the 0/1 code columns keep their raw scale; centring and scaling those columns too
takes kappa to 254. On a logistic loss the curvature falls as the fit improves, so the
classic zigzag dies after one step: what remains is the crawl.
:::

## Constant steps stall; averages settle

- Constant $\eta$: SGD hops in a noise ball, size about $\eta\sigma^2$.
- Average the iterates (Polyak & Juditsky, 1992), or shrink $\eta$ (Robbins & Monro, 1951).
- Here, $\eta$ = 3: the iterate wanders ±0.08; the average ends $9\times10^{-6}$ above.

![Contour map zoomed around the optimum of the attempt's two-weight loss, a white star. Steps 100 to 1,000 of SGD at eta = 3 in thin orange zigzag around it in a cloud about 0.08 wide in each weight; the running average of the iterates, a thick black line, comes in from the lower right and settles on the star.](cee_l13_stochastic_gradient/noise_ball.png)

::: notes
sigma^2 is the variance of a batch gradient near the optimum. The average of the iterates
cancels much of the hopping; "SGD, averaged" in this lecture averages every iterate from the
first step. Momentum (Polyak, 1964) keeps a running sum of past gradients; it speeds travel
along shallow valleys but hops as much as plain SGD at a constant step.
Robbins-Monro conditions: sum eta_t = infinity, sum eta_t^2 < infinity. The average:
w-bar_t = (1/t) sum_{s<=t} w_s. Momentum: m <- beta m + g, w <- w - eta m, beta = 0.9. The
figure: the attempt's model, 1,000 steps (256,000 decks, under one pass); the last iterate's
median gap over steps 100-1,000 is 0.0002 nats, and it wanders by a standard deviation of
0.086 and 0.076 in the two weights.
:::

## Adam sizes each weight's step

$$\mathbf{m}\leftarrow\beta_1\mathbf{m}+(1-\beta_1)\mathbf{g},\qquad\mathbf{v}\leftarrow\beta_2\mathbf{v}+(1-\beta_2)\mathbf{g}^2,\qquad\mathbf{w}\leftarrow\mathbf{w}-\eta\,\frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}}+\epsilon}$$

- $\hat{\mathbf{m}}=\mathbf{m}/(1-\beta_1^t)$, $\hat{\mathbf{v}}=\mathbf{v}/(1-\beta_2^t)$: corrected for starting at zero.
- Each weight moves about $\eta$ per step, however small its gradient: rare codes get full steps.
- $\beta_1$ = 0.9, $\beta_2$ = 0.999, $\epsilon$ = $10^{-8}$ (Kingma & Ba, 2015).

::: notes
Squares and square roots are per weight. Adam descends from AdaGrad (Duchi, Hazan & Singer,
2011), which was built for rare features like a code that appears in 0.05% of the rows. Its
steps do not change if the gradient is rescaled. They do change if an input is rescaled: a
weight whose right value is 10^-6 still moves by about eta. The units slide tests this.
:::

## Your turn: predict the check

```python
def sgd_pass(A, y, w, lr, rng, B=256):
    visit = rng.permutation(len(A))          # a new order each pass
    for j in range(len(A) // B):
        rows = visit[j * B:(j + 1) * B]
        p = expit(A[rows] @ w)               # (B,)
        g = A[rows].T @ (p - y[rows]) / B    # (D,)
        w = w - lr * g
    return w
```

**Predict:** this gradient against PyTorch's autograd? Our Adam against `torch.optim`? One
pass at $\eta$ = 0.3: 0.1, 0.01, or 0.001 nats above Newton's optimum?

::: notes
expit is the logistic function, from scipy.special. The three questions in full: (1) the
largest difference between this gradient and PyTorch's autograd on the same batch; (2) between
our Adam and torch.optim.Adam after 200 steps on the same batches; (3) how far above Newton's
optimum one pass at eta = 0.3 (the step the sweep chose) ends. Take a show of hands on each. Most expect
small but visible differences against PyTorch, and that one pass is far from the optimum.
:::

## Agrees with PyTorch and Newton

| Check | Against | Agreement |
|:-------------------------|:-------------------------------|:-----------------|
| Gradient | PyTorch autograd; finite differences | identical; 2 × 10⁻¹⁰ |
| Momentum and Adam, 200 steps | torch.optim | 2 × 10⁻¹⁶ |
| Newton, 8 steps | scikit-learn's Newton solver | 3 × 10⁻⁸ |
| One pass, η = 0.3 | Newton's optimum | 0.0017 above (seeds: 0.0014–0.0021) |

::: notes
Both optimizers take the same 200 shuffled batches in float64; PyTorch differs only in the order
of a few floating-point operations. scikit-learn: LogisticRegression with C = inf and the
newton-cholesky solver. The optimum's training loss is 0.2960 nats; one pass of plain SGD
gets within 0.0017 of it, from 0.693 at the start (the median of seeds 5 to 9: every step size
in this lecture is chosen on seeds 0 to 4 and reported on seeds 5 to 9).
:::

## One pass beats 100 of gradient descent

![Log-log plot of the training loss above the optimum against passes over the training inspections, from 1/64 to 100. Gradient descent at its best step, 5/L, is at 0.18 after one pass and still at 0.0021 after 100 passes. SGD and Adam fall to about 0.002 within one pass and then hop between about 0.0003 and 0.002 for ten passes. Averaged SGD keeps falling: 0.0003 after one pass and 0.0001 after ten. Newton's steps, diamonds, fall from 0.033 to 0.00008 in three steps and off the chart after that. A dotted line marks the base-rate-only model, 0.038 above the optimum; another marks the estimation floor, D over 2N, 0.00006.](cee_l13_stochastic_gradient/convergence.png)

::: notes
The bracket: a model with only the base rate is 0.038 nats above the optimum; the estimation
floor D/2N = 0.00006 is how far the exact fit itself sits, on average, from the best model of
its form, because it saw 864,669 rows and not all bridges. It is the clustered version, one half
of the trace of H times the sandwich covariance, which allows for the two pairs of each bridge;
here it matches D/2N. Optimizing below that floor buys nothing a new sample of bridges could
detect (Bottou & Bousquet, 2008). After one pass, averaged SGD is 0.0003 above the optimum,
7 times closer than gradient descent after 100 passes at its best step (5/L, from a grid
of steps 1 to 8 over L, fine where it matters). Constant-step SGD, momentum, and Adam hop
between 0.0003 and 0.002 over passes 1 to 10: the noise ball. Newton crosses the floor at
its fourth step.
:::

## Three step sizes, one bowl

![Left, the attempt's two-weight loss surface in 3D with three runs of 300 SGD steps from w = 0, batches of 256: eta = 0.1 in blue creeps along the valley and stops short of the white-star optimum; eta = 3 in green arrives and stays; eta = 30 in orange scribbles across the bottom far around the optimum. Right, the loss above the optimum against the step, log scale: eta = 0.1 falls slowly to 0.011; eta = 3 is below 0.001 by step 100 (0.0002) and hops around 0.0001; eta = 30 keeps bouncing, median 0.07 over the last 100 steps and up to 0.6.](cee_l13_stochastic_gradient/steps3d.png)

::: notes
The attempt's two-weight model, so the whole surface can be drawn. The three runs see the same
batches in the same order; only the step differs. Too small: 300 steps cover a fraction of the
way. Too large: each batch's noise is multiplied by 30 and the run never settles. The numbers on
the next slide come from the full 108-weight model, where the same three regimes appear.
:::

## Too small crawls, too large bounces

:::::: {.columns}
::: {.column}
- One pass, seeds 0–4: $\eta$ = 0.1 leaves 0.0056; $\eta$ = 0.3, 0.0016; $\eta$ = 10, 0.069.
- Averaging tolerates a larger step: $\eta$ = 3, 0.0003.
- Tune $\eta$ on the training loss, never on the test cycle.
:::
::: {.column}
![Log-log plot of the loss above the optimum after one pass against the step size, 0.1 to 10, median of seeds 0 to 4 with their range shaded. SGD falls from 0.0056 at 0.1 to 0.0016 at 0.3, stays close at 1 (0.0017), and rises to 0.069 at 10. Averaged SGD is worse at small steps, 0.012 at 0.1, reaches its best, 0.0003, at 3, and rises to 0.014 at 10.](cee_l13_stochastic_gradient/step_size.png)
:::
::::::

::: notes
Too small: the pass ends before the weights arrive. Too large: the noise ball grows with eta.
SGD's 0.3 and 1 are close (0.0016 against 0.0017, inside the seeds' spread); the rule keeps
0.3. Averaging smooths the hops, so it can take larger steps. Momentum's best step is
0.1 (0.0010) and Adam's 0.01 (0.0015). Every step size here was chosen on the training
loss of seeds 0 to 4; seeds 5 to 9 then report it, so no reported number is the best of the
draws that chose it: at their chosen steps, 0.0017 for SGD, 0.0003 averaged, 0.0014 for
momentum, 0.0017 for Adam. The test cycle stays untouched until the decision.
:::

## Units decide the step

- ADT runs to 919,000 a day: no single $\eta$ suits ADT's weight and the intercept.
- Adam rescales each gradient, not each input: still 31 times worse.

| One pass, best step | SGD | Adam |
|:--------------------|----:|-----:|
| Raw units | 0.15 | 0.050 |
| Scaled | 0.0017 | 0.0016 |

::: notes
Raw units: years, vehicles a day, percent, meters, months. Scaled: each numeric input to mean 0
and standard deviation 1 on the training rows. The same model in both rows: ADT enters linearly
here (not as a logarithm), so the scaled row differs slightly from the earlier slides. Raw
units: each step was chosen on seeds 0 to 4 from a grid wide enough to hold its best inside, and
reported on seeds 5 to 9. Scaled: the main model's steps, with no search. ADT's 99th percentile
is 113,000 a day; the intercept's input is 1. Unscaled, SGD's best step is 3 x 10^-8; ten times
larger, 3 x 10^-7, it blows up (1.2 nats above; at 10^-7 it ends at 0.26). At 3 x 10^-8 every seed
from 0 to 9 ends between 0.152 and 0.165, from a start of 0.40: 88 times the scaled gap. Adam
moves every weight by at most about eta per step: at its best step, 10^-5, no weight travels more
than about 0.03 in one pass of 3,377 steps, yet that step is large for ADT's weight, whose value at the
optimum is −9 x 10^-8 per vehicle a day. It ends at 0.050, better than SGD but 31 times the scaled
0.0016. Scale the inputs first, with the training rows' mean and standard deviation.
:::

## Predict before we test

1. **Published order.** FHWA's file runs state by state, Alabama to Puerto Rico. One pass in
   that order, at its own best $\eta$: better, the same, or worse than shuffled?
2. **One state.** The shuffled pass is 0.0017 above the optimum nationally. Is its Illinois
   forecast within 1%, 5%, or 15% of Newton's 2,688 decks?
3. **The weights.** After one averaged pass, how many of the 108 weights sit within two standard
   errors of Newton's?

::: notes
One minute; students commit in writing. The next slides reveal (1) the order, then (2) and (3)
together: the loss and the weights converge at different speeds.
:::

## Published order never settles

![Two panels. Left, log plot of the loss above the optimum along one pass of averaged SGD, each at its best step size. Shuffled, dashed, falls steadily to 0.0003. Published order, solid, falls to about 0.008 by the end of the 2018 file and then flattens, ending at 0.006, 20 times higher. Right, bars of each state's forecast minus Newton's, in percentage points, for the 52 states in file order. Shuffled bars stay within about 1 point except a few, the largest 3 points; published-order bars reach 4 points above and 11 points below, in Utah, with Maine 9 and California 7 points below.](cee_l13_stochastic_gradient/file_order.png)

::: notes
This is the lecture's decisive figure. At their own best steps, published order ends
20 times above the shuffled pass for averaged SGD and 3 times for the last iterate. Its steps
were chosen on the very pass they report, which favours it: the ratios are generous.
Consecutive batches come from one state, and the 2020 file repeats the tour. Per state, the mean
forecast error is 2.5 points, against 0.65 shuffled; Utah is 11 points low and Maine
9. Illinois lands at 2,687, beside Newton's 2,688: right by luck, as the other bars show.
:::

## Sorted batches shrink the states

| What went wrong | What to do |
|:-------------------------------------|:-------------------------------------|
| consecutive batches came from one state | shuffle before training |
| each state's forecast was pulled toward the national rate | check the forecast per group, not just the loss |
| sorted data are the norm: state, year, county, inspector, sensor | a stream you cannot shuffle: a shuffle buffer, or replay older rows |

::: notes
What the errors follow: each state's drop rate, not its place in the file. Across the 52 states,
the published-order error correlates at r = −0.83 with the state's training drop share above the
national 10.4%, and at r = 0.12 with its place in the file (the first half of the
file averages 2.3 points of error, the second half 2.7). States that drop often are
forecast low, states that rarely drop are forecast high: the shuffled pass shows the same pattern
(r = −0.71), with errors of 0.65 points instead of 2.5. The trap is that published
order still lowers the loss, from 0.69 to 0.006, and looks like training. Nothing in its
loss curve says the batches were biased; the per-state check does.
:::

## The loss settles before the weights

- After an averaged pass, the state weights sit a median 8.3 standard errors from Newton's; the
  other weights, 1.1.
- Directions where the loss is flat move a state's forecast, not the national loss.

| After one pass, seeds 5–9 | Above the optimum | Illinois forecast |
|:---------------------------|------------------:|------------------:|
| SGD, last iterate | 0.0014–0.0021 | 2,614–2,838 |
| SGD, averaged | 0.0003 | 2,601–2,612 |
| Newton | 0 | 2,688 ± 33 |

::: notes
± 33 is the standard error of Newton's forecast from the sampling of bridges (sandwich
covariance, clustered by bridge), over 17,746 Illinois inspections. The weights along flat
directions, the smallest curvatures behind kappa = 11,507, are the last to converge: the
intercept against the state weights. Epoxy-coated reinforcing (Item 108C = 1) is fine after one
pass: −0.158 against Newton's −0.145, standard error 0.012 (odds of a drop 0.87 times). Utah's
weight is 12 standard errors off. After ten averaged passes the median is 1.5 standard errors,
and 44% of the weights are still beyond two.
:::

## When SGD is the wrong tool

| When | Use instead |
|:--------------------------|:-------------------------------------------|
| few weights (D = 108) | Newton: exact in 8 steps |
| weights you will quote (a state, an epoxy effect) | converge, then report standard errors |
| rows you cannot shuffle | sorted streams bias every step |
| millions of rows and weights, no Hessian | SGD's home: Lecture 14's networks |

::: notes
Newton's D x D Hessian costs N D^2 per step and D^2 memory: trivial at 108, impossible at a
million. L-BFGS (Lecture 12) sits between: full gradients, curvature from their history. For
convex models of modest size, full-batch methods are exact and cheap; SGD earns its place when
N and D are both large or the model is not convex.
:::

## Whose forecast is wrong?

| Who | Why |
|:----------------------------------|:----------------------------------------------|
| DC, Delaware, Nevada, Puerto Rico, Rhode Island | the flattest direction: their weights sit near Newton's, their forecasts do not |
| Utah and Maine | unlike the average: 20% and 21% of decks drop, against 10.4% |
| every state's budget | a national loss averages them away: check each state against the exact fit |

::: notes
FHWA and state agencies report condition nationally and plan work state by state. An error that
averages out nationally can be hundreds of decks in one state: one plain SGD pass forecast
Illinois anywhere from 2,614 to 2,838, depending on the seed, against 2,688. Ask: who notices a
forecast that is 6% high, and what does the wrong budget cost the decks that needed it?
The first row lists the largest parts of the Hessian's flattest direction. All five are small
inventories (DC, Rhode Island, and Delaware have the three smallest), but size alone does not
decide it: Hawaii and Alaska have fewer pairs than Puerto Rico. After one averaged pass their
weights are among the closest to Newton's in standard errors (45th to 51st of 51 state
weights, at most 3.2), yet DC, Delaware, Puerto Rico, and Nevada have the 3rd, 6th,
10th, and 11th largest forecast errors of the 52 (Rhode Island the 21st). Few rows give
their weights wide standard errors; a weight near Newton's in those units can still leave the
state's forecast off by points.
:::

## Vote: what do you send IDOT?

It is 2022. IDOT asks how many Illinois decks will lose a rating point by their next inspection.

- **A.** One SGD pass: 2,655 (median of five seeds).
- **B.** One averaged-SGD pass: 2,601 (seed 5).
- **C.** Newton: 2,688 ± 33, with a note that the drop rate fell from 11.1% to 9.7% between the
  two training cycles.
- **D.** Averaged SGD in published order: 2,687, beside Newton's.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
C. A scatters from 2,614 to 2,838 across seeds. B is steady (2,601 to 2,612 over
five seeds) but 86 low on seed 5, 2.6 standard errors: its intercept and Illinois's weight have
not converged together. Restoring Illinois's weight alone to Newton's would overshoot to 3,387;
restoring the intercept alone would drop it to 1,984; both together give 2,632. D matches by
luck, as Utah's 11 points show. At 108 weights Newton is exact in seconds, and its note
matters most, as the next slide shows.
:::

## Drift, not the optimizer, set the miss

- Forecast minus outcome: 329 decks, 6 times chance and estimation combined.
- Each cycle has dropped less: 11.1%, 9.7%, 8.7%. With Newton, the drift is the error left.

| Illinois, 2022 → 2024 | Decks |
|:----------------------|------:|
| Newton's forecast | 2,688 |
| What happened | 2,359 |
| Estimation, 1 SE; chance, 1 SD | ± 33; ± 44 |
| One SGD pass, five seeds | 2,614–2,838 |

::: notes
Chance: even with the right probabilities, the count of drops varies by about 44 (the binomial
standard deviation); the miss is 7.4 times that alone, and 6.0 times chance and estimation
combined. The drift is the shift of Lecture 11: the 2022 to 2024 cycle is not the 2018 to 2022
cycles. It is national, not Illinois's: Newton expects 10.1% of the test cycle's decks to drop,
and 8.7% did. Spend the effort where the error is: here on the drift, by recalibrating
to the latest cycle and checking why ratings drop less (inspection practice, preservation
programs), not on more passes.
:::

## AdamW and one pass train LLMs

| Today's idea | Inside language-model training |
|:----------------------|:----------------------------------------------------|
| Adam | AdamW (Loshchilov & Hutter, 2019), with a warm-up and a decaying step |
| One pass | about one pass (Komatsuzaki, 2019); up to four costs little (Muennighoff et al., 2023) |
| Shuffling | data order engineered: sources mixed and shuffled |
| Averaging | weight averaging (Izmailov et al., 2018) descends from Polyak's |
| **Next** | Lecture 14: the same loop fits a network, where Newton cannot go |

::: notes
AdamW applies weight decay directly to the weights instead of through the gradient, which
Adam's per-weight scaling would distort. Training at scale lives in the regime of Bottou and
Bousquet: more data, fewer passes, an optimizer just good enough. The published-order failure
is the same one that data pipelines guard against when they mix sources.
:::

## Exit question

A colleague streams the 2024 file county by county through one pass of Adam, gets a test loss
within 0.002 of Newton's, and wants to publish a ranking of counties by their weights. What do
you check first, and why?

**In the notebook** (`cee_l13_stochastic_gradient.ipynb`): SGD, momentum, and Adam from
scratch, checked against PyTorch and Newton; the step-size sweep; the published-order test.

::: notes
Answer: two things. The order: county-sorted batches bias every step, so shuffle and refit.
The weights: a close loss says nothing about flat directions, and county weights are exactly
those; fit to convergence (Newton here) and rank with standard errors. Collect the answers;
they seed the next warm-up.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- A random batch is right on average; shuffle to keep it so.
- Constant steps hop in a noise ball; average them.
- Units shape the bowl: scale the inputs.
- The loss settles before the weights.

**Next:** multilayer perceptrons and backpropagation.
:::
::: {.column}
![Four thumbnails of the lecture's key figures: batch steps that agree far from the optimum and scatter near it; the ravine of raw units against the round bowl of scaled ones; SGD at three step sizes on one bowl; the loss along one pass in FHWA's published order.](cee_l13_stochastic_gradient/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Bottou & Bousquet (2008). *NeurIPS* 20.
- Bottou, Curtis & Nocedal (2018). *SIAM Rev.* 60(2).
- Duchi, Hazan & Singer (2011). *JMLR* 12.
- FHWA (1995). *Recording and Coding Guide*, FHWA-PD-96-001.
- Izmailov et al. (2018). *UAI*.
- Kingma & Ba (2015). *ICLR*.
:::
::: {.column}
- Komatsuzaki (2019). arXiv:1906.06669.
- Loshchilov & Hutter (2019). *ICLR*.
- Mishchenko et al. (2020). *NeurIPS*.
- Muennighoff et al. (2023). *NeurIPS*.
- Polyak (1964). *USSR Comput. Math. Math. Phys.* 4(5).
- Polyak & Juditsky (1992). *SIAM J. Control Optim.* 30(4).
- Robbins & Monro (1951). *Ann. Math. Stat.* 22(3).
:::
::::::
