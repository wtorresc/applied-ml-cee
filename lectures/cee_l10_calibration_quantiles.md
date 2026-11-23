---
title: Calibration and Quantile Regression
subtitle: "Lecture 10 · What strength can we count on?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 5–9

On your own, no notes, two minutes:

1. In Lecture 5 we aimed the mix at $f'_c$ plus 9.0 MPa. Where did 9.0 come from, and what
   did it buy?
2. In 2024 the bridge model predicted that 5.0% of bridges would turn poor; 4.0% did. Which
   assumption of the cost threshold failed?
3. On the burst window, why did Huber's scale stop protecting the fit?

**Today:** what strength can a mix be counted on to reach?

::: notes
Give the answers before moving on; recall helps most with feedback.
1. The 90th percentile of over-prediction on held-out 28-day tests: aiming at f'c + 9.0 leaves
   about 10% of such mixes below f'c (Lecture 5). Today: the same idea at 5%, done properly.
2. Calibration: p > 1/(1+r) assumes that of bridges given p, a share p turns poor (Lecture 6).
   Today we measure calibration.
3. The burst, 29% of the readings, inflated the MAD scale from the calm week's 0.016 m to
   0.33 m, so burst readings looked ordinary (Lecture 9). A spread estimated on mixed data
   misleads; today the spread itself changes with strength.
:::

## Sign for the strength you count on

- A mean is exceeded half the time.
- Codes count on the **5% fractile**.

![28-day strength against water-binder ratio for 425 tests, with quantile-regression lines at 5, 25, 50, 75, and 95%, the 5% line thick, shading the 90% band, which is wider at low w/b, where strength is high (34 MPa) than at high w/b (25 MPa). A dashed line, the mean minus 1.645 sigma, runs parallel to the mean and crosses the 5% line.](cee_l10_calibration_quantiles/fan.png)

::: notes
EN 1992-1-1 defines f_ck as the characteristic (5% fractile) cylinder strength at 28 days. In
ACI 318, the specified strength f'c and the acceptance rules play the same role, and phi = 0.90
for tension-controlled flexure reduces the nominal resistance. Today we predict the 5% quantile
of a held-out test's strength, not its mean. P is Lecture 5's proposed mix. For a new design that
5% mostly reflects the model's error (held-out SD 7.6 MPa), not batch-to-batch scatter (repeat
tests, SD 1.4 MPa): it is 95% of designs like P, not 95% of batches of one design.
A mean prediction is exceeded half the time: in Lecture 5, 57% of the 3,179 held-out 28-day
predictions were above the measured strength (58% of the 425 tests). Codes count on a low
fractile: the characteristic strength is the 5% fractile (Eurocode 2), and resistance factors phi < 1 reduce it further. The decision: how much cement
makes 95% of held-out designs like P reach 35 MPa? Data: UCI Concrete, the Lecture 2 groups, 30
splits holding out whole designs: 7,689 held-out predictions. The fan: one input, w/b, to see
the spread change.
:::

## Your turn: how much margin?

| Predicted mean (MPa) | Held-out predictions | Error SD (MPa) | 5% of errors below (MPa) |
|:--|--:|--:|--:|
| below 30 | 2,537 | 5.8 | −10.3 |
| 30 to 50 | 3,724 | 7.1 | −11.4 |
| above 50 | 1,357 | 10.7 | −15.6 |
| all | 7,689 | 7.6 | −12.2 |

::: notes
Errors are measured minus predicted, for the mean model (Lecture 5's least squares on the six
groups), over the 30 splits: 7,689 predictions of the 1,030 tests. Two minutes: a mix is
predicted at 60 MPa; what strength would you sign for, so that 95% of designs predicted there
reach it? And at 25 MPa? Is one margin enough?
Answers: about 60 - 15.6 = 44 MPa, and 25 - 10.3 = 15 MPa. One margin (12.2 MPa, or 1.645 x
7.6 = 12.5) is too large for weak mixes and too small for strong ones. Keep the class's
answers; Break it measures what one margin does.
:::

## Predict the quantile, not the mean

![Diagram of the course pipeline with the loss highlighted. A mix x goes to an encoder of dimensionless groups, giving z; the decoder, a line w-tau transpose z, gives the predicted quantile q-hat-tau; the pinball loss compares it with the measured strength y. Above, 746 training tests with designs held out fix the weights by a linear program at tau equal to 0.05. Below, Lecture 5's proposed mix P, held out: cement 375 kilograms per cubic meter, water 186, no admixtures, 28 days. The mean prediction is 39.77 MPa; the mean minus 1.645 times 7.34 is 27.70; the 5% quantile line gives 28.88 MPa. The measured strength, 38.21 MPa, is above both 5% values.](cee_l10_calibration_quantiles/quantile_pipeline.png)

::: notes
Same encoder as Lectures 2 and 5; the decoder is still a line, but it targets a quantile, and
the loss changes. For P the two 5% values differ by 1.2 MPa; Break it shows where they differ
more.
:::

## The pinball loss finds quantiles

**Assumption:** the $\tau$-quantile of strength is a weighted sum of the groups; the spread may
change with the mix.

$$\rho_\tau(r)=\begin{cases}\tau\,r & r\ge0\\(\tau-1)\,r & r<0\end{cases}\qquad r=y-q$$

- Why it works: $\dfrac{d}{dq}\,\mathbb{E}\,\rho_\tau(Y-q)=F(q)-\tau$, zero at the
  $\tau$-quantile.
- At $\tau=0.05$ a reading below $q$ costs 0.95 per MPa; above, 0.05.
- The mean model's alternative assumes normal errors of one size: $q=\hat{y}-1.645\,\sigma$.

::: notes
Derivation: E rho(Y - q) = tau * integral over y > q of (y - q) dF + (1 - tau) * integral over
y < q of (q - y) dF. Differentiating in q gives -tau (1 - F(q)) + (1 - tau) F(q) = F(q) - tau.
So the expected loss is minimized where F(q) = tau: the loss is minimized by the true quantile,
which makes it a proper scoring rule for quantiles. The asymmetry is the cost ratio of Lecture 6
in another form: over-predicting the 5% value is 19 times worse than under-predicting it.
:::

## The tilt moves the minimum

![Two panels. Left: the pinball loss against the residual for tau 0.05, 0.5, and 0.95: tilted V shapes; at 0.05 a residual below zero costs 19 times one above. Right: the average 5% pinball loss of a constant guess q over the 425 tests at 28 days, a smooth curve over their histogram, lowest at 15.4 MPa, which is the 5% point of the strengths.](cee_l10_calibration_quantiles/pinball.png)

::: notes
The loss charges 0.95 per MPa for a test below the guess and 0.05 per MPa for one above. Moving
the guess down past a test saves 0.95 and costs 0.05 on the rest, so the average falls until
only 5% of the tests lie below: the minimum is the 5% quantile, 15.4 MPa here. With inputs, the
same loss finds a quantile for each mix.
:::

## Quantile regression is a linear program

Split each residual into parts: $y_i-\mathbf{a}_i^\top\mathbf{w}=u_i^+-u_i^-$,
$u^\pm\ge0$:

$$\min_{\mathbf{w},\,\mathbf{u}^\pm}\ \sum_i\big(\tau\,u_i^++(1-\tau)\,u_i^-\big)\quad\text{s.t.}\quad\mathbf{A}\mathbf{w}+\mathbf{u}^+-\mathbf{u}^-=\mathbf{y}$$

- Linear objective, linear constraints: an LP (Koenker & Bassett, 1978).
- An optimal line passes **exactly through** $d+1$ tests, here 7, as a median passes through
  a point.
- With one constant, the solution is the $\lceil\tau n\rceil$-th smallest strength.

::: notes
At the optimum, u+ and u- are not both positive for one test (that would waste cost), so their
weighted sum is the pinball loss. Vertex solutions of the LP interpolate as many tests as there
are weights. The constant case: the minimizer of the mean pinball loss over a constant is the
empirical quantile; the notebook checks that the 38th smallest of 746 strengths, 10.76 MPa, is
that minimizer.
:::

## Your turn: is calibration enough?

A forecaster quotes the same value for every mix: the 5% quantile of all training strengths,
11.0 MPa.

1. What share of held-out tests falls below it?
2. Is it calibrated? Would you use it to design a mix?
3. How would you score it against a model that changes with the mix?

::: notes
Two minutes in pairs. The answers come in Build and Break it: 5.2% below (calibrated on
average), useless for design, and the pinball loss exposes it. A score that rewards only
calibration can be gamed; a proper scoring rule rewards calibration and sharpness together
(Gneiting & Raftery, 2007).
:::

## Your turn: predict the check

```python
def fit_quantile(Z, y, tau):
    A = np.column_stack([np.ones(len(Z)), Z])
    n, d = A.shape
    # variables: w (free), u+ >= 0, u- >= 0 with A w + u+ - u- = y
    c = np.concatenate([np.zeros(d), np.full(n, tau),
                        np.full(n, 1 - tau)])
    A_eq = np.hstack([A, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * d + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds,
                  method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return res.x[:d]
```

**Predict:** of the 746 training tests, how many fall below the fitted 5% line?

::: notes
Take a show of hands: 37 (5% of 746)? fewer? more? The reference: scikit-learn's
QuantileRegressor with alpha = 0 and the HiGHS solver. Most students say 37.
:::

## Sharp and calibrated beats calibrated

LP matches scikit-learn within $10^{-11}$. Pinball: quantile regression **0.75**, mean − 1.645σ 0.77;
their gap, 0.02, has a 95% CI of −0.02 to 0.07.

![Dot plot of the mean pinball loss at tau equal to 0.05 on held-out designs: no model, the constant 5% quantile, 1.40; mean minus 1.645 sigma, 0.77; quantile regression, 0.75; repeat tests, a rough floor, 0.14.](cee_l10_calibration_quantiles/bracket.png)

::: notes
34 below and 7 on the line: 34/746 = 4.6%, and with the 7 on the line, 41 at or below; the LP
puts exactly tau n = 37.3 "between" them. The floor: if each prediction erred only by the noise
of repeat tests (SD 1.4 MPa, Lecture 5), the expected pinball loss at tau = 0.05 would be
1.4 x phi(1.645) = 0.14. Both models sit far above it.
Below the line: 34 training tests; 7 lie on it. Above: the constant 11.0 MPa, 1.40. Below: repeat-
test noise, 0.14 (rough). The gap's interval resamples whole designs (315; 2,000 resamples, seed
0), every held-out prediction of a design together; it covers which designs are tested, and the
models are not refitted. It includes zero: on this score the two models are not separated. Both
are separated from the constant: it loses 0.63 more than mean - 1.645 sigma (95% CI 0.53 to
0.73) and 0.65 more than quantile regression (0.54 to 0.75).
:::

## Predict before we test

1. **Reliability.** For $\tau$ from 5% to 95%, what share of held-out tests falls below each
   predicted quantile, for both methods?
2. **By strength.** Below 30 MPa and above 50 MPa: what share falls below mean − 1.645σ?
3. **No model.** Same question for the constant 11.0 MPa.
4. **Crossing.** Fit the 5% and the 10% lines separately. Can the 5% line lie above the 10%?

::: notes
One minute; students commit in writing. Each of the next slides reveals one answer: overall
calibration (1), calibration where it matters (2), calibration without sharpness (3), and a
defect of separate fits (4).
:::

## Both look calibrated overall

Every level from 5% to 95% within 2.3 points. At 5%: **5.0%** and 5.5%.

![Reliability diagram: held-out tests below the predicted quantile against the nominal level tau, at 5, 10, 25, 50, 75, 90, and 95 percent, for the mean plus or minus z sigma and for quantile regression. Both curves follow the diagonal closely: 5.0 and 5.5 percent at 5 percent, 51.3 and 48.9 at 50, 93.8 and 93.4 at 95.](cee_l10_calibration_quantiles/reliability.png)

::: notes
A reliability diagram groups predictions by their nominal level and compares with the observed
frequency (for probabilities, Lecture 6's tenths of risk). Both methods pass this check. The
largest gap is 2.3 points: mean - z sigma at 25% gives 22.7%. Shares are pooled over all 7,689
held-out predictions, as in the next slides.
At 5%: mean - 1.645 sigma, 5.0%; quantile regression, 5.5%. A reliability diagram over all tests
cannot see where the misses are.
:::

## Calibrated overall, unsafe when strong

Mean − 1.645σ: **13.5%** (95% CI 6.8 to 21.5) of strong mixes fall below it; quantile
regression: 6.4% (2.7 to 11.1).

![Plot of held-out tests below the predicted 5% quantile, in percent, in three bands of predicted mean strength: below 30, 30 to 50, and above 50 MPa. Mean minus 1.645 sigma: 2.9, 3.4, and 13.5 percent. Quantile regression: 7.6, 3.9, and 6.4 percent. No model: 12.9, 0, and 0 percent. A dotted line marks 5 percent.](cee_l10_calibration_quantiles/coverage_bands.png)

::: notes
This is the lecture's decisive figure. The error SD grows from 5.8 to 10.7 MPa across the bands
(the attempt's table), so a constant margin covers 97% of weak mixes and 86.5% of strong ones.
Quantile regression's line has its own slopes, so its margin grows with strength. It is not
perfect: 7.6% below 30 MPa, but its 95% CI, 4.7 to 10.7, includes 5%: not separated from 5%.
Mean - 1.645 sigma: 13.5% of tests predicted above 50 MPa fall below it; 2.9% of those below 30
MPa. Quantile regression: 6.4% and 7.6%. Its line follows the spread. One margin wastes cement
on weak mixes and is unsafe on strong ones. The 95% intervals resample whole designs (2,000
resamples, seed 0): the constant margin's excludes 5%, quantile regression's includes it. The
two intervals overlap; what the data establish is that one margin misses its 5% on strong mixes.
:::

## One margin is flat; the quantile's tilts

![3D plot of the margin below the mean, mean minus the 5% value, over water-binder ratio, 0.30 to 0.70, and age, 7 to 365 days, for mix P's family. Mean minus 1.645 sigma: a flat orange sheet at 12.1 MPa. Quantile regression: a tilted plane from about 2.9 MPa for weak mixes, high w/b at young age, to 20.7 MPa for strong ones, low w/b at old age, crossing the flat sheet.](cee_l10_calibration_quantiles/sheets.png)

::: notes
Mean minus 1.645 sigma subtracts the same 12.1 MPa from every mix. Quantile regression fits its
own line, so the gap below the mean follows the spread: 2.9 MPa where strength is low, 20.7
where it is high. The flat sheet gives weak mixes too much margin and strong mixes too little:
the 13.5% of strong mixes below it on the last slide.
:::

## No model is calibrated too

A constant 11.0 MPa: **5.2%** below, on target. Pinball **1.40**, nearly twice the models'.

![Histograms of the predicted 5% strength for the 7,689 held-out predictions: the constant, no model, is a single tall spike near 11 MPa; quantile regression's values spread mostly from 0 to 50 MPa; 139 of them (1.8%) fall below zero, down to −14.4 MPa.](cee_l10_calibration_quantiles/sharpness.png)

::: notes
Calibration alone can be achieved by ignoring the inputs. Gneiting, Balabdaoui and Raftery
(2007): maximize sharpness subject to calibration. Proper scoring rules (pinball for quantiles,
the log score and the Brier score for probabilities, CRPS for whole distributions) reward both
at once. Use calibration to diagnose, a proper score to compare.
12.9% of weak mixes fall below the constant, and no strong ones: it says nothing about the mix. A
proper score rewards sharpness too: the constant's pinball loss is 1.82 times that of mean - 1.645
sigma and 1.88 times quantile regression's.
:::

## Separate quantiles can cross

Separate 5% and 10% fits: for **32 tests** the 5% line lies above the 10%. Sort, or fit jointly.

![Scatter of the 10% value minus the 5% value against the predicted median for all 1,030 tests, fitted separately: most gaps are 0 to 5 MPa, but 32 tests, in orange, fall below zero, by up to 1.5 MPa, mostly at low predicted strength.](cee_l10_calibration_quantiles/crossing.png)

::: notes
Each level is its own LP, so nothing keeps the lines in order; near the edge of the data their
slopes can cross. Of the 32 tests, 31 have a predicted median below 30 MPa (34% of all tests do),
and 16 are among the tenth of tests with the highest leverage. Sorting the predicted quantiles of each test (rearrangement; Chernozhukov,
Fernández-Val & Galichon, 2010) never makes them worse. Conformalized quantile regression
(Lecture 11) adds a coverage guarantee on top.
For 32 tests (3.1%) the 5% line is above the 10% line; the largest crossing is 1.5 MPa. A "5%
value" above a "10% value" is a contradiction. Fixes: fit the levels jointly with order
constraints, or sort the predicted quantiles.
:::

## Count on the 5% line: 468 kg

Mean at 35 MPa: **337** kg/m³. 95% of designs like P: **468** (quantile), 454 (normal).

![Predicted 28-day strength against cement content for mix P's family, from 250 to 540 kilograms per cubic meter. The mean curve crosses 35 MPa at 337 kilograms. The mean minus 1.645 sigma crosses at 454 and the 5% quantile regression line at 468; the two lower curves cross each other near 420 kilograms, where the quantile line's margin grows past the constant one.](cee_l10_calibration_quantiles/cement_quantile.png)

::: notes
w/c: 0.55 at the mean, 0.41 by the normal margin, 0.40 by the quantile line. These are one
split's fits, the split that holds out P. Over the 30 splits the quantile line needs 441 to 491
kg (median 468), the normal margin 444 to 468 (median 456).
At 468 kg the predicted mean is 48.1 MPa, in the 30 to 50 band, where both margins already
cover more than 95%: 3.4% and 3.9% of tests fall below them (95% CIs 2.0 to 5.0 and 2.2 to
5.7), and the error SD, 7.1 MPa, is below the overall 7.6. So the extra cement does not come
from this band's spread but from the quantile line's tilt: its margin at 468 kg is 13.1 MPa,
against the constant 12.1. Across the splits it asks a median 13 kg more than the normal margin
(-12 to +31; more in 26 of 30). Trial batches and acceptance tests still decide (Lecture 2); the
model sets the starting point. 468 kg is inside the tested range (cement up to 540).
The family of Lecture 5's mix P (water 186 kg/m³). Lecture 5's 90% margin gave 418. Each step of
assurance costs cement and carbon.
:::

## Use quantiles inside the data

| Do not | Because |
|:------------------------------|:------------------------------------|
| go to the far tails | about 37 tests per split define the 5% line; fewer, noisier |
| leave the tested ratios | or use new materials (Lecture 5) |
| check coverage only overall | the risk is spread by strength: report bands |
| replace acceptance tests | cylinders accept concrete; the model plans it |
| trust a 5% value below zero | the line extrapolates: 1.8% of held-out values, to −14.4 MPa |

::: notes
When not to use the method is part of the method. A 1% quantile from 746 tests rests on about 8
tests; extreme quantiles need models of the tail (extreme value theory) or more data.
:::

## Who carries the 5%?

| The 5% | Who |
|:--------------------------|:--------------------------------------------|
| one batch in 20 falls short | covered by resistance and load factors set by code committees |
| a model calibrated only on average | moves the risk onto strong mixes, used where loads are high |
| more cement for more assurance | cost and CO₂ (Lecture 2) |

::: notes
Ask: who decides that 5% is acceptable, and who should know that the model's 5% is 13.5% for
strong mixes? Calibration by subgroup is the same idea as disaggregated evaluation (Lecture 27).
:::

## Vote: which 5% value do you sign?

A strong mix: predicted mean 60 MPa. Specified $f'_c=45$ MPa.

- **A.** Mean − 1.645σ: 60 − 12.1 = 47.9 MPa. Sign.
- **B.** Quantile regression's 5% value for this mix.
- **C.** The mean, 60 MPa: plenty of margin.
- **D.** Trial batches first; the model only plans.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
A covers only 86.5% of strong mixes (13.5% below); C is the mean; B fits the spread, about
6.4% below in this band; D is always right in practice, and B tells you whether the design is
likely to pass its trial batches. The good answer is B and D together. Then ask what would change the
answer: a producer's own records with less scatter, or a mix outside the tested ratios.
:::

## Quantile losses run modern AI

| Where | Today's idea, inside | Lecture |
|:----------------|:------------------------------------------|:----------|
| Forecasting | Temporal Fusion Transformer: pinball loss at three levels (Lim et al., 2021) | 20 |
| Weather | GenCast scored with the CRPS: pinball over all levels (Price et al., 2025) | — |
| Guarantees | conformalized quantile regression (Romano et al., 2019) | 11 |
| Language models | calibrated about their own answers (Kadavath et al., 2022) | 19 |

::: notes
The Temporal Fusion Transformer predicts the 10th, 50th, and 90th percentiles with the quantile
loss. The CRPS of a forecast distribution equals twice the integral of the pinball loss over all
levels tau, so it is the proper score for whole distributions. Romano, Patterson and Candès
calibrate quantile regression on held-out data so that coverage holds without assumptions on the
model. Kadavath et al. find that large language models are reasonably well calibrated on
multiple-choice questions when asked about the probability their own answer is true. Same
question in every case: is "90%" really 90%, for this kind of case?
:::

## Exit question

A producer's quantile model is calibrated over all its mixes: 5.0% fall below the 5% line. Why
is that not enough to sign a high-strength mix? What would you ask for?

**In the notebook** (`cee_l10_calibration_quantiles.ipynb`): the LP checked against
scikit-learn, the reliability diagram, coverage by band, crossing, and the cement decision.

::: notes
Answer: overall calibration can hide miscalibration by subgroup (13.5% below for strong mixes
with a constant margin) and is achieved even by a constant. Ask for coverage by strength band
and a proper score (pinball loss) on held-out designs, and for the share below the 5% line among
mixes like this one. Collect the answers; they seed the next warm-up.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- The pinball loss's tilt finds a quantile.
- Calibrated overall is not enough: check by band.
- A constant can be calibrated: score sharpness.
- One margin is flat; the spread is not.

**Next:** conformal prediction and shift, at Illinois stream gauges.
:::
::: {.column}
![Four thumbnails of the lecture's key figures: the pinball loss and its minimum at the 5% point; the margin below the mean, flat against tilted; coverage by strength band; cement for 35 MPa at the mean and at the 5% line.](cee_l10_calibration_quantiles/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- CEN (2004). *EN 1992-1-1, Eurocode 2*.
- Chernozhukov, Fernández-Val & Galichon (2010). *Econometrica* 78(3).
- Gneiting, Balabdaoui & Raftery (2007). *JRSS B* 69(2).
- Gneiting & Raftery (2007). *JASA* 102.
- Kadavath et al. (2022). arXiv:2207.05221.
:::
::: {.column}
- Koenker & Bassett (1978). *Econometrica* 46(1).
- Lim et al. (2021). *Int. J. Forecasting* 37(4).
- Price et al. (2025). *Nature* 637.
- Romano, Patterson & Candès (2019). *NeurIPS*.
- Yeh (1998). *Cem. Concr. Res.* 28(12).
:::
::::::
