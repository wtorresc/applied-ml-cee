---
title: Outliers and Robust Estimation
subtitle: "Lecture 9 · When did the pipe burst?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 4–6

On your own, no notes, two minutes:

1. A screen flags bridges when $p>0.5$ and scores 96% accuracy. Is it a good screen? What
   would you report instead?
2. Each Newton step for logistic regression solves a weighted least squares. What are the
   weights, and which bridges get the largest?
3. $k$-means spent a center on four buildings that run all night. Error or finding?

**Today:** when did the pipe burst, and can a fit trust readings that contain the burst?

::: notes
Give the answers before moving on; recall helps most with feedback.
1. No: in 2024, p > 0.5 flagged no bridge, and flagging nobody also scores 96%. Report what the
   screen catches and what it flags, or the cost with r, and flag above 1/(1+r) (Lecture 6).
2. p(1 - p): largest for bridges near p = 0.5, the uncertain ones (Lecture 6). Today the
   weights come from the residuals instead, and shrink the points that do not fit.
3. It depends on the question. For typical days, an outlier that takes a center; for an energy
   audit, the finding (Lecture 4). Today: is a reading an error to ignore, or the event we want?
:::

## Find the burst before the bill

- 33 pressure sensors, every 5 minutes.
- A burst pulls n229 below its neighbor's line.

![Map of L-Town's 905 pipes in grey, with its 33 pressure sensors as blue dots; sensor n229 is ringed next to the March 2018 burst on pipe p673, marked with an orange star; its neighbor n636, also ringed, lies a few hundred meters away.](cee_l09_robust_estimation/network.png)

::: notes
BattLeDIM, the Battle of the Leakage Detection and Isolation Methods (Vrachimis et al., 2022),
published the SCADA readings of L-Town, a realistic town network, generated with its hydraulic
model, real demand patterns, and sensor noise; the leak times, sizes, and pipes are released,
which is why we can grade a method. Licence: CC BY 4.0. The burst in pipe p673 started at 15:45
on March 5, 2018, and was repaired on March 23; it leaked 28.4 m3/h, 13% of the town's inflow.
The decision: send a leak crew to the district now, or wait. A false alarm costs a crew day; a
miss loses water. The March 2018 burst leaked 28 m³/h, 680 m³ a day. Data: BattLeDIM's L-Town:
782 junctions, 905 pipes (43 km), 33 pressure sensors read every 5 minutes. Simulated, so every
leak is known: 14 in 2018. The catch: the readings we fit on may already contain a burst, or a
faulty sensor.
:::

## Your turn: did a pipe burst?

| Hourly mean pressure (m) | Mar 4, 14:00 | Mar 4, 17:00 | Mar 5, 14:00 | Mar 5, 17:00 |
|:--|--:|--:|--:|--:|
| Sensor n229 | 52.48 | 51.98 | 51.80 | 50.64 |
| Neighbor n636 | 45.47 | 45.07 | 44.91 | 44.60 |
| Town inflow (m³/h) | 188 | 225 | 242 | 257 |

::: notes
Two minutes on your own: did a pipe burst? If so, near which sensor, and when? Write down the
evidence you used. The two sensors are about 340 m apart; both pressures fall in the evening as
demand rises.
Answers: from 14:00 to 17:00 on March 4, n229 fell 0.50 m and its neighbor 0.40 m: the usual
evening. On March 5, n229 fell 1.16 m and its neighbor only 0.31 m. The calm line's slope
predicts 1.24 × 0.31 = 0.38 m for n229, so n229 lost 0.78 m more than demand explains (0.85 m
more than its neighbor). A burst near n229, between 14:00 and 17:00 (it began at 15:45, about
100 m from n229). Note what you did: you compared n229 with what its neighbor predicts. That is the
model; the question is how to fit it when the data contain the burst.
:::

## The loss decides who is normal

![Diagram of the course pipeline with the loss highlighted. The neighbor sensor n636's pressure x goes to a decoder, a line w0 plus w1 x, giving y-hat; the residual r is the measured pressure of sensor n229, y, minus y-hat; an alarm sounds if the hourly mean of r falls below minus 5 sigma. Above, the robust loss, Huber weights or RANSAC, fits the line once on 2,304 readings; RANSAC keeps 1,626 within tau equal to 0.05 meters. Below, at 17:00 on March 5, 2018: x is 44.64 m; minus 3.786 plus 1.2376 times 44.64 gives y-hat 51.46 m; y is 50.75 m, so r is minus 0.71 m, minus 43 sigma, with sigma 0.017 m; the burst began at 15:45, 100 m away: alarm.](cee_l09_robust_estimation/robust_pipeline.png)

::: notes
This is the single reading at 17:00 on March 5 (50.75 m), 75 minutes after the burst; your
attempt's table showed that hour's mean, 50.64 m. The line comes from
RANSAC on a window that contains the burst's first days. The residual is 43 times the noise of
a calm week. The question for the next slides: which loss lets the fit ignore the burst?
:::

## Assume a few readings lie

**Assumption:** most readings follow one line; a minority, the outliers, follow anything.

- Least squares: $\min_{\mathbf{w}}\sum_i r_i^2$. A point's pull grows with its residual: one
  bad reading can move the line anywhere.
- **Huber** (1964): squared near zero, linear beyond $\delta$:
  $$\rho(r)=\begin{cases}r^2/2 & |r|\le\delta\\ \delta|r|-\delta^2/2 & |r|>\delta\end{cases}\qquad\psi(r)=\rho'(r)=\mathrm{clip}(r,-\delta,\delta)$$
- Scale it: $\delta=1.345\,\hat\sigma$, with $\hat\sigma=\mathrm{median}\,|r_i|/0.6745$, which
  ignores the outliers.

::: notes
The influence of a point on the least-squares fit is proportional to its residual, so it is
unbounded: the breakdown point of least squares is 0 (a single point can carry the fit away).
Huber's psi is bounded: a far point pulls with a constant force delta. The constant 1.345 gives
95% of least squares' efficiency when the noise is normal. MAD/0.6745 estimates sigma for
normal noise, and the median ignores up to half the points. Forsyth §11.2 derives M-estimators
and their fitting by reweighting.
:::

## Huber caps each reading's pull

![Two panels against the residual r in units of delta. Left, the loss: squared, a parabola that keeps steepening; absolute, a V; Huber, a parabola within one delta of zero and straight lines beyond. Right, each reading's weight in iteratively reweighted least squares: 1 for squared everywhere; for Huber, 1 within delta, then delta over the absolute residual, falling to 0.25 at four deltas.](cee_l09_robust_estimation/losses.png)

::: notes
Least squares' pull grows with the residual, so the farthest reading pulls hardest. Huber keeps
the squared loss near zero (efficient when the noise is normal) and grows only linearly beyond
delta, so a far reading's pull is capped. In IRLS the cap is a weight: delta / |r|. The next slide
derives it.
:::

## Huber is least squares, reweighted

Set the gradient of $\sum_i\rho(r_i)$ to zero: $\sum_i\psi(r_i)\,\mathbf{a}_i=\mathbf{0}$.
Write $\psi(r)=w(r)\,r$:

$$\sum_i w_i\,r_i\,\mathbf{a}_i=\mathbf{0},\qquad w_i=\min\!\Big(1,\frac{\delta}{|r_i|}\Big)$$

- Fix the weights: a **weighted least squares** for $\mathbf{w}$. Update the weights from the
  new residuals; repeat (IRLS).
- The same loop as Lecture 6's Newton steps; only the weights differ.
- Far points get weight $\delta/|r|$: they still pull, with a bounded force.

::: notes
Iteratively reweighted least squares: start from least squares, compute residuals and the
scale, set w_i, solve the weighted least squares (with QR or lstsq on sqrt(w) A, never an
inverse), repeat until the weights stop moving. Huber's rho is convex, so the minimum is unique
for a fixed scale. In Lecture 6 the weights were p(1 - p); here they are min(1, delta/|r|).
Ask: which readings will get small weights in our window?
:::

## RANSAC lets the data vote

**Fischler & Bolles (1981):** fit many minimal samples; keep the line most readings agree with.

1. Draw 2 readings at random; fit the line through them.
2. Count **inliers**: $|r_i|<\tau$. Keep the largest set.
3. Refit by least squares on that set.

Trials for probability $p$ of one clean pair, outlier share $\varepsilon$, sample size $s=2$:

$$N=\frac{\ln(1-p)}{\ln\!\big(1-(1-\varepsilon)^s\big)}$$

::: notes
A bad reading cannot vote against the consensus: it is simply not an inlier. The threshold tau
is an assumption about the noise: a reading within tau of the line is believed. RANSAC needs no
outlier model, only the noise scale and a minimal sample.
:::

## RANSAC, trial by trial

![Three panels of the burst window, n229 against n636: two parallel bands of readings, the calm line above and the burst line below. Trial 1: a random pair, both on the calm band, gives a line with 956 readings within tau, in blue. Trial 2: a pair with one reading on each band gives a steep line with only 324 inliers. Best of 200 trials, refit by least squares on its consensus: the calm line, 1,626 inliers, no burst reading among them.](cee_l09_robust_estimation/ransac_build.png)

::: notes
Each trial draws two readings, draws the line through them, and counts the readings within tau
= 0.05 m (three standard deviations of the calm week's noise). A pair from two different bands
gives a line few readings agree with. The best of 200 trials is refit by least squares on its
inliers.
:::

## Your turn: how many trials?

At $p=0.99$ and $s=2$, how many RANSAC trials do you need when

1. $\varepsilon=0.3$ of the readings are outliers?
2. $\varepsilon=0.5$?
3. $\varepsilon=0.7$?

Then: what sets $\tau$, and what happens if it is ten times too large?

::: notes
Two minutes with a calculator. Answers: 7, 17, and 49 trials (round up). Our window holds 29%
burst readings, so about 7 to 9 trials suffice; we run 200. A tau ten times too large lets
readings from both regimes agree with a line between them: the next slides show it happening.
Tau comes from the noise of the relation, measured on a calm week, February 20 to 27: no leak
near n229 (three slow leaks ran elsewhere, the nearest 566 m away). Here sigma = 0.017 m and
tau = 3 sigma = 0.05 m.
:::

## Your turn: predict the check

```python
def huber_irls(A, y, t=1.345, tol=1e-12, max_steps=500):
    w = np.linalg.lstsq(A, y, rcond=None)[0]     # start: least squares
    for step in range(1, max_steps + 1):
        r = y - A @ w
        s = max(np.median(np.abs(r)) / 0.6745, 1e-8)  # scale, never 0
        u = np.abs(r) / s
        wt = np.where(u <= t, 1.0, t / u)         # Huber weights
        sw = np.sqrt(wt)
        w_new = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        if np.max(np.abs(w_new - w)) < tol:
            return w_new, step, s
        w = w_new
    raise RuntimeError("IRLS did not converge")
```

**Predict:** on the burst window, will Huber's slope be near the calm 1.24, or near least
squares' 1.63?

::: notes
Take a show of hands. The reference: statsmodels' RLM with the HuberT norm and the same MAD
scale. Most students expect Huber near the calm slope: it was built for outliers.
:::

## The check passes; Huber does not

IRLS matches statsmodels to $10^{-12}$. Huber: **1.65**. RANSAC, $\tau=0.05$ m: **1.24**, the calm fit's.

![Dot plot of the slope of n229 on n636 fitted on the burst window, against the slope of a fit on the calm readings only, 1.24, marked by a dotted line: moving one to one, 1.00; least squares, 1.63; Huber, 1.65; RANSAC with tau equal to 3 sigma, 1.24, on the dotted line.](cee_l09_robust_estimation/bracket.png)

::: notes
IRLS took 17 steps. No burst reading is among RANSAC's 1,626 inliers of the 2,304 readings.
The verification is of the code, not of the method: IRLS computes Huber's estimate correctly,
and that estimate is wrong here. scikit-learn's HuberRegressor (no penalty), which estimates the
scale jointly, gives 1.66. Why Huber fails: the burst readings are not isolated large residuals;
they form a coherent group, a parallel line 0.73 m lower, and the MAD scale grew from the calm
week's 0.016 m to 0.33 m, so most burst readings look ordinary. The calm fit uses only the
readings before 15:45 on March 5: slope 1.238. Successive residuals are correlated (0.44 five
minutes apart, 0.33 a day apart), so its standard error depends on the window allowed for that:
0.0023 with Newey–West over two hours, 0.0009 over one day (which spans the daily cycle), 0.0012
for independent readings. Every version leaves Huber's 1.65 far outside.
The burst is 29% of the readings, and it inflated the scale to 0.33 m. Baseline "they move 1:1":
slope 1.00.
:::

## Predict before we test

1. **A fault.** Three readings of n636 recorded as 0 m among 1,629 calm ones. Slopes of least
   squares, Huber, RANSAC?
2. **The burst.** Which line does each method fit through the burst window, and how big a
   burst does each report?
3. **The threshold.** RANSAC with scikit-learn's default threshold, over 20 seeds?
4. **The majority.** Grow the window until the burst is 65% of it. Which line does RANSAC keep?

::: notes
One minute; students commit in writing. Each of the next slides reveals one answer, and each
tests a piece of the assumption: isolated bad readings (1), a structured minority (2), the
noise scale behind tau (3), and the minority itself (4).
:::

## One dropout breaks least squares

![Two 3D loss surfaces over the line's level and slope for 1,629 calm readings in which three of n636's readings were recorded as 0 m, each relative to its own minimum. Least squares: the bowl's bottom has moved to slope 0.05, far from a green line that marks the clean slope, 1.24. Huber: the bottom stays at slope 1.22, beside the green line.](cee_l09_robust_estimation/dropout_surface.png)

::: notes
The zeros are injected into the calm part of the window (before the burst) to simulate a
telemetry dropout, a common SCADA fault; the BattLeDIM readings themselves have none. Each zero
is a leverage point about 45 m from the others in x (they average 45.44 m), so least squares
tilts the line through them. Huber's weights fall as delta/|r|; these residuals are about 55 m
(55.18 to 55.39), so the zeros get weights of about 0.0005. Lesson: least squares has breakdown point 0; Huber handles gross, isolated errors.
Three zeros in 1,629 readings: least squares' slope falls from 1.24 to 0.05. Huber: 1.22. RANSAC:
1.24. Both shrug off isolated faults. A dropout recorded as 0 is a sensor fault, not a leak: one
sensor jumps, its neighbors do not. The loss decides who is normal: each zero sits 45 m from the
others, and squared error lets three readings carry the bowl's bottom away.
:::

## A burst is a second line

Least squares and Huber cut between the bands (1.63, 1.65). RANSAC's burst size: **0.73 m**.

![Scatter plot of n229 against n636 pressure for the 2,304 readings of the window. Before the burst the readings lie on a tight line, from 51.45 m where n636 reads 44.63 m to 53.34 m where it reads 46.16 m; after 15:45 on March 5 they lie on a parallel line 0.73 m lower. Least squares, slope 1.63, and Huber, slope 1.65, cut across both groups; RANSAC, slope 1.24, follows the upper, calm line.](cee_l09_robust_estimation/burst_lines.png)

::: notes
This is the lecture's decisive figure. The burst readings are a second, parallel line: a
structured minority, not scattered noise. Huber's assumption, a few readings with large
residuals, fails; RANSAC's, a majority within tau of one line, holds. The burst sizes are the
mean residual during the burst minus the mean before it, so a method that tilts the line also
shrinks the burst it reports.
They report 0.53 and 0.52 m; RANSAC, on the calm line, reports 0.73 m. Huber bounds each reading's
pull, not 675 that agree.
:::

## RANSAC is only as good as τ

Default $\tau$ (MAD of $y$): 0.44 m, 13 of 20 seeds off. $\tau=3\sigma=0.05$ m: all 20 agree.

![Strip plot of RANSAC slopes over 20 random seeds. With the default threshold, 0.44 m, the slopes scatter: 7 near 1.24, two near 1.16, and eleven between 1.75 and 2.0. With 3 sigma of a calm week, 0.05 m, all 20 lie on the clean slope, 1.24, marked by a dotted line.](cee_l09_robust_estimation/ransac_seeds.png)

::: notes
A threshold of 0.44 m is more than half (0.36 m) the 0.73 m between the bands, so a line
midway between them holds readings of both bands as inliers. The default is a
reasonable guess when nothing is known about the noise; here we know it from a calm week. The
tau assumption is engineering knowledge: sensor resolution and the relation's own noise.
scikit-learn's default threshold is the MAD of y: 0.44 m. It measures the daily demand cycle, not
the noise. With it, 13 of 20 seeds return a slope off by more than 0.05 (1.15 to 1.99). With tau
= 3 sigma = 0.05 m from a calm week, all 20 seeds agree: 1.230 to 1.242.
:::

## The biggest tight group wins

RANSAC keeps the calm line up to 56% burst, narrowly. At **65%** the burst becomes "normal".

![Line plot of the fitted slope against the share of burst readings in the window, from 19% to 65%. Least squares stays between 1.58 and 1.66. RANSAC with tau equal to 3 sigma stays at the calm slope, 1.24, up to 56%, past the dotted line at one half, then jumps to 1.35 at 65%.](cee_l09_robust_estimation/breakdown.png)

::: notes
RANSAC keeps the line with the most readings within tau, not the line of the majority. The
burst readings spread more about any one line: the best line through them alone holds 76 to 78%
of them within 0.05 m, the best through the calm readings 100% (1,627 of 1,629). So the calm
line survives past one half: at 56% it holds 1,627 readings against the burst line's 1,601, by
26. At 65% the burst line holds 2,281 and wins. Once the burst
dominates, the residual flips sign: a model refit on recent data learns the leak as normal.
Always fit the baseline on a period you have reasons to trust.
It keeps the calm line while the burst is 19%, 37%, 49%, even 56% of the window. At 65% it
switches, and the calm days look like a pressure rise of 0.68 m. Least squares sits between the
two lines throughout (1.58 to 1.66).
:::

## Alarm in five minutes

Alarm below $-5\sigma=-0.08$ m: first at **15:50**, five minutes after the burst. None before.

![Time series of n229's residual from the calm week's line, March 3 to 8, 2018. Before the burst the residual stays within a few centimeters of zero, above the dashed alarm line at minus 0.08 m. At 15:45 on March 5, marked by a dotted line, it drops to about minus 0.75 m and stays between minus 0.6 and minus 0.8 m, with a daily cycle.](cee_l09_robust_estimation/alarm.png)

::: notes
The hourly mean is the mean of the last 12 readings, so the first alarm comes as soon as two
readings after the burst pull the mean below the threshold. No alarm in the 2,016 readings of
the week before, nor earlier in 2018. The crew goes to n229's district: n229 is about 100 m
from the burst; the neighbor, 370 m. Which district to search comes from which sensor departs
from its neighbors; BattLeDIM also scored how close teams placed the leak. Five sigma: with normal noise,
a reading below it happens about once in 3.5 million; hourly means of correlated readings make
it rarer still. The cost of a false alarm sets the threshold, as in Lecture 6.
Alarm when the last hour's mean residual from the calm week's line is below -5 sigma: a line
fitted once, on the trusted week of February 20 to 27, before the burst. RANSAC's line from the
burst window, fitted through March 7, gives the same alarms. None in the week before. After
the repair, March 24 to December, no alarm at all: no false alarm, and one miss. October's leak
in pipe p158 (24.4 m³/h, nearly the burst's 28.4) lay 220 m from n229 and 140 m from n636. Both
sensors fell (0.31 and 0.24 m over its first 10 days), so the residual moved only from −0.013 m
to −0.019 m; its lowest hourly mean, −0.066 m, stayed above the −0.083 m threshold. A pair
alarm watches the gap between two sensors; a leak that lowers both leaves the gap alone.
:::

## Not for slow leaks or new normals

| Do not | Because |
|:------------------------------|:------------------------------------|
| fit on a majority of leak readings | the baseline learns the leak |
| refit a window over a slow leak | advice, not measured here: the line would move with it |
| alarm on an isolation forest alone | trained on one calm week, it flagged 13% of the hours before the burst, 92–100% of each month from May |
| trust one pair to see every leak | October's leak (24 m³/h) lowered both sensors: no alarm |
| confuse faults with leaks | one sensor jumping is a fault |

::: notes
When not to use the method is part of the method. The isolation forest (Liu, Ting & Zhou, 2008)
isolates unusual points with few random splits; on the hourly residuals of all 33 sensors,
trained on February 20 to 27 (three slow leaks were already running), it flags whatever
differs from its training week. From May it flagged 92 to 100% of hours, while the inflow ran 7
to 30% above the training week's 175.2 m³/h and 4 to 6 leaks were running: it cannot say which.
In the days before the burst no leak started, but two incipient leaks (p461, p427) were still
growing. Eight of 2018's 14 leaks were incipient: they grew for 4 to 98 days (median 50) before
reaching full size.
:::

## Lost water is lost money and energy

| Lost | How much |
|:----------------|:------------------------------------------|
| Water, worldwide | about 126 billion m³ a year never billed: leaks, theft, metering errors, unbilled use (Liemberger & Wyatt, 2019) |
| Money | about US\$39 billion a year |
| Energy | every lost m³ was pumped and treated |
| Who pays | customers, through their bills |

::: notes
Non-revenue water: water produced but not billed. Liemberger and Wyatt estimate 346 million m3 a
day. L-Town's March burst alone lost about 680 m3 a day for 18 days. Ask: in a city with old
pipes and little budget for sensors, who would a slow alarm hurt most?
:::

## Vote: which baseline for the alarm?

Your fitting week may contain an unknown event. Which estimator sets the line?

- **A.** Least squares on the last week.
- **B.** Huber on the last week.
- **C.** RANSAC with $\tau$ from a calm, checked week.
- **D.** RANSAC with the library's default threshold.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
C: on the burst window, A and B fitted 1.63 and 1.65 and shrank the burst to 0.53 and 0.52 m;
D failed for 13 of 20 seeds; C recovered the calm line and the full 0.73 m. Then ask what would
change the answer: a window that is mostly burst (C fails too: fit on a trusted period), or
isolated dropouts only (B is enough, and cheaper).
:::

## Robust losses run modern AI

| Where | Today's idea, inside | Lecture |
|:----------------|:------------------------------------------|:----------|
| Object detectors | smooth L1, a Huber loss on boxes (Girshick, 2015) | 15–16 |
| Deep Q-networks | the TD error clipped: a Huber loss (Mnih et al., 2015) | 26 |
| Point clouds, cameras | RANSAC aligns scans and poses, as in COLMAP (Schönberger & Frahm, 2016) | 22 |
| Gradient boosting | the Huber loss resists bad labels (Friedman, 2001) | 8 |

::: notes
Smooth L1 is the Huber loss with delta = 1. Mnih et al. clip the error term of the update to
[-1, 1], which equals a Huber loss on the temporal-difference error. Structure-from-motion
pipelines such as COLMAP estimate the geometry between images with RANSAC, because feature
matches contain many outliers. Friedman's gradient boosting paper derives the Huber M-estimator
as a loss for boosting. The point: the question "which points may the fit ignore?" comes with
every loss you choose.
:::

## Exit question

Your SCADA vendor proposes refitting each sensor's baseline every night on the last seven days,
with least squares. Name two ways it will miss a leak, and one change you would make.

**In the notebook** (`cee_l09_robust_estimation.ipynb`): Huber IRLS checked against statsmodels,
RANSAC from scratch, the trials formula, the four breaks, and the alarm.

::: notes
Answers: a burst in the last week pulls least squares toward it (the burst window: 0.53 m
reported for 0.73 m); a slow leak becomes part of each new baseline; a dropout tilts the line.
Changes: fit on a trusted, checked period, use RANSAC with tau from the noise, and alarm on
departures from neighbors. Collect the answers; they seed the next warm-up.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- Squared loss lets one reading move the line.
- Huber caps each reading's pull (IRLS).
- RANSAC: the largest set within $\tau$.
- Fit baselines on trusted periods.

**Next:** calibration and quantile regression: strength we can count on.
:::
::: {.column}
![Four thumbnails of the lecture's key figures: squared, Huber, and absolute losses with their weights; the dropout loss surfaces of least squares and Huber; three lines through the burst window; slopes as the burst grows.](cee_l09_robust_estimation/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Fischler & Bolles (1981). *Comm. ACM* 24(6).
- Forsyth (2019). Springer, §11.2.
- Friedman (2001). *Ann. Statist.* 29(5).
- Girshick (2015). *ICCV*.
- Huber (1964). *Ann. Math. Statist.* 35(1).
- Liemberger & Wyatt (2019). *Water Supply* 19(3).
:::
::: {.column}
- Liu, Ting & Zhou (2008). *ICDM*.
- Mnih et al. (2015). *Nature* 518.
- Schönberger & Frahm (2016). *CVPR*.
- Vrachimis et al. (2022). *J. Water Resour. Plann. Manage.* 148(12).
:::
::::::
