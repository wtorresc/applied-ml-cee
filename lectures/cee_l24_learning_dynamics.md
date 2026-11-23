---
title: Learning Dynamics from Data
subtitle: "Lecture 24 · Will the pond hold? Can a network route the flood?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 5–23

On your own, no notes, two minutes:

1. Lecture 5: ridge shrinks every weight. Which penalty sets some weights to exactly zero?
2. Lecture 20: a unit hydrograph turns rain into flow by a convolution. What must be true of
   the catchment for that to hold?
3. Lecture 23: a PINN needed the PDE. What if we do not know the equation?

**Today:** learn the equation itself from data, or learn the solver's input-to-output map.

::: notes
Give the answers before moving on.
1. The lasso's absolute-value penalty (Tibshirani, 1996). Today a simpler cousin: least
   squares, then drop the small coefficients and refit.
2. Linear and time-invariant: twice the rain gives twice the flow, at the same delay. Today
   a river whose big floods travel faster breaks that.
3. Two answers today: search a library of candidate terms for a sparse equation (SINDy), or
   learn the map from input function to output function (a neural operator).
:::

## A pond with an unknown outlet

| The pond (assumed values; records simulated, so the truth is known) | Value |
|:----------------------------------------|-----------------:|
| Area (vertical walls) | 2,000 m² |
| Outlet: an orifice (Q ∝ √h) or a weir (Q ∝ h^1.5)? | unknown |
| Storms recorded in year 1 (inflow measured) | 20 |
| Level read every | 5 min |
| Level noise (one standard deviation) | 0.005 m |
| Design storm: peak inflow | 6 m³/s |
| Rim above the outlet | 2.5 m |

::: notes
The decision: a design storm with a peak inflow of 6 m³/s, about twice
the largest recorded (3.0 m³/s). Will the pond hold it, or must the
city enlarge it? The physics is a water balance, A dh/dt = Q_in − Q_out(h); the outlet's law
Q_out(h) is the unknown. The records are simulated from a weir (Q_out = C h^1.5 with
C = 1.7), and nobody is told.
:::

## Your turn: orifice or weir?

![The largest recorded storm: left, inflow rising to about 3.0 m³/s and falling within hours; right, the pond's level rising and falling more slowly, read every five minutes.](cee_l24_learning_dynamics/attempt.png)

::: notes
Two minutes, alone. Hint: when the inflow has stopped, the level falls at Q_out(h) / A. How
would the fall look if Q_out grew like √h? Like h^1.5? Collect answers and how students would
check: plot the fall rate against the level.
:::

## Sparse regression picks the law

![Six candidate terms (inflow, a constant, h^0.5, h, h^1.5, h^2), their least-squares coefficients scaled by each column's size: the inflow and h^1.5 dominate; the other four get small, non-zero weights.](cee_l24_learning_dynamics/picture_1.png)

::: notes
The one picture, three frames. Frame 1: regress dh/dt (from the level readings, by central
differences) on all six candidate terms. Least squares gives every term some weight: small
here, because these records are clean; with noisier records the small weights grow (the
second reveal).
:::

## Sparse regression picks the law

![The same coefficients with a threshold at 5% of the largest (dotted): the terms below it in orange.](cee_l24_learning_dynamics/picture_2.png)

::: notes
Frame 2: drop every coefficient smaller than a threshold (here 5% of the
largest, on columns scaled to the same size), refit on the terms left, repeat. Sequentially
thresholded least squares, the regression inside SINDy (Brunton et al., 2016).
:::

## Sparse regression picks the law

![After thresholding and refitting: two terms left, the inflow and h^1.5.](cee_l24_learning_dynamics/picture_3.png)

::: notes
Frame 3: two terms survive, the inflow (1/A) and h^1.5: a weir. Read off the pond: area
2,015 m² (true 2,000), weir coefficient 1.70
(true 1.70). An equation, not a black box: it can be checked, compared with a
handbook, and run beyond the data.
:::

## SINDy: a library and a threshold

:::::: {.columns}
::: {.column}
$$\frac{dh}{dt} \approx \Theta(Q_{in}, h)\,\xi$$

Θ: one column per candidate term; ξ: their coefficients, most of them zero.
:::
::: {.column}
| Step | What it does |
|:-----|:-------------|
| Derivative | central differences of the level |
| Library | Q_in, 1, h^0.5, h, h^1.5, h^2 |
| Fit | least squares on all columns |
| Threshold | zero the small ones, refit; repeat |
| Result | a short equation to run |
:::
::::::

::: notes
SINDy, sparse identification of nonlinear dynamics (Brunton et al., 2016). Assumptions: the
true law is a sum of a few library terms; the derivative can be estimated from the data; the
data exercise the terms enough to tell them apart. Each assumption is broken later today. The
threshold is fixed at 5% before any test; the next slides sweep it.
:::

## Your turn: predict the check

```python
xi = stlsq(library(q, h), exact_dhdt, lam)  # no noise
ls = lstsq(library(q, h), dhdt)              # all terms
d  = stlsq(library(q, h), dhdt, lam=0) - ls
v  = (route(long) - base).sum() / (long - base).sum()  # solver
```

How far is `xi` from the truth? `d` from 0? `v` from 1?

::: notes
Two minutes, written. The third line: with a threshold of zero, nothing is dropped. The
fourth: the routing solver, given a flood and 200 more hours of base flow to let it pass;
the volumes are those above the 50-m³/s base flow.
:::

## SINDy recovers an exact law exactly

| Check | Value |
|:------|------:|
| SINDy on exact derivatives: largest coefficient error (relative) | 6e-16 |
| … terms kept | 2 |
| Threshold 0 against least squares (relative) | 2e-15 |
| Solver, 50 floods, 200 h more: volume out ÷ in | 0.947 to 0.998 |
| Solver, half the reach and step: largest change, m³/s | 3.3 |

::: notes
Given the exact derivative, the regression returns the true equation, 2 terms, to round-off: the regression code is right, so later errors come from the data or the library. With a threshold of zero it is plain least squares. The routing solver is variable-parameter Muskingum–Cunge (Cunge, 1969; Ponce & Yevjevich, 1978): even 200 hours after a flood, only 94.7% to 99.8% of the water has come out (the first 50 test floods). That method is known to lose some volume (up to about 8% in Tang et al., 1999); it is the reference anyway, as in practice, and the network is scored against it. Halving the reach and the time step moves the outflow by at most 3.3 m³/s.
:::

## SINDy finds the weir

| Coefficient of | Truth | SINDy | Orifice assumed |
|:---------------|------:|------:|----------------:|
| Q_in (1/A) | 5.00e-04 | 4.96e-04 | 5.80e-05 |
| h^0.5 | 0 | 0 | -6.82e-05 |
| h^1.5 | -8.50e-04 | -8.44e-04 | 0 |

::: notes
SINDy keeps two terms: the inflow (an area of 2,015 m²) and h^1.5 (a weir coefficient of 1.70), both within 1% of the truth. Assume an orifice instead, and least squares does what it can: an area of 17,255 m² against the true 2,000, and the fit explains R² = 0.11 of the level's rate of change, against SINDy's 0.94. Here the wrong form announces itself in the residuals; the first reveal shows one that does not.
:::

## Two terms track the true level best

![Validation error of the pond's level on held-out storms against the threshold: all six terms at threshold 0, 0.00106 m; five terms at 0.001 and 0.003, 0.00125 m; two terms from 0.01 up, 0.00041 m.](cee_l24_learning_dynamics/sweep.png)

::: notes
The threshold trades terms against fit. Against the true level of storms it never saw, the sparse model is also the accurate one: two terms give a level error of 0.00041 m, all six 0.00106 m, five 0.00125 m. All three are far below the gauge's own error: scored against the gauge's readings, which carry 5 mm of noise and err 0.0051 m themselves (the noise floor), they score 0.0051 to 0.0052 m, all at the floor. The clear ranking needs the true level, which only a simulation gives. Real data are rarely this clean; the fourth reveal shows a case.
:::

## The pond holds; the orifice fit says no

![The design storm's level over a day: the truth peaks at 2.31 m, under the rim at 2.5 m; SINDy's model at 2.31 m, on top of it; the fit that assumed an orifice at 2.74 m, over the rim.](cee_l24_learning_dynamics/design.png)

::: notes
The decision. Run each fitted equation through the design storm, about twice the largest recorded peak. SINDy's model gives 2.309 m against the truth's 2.309 m: the pond holds under its 2.5-m rim. The orifice model fitted the same records badly (R² = 0.11) and says 2.74 m: overtopping, and a needless enlargement. A poor fit is a warning to heed before the design storm. Ask: without the truth, how would the city know which to trust?
:::

## Year 2: the weir passes 70% as much

| Year 2 | h^1.5 coefficient ÷ year 1 |
|:-------|---------------------------:|
| SINDy | 0.702 |
| 90% bootstrap interval (200 resamples of the storms) | 0.7004 to 0.7038 |
| Truth (the weir partly blocked) | 0.70 |

::: notes
The same storms a year later, with the weir partly blocked (debris, sediment). SINDy's h^1.5 coefficient falls to 0.702 of year 1's; a bootstrap over storms (resample the 20, refit both years) gives 0.7004 to 0.7038, just above the truth (0.70): the level noise biases it slightly. A coefficient that means something physical can be tracked over time: a maintenance trigger. Ask: what would a black-box model of the level have told the city?
:::

## A river reach that routes floods

| The reach (assumed values) | Value |
|:----------------------------------------|-----------------:|
| Length, in reaches of 2 km | 60 km |
| Width, slope, Manning's n | 50 m, 0.0005, 0.035 |
| Base flow | 50 m³/s |
| Training floods (random peaks 100–600 m³/s) | 1,000 |
| Solver | variable-parameter Muskingum–Cunge |

::: notes
The second half: a 60-km reach whose floods travel faster when they are bigger (median lag 7 h for training-size floods, 5 h for floods up to twice as large): the routing is nonlinear. The task: route a 1,000-member forecast of inflow floods every few hours. Training data: 1,000 random floods routed by the solver.
:::

## The operator learns the routing

![A test flood of median size, over 48 hours: the inflow (dotted), the solver's outflow at the end of the reach, the FNO's outflow on top of it, and the unit hydrograph's.](cee_l24_learning_dynamics/fno.png)

::: notes
A neural operator learns a map from a function (the inflow over time) to a function (the outflow), not from a fixed list of numbers to another (Li et al., 2021). The unit hydrograph is Lecture 20's linear convolution, here from inflow to outflow: Lecture 20's turned rain into flow and was fitted by ridge; this one is fitted by plain least squares, its length chosen on validation (72 hours, the whole record). Being linear, it cannot make big floods travel faster. Ask the room to compare the two peaks with the solver's.
:::

## A Fourier layer: transform, keep, mix

:::::: {.columns}
::: {.column}
$$v \leftarrow \sigma\big(W v + \mathcal{F}^{-1}(R \cdot \mathcal{F} v)\big)$$

F: the Fourier transform over time; R: learned weights on the lowest 12 modes.
:::
::: {.column}
| Part | Size |
|:-----|:-----|
| Lift | inflow and time → 32 channels |
| Fourier layers | 4, each 12 modes |
| Project | 32 → 64 → 1 |
| Weights | 104,801 |
:::
::::::

::: notes
Each Fourier layer transforms the channels over time, keeps the lowest 12 frequencies, multiplies them by learned complex weights, transforms back, and adds a pointwise linear map W and a nonlinearity σ. The weights act on frequencies, not on time steps, so the same network can read a hydrograph sampled hourly or every half hour. Settings fixed before any test, not tuned: 32 channels, 4 layers, Adam for 300 epochs, 3 seeds. Validation error was still falling at the end (2.1% to 6.9% over the last 30 epochs), so these errors are those of 300 epochs, not the network's best; longer training was not run.
:::

## In range, the FNO beats the UH

| 500 test floods | Error, m³/s | Peak error (median) | Peak timing, h (median) | Seconds |
|:----------------|------------:|-----------:|---------------:|--------:|
| Unit hydrograph (72 h) | 17.1 | 4.2% | 1 | |
| FNO, 3 seeds | 0.25–0.28 | 0.07%–0.09% | 0 | 0.08 |
| Solver | 0 | 0 | 0 | 0.65 |

::: notes
On 500 test floods in the training range, the FNO errs 0.25 to 0.28 m³/s across its 3 seeds, the unit hydrograph 17.1; median peak error at most 0.09% against 4.2%. Trained on hourly floods, it routes floods sampled every 30 minutes with an error of 1.09 to 1.10 m³/s, without retraining; on the same 200 floods sampled hourly it errs 0.25 to 0.29, so the finer sampling multiplies its error by 3.8 to 4.3 (the unit hydrograph was not run on them). The solver routes both samplings with the same 900-s internal step, so only the input's sampling changes; that alone moves the solver's own outflow by 0.23 m³/s, and halving the step too would move it by 0.59 m³/s. Speed: 0.08 s for the 500 floods, the solver 0.65 s; training took up to 216 s per seed (timings vary on a shared CPU). This solver is fast already: a surrogate pays when the solver is slow and the runs are many.
:::

## Your turn: predict four failures

1. Pond: leave h^1.5 out of the library.
2. Pond: level noise of 5 cm instead of 5 mm.
3. River: floods up to twice as large as any in training.
4. A real river's recessions (CAMELS, Lecture 20): find the law.

Write what happens to each.

::: notes
Two minutes alone. Then four reveals.
:::

## A missing term fits, then misleads

![The design storm again, now with SINDy fitted on a library without h^1.5: it peaks at 2.19 m, below the truth's 2.31 m.](cee_l24_learning_dynamics/design_missing.png)

::: notes
First reveal. Without h^1.5 in the library, SINDy builds the weir from what it has: +8.37e-05 h^0.5, -5.11e-04 h, -4.04e-04 h^2. Over the recorded levels (up to 1.44 m) the combination fits the rate almost as well as SINDy (R² = 0.934 against 0.940), though its held-out level error is 0.015 m against SINDy's 0.0004; at the design storm's levels it peaks at 2.19 m against 2.31: low, the unsafe direction. The library is an assumption: put in every term the physics allows (orifice, weir), and check the model beyond the recorded range before trusting it there.
:::

## 5 cm of level noise adds false terms

| Level noise | Terms kept | Design peak, m |
|:------------|-----------:|---------------:|
| 0.005 m | 2 | 2.31 |
| 0.020 m | 2 | 2.31 |
| 0.050 m | 5 | 2.39 |

::: notes
Second reveal. The derivative comes from differences of noisy levels, so noise in the level becomes larger noise in dh/dt. At 0.02 m SINDy still keeps the two true terms; at 0.05 m it keeps 5, and its design peak moves to 2.39 m. Remedies: better sensors, smoothing before differencing, or fitting the integrated equation instead of the derivative.
:::

## Past the training range, errors grow

![A flood of median size among those up to twice as large as any in training (peaks 600 to 1200 m³/s; training peaks up to 600): the solver's outflow, the FNO's, and the unit hydrograph's.](cee_l24_learning_dynamics/big.png)

::: notes
Third reveal. On floods of 600 to 1200 m³/s, beyond every training flood, the FNO's error rises from at most 0.28 to 13.5–16.0 m³/s; its median peak error is up to 1.5% across seeds. The unit hydrograph is worse still (81 m³/s). Nothing in the network says it is extrapolating: flag members outside the training range and send them to the solver.
:::

## Real rivers: many laws fit alike

![The home basin's dry-weather recession days, 1989–2004: the daily fall in flow against the flow, log axes, a wide cloud; the power law, SINDy's five-term law, and its two-term law, close at the highest flows and apart below them.](cee_l24_learning_dynamics/recession.png)

::: notes
Fourth reveal. North Fork Vermilion River (Lecture 20's home basin), 1,699 dry falling days to fit, 604 to test. On the test days: the one-term power law −dQ/dt = a Q^b (b = 1.73, the form of recession analysis; Brutsaert & Nieber, 1977) errs 0.0915 mm/day²; SINDy with five terms 0.0932, three 0.0928, two 0.0931; a linear reservoir 0.1524. The curved laws tie, and their coefficients disagree, even in sign (Q^1.5: -0.106 in the five-term law, +0.016 in the three-term law). The library's columns are nearly collinear (correlation up to 0.97). The data cannot pick the law; even the exponent depends on how it is fitted: 1.73 on the falls themselves, 1.26 on their logarithms.
:::

## The operator does not conserve water

| 500 test floods, after 72 h | Lowest | Highest |
|:--------------------------------------------|-------:|--------:|
| Solver: volume out ÷ in | 0.947 | 0.998 |
| FNO, 3 seeds: volume out ÷ in | 0.946 | 1.070 |
| FNO ÷ solver, flood by flood | 0.982 | 1.085 |
| FNO − solver, mean flow, m³/s | −0.30 | 0.31 |

::: notes
After 72 hours the solver has passed 94.7% to 99.8% of each flood's volume. The shortfall is mostly the method's, not water still on its way: routed 200 hours more, a flood gains at most 1.1% of its inflow (the first 50 test floods), and the lowest share, 94.7%, does not move (Muskingum–Cunge is known to lose volume). The FNO, on the same floods, passes up to 107.0%: more water out than came in. In absolute terms that is small: the flood with the most (an inflow peak of 105 m³/s) gains 68,011 m³ over the 72 hours, a mean of 0.26 m³/s, the size of the FNO's in-range error (0.25–0.28 m³/s). Flood by flood, the FNO's volume is 0.982 to 1.085 times the solver's, a mean flow from −0.30 to 0.31 m³/s off. Nothing in its architecture conserves volume; a water-balance check catches it, as in Lecture 23, and a percentage on a small flood makes a small offset look large.
:::

## Vote: what routes tomorrow's forecasts?

- **A.** The solver, every member.
- **B.** The FNO, every member.
- **C.** The unit hydrograph.
- **D.** The FNO, and the solver for members beyond the training floods.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: vote, discuss two minutes, revote, then explain. The evidence. A: the solver took 0.65 s for 500 floods, so 1,000 members take seconds: fast enough every hour. B: 0.28 m³/s in range, but up to 16 on bigger floods and 1.10 on half-hourly ones, and its water balance is off the solver's by up to 0.31 m³/s. C: 17 m³/s even in range. D is the general answer when the solver is slow; with this solver, A is the defensible one: use the model you can check when it is cheap enough.
:::

## When to learn the dynamics

| You have | Reach for |
|:---------|:----------|
| The equation and a fast solver | the solver |
| The equation, unknown parameters | least squares on the solver (Lecture 23) |
| Candidate terms and clean data | SINDy, then check it beyond the data |
| A slow solver and many runs | a neural operator, checked against the solver |

::: notes
The first row is today's river: the solver is cheap. The second is Lecture 23. The third is the pond: candidate terms from the physics, clean data, and a check beyond the data. The fourth is where neural operators are proposed: solvers that take long per run, many runs, inputs inside the training range; score the operator against the solver on new cases.
:::

## Exit question

The city's consultant fits a pond's outlet law with SINDy from one year of records and uses it
to size a spillway for a storm twice as large. What would you check before approving?

**In the notebook** (`cee_l24_learning_dynamics.ipynb`): SINDy and an FNO from scratch, the
checks, and the four failures.

::: notes
Any two of: the library holds every outlet law the physics allows; the fitted model is checked on storms it did not see and beyond the recorded levels (the missing-term model peaked at 2.19 m against 2.31); the level noise and the derivative's quality; a bootstrap over storms for the coefficients; the model's water balance.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- SINDy: a library, least squares, a threshold.
- The library must hold the true term.
- Noise in the derivative adds false terms.
- An operator maps a function to a function.
- Check learned models where you will use them.

**Next:** agents that plan, call tools, and check.
:::
::: {.column}
![Thumbnails of today's figures.](cee_l24_learning_dynamics/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Brunton, Proctor & Kutz (2016). *PNAS* 113(15).
- Tibshirani (1996). *J. R. Stat. Soc. B* 58(1).
- Brutsaert & Nieber (1977). *Water Resour. Res.* 13(3).
- Newman et al. (2015). *Hydrol. Earth Syst. Sci.* 19.
:::
::: {.column}
- Li et al. (2021). Fourier neural operator. *ICLR.*
- Cunge (1969). *J. Hydraul. Res.* 7(2).
- Ponce & Yevjevich (1978). *J. Hydraul. Div.* 104(HY12).
- Tang, Knight & Samuels (1999). *J. Hydraul. Eng.* 125(6).
:::
::::::
