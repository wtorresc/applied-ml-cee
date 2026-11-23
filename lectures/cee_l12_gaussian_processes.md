---
title: Gaussian Processes for Optimization
subtitle: "Lecture 12 · Which valve settings leak least?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 2–11

On your own, no notes, two minutes:

1. k-NN predicted a new mix from its five nearest tests (Lecture 2). What did it tell you
   about how sure it was?
2. Ridge regression adds $\lambda\lVert\mathbf{w}\rVert^2$ to the loss (Lecture 5). What
   does $\lambda$ trade?
3. How many calibration scores does a 90% split-conformal interval need (Lecture 11)?

**Today:** fifteen simulator runs to set two valves.

::: notes
Give the answers before moving on; recall helps most with feedback.
1. Nothing: the spread of the five neighbors is a hint, not an interval. Today's predictor
   also weighs past cases by similarity, and it reports how sure it is.
2. Fit against the size of the weights: a little bias for much less variance. Today's mean
   is kernel ridge regression.
3. At least 9: the cutoff is the ceil((n + 1) x 0.9)-th score, which must exist among n. With
   5 runs there is no 90% conformal interval; today's interval comes from a model of the
   function instead, and it is only as good as that model.
:::

## Two valves, one district, 15 runs

- Lower settings leak less; keep 35 psi (24.6 m).
- 15 runs of the simulator.

![Map of L-Town's pipes by area: Area A, 657 junctions, solid lines, fills most of the town; Area B, 31 junctions, dashed, a small pocket in the south; Area C, 92 junctions, dotted, in the northwest. PRV-1 feeds Area A from the east and PRV-2 from the west; junction n50, in the north, has the lowest pressure.](cee_l12_gaussian_processes/ltown_map.png)

::: notes
BattLeDIM (Vrachimis et al., 2022, CC BY 4.0) published L-Town's EPANET model with its SCADA
data, which Lecture 9 used; WNTR runs it from Python (Klise et al., 2017). The budget stands
for a calibrated city model, which can take minutes to hours per run. One run here simulates the peak day of the model's one-week
demand pattern; the lowest pressure is at n50, at 19:50.

The assumption about leakage, to state aloud: background leakage at every Area A junction,
proportional to the pipe length it serves and to its pressure (an EPANET emitter with exponent
N1 = 1), scaled so that today's settings leak 25.52 m3/h, BattLeDIM's mean leak flow in 2018.
35 psi is the normal working pressure the Ten States Standards ask for; their floor of 20 psi
under all flows, fire flow included, is not simulated here. Area C is fed by the tank and stays
at 24.82 m whatever the two valves do. Here a run takes under half a second (0.38 s), so every
answer in this lecture is checked against the simulator itself, starting with a 225-run map of
the box.
Area A, 657 junctions: fed by PRV-1 (40 m) and PRV-2 (50 m). Decide: lower settings leak less;
every customer keeps 35 psi (24.6 m). Simulator: L-Town in EPANET, one peak day. Budget: 15
runs.
:::

## Your turn: where does it cross 24.6 m?

PRV-1 held at 36 m. Four runs of the simulator, peak day:

| PRV-2 setting (m) | 30 | 40 | 50 | 60 |
|:------------------|---:|---:|---:|---:|
| Lowest pressure (m) | 14.89 | 20.11 | 25.73 | 29.33 |

::: notes
Two minutes on your own: at what PRV-2 setting does the lowest pressure cross 24.6 m? How sure
are you? Where would you spend a fifth run?
Answers: a straight line between the 40 m and 50 m runs crosses at 48.0 m; the simulator says
46.77 m. The fifth run belongs where the answer is uncertain and matters: between 45 and 48 m,
not at 35 m, where we already know the pressure is too low. Keep your answers; the next slides
build a predictor that makes this reasoning precise, and then test it.
:::

## A GP weighs the runs by similarity

![Diagram of the course pipeline with the Gaussian process highlighted, inside a loop. Valve settings go to an encoder that scales them to a box, giving z; the Gaussian process gives a mean and plus or minus two standard deviations; a decision step, expected improvement times the probability of feasibility, picks the next run; the simulator, EPANET over one day, returns a result to the runs so far, which condition the process; the evidence, minus the log marginal likelihood, fits the lengthscale. Below, the attempt's four runs at PRV-1 36 m: PRV-2 at 30, 40, 50, 60 m gave lowest pressures 14.89, 20.11, 25.73, 29.33 m; at 47 m the weights on the runs are minus 0.054, 0.330, 0.800, minus 0.078, so the mean is 22.51 plus the weighted deviations, 24.2 m, with a band of 24.0 to 24.4 m and a lengthscale of 48 m; the probability of feasibility is 0.000, infeasible; the simulator says 24.7 m, feasible.](cee_l12_gaussian_processes/gp_pipeline.png)

::: notes
The mean is the average of the runs plus a weighted sum of their deviations, like k-NN, but
the weights come from the kernel and can be negative. The process is sure that 47 m fails, and
the simulator says it passes. Hold that contradiction; the break-it section explains it.
:::

## Nearby settings, similar pressures

**Assumption:** outputs at nearby settings are strongly correlated, and "nearby" means the
same everywhere in the box (a stationary kernel).

- A Gaussian process: any finite set of outputs is jointly Gaussian, with covariance $k$.
- Matérn 5/2, one lengthscale $\ell_d$ per setting:
  $$k(r)=\sigma^2\Big(1+\sqrt{5}\,r+\frac{5}{3}r^2\Big)e^{-\sqrt{5}\,r},\qquad r^2=\sum_d\frac{(z_d-z'_d)^2}{\ell_d^2}$$
- $\ell_d$: how far a setting moves before the output changes appreciably.

::: notes
Matérn 5/2 functions are twice differentiable: smooth, but rougher than the squared exponential,
which suits physical responses with kinks. It is the usual default in Bayesian optimization
(Snoek et al., 2012). Outputs are standardized (mean 0, standard deviation 1) before fitting, as
scikit-learn's normalize_y does; settings are scaled to the box [25, 60] m.
:::

## Condition on the runs you have

Runs $\mathbf{y}$ at settings $Z$; a new setting $\mathbf{z}_*$, with $\mathbf{k}_*=k(Z,\mathbf{z}_*)$:

$$\mu(\mathbf{z}_*)=\mathbf{k}_*^\top(K+\epsilon I)^{-1}\mathbf{y},\qquad\sigma^2(\mathbf{z}_*)=k_{**}-\mathbf{k}_*^\top(K+\epsilon I)^{-1}\mathbf{k}_*$$

- The mean weighs the runs: $\mathbf{w}=(K+\epsilon I)^{-1}\mathbf{k}_*$.
- The variance shrinks near runs and returns to $k_{**}$ far from them.
- Factor once, $K+\epsilon I=LL^\top$: $O(n^3)$, then $O(n^2)$ per prediction.

::: notes
The formulas are the conditional of a joint Gaussian (Rasmussen & Williams, 2006, §2.2). epsilon
= 1e-6 in standardized units is jitter, not noise: the simulator is deterministic, and the
jitter keeps the Cholesky factor stable. The mean is kernel ridge regression with lambda =
epsilon (Lecture 5). On the attempt: weights -0.054, 0.330, 0.800, -0.078 on the deviations
from 22.51 m give 24.17 m.
:::

## Runs pin down the possible curves

![Two panels of the lowest pressure against the PRV-2 setting with PRV-1 at 36 m. Before the runs: six curves drawn from the Gaussian process prior, smooth but wandering anywhere from 5 to 32 m. After four runs at 30, 40, 50, and 60 m: six curves drawn from the posterior all pass through the four orange runs and stay close together between them, inside a pale band of plus or minus two standard deviations that pinches at each run.](cee_l12_gaussian_processes/samples.png)

::: notes
A GP is a distribution over curves. Before any run, the kernel allows any smooth curve with the
right wiggle length (left: six draws). Conditioning keeps only the curves that pass through the
runs (right: six draws from the posterior). Between runs they spread, and the band is that
spread; at a run they agree. The mean of all posterior curves is the formula on the last slide.
:::

## The evidence picks the lengthscale

$-\log p(\mathbf{y})$ = misfit + complexity + const.

Each pulls the other way: the four runs pick $\ell$ = **48 m**.

![Three curves against the lengthscale from 5 to 120 m for the four runs, signal variance held at its fitted value: the misfit term rises steeply for long lengthscales; the complexity term, one half log determinant, falls; their sum, the negative log evidence, is lowest at 48 m.](cee_l12_gaussian_processes/evidence.png)

::: notes
The determinant term is Occam's razor built into the marginal likelihood (Rasmussen & Williams,
2006, §5.4). Choosing hyperparameters this way is type-II maximum likelihood; it treats the
chosen lengthscale as known. With few runs, that can make the band confidently wrong, as the
break-it section shows.
-log p(y) = 1/2 z^T K^-1 z + 1/2 log|K| + (n/2) log 2 pi, with z the standardized runs. The signal
variance is held at its fitted value in the plot, so the two terms trade visibly.
First term: how well the process fits the runs; a long lengthscale cannot bend through them.
Second term: a short lengthscale fits anything, and pays for it in log|K|. Maximize over the
lengthscales and the signal variance (L-BFGS, three starts).
:::

## Expected improvement, if feasible

$$\mathrm{EI}=\sigma\,\big[u\,\Phi(u)+\phi(u)\big]$$
$$u=(f^+-\mu)/\sigma$$

Times $P(\text{feasible})$. Next run: the peak.

![3D surface of expected improvement times the probability of feasibility over the two settings after the five initial runs, which are black dots on the floor: a ridge across PRV-1 settings of about 31 to 45 m that runs the whole length of PRV-2, near zero above 48 m; its top, at PRV-1 39.5 m and PRV-2 25.0 m, is the next run, marked orange.](cee_l12_gaussian_processes/acquisition.png)

::: notes
EI is E[max(f+ - f, 0)] in closed form (Jones, Schonlau & Welch, 1998). Multiplying by the
probability of feasibility handles the pressure constraint (Gardner et al., 2014). Before any
run is feasible, maximize the probability of feasibility alone. Candidates: every pair on a
0.5 m lattice over the box, 5,041 of them. Five space-filling runs (a Latin hypercube) start
each search; ten more follow the acquisition.
Leakage f ~ N(mu, sigma^2) at a candidate; f+ = least feasible leakage so far. Times
P(lowest >= 24.6 m) = Phi((mu_p - 24.6)/sigma_p), from a second GP. High where the mean is low
(exploit) or the doubt is large (explore). Here, after five runs, the best feasible run leaks
25.95 m³/h. At the peak, PRV-1 = 39.5 m, leakage could beat it by 2.55 m³/h on average (EI) but
the pressure holds with probability only 0.48: the product is 1.22. Along PRV-2 at that PRV-1
setting the product only ranges from 1.21 to 1.22, because five runs set PRV-2's lengthscale at
its upper bound (PRV-1's is 6.5 m): the process believes PRV-2 does not matter yet. PRV-2 = 25 m
wins by a hair.
:::

## Your turn: predict the check

```python
def gp_posterior(Z, y, Zs, ell, var, eps=1e-6):
    K = matern52(Z, Z, ell, var) + eps * np.eye(len(Z))
    L = np.linalg.cholesky(K)                   # K = L L^T
    alpha = cho_solve((L, True), y)             # K^-1 y
    ks = matern52(Zs, Z, ell, var)              # (m, n)
    v = solve_triangular(L, ks.T, lower=True)   # (n, m)
    sd = np.sqrt(np.clip(var - (v**2).sum(0), 0, None))
    return ks @ alpha, sd
```

**Predict:** scikit-learn, same kernel and hyperparameters. Largest difference in the mean?
And when both choose the lengthscale themselves?

::: notes
Outputs y here are standardized; the notebook adds the mean and scale back. Take a show of
hands: 1e-3? 1e-8? 1e-12? And for the second question: will the evidence they reach differ?
:::

## Agrees with scikit-learn

**Verified:** means and standard deviations agree within $10^{-11}$ m; the evidence at the
same hyperparameters, within $10^{-11}$.

- Each choosing its own lengthscales: the same optimum (evidence within $10^{-8}$).
- Fifteen runs in two settings: leakage $\ell$ = 37 and 45 m; lowest pressure, 29 and 22 m.

::: notes
scikit-learn's GaussianProcessRegressor with ConstantKernel x Matern(nu = 2.5), alpha = 1e-6,
and normalize_y, restarting its optimizer five times; ours restarts from three fixed points.
A lengthscale is a correlation distance, not a rate: over the 225-run map the lowest pressure
changes by 0.47 m per metre of PRV-1 and 0.31 m per metre of PRV-2, on average, although
PRV-2's lengthscale is the shorter.
:::

## Fifteen runs, 0.05 from the best

Ten starts: median **0.05** above the best; random 1.57; a 4 × 4 grid 0.58.

![Line plot of the least feasible leakage found after each simulator run, 1 to 15. Bayesian optimization's median falls from 31.3 to 23.65 m3/h by run 14, close to the best possible, 23.60, a dotted line; random settings level off at 25.17; a 4 by 4 grid of 16 runs reaches 24.18. A dashed line marks today's settings, 25.52. Shaded bands show the range over 10 starts.](cee_l12_gaussian_processes/regret.png)

::: notes
The best possible, 23.60 m3/h at (35.9, 46.89) m, comes from the simulator itself: for each
PRV-1 setting, bisection finds the
lowest PRV-2 setting that keeps 24.6 m (leakage rises with both settings, so the best feasible
point is on that boundary), then a scan of PRV-1 every 1 m and every 0.1 m around the best.
The best settings save 7.5% of today's leakage, 1.92 m3/h.
One search: 23.65 m³/h at (35.5, 47.5) m. Ten starts: median 0.05 above the best possible; worst
0.16. Bracket: today 25.52 above; best possible 23.60 below. The searches pick from a 0.5 m
lattice, and the lattice's best feasible setting is that same (35.5, 47.5) m, 0.05 above the best
possible: the median search reaches the lattice's floor, and a finer lattice would close the rest.
:::

## Five runs, then fifteen: doubt shrinks

![Two 3D surfaces of the leakage GP's mean over the two valve settings, coloured by its standard deviation, dark where sure and yellow at 2 cubic meters per hour or more, with the runs as orange pins. After 5 runs: a ridged surface, mostly yellow; median standard deviation 1.6. After the 15 runs of Bayesian optimization: a smooth surface, dark almost everywhere, with most pins clustered near the low corner where the best settings are; median standard deviation 0.4.](cee_l12_gaussian_processes/posterior3d.png)

::: notes
After five runs the evidence put PRV-2's lengthscale at its upper bound (10 box widths): with
five points it cannot yet see PRV-2's effect, so the surface varies only with PRV-1, in ridges
(lengthscale 6.5 m). Ten more runs fix that. The search spends its runs where leakage is low and
the pressure limit allows it, so the surface is surest there, near (35.5, 47.5) m. Far corners
stay less certain; the search never needed them. Ask: what would a fixed, sensible lengthscale
prior have done with the first five runs?
:::

## Predict before we test

1. **Four runs.** Is the ±2σ band at 47 m honest? How often does the truth fall inside?
2. **Past the box.** Run PRV-2 up to 90 m. What does the process predict?
3. **A different question.** Spend 15 runs mapping the 24.6 m boundary instead. Better map?
   Better settings?
4. **The leak model.** Leakage exponent 0.5 or 1.5 instead of 1. Do the best settings move?

::: notes
One minute; students commit in writing. Each of the next slides reveals one answer: the
lengthscale (1), extrapolation (2), the question the runs serve (3), and the modeling
assumption behind the prize (4).
:::

## Four runs cannot pin the lengthscale

![Three panels of the lowest pressure against the PRV-2 setting with PRV-1 at 36 m, each with the simulator's curve, the runs, and the process's mean and band; a dotted line marks 24.6 m. Between the runs: with 4 runs and a lengthscale fixed at 2 m, the mean dips back toward the average and the band is wide everywhere: error 3.37 m, truth inside the band 100% of the time. With 4 runs and the fitted lengthscale, 48 m, the mean follows the curve but the band is very narrow: error 0.94 m, truth never inside. With 7 runs the fitted lengthscale is 17 m: error 0.30 m, truth inside 83% of the time.](cee_l12_gaussian_processes/lengthscale.png)

::: notes
This is the lecture's decisive figure. Errors and coverage are counted between the runs, every
metre except the run settings themselves, where the process repeats the simulator. With four
runs the evidence prefers a long lengthscale, and the band, computed as if that lengthscale were
known, is far too narrow: it holds the truth at none of the 32 settings between the runs. The
simulator's curve is flat up to 33 m, a feature four runs cannot reveal. The cure is runs,
not a cleverer lengthscale: with seven runs the band holds the truth at 24 of 29 settings
between them (83%, for a nominal 95%). Before trusting a band, check it:
leave one run out and see whether the process predicts it inside its band, or make one
confirming run. A fully Bayesian treatment that averages over lengthscales widens the band
(Snoek et al., 2012).
:::

## Beyond the runs, the band widens

PRV-2 fully open from 75 m; at 90 m the GP says 32.86, σ 5.67. Trust the band, not the mean.

![Line plot of leakage against the PRV-2 setting from 25 to 90 m with PRV-1 at 36 m. Inside the box, up to 60 m, the process's mean follows the simulator closely. Beyond 60 m, shaded, the simulator rises to 35.65 m3/h at 75 m and stays flat, because the valve is fully open; the process's mean bends down to about 33 and its band widens to plus or minus 11 at 90 m.](cee_l12_gaussian_processes/extrapolation.png)

::: notes
A stationary kernel assumes the function behaves the same everywhere; a valve that changes
from regulating to fully open is a different regime. Here the band is honest and the mean is
2.8 m3/h low at 90 m. Never optimize over settings the runs did not cover.
Leakage flattens at 35.65 m³/h past 75 m. At 90 m the truth is inside plus or minus 2 sigma.
:::

## The question picks the runs

**Straddle** runs where the boundary is uncertain: $1.96\,\sigma_p-\lvert\mu_p-24.6\rvert$
(Bryan et al., 2005).

| Fifteen runs chosen by | Above the best (m³/h) | Map points misclassified |
|:-----------------------|----------------------:|-------------------------:|
| Bayesian optimization | 0.05 | 2.9% (0.9–8.1%) |
| Straddle | 1.25 | 1.6% (0.9–3.2%) |
| Random settings | 1.57 | 3.1% (1.3–6.7%) |

::: notes
Medians over 10 starts, with the range across them in brackets. "Misclassified": the share of
the 225-run map where the process calls the pressure feasible and the simulator does not, or
the reverse (134 of the 225 points are feasible); map points a search ran itself are left out,
since there the process repeats the simulator. Above the best, the ranges are 0.05-0.16 (BO), 0.44-2.47
(straddle), and 0.68-2.75 (random): BO's worst start beats every other start. On the map the medians are close and the ranges overlap; straddle's advantage is
its worst start, 3.2% against 8.1% for BO and 6.7% for random. Active learning is the same loop
with a different acquisition: the question you ask decides where the runs go. A surrogate meant for many future questions, like Lecture
14's, needs runs spread over the box.
:::

## Where each search spent its runs

![Two maps of leakage over the PRV-1 and PRV-2 settings, 25 to 60 m, with thin contour lines of leakage in m3/h, the infeasible region hatched, and the true 24.6 m boundary as a thick line. Left, Bayesian optimization: after five open circles, its ten squares cluster in the valley near the optimum, a star at 35.9 and 46.9 m, below today's settings, a diamond at 40 and 50 m; its estimated boundary, dashed, follows the truth near the optimum and strays where PRV-1 is above 46 m. Right, straddle: its runs follow the boundary, and its dashed boundary matches the true one almost everywhere.](cee_l12_gaussian_processes/bo_map.png)

::: notes
The boundary has a corner: from a PRV-1 setting of 47 m, any PRV-2 setting in the box
keeps 24.6 m; at 46 m, PRV-2 must be at least 43.17 m. Bayesian optimization (left) spends none
of its ten guided runs past 46 m: the least feasible leakage on the map from 47 m on is 26.59
m3/h, against 23.60 at the optimum. Straddle (right) runs along the boundary there too.
:::

## The GP is sure; the town is not

| At BO's best run | Means |
|:--------------------------|:--------------------------------------------|
| 0.02 m to spare, σ = 0.005 m | the simulator is interpolated well |
| P(feasible) = 1.00 | about the simulator, not the town |
| pipes, demands, gauges differ | the limit needs a margin: what does it cost? |

::: notes
At a setting it has run, the process has almost no uncertainty: it is interpolating a
deterministic simulator. Its probability says nothing about how well the simulator matches the
town. That gap is calibration error, measured against gauges (Lecture 9's SCADA pressures).
:::

## The saving is the margin

Each metre of margin: about **0.53 m³/h**. Today's settings sit on the curve.

![Least leakage of the best settings against the pressure kept above 35 psi: 23.60 cubic meters per hour with no margin, 24.14 with 1 m, 24.67 with 2 m, and 25.51 with today's 3.58 m; today's settings, 40 and 50 m, a square at 25.52, sit on the line.](cee_l12_gaussian_processes/margin.png)

::: notes
Each margin's best settings come from the simulator, the same boundary search as before:
(35.9, 46.89), (36.9, 47.92), and (37.9, 48.94) m. With today's 3.58 m, the best settings are
(39.5, 50.55) m: 0.01 m3/h less than today, two bisection brackets' worth (one bracket, 0.02 m
of PRV-2, changes leakage there by 0.005 m3/h), so equal to the search's resolution. The two metres of margin cost 1.07 m3/h together,
about 0.53 each. So the decision is the margin, and the frontier is what Bayesian optimization
buys: once the margin is set, 15 runs find its best settings. Fire flow (20 psi under the
largest fire demand) is a second limit this lecture did not simulate.
Least leakage with 0, 1, 2 m to spare: 23.60, 24.14, 24.67 m³/h. Today's settings keep 3.58 m and
leak 25.52; the best settings with that same margin, 25.51. Today's settings were already on the
frontier. The 7.5% is the margin given up.
:::

## The leak model sets the prize

| Leakage exponent N1 | 0.5 | 1.0 | 1.5 |
|:--------------------|----:|----:|----:|
| Best settings (m) | 35.9, 46.93 | 35.9, 46.89 | 35.9, 46.84 |
| Leakage saved | 3.9% | 7.5% | 11.0% |

::: notes
Each row recalibrates the leakage so that today's settings leak 25.52 m3/h, then finds the
best settings again with the simulator. The best settings move by under 0.1 m (PRV-2 by 0.09
m; PRV-1, scanned every 0.1 m, not at all): each lies on the 24.6 m boundary, which the
exponent barely shifts. The exponent decides how much water they save. Field tests of N1 (step
tests at night) tell the utility which prize to expect.
:::

## When a GP is the wrong tool

| Not when | Because |
|:------------------------------|:------------------------------------|
| there are many settings | beyond about 20, kernels see every point as far away |
| there are many runs | O(n³): thousands need sparse or neural surrogates (Lecture 14) |
| there are regimes and jumps | a stationary kernel smooths a valve opening, a pipe emptying |
| the simulator is cheap | one run here takes under half a second: run the grid |

::: notes
Bayesian optimization suits fewer than about 20 settings (Frazier, 2018). It pays when each run
is expensive and the settings are few: tuning a
calibrated city model, a structural analysis, a lab test. When runs are cheap, search
exhaustively and check. Nonstationary kernels, or one process per regime, handle known regime
changes.
:::

## Less leakage, thinner margins

| Gain | Who carries the risk |
|:--------------------------|:--------------------------------------------|
| less water lost, less energy to treat and pump it | customers at n50, the highest junction (44.7 m) |
| a thinner margin | anyone the model is wrong about |
| state it first | who bears the residual risk, before lowering anyone's pressure |

::: notes
Pressure management is a standard way to cut real losses, and a model makes it look free. The cost of a wrong model falls on specific streets. Ask: who should be told that
the margin at n50 was set from a model, and who checks it after the change?
:::

## Vote: which settings for tomorrow?

The utility will change the two valves tomorrow.

- **A.** BO's best run, (35.5, 47.5) m: saves 7.3%, 0.02 m to spare.
- **B.** The best settings with 1 m to spare, (36.9, 47.92) m: saves 5.4%.
- **C.** Keep today's (40, 50) m: 3.58 m to spare.
- **D.** B, after a week-long run with fire flows and a logged trial at n50.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
D: A trusts the simulator to 2 cm; C keeps a margin nobody chose; B is the right kind of
answer, but its margin must come from the model's error against the town, and the change goes
in steps with the pressure at n50 logged. Then ask what would change the answer: a gauge at n50
that reads 1.5 m below the model, or a planned fire-flow test.
:::

## Bayesian optimization tunes modern AI

| Where | Today's idea, inside | Lecture |
|:----------------|:------------------------------------------|:----------|
| Hyperparameters | Bayesian optimization (Snoek et al., 2012); Vizier (Golovin et al., 2017); BoTorch (Balandat et al., 2020) | — |
| Wide networks | infinitely wide networks are GPs (Neal, 1996; Lee et al., 2018) | 14 |
| Self-driving labs | choose the next experiment the same way | — |

::: notes
Every large training run is an expensive simulator with a few settings that matter; the same
loop of surrogate, acquisition, and run decides them. The GP-network connection is exact in
the limit of infinite width, and is one reason the two can behave alike; on this simulator,
Lecture 14 compares them run for run.
:::

## Exit question

After 15 runs, the GP says settings (35.5, 47.5) m keep 24.6 m with probability 1.00. A
colleague proposes deploying them tonight. What does that probability mean, what does it not,
and what would you do first?

**In the notebook** (`cee_l12_gaussian_processes.ipynb`): the GP from scratch, its check
against scikit-learn, Bayesian optimization and straddle on the simulator, and the break-it
tests.

::: notes
Answer: the probability is the process's belief about the simulator at a setting it has
already run; it says nothing about the model's error against the town, which is larger than
the 0.02 m to spare. First: set a margin from the model's error against the gauges, check a
week and fire flows, then lower in steps with logging. Collect the answers; they seed the next
warm-up.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- A GP: curves that pass through the runs.
- The evidence trades misfit for complexity.
- EI × P(feasible) picks the next run.
- Outside the runs, trust the band only.

**Next:** stochastic gradient descent: learning from millions of records.
:::
::: {.column}
![Four thumbnails of the lecture's key figures: prior and posterior curves on the slice; the leakage GP after 5 and 15 runs coloured by doubt; the least leakage found per run by each search; leakage past the box with the widening band.](cee_l12_gaussian_processes/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Balandat et al. (2020). *NeurIPS*.
- Bryan et al. (2005). *NeurIPS*.
- Frazier (2018). arXiv:1807.02811.
- Gardner et al. (2014). *ICML*.
- Golovin et al. (2017). *KDD*.
- Jones, Schonlau & Welch (1998). *J. Global Optim.* 13.
- Klise et al. (2017). *Environ. Model. Softw.* 95.
:::
::: {.column}
- Lee et al. (2018). *ICLR*.
- Neal (1996). *Bayesian Learning for Neural Networks*. Springer.
- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Snoek, Larochelle & Adams (2012). *NeurIPS*.
- Vrachimis et al. (2022). *J. Water Resour. Plann. Manage.* 148(12).
:::
::::::
