---
title: Physics-Informed Networks
subtitle: "Lecture 23 · When can the road be paved?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 5–22

On your own, no notes, two minutes:

1. Lecture 5: least squares fits a line's slope. Could it fit a soil's coefficient instead?
2. Lecture 14: backpropagation differentiates the loss with respect to the weights. What else
   can it differentiate?
3. Lecture 20: a 1-D filter with weights [1, −2, 1] / h² slides along a signal. What does it
   compute?

**Today:** a network trained to obey a physical law, not just to fit data, and when it pays.

::: notes
Give the answers before moving on.
1. Yes: least squares fits a model's parameters through its predictions, not only a line's
   slope; today it fits a clay's coefficient of consolidation through Terzaghi's solution.
2. Its inputs. The same automatic differentiation gives ∂u/∂t and ∂²u/∂z² of a network u(z, t),
   which is what the PDE's residual needs.
3. The second derivative, by finite differences: the classical way to solve today's equation,
   and today's baseline.
:::

## Four metres of clay under a new road

| The site (assumed values; readings simulated, so the truth is known) | Value |
|:----------------------------------------|-----------------:|
| Soft clay, drained at the top, rock below | 4 m |
| Coefficient of consolidation, c_v (to be learned) | 4 m²/year |
| Excess pore pressure under 2.5 m of fill | 50 kPa |
| Piezometers, read monthly for 6 months | 1, 2, 3 m deep |
| Reading noise (one standard deviation) | 1 kPa |
| Pave when the clay has consolidated | 90% |

::: notes
The decision: an embankment's fill squeezes water out of the clay below it; the clay settles
as the water drains through the top. Pave too early and the road keeps settling and cracks.
The contractor wants a paving date from six months of readings. The physics is Terzaghi's
(1943): ∂u/∂t = c_v ∂²u/∂z², the excess pore pressure u diffusing out through the drained top.
The values are assumed and the readings simulated from Terzaghi's exact solution with
1-kPa noise, so every method can be scored against the truth: 90%
consolidation at 3.39 years.
:::

## Your turn: when is the clay 90% done?

![Excess pore pressure at 1, 2, and 3 m depth, read monthly for six months after the fill was placed, from 50 kPa down to 19, 33, and 42 kPa; the axis runs on to four years, empty.](cee_l23_physics_informed/attempt.png)

::: notes
Two minutes, alone: write a paving date, in months. Then ask how they got it: a ruler along
the curves, a curve through the points, Terzaghi's chart from the geotechnics course. Collect
the spread. Average consolidation (the share of the final settlement) is not any one
piezometer: at the true t90 the 3-m piezometer still reads
7.3 kPa.
:::

## A network that must obey the PDE

![Depth down, years across: Terzaghi's exact excess pore pressure, dark blue (50 kPa) at the start and deep, fading as the water drains through the top; orange stars mark the 18 readings, all in the first six months.](cee_l23_physics_informed/picture_1.png)

::: notes
The one picture, in three frames. Frame 1: the truth. The pressure falls first near the
drained top and last at the base; the readings see only the first six months. Ask: what does
a network fitted to the stars alone know about year three?
:::

## A network that must obey the PDE

![The same plane: 2,000 grey points where the PDE's residual is scored, orange points along t = 0 where u must be 50 kPa, green points along the base where no water may flow, the readings as stars, and the drained top in blue, built into the network.](cee_l23_physics_informed/picture_2.png)

::: notes
A physics-informed neural network (Raissi et al., 2019) is a network u(z, t) trained on a loss
read at these points. Grey: the residual ∂u/∂t − c_v ∂²u/∂z², computed from the network by
automatic differentiation, must vanish. Orange: the initial condition. Green: no flow through
the rock, ∂u/∂z = 0. The drained top needs no points: the network's output is multiplied by
z, so u = 0 there exactly (a hard constraint). Stars: the readings, when we have them. The
residual points need no data: they cost only computing.
:::

## A network that must obey the PDE

![The trained network's error against the exact solution, on a log scale from 0.01 to 10 kPa: over 1 kPa only in the top 1.0 m and the first 0.03 years, the corner where the drained top meets the start; worst 3.9 kPa.](cee_l23_physics_informed/picture_3.png)

::: notes
The network after training on the PDE alone (c_v known, no readings; seed 23): it reproduces the exact
solution to a relative error of 1.0%, except at the corner where the drained top (u = 0) meets the start (u = 50 kPa). There the exact solution jumps, and the water's flow out of the top is infinite at t = 0; a smooth network cannot follow a jump. Its error peaks there at 3.9 kPa and is at most 0.80 kPa after three months. Ask: which part of the field matters for a paving date years later?
:::

## Grids, graphs, sets, and now physics

| Data | What the model is told | How | Lecture |
|:------------|:-----------------------|:-------------------|:-------|
| Pixels on a grid | a pattern means the same anywhere | shared 3 × 3 weights | 15, 16 |
| Nodes and edges | the same rules at every node | messages summed | 21 |
| Points in space | order does not matter | a max over points | 22 |
| A field u(z, t) | a PDE holds everywhere | a penalty in the loss | today |

::: notes
Each structure gets the model that respects it. The first three build the structure into the
architecture: it holds exactly, for any weights. A PINN mostly asks for it through the loss: it
holds only as well as training drives the residual down, and only where the residual is
scored. The drained top is the exception today: built in, exact.
:::

## The loss: residual, boundaries, data

:::::: {.columns}
::: {.column}
$$r = \frac{\partial u}{\partial t} - c_v\, \frac{\partial^2 u}{\partial z^2}$$

$$L = L_r + L_{\text{base}} + L_{\text{start}} + L_{\text{data}}$$

each the mean of a squared mismatch over its points.
:::
::: {.column}
| Term | Points | Asks |
|:-----|:-------|:-----|
| Residual | 2,000 random | the PDE holds |
| Base | 200 | no flow into rock |
| Start | 200 | u = 50 kPa |
| Readings | 18 | match the piezometers |
:::
::::::

::: notes
The network: 4 tanh layers of 32, 3,297 weights,
inputs scaled to [−1, 1]. Its derivatives with respect to z and t come from the same automatic
differentiation as backpropagation (Lecture 14), applied to the inputs. tanh, not ReLU: a ReLU
network's second derivative is zero almost everywhere. Training: Adam (3,000
steps), then L-BFGS (up to 1,000 iterations), in 64-bit floats on the CPU.
The forward problem knows c_v; the inverse problem makes it one more trainable parameter and
adds the readings. Equal weights on the four terms; choosing them is a known difficulty (Wang
et al., 2021).
:::

## Your turn: predict the check

```python
u_zz = grad(grad(net(z, t), z), z)    # autograd, twice
fd   = (net(z+h,t) - 2*net(z,t) + net(z-h,t)) / h**2
r    = residual(terzaghi_series, z, t) # same code
U1   = drained(terzaghi_series, 3 mo, t90)  # flux, top
cv   = series_fit(readings_without_noise)
```

How far is `u_zz` from `fd`? `r` from 0? `U1` from the change in U? `cv` from 4?

::: notes
Two minutes, written. h = 10⁻⁴ in 64-bit floats. The drained route: the water leaving through
the top, the integral of k ∂u/∂z there, must equal the drop in the layer's average pressure.
:::

## The code passes every check

| Check | Value |
|:------|------:|
| Autograd ∂²u/∂z² against finite differences of the network | 1e-08 |
| Residual of Terzaghi's exact series | 2e-15 |
| Residual of the river's exact solution (Péclet 1,000) | 5e-15 |
| Water drained against the drop in pressure, exact solution | 6e-08 |
| c_v from noise-free readings, error in m²/year | 9e-16 |
| Finite differences at 0.2, 0.8, 3.39 years, 20 → 40 → 80 cells: max error, kPa | 0.034 → 0.009 → 0.0021 |

::: notes
The residual code scores the exact solutions at round-off, so it computes the right PDE. The
second difference matches autograd to round-off at h = 10⁻⁴ (Lecture 20's filter, now a check). The drained
route agrees with the pressures on the exact solution, so a disagreement later is the
network's, not the bookkeeping's. The finite-difference solver (Crank–Nicolson) quarters its
error when the cells halve: second order (observed orders 2.00 and 2.00). It is scored here at three times;
the next slide scores it at every time from 0.01 years, where the corner makes its error larger.
:::

## The classical solver wins, by far

| Forward problem, c_v known | Max error, kPa | After 3 months | Seconds |
|:---------------------------|---------------:|---------------:|--------:|
| Finite differences, 80 cells, 8,000 steps | 0.038 | 0.0019 | 0.01 |
| PINN, 5 seeds | 2.3–25.5 | 0.35–1.53 | 180 (median) |

::: notes
Both are scored at the same 81 depths and 500 times. Both err most in the first days near the drained top, where the exact solution jumps; the best PINN errs 59 times more than finite differences there and 184 times more after three months, and takes minutes, not milliseconds: 180 s against 0.01 s on this run's shared CPU (4 threads per fit; timings vary from run to run). Across 5 seeds the PINN's worst error ranges 2.3 to 25.5 kPa: the seed matters, and the training loss does not tell which to keep. The lowest loss, 8.8e-05 (seed 26; the others 0.013 to 0.031), belongs to the run with the largest error, 25.5 kPa: choose among runs by a physics check (conservation, a classical solution), not by the loss. For diffusion in 1-D with known coefficients, use the classical solver. Grossmann et al. (2024) found the same in a systematic comparison: physics-informed
networks did not beat the finite element method on time or accuracy in their tests; McGreivy
and Hakim (2024) found that most papers claiming otherwise compared against weak baselines.
:::

## Learning c_v: the series fit wins

![Left: average consolidation against years on the site's readings (the six months read are shaded): the exact curve; Terzaghi's series with c_v fitted, on top of it; the PINN with c_v learned, close to it; the network fitted to the readings alone (and the start), rising to 1.17 at two years, past 90% at 1.16 years, then falling back. Right: years to 90% on 10 noise draws: the series fit from 3.23 to 3.52, the PINN from 3.14 to 3.84; the truth, 3.39, as a line.](cee_l23_physics_informed/inverse.png)

::: notes
On the site's readings (left) both physics-based methods land near the truth. The network fitted to the readings alone gets the start as the PINN has it (the same start term: 50 kPa at 200 depths at t = 0) and has nothing to hold it beyond its readings: its average consolidation first passes 90% at 1.16 years, against the truth's 3.39, rises to 1.17 at two years (above 1 means negative pore pressures), and falls back to 0.88 at three: the clay un-consolidates. Over 10 noise draws (right), the series fit's worst t90 is off by 4.8%: that spread is the readings' noise. The PINN's worst is off by 13.2%, and it is further off than the fit on 10 of 10 draws: the noise plus its own approximation error. Its c_v ranges 3.56 to 4.36 m²/year, the fit's 3.85 to 4.20. Each PINN took about 144 s; each fit at most 0.01 s. The readings alone, over the same draws: years to 90% from 0.58 to 4.15, and never within five years on 3; average consolidation above 1 on 6; pore pressures from -204 kPa up to 120 kPa, above the 50-kPa load. The physics is what makes extrapolation possible; the series fit and the PINN both carry it.
:::

## Your turn: predict four failures

1. Train the PINN without the initial-condition term.
2. Score the residual only where the readings are: the first six months.
3. Read the settlement from the water drained, not from the pressures.
4. A spill in a river: the same method at Péclet numbers 1 to 10,000.

Write what happens to each.

::: notes
Two minutes alone. Then four reveals.
:::

## No start condition: settled at once

| Average consolidation | 3 months | 1 year | t90 |
|:----------------------|---------:|-------:|----:|
| Exact | 0.282 | 0.562 | 0.900 |
| PINN with the start term (seed 23) | 0.286 | 0.566 | 0.898 |
| PINN without it, seed 23 | 1.000 | 1.000 | 1.000 |
| PINN without it, seed 24 | 1.000 | 1.000 | 1.000 |
| PINN without it, seed 25 | 1.000 | 1.000 | 1.000 |

::: notes
First reveal. Without the start term, u = 0 everywhere satisfies everything left in the loss: the PDE, the no-flow base, the drained top. Training finds it: the network's pore pressure is at most 0.041 kPa anywhere, so the clay is 'consolidated' from the first day. A loss that a trivial answer satisfies will be satisfied by it. Every condition the physics needs must be in the loss or built into the network; then check the answer against something known, here 50 kPa at the start.
:::

## No residual points, no physics

![Average consolidation against years on the site's readings: the exact curve; the PINN with residual points over five years, close to it; the PINN with residual points only in the first six months (shaded), which follows the truth there and then departs: 0.36 at five years.](cee_l23_physics_informed/window.png)

::: notes
Second reveal. The residual is scored only where its points are. Beyond six months the network still meets the drained top (built in) and the no-flow base (scored over all five years), but nothing ties how it changes in time to the PDE. With points only in the first six months, over 10 draws: the field never reaches 90% within five years on 8 draws, and its average consolidation at five years ranges from 0.12 to 1.36. The learned c_v is off too: 2.59 to 3.89 m²/year against 4. Residual points cost no data: put them everywhere the network will be asked a question.
:::

## Two readings of one settlement disagree

![For each of 5 seeds, the PINN's consolidation at the true t90 read two ways: from its pore pressures, 0.898 to 0.901 (the truth is 0.90); from the water drained through the top, 0.878 to 1.429.](cee_l23_physics_informed/conservation.png)

::: notes
Third reveal. Water is conserved: what leaves through the drained top must equal the drop in the layer's pore pressures, and on the exact solution the two agree to 6e-08. The PINN's two readings differ by more than 1% on 3 of 5 seeds; on seed 26 the water drained says 1.43, more water out than the layer held. The PDE holds only approximately, worst near the drained top at the start; the water drained is read from the network's slope at the top from t = 0 on, where its errors are largest. A model that conserves only approximately can be right on one output and wrong on another: check every quantity you report, and prefer a method that conserves by construction (finite volumes) when a balance matters.
:::

## Sharp fronts break the PINN

![Left: concentration along the river at τ = 0.1, the front a tenth of the way down, at Péclet 1,000: the exact solution, a steep front, and five PINN seeds that miss it by up to 0.31. Right: each run's largest error against the Péclet number, log axes: the PINN's median is 0.016 at Péclet 1 and 0.48 at 10,000; finite differences stay at or below 0.008; refined PINNs (green) at 1,000 and 10,000.](cee_l23_physics_informed/peclet.png)

::: notes
Fourth reveal. A spill upstream: the advection–dispersion equation ∂c/∂τ + ∂c/∂X = (1/Pe) ∂²c/∂X², with Ogata and Banks's (1961) exact solution. The Péclet number Pe = vL/D compares carrying with spreading: for example, 5 km of river at 0.5 m/s with D = 2.5 m²/s (assumed) gives 1,000. The front's width shrinks like 1/√Pe, and the error sits on it: at Pe 1,000 every seed's largest error is within 0.006 of the front's centre, at the earliest time scored (τ = 0.05), when the front is sharpest. From Pe = 1,000 every seed's largest error exceeds 0.1 (5 of 5 at 1,000, 5 at 10,000). Krishnapriyan et al. (2021) report the same failure for convection and trace it to a hard optimization problem, not to the network's size. The mass balance does not raise this alarm (worst 0.7% at 1,000), but it flags low-Pe runs (worst 6.3% at 10): this check missed this failure. Finite differences: 0.008 at worst, at most 107 s, with 40,000 cells at 10,000.
:::

## Refinement helps; FD still wins

| Max error, five seeds | Péclet 1,000 | Péclet 10,000 |
|:----------------------|-------------:|--------------:|
| PINN | 0.128–0.43 | 0.20–0.77 |
| PINN, refined | 0.008–0.35 | 0.09–0.22 |
| PINN, 1,000 random points added | 0.099–0.42 | 0.17–0.74 |
| Finite differences | 0.005 | 0.002 |

::: notes
After Adam, score the residual at 20,000 fresh random points and add the 1,000 worst to the training points, then finish with L-BFGS: one round of residual-based refinement (Lu et al., 2021, repeat the round until the residual is small enough). It lowered the error on 5 of 5 seeds at Pe 1,000 and 4 of 5 at 10,000, and nearly closed the mass balance (worst 0.10% at 1,000). Finite differences still beat the best refined seed: an error 1.5 times smaller at 1,000 and 54 times smaller at 10,000. Is it the extra points or where they go? Adding 1,000 random points from the same pool instead, refinement still wins on 5 of 5 seeds at 1,000; at 10,000 it wins on 3 (close, seed by seed), with a median of 0.19 against the random points' 0.51 (the random points' worst mass error 0.61% at 1,000).
:::

## Vote: who sets the paving date?

- **A.** A network fitted to the readings alone.
- **B.** Terzaghi's solution with c_v fitted to the readings.
- **C.** The PINN, with c_v learned.
- **D.** B now, refitted as each month's readings arrive.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: vote, discuss two minutes, revote, then explain. The evidence. A: a network fitted to the readings alone gives years to 90% from 0.58 to never (3 of 10 draws), and consolidation above 1 on 6. B: the worst of 10 noise draws is off by 4.8%, in milliseconds. C: further off than B on 10 of 10 draws, minutes per fit, and the conservation and corner checks still to pass. D: B, checked against the readings as they arrive, and the defensible answer: the site's c_v is an assumption until the readings confirm it, and refitting each month is Peck's (1969) observational method: the prediction is checked against the site, not trusted.
:::

## When a physics-informed network pays

| Situation | Reach for |
|:----------|:----------|
| PDE known, parameters known | a classical solver: finite differences, finite elements |
| PDE known, a few unknown parameters | least squares on the solver |
| PDE known, an unknown field, scattered data | a PINN, against a classical inverse method |
| PDE partly unknown | learn the missing terms: Lecture 24 |

::: notes
The first two rows are today's measurements. The third is where PINNs are proposed for inverse problems
(Raissi et al., 2019, recover unknown coefficients and hidden fields from scattered data), and
where the classical alternatives (adjoint and regularized inversion) are the baseline to beat.
The fourth is next lecture: sparse regression on data finds the equation itself. The impact
question: when a network replaces an analysis that a design code sanctions, who signs for the
result? The engineer of record still does, so the engineer needs the checks from today.
:::

## Exit question

A consultant's report predicts an embankment's settlement with a PINN. Which two checks would
you ask for before signing off, and why?

**In the notebook** (`cee_l23_physics_informed.ipynb`): the PINN from scratch, the checks,
the inverse problem, and the four failures.

::: notes
Any two of: the same problem solved by a classical method (finite differences, or the series fitted) and compared; the error where the solution is steep, at boundaries and corners; a conservation check read two ways (settlement from pore pressures and from water drained); several seeds, since one can be badly off (2.3 to 25.5 kPa worst error across seeds here); residual points everywhere the prediction is used. Who is accountable: the engineer who signs, so the method's failure modes are theirs to check.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- A PINN: a network, its PDE residual by autograd, a loss.
- The physics holds only where the residual is scored.
- Build in what you can: the drained top was exact.
- Check conservation two ways.
- Beat the classical solver, or use it.

**Next:** SINDy and neural operators.
:::
::: {.column}
![Thumbnails of today's figures.](cee_l23_physics_informed/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Terzaghi (1943). *Theoretical Soil Mechanics.* Wiley.
- Raissi et al. (2019). *J. Comput. Phys.* 378.
- Krishnapriyan et al. (2021). *NeurIPS.*
- Wang et al. (2021). *SIAM J. Sci. Comput.* 43(5).
- Lu et al. (2021). DeepXDE. *SIAM Rev.* 63(1).
:::
::: {.column}
- Ogata & Banks (1961). USGS Prof. Paper 411-A.
- Peck (1969). *Géotechnique* 19(2).
- Grossmann et al. (2024). *IMA J. Appl. Math.* 89(1).
- McGreivy & Hakim (2024). *Nat. Mach. Intell.* 6(10).
:::
::::::
