---
title: Neural Networks and Backpropagation
subtitle: "Lecture 14 · Can a network stand in for the simulator?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 6–13

On your own, no notes, two minutes:

1. Lecture 6's logistic regression is $\sigma(\mathbf{w}^\top\mathbf{z})$. Drawn as a network,
   how many layers and units? Can its boundary bend?
2. Past the box, Lecture 12's GP band on leakage widened: sd 0.77 m³/h at 60 m, 5.67 at
   90 m. Why?
3. Adam needs the gradient of every weight (Lecture 13). By finite differences, how many
   forward passes for 4,482 weights?

**Today:** all 4,482 gradients from one backward pass.

::: notes
Give the answers before moving on.
1. One unit and no hidden layer: its boundary is a plane in z. A hidden layer lets it bend,
   which is today's first idea.
2. Far from every run, the kernel's correlation with the runs fades, so the posterior variance
   climbs back toward the prior's. Today's networks have no such mechanism; the break-it test
   shows what they do instead.
3. Central differences take two forward passes per weight: 8,964. Backpropagation takes one
   forward and one backward pass.
:::

## Two valves, any setting, instantly

:::::: {.columns}
::: {.column}
- Lecture 12: 15 runs found the settings.
- Now a control room asks about **any** pair, instantly.
- **Decide:** which surrogate, how many runs, where to trust it.
:::
::: {.column}
![Map of L-Town's pipes by area: Area A, solid, fills most of the town; Area B, dashed, a small pocket; Area C, dotted. PRV-1 and PRV-2 feed Area A; junction n50, marked, is the customer with the lowest pressure.](cee_l14_neural_networks/ltown.png)
:::
::::::

::: notes
The setting is Lecture 12's: L-Town (BattLeDIM's EPANET model, run with WNTR; Klise et al.,
2017), the peak day, background leakage calibrated to BattLeDIM's 2018 mean leak flow, and the
limit of 35 psi (24.6 m) at every customer of Areas A and B. A control room wants leakage and
the lowest customer pressure for any pair of settings in the box (25–60 m), instantly. Runs made
in a batch, from 16 to 1,024, train a surrogate. On a calibrated city model a run can take
minutes to hours; here it takes under half a second (0.43 s), so a 1-m map of the box (1,296 runs, which
no surrogate trains on) checks every answer.
:::

## The valves switch regimes

![Contour map of the lowest customer pressure over the two valve settings, 25 to 60 m each, from 1,296 simulator runs, with contours every 4 m and the 24.6 m limit drawn thick. A dashed diagonal line marks where PRV-2 shuts, about 3.6 m below PRV-1's setting, and a dash-dot line in the upper left marks where PRV-1 shuts, about 22 m above it. The contours bend sharply at both lines.](cee_l14_neural_networks/regimes.png)

::: notes
A valve passes no water when the other valve's water reaches its outlet with more head than its
own setting: PRV-2 is closed all day below the line PRV-2 = PRV-1 - 3.6 m (538 of the 1,296
settings), PRV-1 above PRV-2 = PRV-1 + 22 m (105). Lines fitted to where the simulator reports
each valve closed all day; slopes 0.94 and 1.00. The lowest customer is n50 at 1,031 settings;
at the other 265 it is n206 in Area B, which PRV-3 holds at 33.19 m: the surface's ceiling.
:::

## Your turn: where does it bend?

![Six runs of the simulator with PRV-1 at 36 m: the lowest pressure is 14.89 m at PRV-2 settings of 25 and 30 m, then 16.12, 20.11, 23.69, and 25.73 m at 35, 40, 45, and 50 m.](cee_l14_neural_networks/ramp0.png)

::: notes
Three minutes, alone or in pairs. (1) Sketch a curve through the six runs. Where does it bend,
and why there? (2) Write it as a + b max(0, PRV-2 - c): choose a, b, and c. (3) What would one
more ramp fix? The next two slides reveal the answer.
:::

## One ramp: PRV-2 opens at 33 m

![The same six runs with one ramp drawn: flat until 33 m, where PRV-2 opens, then rising about 0.8 m per metre; it overshoots at 45 and 50 m.](cee_l14_neural_networks/ramp1.png)

::: notes
Flat at 14.89 m until about 33 m: below that PRV-2 passes no water (the regimes map's dashed
line), so its setting changes nothing. Then about 0.8 m per metre: from 35 to 40 m the pressure
rises 3.99 m. By hand, with a = 14.89, b = 0.8, and c = 33, the ramp gives 14.89 + 0.8 x 17 =
28.49 m at 50 m, against 25.73: it overshoots.
:::

## Two ramps fit the slice

![The same runs with the simulator's curve every metre, the one ramp dotted, and two ramps summed: a second kink at 43 m lowers the slope; the sum follows the simulator closely.](cee_l14_neural_networks/ramp2.png)

::: notes
A second ramp with a negative slope from about 43 m bends the line back. On the whole slice
(every metre from 25 to 60 m), each pair of kinks gets a least-squares fit, and the pair whose fit
misses least at its worst point is kept: kinks at 33 and 43 m, a = 14.76, b = 0.81, and a second
slope of -0.46; it misses by at most 0.64 m, and the least-squares straight line by 1.90 m. Two ramps are a hidden layer of two units: today's network has 64, then 64 more.
:::

## Ramps forward, gradients backward

![Diagram of the course pipeline with the network highlighted. The two valve settings go to an encoder that standardizes them, giving z; the network, 64 ramps then 64 ramps then 2 outputs, gives y-hat; the loss is the mean squared difference from the simulator's outputs. Above, 1,024 runs feed both the encoder and the simulator, EPANET over one day, which gives y. A dashed arrow from the loss back to the network is labeled backpropagation: all 4,482 gradients, one pass. Below, one run, PRV-1 at 36 m and PRV-2 at 47 m: z is minus 0.64, 0.45; 36 of 64 ramps are on in the first layer and 32 of 64 in the second; the network predicts 23.66 cubic metres per hour of leakage and a lowest pressure of 24.69 m; the simulator gives 23.66 and 24.72.](cee_l14_neural_networks/mlp_backprop.png)

::: notes
The encoder standardizes the two settings with the training runs' mean and standard deviation;
the outputs are standardized the same way and converted back. The network: 2 inputs, 64 ReLU
units, 64 more, 2 outputs: 2 x 64 + 64 + 64 x 64 + 64 + 64 x 2 + 2 = 4,482 weights. Training:
Adam (Lecture 13) on batches of 128 runs, reshuffled every pass, 20,000 steps, with a step that
decays from 0.003 to 0 (a cosine schedule). The inset is one of five networks trained on 1,024
runs; the five average 23.67 and 24.71 at this setting.
:::

## A ramp passes or blocks

![Two panels. Left, a ramp unit, h equals max of 0 and PRV-2 minus 33, flat at zero until 33 m and rising with slope 1 after. Right, its derivative: 0 below 33 m, labeled off, blocks the gradient, and 1 above, labeled on, passes it.](cee_l14_neural_networks/relu.png)

::: notes
A ReLU unit computes h = max(0, w^T z + b). Its derivative is 1 where it is on and 0 where it is
off, so in the backward pass it either passes the gradient unchanged or blocks it. The guarantee
behind stacking them: networks of such units approximate any continuous function on a box
(Cybenko, 1989, for sigmoids; Leshno et al., 1993, for any activation that is not a
polynomial). It promises that a network exists, not that training finds it from a few runs, and
nothing outside the runs.
:::

## Ramps fold the settings plane

![Three 3D surfaces over the valve-setting box, PRV-1 and PRV-2 from 25 to 60 m. Left, one first-layer unit of a trained network: a flat half-plane at zero and a tilted half-plane, joined along a straight diagonal crease, drawn in orange. Middle, the five networks' surface of the lowest pressure; right, the simulator's. The two look alike: a deep trough where both settings are low, folded along sharp creases where the valves switch.](cee_l14_neural_networks/unit_maps.png)

::: notes
With two settings, each first-layer unit is a sheet folded once: flat at zero on one side of a
straight crease, tilted on the other. The unit shown is the one whose crease lies nearest PRV-2's
shut-off line (offset −3.91 m at the box's centre). The second layer adds 64 such folded sheets
and folds the sum again; the result (middle) has the simulator's creases (right) where the
valves switch. Ask: where would a smooth fit, a GP or a polynomial, round these creases off?
:::

## Forward: values flow right

![Network diagram of the attempt's two-ramp network at PRV-2 = 47 m: the upper ramp, kink 33 m, gives 14; the lower ramp, kink 43 m, gives 4; the output is a + 0.8 x 14 + (-0.4) x 4 = 14.89 + 11.2 - 1.6 = 24.49 m, against the simulator's 24.72 m; the loss is 0.0529.](cee_l14_neural_networks/backprop_1.png)

::: notes
The attempt's hand-made network: a = 14.89, b = 0.8, a second slope d = -0.4, kinks at 33 and
43 m. The simulator's lowest pressure at (36, 47) m is 24.72 m to the centimetre. Ask before
the next slide: if the prediction should rise, which numbers should move, and which way?
:::

## The loss sends back −0.46

![The same diagram with the loss's gradient highlighted: d L by d y-hat equals 2 times 24.49 minus 24.72, which is minus 0.46.](cee_l14_neural_networks/backprop_2.png)

::: notes
The derivative of (y-hat - y)^2 is 2(y-hat - y). Negative: raising y-hat lowers the loss. Every
gradient that follows is this number times local derivatives along the path back.
:::

## Each weight: its input × −0.46

![The same diagram with the output weights' gradients highlighted: d L by d b is minus 0.46 times 14, minus 6.44; d L by d d is minus 0.46 times 4, minus 1.84; d L by d a is minus 0.46.](cee_l14_neural_networks/backprop_3.png)

::: notes
A weight's gradient is the gradient arriving at its output times the input it multiplies. A step
of gradient descent raises a, b, and d, and b most, because its ramp is the larger input. The
same -0.46 appears in every gradient: computed once, reused. That reuse is what makes
backpropagation cheap.
:::

## Through the ramps, the kinks move

![The same diagram with the kinks' gradients highlighted: both ramps are on, so each kink gets minus minus 0.46 times its weight: the upper kink plus 0.368, so it moves left; the lower kink minus 0.184, so it moves right.](cee_l14_neural_networks/backprop_4.png)

::: notes
The ramp max(0, PRV-2 - kink) has derivative -1 with respect to its kink when it is on, so the
kink's gradient is -(-0.46) x its outgoing weight. Gradient descent moves the upper kink left
(+0.368) and the lower kink right (-0.184): both raise the prediction at 47 m, toward 24.72. Had a
ramp been off, its kink would get no gradient at all: the ReLU blocks it.
:::

## Same rule, every layer

![Diagram of the network as arrays for a batch of 128 runs: Z, 128 by 2; A1 and H1, 128 by 64; A2 and H2, 128 by 64; Y-hat, 128 by 2; and the loss. Forward arrows above each step, dashed backward arrows below. Under each array its backward rule: G equals 2 times Y-hat minus Y over n m; the gradient of W3 is H2 transposed times G and that of H2 is G times W3 transposed; through a ramp layer the gradient is masked where the unit was off; and so on down to the gradient of W1, Z transposed times the gradient of A1. Each bias gets the column sums of the gradient below it.](cee_l14_neural_networks/layers.png)

::: notes
n rows in the batch, m = 2 outputs. Each operation needs only its own local rule and the values
stored on the way forward. A product's backward step is two products of the same size (one for
the weights, one for the input), so the backward pass costs about twice the forward. Reverse
mode is cheap because there is one output, the loss, and many inputs, the weights; forward mode
would need a pass per weight (Griewank & Walther, 2008). Rumelhart, Hinton & Williams (1986)
made it the standard way to train networks.
:::

## Your turn: predict the check

```python
def matmul(a, b):  # g @ b.T flows to a; a.T @ g flows to b
    return Var(a.value @ b.value, (a, b),
               lambda g: (g @ b.value.T, a.value.T @ g))

def backward(out):
    out.grad = np.ones_like(out.value)
    for v in reversed(topo_order(out)):  # users before inputs
        if v.back:
            for p, g in zip(v.parents, v.back(v.grad)):
                p.grad = p.grad + g
```

A `Var` holds a value, its gradient, its parents, and `back`: how to send a gradient to
each parent.

**Predict:** against PyTorch's autograd, how many digits agree? Against finite differences?

::: notes
The other operations follow the same pattern: add passes g to both inputs, summing over the rows
for a bias; relu passes g where its input was positive; the mean squared error sends
2(y-hat - y)/size. topo_order lists every Var after the Vars it was computed from; walking it in
reverse visits each Var after every Var that used it, so its gradient is complete before it
passes it on. Take a show of hands.
:::

## Agrees with PyTorch

| Check (16 runs, 4,482 weights, float64) | Largest difference |
|:----------------------------------------|-------------------:|
| Gradient vs PyTorch's autograd | 6 × 10⁻¹⁷ |
| Gradient vs central differences (8,964 passes) | 2 × 10⁻¹⁰ |
| Weights after 1,000 Adam steps vs PyTorch's | 2 × 10⁻¹⁴ |

::: notes
PyTorch computes the same products in the same order, so the gradients agree to the last bit or
two. Finite differences use a step of 10^-6 on each weight in turn; their error, against a
largest gradient of 0.23, is the difference quotient's truncation and round-off, not ours. Both
Adam runs start from the same weights and take the same steps.
:::

## Training folds the sheet into shape

![Four 3D views of one network's surface of the lowest pressure over the two valve settings, at Adam steps 0, 10, 50, and 20,000, each drawn over the simulator's surface as a grey mesh. At step 0 the network's surface is a nearly flat sheet (RMS error over the map 7.33 m); at step 10 it tilts (3.20 m); at step 50 it bends toward the mesh (0.94 m); at step 20,000 it lies on the mesh with sharp creases (0.02 m).](cee_l14_neural_networks/sheet.png)

::: notes
Network 1 on the 1,024 runs, drawn at four steps of its training; the grey mesh is the
simulator. The default start has small weights, so the first prediction is a nearly flat sheet
near the mean pressure (7.33 m RMS over the map). Ten steps tilt it (3.20 m); by step 50 the
trough has formed (0.94 m). The remaining 19,950 steps sharpen the creases, from 0.94 to
0.02 m. Ask: why does the tilt come first? Gradient descent fits the broad, smooth part of
the error first, here the overall slope, and the sharp creases last (spectral bias; Rahaman et
al., 2019).
:::

## Longer training still helps

![Log plot of the RMS error of the lowest pressure against the Adam step, for one network on 1,024 runs, on its training runs and on 256 validation runs from a separate design, for two schedules. With 20,000 steps both errors keep falling and end at 0.012 and 0.014 m. Stopped at 4,000 steps, with the same decay compressed, they end at 0.041 and 0.044 m.](cee_l14_neural_networks/training.png)

::: notes
Each schedule decays its step to zero by its last step. With 4,000 steps the training error is
0.041 m against 0.044 m on the validation runs, 256 runs of a fourth Sobol design; with 20,000
steps, 0.012 against 0.014 m. Trained to each length, the validation error keeps falling: 0.057 m
at 2,500 steps, 0.038, 0.022, 0.014 at 20,000, and 0.011 at 40,000; each doubling still removes
about a quarter or more. A deterministic simulator has no noise to overfit, so 20,000 steps is a
compute budget, not the point where the curve flattens; on 1,024 runs, more steps could only lower
the networks' errors in the comparison that follows (on 16 runs the training loss is already at
round-off). The recipe (PyTorch's default start, Adam with a peak step of 0.003) was not tuned;
on the validation runs a peak step of 0.01 does better (0.011 m against 0.014) and 0.001 worse
(0.021), so the networks' errors that follow are, if anything, high. The 1-m map is never used to
choose anything: it judges once.
Judging the short run would have made the network look worse than it is.
:::

## Zoom in: a bowl. Zoom out: valleys

![Two 3D surfaces of the squared error on the 1,024 training runs. Left, around network 1 along two random directions in weight space: a smooth bowl with network 1 at its bottom. Right, over the plane through the weights of networks 1, 2, and 3, trained from three seeds: three valleys, one per network, with ridges of 25 to 40 square metres on the straight lines between them; an orange path runs straight from network 1 over the ridge to network 2.](cee_l14_neural_networks/landscape.png)

::: notes
Each network has 4,482 weights, so its loss is a surface in 4,483 dimensions; we can draw only
2D slices of it. Left: around network 1 along two random directions, each scaled unit by unit
to the size of that unit's weights (Li et al., 2018). Close up, the loss is a smooth bowl, which
is what gradient descent sees: 0.00015 m² at the bottom, 6.4 m² (median) at the slice's edge.
Right: the plane through three networks trained on the same runs from three seeds. Three
valleys, all near zero (0.00015, 0.00006, 0.00015 m²); the straight lines between them cross
ridges of 39.8 m² (networks 1 and 2), 25.4 (1 and 3), and 30.1 (2 and 3).
The orange path is the straight line from network 1 to network 2; the next slide plots it.
:::

## Two seeds, two valleys

![Log plot of the RMS error on the training runs along the straight line between the weights of two trained networks: low at both ends, 0.012 and 0.008 m, and rising to 6.31 m in between, a ridge between two valleys.](cee_l14_neural_networks/barrier.png)

::: notes
Two networks trained on the same runs from different seeds both fit them (0.012 and 0.008 m),
yet the average of their weights misses by 6.31 m: the straight line between them crosses a
ridge. That does not make them separate basins. Part of it is symmetry: swapping two hidden
units changes the weights, not the function, and lining up the units first removes much of such
a barrier (Entezari et al., 2022; Ainsworth et al., 2023); curved paths of low loss often join
minima like these (Garipov et al., 2018; Draxler et al., 2018). Which minimum SGD reaches depends
on the seed: five seeds give five different networks.
:::

## Hidden units found the valves

:::::: {.columns}
::: {.column}
- Every network: a kink on PRV-2's shut-off line.
- Four of five: one on PRV-1's.
- Nobody told them. The runs bend there.
:::
::: {.column}
![Strip plot of the first-layer kink lines that run parallel to the shut-off lines, for five networks trained on 1,024 runs, each kink placed by PRV-2's setting minus PRV-1's where it crosses the box's centre. Ten to sixteen kinks per network lie between minus 6 and plus 32 m. Every network has a kink on the dashed line at minus 3.6 m, where PRV-2 shuts, and four have one on the dash-dot line at 22 m, where PRV-1 shuts.](cee_l14_neural_networks/kinks.png)
:::
::::::

::: notes
A first-layer unit's kink is the line where its input crosses zero. Lines with slope 0.8 to 1.25
run parallel to the shut-off lines; each is placed by its offset, PRV-2 minus PRV-1, at the box's
centre (PRV-1 = 42.5 m). PRV-2's line has a kink within 0.59 m in all five networks; PRV-1's,
within 0.23 m in four and 2.22 m in the fifth. At the start, the same seeds put their nearest
kinks 0.41 to 3.04 m from PRV-2's line and 2.63 to 14.36 m from PRV-1's: training moved them
there. The shut-off lines come from the simulator's valve states, which the networks never saw.
:::

## Networks lead on pressure

![Two log-log panels of error over the 1,296-run map against runs used for training, 16 to 1,024, median of three designs with bars spanning their range, for the GP and for five networks averaged, each as the worst error, solid, and the RMS error, dashed. Leakage, left: the two are close at every budget, the networks lower at 16 runs and the GP from 64; at 1,024 runs the RMS errors are both 0.004 cubic metres per hour and the worst errors 0.023 for the GP and 0.034 for the networks. Lowest pressure, right: the networks are lower at every budget; RMS 0.99 against 1.26 m at 16 runs and 0.011 against 0.026 at 1,024; worst 3.6 against 7.8 m at 16 runs and 0.17 against 0.34 at 1,024.](cee_l14_neural_networks/curves.png)

::: notes
Same runs for both, from three Sobol designs; errors over the 1,296-run map, which neither saw;
medians over the designs, with bars for their range. On pressure the networks beat the GP in all
12 pairs of budget and design, not only in the medians. The GP is Lecture 12's (Matern 5/2, one lengthscale per valve, chosen
by the evidence with five restarts), one per output; no fit ended at one of Lecture 12's
bounds. The networks: five seeds, averaged. Lowest
pressure, RMS error: 0.99 against 1.26 m at 16 runs, 0.19 against 0.25 at 64, 0.028 against
0.081 at 256, and 0.011 against 0.026 at 1,024. Worst error at 1,024 runs: 0.17 m, half the
GP's 0.34. Where the lead comes from: both models' errors are larger within 2 m of a valve's
shut-off line (192 of the 1,296 settings) than elsewhere (medians of three designs). At 16 to 256 runs the networks' lead is about the same there as
elsewhere; at 1,024 runs it is larger there: the GP's RMS error is 3.5 times the networks' near
the lines (0.053 against 0.015 m) and 1.9 times away from them (0.019 against 0.010), in all
three designs. Leakage, which is smooth, is close at every budget: the networks lower in all
three designs at 16 runs, the GP in two of three at 64 and 256 and in all three at 1,024. The
bracket: a plane fitted to 256 runs misses the pressures by 2.44 m RMS; the floor is zero,
because the simulator is deterministic.
:::

## Predict before we test

1. **Bad starts.** All weights 0; all equal; zero biases; settings left in metres. Rank them.
2. **Past the box.** At PRV-2 = 90 m, how far apart are the five networks, and from the truth?
3. **Deciding.** How often is a surrogate's best pick really short of 24.6 m?

::: notes
One minute; students commit in writing. Today's start reaches 0.04 m RMS on 256 runs (the median
of five seeds); inside the
box the five networks agree within 0.08 m along this slice (0.15 m over the whole map). The next slides reveal (1), then (2), the decisive
test, then (3).
:::

## Where the kinks start

![Three panels of the 64 first-layer kink lines at the start of training, over a box of valve settings from 25 to 60 m: with zero biases all 64 lines pass through the box's centre; with uniform weights and biases they are scattered, 60 crossing the box; with settings left in metres they radiate from the origin and only 17 cross the box.](cee_l14_neural_networks/kinks_start.png)

::: notes
Random values break the symmetry between units; their size keeps each layer's output from
growing or vanishing (Glorot & Bengio, 2010). PyTorch's default for a linear layer draws weights
and biases uniformly in plus or minus 1/sqrt(fan-in): the kinks start scattered across the box.
He et al. (2015) draw N(0, 2/fan-in) weights with zero biases, which suits deep ReLU stacks; here
every first-layer kink then starts through one point, the centre of the runs. Settings in metres,
near 40, against weights and biases of at most 0.71, put the kinks near the origin, far from the
box.
:::

## A bad start never recovers

| Start (256 runs, 5 seeds) | RMS (m) | Worst (m) |
|:------------------------------|--------:|----------:|
| Uniform, scaled (today's) | 0.04 (0.03–0.05) | 0.31 (0.18–0.33) |
| Outputs left unscaled | 0.04 (0.03–0.10) | 0.31 (0.24–0.63) |
| He's weights, zero biases | 0.09 (0.09–0.12) | 0.70 (0.63–0.79) |
| Settings in metres | 0.15 (0.13–0.18) | 1.23 (0.76–1.73) |
| Every weight 0.1 | 2.49 | 8.88 |
| Every weight 0 | 6.98 | 19.24 |

::: notes
Errors of the lowest pressure over the 1,296-run map: the median of five networks per start
(seeds 0 to 4), with the range in brackets; each trained for 20,000 steps. Zeros and equal
weights give the same network whatever the seed. Zeros: every hidden unit outputs 0, so no gradient reaches the weights; only
the output bias moves, to the training mean (checked: every prediction equals it). Equal
weights: every unit in a layer computes the same thing and receives the same gradient; only
round-off separates them (6 x 10^-16 in the first layer, 0 in the second), so the network acts
as one unit per layer, no better than a plane (2.44 m). Settings in metres (outputs still scaled;
seed 0): 17 of 64 kinks cross the box at the start and 18 after training, though not the same
units, against 60 of 64 when the settings are scaled. Outputs left in m and m3/h (settings
scaled) give the same median, 0.04 m, with a wider range.
:::

## Five networks agree, all wrong

![Two panels along PRV-2's setting from 25 to 90 m, with PRV-1 at 36 m, for five networks trained on 1,024 runs in the 25 to 60 m box, Lecture 12's GP band on the same runs, and the simulator. Inside the box all three coincide. Past 60 m, shaded, the simulator's leakage, left, rises to 35.65 cubic metres per hour at 75 m and stays flat, while the five networks keep rising to 38.5 to 42.2 at 90 m; the GP band widens to cover the truth. Lowest pressure, right: the simulator stops at 33.19 m from 65 m; the networks climb to 45.6 to 51.1 m at 90 m; the GP band widens to cover 33.19.](cee_l14_neural_networks/beyond.png)

::: notes
The lecture's decisive figure. Inside the box the five networks agree within 0.08 m on pressure
and 0.05 m3/h on leakage. Past it, each continues its last ramps. At 90 m they predict 45.56 to
51.06 m of lowest pressure, where the simulator gives 33.19 m: PRV-3 holds Area B there, so no
setting of these two valves can raise it. Leakage: 38.51 to 42.22 m3/h against 35.65; it is
flat from 75 m, where PRV-2 is fully open (Lecture 12). The GP's band widens and covers both:
25.40 plus or minus 2 x 4.24 m, and 35.10 plus or minus 2 x 4.70 m3/h. The five networks saw
the same runs, so they share the same blind spot. The other two designs' networks do the same:
at 90 m they predict 44.01 to 51.43 and 42.55 to 50.69 m.
:::

## Past the runs, the slopes stay up

![Plot of the slope of the predicted lowest pressure along PRV-2, PRV-1 at 36 m, for the five networks and the simulator. Inside the box the networks' slopes follow the simulator's rise, dip, and rise. Past 60 m the simulator's slope drops to 0 once Area B's ceiling binds, near 65 m; the networks' slopes stay between 0.40 and 0.63 m per metre all the way to 90 m.](cee_l14_neural_networks/slopes.png)

::: notes
Why the networks go wrong together. Their kinks do not stop at the box: each network's last kink
along this slice falls between 82.5 and 87.5 m. But no run showed them Area B's ceiling, so
nothing bends their slopes down: at 90 m they are 0.40 to 0.63 m per metre, against the
simulator's 0. Ensembles often help inside the data (Lakshminarayanan et al., 2017); here the
five averaged reach a worst pressure error of 0.17 m at 1,024 runs (median of three designs),
within the 0.07 to 0.25 m of the 15 single networks (median 0.15): no clear gain. Outside the data, an
ensemble's spread measures disagreement, not error. Check the box before you
ask; outside it, refuse, or run the simulator.
:::

## A pick on the line needs a margin

![Zoom on the settings near the optimum, PRV-1 from 35 to 37.2 m and PRV-2 from 45.8 to 48 m: grey leakage contours; the simulator's 24.6 m line, solid; the networks' 24.6 m line, dashed; the line with the margin, dash-dot. A note reads: networks, 4 of 12 picks truly short; with margins, 0 of 12; the margin here, 0.02 m, from 256 held-out runs. The networks' pick without a margin, a square, sits on their own line; the pick with the margin, a star, sits at 36.0 and 46.8 m, the best setting on the 0.2 m grid.](cee_l14_neural_networks/margin.png)

::: notes
Picks: each of the two surrogates, three designs, four budgets, on a 0.2-m grid of 30,976
candidate settings; one simulator run checks each pick, solved 100 times more tightly than the
model file asks (EPANET accuracy 10^-4 instead of 0.01; pressures move by at most 0.2 mm), so a
shortfall is the surrogate's, not the solver's. A pick on the boundary is short whenever the
surrogate overpredicts the pressure there, by however little: 7 of 24 were (4 of the networks'
12, 3 of the GP's 12), by up to 0.09 m; three of them by only 1.7 mm. The margin is the 95%
split-conformal quantile (Lecture 11) of the networks' overprediction on 256 runs they never saw
(another design's); on the map it covers 96.2% of settings (one-sided, nominal 95%), a guarantee
over the box, not at the boundary. The GP's 2-sd band covers 91.9% of the map both ways, and
95.1% for the one-sided bound its own pick uses (nominal 97.7%). Here it is 0.02 m, and the pick is (36.0, 46.8) m: 23.61 m3/h at 24.63 m, the best on the
grid. Every design and budget with its own margin, from 0.01 m at 1,024 runs to 2.15 m at 16:
none of the 12 picks is short; the price is leakage, up to 1.35 m3/h above the best at 16 runs
and 0.01 at 1,024. Lecture 12's lesson still holds: this margin covers the surrogate's error
against the simulator only; nothing here measures the simulator's error against the town, which
for a city model is likely larger than 0.02 m, so that margin must come from gauges.
:::

## When a network is the wrong tool

| Situation | Better choice | Seen today |
|:----------|:--------------|:-----------|
| Few runs, smooth response | Lecture 12's GP | Lecture 12: 15 runs and a band |
| Settings past the runs | the simulator | five networks, all wrong |
| An error bar is needed | a held-out margin | spread is not error |
| Many runs, switches, many inputs | a network | pressure: half the error |

::: notes
Lecture 12's GP took 15 runs and gave an honest band; for a search with few runs, it is still
the tool. The network earned its place here on the pressure, the response with switches, once
there were hundreds of runs, and it scales to thousands of inputs, as Lecture 15's images need.
:::

## Fast answers invite far questions

:::::: {.columns}
::: {.column}
- At the chosen settings, 7 customers sit within 3 m of the limit; today, none.
- A surrogate's error lands on them first.
:::
::: {.column}
![Map of L-Town with the 7 customers whose lowest pressure at the chosen settings, 36.0 and 46.8 m, lies within 3 m of the 24.6 m limit marked as triangles; they cluster around n50 on the high ground in the north.](cee_l14_neural_networks/near_limit.png)
:::
::::::

::: notes
At today's settings no customer is within 3 m of the limit (today's margin is 3.58 m, Lecture
12); at the chosen settings, 7 are, around n50. A surrogate answers in milliseconds, so
operators will ask it more, and further out. The cost of
computing here is small: minutes of simulator runs and of training on one workstation; it is
not small for the large models of Lectures 17 to 19 and 25 to 26. The social cost is the one
to name: a fast, confident answer outside its runs looks the same as one inside, and its error
lands on specific streets. Log every query, flag those outside the runs, and retrain when the
city model is recalibrated. Ask: who reviews the tool's answers, and who is told when the city
model changes?
:::

## Vote: what goes in the control room?

The utility wants instant answers for any pair of settings.

- **A.** Lecture 12's GP on 15 runs: it found the optimum.
- **B.** Five networks on 1,024 runs: half the GP's worst pressure error.
- **C.** B, with a margin from held-out runs and a refusal outside 25–60 m.
- **D.** The GP on 1,024 runs: its band covers past the box.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
C. A: a search's runs pin the optimum, not the box; even 16 runs spread over the box leave the
GP a worst pressure error of 7.8 m over the map (median of three designs). B: without a margin,
picks fall short (4 of the networks' 12 picks), and past the box it answers with pressures no
setting can reach. C: with margins, none of 12 picks fell short. D is defensible: its one-sided
band covers 95.1% of the map, about as much as C's margin (96.2%); its weakness is the worst
pressure error, 0.34 m against the networks' 0.17. Then ask what would change the answer: a gauge at n50 reading 1.5 m below the
model (the margin must come from the model's error against the town), or a request for PRV-2 at
70 m during a main break (refuse; run the simulator).
:::

## Inside every LLM: MLPs and autograd

![Diagram of one transformer layer: tokens go through attention, with 4 d squared weights, then an MLP from d to 4 d to d, highlighted, with 8 d squared weights, then on to the next layer. A bar below splits the layer's weights: attention one third, MLP two thirds. A dashed arrow loops back: the same backward pass, by PyTorch's autograd.](cee_l14_neural_networks/transformer.png)

::: notes
Per layer, attention has four d x d projections, 4d^2 weights; the MLP, d to 4d to d, has 8d^2:
two-thirds (Vaswani et al., 2017; Lecture 17). The activation is usually GELU rather than ReLU;
the backward pass is the one on today's slides, run by PyTorch's autograd over billions of
weights (Paszke et al., 2019). Neural surrogates replace simulators too: GraphCast, a graph
network (Lecture 21), forecasts 10 days in under a minute (Lam et al., 2023); neural operators
(Lecture 24) learn whole solution fields. In water systems, Broad, Dandy & Maier (2005) replaced
EPANET with networks inside a genetic algorithm; Razavi, Tolson & Burn (2012) review surrogates
in water resources.
:::

## Exit question

A colleague trains five networks on 1,024 runs of a flood model with rainfall up to 100 mm. For
a 150 mm storm the five agree within 1 cm, and the colleague calls the prediction certain. What
do you say, and what do you check first?

**In the notebook** (`cee_l14_neural_networks.ipynb`): the autograd from scratch, its check
against PyTorch and finite differences, the network against the GP, and the break-it tests.

::: notes
Answer: agreement is not accuracy. 150 mm is outside the runs, where every ramp keeps its last
slope and every member shares the same blind spot, as at 90 m today. Check the training range
first; outside it, run the model, and add runs there if such storms matter. Collect the answers;
they seed the next warm-up.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- A unit is a ramp; a network sums and folds ramps.
- Backprop: one backward pass, every gradient.
- Start random, small, and scaled.
- Past the runs, networks go wrong together.

**Next:** convolutional networks, weights shared across an image.
:::
::: {.column}
![Four thumbnails of the lecture's key figures: two ramps fitting the slice; where the kinks start under three starts; the networks' kinks on the valves' shut-off lines; five networks agreeing and all wrong past the box.](cee_l14_neural_networks/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Ainsworth et al. (2023). *ICLR*.
- Broad et al. (2005). *JWRPM* 131(5).
- Cybenko (1989). *MCSS* 2(4).
- Draxler et al. (2018). *ICML*.
- Entezari et al. (2022). *ICLR*.
:::
::: {.column}
- Garipov et al. (2018). *NeurIPS*.
- Glorot & Bengio (2010). *AISTATS*.
- Griewank & Walther (2008). SIAM.
- He et al. (2015). *ICCV*.
- Klise et al. (2017). *Environ. Model. Softw.* 95.
:::
::::::

## References, continued

:::::: {.columns}
::: {.column}
- Lakshminarayanan et al. (2017). *NeurIPS*.
- Lam et al. (2023). *Science* 382.
- Leshno et al. (1993). *Neural Netw.* 6(6).
- Li et al. (2018). *NeurIPS*.
- Paszke et al. (2019). *NeurIPS*.
:::
::: {.column}
- Rahaman et al. (2019). *ICML*.
- Razavi et al. (2012). *Water Resour. Res.* 48(7).
- Rumelhart et al. (1986). *Nature* 323.
- Vaswani et al. (2017). *NeurIPS*.
- Vrachimis et al. (2022). *JWRPM* 148(12).
:::
::::::
