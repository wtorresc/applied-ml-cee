---
title: Graph Neural Networks
subtitle: "Lecture 21 · Can 33 sensors tell the pressure at 782 junctions?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 12–20

On your own, no notes, two minutes:

1. Lecture 12: what pressure must every L-Town customer keep, in psi and in metres of water?
2. Lecture 16: after two 3 × 3 convolutions, which input pixels can change one output pixel?
3. Lecture 17: in attention, which tokens does each token look at?

**Today:** the pressure at every junction from 33 sensors, by passing messages along the pipes.

::: notes
Give the answers before moving on.
1. 35 psi, 24.6 m of water (Ten States Standards, Lecture 12). Today's flag: a junction
   estimated below it.
2. A 5 × 5 patch: each layer widens the receptive field by one pixel on each side. Today the
   same holds on a graph: each layer reaches one more pipe away.
3. All of them. A graph says which nodes may talk: only those joined by a pipe.
:::

## 33 sensors, 782 junctions

| L-Town (BattLeDIM) | Value |
|:-------------------|------:|
| Junctions | 782 |
| Pressure sensors, read every 5 minutes | 33 |
| Links: pipes, 3 pressure-reducing valves, 1 pump | 909 |
| Junctions per pressure zone (valves and the pump separate them) | 657, 92, 31, 1, 1 |
| Pipes from a junction to its nearest sensor: median, most | 4, 11 |

::: notes
The decision: flag, every 5 minutes, the junctions whose pressure is below 35 psi (24.6 m)
though no sensor reads them. A missed flag is a customer without service, or a backflow risk;
a false flag sends a crew. L-Town is the BattLeDIM benchmark's town (Vrachimis et al., 2022;
Lectures 12 and 14): its EPANET model, and two years of its SCADA readings. We simulate days
with the model to learn from, and test on simulated days and on 2019's measurements.
:::

## Your turn: estimate the pressure at n213

| At 21:00 on a test day | Links away | Elevation, m | Pressure, m |
|:---------------------------------------|:----------:|-------------:|------------:|
| Junction n213 | 0 | 3.24 | ? |
| Sensor n215 | 3 | 2.01 | 39.08 |
| Sensor n229 | 8 | 21.58 | 59.95 |

::: notes
Estimate the pressure at n213. Which sensor do you trust, and why? Two
minutes, alone. The junction was drawn at random (seed 21) among junctions exactly 3
links from their nearest sensor; the hour is the first test day's lowest-head moment. Hint
withheld: head = pressure + elevation, and water flows from high head to low head.
:::

## Information travels one pipe per layer

![L-Town's pipes in grey, the 33 sensors as orange stars, the 2 reservoirs and the tank as navy squares. Nothing else is marked yet.](cee_l21_graph_networks/picture_1.png)

::: notes
The one picture, in three frames. A graph: junctions are nodes, pipes are edges. Each layer
of a graph network lets every node read its neighbours, so after k layers a node has heard
from everything within k links (pipes, valves, the pump). Here: what a reading at a sensor can reach.
:::

## Information travels one pipe per layer

![The same map; the junctions within 2 links of a sensor are marked in blue: 212 of 782.](cee_l21_graph_networks/picture_2.png)

::: notes
After 2 layers, 212 of 782 junctions have heard from a sensor; the rest have heard only the
town-wide mean (each layer's average over every node: "A layer: send, sum, update"), whatever
the weights. Ask: how many layers until every junction has? (The most links to a sensor: 11.)
:::

## Information travels one pipe per layer

![The same map after 12 layers: every junction, 782 of 782, is within reach of a sensor.](cee_l21_graph_networks/picture_3.png)

::: notes
After 11 layers every junction can hear from a sensor; we use 12. This is Lecture 16's receptive
field on a graph: a convolution widens it by a pixel per layer, a graph layer by a pipe.
:::

## Grids, sequences, and now graphs

| Data | Neighbours | Lecture |
|:-----|:-----------|:--------|
| Pixels on a grid | the 3 × 3 around each pixel, same weights everywhere | 15, 16 |
| Tokens in a sequence | every token (attention) | 17 |
| A fixed list of readings | none: one weight per input | 14 (MLP) |
| Nodes and edges | the nodes a link joins, same weights for every node and link | today |

::: notes
The graph is the structure the network is told about. A convolution assumes a grid; attention
assumes nothing; a graph network assumes that what matters at a node reaches it along the
edges. For a water network the edges are the 909 links: the pipes, the 3 valves, and the pump.
:::

## A layer: send, sum, update

$h_i \leftarrow h_i + U\big(h_i,\ \sum_{j \to i} M(h_j, h_i, e_{ji}),\ \bar h\big)$

![The attempt's junction n213 and everything within 3 links of it (10 nodes), coloured by pressure; an arrow from its one neighbour into it marks the message it receives in one layer; the sensor n215 is a star. Labels: n213, head 41.09 m, elevation 3.24 m; n215, head 41.09 m, elevation 2.01 m.](cee_l21_graph_networks/neighborhood.png)

::: notes
Assumption: a node's state depends on its neighbours through the same rules at every node and
every link. M (the message) and U (the update) are small MLPs shared by all; e_ji are the
pipe's features (its Hazen-Williams resistance from length, diameter, and roughness; for a
valve, the head it holds). In the code each layer first normalizes the states (LayerNorm), and M
and U read the normalized ones. The sum makes the layer indifferent to the order of neighbours;
h-bar, the mean over all nodes, is the same input at every node: a town-wide summary that no
neighbourhood holds (a graph-level attribute, Battaglia et al., 2018). A graph convolution
(Kipf & Welling, 2017) is the special case with a linear message and a degree-scaled sum. Worked on the attempt:
n213 sits at the end of a branch; its one neighbour passes it what the branch knows; the
sensor's head reaches it after 3 layers. The two heads agree to the centimetre
(41.09 and 41.09 m), so the two pressures differ by
the elevations.
:::

## Your turn: predict the check

```python
agg = zeros(N, d).index_add_(0, dst, m)  # each node's incoming sum
dense = B @ m                     # B[i, k] = 1 if edge k ends at i
out = model(relabel(x))           # shuffle the node numbers
# Two purely local layers; add 5 m to one sensor's reading:
moved = nodes whose estimate changes
```

How close are `agg` and `dense`? Does relabelling change the answers? Which nodes move?

::: notes
Two minutes, written. The incidence matrix B has 785 rows (nodes) and 1,818 columns (each link in
both directions).
:::

## Same sums; the reach holds

| Check | Value |
|:------|------:|
| index_add_ against the incidence product, largest difference | 9e-16 |
| Nodes relabelled at random: largest change in any estimate, m | 8e-06 |
| 2 local layers, one sensor nudged: nodes that move | 3 |
| … nodes within 2 links of that sensor | 3 |
| … largest change beyond 2 links, m | 0 |
| The trained 2-layer network (with the town-wide mean): nodes that move | 783 |

::: notes
The sum is a sparse matrix product, to float64 round-off. Relabelling permutes the answers and
changes nothing else, to float32 round-off: the network has no notion of a node's number, only
of its neighbourhood (permutation equivariance). With purely local layers, nothing beyond 2
links can move, whatever the weights (an untrained network shows it). The trained network adds
the town-wide mean, so a nudge reaches 783 of the 785 nodes, but only through that one average
(largest change beyond 2 links: 2e-03 m).
:::

## Fixed network: MLP and GNN are close

| Test days (30, simulated), all 33 sensors | Head error, m | Low-pressure flags found | Flags right |
|:------------------------------------------|--------------:|-------------------------:|------------:|
| Each junction's training mean | 8.85 | 0% | none raised |
| Nearest observed node in the zone | 0.25 | 100% | 99% |
| MLP on the 33 readings | 0.18 | 99% | 97% |
| GNN, 12 layers (budget-limited) | 0.19 | 99% | 99% |

::: notes
First result, on 30 simulated test days like the training days (demand ×0.7–1.3, every
sensor read). The MLP's error, 0.18 m, and the GNN's,
0.19 m, are close: one training seed each, a gap of a few
centimetres; the nearest observed node in the zone, with no learning, gets
0.25 m. All three find about 99% of the low-pressure flags
(1.0% of unmeasured junction-hours are below 35 psi); each
junction's training mean raises 0 flags. Zone by zone (next slide) the GNN is ahead in Area A (0.125 m against 0.170) and Area B (0.021 against 0.223); the MLP leads in Area C (0.198 against 0.445). Weights: MLP
701,201, GNN 358,849. Same protocol
for every model: Adam rates from 3e-4 and 1e-3, widened a step at a time until the best on the
validation days lies inside the range tried (Lecture 13); one seed each. Why "budget-limited":
the GNN's kept rate, 1e-3, stopped early (best epoch 178 of 193), but the rates on either side, 3e-4 and
3e-3, were still improving at the 200-epoch budget (best epochs 197 and 199). A longer budget
can only lower each run's best validation error, so the search might keep another rate: the
12-layer GNN's errors in this lecture may be lower with a longer budget (only its validation
error is guaranteed not to rise). The MLP's kept rate and both of its neighbours stopped early.
:::

## The GNN errs most behind the pump

![Map: each unmeasured junction coloured by the 12-layer GNN's head error over the test days, light (0) to dark, the darkest 1% beyond the colour scale; sensors as stars. Area C, behind the pump and tank, is darkest: the GNN's error there is 0.45 m, against 0.13 m in Area A.](cee_l21_graph_networks/error_map.png)

::: notes
By pressure zone, test days. Area A, the main zone (29 sensors,
657 junctions): GNN 0.125 m, MLP
0.170 m, the nearest sensor 0.269 m.
Area C, behind the pump and tank (3 sensors, 92
junctions): GNN 0.445 m, MLP 0.198 m,
the nearest sensor 0.033 m. Area B (1
sensor, 31 junctions): GNN 0.021, MLP
0.223, nearest 0.006 m. Area C's
heads move together: in one hour they spread 0.027 m across its junctions,
while over the test days each moves 0.52 m (standard deviations). So a
neighbour's reading is nearly right. Both networks err more in Area C than the nearest sensor does (GNN 0.445, MLP 0.198, nearest 0.033 m; the GNN's
errors are budget-limited, see the fixed-network slide). Along the pipes the GNN's error grows with distance
from a sensor, 0.14 m at 1 link to
0.31 m at 10, while the MLP's stays at 0.17–0.19 m (next slide). One town-wide number hides a zone
served worse: report errors zone by zone.
:::

## Your turn: predict four failures

1. Use 2 message-passing layers instead of 12.
2. Close one pipe (p182, 200 mm) on the test days.
3. Raise the town's demand to ×1.5–1.8, beyond training's ×0.7–1.3.
4. Estimate 2019's measured sensors, one hidden at a time, from the other 32.

Write what happens to each, for the GNN and for the MLP.

::: notes
Two minutes alone. Then four reveals.
:::

## Two layers cannot see three links

![Head error by links to the nearest sensor (junction counts below each), for five methods. GNN with 2 layers: 0.48 m at 2 links, then 1.69 m at 3. GNN with 12 layers: 0.14 m at 1 link to 0.31 m at 10. MLP: near 0.18 m at every distance.](cee_l21_graph_networks/hops.png)

::: notes
First reveal. The jump sits exactly at the reach: beyond 2 links, a 2-layer network hears no
sensor, only the town-wide mean, so its error leaps from 0.48 to
1.69 m (overall 1.43 m). Five layers
move the step out: 0.27 m at 3 links, 0.46 m
at 5, then flat (overall 0.38 m). The two-layer network stopped early (best at epoch 41), so its step is not a short training budget; the five-layer network's kept run was still improving at epoch 197 of 200, so its errors may be lower with a longer budget, as may the 12-layer network's (its neighbouring rates: best epochs 197 and 199 of 200). Beyond its reach a two-layer network has only the town-wide mean of the readings: at 3 links it errs 1.69 m, against 8.97 m for each junction's training mean. The MLP sees all 33 readings at once: distance means nothing to it. Rule: depth at least the
most links to a sensor (11).
:::

## A closed pipe: told, the GNN errs least

| Pipe p182 closed (200 mm), 30 test days | Head error, m | Flags right |
|:-----------------------------------------------|--------------:|------------:|
| GNN, told: pipe cut from graph (budget-limited) | 0.45 | 98% |
| GNN, not told (budget-limited) | 0.68 | 98% |
| MLP on the 33 readings | 2.34 | 79% |
| Nearest observed node in the zone | 0.83 | 98% |

::: notes
Second reveal. Closing the pipe moves the heads by 9.06 m (RMSE over the
junctions and test days), and 3.8% of junction-hours drop below
35 psi. No model saw a closed pipe in training. The GNN can be told: its rules are local, and the
closure is a change of neighbours. Told (the pipe removed from its graph), its error rises from 0.19 to 0.45 m, the least of the four. The MLP has no way to be told; its error grows to 2.34 m,
and 79% of its flags are right (crews sent for nothing).
Not told, the GNN gets 0.68 m. Both GNN errors are budget-limited (fixed-network slide). The pipe: of Area A's ten busiest
pipes whose loss keeps the network connected, the one whose closing moved the heads most on 3
validation days.
:::

## Higher demand: every method errs more

| Demand ×1.5–1.8, 30 test days | Head error, m | Flags found | Flags right |
|:------------------------------------------|--------------:|------------:|------------:|
| GNN, 12 layers (budget-limited) | 0.31 | 97% | 99% |
| MLP on the 33 readings | 0.31 | 98% | 95% |
| Nearest observed node in the zone | 0.47 | 99% | 98% |

::: notes
Third reveal. Thirty test days with the town's demand at ×1.5–1.8, beyond anything in
training. The GNN's error grows from 0.19 to 0.31 m, the MLP's from 0.18 to 0.31, the nearest sensor's from 0.25 to 0.47: even the method that learns nothing errs more, so these days are harder, not only unfamiliar (the head lost along a pipe doubles: 0.029 to 0.058 m on average). The GNN and the MLP stay close (one seed each; the GNN's 0.31 m is budget-limited). One zone breaks the pattern: in Area C, 174 junction-hours fall below 35 psi; the GNN flags 0% of them, the MLP 6%, the nearest sensor 100%.
:::

## 2019: sensors beat the model alone

![Each of the 33 sensors hidden in turn, its head estimated from the other 32 over 2019's first four weeks (hourly), error per sensor on a log scale with the median: model alone 0.29 m, nearest in zone 0.10 m (one sensor alone in its zone has no estimate), MLP 0.11 m, GNN 0.12 m.](cee_l21_graph_networks/real.png)

::: notes
Fourth reveal, the decisive one: real readings. Each of the 33 sensors hidden in turn, its
head estimated hourly from the other 32 over 2019's first four weeks
(672 hours). Medians per sensor: the hydraulic model alone
0.29 m; nearest in zone 0.10 m;
MLP 0.11 m; GNN 0.12 m. Over all
sensors and hours (RMSE): 0.31, 0.18
(without the one sensor alone in its zone), 0.13, and
0.24 m. Every data-driven method cuts the model's median error by more than half. 2019 also holds BattLeDIM's leaks, which no method here knows about. The
networks learned only from simulated days and held up on real ones (simulated hold-outs:
MLP 0.17, GNN 0.25 m). By median the nearest sensor in the zone (0.103 m, on the 32 sensors it can estimate) is as good as both networks, within centimetres and with one seed per network (MLP 0.112, GNN 0.124, on all 33 sensors); by RMSE the MLP is lowest (0.130, nearest 0.179, GNN 0.235; the GNN's errors are budget-limited).
:::

## Who runs the low-pressure flags?

| Do not use it alone when | Because, today |
|:-------------------------|:---------------|
| A pipe was closed and the MLP not retrained | 79% of its flags right (GNN told: 98%) |
| Demand beyond training, in Area C | GNN found 0% of 174 low junction-hours; the nearest sensor 100% |
| A zone is fed by one tank (Area C) | GNN: 0.45 m; the nearest sensor: 0.03 m |
| A zone has one sensor | nearest in zone: no estimate for that sensor |

::: notes
The flags send crews: a false flag costs a crew's hours; a missed one leaves a customer below
35 psi, where backflow can draw contaminated water into the mains. Impact (social, equity): a
town-wide error hides the zones served worse (the GNN's 0.19 m
overall, budget-limited, is 0.45 m in Area C; the MLP's is
0.22 m in Area B). Zone by zone the picture changes: at higher demand the GNN found none of Area C's low junction-hours (0% of 174) and the nearest sensor found 100%. Report and audit errors zone by zone. The utility's operations log (which valves
and pipes are closed) is part of the model: the GNN is only as good as the graph it is told.
:::

## Vote: what estimates the pressures?

- **A.** The MLP: lowest overall error (RMSE), simulated and real.
- **B.** The GNN, told of every valve and pipe closure.
- **C.** The nearest sensor in the zone: no training at all.
- **D.** The hydraulic model alone, as today.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: vote, discuss two minutes, revote, then explain. The evidence: A has the lowest
overall error (RMSE) on a fixed network, close to the GNN's with one seed each, and on real
readings; with one pipe closed its error grows to
2.34 m. B needs every closure recorded and fed to the graph; then
it errs least of the four (0.45 m, budget-limited). C needs
no training and matches the networks by median on real data, but gives nothing for a sensor
alone in its zone and 0.83 m with the pipe closed. D, today's
practice, errs most (0.29 m median). A defensible answer: B in the main zone
where the closure log is reliable, C in the small zones (Area C:
0.033 m) and as a running cross-check; A only if retrained after
every change to the network. The vote is about the utility's records as much as about the model.
:::

## Where graph networks live

| Today | What the graph is | Lecture |
|:------|:------------------|:--------|
| GraphCast weather forecasts | points on the globe, joined to their neighbours | today |
| Google Maps arrival times | road segments, joined where they meet | today |
| Traffic forecasting (DCRNN) | freeway sensors, joined by road distance | today, 20 |
| Molecules and materials | atoms, joined by bonds | today |
| Language models | tokens, every one joined to every other | 17, 19 |

::: notes
GraphCast (Lam et al., 2023) passes messages on a mesh over the globe and forecasts ten days
of weather. Google Maps' arrival times come from a graph network over road segments
(Derrow-Pinion et al., 2021). DCRNN (Li et al., 2018) forecasts freeway speeds with diffusion
on the sensor graph, Lecture 20's sequences on a graph. Message passing began in chemistry,
predicting molecules' properties (Gilmer et al., 2017). A transformer is a graph network on the
complete graph: attention is a learned message from every token to every other.
:::

## Exit question

A consultant offers a graph network trained on another utility's network. What would you
check before using it on yours, and what must be true of the two networks for it to transfer?

**In the notebook** (`cee_l21_graph_networks.ipynb`): message passing from scratch, the three
checks, the GNN against the MLP, and the four failures.

::: notes
What to check: that the other network's graph is built the same way (node and pipe features in
the same units, valves and pumps as edges), that 12 layers reach its farthest junction from a
sensor (depth at least its largest number of links), and its error on simulated days of your
own network before any real use. The MLP cannot transfer at all: its first layer is wired to
this town's 33 sensors. The GNN's rules are local, so they can run on any graph; whether they
are right there is an empirical question, answered on your network's simulations.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- A graph layer: send, sum, update; shared rules.
- k layers reach k links; plan depth by distance.
- Beyond its reach a layer sees only the town-wide mean.
- On a fixed network an MLP is as good; graphs pay when the network changes.
- Check errors zone by zone; a baseline can win in one.

**Next:** point clouds and 3-D scans.
:::
::: {.column}
![Thumbnails of today's figures.](cee_l21_graph_networks/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Gilmer et al. (2017). *ICML*.
- Kipf & Welling (2017). *ICLR*.
- Battaglia et al. (2018). arXiv.
- Vrachimis et al. (2022). *J. Water Resour. Plann. Manage.* 148(12).
:::
::: {.column}
- Klise et al. (2017). *Environ. Model. Softw.* 95.
- Rossman (2000). EPANET 2. EPA.
- Lam et al. (2023). GraphCast. *Science*.
- Derrow-Pinion et al. (2021). *CIKM*.
- Li et al. (2018). DCRNN. *ICLR*.
:::
::::::
