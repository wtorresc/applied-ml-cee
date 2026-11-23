---
title: Trees, Forests, and Their Explanations
subtitle: "Lecture 8 · What makes a pavement wear out fast?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 3–7

On your own, no notes, two minutes:

1. Random folds picked $k=2$ for bridge ratings; the 2024 inventory disagreed. What did the
   random folds reuse?
2. In 2024 the engineer's rule "flag every 5" tied the bridge model. What must any model beat?
3. BIC was lowest at $K=10$ components, the largest tried; the meter needed two regimes.
   What decided $K$?

**Today:** what makes a pavement get rough fast, and can a model trained on the national test
sections tell a state DOT?

::: notes
Give the answers before moving on; recall helps most with feedback.
1. The same bridge's other inspections, its twin, and its neighbors: near-copies of each
   held-out record (Lecture 3). Today: sections at the same site.
2. The simplest credible baseline: then the engineer's rule; today, the mean (Lecture 6).
3. The decision, not the fit (Lecture 7). Today: the decision decides which explanation to
   trust.
:::

## Which pavements wear out fastest?

- Roughness (IRI): how drivers feel a road.
- Target: the **rate** it grows, m/km per year.

![IRI at each roughness visit against age for the eight Kansas SPS-1 sections at one site: section 0105, in orange, climbs from 0.9 to 2.1 m/km in under five years, 0.263 m/km per year; section 0112, in blue, stays near 0.85, 0.004 per year; the other six lie between.](cee_l08_trees_explanations/roughening.png)

::: notes
23 CFR 490 sets the pavement condition measures: IRI below 95 inches per mile (1.5 m/km) is
good, above 170 (2.7 m/km) is poor. The decision: which designs to build and which roads to
resurface first; it needs to know what drives the rate of roughening. Data: FHWA's Long-Term
Pavement Performance program, Standard Data Release 40 (September 2026), public: 360 asphalt
sections in 53 states and provinces, 203 sites, 3,124 roughness visits. GPS-1 and GPS-2
(asphalt on granular and on bound base, in service when LTPP began) and SPS-1 (new sections
built to a factorial of asphalt thickness, base type, and drainage), original construction
only. The 9 inputs: asphalt, granular-base, and treated-base thickness; trucks (equivalent
single-axle loads per year); freezing index, precipitation, and mean temperature; the first
IRI measured and the age at that visit. Target: the least-squares slope of IRI on age, from at
least three visits over four years; median 0.026 m/km per year. The figure: the attempt's
site; the slopes of these lines are the rates in the table.
:::

## Your turn: what slows the roughening?

| Kansas SPS-1 | 0103 | 0105 | 0111 | 0112 | 0104 | 0110 | 0109 | 0159 |
|:------------------|------:|------:|------:|------:|------:|------:|------:|------:|
| Asphalt (in) | 3.6 | 4.2 | 4.2 | 4.7 | 6.6 | 6.7 | 8.1 | 11.3 |
| Treated base (in) | 7.1 | 3.8 | 11.9 | 15.9 | 12.1 | 8.2 | 3.3 | 0 |
| Granular base (in) | 0 | 4.0 | 0 | 0 | 0 | 0 | 12.0 | 0 |
| IRI rate (m/km/yr) | 0.174 | 0.263 | 0.024 | 0.004 | 0.044 | 0.046 | 0.083 | 0.083 |

::: notes
Eight sections of one SPS-1 project in Kansas: same site, same traffic (about 194,000 ESAL a
year), same climate (freezing index 211), different designs. Two minutes on your own: what
slows the roughening here? Write the rule you would take to the DOT.
Answers: the two fastest (0103, 0105) have thin asphalt (under 4.5 in) on 7 to 8 in of base;
thin asphalt on 12 to 16 in of treated base (0111, 0112) stays smooth; thick asphalt alone does
not guarantee it (0109, 0159). Within
one site, structure matters, and both layers count. Keep your rule: the next slides ask whether
a model trained on all 360 sections finds it.
:::

## A forest predicts; Shapley explains

![Diagram of the course pipeline with the decoder highlighted. A section x goes to an encoder of 9 inputs, giving z; the decoder, a forest, averages 500 regression trees, giving y-hat, compared with the measured IRI growth rate by the squared error. Above, 349 sections in 52 states, all but Kansas, train each tree on a bootstrap sample, trying 3 random inputs at each split. Below, Kansas SPS-1 section 0105, with Kansas held out of training: asphalt 4.2 inches, first visit at 0.5 years, first IRI 0.9 m/km, 200,000 ESAL a year. The average prediction is 0.041; the age at the first visit changes it by minus 0.005, the first IRI by minus 0.004, trucks by minus 0.003, and 6 more inputs by minus 0.003, so the forest predicts 0.026 m/km per year. The measured rate was 0.263, about ten times faster.](cee_l08_trees_explanations/forest_pipeline.png)

::: notes
Section 0105 is the fastest in the attempt. The forest, trained without Kansas, predicts a rate
near the national average, and the Shapley values split that prediction among the inputs. The
explanation is exact and clear; it explains a prediction that is wrong by a factor of ten.
:::

## Trees split until leaves agree

$$\min_{j,t}\ \sum_{i\in L}(y_i-\bar{y}_L)^2+\sum_{i\in R}(y_i-\bar{y}_R)^2$$

Greedy splits; a leaf predicts its mean.

![The plane of freezing index against first IRI, with the 360 sections as dots, cut by a depth-3 tree into six rectangles shaded by their mean rate: most sections, with freezing below about 1,088, fall in boxes predicting 0.019 to 0.064 m/km/yr; a band from about 1,088 to 1,451 predicts 0.196, from 5 sections in 2 states; beyond, 0.091.](cee_l08_trees_explanations/partition.png)

::: notes
Assumption: the rate is roughly constant within boxes of input space, found by splits. CART
(Breiman, Friedman, Olshen and Stone, 1984): at each node, the split x_j <= t that most reduces
the squared error. Sorting each feature and sweeping thresholds with running sums gives every
candidate split in O(n log n) per feature. Units do not matter: splits depend on the order of
values only, so no scaling is needed. Interactions come free. Deep trees have low bias and high
variance (Lecture 3). Forsyth §2.2.1–2.2.2 builds classification trees the same way, choosing
splits by information gain instead of squared error. The figure: a depth-3 tree on two inputs,
to see the boxes. The tall 0.196 box rests on 5 sections from 2 states, the fewest a leaf may
hold here: a tree reports such a box as confidently as the 170-section one.
:::

## Average trees; boost the residuals

- **Bagging:** fit $B$ trees on bootstrap samples; average. For trees with variance $\sigma^2$
  and correlation $\rho$:
  $$\mathrm{Var}\Big(\frac{1}{B}\sum_b T_b\Big)=\rho\,\sigma^2+\frac{1-\rho}{B}\,\sigma^2$$
- **Random forests** lower $\rho$: each split looks at a random subset of inputs.
- **Boosting:** $F_m=F_{m-1}+\nu\,h_m$, each small tree $h_m$ fit to the residuals of
  $F_{m-1}$ (the negative gradient of squared error).

::: notes
Breiman (2001) for random forests: ours tries 3 of the 9 inputs at each split (p/3, his
choice for regression); scikit-learn's regressor tries all 9 by default, which is bagging.
Friedman (2001) for gradient boosting, with shrinkage nu (here 0.03, 200 trees of depth 3). Bagging attacks variance; boosting attacks bias, one weak tree
at a time, and overfits if run too long. Forsyth §12.1.3 fits each new regression tree to the
residuals, as greedy stagewise regression.
:::

## A tree is terraces; a forest, soft steps

![Two 3D surfaces of predicted IRI growth over freezing index and first IRI, both trained on all 360 sections. One tree of depth 3: six flat terraces joined by vertical cliffs, with one tall block at 0.196 m/km/yr where the freezing index is 1,088 to 1,451. Three hundred trees of the same depth, each on a bootstrap sample, averaged: the rise sits in the same band but tops out at 0.14, climbing in many small steps.](cee_l08_trees_explanations/terraces.png)

::: notes
Two inputs, to see it. A tree is a set of boxes, so its prediction is a set of flat terraces
with cliffs at the splits: 6 levels here. Averaging 300 trees grown the same way (depth 3, at
least 5 sections a leaf), each on a bootstrap sample, puts the cliffs in slightly different
places, so the average climbs in many small steps (1,577 distinct levels on the plotted grid).
The average's peak is 0.14 against the tree's 0.196: the tall block rests on only 5 sections
from 2 states, and averaging lowers the variance. (Both inputs are tried at every split: with two inputs, sampling them
has little to do, so this is bagging.)
:::

## Boosting fits small trees to residuals

![Three panels of IRI growth against first IRI for the 360 sections, with a boosted model of depth-2 trees, learning rate 0.1, in orange. After 1 tree: almost flat near 0.045. After 10 trees: a gentle staircase rising with first IRI. After 200 trees: spikes and drops that follow single sections, chasing noise.](cee_l08_trees_explanations/boosting.png)

::: notes
Each round fits a small tree to what the current model gets wrong and adds a tenth of it. Early
rounds pick up the broad pattern; late rounds fit individual sections. Boosting attacks bias, one
weak tree at a time, and overfits if it runs too long, so the number of rounds must be chosen
on held-out data, by the unit of use: the next slides choose it inside each fold, on the
training states only. One input here, to see it; the lecture's boosting uses all nine (rate
0.03, depth 3).
:::

## Three ways to explain a model

**Shapley:** split one prediction among the inputs, exactly.

Kansas 0105: from 0.041 to 0.026. Measured: **0.263**.

![Waterfall of exact Shapley values for the forest's prediction for Kansas section 0105, trained with Kansas held out: from the average prediction, 0.041 m/km/yr, age at first visit, first IRI, trucks, and freezing index each lower it, and five small terms barely move it, ending at the prediction 0.026; the measured rate, 0.263, lies far off the right edge.](cee_l08_trees_explanations/waterfall.png)

::: notes
Permutation importance: shuffle one input; how much does R^2 drop? Partial dependence: average
prediction as one input varies, the others as in the data. Shapley values: split one prediction
among the inputs (Lundberg & Lee, 2017): phi_j = sum over subsets S without j of
|S|! (d - |S| - 1)! / d! times [v(S + j) - v(S)], where v(S) is the average prediction with the
inputs in S fixed to the section's values and the rest taken from 100 background sections.
Shapley (1953) derived these weights as the only division satisfying efficiency, symmetry, the
null player, and additivity. With 9 inputs there are 512 subsets: the notebook computes them
exactly. These are interventional values: mixing a section's freezing index with a background
section's temperature also asks the forest about climates that do not exist (the two
correlate at -0.83). The phi_j add up to the prediction minus the average: the waterfall
explains, exactly, a prediction ten times too low. All three describe the model; whether they
describe pavements depends on whether the model predicts pavements: Break it checks.
:::

## Your turn: predict the check

```python
def best_split(X, y, min_leaf=1):
    best, n = (-1, 0.0, 0.0), len(y)
    sse0 = ((y - y.mean()) ** 2).sum()
    for j in range(X.shape[1]):
        o = np.argsort(X[:, j], kind="stable")
        xs, s, s2 = X[o, j], np.cumsum(y[o]), np.cumsum(y[o] ** 2)
        for i in range(min_leaf, n - min_leaf + 1):
            if xs[i - 1] == xs[i]:
                continue  # no threshold between equals
            sse_l = s2[i - 1] - s[i - 1] ** 2 / i
            sse_r = s2[-1] - s2[i - 1] - (s[-1] - s[i - 1]) ** 2 / (n - i)
            gain = sse0 - sse_l - sse_r
            if gain > best[2] + 1e-12:
                best = (j, (xs[i - 1] + xs[i]) / 2, gain)
    return best
```

**Predict:** at depth 3, which input splits first? What training R²?

::: notes
The reference: scikit-learn's DecisionTreeRegressor(max_depth=3, min_samples_leaf=5). Take a
show of hands for the first split (asphalt? trucks? freezing?) and for the training R^2.
:::

## The tree fits; can it predict?

Matches scikit-learn to $10^{-16}$. First split: **freezing index** at 1,088 °C-days.

![Horizontal bars of R squared against the mean on the training sections: full-depth tree 1.00, depth-3 tree 0.21, random forest 0.67.](cee_l08_trees_explanations/overfit_train.png)

::: notes
Training R^2 at depth 3: 0.21. A full-depth tree: 360 leaves, training R^2 = 1.00; the forest:
0.67. The full tree has one leaf per section: it reproduces the data and says nothing about new
sites. The brackets: above, the mean, RMSE 0.0554 m/km/yr with states held out; below, four
groups of SPS-1 sections built alike at one site (the same three thicknesses when rounded to the
inch), whose rates differ by an RMS 0.035 m/km/yr, a rough measure of what no input can explain
(construction, local soil, measurement). Can it predict? The break-it step holds out whole
states, as a DOT would use a model built elsewhere.
:::

## Predict before we test

1. **Folds.** The forest's $R^2$ against the mean: random sections, sites held out, states held
   out?
2. **Importance.** Trucks rank high on the training states. On held-out states?
3. **Depth.** The full tree, $R^2=1.00$ in training. With states held out?
4. **Asphalt.** Effect of an inch of asphalt on the rate: across all 360 sections, and within
   SPS-1 sites?

::: notes
One minute; students commit in writing. Each of the next slides reveals one answer: validation
(1), explanation without skill (2), variance (3), and design against observation (4).
:::

## Random folds promise; states deny

Tuned forest $R^2$: **+0.14** random sections, **+0.02** sites held out, **−0.001** states held out.

![Line plot of R squared against the mean for three ways of holding out sections, each model tuned inside every fold. Forest, tuned: 0.14 with random sections, 0.02 with sites held out, minus 0.001 with states held out. Boosting, tuned: 0.08, minus 0.002, minus 0.02. Linear: 0.01, minus 0.03, minus 0.05. Boosting at a fixed 200 rounds: 0.09, minus 0.08, minus 0.22. Vertical bars show each model's range over five assignments of sites or states to the folds: no bar for states rises above zero. A dotted line at zero marks the mean.](cee_l08_trees_explanations/folds.png)

::: notes
This is the lecture's decisive figure, and Lecture 3's lesson in a new place. Each outer fold
tunes the forest's leaf size and boosting's rounds by five inner folds of its own training
rows, grouped like the outer ones: with states held out, the inner folds hold out training
states. Tuned for held-out states, both shrink toward the mean: leaves of 40 to 80 sections,
0 to 10 rounds (in 2 of 5 folds, none: the mean itself). Over five assignments of the groups
to folds, the tuned models reach at most +0.016 across sites and stay at or below −0.001
across states: no model beats the mean by more than 0.02. A state DOT applying the national
model to its own roads is in the third column. Sections at one site share soil, climate,
traffic, and crews; random folds leak them. The break-it row: boosting left at 200 rounds,
−0.22 across states.
:::

## Importance without skill

Training: trucks rank fourth (0.13). Held-out states: shuffling trucks **helps** (−0.09), mostly in one fold.

![Dot plot of permutation importance for the nine inputs, on training states and on held-out states, averaged over five folds. On training states: first IRI 0.20, freezing index 0.17, age at first visit 0.17, trucks 0.13, precipitation 0.12, mean temperature 0.11, asphalt thickness 0.07, treated base 0.07, granular base 0.04. On held-out states all are near zero except first IRI, 0.06, and trucks, minus 0.09; freezing index, precipitation, and mean temperature are each about minus 0.02.](cee_l08_trees_explanations/importance.png)

::: notes
Importance measured on the training data rewards whatever the model memorized; measured on
held-out states, it measures what transfers. The lecture's forest (leaves of 3, untuned) is
explained here. On the training states, permutation importance ranks first IRI (0.20), freezing
(0.17), age (0.17), and trucks (0.13). Held out, trucks are negative in four of five folds, but
one fold carries most of the mean: -0.084, -0.024, -0.300, +0.008, -0.035; without the third,
-0.034. Freezing and temperature are correlated (-0.83), so shuffling one alone creates
impossible climates (Hooker, Mentch & Zhou, 2021). Climate inputs on held-out states: -0.02.
Only first IRI keeps some
importance (0.06): sections that start rougher roughen faster (correlation 0.23 between first
IRI and the rate).
:::

## A deep tree memorizes

Full depth: one leaf per section, training $R^2$ = 1.00, states held out **−1.66**.

![Horizontal bars of R squared against the mean, training against states held out. Full-depth tree: 1.00 in training, minus 1.66 with states held out. Depth-3 tree: 0.21 and minus 0.15. Random forest: 0.67 and minus 0.12.](cee_l08_trees_explanations/overfit.png)

::: notes
The full tree's held-out RMSE is 0.090 m/km/yr against 0.055 for the mean. A depth-3 tree:
-0.15. The lecture's forest: -0.12, far better than one deep tree, as bagging promises, and
still worse than the mean. Tuned inside each fold, its leaves grow to 40 to 80 sections and it
ties the mean (-0.001). Averaging and tuning cut the variance; neither creates signal the inputs
lack.
:::

## Design found what mining missed

All 360 sections: an inch of asphalt, **+0.0006**. Within 15 SPS-1 sites: **−0.0043** per inch.

![Plot of the change in IRI growth rate per inch of asphalt. Designed, within 15 SPS-1 sites: minus 0.0043, with a cluster-robust 90% interval from minus 0.0075 to minus 0.0012, below zero. Observational, across 360 sections: plus 0.0006, just above zero.](cee_l08_trees_explanations/designed.png)

::: notes
Within sites, built side by side, comparing sections removes soil, climate, and traffic. The
within-site estimate subtracts each site's mean from rates and thicknesses (site fixed effects)
and fits asphalt, treated base, and granular base together; drainage, the factorial's other
factor, is not in these data. The interval is cluster-robust (CR1, sites as clusters; its
small-sample factor counts the 15 site means, the more conservative convention) with t on 14
degrees of freedom: −0.0075 to −0.0012. A percentile bootstrap of whole sites (500 resamples)
gives −0.0073 to −0.0021, but with only 15 clusters it undercovers (Cameron, Gelbach & Miller,
2008). Leaving out one site at a time, the estimate stays between −0.0050 and −0.0031. Treated
base: −0.0031 per inch, 90% interval −0.0055 to −0.0008. An inch of asphalt on a road
roughening at 0.03 m/km/yr slows it by about 14%. Across all sections the correlation is 0.025,
and the nine measured inputs do not explain why: thickness barely tracks trucks (correlation
0.17), and a linear regression on all nine inputs still gives +0.00014 per inch: adjusting for
what was measured does not recover the effect. LTPP built the SPS experiments for this reason.
:::

## Explain only what predicts

| Step | Here |
|:--------------------------|:--------------------------------------------|
| 1. Held-out skill, by the unit of use | states held out: tuned or not, no model beats the mean |
| 2. Then explain, if it predicts | trucks' importance does not survive |
| 3. Decide from design where you can | SPS-1: thicker asphalt and treated base slow roughening |
| 4. Collect what the model lacks | as-built smoothness, materials, drainage, maintenance |

::: notes
The decision for the DOT: the national forest adds nothing over the mean for its roads; the
experiment says thicker asphalt and treated bases slow roughening at a given site. Data that
could help: the analysis-ready LTPP tables, materials tests, and the state's own pavement
management records, validated by holding out its districts.
:::

## When explanations mislead

| Do not | Because |
|:------------------------------|:------------------------------------|
| explain a model without held-out skill | SHAP explains the model, not the road |
| trust training-data importance | it rewards memorization |
| split credit between correlated inputs | freezing and temperature (−0.83); PDPs extrapolate |
| read explanations as causes | across sections thickness was not assigned: design was |

::: notes
When not to use the method is part of the method. Each row is a failure this lecture's data
showed. Ask of every explanation: does the model predict held-out units, and would an
experiment agree?
:::

## Plausible stories steer budgets

| A plausible story | Here |
|:--------------------------|:--------------------------------------------|
| "Trucks drive roughening" | trucks hurt the predictions they "explain" |
| Move money to heavy routes | roads already poor keep costing drivers repairs and fuel |
| Mining instead of experiments | LTPP's designed sections found the thickness effect |

::: notes
The social and economic side: an explanation looks like evidence, and decisions follow it.
Ask: who would notice if a DOT's resurfacing priorities followed the training-data importance?
:::

## Vote: what do you tell the DOT?

The DOT asks which factor to act on when it builds or resurfaces.

- **A.** Trucks: high importance in the national forest.
- **B.** Freezing: the depth-3 tree's first split.
- **C.** Asphalt and treated-base thickness: the SPS-1 within-site contrast.
- **D.** Nothing: no model beats the mean.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it. C:
it comes from a designed comparison with an interval that excludes zero. A and B come from
models with negative held-out R^2, and B's split puts only 10 of the 360 sections on its cold
side. D is right about the model and too strong about the data.
Then ask what would change the answer: the state's own records with a held-out district, or a
model that beats the mean across states.
:::

## Trees still run tabular AI

| Today | At scale | Lecture |
|:--------------------|:----------------------------------------------|:----------|
| Forests, boosting | still beat deep networks on typical tables (Grinsztajn et al., 2022); XGBoost, LightGBM | — |
| Shapley values | TreeSHAP: one call in XGBoost or LightGBM; the check is not | — |
| Small tables | tabular foundation models predict in one pass (Hollmann et al., 2025) | 17 |
| The test | held-out skill by the unit of use, before any explanation | today |

::: notes
Grinsztajn, Oyallon and Varoquaux benchmarked 45 datasets; Chen and Guestrin (2016) introduced
XGBoost; Lundberg et al. (2020) made exact Shapley values fast for trees (TreeSHAP). TabPFN, the tabular foundation model in Nature (Hollmann et al., 2025), handles small
tables like ours. Lecture 17 covers transformers, the architecture behind it.
:::

## Exit question

A consultant shows the DOT a SHAP plot in which trucks drive roughening, from a model with
$R^2=0.9$. What two questions do you ask before believing it?

**In the notebook** (`cee_l08_trees_explanations.ipynb`): CART from scratch checked against
scikit-learn, forests and boosting with three kinds of folds, importance on training and held-out
states, exact Shapley values, and the SPS-1 contrast.

::: notes
Answers: (1) Was the 0.9 measured on held-out units like the DOT's roads (sites, states, years),
or on random folds or the training data? (2) Do the explanations agree with a designed
comparison, or could confounding explain them, as it hid the asphalt effect here? Collect the
answers; they seed the next warm-up.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- Trees: terraces. Forests: averaged. Boosting: residuals.
- Validate by the unit of use.
- Explanations describe the model.
- Design beats mining.

**Next:** outliers and robust estimation: finding a burst pipe.
:::
::: {.column}
![Four thumbnails of the lecture's key figures: one tree's terraces against a forest's softer steps; R squared by kind of fold; permutation importance on training and held-out states; the within-site effect of asphalt thickness.](cee_l08_trees_explanations/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- 23 CFR 490 (2017). Pavement condition measures.
- Breiman (2001). *Machine Learning* 45.
- Breiman et al. (1984). *Classification and Regression Trees*.
- Cameron et al. (2008). *REStat* 90.
- Chen & Guestrin (2016). *KDD*.
- FHWA (2026). LTPP InfoPave, SDR 40.
- Forsyth (2019). Springer, §2.2, §12.1.
:::
::: {.column}
- Friedman (2001). *Ann. Statist.* 29(5).
- Grinsztajn, Oyallon & Varoquaux (2022). *NeurIPS*.
- Hollmann et al. (2025). *Nature* 637.
- Hooker, Mentch & Zhou (2021). *Stat. Comput.* 31.
- Lundberg & Lee (2017). *NeurIPS*.
- Lundberg et al. (2020). *Nat. Mach. Intell.* 2.
- Shapley (1953). *Contrib. Theory of Games* II.
:::
::::::
