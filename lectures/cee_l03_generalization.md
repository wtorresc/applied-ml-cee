---
title: Generalization and Validation
subtitle: "Lecture 3 · How wrong will the model be on the next bridge?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: questions from Lectures 1–2

On your own, no notes, two minutes:

1. You tried ten values of $k$ and kept the one with the best test score. What went wrong?
2. Why did water carry only 1.2% of the raw distance between two concrete mixes?
3. A random split of the concrete tests gave 5.7 MPa; holding out whole designs gave 6.9 MPa.
   Which do you report to a client, and why?

**Today:** how wrong will a model of bridge condition be on the next bridge?

::: notes
Give the answers before moving on; recall helps most with feedback.
1. The test set chose k, so it is no longer a test: the best of ten scores is optimistic.
   Choose k on validation data and look at the test set once, at the end (Lecture 1).
2. A raw distance weighs each input by its variance, and water varies little in kg/m³ (only
   superplasticizer varies less); cement and the aggregates dominate. Units, not importance,
   decided the distance (Lecture 2).
3. Designs held out: the client's mix is a new design, and the random split put the same design,
   tested at other ages or entered twice, in training (Lecture 2).
Question 3 is today's lecture in miniature.
:::

## Rank bridges before they fail

- Inspectors rate each part 0–9; the lowest at 4 or less is **poor**.
- Which bridges first? Wrong "fair": a poor bridge waits.

![Bar chart of the lowest condition rating of the 22,015 Illinois bridges in the 2018 inventory: 29 at 0, 12 at 1, 145 at 2, 477 at 3, and 1,444 at 4, together 9.6% poor, hatched orange; 3,655 at 5, 5,861 at 6, 5,349 at 7, 4,707 at 8, and 336 at 9.](cee_l03_generalization/ratings.png)

::: notes
Frame the problem by what an error costs before talking about accuracy. The rating scale comes
from FHWA's Recording and Coding Guide (1995): 7 good, 5 fair, 4 poor, 3 serious. FHWA's
good/fair/poor classification uses the lowest of the three ratings; 4 or less is poor.
The data: 131,851 bridge-year records of 23,464 bridges in 102 counties, every two years; between
21,935 and 22,015 bridges a year. A model could rank bridges between inspections, or where
records are missing.
A model could predict the rating from the inventory. A wrong "fair": a poor bridge waits, and
the public carries the risk. A wrong "poor": a crew and a budget go to a sound bridge. Data:
FHWA National Bridge Inventory (NBI), Illinois, 2014-2024, about 22,000 bridges a year,
culverts aside.
:::

## Your turn: rate bridge Q

| | Bridge, inventory | County | Built | Vehicles a day | km from Q | Rating |
|:--|:-----------------|:-------|-----:|----------:|-------:|------:|
| Q | IL 53 north, 2018 | Cook | 1963 | 157,100 | 0 | ? |
| A | IL 53 south, 2018 | Cook | 1963 | 157,100 | 0.01 | 4 |
| B | Q itself, 2016 | Cook | 1963 | 147,600 | 0 | 4 |
| C | I-290 west, 2018 | DuPage | 1970 | 75,750 | 15.9 | 6 |
| D | I-290 east, 2018 | DuPage | 1971 | 75,750 | 15.9 | 6 |

::: notes
Q is IL 53 northbound over Kirchoff Road; A, IL 53 southbound; C and D, I-290 westbound and
eastbound over Addison Road. All five are steel beam bridges, described by the same 12 inputs: age, years since work,
traffic, trucks, length, width, material, and location. Two minutes on your own: estimate Q's
2018 rating and write down which rows you relied on. Then a second question: if Cook County had never shared its records, which rows would not exist?
Cover them and estimate again.
Answers: A (its twin, a few meters away) and B (Q itself two years earlier) both say 4; the mean
of all four rows is 5.0. Without A and B, only C and D remain (6). Measured: 4, poor.
Keep the class's answers; the next slide traces Q, and the Break-it step returns to them.
How Q was chosen: among poor twin bridges carrying at least 50,000 vehicles a day in the first
block of held-out counties, whose nearest record under random folds is their twin, the one where
the two ways of holding out disagree most (ties to the busier bridge). A and B are Q's two
nearest training records under random folds; C and D, its two nearest with Cook held out. C and
D are themselves twins: I-290 over Addison Road.
:::

## The split decides who is near

![Diagram of the course pipeline with the held-out error highlighted. A bridge x goes to an encoder that scales 12 inputs, giving z; the decoder averages the ratings of the k nearest training records, which are stored, 55,155 records from 2014 to 2018; the prediction y-hat is compared with the rating y by the held-out error, the square root of the mean squared difference, honest only if the held-out bridges look like the next ones. Below, bridge Q, IL 53 northbound in Cook County, built 1963, never rebuilt, 157,100 vehicles a day; age 54 years, 54 years since work, ln ADT 12.0. With k equal to 1: under random folds its nearest record is its twin, 10 meters away, prediction 4, error 0; with its county held out its nearest record is in DuPage County, 16 kilometers away, prediction 6, error +2. Its rating is 4, poor.](cee_l03_generalization/validation_pipeline.png)

::: notes
This is bridge Q from your attempt, with k = 1. The model is the same in both rows; only what
was held out changed. With random folds, the twin answers; with Cook held out, a bridge in
another county answers, and a poor bridge comes out fair. With k = 50, the two predictions are
5.5 and 5.2: averaging 50 records dilutes the twin.
The training set holds each inspection once: 65,928 inventory records from 2014 to 2018
repeat 10,773 inspections already reported two years earlier (bridges on a 48-month
interval); we keep 55,155 inspections of 22,720 bridges.
:::

## Validation assumes look-alike cases

**Assumption:** held-out cases are exchangeable with the cases the model will face.
$$\hat{L}=\frac{1}{n}\sum_{i=1}^{n}\ell_i,\qquad\mathbb{E}\,\hat{L}=L,\qquad\mathrm{Var}\,\hat{L}=\frac{\mathrm{Var}\,\ell}{n}$$

Train: 2014–2018 inventories. Test: 2024, which chooses nothing.

::: notes
Derivation: if each held-out (x_i, y_i) is drawn like a future case and the model never saw it,
each loss l_i has expectation L, so by linearity E L-hat = L. If the losses are independent,
Var L-hat = Var(l)/n, and Chebyshev's inequality gives P(|L-hat − L| ≥ ε) ≤ Var(l)/(n ε²)
(Forsyth, §3.1). Dependence breaks both: a held-out case with a near-copy in training has a
smaller loss than a future case (bias), and correlated losses shrink the effective n.
Choosing k by the smallest of several validation errors also biases that error low:
E[min_k L-hat_k] ≤ min_k E[L-hat_k] (Cawley & Talbot, 2010). Hence a test set that only
scores the choices validation made, and makes none.
Both identities hold only if the held-out cases are drawn like future ones, independently, and
were not used in training.
:::

## Twins make leave-one-out lie

| The five records | Rating | Leave one out: nearest | Predicted | Leave its county out: nearest | Predicted |
|:--|--:|:--:|--:|:--:|--:|
| Q | 4 | A | 4 | C | 6 |
| A | 4 | Q | 4 | C | 6 |
| B | 4 | Q | 4 | C | 6 |
| C | 6 | D | 6 | B | 4 |
| D | 6 | C | 6 | B | 4 |
| **RMSE** | | | **0.0** | | **2.0** |

::: notes
Leave one record out, 1-NN: each record finds its twin or its own other inspection, so the RMSE
is 0.0. Leave its county out: Q, A, and B fall back on the DuPage pair (6), and C and D on B
(4): RMSE 2.0. Same model, same five records; only what was held out changed.
Distances use the 12 scaled inputs, with the scaler of all training records. B is Q itself two
years earlier, a twin in time. Ask: which of the two RMSEs describes a bridge in a county with
no records? Work the distances by hand in the notebook.
:::

## Error = noise + bias² + variance

Assume $y=f(\mathbf{x})+\varepsilon$ with $\mathbb{E}\,\varepsilon=0$,
$\mathrm{Var}\,\varepsilon=\sigma^2$, independent of the training set $\mathcal{D}$; let
$\bar{f}=\mathbb{E}_{\mathcal{D}}\hat{f}_{\mathcal{D}}$. At a fixed $\mathbf{x}$:

$$\mathbb{E}\big[(y-\hat{f}_{\mathcal{D}})^2\big]=\mathbb{E}\big[(\varepsilon+(f-\bar{f})+(\bar{f}-\hat{f}_{\mathcal{D}}))^2\big]$$

$$=\sigma^2+(f-\bar{f})^2+\mathbb{E}_{\mathcal{D}}\big[(\hat{f}_{\mathcal{D}}-\bar{f})^2\big]$$

- The cross terms vanish: $\varepsilon$ has mean 0 and is independent of $\mathcal{D}$;
  $\hat{f}_{\mathcal{D}}-\bar{f}$ has mean 0 over $\mathcal{D}$; $f-\bar{f}$ is fixed.
- $\sigma^2$ is rating noise between inspectors: no model removes it.

::: notes
noise + bias² + variance. Bias: even the average model misses f. Variance: each trained model
wanders around the average one. Forsyth §11.1.1 derives the same result (he writes the noise as
ξ); Geman, Bienenstock and Doursat (1992) introduced it to machine learning.
:::

## Small k follows noise; huge k, the mean

![Three panels of k-NN predictions of the condition rating from one input, age, each fitted to six random samples of 1,000 bridges from the 2018 inventory, with the mean rating at each age over all of its bridges as a dashed line falling from about 8.4 at age 0 to about 4.8 at 110 years; the vertical axis runs over every rating, 0 to 9. k = 1: the six fits jump between whole ratings from 0 to 9, variance 0.99, bias squared 0.22. k = 30: the six fits hug the dashed line, variance 0.04, bias squared 0.01. k = 1,000: all six are the same flat line near 6.3, variance 0.00, bias squared 1.09.](cee_l03_generalization/resample.png)

::: notes
The derivation's two terms, seen: each panel fits k-NN to six different samples of 1,000 bridges
(one input, age, to see it in 2D). Variance is how much the six fits disagree; bias² is how far
their average sits from the dashed truth. k = 1 copies one bridge's rating: the fits disagree
everywhere. k = 1,000 averages every bridge: the fits agree, and all are wrong. The next slide
measures the same split with all 12 inputs, on held-out counties.
:::

## k trades variance for bias

Held-out error lowest near $k=100$ (1.13). Training error says nothing: 0.001 at $k=1$.

![Line plot of squared error in rating points squared against k from 1 to 1,000 on a log axis, for 18 held-out counties and 20 half-size training sets. Held-out error falls from 2.1 at k = 1 to its minimum of about 1.1 near k = 100, then rises to 1.24 at k = 1,000. Variance falls from 0.67 at k = 1 to near zero. Bias squared plus noise is the difference, lowest near k = 50. Training error rises from near 0 at k = 1.](cee_l03_generalization/bias_variance.png)

::: notes
Every error on this slide is squared, in points². Variance is the spread of predictions over
the 20 training sets, averaged over the held-out records; the training sets are random halves
of one pool of bridges, so this is the spread over such halves, not over independent samples
from the population. Bias² + noise is the squared distance of the average prediction to the
rating; the two add up exactly to the held-out error. Variance falls by a factor of about 5.6
from k = 1 to 10 (0.67 / 0.12), not 10: the neighbors themselves change from one training set
to the next. At small k, bias² + noise
is also high: the average prediction still carries the few neighbors' own rating noise. The
training error at k = 1 is not exactly 0 because a few records share their inputs with another
record of a different rating.
With the k neighbors fixed, only their noise varies: Var f-hat = sigma^2 / k. Measured on 18
held-out counties, 20 half-size training sets: variance 0.67 points^2 at k = 1, 0.12 at k = 10.
By k = 1,000, bias raises the held-out error to 1.24.
:::

## Cross-validation is a loop

```python
def cv_predict(X, y, folds, ks):               # ks: array of k values
    pred = np.full((len(y), len(ks)), np.nan)
    for tr, te in folds:                       # te: one held-out block
        sc = StandardScaler().fit(X[tr])       # training rows only
        Z_tr, Z_te = sc.transform(X[tr]), sc.transform(X[te])
        nn = NearestNeighbors(n_neighbors=ks.max(), algorithm="brute")
        _, idx = nn.fit(Z_tr).kneighbors(Z_te) # nearest first
        total = np.cumsum(y[tr][idx], axis=1)  # sums of nearest 1..K
        pred[te] = total[:, ks - 1] / ks
    assert not np.isnan(pred).any()            # each row held out once
    return pred
```

**Predict:** at $k=50$, will it match scikit-learn on every record?

::: notes
Counties held out, all 55,155 records. Take a show of hands: on every record, on almost every
record, or often different. The folds:
KFold(5, shuffle=True) for random records; GroupKFold(5) by bridge, or by county. Ten bridges
change county code between inventories, so each bridge goes to its first county's fold. The reference
is scikit-learn's cross_val_predict with a StandardScaler and KNeighborsRegressor pipeline and
the same folds. One neighbor search per fold gives every k at once: the mean of the nearest k is a running sum
divided by k. The scaler is part of the model, so it is fit inside each fold.
:::

## Between the mean and the twins

Counties held out: **1.09** at $k=50$. Matches scikit-learn except at ties.

![Dot plot of RMSE in rating points with counties held out: predicting the mean 1.44; k-NN with k = 50, 1.09; k-NN with k = 1, 1.45; twin bridges rated on one visit, a floor, 0.41.](cee_l03_generalization/bracket.png)

::: notes
Ties: at k = 50, 865 records have two training records equally far away at rank 50;
which one counts is arbitrary in both codes. On this machine 48 of them changed the mean, by
at most 0.06; the count depends on the last bit of the distances, so it changes with the
machine and its number of BLAS threads. At k = 1 the model is no better than predicting the mean. The floor: 872 pairs of twins
in the 2018 inventory (parallel structures, same year, less than 200 m apart) differ by RMS
0.58, and 0.58/√2 = 0.41; 77% have the same rating and 97% are within one point. With equal
true conditions and independent errors, 0.41 would be σ, the noise of one rating. But 772 of
the 872 pairs (89%) were inspected in the same month, likely by the same crew: correlated
errors (ρ) shrink the difference to 2σ²(1 − ρ), and the other 100 pairs disagree more, 0.66.
So 0.41 is the disagreement within one visit, a lower bound on the noise between inspectors.
A national study found more spread between inspectors from different states: 68% of ratings
within one point of the average, 95% within two, σ ≈ 1 (Moore et al., 2001; 49 inspectors
from 25 states).
At k = 1: 1.45. Above: predicting the mean, 1.44. Below: twins rated on one visit, 0.41.
:::

## Predict before we test

1. **Random folds.** Hold out random records instead of whole counties. Higher or lower error?
   Which $k$ will it pick?
2. **The future.** Train on 2014–2018, then test on the 2024 inventory. Nearer the random-fold
   number or the county number?
3. **Leaky columns.** Add the inspection interval and the structural evaluation from the same
   record. What happens to the error? Would they help in 2024?

::: notes
Give one minute; students commit in writing. Each of the next slides reveals one answer. Ask
who predicted it, and what the others missed.
:::

## Random folds reuse the same bridges

Random **0.90** ($k=2$) · bridges 1.03 · counties 1.09. 88% of held-out records have their bridge in training.

![Two maps of Illinois drawn from the 2018 bridge records. Left, random folds: the held-out records, blue squares, are scattered among the training records, grey dots, in all 102 counties. Right, counties held out: the held-out records fill 18 whole counties, in blocks.](cee_l03_generalization/folds_map.png)

::: notes
Holding out bridges is not enough for a new county: neighbors share corridors, eras, and
inspection crews. Each map shows the first of five folds. Random folds put a bridge's other inspections, its
twin, and its neighbors on the same road in training. Holding out whole bridges removes the
first; holding out counties also removes the twins, but a neighbor across a county border
stays in training. Which one matches the use depends on the
bridges the model will face (Roberts et al., 2017, on blocked cross-validation).
Bridges held out: best at k = 30; counties held out: best at k = 70. 99% of twins have their
twin in training under random folds.
:::

## Random folds pick the wrong k

Random folds pick $k=2$, promise **0.90**; in 2024 it scores **1.12**. Honest folds' $k$: 1.01.

![Line plot of RMSE against k from 1 to 100 on a log axis. Random folds: lowest, 0.93 at k = 1 and 0.90 at k = 2, then rising to 1.03 at k = 100; an arrow marks that random folds pick k = 2. Bridges held out: from about 1.3 at k = 1 down to 1.03 near k = 30. Counties held out: from 1.45 down to 1.09 near k = 70. The 2024 inventory, a thick dotted line labeled post hoc: from 1.20 at k = 1 down to 1.01 from k = 30 to 70, close to bridges held out and far above random folds.](cee_l03_generalization/k_curves.png)

::: notes
This is the lecture's decisive figure. Random folds are not only optimistic; they pick the wrong
model. Their curve rises with k while every honest curve falls: a small k copies the twin or the
bridge's own other inspection, which random folds leave in training. The 2024 test scores
only the k each validation chose (2, 30, 70). Its full curve is a post-hoc diagnostic, drawn
after the choices: reading k off it would turn the test set into a validation set.
Honest folds pick k = 30 and 70; at both, the 2024 inventory scores 1.01. Counties new to the model
in 2024: 1.06 at k = 70, as counties held out promised (1.09).
:::

## Old ratings go stale

1-NN error grows with time: 0.93, 1.07, 1.20 at 2, 4, 6 years after the 2018 inventory.

![Line plot of mean condition rating against age at inspection in eight age groups, for the 2014 and 2024 inventories. Both fall with age, from about 8 for bridges under 10 to about 5 for bridges over 80. For bridges under 30 years old the 2024 line lies below the 2014 line: 7.58 against 7.96 under 10 years; from 30 to 49 years the two agree.](cee_l03_generalization/drift.png)

::: notes
The three 1-NN errors are a post-hoc diagnostic: no validation chose k = 1, and the 2024 test
scores it only to show how a copied inspection ages, not to pick a model. The horizons count
inventories after 2018; a bridge's own last inspection in training can be older (intervals
reach 48 months).
With 1-NN, 70% of 2020 predictions copy the bridge's own earlier inspection, and 42% of 2024
predictions; the copy goes stale. The bias is +0.20 at k = 1 and +0.11 at k = 50: the model
expects better condition than the inspectors found. Same age, lower rating: validate forward in
time, at the horizon you will use. Test records are 2024 inspections not already in training (a 48-month bridge can report the same
inspection in two inventories). The horizon matters: a forward split with a two-year horizon
promises what random folds promise, so validate at the horizon of the use. Why newer bridges
rate lower in 2024 than in 2014 is a question for the inventory's owners: rating practice, the
bridges built, or both. The model cannot know; it only sees the past.
It overrates 2024 bridges by 0.20 on average. Bridges under 10 years old: mean 7.96 in 2014, 7.58
in 2024.
:::

## Some columns already know the answer

Interval and structural evaluation are set from the ratings: 1.10 → **0.61**. **Leakage.**

![Dot plot of RMSE with counties held out and k = 50: inventory only 1.10; adding the inspection interval 0.94; adding the structural evaluation 0.61; adding both 0.61.](cee_l03_generalization/leakage.png)

::: notes
Counties held out, k = 50: 1.10 with the inventory alone, 0.94 with the interval, 0.61 with the
structural evaluation. Both columns sit in the same record as the rating, so they look like
ordinary inputs; using them is leakage (Kaufman et al., 2012). The
Coding Guide says the inspection interval "could vary from inspection to inspection depending on
the condition of the bridge at the time of inspection", and that Item 67 uses the lowest of the
superstructure (Item 59), substructure (Item 60), and a load-based code (FHWA, 1995). Random
folds with both columns: 0.57. The question to ask of every column: would I have it when I
make the prediction? These runs use the 55,045 records that have both columns (110 lack
one), so the first number is 1.10, not 1.09.
Inspection interval (12, 24, 48 months) is set from the condition: mean rating 3.8, 5.7, 7.6.
The structural evaluation is computed from the ratings. Neither exists before the inspection.
:::

## Optimism misses poor bridges

Flag the worst 10%. Random folds promise **60%** of poor records caught; 2024 delivers **43%**.

![Curves of the percentage of poor records caught against the percentage of records flagged for inspection, from 0 to 40 percent flagged. Promised by random folds for k = 2, on the training records: the highest curve, 60 percent caught at 10 percent flagged. Delivered in 2024: 43 percent for k = 2 and 37 percent for k = 50 at 10 percent flagged. A dotted line marks 10 percent.](cee_l03_generalization/screening.png)

::: notes
Records are ranked by predicted rating, ties broken at random with a fixed seed. The promise is
computed on the 2014-2018 training records, several inspections per bridge, where 10.5% are
poor; in 2024 each record is one bridge, and 2,307 of them are poor.
A missed poor bridge is the costly error; how many bridges to flag is a budget decision, and the
validation error decides how much to trust the ranking.
The k = 50 model catches 37%. The lowest RMSE is not the best screen: choose the metric by the
decision (Lecture 6).
:::

## Rank bridges; inspect them anyway

| Use the model to | Not to |
|:------------------------------|:------------------------------|
| rank bridges for the next inspection | rate a bridge: an inspector does |
| plan budgets, with an error of about one point | replace the 24-month inspection (FHWA, 1995) |
| flag records that look out of line | predict new designs, other states, a later decade, unvalidated |
| | use leaky columns or a random-fold error |

::: notes
When not to use the method is part of the method. The honest error (about one point) is close
to the national spread between inspectors (Moore et al., 2001), so the model is useful to rank
and to plan, not to replace the inspector's rating.
An honest error of about one rating point means a predicted 6 can be a poor 4: the model ranks;
it does not rate. Uncertainty with a guarantee: calibration (Lecture 10), conformal prediction
(Lecture 11).
:::

## Whose bridges are in the data?

- Complete inventory, unequal error.
- Report the error **per county**, with its noise floor.

![Map of Illinois counties, each a dot at the centre of its bridges, sized by its number of records and coloured by RMSE with counties held out, k = 50, from 0.76 to 1.39. Franklin County, 0.76, and White County, 1.39, ringed, both in the south, one county apart.](cee_l03_generalization/county_map.png)

::: notes
Measurement bias becomes allocation bias. Counties with at least 300 records: 79; Cook
County, 1.06. Is the spread real? Deal the bridges to counties at random, 500 times: chance
alone spreads the counties by 0.06 around the state's 1.09, with a best and worst of about
0.95 and 1.25. The real spread is twice that, 0.13, and Franklin and White lie beyond all 500
random deals: the unequal error is real. But part of each extreme is chance, as in Lecture 1's
winner's curse: chance alone moves the best and worst of 79 counties that far from 1.09.
Ask: which counties would a model built on the state's own bridges serve worst,
and who would notice?
The NBI is a complete inventory: every public highway bridge, inspected on a schedule.
Crowd-reported data are not: Boston's Street Bump app found potholes through drivers'
smartphones, so its map followed who owned one (Crawford, 2013). A complete inventory is still
not served equally: with counties held out, k = 50 errs by 0.76 in Franklin County and 1.39 in
White County, both in the south with only Hamilton County between them. A ranking that is less accurate in some counties moves inspections and
money between them.
:::

## Foundation models face the same test

| Where | What validation hid |
|:----------|:------------------------------------------|
| Language models | Benchmark test sets in the training data: "contamination causes an overestimation of the performance" (Sainz et al., 2023) |
| Maps | Random validation: "predicts more than half"; spatial: "quasi-null predictive power" (Ploton et al., 2020) |
| Geospatial foundation models | 1 TB of imagery in pretraining: which places and dates do test tiles share? (Jakubik et al., 2023) |

::: notes
Lecture 18 mentions geospatial foundation models (Prithvi); Lecture 19 shows how to evaluate
language models.
The fix for contamination is today's temporal split: test on data newer than the training data.
Same assumption, larger models: the test cases must be exchangeable with the cases the model
will face, and must not have been seen in training. Ploton et al. trained a random forest on an
inventory of 11.8 million trees in central Africa. Prithvi (Jakubik et al., 2023) was pretrained
on Harmonized Landsat Sentinel-2 imagery.
:::

## Vote: which error do you quote?

A county engineer will rank her bridges with your model. Her county has never shared its
records. Which error do you quote?

- **A.** 0.90: random folds ($k=2$)
- **B.** 1.03: bridges held out ($k=30$)
- **C.** 1.09: counties held out ($k=70$)
- **D.** 1.01: the 2024 inventory, at B's $k=30$

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
C matches her use: her county is new to the model. A reuses her neighbors' bridges and her own
bridges' other inspections. B and D still saw her county's other bridges, twins included; D,
the 2024 score of the model B chose (not a k picked on the test set), is the right test for a
different use, the next inventory of the same bridges. The check:
counties new to the model in 2024 scored 1.06 at k = 70, close to C. Then ask what would
change the answer: her records joining the training data, or a forecast for the same bridges.
:::

## Exit question

Your model will forecast the next inspection of the same bridges. Which split do you validate
with, and which columns do you drop? Why?

**In the notebook** (`cee_l03_generalization.ipynb`): write `cv_predict`, match scikit-learn,
work the five records by hand, and reproduce the random, county, and 2024 errors and the leak.

::: notes
Answer: train on earlier inventories and validate on a later one, at the same horizon as the
use, with each inspection kept once. Drop the inspection interval and the structural evaluation:
the rating sets both. Collect the answers on paper or in a poll; they show what to revisit in
the next warm-up.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- Honest only for cases like the future ones.
- Hold out what the use holds out.
- Error = noise + bias² + variance.
- Drop columns the target decides.

**Next:** retrieval, clustering, and PCA, on building load profiles.
:::
::: {.column}
![Four thumbnails of the lecture's key figures: k-NN fits on six samples for k = 1, 30, 1,000; random folds against counties held out on the map; validation error against k for three fold schemes and 2024; the leaky columns cutting the error.](cee_l03_generalization/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Cawley & Talbot (2010). *JMLR* 11.
- Crawford (2013). *Harvard Business Review*.
- FHWA (1995). *Recording and Coding Guide*. FHWA-PD-96-001.
- Forsyth (2019). *Applied Machine Learning*. Springer.
- Geman, Bienenstock & Doursat (1992). *Neural Computation* 4(1).
- Jakubik et al. (2023). arXiv:2310.18660.
:::
::: {.column}
- Kaufman et al. (2012). *ACM TKDD* 6(4).
- Moore et al. (2001). FHWA-RD-01-020.
- Ploton et al. (2020). *Nature Communications* 11.
- Roberts et al. (2017). *Ecography* 40(8).
- Sainz et al. (2023). *Findings of EMNLP 2023*.
:::
::::::
