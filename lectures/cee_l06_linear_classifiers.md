---
title: Linear Classifiers and Costs
subtitle: "Lecture 6 · Which bridges will be poor in four years?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 3–5

On your own, no notes, two minutes:

1. You will forecast the next inspection of the same bridges. Which split do you validate
   with, and which columns do you drop?
2. Why solve least squares with QR or the SVD, and never with $(X^{\top}X)^{-1}$?
3. Why divide each building's daily profile by its mean before clustering? What do you lose?

**Today:** which bridges that are not poor today will be poor in four years?

::: notes
Give the answers before moving on; recall helps most with feedback.
1. Train on earlier inventories and validate on a later one, at the horizon of the use; keep
   each inspection once; drop the columns the target sets, such as the inspection interval
   from the record being predicted (Lecture 3). Today's split is exactly that.
2. Forming X^T X squares the condition number, so the inverse loses twice as many digits; QR
   and the SVD work on X itself (Lecture 5). Today, each Newton step is a weighted least
   squares, solved the same way.
3. To compare schedules, not sizes; you lose how much is at stake, so keep the mean as a
   number of its own (Lecture 4).
:::

## Catch bridges before they turn poor

- Not poor today; poor in four years?
- A miss costs $r=20$ false alarms.

![Three arrows on a timeline of inventory years. Train: 2014 to 2018, 19,748 bridges, 841 turned poor. Validate: 2016 to 2020, 19,777 bridges, 898. Test, once: 2020 to 2024, 19,637 bridges, 793.](cee_l06_linear_classifiers/timeline.png)

::: notes
The 20 is the lecture's working assumption; the owner sets it (the Decide step). The setting:
an engineer in 2021 has the inventories through 2020, so the last four-year outcome she can
see is 2016 to 2020. The label is the lowest rating in the inventory four years later, so a
bridge poor in 2022 and repaired by 2024 counts as not poor; about 1% of bridges leave the
inventory in four years (replaced or removed) and are dropped. They are the riskier ones: 150 of
the 19,787 bridges of 2020 left by 2024; 65% of them were rated 5, against 19% of the rest, and
the model gives them a mean p of 15%, about 23 expected poor beside the 793 counted. The test
costs leave them out (informative censoring).
The 16 inputs: the 12 inventory inputs of Lecture 3 (age, years since work, traffic, trucks,
length, width, material, location) and the deck, superstructure, and substructure ratings and
the lowest of them, all known when the decision is made. In Lecture 3 the ratings were the
target; today they come from the earlier record, so they are not leakage. About 1% of
bridges have no deck rating (no deck); for them we use the lower of the other two.
Inspectors rate the deck, superstructure, and substructure from 0 to 9; the lowest at 4 or below
is poor (Lecture 3). The decision: which bridges rated 5 or more get preventive work and a closer
look now. A false alarm: work on a bridge that would have held. A miss: a bridge turns poor:
repair, load posting, detours. Data: NBI, Illinois.
:::

## Your turn: which bridges do you flag?

| | Road, over, 2020 | Built | Vehicles a day | Deck, super, sub | Poor by 2024? |
|:--|:--------------|-----:|------:|:-----:|:------:|
| A | Duncan Rd, Phinney Branch | 1969 | 8,400 | 6, 6, 7 | ? |
| B | Duncan Rd, Copper Slough | 1976 | 8,000 | 5, 5, 6 | ? |
| C | FAU 7154, Copper Slough | 1988 | 7,000 | 6, 6, 7 | ? |
| D | Staley Rd, Phinney Branch | 2006 | 7,000 | 6, 6, 8 | ? |
| E | Mattis Ave, Phinney Branch | 1977 | 5,750 | 5, 5, 7 | ? |
| F | Broadway, Saline Branch | 1982 | 3,350 | 5, 5, 5 | ? |

::: notes
Six city bridges in Champaign-Urbana in 2020; the budget allows two flags. Which two?
How the six were chosen: the six busiest bridges in Champaign County owned by a city (NBI Item
22 = 04) whose lowest rating in 2020 was 5 or 6. All six are prestressed concrete; C was rebuilt
in 2003. Two minutes on your own: pick two. Then a second question: if a miss costs 20 false
alarms, how many would you flag, and which? Write a chance for each.
Answers: B (rated 2, critical, in 2024) and F (rated 4, poor). A, C, D, and E did not turn poor.
The model ranks F, B, E, A, C, D: its two flags are both right. At r = 20 it flags F, B, E, and
A. Keep the class's answers; the next slides trace B, and Derive returns to all six.
:::

## The decoder outputs a probability

![Diagram of the course pipeline with the decoder highlighted. A bridge x goes to an encoder that scales 16 inputs, giving z; the decoder computes sigma of w0 plus w transpose z, giving a probability p; a decision step flags the bridge if p is above 1 over 1 plus r, where a miss costs r false alarms. Above, 19,748 training bridges from 2014 to 2018 and a cross-entropy loss, minus log p if the bridge turned poor and minus log of 1 minus p if not, fix the weights by Newton's method. Below, bridge B in 2020, Duncan Road over Copper Slough, built 1976, 8,000 vehicles a day, deck 5, superstructure 5, substructure 6. Its lowest rating is 1.60 standard deviations below the mean, adding 1.67 to the score; superstructure adds 0.62; the deck adds 0.28; 13 more inputs add 0.12; the intercept is minus 4.21; the score is minus 1.52, so p is 0.18, above 1 over 21, 0.048: flag. In 2024 its rating was 2, poor.](cee_l06_linear_classifiers/classifier_pipeline.png)

::: notes
This is bridge B from your attempt. The rating terms dominate: a lowest rating of 5 is 1.6
standard deviations below the training mean. sigma(-1.52) = 0.18: about one in five bridges
like B is poor in the inventory four years later. B was rated 2, critical, in 2024.
Material is read with prestressed, the most common, as the baseline. The model's own coding
measures each material against the 14 bridges of "other" material, 1 of which turned poor, so
each material weight is uncertain (prestressed -0.51, standard error 0.55). Read alone, B's
prestressed column subtracts 0.49, and the other material columns add most of it back: its
material terms sum to -0.15 in either coding. The same model, the same p; only the attribution
moves. Read a categorical input's columns together.
:::

## Model the log-odds, not the label

**Assumption:** the log-odds of turning poor are a weighted sum of the inputs.

$$\ln\frac{p}{1-p}=w_0+\mathbf{w}^{\top}\mathbf{z}=s$$

$$p=\sigma(s)=\frac{1}{1+e^{-s}}$$

- A line on 0/1 labels leaves $(0,1)$; $\sigma$ does not.
- Lowest rating alone: −1.44 per point.

![Plot of the log-odds of being poor four years later against the lowest rating now, for the 2014 to 2018 training bridges. The share that turned poor is 18.2% at rating 5, 4.1% at 6, 1.3% at 7, and 0.4% at 8; on the log-odds scale the four points lie close to a dashed straight line falling 1.44 per rating point. None of the 746 bridges rated 9 turned poor.](cee_l06_linear_classifiers/log_odds.png)

::: notes
The odds p/(1-p) range over (0, infinity) and the log-odds over the whole line, the range of a
weighted sum. The fitted line uses the rating alone; the model uses all 16 inputs. None of the
746 bridges rated 9 turned poor, so rating 9 has no finite log-odds and is not plotted.
Forsyth §11.3.1 derives logistic regression as a generalized linear model.
:::

## The probability is an S-shaped sheet

![Two panels. Left: in 3D, the logistic probability of turning poor in four years over the lowest rating, 5 to 9, and age, 0 to 110 years: an S-shaped sheet rising to 0.14 (age 0) to 0.21 (age 110) at rating 5, near zero from rating 7 up, with the cost threshold p = 1/21 drawn on it in orange. Right: the same seen from above, with 1,500 training bridges by rating and age, those that turned poor in orange: the threshold line is almost vertical, near rating 6: 5.84 at age 0, 6.22 at age 110; age barely tilts it.](cee_l06_linear_classifiers/sheet.png)

::: notes
Two inputs, to see it: p = sigma(w0 + w1 rating + w2 age), fitted on the 2014 to 2018 bridges.
The weighted sum is a tilted plane; sigma bends it into a sheet between 0 and 1. The line where
the sheet crosses 1/21 is straight, because it is where the plane crosses a constant: a linear
classifier. Here the rating does almost all the work (-1.39 per point); age adds 0.005 per year.
:::

## Likelihood gives cross-entropy

Each label is a coin flip with chance $p_i$; maximize the likelihood, i.e. minimize

$$\mathcal{L}(\mathbf{w})=-\sum_{i}\big[y_i\ln p_i+(1-y_i)\ln(1-p_i)\big]$$

$$\nabla\mathcal{L}=\mathbf{A}^{\top}(\mathbf{p}-\mathbf{y}),\qquad\nabla^2\mathcal{L}=\mathbf{A}^{\top}\mathbf{S}\mathbf{A},\qquad\mathbf{S}=\mathrm{diag}\,p_i(1-p_i)$$

- $\mathbf{A}=[\mathbf{1}\ \mathbf{Z}]$. The Hessian is positive semidefinite: convex, one
  minimum, no closed form.
- **Newton:** $\mathbf{w}\leftarrow\mathbf{w}-(\mathbf{A}^{\top}\mathbf{S}\mathbf{A})^{-1}\mathbf{A}^{\top}(\mathbf{p}-\mathbf{y})$,
  a weighted least squares at each step (Lecture 5).

::: notes
Derivation: the likelihood is the product of p_i^y_i (1 - p_i)^(1 - y_i); its negative log is
the cross-entropy. With p = sigma(s), dp/ds = p(1 - p), so the derivative of one term with
respect to w is (p_i - y_i) a_i, where a_i is row i of A: the gradient is A^T(p - y). Its
derivative is the sum of p_i(1 - p_i) a_i a_i^T, a sum of positive semidefinite matrices.
Newton's step solves (A^T S A) delta = A^T (p - y): iteratively reweighted least squares, solved
with a factorization, never an inverse. If a hyperplane separates the classes perfectly, the
likelihood has no maximum and the weights grow without bound; a ridge penalty fixes that
(Lecture 5).
:::

## Newton walks down one bowl

![Two panels. Left: in 3D, the mean cross-entropy over the intercept and the slope of the one-input model, lowest rating standardized, a single smooth bowl; Newton's path from zero, in orange, drops to its bottom in a few steps. Right: on a log scale, the loss above the minimum after each step falls from 0.08 to 0.02, 0.005, 0.0005, 0.00001, then 5 billionths: close to the bottom, each step roughly squares the error.](cee_l06_linear_classifiers/newton.png)

::: notes
The Hessian A^T S A is positive semidefinite, so the loss is convex: one bowl, no valleys to
get stuck in (compare Lecture 14's networks). There is no closed form, but Newton's method fits
a quadratic bowl at each point and jumps to its bottom; near the minimum the error squares at
every step. Nine steps to 1e-10 for this model; nine for the full 16-input model too.
:::

## Your turn: where to draw the line

A false alarm costs 1. A miss costs $r$. Correct decisions cost 0. The model says $p$.

1. Write the expected cost of flagging a bridge, and of letting it wait.
2. Find the threshold on $p$ above which you flag. Evaluate it at $r=20$.
3. The model's $p$ for the six bridges: F 0.20, B 0.18, E 0.13, A 0.049, C 0.044,
   D 0.038. Which do you flag?

::: notes
Three minutes, in pairs. The next slide answers. Ask who flagged more than two bridges in the
attempt, and why.
:::

## Costs set the threshold

Flag when $p\,r>1-p\iff p>\dfrac{1}{1+r}$. At $r=20$: **0.048**, not 0.5.

![Two lines against the model's p from 0 to 0.15: the expected cost of flagging, 1 minus p, nearly flat near 1; the expected cost of waiting, 20 p, rising steeply from 0. They cross at p = 1/21 = 0.048; to the right, shaded, flagging is cheaper.](cee_l06_linear_classifiers/cost_lines.png)

::: notes
Costs are counted relative to the correct decision, so only the two errors carry a cost. A
fixed budget, flag the top k, gives the same rule with an implied r. With a budget of two, the
model picks F and B, both right; compare with the class's picks. A at 0.049 is on the line: at
r = 19 it would wait. That is why r must be set before the list is made, not after.
Flag: expected cost (1 - p) times 1. Wait: p times r. Elkan (2001). The six: flag F, B, E, and A
(0.049, on the line). B and F turned poor. The rule needs calibrated p: of bridges given 0.05,
about 5 in 100 must turn poor.
:::

## The SVM keeps a margin, not odds

Squared hinge: zero beyond the margin. Same ranking (AUC 0.815), but $s>0$ for no bridge.

![Three losses against y times the score: the 0-1 loss, a step from 1 to 0 at zero; the logistic loss, smooth and never zero; the squared hinge, falling to exactly zero at the margin, y s = 1, and steep for wrong-side points.](cee_l06_linear_classifiers/losses.png)

::: notes
Forsyth §2.1 derives the SVM from the hinge, max(0, 1 - y s) (Cortes & Vapnik, 1995). We fit
its square, the L2-SVM: scikit-learn's LinearSVC with loss="squared_hinge" (its default), C = 1,
on the same 16 standardized inputs. LinearSVC (liblinear) also penalizes the intercept, as the
weight of a constant input. The plain hinge fails on these bridges: its minimum puts every
bridge at s = -1 with w near 0, so nothing is left to rank (notebook, Gotchas). The bridges
with 1 - y_i s_i > 0 are the support vectors. Why no positive score: with 4% positives and equal
costs in the loss, the boundary that minimizes the training loss flags nobody, even on the
training bridges; weight the classes equally (class_weight="balanced") and it flags 6,221 of
them. AUC is the chance that a random bridge that turned poor scores above a random one that
did not; it measures ranking only. Logistic regression's cost at 1/21 is 0.45 (Break it).
Objective: lambda/2 ||w||^2 + (1/n) sum max(0, 1 - y_i s_i)^2, labels plus or minus 1. Only
bridges near the boundary shape it; margin width 2/||w||. Scores are not probabilities. Platt
scaling fits sigma(a s + b) on validation data (Platt, 1999): then its cost at r = 20 is 0.44.
:::

## Ratings are ordered: cut once

$$P(y\le j\mid\mathbf{z})=\sigma(\theta_j-\mathbf{w}^{\top}\mathbf{z})$$

One weight vector, five cutpoints: every later rating teaches it.

![Five S-shaped curves of the probability that the later rating is at most j, for j = 4 (poor), 5, 6, 7, and 8, against the score w times z: the same curve shifted left or right by its cutpoint, the poor curve furthest left.](cee_l06_linear_classifiers/cutpoints.png)

::: notes
The proportional-odds, or cumulative-logit, model. The chance of each outcome is a difference
of two adjacent cumulative probabilities. Fit by maximum likelihood; the notebook writes it
from scratch and checks it against statsmodels' OrderedModel. Its P(y <= 4) ranks bridges for
today's decision. Break it tests the assumption.
Six later outcomes: poor (4 or less), 5, 6, 7, 8, 9 (McCullagh, 1980). It learns from every later
rating, not only poor or not: 841 bridges turned poor, but all 19,748 have a later rating.
Assumption: what moves a bridge from 8 to 7 also moves it from 5 to 4.
:::

## Your turn: predict the check

```python
def fit_logistic(Z, y, tol=1e-10, max_steps=50):
    A = np.column_stack([np.ones(len(Z)), Z])   # intercept column
    w = np.zeros(A.shape[1])
    for step in range(1, max_steps + 1):
        p = expit(A @ w)                         # sigma(s), s = A w
        g = A.T @ (p - y)                        # gradient of the NLL
        H = A.T @ (A * (p * (1 - p))[:, None])   # Hessian A^T S A
        delta = np.linalg.solve(H, g)            # Newton step
        w -= delta
        if np.abs(delta).max() < tol:
            return w, step
    raise RuntimeError("Newton did not converge")
```

**Predict:** how many steps? Pass against scikit-learn's default solver?

::: notes
Ask for the number of steps from w = 0, and for pass or fail at rtol = 1e-6. Take a show of
hands on each. The reference: LogisticRegression(C=np.inf), no penalty, with
its default solver, L-BFGS; and the same with solver="newton-cholesky". Most students expect
dozens of steps and a pass.
:::

## The rule is the bar to clear

9 Newton steps match scikit-learn to $10^{-12}$. All bridges: model **0.43**, rule 0.45; twins 0.52, 0.61, floor 0.37.

![Dot plot of cost per bridge at r = 20 on the 1,546 twin records of the 2016 to 2020 validation period: flag all 0.95; flag none 1.07; the rule, flag every bridge rated 5, 0.61; the model, flagging p above 1 over 21, 0.52; flagging a twin when its twin turned poor, a rough floor, 0.37.](cee_l06_linear_classifiers/bracket.png)

::: notes
The default solver, L-BFGS with tolerance 1e-4, stops after 19 iterations: the losses differ
by 1e-5, but one weight differs by 0.4 and one probability by 0.13. The 14 bridges of "other"
material make a flat direction in the loss, and L-BFGS stopped on it. With tolerance 1e-8 it
agrees to 4e-6. A check needs a reference whose tolerance you know.
A ridge penalty: the best C improves the validation log loss by 5e-5; with 19,748 bridges and
16 inputs the unpenalized fit is enough.
The rule is the engineer's baseline: it uses the same inspection. The floor: 1,546 twin records
in 2016 (parallel bridges, same year, side by side); flag a twin if its twin turned poor; 96.4%
of twins agree. Rough, because twins are not typical bridges (5.4% turned poor, against 4.5%),
and the cost per bridge grows with that share, so the figure costs everything on the same twin
records: flag none 1.07, the rule 0.61, the model 0.52, the floor 0.37. On all 19,777 bridges:
the model 0.43, the rule 0.45, flag none 0.91. Validation flags 28% and catches 80%; the rule
flags 16% and catches 65%. Validation shares 2016 to 2018 and the same bridges with training (an
engineer in 2021 has no later four-year outcome). Its AUC, 0.847, is above the test period's,
0.815, but checked in hindsight the overlap is not why: 2018 to 2022 ranks like validation
(0.843), and the rating alone drops the same way (0.823, 0.824, then 0.796 in 2020 to 2024).
Its default solver fails: 0.4 off. Validation, 2016 to 2020, at r = 20, on all bridges: model
0.43, rule 0.45. On the twin records: flag none 1.07, rule 0.61, model 0.52, floor 0.37 (rough).
:::

## Predict before we test

1. **0.5.** Flag when $p>0.5$. How many of the 793 bridges that turned poor by 2024 does it
   catch? What is its accuracy?
2. **Costs.** Flag when $p>1/21$. Validation promised 0.43 against the rule's 0.45. In 2024?
3. **Ordinal.** The model that sees every later rating: will it catch more of the poor?
4. **2024.** The model learned 2014 → 2018. Will its probabilities hold for 2020 → 2024?

::: notes
One minute; students commit in writing. Each of the next four slides reveals one answer, and
each tests a piece of the method: the default threshold (1), the cost threshold and the rule
(2), the ordinal assumption (3), and calibration over time, which the cost threshold needs (4).
:::

## 0.5 flags no bridge

Highest $p$ in 2020: **0.497**. Flags nobody; accuracy 95.96%, same as flagging nobody.

![Histogram on a log scale of the model's p for the 2020 bridges, those that stayed above poor in grey and those that turned poor by 2024 in orange: almost all below 0.4, the highest at 0.497; a dashed line at 0.5 has no bridge to its right, while the solid line at 1/21 cuts through the mass.](cee_l06_linear_classifiers/p_hist.png)

::: notes
Accuracy is the cost with r = 1, and 0.5 is the r = 1 threshold. Many libraries' predict()
uses 0.5. For a rare event, report what you catch and what you flag, or the cost; never
accuracy alone.
p > 0.5 catches none of the 793 that turned poor. 4.0% turned poor. Accuracy counts a miss as one
false alarm, r = 1. Here r = 20.
:::

## Costs, not 0.5, set the threshold

$p>1/21$ catches **80%** (631 of 793), cost 0.45. "Flag every 5": **0.44**. A tie.

![Curve of cost per bridge at r = 20 for the 2020 bridges judged in 2024, against the threshold on p from 0.005 to 0.7 on a log axis. The cost falls from 0.76 to a minimum of about 0.44 near 0.07, then rises to 0.81, the cost of flagging none, above 0.3. A square marks 1 over 21, 0.048, at 0.45; a diamond marks 0.5 at 0.81. Horizontal lines: flag all 0.96, flag none 0.81, and the rule, flag every bridge rated 5, 0.44, dashed.](cee_l06_linear_classifiers/cost_threshold.png)

::: notes
This is the lecture's decisive figure. The bottom of the curve is wide: any threshold from 0.045
to 0.12 costs within about 0.02 of the best. A tie: the model costs 0.011 per bridge more than
the rule; resampling whole counties, the 95% interval runs from -0.019 to +0.042. The default 0.5 sits on the flag-none line. The
best threshold in hindsight, 0.074, costs 0.44, the same as the rule: at r = 20 the lowest
rating does most of the work, and the other 15 inputs add little. Ranked by the lowest rating
alone, the 2020 bridges reach an AUC of 0.796; the 16-input model, 0.815. A model must beat the engineer's
rule, not only the coin flip.
p > 1/21 flags 31.5% and misses 162 of the 793. At 0.5 the cost is 0.81. AUC from the rating
alone 0.796, the model 0.815. Validation promised 0.43 against 0.45; the next two slides say
why it shrank.
:::

## One weight vector misfits every cut

Age matters for "7 or less" (0.26), not for poor (0.01). One shared weight fits neither.

![Two panels of weights toward a lower later rating, fitted separately at each cut, poor or 4 or less, 5 or less, 6 or less, 7 or less, 8 or less, with the ordinal model's single weight as a dashed line. Age: 0.01, 0.04, 0.09, 0.26, and minus 0.27, against one weight of 0.10. Superstructure rating: minus 0.31, minus 0.21, minus 0.44, plus 0.33, and minus 0.27, against one weight of minus 0.22.](cee_l06_linear_classifiers/ordinal_weights.png)

::: notes
Both models use the same 16 inputs and the 2014 to 2018 bridges. AUC: 0.811 ordinal, 0.815
binary. Costs at r = 20: 0.44 ordinal, 0.45 binary. The ordinal model is a good idea when the
assumption holds; here, what moves a bridge from 8 to 7 (age, size) is not what moves it from
5 to 4 (the condition of the superstructure). Separate fits, one per cut, are the check.
At a 10% budget the ordinal model catches 39% of the poor; the binary model, 43%. At r = 20 their
costs differ by 0.01. More labels did not help: above 5, later ratings answer other questions.
:::

## 2024 probabilities run high

Predicted 5.0% would turn poor by 2024; **4.0%** did. Nothing seen by 2021 showed it.

![Reliability plot: observed percentage of bridges turning poor against the predicted percentage, in ten groups of predicted risk, for 2016 to 2020 (validation) and 2020 to 2024 (test). The validation points lie on the diagonal, up to 20% predicted and 22% observed. The test points fall below it: 21% predicted, 17% observed in the riskiest tenth.](cee_l06_linear_classifiers/calibration.png)

::: notes
The ranking held (AUC 0.847 in validation, 0.815 in test), but the level moved: fewer bridges
turned poor from 2020 to 2024 than the 2014 to 2018 rates predicted. The cost threshold assumes
calibrated probabilities, so it flags too many. Could the 2021 engineer have fixed it? Not with
2016 to 2020, the latest period she had (checked in hindsight): Platt scaling on it still
predicts 5.06% for 2024 (cost 0.447), and a refit on it 5.07% (0.445). The slide's Newton
listing does not even converge on that period: none of its 13 "other"-material bridges turned
poor, so the likelihood has no maximum (a ridge penalty fixes that). The drift shows only after
2020: 2018 to 2022, not known in 2021, already ran high (4.79% predicted, 4.09% observed).
Recheck calibration as each new period arrives. Checking calibration: Lecture 10; coverage
guarantees under shift, Lecture 11.
2016 to 2020: predicted 4.5%, observed 4.5%. 2020 to 2024, the riskiest tenth: predicted 21%,
observed 17%. So
p > 1/21 flags too many: 0.45, against 0.44 at the best threshold in hindsight.
:::

## Flag above 1/(1+r); the owner sets r

$r=10$: catch 64%. $r=20$: 80%. $r=50$: 89%. One dial for every $r$.

![Curve of the percentage of bridges that turned poor by 2024 caught against the percentage flagged. It rises steeply to 64% caught at 18% flagged, the point for r = 10, then to 80% at 31.5% for r = 20 and 89% at 49.5% for r = 50; it reaches 99.5% at 90% flagged, and 100% only at 99.9%. The rule of flagging every bridge rated 5 sits on the curve at 19% flagged and 67% caught, next to r = 10; flagging every bridge rated 5 or 6 sits at 49% and 89%, next to r = 50.](cee_l06_linear_classifiers/dial.png)

::: notes
Costs in 2024: r = 10, the model 0.300 and "every 5" 0.302; r = 50, the model 0.68 and "every 5"
0.84 (the model's lead, 0.164 per bridge, holds when whole counties are resampled: 0.071 to
0.265), while "every 5 or 6" catches 89% for 49% flagged, like the model. At the rule's own
budget (19.4% flagged) the model catches 66.3%, the rule 66.7%. So: the rating does most of the
work. The model earns its place where the right budget falls between the rules, and when someone
must defend r with numbers.
Flagged: 18%, 31.5%, 49.5%. The rules sit on the curve: "every 5" is about r = 10; "every 5 or 6",
about r = 50. Recheck calibration as each period arrives.
:::

## Rank to act, inspect to know

| Do not use it | Because |
|:------------------------------|:------------------------------------|
| instead of inspection | federal rules require routine inspections, every 24 months by default (23 CFR 650.311) |
| with 0.5 or accuracy | for rare events, price the miss |
| where the rule does as well | at r = 20 in 2024, "flag every 5" needs no model |
| without rechecking calibration | each new period, or bridges unlike these: culverts, new materials, other states |
| for one bridge's fate | p = 0.18 means about one in five bridges like it |

::: notes
When not to use the method is part of the method. The 2022 National Bridge Inspection Standards
set a regular interval of at most 24 months; a lowest component rating of 3 or less requires 12
months at most, and Method 1 allows up to 48 months only when every component is rated 6 or more
and other criteria hold (23 CFR 650.311(a)). A model can set priorities inside those rules; it
cannot replace them.
:::

## Who carries a missed bridge?

- Township bridges: 48% of the inventory, **86 of 162** misses.
- The owner sets $r$, in the open.

![Horizontal bars of the share of bridges that turned poor by 2024 that the r = 20 threshold caught, by owner: state 84%, 33 missed; county 82%, 26 missed; township 75%, 86 missed; city 87%, 13 missed; other owners 56%, 4 missed.](cee_l06_linear_classifiers/owner.png)

::: notes
One threshold allocates attention unevenly. By owner (Item 22), at r = 20 in 2024: state
bridges, 46% flagged, 84% caught (33 missed of 203); county, 27%, 82% (26 of 145); township,
23%, 75% (86 of 339); city, 40%, 87% (13 of 97); the 544 bridges of other owners (toll
authority, parks, railroads, federal and other agencies), 35%, 56% (4 of 9). 162 missed in all.
State bridges get a higher p than township bridges (mean 6.9% against 3.9%): their scores are
higher by 0.93, mostly through the ratings (+0.70; 26% of state bridges are rated 5, 16% of
township bridges) and material (+0.19); traffic and size add only 0.02. The model runs high for
state bridges (6.9% predicted, 3.7% turned poor) and is close for townships (3.9% against
3.6%). The gap in the share caught, 84% against 75%, is 9.1 points; resampling whole counties,
its 95% interval runs from -3.2 to +19.7: within county-to-county variation. The misses, 86 of
162, still fall on township owners.
Ask: who would notice the 86, and who would answer for them?
r prices risk to the people who cross the bridge: the owner sets it, not the modeler. Report misses
by owner.
:::

## Vote: what do you recommend?

The county will act on your list. A miss costs 20 false alarms.

- **A.** Flag $p>0.5$: the highest accuracy.
- **B.** Flag every bridge rated 5: no model.
- **C.** Flag $p>1/21$: the cost threshold.
- **D.** Flag every bridge.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
2024 costs: B 0.44, C 0.45, A 0.81, D 0.96. B is as good here and anyone can audit it; C adapts
to r and states a probability per bridge, but its calibration needs rechecking each period; A is the accuracy trap; D
spends everything. Then ask what would change the answer: r = 50 (B costs 0.84, C 0.68), a cap
on the budget, or a decade with drift.
:::

## Logistic regression runs modern AI

| Where | Today's model, inside | Lecture |
|:----------------|:------------------------------------------|:----------|
| Last layer | softmax regression on cross-entropy; a language model's next token | 17, 19 |
| CLIP | cross-entropy over image–text pairs in a batch (Radford et al., 2021) | 18 |
| Reward models | P(A preferred to B) = σ(r_A − r_B) (Ouyang et al., 2022) | 26 |
| Calibration | large networks are overconfident; temperature scaling (Guo et al., 2017) | — |

::: notes
Softmax regression is logistic regression with more than two classes (Forsyth §11.3.2). A
language model's last layer scores every token of its vocabulary, and training minimizes the
cross-entropy of the next token: today's loss, with tens of thousands of classes. The reward
model in InstructGPT is trained on pairs of answers ranked by people, with the logistic
(Bradley-Terry) loss. Temperature scaling divides the scores by one learned number before the
softmax, fitted on validation data: Platt scaling with one parameter. The point: the decoder
and loss you derived today are inside the largest models, and today's threshold question
(what does a probability cost?) comes with them.
:::

## Exit question

The owner raises $r$ from 20 to 50. What changes: the threshold, the list, and the rule the
model matches? What would you check before using the model in 2026?

**In the notebook** (`cee_l06_linear_classifiers.ipynb`): Newton's method checked against
scikit-learn, the cost threshold, the SVM and Platt scaling, the ordinal model, and the
calibration check.

::: notes
Answer: the threshold falls to 1/51 = 0.020; the list grows from 31.5% to 49.5% of bridges and
catches 89% instead of 80%; it matches "flag every 5 or 6". Before 2026: check calibration on
the latest four-year period, 2020 to 2024, and recalibrate or refit. Collect the answers on
paper or in a poll; they show what to revisit in the next warm-up.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- The log-odds are a plane; $\sigma$ bends it into a sheet.
- Cross-entropy is one bowl: Newton walks it.
- Flag above $1/(1+r)$, not 0.5.
- Beat the engineer's rule; recheck calibration.

**Next:** naive Bayes, mixtures, and EM: free-flow and congested traffic.
:::
::: {.column}
![Four thumbnails of the lecture's key figures: the logistic sheet over rating and age; the costs of flagging and waiting crossing at 1/21; the 2020 bridges' p, none above 0.5; the reliability plot of 2024 probabilities running high.](cee_l06_linear_classifiers/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- 23 CFR 650.311 (2022). Inspection interval.
- Cortes & Vapnik (1995). *Machine Learning* 20.
- Elkan (2001). *IJCAI*.
- FHWA (1995). *Recording and Coding Guide*.
- Forsyth (2019). Springer, §2.1, §11.3.
- Guo et al. (2017). *ICML*.
:::
::: {.column}
- McCullagh (1980). *JRSS B* 42(2).
- Ouyang et al. (2022). *NeurIPS* 35.
- Platt (1999). *Advances in Large Margin Classifiers*. MIT Press.
- Radford et al. (2021). *ICML*.
:::
::::::
