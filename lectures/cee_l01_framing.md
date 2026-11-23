---
title: When Predictions Carry Loads
subtitle: "Lecture 1 · What would you need before signing a model's output?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions to guess

On your own, two minutes. Write a guess; nobody expects the answers yet.

1. A model's error on its test set is 8.1 MPa. Would you sign off on a mix design that rests on
   its prediction? What else would you ask?
2. You try 255 sets of inputs and keep the one with the best test score. As an estimate of its
   error on new mixes, is that score too low, too high, or about right?
3. Do students learn more in lectures where they solve problems or in lectures where they
   listen? In which do they *feel* they learn more?

**Today:** what you need before signing a model's output.

::: notes
These are prequestions: asked before the lesson, answered during it. Guessing first helps
learning even when most first guesses are wrong, provided the correct answers follow (Pan &
Carpenter, 2023). Do not answer now; collect a few guesses and keep them.
Question 1 is answered on "Before you sign: four questions", question 2 on "The best of many
scores is too good" and "The winner's score is partly luck", question 3 on "Every lecture runs
the same steps". From Lecture 2 on, the warm-up is recall of earlier lectures instead.
:::

## A model's error sets the margin

:::::: {.columns}
::: {.column}
- Approve a mix design from its **predicted** strength, before any cylinder?
- Specified $f^{\prime}_c=35$ MPa at 28 days.
- 1,030 tests of 428 mixes (Yeh, 1998).
:::
::: {.column}
![Histogram of measured 28-day strengths for 330 held-out predictions between 36 and 40 MPa, pooled over 30 splits that hold out whole designs. A vertical line marks f'c = 35 MPa; the 37% of bars below it, hatched, are weak mixes; the rest, up to 72 MPa, are strong.](cee_l01_framing/margin.png)
:::
::::::

::: notes
Frame every model by its input, its output, and the decision it feeds, then by what each kind
of error costs, before looking at any accuracy number. The decision here is the producer's;
the engineer who signs needs to know how far to trust the number. 425 of the 1,030 tests are at
28 days; strengths run from 2.3 to 82.6 MPa. One mix is often tested at several ages, and some
designs were entered twice with rounded proportions: 315 designs in all. Lecture 2 returns to
why that matters.
The picture: every held-out 28-day prediction between 36 and 40 MPa, near the traced design's
38.1 (330 predictions of 111 tests, pooled over the 30 design-held-out splits). 37% of those
predictions measured below 35 MPa. A prediction above f'c is not a pass: the error decides how far above it must
be. Too small a margin approves weak concrete, and the owner and the public carry the risk;
too large spends cement and money for nothing.
:::

## Your turn: which error would you quote?

A model predicts a new design at 38.1 MPa; $f^{\prime}_c=35$ MPa. Four teams measured its
error.

Alone, two minutes: rank A–D, most trustworthy first. Would you approve the design?

| | Error measured on | RMSE (MPa) |
|:--|:----------------------------------------------------------|----------------:|
| A | the tests it was fit to | 7.0 |
| B | random tests held out | 8.9 |
| C | the best of 255 input sets, on the test set that picked it | 8.1 |
| D | whole designs held out, looked at once | 9.5 |

::: notes
The model averages the strengths of the five most similar tested mixes (Lecture 2 builds it);
RMSE is the root-mean-square error. All four numbers are means over 30 splits: A, C and D over
Lecture 2's 30 splits that hold out a quarter of the designs; B over its 30 random splits.
Collect rankings. Most students trust A, the smallest number. Do not settle it yet.
Answers, kept for later:
D is the honest number for a new design: designs the model never saw, looked at once.
A is the training error: each test is one of its own five neighbors.
B mixes the same mix, tested at other ages, into training: on average over the 30 random
splits, 74% of test rows have their own mix in training (Break it).
C is the best of 255 scores on one test set, partly luck. Chosen on validation and tested once,
the same search scores 8.7 (Build and verify).
Approve? 38.1 is above 35 with no margin; after subtracting any of the four errors, it is not.
The next slides trace this design.
:::

## Engineers write rules; ML learns them

![Diagram in two rows about concrete strength. Classical programming, where an engineer writes the rule: the rule f_c = A / B^(w/c) and a new mix go into a program, which outputs the predicted strength; a note says Abrams' law: strength falls as the water-cement ratio w/c rises. Machine learning, where the rule is learned from examples: tested mixes x_i and their measured strengths y_i go into a learning algorithm, which outputs a learned rule f-hat; a note says 1,030 tests of 428 mixes (Yeh, 1998), judged on mixes it has not seen.](cee_l01_framing/ml_vs_programming.png)

::: notes
The two-row contrast follows Chollet (2017). In Abrams' law the engineer chose the form of the
rule; A and B are fitted to a cement and a test method. In machine learning the data choose the
rule from a whole family of candidates, and the learned rule is then used exactly like the
program in the top row. The last words of the note are the whole course: a rule that only
reproduces the tested mixes is useless; it is judged on mixes it has not seen.
:::

## Encoder, decoder, loss: every model

![Diagram of the course pipeline. The input x goes to an encoder z = f_phi(x), then to a decoder y-hat = g_theta(z); together they form the model. A loss L(y-hat, y) compares the prediction with the target y. Dashed arrows show that training adjusts phi and theta to lower the loss on 746 training tests. Below, one held-out design tested at 28 days: cement 237, slag 92, fly ash 71, water 247, superplasticizer 6, coarse aggregate 853, fine aggregate 695 kilograms per cubic meter. Standardized, its water is +3.13 and its cement -0.41 standard deviations from the training means. Its 5 nearest training tests reached 35.2, 35.2, 37.2, 37.2, and 45.9 MPa, mean 38.1. The prediction is 38.1 MPa; the measured strength is 28.6 MPa; the squared error is 90.3; against f'c = 35 MPa this is a wrong yes.](cee_l01_framing/encoder_decoder.png)

::: notes
This is the design from your attempt. Encoder: standardize each input with the training means
and spreads (phi). Decoder: average the strengths of the 5 nearest training tests (theta is
the stored tests). Loss: squared error. Measured 28.6 MPa: a wrong "yes" of 9.5 MPa, exactly
the size Team D reported. Its water is 3.13 standard deviations above the training mean
(w/b 0.62), but its neighbors have w/b 0.46 to 0.60: Lecture 2 shows why raw inputs choose
them. Every later lecture reuses this picture with its own step highlighted. How the design
was chosen: among split 0's held-out 28-day tests (designs not used as examples in later
lectures), the wrong "yes" whose error is closest to 9.9 MPa, the held-out error on 28-day
tests. Its design was entered twice, both times in the test set.
:::

## Three datasets open the course

![Three panels. Concrete: strength against age, 1 to 365 days, for 1,030 tests; tests below the dashed f'c = 35 MPa line in orange, above in blue. Illinois bridges: a map of 21,966 bridges in 2024, 2,308 rated poor in orange. Buildings: the typical weekday load, as a multiple of the daily mean, for 40 buildings in grey and the median in blue, low at night and high from 9 to 17.](cee_l01_framing/datasets.png)

::: notes
Frame each one: input, output, decision; ask what an error costs and who carries it. All
three are public and already in the course data folder. The bridge inventory covers 102
counties, about 22,000 bridges a year (21,935 to 22,015), every two years. The buildings are
at 19 sites; Lecture 4 keeps 1,159 buildings at 15 sites with complete hourly data. Later
units add, among others: loop detectors on Portland's I-205, pavement performance (FHWA
LTPP), a water network with leaks (BattLeDIM), USGS stream gauges in Illinois, crack images
(SDNET2018, 3DCrack), flooded buildings seen from satellites (SpaceNet 8), a laser scan of a
building (CRAS lab), OSHA injury reports, and the traffic-control manual (FHWA MUTCD).
Sources: concrete, 1,030 tests of 428 mixes (Yeh, 1998), Lectures 2, 5, 10; Illinois bridges,
131,851 records of 23,464 bridges, 2014-2024 (FHWA NBI), Lectures 3, 6; buildings, 1,578
metered hourly, 2016-2017 (Miller et al., 2020), Lecture 4. On the map, poor means the lowest
of deck, superstructure, and substructure ratings is 4 or less.
:::

## Held-out data must look like the future

![Thirty-six concrete designs, one column each, with a dot for each test at its age, 1 to 365 days. Every design's tests share one role: whole columns are training (blue circles), validation (green triangles), or test (orange squares).](cee_l01_framing/split_designs.png)

::: notes
State the assumption before the method; the Break-it step tests it. Team A's 7.0 flatters
because each training test is one of its own five neighbors. Derivation: if each
held-out case is drawn like a future case and played no part in fitting or choosing, each
squared error has expectation L, so by linearity E L-hat = L (Forsyth, Section 3.1). The RMSE
is the square root of L-hat; it is what we quote, in MPa. Validation can be a separate set or
cross-validation: folds of the training designs, each held out in turn. Every word of the
assumption is tested today: "drawn like" (designs, not tests; no new materials) and "never
used to choose" (the test set, once).
The picture is split 0: its test designs (orange), and inside the training designs the first of five
validation folds (green). Each design's tests go to one side together: the next mix the
producer designs is new, so the held-out cases must be new designs.
Assumption, in symbols: if each held-out case is drawn like a future case and was never used to
fit or choose, $\mathbb{E}\,\hat{L}=L$.
:::

## The best of many scores is too good

![Three panels of validation RMSE for equally good models whose true error is 9.5 MPa, each score drawn with spread 0.9 MPa. Best of 2: the expected best score is 0.5 below the truth. Best of 16: 1.5 below. Best of 255: 2.5 below; the chosen score, ringed, sits at the bottom of the cloud.](cee_l01_framing/winners_curse.png)

::: notes
Two models: min(a, b) = (a + b)/2 - |a - b|/2. For independent normal scores with mean L and
standard deviation sigma, a - b has standard deviation sigma times root 2, and E|a - b| = 2 sigma /
root pi, so E min = L - sigma / root pi. sigma = 0.9 MPa is the spread of Team D's score over
the 30 splits. Two models scored on one test set are not independent: they err on the same hard
cases, so their scores are positively correlated (rho), a - b has standard deviation sigma times
root 2(1 - rho), and the optimism is sigma times root (1 - rho) / root pi. So 0.5 MPa is an upper
bound. "Unbiased" holds for L-hat, the mean squared error; its square root, the RMSE we quote,
is slightly low on average (Jensen's inequality). Cawley and Talbot (2010) show that this
optimism can be as large as the differences between learning algorithms. Answer to warm-up
question 2: too low.
The picture assumes equally good models with independent scores: the expected best of M normal
scores is an order statistic (best of 2: 9.02, of 16: 7.96, of 255: 7.04 MPa, against a true
9.51). The real gap, four slides on, is smaller: 1.0 MPa at 255 input sets. The picture's
assumptions do not hold there: real input sets differ in true error, their scores on one test
set are correlated, and 0.9 MPa is the spread over splits, where the training data change too,
not the noise of one test set.
In symbols: $\mathbb{E}\,\min_m\hat{L}_m\le\min_m L_m$. The more models compared on the same
data, the luckier the winner looks.
:::

## Code: validation chooses the inputs

```python
def val_rmse(X: np.ndarray, y: np.ndarray, d: np.ndarray,
             k: int = 5, folds: int = 5) -> float:
    """RMSE over folds of whole designs; scaler fit inside."""
    pred = np.empty(len(y))
    for fit, val in GroupKFold(folds).split(X, y, groups=d):
        sc = StandardScaler().fit(X[fit])        # fit rows only
        knn = KNeighborsRegressor(k).fit(sc.transform(X[fit]), y[fit])
        pred[val] = knn.predict(sc.transform(X[val]))
    return float(np.sqrt(np.mean((pred - y) ** 2)))

scores = [val_rmse(X[:, c][tr], y[tr], d[tr]) for c in subsets]
best = subsets[int(np.argmin(scores))]   # validation chooses
```

**Predict:** (1) Will `val_rmse` match scikit-learn's `cross_val_predict` on all 255 input
sets? (2) Over 30 splits, will the chosen set's test score be higher or lower than its
validation score?

::: notes
d holds each test's design id, so a design is never split between fitting and validating; the
scaler is part of the model, so it is fit inside each fold. tr is the 75% of designs used for
training; the test designs are not touched until the end. Take a show of hands on each
prediction. The next slide answers both.
:::

## Validation flatters a little; test once

- `val_rmse` matches `cross_val_predict` on all 255 sets, within $10^{-12}$ MPa.
- Chosen on validation: 8.5. **Tested once: 8.7.**

![Dot plot of RMSE in MPa with designs held out. All 8 inputs: 9.5. The input set chosen on validation: validation score 8.5, a hollow square; test score, once, 8.7. A dotted line marks 17.0 for predicting the mean; a dashed line marks 1.4 for the scatter of repeat tests, labelled rough.](cee_l01_framing/bracket.png)

::: notes
The check against an independent implementation, at a stated tolerance, is what lets us trust
the loop. Validation over five folds of all training designs is barely optimistic (0.2 MPa on
average), because it compares the 255 sets on hundreds of tests. The set chosen most often:
cement, slag, water, SP, and age (11 of 30 splits); dropping fly ash and the aggregates helps
this plain model. Bracket: predicting the training mean scores 17.0 MPa; the same mix at the
same age, tested more than once, scatters with a median standard deviation of 1.4 MPa
(Lecture 2): a rough floor, an indication rather than a measurement, because it rests on 9
groups (20 tests) from only 3 mixes.
No design is on both sides of any check. The test score is higher than the validation score in
19 of 30 splits.
:::

## Predict before we test

1. **Many models.** Half the test designs pick the best of 255 input sets; the other half
   checks it. How far apart are the two scores?
2. **Random split.** Hold out random tests instead of whole designs. Higher or lower error?
   Why?
3. **New material.** Fit the model to mixes without fly ash, then predict mixes with fly ash.
   Does validation's promise hold?

::: notes
Give one minute; students commit in writing. Each of the next three slides reveals one answer.
Ask who predicted it, and what the others missed. Each tests one word of the assumption: never
used to choose (1); drawn like the future, independently (2 and 3).
:::

## The winner's score is partly luck

Half the test designs pick; the other half checks. Team C's 8.1 was luck: honestly, 8.7.

![Line plot of the winner's RMSE against the number of input sets tried, M, from 1 to 255 on a log axis. On the half of the test designs that picked it, the winner's score falls from 12.95 to 7.87. On the other half, fresh, it falls from 13.00 to 8.84. At M = 255 an arrow marks the gap: 1.0 MPa too good.](cee_l01_framing/selection_curve.png)

::: notes
This is the lecture's decisive figure. Each point averages 30 splits, both halves as the picker,
and 20 random orders of the 255 input sets. With one model the two halves agree: a score used
once is honest. Every extra model is another lottery ticket for the picking half, and the gap
grows to 1.0 MPa. Both curves fall because trying more sets does find better models; only the
fresh half says how much better. Hence one test set, looked at once, after every choice.
A half holds fewer designs than the whole test set, so its scores are noisier and its winner
luckier: on the whole test set, Team C's 8.1 sits 0.6 MPa below the 8.7 that the set chosen on
validation scores, less than the halves' 1.0.
:::

## Random splits reuse mixes: 8.9, not 9.5

![The same 36 designs, with random tests held out instead: every column now mixes blue training circles and orange test squares, and every test square is ringed because its design also has tests in training.](cee_l01_framing/split_random.png)

::: notes
The rows are tests; the independent units are designs. A random split puts the same mix at 7,
28, and 90 days on both sides, and designs entered twice with rounded proportions. The model
recognizes them rather than predicting (Roberts et al., 2017, on validation with dependent
data). Lecture 2 measures it on concrete; Lecture 3 on bridges, where neighbors and repeated
inspections play the same part.
In the picture, 36 designs tested at least five times each, every random test row has its
design in training. Over the 30 random splits, on average 88% of test rows have their design in
training (83% to 93% by split), and 74% their identical mix (67% to 80%); random splits score
8.9 MPa, whole designs 9.5.
:::

## New materials void the promise

Fit without fly ash: 566 tests. On 464 with fly ash: **13.7 MPa**, worse than validation's 9.8.

![Dot plot of RMSE in MPa for a 5-NN model fitted to mixes without fly ash. Validation on held-out designs without fly ash: 9.8 with an error bar of one standard deviation over 30 splits. New mixes with fly ash: 13.7. A dotted line marks 14.4 for predicting the mean on the fly-ash mixes.](cee_l01_framing/new_material.png)

::: notes
Fly ash is zero in every training test, so its standardized column is the same for all of them
and adds the same amount to every distance: it never changes which tests are nearest. The model
answers as if the fly ash were not there. Its 13.7 MPa is barely better than predicting the
training mean for every fly-ash mix (14.4). No split of the training data could warn us: every
validation case was drawn from mixes without fly ash. The test cases must look like the cases
the model will face; when the future holds a new material, collect tests of it first.
:::

## Before you sign: four questions

| Ask | Today's answer |
|:--------------------------------|:------------------------------------|
| The error on cases like this one? | 9.9 MPa: 28-day tests, designs held out |
| Is the case inside the data? | Not with fly ash: 13.7, not 9.8 |
| What does an error cost, and who carries it? | Predicted 38.1, measured 28.6: a wrong "yes" |
| Who is accountable? | The engineer who signs, not the model |

::: notes
Answer to warm-up question 1. Team D's 9.5 MPa is the error over all ages; the decision is
about 28-day tests, and on held-out 28-day tests (designs held out, tested once) the error is
9.9 MPa (the mean of the 30 splits' RMSEs; pooled over their predictions, 9.92). With that as the margin (approve if the prediction minus 9.9 reaches 35 MPa), 1.1% of
weak 28-day predictions are approved, but 59.2% of strong ones are rejected (3,179 held-out
predictions of 425 tests at 28 days, over the 30 splits; 51% weak). With no margin: 13.0% of
weak approved, 30.5% of strong rejected. An honest error can say that a model is not ready:
this one could screen designs, not approve them. Lecture 2 cuts the error with engineering
knowledge.
:::

## Sometimes the answer is not ML

![Flowchart of four questions in a row. A trusted rule already answers? If yes, use the rule: a code, Abrams' law. If no: is the case inside the data? If no, test first: fly ash, new ages. If yes: is the error measured on cases like it? If no, do not sign: no honest error. If yes: is a wrong yes caught before it matters? If no, test first: cast cylinders. If yes: ML, with its honest error.](cee_l01_framing/not_ml.png)

::: notes
When not to use a method is part of the method. A model can still help in these cases as a
screen, a first guess, or a check on another method, with the decision resting on something
else. Lecture 2 makes the last point concrete: a model can screen mixes; cylinders accept them.
:::

## Someone else carries the error

:::::: {.columns}
::: {.column}
- **Ethical:** optimism moves risk to the public.
- **Social:** fly-ash producers meet 13.7, not 9.8.
- **Economic, environmental:** redesigns, tests, cement.
:::
::: {.column}
![Two curves against the margin, 0 to 16 MPa, over 3,179 held-out 28-day predictions. Weak mixes approved, the public's risk: 13% with no margin, falling to 1.1% at the honest margin of 9.9 MPa. Strong mixes rejected, the cost: 30.5% with no margin, rising to 59.2% at 9.9 MPa and 77.7% at 16.](cee_l01_framing/costs.png)
:::
::::::

::: notes
The Task Force asks that social, ethical, environmental, and economic considerations be
integrated throughout AI curricula rather than isolated in stand-alone courses; this course
gives each unit one such note. Ask: who would notice if the model were 1.0 MPa more optimistic
than it should be, and when? Lecture 27 returns to whose cases are in the data; Lecture 28 to
AI itself as infrastructure, drawing on power grids and water systems.
The curves: approve when the prediction minus the margin reaches 35 MPa. A margin is a choice of
who carries the error: with none, 13% of weak mixes are approved; with the honest 9.9 MPa,
1.1%, but 59.2% of strong mixes are rejected.
:::

## ML already informs CEE decisions

| Where | What runs today | Lectures |
|:------|:--------------------------------------|:------|
| Flood warnings | AI forecasts of extreme floods in ungauged basins, 5 days ahead, public in 80+ countries (Nearing et al., 2024) | 11, 20 |
| Weather | ECMWF's machine-learned forecast, operational since 25 Feb 2025, beside its physics model | 17 |
| Both | Tested on cases like the ones they face? Who signs the decision? | today |

::: notes
Where today's ideas live in current practice. Weather and flood forecasts feed decisions on
stormwater, construction schedules, and emergency response. ECMWF runs its AI model side by side with the
physics-based one, not instead of it. Ask of each what an honest test looks like: for a
forecast, later years held out; for flood warnings in ungauged basins, whole basins held out.
Lecture 11 covers streamflow intervals at
Illinois gauges; Lecture 17, transformers; Lecture 20, rainfall-runoff models and pretrained
forecasters.
:::

## Vote: would you sign this mix?

A model fit to mixes without fly ash predicts 47.3 MPa for a new design with 77 kg/m³ of fly
ash. $f^{\prime}_c=35$ MPa. Validation, designs held out: 9.8 MPa.

- **A.** Sign: 47.3 − 9.8 = 37.5 MPa clears $f^{\prime}_c$.
- **B.** Sign if a second model, fit to the same tests, agrees.
- **C.** Not yet: no training test had fly ash. Cast cylinders first.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
C. The 9.8 MPa was measured on mixes like the training mixes; this one is outside the data.
On fly-ash mixes the same model errs by 13.7 MPa, and 47.3 - 13.7 = 33.6 MPa does not clear
35. B: a second model fit to the same tests shares the blind spot. The mix is real: measured
33.8 MPa, below f'c. How it was chosen: among 28-day fly-ash tests that A's rule would approve
but that measured below 35 MPa, the one whose error is closest to 13.7 MPa. Then ask what
would change the answer: tests of fly-ash mixes in training, or a cheaper wrong "yes".
:::

## Every lecture runs the same steps

![Diagram of the eight steps of every lecture in two rows: 0 warm-up, recall; 1 question, you try first; 2 one picture; 3 derive from the assumption; 4 build and verify; 5 break it; 6 decide, vote twice; 7 exit question. Notes give the evidence: recall beats rereading; try first, effect size 0.36 over 53 studies; predict before every reveal; peer discussion between votes. A side note: 1.5 times as likely to fail with lecture only; active classes learn more but feel they learn less.](cee_l01_framing/arc.png)

::: notes
Answer to warm-up question 3: more in the active class, yet students there felt they learned
less, partly because the effort feels like confusion (Deslauriers et al., 2019, a randomized
comparison with identical materials). Expect that feeling here; it is the work of learning,
not a sign that it is failing. Today you ran every step: the warm-up, the four teams, the
picture, the derivation, the code, three breaks, the vote. Recall helps most at a delay:
students who reread did better five minutes later, but those who were tested remembered more
days later (Roediger & Karpicke, 2006). Students who only watch a demonstration understand it
no better than students who never see it; predicting first helps (Crouch et al., 2004).
Effect size 0.36 is Hedges' g for problem solving before instruction against instruction first.
Sources: Roediger & Karpicke (2006); Sinha & Kapur (2021); Crouch et al. (2004); Freeman et al.
(2014); Deslauriers et al. (2019).
:::

## Grades: XP, two exams, a project

$$\text{Grade}=\frac{\text{XP}+\text{2 best exams}+\text{project}}{\max(\text{XP},\,\text{XP}_{\text{target}})+300}$$

| Part | Counts |
|:------------------------------------|:-------------------------------|
| Five homeworks: 100-point core plus stretch goals | XP; target 400 (3 credits) or 525 (4) |
| Three CBTF exams | the best two, 100 points each |
| Final project on custom CEE data, oral defense | 100 points |

::: notes
XP also comes from in-class activities. Ten free late days for homework; the final project
cannot be late. Final project: teams of two to four; a datasheet for the dataset, a model card
with disaggregated results, a two-page engineer's memo, an impact statement on the four
dimensions, and the oral defense. CBTF: the Computer-Based Testing Facility, proctored.
XP beyond the target raises the grade. Every homework has one critique-the-machine problem: an
AI-generated analysis with a seeded error to find, explain, and fix.
:::

## AI use is declared on every task

| Component | AI-use level | Why |
|:----------|:---------|:--------------|
| CBTF exams | Restricted | They measure unaided mastery. |
| Homework cores: from-scratch code and analysis | Limited Use | AI may explain and help debug; producing the code and the analysis is how you learn. |
| Critique-the-machine problems | Use When Instructed | You work with AI output by design. |
| Stretch goals and final project | All Use, with a disclosure log | The grade rests on your data, your verification, and your oral defense. |

::: notes
The four levels are the Grainger College of Engineering's (AI Guidance, 2026): Restricted, Limited
Use, Use When Instructed, All Use. Exams are where the course certifies fluency, so they are
Restricted. Critique-the-machine: a plausible AI-generated analysis with a seeded physical or
statistical error. The assessments are designed to hold up when students use AI: proctored
exams, oral defenses, and analyses you must verify.
:::

## Exit question

A city wants a model that flags water mains likely to break next year. Name its input, its
output, and the decision it feeds. How would you split the records so the test set looks like
next year?

**In the notebook** (`cee_l01_framing.ipynb`): rank the four teams yourself, write `val_rmse`
and match scikit-learn, reproduce the winner's-curse curve, and break the model with fly ash.

::: notes
One answer. Input: material, diameter, age, soil, pressure, past breaks of each main. Output:
the chance it breaks next year. Decision: which mains to inspect or replace first, within a
budget. Split by time: fit and validate on years up to the last one, test once on the last
year; and keep each main, or each neighborhood, on one side. A random split of main-years would
put a main's own history on both sides. Collect the answers; they seed the next warm-up.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- Frame it: input, output, decision.
- Training fits, validation chooses, test reports once.
- The best of many scores is too good.
- Honest only for cases like the held-out ones.

**Next:** nearest neighbors: which mixes are similar?
:::
::: {.column}
![Four thumbnails of the lecture's key figures: whole designs held out as columns; the best of many equally good models scoring too well; the fly-ash mixes' error far above validation's; weak mixes approved and strong mixes rejected against the margin.](cee_l01_framing/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- AI Curriculum Task Force (2026).
- Cawley & Talbot (2010). *JMLR*.
- Chollet (2017). Manning.
- Crouch et al. (2004). *Am. J. Phys.*
- Deslauriers et al. (2019). *PNAS*.
- ECMWF (2025). AIFS operational.
- FHWA. National Bridge Inventory.
- Forsyth (2019). Springer.
- Freeman et al. (2014). *PNAS*.
:::
::: {.column}
- Grainger (2026). AI Guidance.
- Miller et al. (2020). *Sci. Data*.
- Nearing et al. (2024). *Nature*.
- Pan & Carpenter (2023). *EPR*.
- Roberts et al. (2017). *Ecography*.
- Roediger & Karpicke (2006). *Psych. Sci.*
- Sinha & Kapur (2021). *RER*.
- Yeh (1998). *Cem. Concr. Res.*
:::
::::::

::: notes
Full references.
Campus AI Curriculum Task Force, AY2025-26. Office of the Provost, University of Illinois Urbana-Champaign. provost.illinois.edu/about/committees/campus-ai-curriculum-task-force/
Cawley, G. C. & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. Journal of Machine Learning Research 11, 2079-2107.
Chollet, F. (2017). Deep Learning with Python. Manning.
Crouch, C. H., Fagen, A. P., Callan, J. P. & Mazur, E. (2004). Classroom demonstrations: Learning tools or entertainment? American Journal of Physics 72(6), 835-838. doi:10.1119/1.1707018
Deslauriers, L., McCarty, L. S., Miller, K., Callaghan, K. & Kestin, G. (2019). Measuring actual learning versus feeling of learning in response to being actively engaged in the classroom. PNAS 116(39), 19251-19257. doi:10.1073/pnas.1821936116
ECMWF (2025, 25 February). ECMWF's AI forecasts become operational. ecmwf.int/node/29308
FHWA. National Bridge Inventory. fhwa.dot.gov/bridge/nbi.cfm
Forsyth, D. (2019). Applied Machine Learning. Springer. doi:10.1007/978-3-030-18114-7. Chapters 1 and 3.
Freeman, S. et al. (2014). Active learning increases student performance in science, engineering, and mathematics. PNAS 111(23), 8410-8415. doi:10.1073/pnas.1319030111
Grainger College of Engineering (2026). AI Guidance: Sample Statements for Engineering. genai.grainger.illinois.edu/ai-guidance
Miller, C. et al. (2020). The Building Data Genome Project 2. Scientific Data 7, 368. doi:10.1038/s41597-020-00712-x
Nearing, G. et al. (2024). Global prediction of extreme floods in ungauged watersheds. Nature 627, 559-563. doi:10.1038/s41586-024-07145-1
Pan, S. C. & Carpenter, S. K. (2023). Prequestioning and pretesting effects. Educational Psychology Review 35, 97. doi:10.1007/s10648-023-09814-5
Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40(8), 913-929. doi:10.1111/ecog.02881
Roediger, H. L. & Karpicke, J. D. (2006). Test-enhanced learning. Psychological Science 17(3), 249-255. doi:10.1111/j.1467-9280.2006.01693.x
Sinha, T. & Kapur, M. (2021). When problem solving followed by instruction works. Review of Educational Research 91(5), 761-798. doi:10.3102/00346543211019105
Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. Cement and Concrete Research 28(12), 1797-1808. doi:10.1016/S0008-8846(98)00165-3
:::
