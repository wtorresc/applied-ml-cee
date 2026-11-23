---
title: Nearest Neighbors and Representation
subtitle: "Lecture 2 · Will this concrete mix reach its specified strength?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions from Lecture 1

On your own, no notes, two minutes:

1. Which part of the data may you look at only once, and why only once?
2. A model ranks bridges for inspection from their records. Name its input, its output, and the
   decision it feeds.
3. Name one thing you would need before signing a decision based on a model's output.

**Today:** will a proposed concrete mix reach its specified strength?

::: notes
Give the answers before moving on; recall helps most with feedback.
1. The test set, once, at the end. Every look that changes the model makes its score optimistic.
2. Input: age, materials, traffic, past condition ratings. Output: a condition rating, or the
   chance of poor condition. Decision: which bridges to inspect or repair first.
3. Any of: the error on cases like this one; whether this case lies inside the data; what an
   error costs, and who carries it; who is accountable for the decision.
Every lecture opens this way, with a few questions on earlier lectures, mixed.
:::

## Past tests can answer today

A slab at $f^{\prime}_c=35$ MPa. Predict from the most similar tested mixes.

| | Mix is strong | Mix is weak |
|:--------|:-------------------|:-------------------|
| **Accept** | ✓ | weak concrete in place: cores, strengthening, removal |
| **Reject** | extra cement: money, carbon, a redesign | ✓ |

::: notes
Frame the problem by what an error costs before talking about accuracy. We return to the two
kinds of error in the Decide step.
The owner and the public carry a wrong "yes"; the producer pays for a wrong "no". 35 MPa is
about 5,000 psi.
:::

## 1,030 tests, but only 428 mixes

1,030 tests, 14 ages, 2.3 to 82.6 MPa. 246 mixes tested once.

**One mix, many rows:** remember it at Break-it.

![Strength against age on a log axis, one blue line per mix tested at several ages, from 1 to 365 days, rising roughly linearly in log age; orange dots for the 246 mixes tested once, almost all at 28 days. A dashed line marks 35 MPa.](cee_l02_nearest_neighbors/mixes.png)

::: notes
Two minutes on your own: estimate the proposed mix's 28-day strength, and write down which mixes
you relied on, and why. All five mixes were tested at 28 days. Collect a few estimates and ask
what each student matched on. Matching on the kilograms points to
A, the nearest in raw units, which was tested twice (59.5 and 62.9 MPa, mean 61.2). Matching on
the water–binder ratio, as Abrams' law suggests, points to B, C, and D (w/b 0.40 to 0.42), mean
45.3 MPa. Do not reveal the answer yet: the next slide traces this mix. Keep the class's
answers; we return to them on the neighbors slide.
:::

## The encoder decides who is near

![Diagram of the course pipeline with the decoder highlighted. A mix x goes to an encoder that computes dimensionless groups, giving z; the decoder averages the strengths of the k nearest training tests, which are stored rather than fitted (746 tests of 236 designs); the prediction y-hat is compared with the measured strength y by the squared-error loss. Below, one held-out design tested at 28 days: cement 298, slag 0, fly ash 107, water 164, superplasticizer 13, coarse aggregate 953, fine aggregate 784 kg per cubic meter. Its water-binder ratio is 0.40 and ln age 3.33. The 5 nearest training tests reached 51.3, 38.5, 38.5, 37.3, and 47.4 MPa, mean 42.6 and spread (sample standard deviation) 6.4. The prediction is 42.6 MPa; the measured strength is 35.9 MPa, an error of +6.7 MPa, against f'c = 35 MPa.](cee_l02_nearest_neighbors/knn_pipeline.png)

::: notes
This is the proposed mix from your attempt: it measured 35.9 MPa. The encoder is where
engineering knowledge enters; the decoder is today's method. The value 38.5 appears twice among
the neighbors because the same test is recorded twice in the data, a first hint for the Break-it
step. The traced design is the 28-day held-out test closest to 35 MPa in the first held-out split.
:::

## Similar mixes, similar strengths

**Assumption:** similar mixes reach similar strengths.
$$\hat{y}(\mathbf{z})=\frac{1}{k}\sum_{i\in\mathcal{N}_k(\mathbf{z})}y_i$$

![3D surface of the 5-NN prediction over water-binder ratio, 0.25 to 0.80, and age, 1 to 365 days, with the proposed mix's other ingredients fixed. The surface is a set of flat terraces joined by steps, high at low w/b and old age, low at high w/b and young age; the proposed mix is a dot at w/b 0.40 and 28 days.](cee_l02_nearest_neighbors/terraces.png)

::: notes
State the assumption before the method; every method rests on one. The Break-it step tests this
one: where the neighbors are not really similar, or where no tested mix is similar at all.
$\mathcal{N}_k(\mathbf{z})$ are the $k$ training tests closest to $\mathbf{z}$ in Euclidean distance;
"training" only stores the tests, and each prediction costs O(ND) by brute force. The surface:
between tested mixes the prediction is flat (82% of the grid steps along w/b change nothing),
and it steps where the set of five neighbors changes. Everything rests on z: what makes two
mixes similar?
:::

## Raw kilograms hide the water

- For two independent tests:
  $$\mathbb{E}\,(x_j-u_j)^2=2\,\mathrm{Var}(x_j)$$
- Input $j$'s share is $\mathrm{Var}(x_j)/\sum_l \mathrm{Var}(x_l)$: water **1.2%**,
  cement 27.7%.
- **Check:** add 30 kg/m³ of water to the proposed mix. Near or far?

![Scatter plot of the four attempt mixes, raw distance to the proposed mix against water-binder ratio. A: 75.8 kilograms per cubic meter away, w/b 0.30, 59.5 and 62.9 MPa. B: 77.8 away, w/b 0.42, 47.4 MPa. C: 90.7 away, w/b 0.41, 37.3 MPa. D: 104.5 away, w/b 0.40, 51.3 MPa. A dashed line marks the proposed mix's w/b of 0.40.](cee_l02_nearest_neighbors/attempt_distance.png)

::: notes
Derivation: for independent x and u from the same distribution, E(x_j - u_j)^2 = Var(x_j) +
Var(u_j) = 2 Var(x_j), because the cross term has mean zero. The shares are the variances of the
columns over all 1,030 tests; the aggregates together take 31.6%. In the four mixes, A is the
nearest in kilograms but the farthest in w/b; water is 1.7% of its squared distance.
Check: in raw units, 30 kg/m³ of water is nearer than any tested mix (the nearest is 33.1 kg/m³
away), so the changed mix looks almost identical. Abrams' fit says otherwise: w/b goes from 0.40
to 0.48 and strength drops from 39.8 to 32.4 MPa, from above f'c to below it.
:::

## Strength follows w/b and log age

![Two panels. Left: 28-day strength against water-binder ratio for 425 tests, falling from about 80 MPa near w/b 0.3 to about 15 MPa near 0.8, with the fitted Abrams curve f = 124 / 16.5 to the power w/b. Right: strength against age on a log scale for the four mixes tested at the most ages; each rises roughly linearly in log age from 1 or 3 days to 365 days.](cee_l02_nearest_neighbors/abrams_and_age.png)

::: notes
Fit: least squares of ln f on w/b over the 28-day tests gives A = 123.8 MPa and B = 16.5. The
age gain is the median over the 133 mixes tested at 4 or more ages. Standardize with training
statistics only.
The curve is a summary, not a model of every mix: the scatter around it is what the other
ingredients and the testing add.
Standardizing weighs every ingredient equally; strength depends on proportions, so equal is
arbitrary. Abrams (1918): strength falls exponentially as the water-cement ratio rises,
f = A / B^(w/c). With slag and fly ash in the binder b, use w/b: on the 425 28-day tests it
explains 60% of the variance of ln f; w/c explains 42%. Age: a day of curing matters more at
day 3 than at day 300; the median gain is 8.7 MPa per unit of ln(age).
:::

## Encode a mix as an engineer would

```python
def mix_groups(df: pd.DataFrame) -> np.ndarray:
    """Dimensionless descriptors of each mix, plus log age."""
    b = df.cement + df.slag + df.fly_ash         # binder, kg/m³
    return np.column_stack([
        df.water / b,                            # Abrams: w/b
        df.slag / b, df.fly_ash / b,             # binder makeup
        df.superplasticizer / b,
        (df.coarse_agg + df.fine_agg) / b,       # aggregate-binder
        np.log(df.age),                          # gain ~ log time
    ])
```

Ratios to the binder $b$ = cement + slag + fly ash, and log age; standardized with training
statistics.

::: notes
This is dimensional analysis put to work as an encoder (Buckingham, 1914): describe a physical
system by dimensionless groups, and similar groups mean similar behavior. Eight inputs with units
become six dimensionless numbers. Fit the StandardScaler on the training rows only, then transform
both sets.
:::

## The decoder: distance, sort, mean

```python
def knn_regress(Z_tr: np.ndarray, y_tr: np.ndarray, Z_te: np.ndarray,
                k: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and spread of the k nearest training strengths."""
    d2 = ((Z_te**2).sum(1)[:, None] - 2 * Z_te @ Z_tr.T
          + (Z_tr**2).sum(1)[None, :])
    assert d2.shape == (len(Z_te), len(Z_tr))
    nn = np.argsort(d2, axis=1, kind="stable")[:, :k]  # k nearest
    return y_tr[nn].mean(axis=1), y_tr[nn].std(axis=1, ddof=1)
```

**Predict:** on 284 held-out tests with $k=5$, will this match scikit-learn's
`KNeighborsRegressor` on every test?

::: notes
Take a show of hands: on every test, on almost every test, or often different. The next slide
answers. The spread is the sample standard deviation of the k strengths (ddof = 1: divide by
k − 1, since the k neighbors are a sample).
:::

## Raw units pick the wrong neighbors

Raw units pick A and B; the groups pick D, C, B. Matches scikit-learn on 283 of 284 tests (the one difference is a tie; 8 tests have ties).

| Rank | Raw: w/b | Raw: MPa | Groups: w/b | Groups: MPa |
|:--|--:|--:|--:|--:|
| 1 | 0.40 | 38.5 | 0.40 | 51.3 |
| 2 | 0.40 | 38.5 | 0.40 | 38.5 |
| 3 | 0.30 | 59.5 | 0.40 | 38.5 |
| 4 | 0.30 | 62.9 | 0.41 | 37.3 |
| 5 | 0.42 | 47.4 | 0.42 | 47.4 |
| Mean | | 49.4 | | 42.6 |

::: notes
Ties: 8 of the 284 tests have two training records equally far away at rank 5. Four are one
design tested at two ages equally far in log age (14 and 56 days around 28, or 7 and 28 around
14); three come from a series that steps slag up and down around the test's design; one is two
designs the encoder cannot tell apart (only the coarse/fine split differs), measured 25.6 and
26.2 MPa. Which tied record counts is arbitrary in both codes; only that last one changes the
mean. The 38.5 MPa records at ranks 1 and 2 are one more
design, entered twice, left out of the attempt because both encoders find it.
Return to the attempt: matching on w/b, as most engineers do, is what the groups encode. Error:
+13.5 MPa with raw units, +6.7 MPa with the groups; measured 35.9. Work this by hand in the
notebook.
:::

## Groups find tests like the mix

![Two panels of the proposed mix's 30 nearest training tests, plotted by water-binder ratio and age, the proposed mix as a star at w/b 0.40 and 28 days. Raw kilograms and days: the 30 spread over ages from 3 to 56 days, only 15 at 28 days. Dimensionless groups: 24 of 30 at 28 days, the rest at 14 and 56, with w/b from 0.31 to 0.52.](cee_l02_nearest_neighbors/near.png)

::: notes
All eight inputs, the proposed mix's 30 nearest training tests (more than 5, to see the pattern).
Raw units measure age in days, so a test at 3 days is as close as one with 25 kg/m³ more
cement: age and the big-kilogram ingredients decide. The groups compare w/b and log age, what
Abrams and curing say matter: 24 of the 30 are 28-day tests.
:::

## Representation did most of the work

Raw 10.0 → groups **6.9 MPa**. Engineering knowledge, not the algorithm, did it.

![Dot plot of test RMSE in MPa with one-standard-deviation error bars over 30 splits for 5-NN: raw kilograms and days 10.0, standardized 9.5, standardized with ln age 8.0, dimensionless groups 6.9. A dotted line marks 17.0 for predicting the mean; a dashed line marks 1.4 for the scatter of repeat tests.](cee_l02_nearest_neighbors/representation_rmse.png)

::: notes
Same 5-NN and 30 splits, designs held out. The two big steps: log time (9.5 to 8.0) and Abrams'
ratio (8.0 to 6.9). Error bars are one standard deviation over 30 splits; the bars overlap, but
the encoders are scored on the same splits, so compare them split by split: log age beats
standardized in 30 of 30 splits (by 1.54 MPa on average, standard deviation 0.42), and the
groups beat log age in 29 of 30 (by 1.10, standard deviation 0.57). The four encoders were
fixed in advance from engineering knowledge, not searched on these splits. The floor comes from tests with identical proportions and age
whose strengths differ (exact duplicates left out): 9 groups, 20 tests, median standard deviation
1.4 MPa, largest 16.5 MPa. The 9 groups come from only 3 mixes, two of which supply 8, so treat
the floor as an indication: no model of these inputs can beat the scatter of the test itself,
and 6.9 MPa leaves room for better models.
Above: predicting the mean, 17.0 MPa. Below: repeat tests of one mix differ by a median 1.4 MPa
(9 groups from 3 mixes; an indication).
:::

## Predict before we test

1. **Splits.** Random tests; identical mixes held out; whole designs held out. Rank their errors,
   lowest first. How far apart?
2. **Ages.** Train only on tests up to 28 days, then predict the tests at 56 to 365 days. Too
   high, too low, or both?
3. **Physics.** Raise only the water of one mix, step by step. Does the prediction always fall?

::: notes
Give one minute; students commit in writing. Each of the next three slides reveals one answer.
Ask who predicted it, and what the others missed.
:::

## Random splits flatter the model

Random **5.7**, mixes held out 6.3, designs held out **6.9** MPa. The model will face new designs.

![Line plot of test RMSE against k from 1 to 20 for three splits, dimensionless groups. Random split of tests: lowest, about 5.6 to 6.6 MPa. Identical mixes held out: 6.3 to 6.8. Designs held out: highest, 6.8 to 8.0. The title marks the plot as post hoc: every k scored on the test splits, with k = 5 fixed in advance.](cee_l02_nearest_neighbors/leakage_vs_k.png)

::: notes
The curves are a post-hoc diagnostic: they score every k on the test splits to show how the
split changes the error at any k, not to choose k (that would make the test splits a
validation set). The lecture's k = 5 was fixed before.
The data combine records in which the same design appears with rounded and unrounded
proportions, for example fine aggregate 806.0 and 806.9 kg/m³. One 28-day test even appears
twice, once with 140 kg and once with 139.7 kg of cement, with the identical strength 35.2 MPa.
Ask: what will the model face in practice? A new design. So hold out designs (Kaufman et al.,
2012, on leakage; Roberts et al., 2017, on structured cross-validation). Also: ten mix–age pairs
are recorded more than once with identical strength, which is the same record entered twice.
Over the 30 random splits, on average 74% of test rows (67% to 80% by split) have their own mix,
at another age, in training. Holding out identical mixes still leaves 22% (15% to 28%) with a
near-identical design in training.
:::

## k-NN cannot see past tested ages

Trained to 28 days, predicting 56–365: mean error **−9.0 MPa** (all ages trained: −1.4).

![Scatter plot of 5-NN predictions against measured strength for tests at 56 days or older. Squares, trained only on ages up to 28 days, fall well below the diagonal; circles, with all ages in training, scatter around it.](cee_l02_nearest_neighbors/age_extrapolation.png)

::: notes
RMSE 10.7 MPa when the late ages are outside training, 7.4 MPa when they are inside.
Not a ceiling on strength: only 1 of the 281 late tests is stronger than the strongest training
test (81.8 MPa). The input is the problem: a late test's ln(age) lies beyond every training
value, so its nearest neighbors are the 28-day tests (99.9% of them), and 95% of the late tests
have their own mix among their five. k-NN is flat beyond the training range of an input:
extrapolation in the inputs.
Caveat: the −9.0 run keeps each mix's early tests in training, while the −1.4 run holds whole
designs out, so the two differ in two ways at once.
749 training tests at ages up to 28 days; 281 tests at 56 days or later. Age leaves the training
range: 99.9% of the neighbors are 28-day tests, so k-NN predicts 28-day strengths.
:::

## k-NN can contradict physics

Only the water rises. 5-NN goes **up** in 3 of 13 steps; past $w/b\approx0.57$ it sticks at 30.1 MPa.

![Step plot of the 5-NN 28-day strength prediction as only the water changes, against water-binder ratio from 0.30 to 0.65, with the fitted Abrams curve dashed. The prediction falls from 48.7 to 30.1 MPa in 13 steps, rising in three of them, and stays flat at 30.1 MPa beyond w/b 0.57, while the Abrams curve keeps falling to 20 MPa.](cee_l02_nearest_neighbors/monotonicity.png)

::: notes
This is the lecture's decisive figure. Binder 405 kg/m³. From w/b 0.57 on, the same five
tests stay nearest, and all five are drier (w/b at most 0.55). Training holds 159 wetter tests,
but none is like this mix in its other groups (each at least 1.65 standardized units away in
them; the five, at most 0.73), so none ever comes nearer (at w/b 0.65 the nearest wetter test
is 1.72 away, the five 0.91). The distance does not warn: at w/b 0.57 the nearest test is 0.48
away, nearer than the proposed mix's own nearest (0.64), and held-out tests' nearest distances
have median 0.37 and 90th percentile 0.74.
A model can fit the held-out tests well and still contradict physics between and beyond them.
Check models against physics before trusting them; Lecture 23 builds the physics into the
model. Abrams: strength only falls; at w/b 0.65 it gives 20.0 MPa. The increases reach 2.6 MPa.
The assumption fails twice: neighbors not similar enough, then the same five drier mixes answer
every wetter query.
:::

## Every margin is a trade-off

Accept if prediction − margin ≥ 35. No margin: 13.9% of weak accepted; 5 MPa: 4.4%.

![Trade-off curve of the percentage of weak mixes accepted against the percentage of strong mixes rejected, as the margin grows from 0 to 15 MPa. No margin: 13.9% accepted, 14.9% rejected. A 5 MPa margin: 4.4% and 36.5%. A margin equal to the neighbors' spread: 4.8% and 37.0%, a little right of the curve, by less than resampling the designs moves it.](cee_l02_nearest_neighbors/decision_tradeoff.png)

::: notes
Weak means measured below 35 MPa (51% of these predictions). Each test is held out in several
of the 30 splits, so the rates count predictions, not distinct tests. The cost of a wrong "yes" chooses the
point on the curve. Design codes already require a mix to target an average strength above
the specified strength (ACI 318); this is the same idea, applied to a prediction.
3,179 held-out predictions of the 425 tests at 28 days, over 30 splits. No margin: 13.9% of weak
mixes accepted, 14.9% of strong rejected. 5 MPa: 4.4% and 36.5%. The neighbors' spread
(sample standard deviation, median 4.7 MPa): 4.8% and 37.0%. A fixed 4.62 MPa margin accepts
the same 4.8% of weak mixes and rejects 34.8% of strong ones: 2.2 points fewer. Resampling the
308 designs (400 draws, the margin matched again in each) puts that gap between −5.2 and +8.9
points, at or below zero in 29% of draws: the spread is no better than a fixed margin.
:::

## Screen with k-NN; accept with cylinders

| Use 5-NN to | Not to |
|:------------------------------|:------------------------------|
| screen proposed mixes inside the tested range | predict beyond tested ages, w/b, or binders |
| rank trial batches to cast first | judge new cements, admixtures, aggregates |
| flag a mix for a closer look | accept concrete: cylinders do (ACI 318) |

::: notes
When not to use the method is part of the method. Duplicated records also make the spread too
small: two copies of one test agree perfectly.
The neighbors' spread (proposed mix: 42.6 ± 6.4 MPa, measured 35.9) is no better than a fixed
margin: a fixed 4.62 MPa accepts as few weak mixes (4.8%) and rejects 34.8% of strong ones
against 37.0%, a gap that resampling the designs puts between −5.2 and +8.9 points. With k = 5
the spread is noisy itself. The engineer of record signs the design, not
the model. Uncertainty with a guarantee comes later: calibration (Lecture 10), conformal
prediction (Lecture 11).
:::

## The margin costs carbon or safety

- Each MPa of margin: about **9 kg** more binder per m³.
- Cement: roughly 8% of global CO₂.
- A narrow margin moves risk to the public.

![Binder needed to reach 35 MPa plus a margin, from Abrams' fit at the proposed mix's 164 kg/m³ of water: 364 kg/m³ with no margin, 407 with 5 MPa, 454 with 10, rising to about 507 with 15.](cee_l02_nearest_neighbors/binder.png)

::: notes
This is the trade-off two slides back, priced in carbon and in safety. The vote that follows asks
the class to choose.
Binder from Abrams' 28-day fit, f = 123.8 / 16.5^(w/b), solved for w/b at 35 MPa plus the
margin, with the water held at the proposed mix's 164 kg/m³: 364, 407, 454 kg/m³ at 0, 5, 10
MPa. Cement share of global CO₂: Lehne & Preston (2018). A model that lets a producer cut
cement must show that it does not raise the share of weak mixes accepted.
:::

## Vote: would you accept this mix?

$f^{\prime}_c=35$ MPa. The 5-NN prediction is 37 MPa; the neighbors' spread is 6 MPa.

- **A.** Accept: the prediction is above 35 MPa.
- **B.** Reject: 37 − 35 is smaller than the spread.
- **C.** Not yet: require a trial batch and cylinder tests first.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
A ignores the uncertainty: with no margin, 13.9% of weak mixes were accepted. B treats a rough
spread as if it were calibrated. C is the engineer's answer when a wrong "yes" is costly: codes
already ask a mix to target an average strength above f'c (ACI 318), and a prediction 2 MPa
above f'c with a 6 MPa spread does not do that. Then ask what would change the answer: a
calibrated interval (Lectures 10–11), a cheaper wrong "yes", or more tested mixes like this one.
:::

## k-NN lives on in retrieval

![Diagram of retrieval-augmented generation in the course's terms. A question goes to an encoder that maps it to an embedding vector; a nearest-neighbor search finds the k closest passages in an index of documents; the passages and the question go to a language model, the decoder, which writes the answer. Notes: the encoder decides what near means, as the groups did for concrete; at scale the search is approximate, over a billion vectors on GPUs; with no truly similar passage, the answer can still sound sure.](cee_l02_nearest_neighbors/rag.png)

::: notes
Where today's method lives in current systems. The failure is the one from Break it: when no
tested mix is similar, k-NN still returns an average; when no passage answers the question,
RAG still returns an answer. Evaluating retrieval is part of Lecture 19.
Retrieval-augmented generation: Lewis et al. (2020). Approximate search on GPUs: Johnson et al.
(2021). Lecture 19 builds one: question answering over the MUTCD (FHWA, 11th Edition), with
its retrieval evaluated.
:::

## Exit question

Which change cut the error most, and which split would you use to report the error to a client?
Why?

**In the notebook** (`cee_l02_nearest_neighbors.ipynb`): build the encoder and `knn_regress`,
match scikit-learn, work the four mixes by hand, and reproduce the leakage and extrapolation
tests.

::: notes
Answer: log age (9.5 to 8.0 MPa), then Abrams' ratio (8.0 to 6.9 MPa), both from engineering
knowledge. Hold out whole designs, because the client's mix is a new design. Collect the answers
on paper or in a poll; they show what to revisit in the next warm-up.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- k-NN: the engineering is in "similar".
- Ratios and log age: 10.0 → 6.9 MPa.
- Hold out whole designs.
- Check extrapolation and physics.

**Next:** generalization, and validation the way infrastructure data fail.
:::
::: {.column}
![Four thumbnails of the lecture's key figures: raw kilograms against the groups picking the proposed mix's neighbors; the 5-NN prediction as terraces; error against k for three kinds of split; the prediction rising as water rises.](cee_l02_nearest_neighbors/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Abrams (1918). *Design of Concrete Mixtures*. Lewis Institute, Bulletin 1.
- ACI Committee 318 (2019). *Building Code Requirements for Structural Concrete* (ACI 318-19).
- Buckingham (1914). *Physical Review* 4(4).
- Forsyth (2019). *Applied Machine Learning*. Springer, Ch. 1.
- Johnson, Douze & Jégou (2021). *IEEE Trans. Big Data* 7(3).
:::
::: {.column}
- Kaufman et al. (2012). Leakage in data mining. *ACM TKDD* 6(4).
- Lehne & Preston (2018). *Making Concrete Change*. Chatham House.
- Lewis et al. (2020). Retrieval-augmented generation. *NeurIPS* 33.
- Roberts et al. (2017). *Ecography* 40(8).
- Yeh (1998). *Cement and Concrete Research* 28(12).
:::
::::::
