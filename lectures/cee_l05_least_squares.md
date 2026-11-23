---
title: Least Squares and Regularization
subtitle: "Lecture 5 · How much cement does a 35 MPa mix need?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 2–4

On your own, no notes, two minutes:

1. 5-NN on raw kg/m³ chose a mix with $w/b=0.30$ as a neighbor of one with $w/b=0.40$. Why,
   and what fixed it?
2. Bridges in one county share climate, traffic, and inspectors. Why does a random split of
   bridge records overstate a model's accuracy? How would you split instead?
3. Stack buildings' daily load profiles as rows and center the matrix. What do its right
   singular vectors and its singular values tell you?

**Today:** how much cement does a 35 MPa mix need?

::: notes
Give the answers before moving on; recall helps most with feedback.
1. In raw units the squared distance is dominated by the inputs with the largest variance
   (cement and the aggregates); water carried 1.2% of it. The Lecture 2 encoder (w/b, binder
   fractions, ln age) cut the 5-NN error from 10.0 to 6.9 MPa.
2. Bridges near each other are alike, so a random split puts near-copies of each test bridge
   in training. Hold out whole spatial blocks, such as counties (Lecture 3).
3. The right singular vectors are the principal directions: the shapes in which daily profiles
   differ most from the average day. The singular values give the spread along each:
   sigma_i^2 over the sum of all sigma_j^2 is the share of variance. Today the SVD returns, as
   a way to solve least squares.
:::

## Set the cement for 35 MPa

- Water 185 kg/m³, fixed by workability. How much cement?
- Too little: weak concrete. Too much: cost and carbon.

![Scatter of 58 tests at 28 days with water between 175 and 195 kilograms per cubic meter and no slag or fly ash: strength rises with cement; the tests span 200 to 528 kilograms of cement and 12.2 to 56.8 MPa. The four attempt mixes, A to D, are ringed; a dashed line marks 35 MPa.](cee_l05_least_squares/cement_question.png)

::: notes
Frame the problem by the cost of each error before accuracy. Lecture 2 asked whether one
proposed mix would pass; today we design the mix, so the model must be something we can solve
for an input. A weighted sum is the simplest such model.
f'c = 35 MPa at 28 days. Too little cement: weak concrete in place, then cores, repair, or
removal; the owner and the public carry that risk. Too much: cost and carbon in every cubic
meter (Lecture 2). Same data as Lecture 2: 1,030 tests, 425 at 28 days; designs held out, the
same 30 splits. Today's tool: a weighted sum of features, fitted by least squares, that we can
solve for the cement.
:::

## Your turn: predict the water effect

Exercise 1/5, **Predict** (3 min, alone, then pairs). Water 185 kg/m³, no admixtures; only
the cement changes.

1. In a model of strength, what sign will water's weight have? Why?
2. Estimate the proposed mix P at 28 days. Does it reach 35 MPa?

| Mix | Cement (kg/m³) | Water (kg/m³) | w/c | MPa at 28 days |
|:--|--:|--:|--:|--:|
| A | 397 | 185 | 0.47 | 39.1 |
| B | 339 | 185 | 0.55 | 31.6 |
| C | 295 | 185 | 0.63 | 25.2 |
| D | 252 | 185 | 0.73 | 19.7 |
| P | 375 | 186 | 0.50 | ? |

::: notes
Today's exercise runs through the lecture, 25 minutes in groups of three with one laptop per
group: Predict 3, Derive 5, Implement 7, Delegate 7, Debrief 3. Students commit on the
worksheet in the notebook. The four mixes are one laboratory series with no slag, fly ash, or
superplasticizer.
Answers, kept for later: (1) negative. More water at the same binder raises w/c, and Abrams'
law (1918) says strength falls exponentially as w/c rises. Water is constant in this table, so
the sign comes from physics, not from these rows; the fitted model confirms it in the Build
step. (2) Between A (0.47, 39.1) and B (0.55, 31.6), straight-line interpolation at 0.50 gives
36.3 MPa. Measured: 38.2 MPa. Do not reveal it yet; the next slide traces P.
All five are 28-day tests; A to D are in the first split's training set, P is a held-out
design. Coarse and fine aggregate vary a little (1,038 to 1,111 and 734 to 784 kg/m³).
:::

## A weighted sum, fitted once

![Diagram of the course pipeline with the decoder highlighted. A mix x goes to an encoder that computes dimensionless groups, giving z; the decoder is a weighted sum w0 plus w-transpose z, whose 7 weights least squares fits once, by QR, from the training set of 746 tests of 236 designs; the prediction y-hat is compared with the measured strength by the squared-error loss. Below, the proposed mix P, a held-out design: cement 375, water 186, coarse aggregate 1038, fine aggregate 758 kilograms per cubic meter, no slag, fly ash, or superplasticizer, tested at 28 days. Its groups: w/b 0.50, aggregate over binder 4.79, ln age 3.33, the other three zero. The decoder adds w0 = 54.0, a w/b term of minus 48.0, an aggregate term of plus 6.0, and an age term of plus 27.8, for a prediction of 39.8 MPa. Measured: 38.2 MPa, an error of plus 1.6 MPa, against f'c = 35 MPa.](cee_l05_least_squares/ls_pipeline.png)

::: notes
Same encoder as Lecture 2; a new decoder. k-NN stored 746 tests; least squares keeps 7 numbers.
Each term is a weight times a group, so you can read where the prediction comes from: water
costs 48.0 MPa here, curing to 28 days adds 27.8. The weights come from the first
design-held-out split; P is not in it.
:::

## Your turn: derive the normal equations

- **Assumption:** strength is a weighted sum of the features, plus noise that is independent
  from test to test and has the same spread for every mix.
- Abrams' law in logs is such a sum: $\ln f_c=a+b\,(w/c)$, with $A=e^{a}$ and $B=e^{-b}$.
- Stack the four mixes: $X$ has rows $(1,\ w/c)$; $\mathbf{y}$ holds $\ln f_c$.

**Exercise 2/5, Derive** (5 min, groups of three). Write
$\mathcal{L}(\mathbf{w})=\lVert X\mathbf{w}-\mathbf{y}\rVert^2$ and set its gradient to zero.
Why solve with QR or the SVD rather than forming $(X^\top X)^{-1}$?

::: notes
State the assumption before the method; the Break-it step tests it. With Gaussian noise of
constant variance, maximizing the likelihood is exactly least squares (Forsyth, Section
10.2.2); other noise gives other losses.
Answer: grad L = 2 X^T (X w - y) = 0 gives X^T X w = X^T y. The Hessian 2 X^T X is positive
semidefinite, so this is a global minimum. Why not the inverse: forming X^T X squares the
condition number, and the inverse does not exist when a column depends on the others. The next
two slides show both.
:::

## Four mixes, solved by hand

Centred, the columns are orthogonal: $b=\sum\Delta x\,\Delta y/\sum\Delta x^2=-2.65$. P: 36.0 MPa.

![Fitted line through the four mixes on a logarithmic strength axis against water-cement ratio: A at 0.47 and 39.1 MPa as a blue circle, B at 0.55 and 31.6 as a vermillion square, C at 0.63 and 25.2 as a green triangle, D at 0.73 and 19.7 as a purple diamond. The line ln f = 4.91 minus 2.65 times w/c passes almost through all four. P at w/c 0.50 is a hollow circle on the line at 36.0 MPa, just above a dashed line at 35 MPa.](cee_l05_least_squares/attempt_fit.png)

::: notes
Values as the table shows them: x = w/c to two decimals, y = ln of the strength to one
decimal. Centered x: -0.125, -0.045, +0.035, +0.135; Delta y = ln f - 3.332. The slope is
-0.0983 / 0.0371 = -2.65. The intercept is ybar = 3.332 at the centered origin, so
a = 3.332 + 2.65 x 0.595 = 4.91. Lecture 2's constants come from all 425 tests at 28 days.
The largest residual in ln f is 0.012: on one lab's series, Abrams' law is nearly exact.
Back to the attempt: the interpolation (36.3) and the fit (36.0) agree; P measured 38.2.
For 35 MPa: w/c = (ln 35 - 4.91) / (-2.65) = 0.511, and cement = 185 / 0.511 = 362 kg/m³.
Zero gradient: X^T X w = X^T y. With Delta x = w/c - 0.595 the columns are orthogonal. A = 135.5
MPa, B = 14.2 (Lecture 2: 123.8, 16.5). For 35 MPa: w/c at most 0.51, 362 kg/m³.
:::

## The fit sits at the bottom of a bowl

![Two 3D surfaces of the four mixes' squared error in ln f over the intercept and the slope b. Raw columns, 1 and w/c: a long, narrow valley running diagonally, its floor tilted, the minimum marked at intercept 4.91, slope minus 2.65. Centred columns, 1 and w/c minus 0.595: the same valley lies along the slope axis, minimum at intercept 3.33, slope minus 2.65. Contours on the floor show the same shapes.](cee_l05_least_squares/bowl.png)

::: notes
The loss L(w) = ||Xw - y||^2 is a quadratic: a bowl whose bottom is where the gradient is zero,
the normal equations. Its shape is X^T X. Raw, the two columns are nearly parallel (w/c varies
little around 0.6), so intercept and slope trade off: a long valley on the diagonal (ratio of
curvatures 198). Centred, the columns are orthogonal: the valley lies along an axis, and each
weight is found on its own, which is why the hand solution decoupled. The valley stays long
(ratio 108) because the four w/c values span only 0.47 to 0.73: the slope is the poorly
determined direction. Lecture 13 walks gradient descent down such valleys.
:::

## Least squares is a projection

Normal equations: $X^\top(\mathbf{y}-X\hat{\mathbf{w}})=\mathbf{0}$. The residual is perpendicular to every column.

![Diagram in 3D: two column vectors of X span a shaded plane; the target vector y rises above it; its shadow on the plane, X w-hat, is the fitted prediction; the dashed residual y minus X w-hat drops straight down to the plane, with a right-angle mark.](cee_l05_least_squares/projection.png)

::: notes
Each prediction X w is a point of the plane spanned by the columns. The closest point to y is
its perpendicular shadow, so the residual is orthogonal to every column: X^T (y - X w-hat) = 0,
the normal equations again, now as geometry. The four mixes live in four dimensions; the
picture is three, to see it. A dependent column adds no new direction to the plane: the
shadow stays the same, but the weights that reach it are no longer unique (two slides on).
:::

## Solve with QR or SVD, not the inverse

- $\kappa(X^\top X)=\kappa(X)^2$: raw kg/m³, $10^{5}\to10^{10}$.
- **QR:** $R\mathbf{w}=Q^\top\mathbf{y}$, and $\kappa(R)=\kappa(X)$.
- **SVD:** $\mathbf{w}=V\Sigma^{+}U^\top\mathbf{y}$; a zero $\sigma$ is skipped.

::: notes
Rule of thumb: a solve can lose about log10(kappa) digits, so forming X^T X doubles the digits
at risk. QR and the SVD work on X itself and are backward stable; solving the normal equations
is not (Trefethen & Bau, 1997, Parts II and III). np.linalg.lstsq and scikit-learn's
LinearRegression use the SVD. The Lecture 2 groups: kappa(X) = 640, kappa(X^T X) = 4.0e5.
Raw kg/m³ with an intercept: kappa(X) = 1.0e5, so kappa(X^T X) = 1.0e10. QR: X = QR with Q^T Q = I,
a triangular solve. SVD: X = U Sigma V^T; a dependent column has sigma = 0, Sigma-plus skips it
and returns the minimum-norm weights. Centering by hand did what QR does for any number of
columns: it made them orthogonal.
:::

## Ridge shrinks the shaky directions

$$\mathbf{w}_\lambda=(X^\top X+\lambda I)^{-1}X^\top\mathbf{y}$$

Ridge shrinks both; lasso drops one. **Standardize first.**

![Contour ellipses of the squared error over the weights of standardized w/b and binder, tilted because the two are correlated, r = minus 0.89. Least squares at their centre, minus 6.21 and 4.72. As the penalty grows, the ridge path curves smoothly to the origin, shrinking both weights; the lasso path runs straight to the w/b axis, sets binder's weight to exactly zero, then shrinks w/b to zero.](cee_l05_least_squares/paths.png)

::: notes
Derivation: the gradient 2 X^T (X w - y) + 2 lambda w = 0 gives (X^T X + lambda I) w = X^T y;
adding lambda I lifts every eigenvalue of X^T X by lambda. Bayesian view: a Gaussian prior on w
gives ridge with lambda = sigma^2 / tau^2. Choose lambda by cross-validation with designs held
out. Here ridge does not lower the error (7.55 MPa with lambda chosen inside each split,
median 3.16; the 2 splits that pick the grid's smallest penalty are at the least-squares
limit): at lambda = 3.16 it shrinks even the weakest direction of the six groups by only 4.8%.
Its value today is stability, which the Break-it step shows. At its cross-validated penalty,
lasso drops w/b on the first split (it keeps 13 of 14) but on only 12 of 30 splits; the
dropped set changes from split to split (22 different sets, 8 to 14 features kept). With
correlated features, which one lasso keeps is unstable (Zou & Hastie, 2005).
Penalize large weights: minimize ||Xw - y||^2 + lambda ||w||^2. Unique for every lambda > 0, even
with a dependent column. Along singular direction i the fit shrinks by sigma_i^2 / (sigma_i^2 +
lambda): directions the data barely span shrink most (Hoerl & Kennard, 1970). Standardize first,
so the penalty does not depend on units; leave the intercept out. Lasso, lambda ||w||_1, sets
weak weights to exactly zero (Tibshirani, 1996). Offered 8 raw inputs and 6 groups, it admits
w/b first, then ln age. The figure: all 1,030 tests, strength centred.
:::

## Your turn: implement and check

**Exercise 3/5, Implement** (7 min, one laptop per group).

- Fit the six-group model to the 746 training tests of the first split in about ten lines of
  NumPy.
- Compare with scikit-learn's `LinearRegression` using `np.allclose` at a tolerance you state.
  The check is part of the answer.

**Predict first:**

1. Your QR solution against scikit-learn at `rtol=1e-8`: pass or fail?
2. The textbook formula, `inv(X.T @ X) @ X.T @ y`, on raw kg/m³, where
   $\kappa(X^\top X)=10^{10}$: pass or fail?

::: notes
Take a show of hands on each. Most students expect the second to fail after the previous slide.
The next slide answers both.
:::

## The check is part of the answer

```python
from scipy.linalg import solve_triangular
def fit_least_squares(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weights [w0, w] minimizing ||[1 Z] w - y||^2, via QR."""
    A = np.column_stack([np.ones(len(Z)), Z])   # intercept column
    Q, R = np.linalg.qr(A)                      # A = QR, Q^T Q = I
    assert Q.shape == A.shape and R.shape == (A.shape[1],) * 2
    return solve_triangular(R, Q.T @ y)         # R w = Q^T y
w = fit_least_squares(Z_tr, y_tr)               # 746 tests, 6 groups
ref = LinearRegression().fit(Z_tr, y_tr)
assert np.allclose(w, [ref.intercept_, *ref.coef_], rtol=1e-8, atol=0)
```

**Both pass:** QR to 4e-14, the inverse to 6e-12. Water's weight is negative, as
predicted: −96.83 MPa per unit $w/b$.

::: notes
Relative differences, max over the weights, against scikit-learn: groups QR 4e-14, SVD 4e-14,
inverse 1e-12; raw kilograms QR 1e-13, SVD 2e-13, inverse 6e-12. The squared condition number
cost the inverse about two digits here, not ten, because the bound is a worst case. Ask: which
principle justifies trusting the code now? The check against an independent implementation, at
a stated tolerance. Then delegate to the library.
The 6e-12 is the inverse on raw kg/m³. A bound is not a forecast; the inverse fails when a
column depends on others (Break it).
Read the weights: in raw units water is -0.16 MPa per kg/m³; ln age +8.34 MPa per unit
(Lecture 2's median gain: 8.7); test RMSE on this split 7.82 MPa.
:::

## Predict before we test

1. **Features.** Least squares on raw kg/m³ and days, on raw kg/m³ with $\ln$ age, and on the
   Lecture 2 groups. Rank the errors, designs held out.
2. **A dependent column.** Add binder content (cement + slag + fly ash) as a ninth input. What
   happens to the weights, and to the predictions?
3. **Beyond the data.** Train on mixes with at most 350 kg/m³ of cement; predict those
   above 400. Too high or too low?
4. **Units.** Ridge with one $\lambda$, age in days or in years. Same prediction?

::: notes
One minute; students commit in writing. Each of the next four slides reveals one answer, and
each tests a piece of the assumption: a weighted sum of these features (1), weights that each
mean something (2), the same sum outside the data (3), and a penalty that treats features
alike (4).
:::

## Inside the data, log age does the work

Raw 10.6 → with $\ln$ age **7.4** MPa; Lecture 2's groups 7.5; 5-NN 6.9.

![Dot plot of test RMSE in MPa, designs held out, with one-standard-deviation error bars over 30 splits. Least squares on raw kilograms and days: 10.6. On raw kilograms with ln age: 7.4. On the Lecture 2 groups: 7.5. A hollow square marks 5-NN on the groups, from Lecture 2: 6.9. A dotted line marks 17.0 for predicting the mean; a dashed line marks 1.4 for the scatter of repeat tests.](cee_l05_least_squares/features_rmse.png)

::: notes
Same 30 splits as Lecture 2, designs held out; error bars are one sample standard deviation
over the 30 splits. Strength
grows linearly in ln(age), not in days, so a weighted sum of days cannot follow it: the
largest step, 10.6 to 7.4. Leakage matters less here than for k-NN: a random split flatters the
groups model by 0.2 MPa (7.38 against 7.55), against 1.2 MPa for 5-NN. Least squares has
little to memorize: on its own training rows its error is 7.38, barely below the 7.55 on new
designs. The floor is Lecture 2's: the median standard deviation of repeat
tests (one mix, one age, different strengths). It rests on only 9 groups from 3 mixes, so it is
rough.
Above: predicting the mean, 17.0. Below: repeat tests, 1.4 (rough). Here the ratios add nothing to
ln age; the break-it step tests them beyond the data.
:::

## A dependent column breaks the inverse

Binder = cement + slag + fly ash: rank 9 of 10. Inverse: wrong signs. SVD, ridge: 10.6 MPa.

![Strip plot of the cement weight in MPa per kilogram per cubic meter over 30 splits. Eight inputs fitted by QR: tightly between 0.10 and 0.15. With binder added and the explicit inverse: scattered on both sides of zero. With binder and the SVD: tightly between 0.04 and 0.05. With binder and ridge: also 0.04 to 0.05.](cee_l05_least_squares/collinear_weights.png)

::: notes
The digits of the inverse and of QR here are round-off: they change with the BLAS, the
machine, even the order of the columns. What holds everywhere: kappa ~ 1e16 means the smallest
singular value sits at round-off; the inverse's cement weight takes both signs over the 30
splits, and its test RMSE (tens of MPa) is worse than predicting the mean (17.0). QR is
unpivoted: its weights, ~1e12, cancel only because test rows obey the same sum. The SVD gives
exactly the 8-input model's error, 10.6 MPa, and ridge 10.6 too; no single weight means
anything now.
The identity cement + slag + fly ash - binder = 0 holds exactly in every row, so X has a null
direction: any multiple of (1, 1, 1, -1) can be added to those four weights without changing a
prediction. The SVD picks the shortest weight vector; ridge picks a unique one for any
lambda > 0 (here 1, standardized). Both split cement's effect between cement (0.04) and binder.
Real data have near-dependencies too: the seven ingredients add up to the mass of a cubic meter
of concrete, 2,195 to 2,551 kg. Report predictions, not single weights, when inputs depend on
each other.
kappa(X) ~ 1e16. Inverse: the cement weight flips sign; RMSE worse than predicting the mean. QR:
weights ~1e12 that cancel. SVD and ridge: stable, 10.6 MPa, as before.
:::

## A copy of a column flattens the bowl

![Two 3D loss surfaces over two weights for the 425 28-day tests, ln f against standardized w/b entered twice. Left, least squares: a trough with a flat floor, drawn in orange, along which every pair of weights summing to minus 0.33 fits equally well; the minimum-norm answer, both weights minus 0.164, is marked. Right, ridge with lambda 425: a round bowl with one bottom, both weights minus 0.11.](cee_l05_least_squares/trough.png)

::: notes
The binder column is the sum of three others; the simplest such case is one column entered
twice. Then only w1 + w2 matters: the loss is constant along a line, a trough with a flat floor,
and X^T X has a zero eigenvalue, so the inverse does not exist. The SVD returns the point of the
floor nearest the origin (minimum norm). Ridge adds lambda ||w||^2, a bowl of its own, which
lifts the floor everywhere except at one point: one answer, shrunk toward zero (each weight
-0.11 instead of -0.164, the factor 2n/(2n + lambda) = 2/3).
:::

## Ratios hold up beyond the data

Train cement ≤ 350; test > 400. Raw kg/m³ over-predicts by **4.7 MPa** (95%: 1.9 to 6.9).

![Scatter plot of predictions against measured strength for the 136 tests with more than 400 kilograms of cement, from models trained on mixes with at most 350. Squares, least squares on raw kilograms with ln age, lie mostly above the diagonal, mean error plus 4.7 MPa. Circles, least squares on the Lecture 2 groups, lie close to it, mean error minus 2.3. Triangles, 5-NN on the groups, lie below it, mean error minus 8.5.](cee_l05_least_squares/cement_extrapolation.png)

::: notes
This is the lecture's decisive figure. Mean errors over all ages: raw kg/m³ with ln age +4.7
(95% interval 1.9 to 6.9, resampling the 39 test designs), the groups -2.3, 5-NN -8.5 MPa.
The over-prediction comes from the other ages: on the 43 tests at 28 days the raw model errs
+0.4, the groups -5.5. RMSE: raw with ln age 9.9, groups 8.9, 5-NN 12.6; raw kg/m³ and days
14.6 (+6.1). The gap between the first two is within design resampling (95%: -1.2 to +3.1
MPa): a similar RMSE, but the groups' errors sit on the safe side. In range the two least-squares models were
equal (7.4 and 7.5). Extrapolation depends on the features: in kilograms these mixes are new;
in ratios most are not (82% lie inside every ratio's training range). The raw model's slope, 0.16 MPa per kg/m³, comes from low-cement mixes; Abrams'
law (28-day fit, 185 kg/m³ of water) says the gain per kg shrinks as cement grows. 5-NN
averages five lower-cement neighbours: 64 of the 136 tests measured above all five, while the
72 within their neighbours' range err by only -2.4 MPa. No test exceeds the training maximum,
81.8 MPa, so the cap of Lecture 2 is not the cause here.
774 training tests, 136 test (39 designs). Their cement is new; 92% of their w/b values are not. Abrams: +0.13
MPa per kg at 300 kg/m³, +0.09 at 500. The raw line keeps 0.16.
:::

## Unscaled ridge depends on units

One $\lambda=100$, the 600 kg/m³ mix: age in years makes the penalty bite. Standardized, units do not matter.

| Ridge, λ = 100 | Age in days | Age in years |
|:--|--:|--:|
| Not standardized | 61.1 | 63.9 |
| Standardized | 60.4 | 60.4 |

::: notes
The penalty acts on the size of each weight, and a weight's size depends on its feature's
units. Without standardizing, a feature measured in large numbers (days, kilograms) is barely
penalized and one measured in small numbers (years, ratios) is penalized hard. Standardizing
makes lambda mean the same for every feature; it is fitted on training rows only. With
scikit-learn's default lambda = 1, unscaled ridge on kg/m³ and days is least squares in
disguise.
All 1,030 tests, one lambda = 100, the 600 kg/m3 mix of the next exercise. In days the penalty
is negligible: 61.1 MPa, the same as least squares. In years, age's weight is 365 times larger,
so the penalty bites: ridge keeps 22% of it. Standardized, the answer is the same in any units.
:::

## Your turn: delegate, then audit

**Exercise 4/5, Delegate** (7 min, groups of three). Ask an AI assistant:

> "With the UCI concrete data, fit a ridge regression and predict the 28-day strength of:
> cement 600 kg/m³, water 165, superplasticizer 10, coarse 1,000, fine 650, no slag or fly ash."

**Audit:**

1. Did it warn about extrapolation? Tested cement stops at 540 kg/m³.
2. Did it standardize before penalizing, and leave the intercept out?
3. Is the prediction physically plausible, given part 1?

::: notes
Audit key, all 1,030 tests: least squares on raw kg/m³ and days 61.1 MPa, with ln age 65.5,
on the groups 59.9, 5-NN 53.9. In ratios the mix is inside the data: w/c 0.275, tested w/b
0.24 to 0.90; aggregate/b 2.75, tested 2.38 to 9.85. 42 28-day tests (35 mixes) with w/b within
0.03 of 0.275 reached 33.7 to 80.0 MPa, so any answer in that band is plausible, and an answer
that falls as cement rises, or has a positive water weight, is not. Unscaled ridge at the
default lambda = 1 gives the least-squares answer, 61.1. The prompt is the exercise's; students
may vary it, and should record the assistant's code and answer on the worksheet.
:::

## Uncertainty costs cement

Line: 35 MPa at 337 kg/m³. 90% reach it at 44.0 MPa: **418** kg/m³ (+24%).

![Predicted 28-day strength against cement content for the proposed mix's family, water 186 kilograms per cubic meter, from 250 to 540 kilograms. The solid prediction curve rises from about 19 to 53 MPa and crosses 35 MPa at 337 kilograms, marked by a circle. The dashed curve, 9.0 MPa lower, crosses 35 MPa at 418 kilograms, marked by a square. Grey dots show tested 28-day mixes with water near 185 kilograms, scattered around the solid curve.](cee_l05_least_squares/cement_margin.png)

::: notes
Of the 3,179 held-out 28-day predictions, 57% were above the measured strength (58% of the
425 tests, averaging each test's predictions), so a mix designed to the line fails more often
than it passes. In practice the margin comes from the standard deviation of the producer's
own test records (ACI 301), not from many laboratories' mixes.
The margin is the 90th percentile of prediction minus measured strength over 3,179 held-out
predictions of the 425 distinct 28-day tests, pooled over the 30 splits (each test is held out
in several). If this mix errs like those, a prediction of 44.0 MPa
leaves about 10% of such mixes below 35. The margin is itself uncertain: each split's own
90th percentile runs from 5.0 to 11.6 MPa (median 8.8); Lecture 10 returns to it. The four-mix fit gave 362 kg/m³ for 35 MPa: one lab's
series, with almost no scatter. Design codes already make a mix target an average strength
above f'c (Lecture 2); this is the same idea, priced in cement.
The verified model, P's family (water 186 kg/m³). 57% of held-out predictions were above the
measured strength; 90% were at most 9.0 MPa above. A producer's own records set its margin.
:::

## Use the line inside the data only

| Do not | Because |
|:------------------------------|:------------------------------------|
| predict beyond tested ratios and ages | check every feature's range, not only the inputs' |
| use it for new materials | other cements, admixtures, aggregates are not in these tests |
| read single weights of dependent inputs | report predictions |
| use one margin for all mixes | scatter grows with strength: 6.5 MPa below 30, 11.1 above 50 |
| accept concrete with it | cylinder tests accept concrete (Lecture 2) |

::: notes
When not to use the method is part of the method. The scatter figures are standard
deviations of the held-out 28-day errors for predictions below 30 MPa and above 50 MPa. A log
target does not even it out here (6.4 MPa below 30, 11.6 above 50) and raised the in-range
error to 9.01 MPa (the notebook tries it). Lecture 10 calibrates intervals; Lecture 11 gives them coverage guarantees.
:::

## The engineer signs, not the assistant

AI fluency: "effectively, reflectively, and responsibly" (Campus AI Curriculum Task Force).

| Audit | Today's check |
|:----------------|:------------------------------------------|
| Range | tested cement stops at 540 kg/m³; the query asks 600 |
| Units | standardize before the penalty; intercept unpenalized |
| Sign | water's weight negative, as Abrams says |
| Accountability | the engineer of record signs the mix design |

::: notes
This is the ethical and economic side of today's decision. The quotation is from the Task
Force's page at the Office of the Provost (University of Illinois). Ask: who would carry the
cost of the 61.1 MPa answer if the 600 kg/m³ mix had been built from it unchecked?
A confident number arrives in seconds; whether it checked range, units, and sign is what today's
audit found out. An unaudited answer can cut cement below what is safe, or add cement no one
needs. Both costs fall on people who never saw the model.
:::

## Vote: how much cement for 35 MPa?

$f^{\prime}_c=35$ MPa at 28 days; 186 kg/m³ of water, no admixtures; materials like those
tested.

- **A.** 337 kg/m³: where the fitted line reaches 35 MPa.
- **B.** 418 kg/m³: where the line minus 9.0 MPa, its 90th-percentile over-prediction, reaches 35 MPa.
- **C.** 540 kg/m³: the most cement ever tested, to be safe.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
A ignores the scatter: 57% of held-out 28-day predictions were above the measured strength. C is safe but
not free: 203 kg/m³ more cement than A (w/c 0.34), with its cost and carbon, and it sits at
the edge of the tested range. B prices the uncertainty the data show (w/c 0.44), and trial
batches confirm it before production. Then ask what would change the answer: the producer's
own records (ACI 301 sets the margin from them), a cheaper wrong "yes", or new materials.
:::

## Least squares lives inside modern ML

| Where | Today's problem, inside | Lecture |
|:----------------|:------------------------------------------|:----------|
| Last layer | a regression network ends in a weighted sum on squared error | 14 |
| Linear probes | a linear model on a frozen foundation model's embeddings (Radford et al., 2021) | 15, 17–19 |
| Weight decay | the ridge penalty in training; for Adam, AdamW (Loshchilov & Hutter, 2019) | 13–14 |
| At scale | gradient descent on the same loss, instead of QR | 13 |

::: notes
Lectures 15 and 17-19 fit such probes on the frozen features of site photos and documents.
Weight decay as the ridge penalty in network training goes back to Krogh and Hertz (1991). The point
for students: the closed form they derived today keeps reappearing inside the
largest models, as a last layer, as a probe that measures what a representation knows, and as
a penalty. For a regression target on frozen embeddings, the probe is ridge regression, solved
exactly as today. Loshchilov and Hutter show that L2 regularization and weight decay are
equivalent for SGD (up to the learning rate) but not for Adam. When N and D are too large for
a QR factorization, stochastic gradient descent minimizes the same loss (Lecture 13).
:::

## Debrief: one-minute paper

**Exercise 5/5, Debrief** (3 min, alone, on paper). Complete:

> "One thing the AI did that I would not have caught without step 3."

Step 3 is Implement. Responses seed the next lecture's warm-up.

**In the notebook** (`cee_l05_least_squares.ipynb`): the group worksheet, least squares by QR
and SVD checked against scikit-learn, the four breaks, ridge and lasso paths, and the cement
decision.

::: notes
This is the exit question. Read the papers before the next lecture; the most common answers
become its warm-up. Expected answers: no warning about the cement range; ridge without
standardizing; a prediction quoted to a decimal with no error bar; a model on raw kilograms.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- The loss is a bowl; solve with QR or the SVD.
- A dependent column: a trough. Ridge: a bowl.
- $\ln$ age and ratios carry the physics.
- Delegate only what you can check.

**Next:** linear classifiers, when a miss costs more than a false alarm.
:::
::: {.column}
![Four thumbnails of the lecture's key figures: the loss bowl raw and centred; a column entered twice making a trough, and ridge making it a bowl; test error by features; predictions beyond the tested cement.](cee_l05_least_squares/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Abrams (1918). *Design of Concrete Mixtures*. Lewis Institute, Bulletin 1.
- ACI Committee 301 (2020). *Specifications for Concrete Construction* (ACI 301-20).
- Campus AI Curriculum Task Force (2026). Office of the Provost, University of Illinois.
- Forsyth (2019). *Applied Machine Learning*. Springer, Ch. 10–11.
- Hoerl & Kennard (1970). *Technometrics* 12(1).
:::
::: {.column}
- Krogh & Hertz (1991). *NeurIPS* 4.
- Loshchilov & Hutter (2019). *ICLR*.
- Radford et al. (2021). *ICML*, PMLR 139.
- Tibshirani (1996). *JRSS B* 58(1).
- Trefethen & Bau (1997). *Numerical Linear Algebra*. SIAM.
- Yeh (1998). *Cement and Concrete Research* 28(12).
- Zou & Hastie (2005). *JRSS B* 67(2).
:::
::::::
