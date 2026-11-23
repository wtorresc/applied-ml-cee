---
title: Mixtures, EM, and Naive Bayes
subtitle: "Lecture 7 · When does the freeway break down?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 4–6

On your own, no notes, two minutes:

1. $k$-means gives each building to its nearest center. What does that assume about the
   clusters' shapes and sizes?
2. Least squares is maximum likelihood under which noise model?
3. Logistic regression models $P(\text{poor}\mid\mathbf{x})$ directly. What else could you model
   instead, and use Bayes' rule?

**Today:** at what occupancy does a freeway break down from free flow to congestion?

::: notes
Give the answers before moving on; recall helps most with feedback.
1. Round clusters of the same size: hard assignments by distance alone (Lecture 4). Today's
   mixtures give each cluster its own spread and weight, and assign softly.
2. Independent normal errors of one variance (Lecture 5). Today EM maximizes a likelihood too,
   with a hidden variable.
3. How x is distributed in each class, p(x | class), and the class shares; Bayes' rule turns
   them into P(class | x). Generative against discriminative (Ng & Jordan, 2002).
:::

## Meter the ramp before it breaks

- Hold **occupancy** just below breakdown.
- Too low: ramp queues. Too high: stop-and-go.

![One day at Glisan, October 6, 2011: occupancy in percent and speed in mph through the day. Occupancy is about 1% at night and 6.7 to 9.4% from 8:00 to 14:00, with speed near 57 mph. Speed drops below 45 mph at 14:40, shaded as congestion until 18:25, the last congested record; occupancy passes 20% at 15:20, and speed bottoms out at 17 mph at 17:30.](cee_l07_mixtures_bayes/day.png)

::: notes
ALINEA, the classic local ramp-metering rule, adjusts the ramp's rate each cycle to hold the
measured downstream occupancy at a target near the critical occupancy (Papageorgiou,
Hadj-Salem & Blosseville, 1991). Data: the USDOT Data Capture and Management test data set
(Portland State University, 2012; CC BY-SA 3.0), three lanes per station. A record is kept when
at least 40 of its 45 lane readings are present. Its volume sums only the readings present, so
the 996 records with 40 to 44 are scaled up by 45/readings. Flow is vehicles per hour per lane.
Occupancy is the share of time a loop is covered. The decision: the target occupancy at each
station. Data: I-205 southbound, Portland, two stations, 20-second dual loops, Sept 15 to Nov 14,
2011: 17,036 five-minute records at Glisan, 17,154 at Stark/Washington. Most loops measure only
volume and occupancy; these also measure speed: a check on what we learn without it.
:::

## Your turn: free or congested?

| Glisan SB, Oct 6, 2011 | 14:00 | 14:30 | 14:45 | 15:35 | 18:20 | 18:35 |
|:------------------|----:|----:|----:|----:|----:|----:|
| Occupancy (%) | 10.6 | 11.1 | 16.8 | 24.9 | 24.4 | 6.0 |
| Flow (veh/h/lane) | 1,550 | 1,620 | 1,460 | 1,620 | 1,400 | 1,150 |
| Speed (mph) | ? | ? | ? | ? | ? | ? |

::: notes
Two minutes on your own: which records are congested? Where did the freeway break down? Write
the rule you used. Which row would you not trust alone?
Answers: speeds 55, 51, 30, 25, 23, 55 mph. Breakdown between 14:30 and 14:45; recovery by
18:35. Flow cannot tell: 1,620 vehicles per hour per lane at 14:30 (free) and at 15:35
(congested). Occupancy can: about 11% free, about 25% congested. The rule you wrote is a
threshold on occupancy. Today we find it without the speeds.
:::

## Bayes' rule decodes the regime

![Diagram of the course pipeline with the decoder highlighted. A loop reading goes to an encoder that takes its 5-minute occupancy x; the decoder applies Bayes' rule to a mixture of two normal densities, pi-k times N of x with mean mu-k and spread sigma-k, giving the probability of congestion given x; a ramp meter turns on if that probability exceeds 0.5. Above, 17,036 unlabeled readings and the negative log-likelihood fix the mixture by EM. Below, Glisan southbound at 14:45 on October 6, 2011: occupancy 16.8% and flow 1,460 vehicles per hour per lane. The free regime has weight 0.90, mean 4.4 plus or minus 3.0%, and pi times N of 1.8 times 10 to the minus 5; the congested regime has weight 0.10, mean 23.8 plus or minus 4.8%, and 2.9 times 10 to the minus 3. The probability of congestion is 0.99: meter on. The measured speed was 30 mph.](cee_l07_mixtures_bayes/mixture_pipeline.png)

::: notes
This is the 14:45 record of your attempt. The two products are proportional to the joint
probability of the regime and the reading; dividing by their sum gives the posterior. The
congested regime is rarer (10%) but explains 16.8% far better. Nothing here used a speed.
:::

## Model how readings are made

**Assumption:** each reading comes from one of $K$ hidden regimes, each with its own density.

$$p(x)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(x;\mu_k,\sigma_k^2),\qquad P(k\mid x)=\frac{\pi_k\,\mathcal{N}(x;\mu_k,\sigma_k^2)}{p(x)}$$

- **Generative:** model $p(x\mid k)$ and $\pi_k$, then Bayes' rule. Lecture 6 modeled
  $P(k\mid x)$ directly.
- **Naive Bayes**, several inputs, labels known:
  $P(k\mid\mathbf{x})\propto\pi_k\prod_j p(x_j\mid k)$: inputs independent **within** each class.

::: notes
A generative classifier can work without labels (the mixture) and can say how likely a reading
is at all. Naive Bayes multiplies one-dimensional densities: cheap, and with few data often
competitive (Ng & Jordan, 2002, compare it with logistic regression). Its assumption is about
conditional independence; Break it tests it on our two inputs.
:::

## Two regimes, two hills

![Two panels for Glisan, 17,036 five-minute records. Left: in 3D, the log density of a two-component Gaussian mixture over occupancy and flow: a narrow ridge, free flow, rising steeply with occupancy up to 10%, and a wide, low dome, congestion, centred at 22.5% occupancy and 1,434 vehicles per hour per lane. Right: the records, those slower than 45 mph in orange, with each component's 2-standard-deviation ellipse: the blue ellipse covers the grey free-flow arm, the orange one the orange cloud.](cee_l07_mixtures_bayes/hills.png)

::: notes
A mixture says how the readings are made: pick a regime (free with weight 0.89, congested 0.11),
then draw a reading from that regime's Gaussian. The density is the sum of the two hills; the
log scale lets the small congested hill show beside the tall free-flow one. EM fitted it without
the speeds; the orange colour is the speed label, used only to check.
:::

## A density needs a bandwidth

Without a formula, a **kernel density estimate** puts a bump on every reading:

$$\hat{f}_h(x)=\frac{1}{nh}\sum_{i=1}^{n}\phi\!\Big(\frac{x-x_i}{h}\Big)$$

- $h$ small: every reading is a peak (variance). $h$ large: the regimes blur (bias). Lecture
  3's trade-off again.
- Scott's rule as SciPy applies it: $h=\hat\sigma\,n^{-1/5}$.
- A KDE describes the data; a mixture explains it with a few regimes we can name.

::: notes
phi is the standard normal density. SciPy's "scott" bandwidth is the sample SD times n^(-1/5).
Scott's normal-reference rule for a Gaussian kernel (Scott, 1992) is 1.06 times that: the
bandwidth that minimizes the asymptotic mean integrated squared error if the data are normal.
A normal reference is too wide for multimodal data. Break it shows three bandwidths on the
speeds.
:::

## EM: guess, weigh, refit

Labels are hidden, so alternate (Dempster, Laird & Rubin, 1977):

- **E-step:** weigh each reading by its responsibility
  $r_{ik}=P(k\mid x_i)$ under the current parameters.
- **M-step:** refit each regime as a weighted fit:
  $$\mu_k=\frac{\sum_i r_{ik}x_i}{\sum_i r_{ik}},\quad\sigma_k^2=\frac{\sum_i r_{ik}(x_i-\mu_k)^2}{\sum_i r_{ik}},\quad\pi_k=\frac{1}{n}\sum_i r_{ik}$$
- Each round never lowers the likelihood; it can stop at a local optimum.

::: notes
Why the likelihood cannot fall: the log-likelihood is at least the expected complete-data
log-likelihood under the responsibilities plus their entropy (Jensen's inequality), with equality at
the current parameters; the M-step maximizes that lower bound, so the likelihood rises or stays.
k-means is the limit of EM with equal, vanishing spreads: hard responsibilities (Lecture 4).
:::

## EM, step by step

![Four panels of Glisan's occupancy histogram with the two weighted Gaussian components, free in blue and congested dashed in orange, and their sum dotted. Start: means 4.9 and 27.4%, both wide and equally weighted. Step 1: means 4.5 and 23.4%, the free component tall and narrow over the main peak, the congested one low over the shoulder near 24%. Step 3: means 4.4 and 23.7%. Converged, step 18: means 4.4 and 23.8%.](cee_l07_mixtures_bayes/em_build.png)

::: notes
The start puts the means at the 50th and 98th percentiles of occupancy, each with the overall
standard deviation and weight one half. One E-step and M-step already moves most of the way:
the responsibilities split the readings at the valley, and each component refits to its share.
The remaining 17 steps move the congested mean another 0.4 points, from 23.4 to 23.8,
and the free mean 0.1. The likelihood rises at every step.
:::

## Your turn: one EM step by hand

Occupancies (%): 3, 4, 5, 20, 24, 28. Start: $\mu=(5,\,20)$, $\sigma=(5,\,5)$, $\pi=(0.5,\,0.5)$.

1. E-step: the responsibility of the congested regime for $x=5$ and for $x=20$.
2. M-step: the new means, to one decimal.
3. Where is the boundary $P(\text{congested}\mid x)=0.5$ after the step?

::: notes
Three minutes, in pairs. Answers: r(5) = 0.011 and r(20) = 0.989 (the two densities differ by
a factor exp(15^2/50) = 90); the new means are 4.1 and 23.9, nearly the plain averages, because
the responsibilities are nearly 0 or 1; the spreads shrink to 1.3 and 3.6, and the boundary
moves to 9.6%. A few more steps settle it. The notebook checks these numbers.
:::

## Your turn: predict the check

```python
def em_gmm_1d(x, mu, sd, pi, tol=1e-10, max_steps=10_000):
    history = []
    for step in range(1, max_steps + 1):
        logd = np.log(pi) + norm.logpdf(x[:, None], mu, sd)  # n x K
        ll = logsumexp(logd, axis=1)                 # log p(x_i)
        r = np.exp(logd - ll[:, None])               # E-step
        nk = r.sum(axis=0)                           # M-step
        mu = (r * x[:, None]).sum(axis=0) / nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(axis=0) / nk)
        pi = nk / len(x)
        history.append(ll.sum())
        if step > 1 and history[-1] - history[-2] < tol:
            return mu, sd, pi, history
    raise RuntimeError("EM did not converge")
```

**Predict:** steps to converge at Glisan? Agreement with the speed labels, which EM never sees?

::: notes
Start: means at the 50th and 98th percentiles of occupancy, both spreads equal to the overall
SD, equal weights. The reference: scikit-learn's GaussianMixture from the same start, with
reg_covar = 0. The E-step works in logs: logsumexp adds the two components without the
underflow that summing their densities risks for a reading far from both. Take a show of
hands on the agreement: 80%? 90%? 99%?
:::

## No labels, 99.7% agreement

Free **4.4 ± 3.0%**, congested **23.8 ± 4.8%**. Posterior 0.5 at **13.7%**; speed labels say 13.4%.

![Glisan's occupancy histogram with the fitted free component, tall near 4%, and the congested one, low and wide near 24%; on a second axis, the posterior probability of congestion rises from 0 to 1 between about 12 and 15% occupancy, crossing 0.5 at 13.7%.](cee_l07_mixtures_bayes/posterior.png)

::: notes
The 95th percentile of flow, in 2-point occupancy bins, peaks at 1,810 vehicles per hour per
lane in the 12–14% bin: the station's capacity, at the critical occupancy. At Glisan the
unsupervised crossing (13.7%) and the labels' (13.4%) both fall in that bin; Stark/Washington
is less tidy (the decisive slide). The label-based number fits P(speed below 45 mph |
occupancy) with logistic regression and reads off the 0.5 crossing.
Verified: 18 steps; scikit-learn agrees to 2e-6. Likelihood: -61,637 to -48,881, never falling.
Weights 90% and 10%. Flow peaks at 12-14% occupancy. Above: "always free" agrees with the speed
labels 89.9% of the time; EM, 99.7%.
:::

## Predict before we test

1. **Bandwidth.** A KDE of Glisan's speeds with $h=0.2$, 1.4, and 3.9 mph. How many peaks?
2. **Starts.** EM on (occupancy, flow) from 20 random starts, at scikit-learn's default
   tolerance. Do all find the regimes?
3. **How many regimes?** BIC for $K=1$ to 10. Which $K$ wins?
4. **Another station.** Glisan's model at Stark/Washington, 0.8 miles downstream. Better or
   worse with flow added to occupancy?

::: notes
One minute; students commit in writing. Each of the next slides reveals one answer: the
density's knob (1), EM's convergence (2), components against regimes (3), and the transfer of
a threshold and of three classifiers (4).
:::

## Bandwidth decides the peaks

$h=0.2$ mph: **42** peaks. Scott's 1.4: 3. $h=3.9$: 2, valley at **41.5 mph**.

![Kernel density estimates of 5-minute speeds at Glisan with three bandwidths, from 5 to 75 mph. All three show a large peak near 58 mph and a small one near 25 mph. The narrow one, h = 0.2 mph, is spiky; Scott's rule, h = 1.4 mph, adds a small bump near 45 mph; the wide one, h = 3.9 mph, is smooth, with a valley near 41.5 mph. A dotted line marks 45 mph.](cee_l07_mixtures_bayes/kde_bandwidth.png)

::: notes
The 45 mph bump under Scott's rule is records in transition: 72.6% of the 84 records between
43 and 48 mph sit next to a change of regime (the record before or after is on the other side
of 45 mph), against 1.6% of all records. Choose h by held-out likelihood, or by the use: to separate two regimes, a wide
kernel; to see the transition, a narrower one. The trade-off is Lecture 3's, in a density.
h = 0.2 mph: 42 peaks, mostly noise. h = 1.4 (Scott): 3 peaks, 25, 45, and 58 mph. h = 3.9 mph:
2 peaks, 25 and 58 mph, valley at 41.5 mph, near the 45 mph label we used; the number of regimes
is a modeling choice.
:::

## EM can stall on a plateau

![Two panels of Glisan's records, flow against occupancy, each with two fitted components drawn as 2-standard-deviation ellipses and crosses at their means. Left, random start 3 at scikit-learn's default tolerance: stalled, both components over the free-flow arm, one ellipse stretched over everything, log-likelihood minus 180,599. Right, best of 5 starts: one narrow ellipse along the free-flow arm and one over the congested cloud, log-likelihood minus 153,939.](cee_l07_mixtures_bayes/stall.png)

::: notes
These were slow climbs, not true local optima: the default tolerance (1e-3 on the mean
log-likelihood per reading) stopped them on a plateau. EM can also converge to genuine local
optima; the defense is the same: several starts, compare likelihoods, and look at the
components. With both inputs, the best fit agrees with the speed labels for 99.2% of records.
On (occupancy, flow), 20 starts from random responsibilities (init_params="random") at the
default tolerance: 5 of 20 stop with both means near 6% occupancy, no congested regime. Their
log-likelihoods run from -180,606 to -180,069, against -153,934 for the regimes. With a tighter
tolerance all 20 find the regimes; so do all 20 from scikit-learn's default start, k-means. Run
several starts; keep the best.
:::

## Components are not regimes

BIC is lowest at $K=10$, the largest tried. Extra components split free flow, then congestion.
Choose $K$ by the decision.

![Line plot of BIC in thousands against the number of components K from 1 to 10, for EM on occupancy and flow at Glisan. BIC falls steeply from 361,261 at K = 1 to 307,985 at K = 2, labeled 2 regimes, then slowly, with one rise at K = 9, to its lowest, 288,380, at K = 10.](cee_l07_mixtures_bayes/bic.png)

::: notes
BIC = -2 log-likelihood + (parameters) log n; with 17,036 records, small improvements in fit
pay for each new component. A mixture component is a convenient density shape, not necessarily
a physical state. For K from 2 to 6 one component holds congestion, its mean between 22.5% and
23.3% occupancy, and the others split free flow (at K = 6, means from 0.7% to 7.5%). From K = 7
congestion splits too: two components lie above Glisan's 13.7% at K = 7, three from K = 8. BIC
rose once, at K = 9, and was lowest at K = 10, the largest tried. Free flow is not one Gaussian.
A meter needs two regimes and the line between them.
:::

## Each station breaks at its own occupancy

![Two scatter plots of flow in vehicles per hour per lane against occupancy, one per station, colored by the EM regime. At Glisan, free-flow records rise to a 95th percentile of 1,810 vehicles per hour per lane at 12 to 14% occupancy; congested records fall from a median of 1,540 at 15 to 20% occupancy to 1,010 at 35 to 40%; a dashed line marks 13.7%. At Stark/Washington the free branch peaks at 1,980 at 16 to 18%, the congested records stay high, a median of 1,740 at 15 to 20% and 1,600 at 35 to 40%, and the dashed line marks 15.6%.](cee_l07_mixtures_bayes/fundamental.png)

::: notes
This is the lecture's decisive figure: the fundamental diagram of traffic flow, recovered from
unlabeled data. EM, no labels: Glisan 13.7%, Stark/Washington 15.6%; the speed labels say 13.4%
and 17.7%. Flow peaks at 12-14% at Glisan and at 16-18% at Stark/Washington (1,980 vehicles
per hour per lane). At Glisan both crossings fall in the flow peak's bin; at Stark/Washington
only the labels' does. A threshold learned at one detector is a property of that detector and
its traffic: it also moves with the share of congested records, from 14.2% to 12.7% at Glisan
and from 18.0% to 14.5% at Stark/Washington as that share goes from 5% to 30%. With its own EM
threshold, Stark/Washington has 301 false alarms and misses 161 of 2,241 congested records; with
Glisan's, 738 and 67. At Stark the congested component is wider (spread 9.0 points, against 4.8
at Glisan), and EM and the labels differ by 2.1 points, against 0.2 at Glisan.
:::

## Borrowed models misfire

Adding flow at Stark/Washington: naive Bayes **754 → 1,237** false alarms; full covariance
**754 → 2,675**; logistic **829 → 613**.

![Free-flow records at Glisan, flow against occupancy, correlated at 0.97: the real 2-standard-deviation ellipse is long and tilted along the data; naive Bayes' ellipse, built as if the two were independent, is a wide upright oval that covers mostly empty space.](cee_l07_mixtures_bayes/naive.png)

::: notes
On Glisan's own records, the ones they were trained on, every model classifies 99.6 to 99.8%
correctly; the differences appear only away from home. Trained at Glisan, applied to
Stark/Washington's 14,913 free records: Glisan's EM threshold, 13.7%, gives 738 false alarms.
Naive Bayes multiplies p(occupancy | class) by p(flow | class) as if they were separate clues;
in free flow they correlate at 0.97. Modeling the correlation does not rescue it: a
full-covariance Gaussian per class, the same model without the independence assumption, raises
2,675 false alarms, more than naive Bayes' 1,237. Adding flow hurt both models of p(x | class)
and helped the model of P(class | x). Nor are naive Bayes' posteriors pushed to 0 or 1 here:
with flow, 16.3% of them lie between 0.1 and 0.9 at Stark/Washington, against 7.8% on occupancy
alone. Accuracy at Stark/Washington: naive Bayes 95.2% and 92.4%, full covariance 84.4%,
logistic regression 94.8% and 96.3%. The logistic fits use scikit-learn's default ridge penalty
(C = 1); without it, 617 false alarms on both inputs. Ng and Jordan (2002): naive Bayes
approaches its (higher) asymptotic error faster; logistic regression wins with enough data.
:::

## Set each meter from its own data

Fit EM to each station's own occupancy; target just below its 0.5 point. Refit after changes.

![Plot of the probability of congestion given occupancy, from EM, for the two stations. Glisan's curve rises steeply and crosses 0.5 at 13.7%; Stark/Washington's rises more gradually and crosses at 15.6%. Dotted lines mark the two crossings.](cee_l07_mixtures_bayes/transfer.png)

::: notes
ALINEA's target is usually set at or slightly below the critical occupancy. The posterior curve
also says how sure the classification is: at Glisan it goes from 0.1 to 0.9 within 2.9 points
of occupancy; at Stark/Washington within 8.1. A flatter curve calls for a more cautious target.
Check the target against the flow peak, and with speed where dual loops exist. Refit after changes
to lanes, ramps, or detectors.
:::

## Mixtures fail off their assumptions

| Do not | Because |
|:------------------------------|:------------------------------------|
| borrow across stations | 805 errors with a borrowed threshold, 462 with its own |
| trust it in incidents or weather | untested: no incident or weather labels here |
| feed it faulty loops | a stuck detector is a third "regime": check status flags |
| add inputs to a borrowed density | with flow: 1,237 false alarms (naive), 2,675 (full) |

::: notes
When not to use the method is part of the method. The data come with an ODOT status field and
data-quality flags; we kept records with at least 40 of 45 lane readings. Two months of 2011
cannot show seasonal or long-term changes in capacity.
:::

## Meters move the wait onto ramps

| Target | Who waits |
|:----------------|:------------------------------------------|
| too low | short trips, on the ramps, so long trips flow |
| too high | everyone, in stop-and-go |
| switched off | Twin Cities, fall 2000: all 430 meters off for a legislated study; they came back on (Cambridge Systematics, 2001) |

::: notes
The study's evaluation found metering cost-effective for the region. Ramp waits fall on
drivers who enter inside the metered area. Ask: which neighborhoods' ramps would you meter
hardest, and who should decide?
:::

## Vote: Stark/Washington's target

The Stark/Washington meter needs a target occupancy. You have no speeds there.

- **A.** 13.7%: Glisan's, verified with speed labels.
- **B.** 15.6%: EM on Stark/Washington's own occupancy.
- **C.** 20.5%: the mean of its congested regime.
- **D.** 0.5 from naive Bayes with occupancy and flow, trained at Glisan.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
B: the station's own unlabeled data put the line at 15.6%, near its flow peak (16–18%): 462
errors (301 false alarms, 161 missed), 2.1 points from the labels' 17.7%. A: 805 errors (738
false alarms, 67 missed), 4.1 points from the labels. C, the mean of Stark/Washington's
congested component, is deep in congestion, too late for a meter; D raised 1,237 false alarms.
Then ask what would change the answer: a short dual-loop survey at Stark (labels would say
17.7%), or a lane drop downstream.
:::

## Mixtures run modern AI

| Where | Today's idea, inside | Lecture |
|:----------------|:------------------------------------------|:----------|
| Mixture of experts | a softmax gate routes each token to a few experts (Shazeer et al., 2017; Jiang et al., 2024) | — |
| VAEs, diffusion | maximize EM's lower bound, with a network inside (Kingma & Welling, 2014; Ho et al., 2020) | — |
| Naive Bayes | a strong baseline for short texts (Wang & Manning, 2012) | — |
| Densities | a ratio of two densities reweights for covariate shift | 11 |

::: notes
In a mixture of experts the gate's softmax plays the role of the responsibilities; Mixtral 8x7B
routes each token to 2 of 8 experts per layer. The evidence lower bound of a variational
autoencoder is the bound in EM's proof, with the posterior replaced by an encoder network.
Wang and Manning show naive Bayes features with an SVM rival more complex text models on short
snippets. No later lecture covers mixtures of experts or naive Bayes for text; Lecture 17's
transformers are the networks Mixtral's experts sit in.
:::

## Exit question

A new station has single loops only: volume and occupancy, no speed. How would you set its
meter target, and how would you check it without speeds?

**In the notebook** (`cee_l07_mixtures_bayes.ipynb`): EM from scratch checked against
scikit-learn, the hand EM step, KDE bandwidths, restarts and BIC, and naive Bayes across
stations.

::: notes
Answer: fit a two-regime mixture to its own occupancy (several starts), take the posterior-0.5
point, check it against the peak of the flow-occupancy diagram, and refit after changes; a
short survey with probe speeds (or a temporary dual loop) can label a sample. Collect the
answers; they seed the next warm-up.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- A mixture: how readings are made.
- EM: weigh, refit; likelihood never falls.
- Several starts; $K$ by the decision.
- Away from home, flow hurt generative models.

**Next:** decision trees, forests, and boosting: pavement deterioration.
:::
::: {.column}
![Four thumbnails of the lecture's key figures: the two hills of the traffic mixture; EM's components at the start, step 1, step 3, and convergence; naive Bayes' upright ellipse against the tilted data; the posterior curves of the two stations.](cee_l07_mixtures_bayes/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Cambridge Systematics (2001). *Twin Cities Ramp Meter Evaluation*. Mn/DOT.
- Dempster, Laird & Rubin (1977). *JRSS B* 39(1).
- Ho, Jain & Abbeel (2020). *NeurIPS*.
- Jiang et al. (2024). arXiv:2401.04088.
- Kingma & Welling (2014). *ICLR*.
- Ng & Jordan (2002). *NeurIPS*.
:::
::: {.column}
- Papageorgiou, Hadj-Salem & Blosseville (1991). *Transp. Res. Rec.* 1320.
- Portland State University (2012). doi:10.21949/1504472.
- Scott (1992). *Multivariate Density Estimation*. Wiley.
- Shazeer et al. (2017). *ICLR*.
- Wang & Manning (2012). *ACL*.
:::
::::::
