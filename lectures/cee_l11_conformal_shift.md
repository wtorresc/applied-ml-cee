---
title: Conformal Prediction and Shift
subtitle: "Lecture 11 · How sure is next week's river forecast?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 7–10

On your own, no notes, two minutes:

1. The 5% strength line was calibrated over all mixes, yet 13.5% of strong mixes fell below
   it. What is the general lesson?
2. RANSAC kept the calm line until the burst made up 65% of the window. What did it assume
   about the readings?
3. Glisan's breakdown threshold misfired at Stark/Washington. What differed between the data
   it learned from and the data it was used on?

**Today:** how sure is next week's river forecast, decades after it was built?

::: notes
Give the answers before moving on; recall helps most with feedback.
1. Calibration on average can hide miscalibration by subgroup: check coverage where the risk is
   (Lecture 10). Today: the weeks when the river rises.
2. That the calm readings were the largest tight group, a kind of majority (Lecture 9). Today's
   method assumes the future looks like the calibration data.
3. The relation between occupancy and congestion, and the station's traffic (Lecture 7): the
   use was not like the training. Today we name that: shift.
:::

## Plan the intake with a range

- Lake Decatur's supply: next week's flow, with a 90% range.
- Built in 1980; used ever since.

![Daily flow of the Sangamon River at Monticello on a log scale, 1908 to 2025, from the 0.01 cubic meters per second floor, where the zero-flow days are drawn, to a peak of 529.5; shaded bands mark the fit period, 1950 to 1969, the calibration period, 1970 to 1979, and the test period, 2000 to 2025.](cee_l11_conformal_shift/record.png)

::: notes
USGS daily mean discharge, public domain, read from the USGS Water Data API (the legacy
service closes in February 2027). 567 days are missing, all in 1912–1914. The 59 zero-flow days,
and all 135 days below 0.01 m3/s, are raised to 0.01 m3/s before the log. The forecast: log
flow seven days ahead from the log flows of today, 1, 3, and 7 days ago, their 30-day mean, and
the season, by least squares (Lecture 5). Fit on 7,305 days, calibrated on 3,652.
The Sangamon River feeds Lake Decatur, Decatur's water supply. The decision: plan withdrawals and
releases for a 90% range of next week's flow. Data: USGS gauge 05572000, Sangamon River at
Monticello (550 mi²), daily flow since 1908: 42,499 days. The setting: built in 1980. Fit on
1950-1969, calibrated on 1970-1979, used ever since.
:::

## Your turn: build the interval

| Forecast | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|:------------------|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Error factor (times off) | 1.2 | 1.5 | 1.1 | 2.0 | 1.3 | 3.1 | 1.4 | 1.8 | 2.4 |

::: notes
Nine forecasts from a calibration year, each with its error factor: how many times too high or
too low the flow came. Two minutes on your own: a tenth forecast is made. Which factor would you use so that, with
90% chance, the flow lands within the forecast times or divided by that factor? And for 80%?
Answers: sort the factors; with n = 9, take the ceil((9 + 1) x 0.9) = 9th smallest, 3.1, for
90%; the 8th, 2.4, for 80%. The reason: if the tenth forecast is like the nine, its error is
equally likely to rank anywhere among the ten. That is conformal prediction. The nine factors
are illustrative; the real calibration set has 3,652.
:::

## Calibration turns forecasts into ranges

![Diagram of the course pipeline with the conformal step highlighted. Past flows go to an encoder of lags and season, giving z; the decoder, a line fitted on 1950 to 1969, forecasts the log of the flow in 7 days; the conformal step, calibrated on 1970 to 1979, adds plus or minus q-hat, giving a 90% interval. Below, a forecast issued June 3, 2024, for June 10: today's flow is 10.48 cubic meters per second; the forecast is 9.31; q-hat is the 3,288th of 3,652 calibration scores, 1.277, a factor of 3.586, so the interval runs from 2.60 to 33.4; the flow that came, 10.31, was inside.](cee_l11_conformal_shift/conformal_pipeline.png)

::: notes
The forecast is on the log scale, so the interval is multiplicative: forecast times or divided
by exp(1.28) = 3.6. A wide range, honest about a week ahead without rain forecasts. The
question for the rest of the lecture: does "90%" still mean 90% in 2024?
:::

## The guarantee rests on exchangeability

**Assumption:** calibration and new cases are exchangeable.
$$1-\alpha\le P\big(s_{n+1}\le\hat{q}\big)\le1-\alpha+\tfrac{1}{n+1}$$

![Histogram of the 3,652 calibration scores of the 1970s, the absolute errors in log flow: most, 59.3%, below 0.5, a long tail to 3.44, where the plot stops at the 99.5th percentile. The 90% below the cutoff q-hat = 1.28 are blue; the 10% above, hatched orange. The cutoff makes the interval times or divided by 3.6.](cee_l11_conformal_shift/scores.png)

::: notes
Proof: by exchangeability, s_{n+1} is equally likely to be the 1st, 2nd, ..., (n+1)-th smallest
of the n + 1 scores (ties aside). It is at most q-hat exactly when its rank is at most
ceil((n + 1)(1 - alpha)), which has probability ceil((n + 1)(1 - alpha))/(n + 1) >= 1 - alpha;
the upper bound holds without ties (Vovk, Gammerman & Shafer, 2005; Lei et al., 2018). Here n =
3,652, so the guarantee is 90.0% to 90.03%. The guarantee is marginal: averaged over cases, not
for each case or each kind of case.
Scores s_i = |ln y_i - ln y-hat_i| on n calibration cases; q-hat is the ceil((n + 1)(1 - alpha))-
th smallest. The new score's rank among n + 1 is uniform: no model, no distribution assumed. The
model is fixed before calibration (split conformal).
:::

## When the future differs

- **Covariate shift:** inputs change, the relation does not. Reweight calibration scores by
  $w(x)=p_{\text{new}}(x)/p_{\text{cal}}(x)$ (Tibshirani et al., 2019).
- **Nonstationarity:** the relation itself drifts. Adapt the level online (Gibbs & Candès,
  2021):
  $$\alpha_{t+1}=\alpha_t+\gamma\,(\alpha-\mathrm{err}_t)$$
- A miss lowers $\alpha_t$ (wider); a hit raises it. Long-run coverage returns to $1-\alpha$.

::: notes
"Stationarity is dead" (Milly et al., 2008): river records are no longer a fixed distribution
to sample from, because of climate, land use, and drainage. Adaptive conformal inference
guarantees |average error - alpha| <= (max(alpha_1, 1 - alpha_1) + gamma)/(gamma T) over T
steps, whatever the data do; it gives no guarantee for any single period. Here gamma = 0.005,
and a forecast is judged seven days after it is issued: with feedback that late, alpha_t can
overshoot its limits by about 7 gamma, which loosens the bound slightly; the average still
returns to alpha.
:::

## Your turn: predict the check

```python
def conformal_quantile(scores, alpha=0.10):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))   # rank of the cutoff
    if k > n:
        return float("inf")                   # too few scores
    return float(np.sort(scores)[k - 1])
q = conformal_quantile(np.abs(y_cal - pred_cal))
covered = np.abs(y_test - pred_test) <= q
```

**Predict:** split the 1970s into two halves by week, 200 times; calibrate on one, test on the
other. Mean coverage? Spread?

::: notes
Weeks, not days, are held out together, because days in a week share a storm. Take a show of
hands for the mean (90%?) and for the 5th to 95th percentile of the 200 coverages.
:::

## Exchangeable halves: 90.1%

Mean coverage **90.1%** over 200 splits; 5th–95th percentile, 86.2% to 93.2%. Width ×/÷ **3.6**.

![Histogram of coverage on the held-out half over 200 random halvings of the 1970s weeks: centred on 90.1%, close to the dashed 90% target, the middle 90% of splits between 86.2 and 93.2%.](cee_l11_conformal_shift/halves.png)

::: notes
The guarantee is about the average over calibration sets and new cases; one split can be
off by about four points. Halving by day instead, which ignores the days' order, gives 88.5% to
91.6%: days within a few weeks are correlated, so the effective sample is smaller than 1,826
days. The width bracket: climatology ignores today's flow;
persistence uses only today's; the model adds the recent history and the season.
Verified: mean coverage 90.1% over 200 splits; 5th to 95th percentile, 86.2% to 93.2%. Interval
width, 90%, a week ahead: times or divided by 3.6 with the model. Above: no model (the 1950-1969
median), times or divided by 13.8; persistence (next week = today), 4.4.
:::

## Predict before we test

1. **Later decades.** The 1970s interval, used from 1980 to 2025. Coverage by decade?
2. **One year at a time.** The worst year since 2000?
3. **Rising weeks.** Weeks when the flow rises 2.7-fold or more: coverage?
4. **By month.** Calibrate a separate interval for each month (Mondrian). Fixed?

::: notes
One minute; students commit in writing. Each of the next slides reveals one answer: coverage
across decades (1), the variance of coverage (2), conditional coverage (3), and a common fix (4).
:::

## Later decades fall short

Fixed interval since 2000: **87.9%**. Adaptive conformal from 1980: **89.8%**.

![Line plot of the share of forecasts inside the 90% interval by the decade the forecast is issued, 1900 to 2020. The fixed interval, calibrated in the 1970s, is near 90% only in the fit and calibration decades, 1950 to 1979, between 87.0 and 89.3% in the four full decades since, and between 88.8 and 89.0% from 1910 to 1949. Adaptive conformal, from 1980, stays between 89.2 and 90.6%. Open markers show the decades the record covers only in part: 1908 to 1909, at 91.8%, and 2020 to 2025, at 89.2% fixed and 90.1% adaptive. A dotted line marks 90%.](cee_l11_conformal_shift/coverage_decades.png)

::: notes
Why short? The 1950s and 1960s are the fit data and the 1970s the calibration data, so those
three decades sit near 90% by construction. The 1970s were also calm: the 90% score of each fit
or calibration decade is 1.26 to 1.28, against 1.33 to 1.47 in every decade from 1910 to 1949
and from 1980 on. Calibrated on any other full decade outside the fit years, the interval covers
89.3% to 91.2% of the rest; calibrated on the 1970s, 88.4%, the lowest. The shortfall says as
much about the calm calibration decade as about a changing river. The spread of the halves (86.2%
to 93.2%) is no yardstick here: it is for five-year tests with half the calibration scores, and
26 years against all 3,652 vary far less; recalibrating on the other decades (89.3% to 91.2%) is
the fair comparison. Two points means about seven extra misses
a year. Adaptive conformal widens the interval after misses and narrows it after hits.
The fixed interval: 87.5% in the 1980s, 87.0% in the 2010s; 87.9% for 2000-2025. Earlier decades
also fall short: 88.8% to 89.0% from 1910 to 1949. Adaptive conformal: 89.2% to 90.6% by
decade. Open markers: the 1900s (1908-1909, 668 forecasts) and the 2020s (2,185) are partial.
:::

## One year: 75% inside

Year by year since 2000: **75%** (2014) to 93%. 90% is a long-run rate.

![Bar chart of the share of forecasts inside the fixed 90% interval in each year from 2000 to 2025: the other years between 83 and 93%, with 2014 lowest, at 75%, marked in orange; a dashed line marks 90%.](cee_l11_conformal_shift/years.png)

::: notes
Coverage over a year averages 365 correlated outcomes: storms cluster, and a wet year produces
many rises. Report the long-run coverage and its year-to-year spread together.
A planner who lived through 2014 saw one forecast in four miss. The yardstick for one year:
within a decade, a year covered by the interval calibrated on the other nine ranges from 84.4%
to 96.2% in the 1970s, and from 79.7% to 98.4% over 87 years in nine decades. 2014's 75% is
below all of them.
:::

## 90% on average, 28% on rises

Rising 2.7-fold or more: **28.1%** inside. Steady: 99.7%. The weeks that matter miss.

![Horizontal bar chart of coverage of the 90% interval, 2000 to 2025, by what the river does in the forecast week: steady, 99.7%, 49% of forecasts; falling, 96.1%, 29%; rising 1.6- to 2.7-fold, 93.6%, 8%; low flow now, 81.1%, 8%; rising 2.7-fold or more, 28.1%, 14% of forecasts. A dotted line marks 90%.](cee_l11_conformal_shift/coverage_states.png)

::: notes
This is the lecture's decisive figure. The guarantee is marginal; it says nothing for a subgroup
defined by the outcome. A rise is caused by rain the model never sees: no calibration on
these inputs can single out the rises. Lecture 10 found the same with strong mixes.
Weeks when the river rises 2.7-fold or more are 14% of daily forecasts. Steady (49%): 99.7%.
Falling (29%): 96.1%. Rising 1.6-2.7-fold (8%): 93.6%. Low flow now (8%): 81.1%, a number that
rests on the 0.01 m3/s floor for zero flows: at 0.03 or 0.1 it would be 84.0% or 89.7%, while
all weeks (87.9% to 88.5%) and rises (28.0% to 28.1%) barely move.
:::

## The interval chases the flood

![Time series of the Sangamon's daily flow, March to early July 2013, on a log scale, with the 90% interval issued seven days earlier as a shaded band. The band follows the flow on recessions; at each sharp rise, in mid April, early June, and late June, the flow jumps above the band, marked with crosses, and the band widens only a week later.](cee_l11_conformal_shift/example_spring.png)

::: notes
Spring 2013: 19 of 122 forecasts missed; 18 of the misses were rises. The April 19 peak,
203 m3/s, came above the interval issued for it a week before. After a flood, the interval is
wide when the river is already falling. The forecast uses only past flows, so it learns of a
storm a week late. Forecasts of rising
rivers need precipitation forecasts and a hydrologic model: the National Weather Service's river
forecasts do this. Conformal prediction can then calibrate those forecasts' intervals.
:::

## Calibrating by month does not help

One interval per month (Mondrian): **87.6%** overall; months 74% to 95%; rises 29.0%.

![Coverage by month, 2000 to 2025, for one interval for all months and for one interval per month calibrated on that month's 1970s scores: both wander between about 75 and 95%, October lowest for both, near 78% and 74%; neither stays on the dotted 90% line.](cee_l11_conformal_shift/months.png)

::: notes
Mondrian conformal guarantees coverage within each group when groups are exchangeable between
calibration and use; here each month's 1970s scores are fewer (about 300) and the decades
differ. Grouping by the outcome (rises) is impossible at forecast time. October is the worst
month for the fixed interval (78%).
Grouping by a known input fixes coverage only for that group, and only without drift.
:::

## Adapt the level, add the rain

Adaptive conformal holds 90% over time. For rises: a forecast with rain in its inputs.

![Three-year rolling coverage from the end of 1982 to 2025: the fixed 1970s interval wanders between 81.4 and 93.2%, lowest in 1991 and 2015; adaptive conformal stays within 87.9 to 92.1%, around the dotted 90% line.](cee_l11_conformal_shift/aci.png)

::: notes
The decision for the planners: an adaptive interval for routine planning, and the National
Weather Service's rain-aware river forecast when storms are coming. Recalibrate after changes
in the watershed or the reservoir's operation.
Adaptive conformal keeps the long-run 90% (89.8% since 2000). For rises: use a forecast with rain
in its inputs, then calibrate its intervals the same way. Report coverage by decade, by year,
and by river state, not one number.
:::

## A range promises only cases like these

| Do not promise | Because |
|:------------------------------|:------------------------------------|
| after the world changes | land use, drainage, climate, dams: recalibrate or adapt |
| for a subgroup | unless calibrated for it; never for groups defined by the outcome |
| for a single year | 90% is a long-run rate |
| with few calibration cases | n = 9 gives the largest score; fewer than 9, no 90% interval |

::: notes
When not to use the method is part of the method. With n calibration scores, the finest level
is 1/(n + 1); at n = 9 the 90% interval is the largest score, and fewer than 9 cannot give 90%.
:::

## Stationarity is dead

| Assumed | Met |
|:--------------------------|:--------------------------------------------|
| the future draws from the past's distribution (Milly et al., 2008) | a different river |
| 1970s statistics for floods, droughts, supply | 87.9% coverage since 2000 |
| one coverage number | 28% on rising weeks: misses fall downstream |

::: notes
Milly and colleagues argued in Science that climate change and human disturbance have ended
stationarity as a default assumption in water management. Ask: who should be told that the
interval's 90% is 28% on rising weeks, and how?
:::

## Vote: which range for Decatur?

Planners will use next week's range for withdrawals and flood preparation.

- **A.** The 1970s interval: calibrated, ×/÷ 3.6.
- **B.** Adaptive conformal on the same forecast.
- **C.** Adaptive conformal, plus a rain-aware forecast when storms come.
- **D.** A 99% interval instead, to be safe.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
C: B restores the long-run 90% (89.8%) but covers 39.0% of rises; A fell to 87.9%; D covers
89.3% of rises only with a range of ×/÷15.5, wider than no model at all (×/÷13.8). Then ask what would change the answer: a dam
upstream, or a decade of drought.
:::

## Conformal runs modern AI

| Where | Today's idea, inside | Lecture |
|:----------------|:------------------------------------------|:----------|
| Language models | conformal sets of answers with a guaranteed chance of a correct one (Quach et al., 2024) | 19 |
| Quantile networks | conformalized quantile regression (Romano et al., 2019) | 10 |
| Deployed models | weighted and adaptive conformal (Tibshirani et al., 2019; Gibbs & Candès, 2021) | today |
| A field guide | Angelopoulos & Bates (2023) | — |

::: notes
Conformal language modeling samples answers until a calibrated stopping rule says the set
contains an acceptable one with the promised probability. The point: any model, however large,
can get a guarantee of this kind, and the same exchangeability assumption limits it.
:::

## Exit question

Your 90% interval covered 90% of the 1970s and 88% since. A colleague proposes widening it to
95% to "fix" the shortfall. What does that fix, what does it not, and what would you do instead?

**In the notebook** (`cee_l11_conformal_shift.ipynb`): split conformal from scratch,
exchangeable halves, coverage by decade, year, and river state, adaptive conformal, and
Mondrian.

::: notes
Answer: a 95% interval lifts rises from 28.1% to 62.8%, and all weeks to 94.5%, by widening
every week's range from ×/÷3.6 to ×/÷6.5; steady weeks were already at 99.7%. Rises stay
far below 95% (a 99% interval reaches 89.3%, at ×/÷15.5), and every week's range is less
useful. Instead: adaptive conformal for the long-run rate, a rain-aware forecast for rises, and
coverage reported by river state. Collect the answers; they seed the next warm-up.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- Conformal: a rank, if exchangeable.
- The guarantee is marginal: 28% on rises.
- Calm 1970s, 87.9% after 2000.
- Adapt the level; add the inputs.

**Next:** Gaussian processes and surrogates: emulating a water network.
:::
::: {.column}
![Four thumbnails of the lecture's key figures: the calibration scores and their 90% cutoff; coverage by decade; coverage by river state; rolling coverage of the fixed and adaptive intervals.](cee_l11_conformal_shift/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Angelopoulos & Bates (2023). *Found. Trends ML* 16(4).
- Gibbs & Candès (2021). *NeurIPS*.
- Lei et al. (2018). *JASA* 113.
- Milly et al. (2008). *Science* 319.
- Quach et al. (2024). *ICLR*.
:::
::: {.column}
- Romano, Patterson & Candès (2019). *NeurIPS*.
- Tibshirani, Barber, Candès & Ramdas (2019). *NeurIPS*.
- U.S. Geological Survey (2026). Water Data API, site 05572000.
- Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World*. Springer.
:::
::::::
