---
title: Fairness and Accountability
subtitle: "Lecture 27 · Does Lecture 6's bridge model serve Illinois's vulnerable counties as well as the rest?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 6–26

On your own, no notes, two minutes:

1. Lecture 6: where did the flagging threshold come from?
2. Lecture 6: whose bridges did the model catch least, by owner?
3. Lecture 26: what did the MDP do when being poor cost nothing?

**Today:** the same bridge model, asked whom it serves.

::: notes
Give the answers before moving on.
1. From the costs: a miss costs 20 false alarms, so flag when p > 1/21.
2. Townships': 75% of their bridges that turned poor were caught, against 82% to 87% for the state,
   counties, and cities. Today the groups are counties, ranked by social vulnerability.
3. It never replaced anything: the optimizer did what the reward said. Today the question is what
   "fair" should say.
:::

## Does the model serve every county?

![Illinois's 102 counties, each drawn at the mean position of its bridges and sized by how many it has, coloured by the CDC/ATSDR Social Vulnerability Index 2022 quartile within Illinois, from low to high; Cook County (Chicago), in the top quartile, is the largest dot.](cee_l27_fairness/county_map.png)

::: notes
Lecture 6's classifier flags bridges that are not poor today but may be poor in four years, so they
get preventive work and a closer look. Its test: the 2020 bridges, judged by the 2024 inventory
(19,637 bridges, 793 turned poor). The Social Vulnerability Index ranks each county on 16 census
factors (poverty, no vehicle, disability, language, housing, and more); the top quartile (ranks
above 0.75) is the most vulnerable. The model never sees the index. Source: CDC/ATSDR SVI 2022
Database, Illinois; its use here does not imply endorsement by CDC, ATSDR, HHS, or the U.S.
Government; the data are available at no charge from ATSDR.
:::

## Your turn: catch and false-alarm rates

| SVI quartile, 26 counties each | Turned poor | Caught | Stayed healthy | Healthy, flagged |
|:-------------------------------|------------:|-------:|---------------:|--------:|
| Low | 164 | 130 | 3,329 | 854 |
| High | 252 | 202 | 5,911 | 1,936 |

::: notes
Two minutes: compute each quartile's catch rate and false-alarm rate. Which difference matters?
Catch rate (true-positive rate): 130 / 164 = 79% low, 202 / 252 = 80% high. False alarms: 854 /
3,329 = 26% low, 1,936 / 5,911 = 33% high. The high quartile gets more flags it did not need. Is
that unfair to it? Keep your answer; the lecture asks what a flag means here.
:::

## Five things called bias

| Bias in … | Means | Here |
|:----------|:------|:-----|
| ML | a model too simple for the truth | a linear score (Lecture 6) |
| Statistics | an estimate off on average | a loss scored on the training years |
| Data | training unlike use | 2014-2018 trains, 2020-2024 tests |
| Thinking | confirmation, automation bias | trusting the flag list |
| Algorithms | unfair treatment of groups | today's question |

::: notes
CS 441's list, on our problem. The same word means five things; today is about the last, and the
first four feed it. "Protected attributes" in law (race, sex, age, religion) are not in the
inventory; the vulnerability index stands in for communities that a bridge program can underserve.
:::

## Where bias enters

![The pipeline: inspections, the inventory (NBI), the model (Lecture 6), the flag (p above 1/21), preventive work.](cee_l27_fairness/picture_1.png)

::: notes
The one picture, three frames. Frame 1: how the model reaches a bridge. Each box is a place where a
choice is made, by people or by code.
:::

## Each step can tilt it

![The same pipeline with a note under each step: labels are an inspector's rating, a proxy for need; selection decides which bridges are in the inventory; proxies are inputs that stand for the county; the threshold decides who is caught and who is missed; and a flag brings resources.](cee_l27_fairness/picture_2.png)

::: notes
Frame 2. Labels: "poor" is an inspector's rating, a proxy for need; it does not count the traffic a
bridge carries or the detour if it closes. Selection: the inventory holds public highway bridges
over 20 feet; we left culverts out. Proxies: inputs that encode the county. The threshold: one
number trades misses for false alarms. And the decision's meaning: here a flag brings resources. CS
441's proxy example: a health algorithm predicted cost as a stand-in for need, and spent less on
patients with the same need but less access (Obermeyer et al., 2019).
:::

## And where we check it

![The same pipeline and notes, with a band beneath: evaluate each group separately, with intervals; publish a model card and a datasheet.](cee_l27_fairness/picture_3.png)

::: notes
Frame 3: the checks this lecture builds. Disaggregated evaluation: every metric per group, with an
interval. Documentation: a model card for the model, a datasheet for the data.
:::

## Unawareness is not enough

![ROC curves of telling a bridge in a high-vulnerability county from Lecture 6's 16 inputs alone: AUC 0.70 with all of them, 0.65 without latitude and longitude.](cee_l27_fairness/proxies.png)

::: notes
The model never sees the index, but its inputs carry part of it: a logistic regression trained on
the 2014 bridges tells a high-vulnerability county from the rest with AUC 0.697 on the 2020 bridges,
and still 0.646 without location (traffic and material alone reach 0.644). On counties held out of
training (ten folds of counties, ten shuffles), the AUC is 0.599, and 0.572 without location: about
half of the signal above chance carries to a county the model never saw; the rest is recognizing
these counties. For a model reused on the same counties every cycle, the first number is the one
that matters. Leaving the attribute out does not keep it out; and to measure a gap at all, you need
the attribute. CS 441: Amazon's résumé screener learned "women's" from the words.
:::

## Three ways to call a model fair

| Criterion | Requires, for every group |
|:----------|:--------------------------|
| Demographic parity | the same flag rate |
| Equalized odds | the same catch rate (TPR) and false-alarm rate (FPR) |
| Calibration | P(poor ∣ p) = p: a 10% bridge turns poor 10% of the time |

::: notes
Hardt et al. (2016) for equalized odds; "equal opportunity" is its half: equal catch rates only.
Each says something different about a group, and they can conflict, as the next slides show.
:::

## Which error is the harm here?

| | COMPAS (CS 441) | Bridges (today) |
|:--|:--|:--|
| A flag means | "high risk": detention | preventive work, a closer look |
| A false alarm | harms the person | spends money; helps the county |
| A miss | releases a risk | a bridge turns poor unwatched |
| The rate that matters most | false positives | misses: catch rate |

::: notes
The same confusion matrix, opposite meanings. In criminal risk scores, a flag is a burden, so
false-positive rates by group matter most (ProPublica, 2016). Here a flag brings resources, so the
harm is a miss, and the criterion that matters is equal opportunity: equal catch rates among the
bridges that will turn poor. The high quartile's extra false alarms are extra attention.
:::

## Your turn: is the gap real?

On the test years, the model catches 85.1% of the bridges that turned poor in the low-mid quartile
and 72.7% in the mid-high quartile.

Predict: does this gap mean the model serves mid-high counties worse? What would convince you?

::: notes
Two minutes, written. Ask how many bridges stand behind each rate, and how much the rate would move
with other counties or another year.
:::

## Catch rates differ; intervals overlap

![By SVI quartile on the test years, with 95% intervals that resample whole counties: flagged 26% to 35%; caught 72.7% (interval 58.8 to 85.4) to 85.1% (78.8 to 91.3); false alarms 24.5% to 32.8%; and calibration: mean predicted chance against the share that turned poor, 5.2% against 3.6% in the low-mid quartile and 5.7% against 4.1% in the high quartile, about equal in the other two.](cee_l27_fairness/disaggregated.png)

::: notes
Every interval resamples counties, not bridges: bridges in one county share inspectors, budgets, and
weather. The catch rates' intervals overlap widely; 176 bridges turned poor in the mid-high
quartile. Flag rates (demographic parity) differ by up to 8 points, false alarms by about as much.
The model over-predicts this cycle as a whole (5.0% predicted, 4.0% turned poor; the cycle before,
4.5% and 4.5%), most in the low-mid and high quartiles: 1.6 points each (intervals 0.5 to 2.7 and
0.1 to 2.6).
:::

## Last cycle, the quartiles were equal

| SVI quartile | Caught, validation (2016-2020) | Caught, test (2020-2024) |
|:-------------|-------------------------------:|-------------------------:|
| Low | 79.8% | 79.3% |
| Low-mid | 80.8% | 85.1% |
| Mid-high | 79.5% | 72.7% |
| High | 80.6% | 80.2% |

::: notes
The same model, the cycle before: every quartile between 79.5% and 80.8%, around the 80.3% overall.
The test year's spread is consistent with a noisy year: each quartile's last-cycle rate lies inside
its test interval. The bracket for a disparity is its own year-to-year noise.
:::

## One county drives the false alarms

![The top quartile split in two. False alarms: Cook County 53.7%, the other 25 counties 27.5%. Calibration: in Cook, 9.3% predicted against 5.4% that turned poor; in the other 25, 4.8% against 3.8%.](cee_l27_fairness/cook.png)

::: notes
Cook County holds 1,246 of the top quartile's bridges (20%), and the model flags 53.7% of its
healthy ones; without Cook, the quartile looks like the rest (27.5%). The quartile was never one
community: Chicago's county, with heavy traffic and city and state bridges, sits in it beside 25
other counties, from Peoria and Winnebago (Rockford) to rural southern ones. A county-level index
also hides the differences inside a county: a group label is an average, the ecological fallacy.
Calibration is not a fixed trait either: the cycle before, Cook was under-predicted (100.7 expected,
116 turned poor; this cycle 115.5 expected, 67), while its false alarms were again about twice the
rest's (46.2% against 24.6%).
:::

## Equal precision forces unequal errors

![False-alarm rate against base rate, for a catch rate of 0.80 and precision 0.09, 0.11, and 0.13: curves rising from the origin. The four quartiles sit near them: high and low-mid near precision 0.09, mid-high near 0.11, low near 0.13.](cee_l27_fairness/identity.png)

::: notes
Chouldechova (2017): FPR = p / (1 − p) × (1 − PPV) / PPV × TPR, an identity of the confusion matrix.
It holds exactly in every quartile here (high: 32.75% computed, 32.75% from the identity). If base
rates p differ, equal precision and equal catch rates force unequal false alarms; Kleinberg et al.
(2017) prove the general version for calibrated scores. Which criterion to give up is a value
judgement, not a technical one. CS 441's COMPAS case is the same arithmetic.
:::

## Thresholds per quartile fix nothing here

![Catch rate by quartile on the test years: one threshold (1/21) against a threshold per quartile chosen on the validation years to catch 80.3% in each. Low 79 and 79; low-mid 85 and 83; mid-high 73 and 73; high 80 and 78.](cee_l27_fairness/thresholds.png)

::: notes
Post-processing (Hardt et al., 2016): pick each group's threshold so its catch rate meets a target.
Chosen on the validation years, the four thresholds land between 0.0474 and 0.0509, next to 1/21,
because the quartiles were already equal there. On the test years they change almost nothing: 5,978
flags instead of 6,176, 621 caught instead of 631, an expected cost of 0.448 against 0.447 per
bridge. A gap that is noise cannot be thresholded away; a real one, measured over several cycles,
could be, and using the group in the decision is then a policy choice that needs its own approval.
:::

## Your turn: split it further

The township bridges in high-vulnerability counties: how many turned poor on the test years? How
wide is the interval of their catch rate?

::: notes
Intersectional evaluation (CS 441): crossing groups finds the worst-served cell, and shrinks every
cell. Guess before the reveal.
:::

## Small cells, wide intervals

![Catch rate by SVI quartile and owner, with the number of bridges that turned poor in each cell: from 15 to 104; township bridges in high-vulnerability counties, 65% of 55; city bridges in mid-high counties, 10 of 16.](cee_l27_fairness/intersection.png)

::: notes
Township bridges in the top quartile: 55 turned poor, 65.5% caught, interval 49.3% to 82.9%. 5 of
the 16 cells hold fewer than 30 bridges that turned poor. Most cells this small have intervals more
than 20 points wide: they can neither confirm nor rule out a gap that size. The remedy is more
cycles of data, reported with intervals, not a smaller cell.
:::

## Where to intervene

| Stage | Change | On bridges |
|:------|:-------|:-----------|
| Data | collect, reweight, better labels | inspect where ratings are thin |
| Training | a fairness penalty, adversarial | catch rates as a constraint |
| Decision | thresholds per group | tried today: nothing to fix |
| Always | evaluate each group, with intervals | every cycle, published |

::: notes
CS 441's three stages. Adversarial learning (Zhang et al., 2018) trains a representation from which
an adversary cannot recover the group. The fourth row is not optional: without disaggregated
evaluation none of the first three can be checked.
:::

## A model card for Lecture 6's model

| Field | Entry |
|:------|:------|
| Use | flag non-poor Illinois bridges for preventive work, 4 years ahead |
| Not for | moving budgets between counties; ranking communities |
| Data | NBI Illinois, trained 2014-2018, tested 2020-2024 |
| Overall | catches 79.6% of 793; false alarms 29.4% |
| By quartile | caught 72.7% to 85.1%; intervals overlap |
| Caveats | labels are ratings; Cook's calibration swings by cycle |

::: notes
Mitchell et al. (2019): a short, standard report that travels with a model: its intended use, its
data, its metrics overall and by group, and its limits. The numbers above are today's.
:::

## A datasheet for the inventory

| Question | National Bridge Inventory, Illinois |
|:---------|:------------------------------------|
| Who made it, why | FHWA and states: bridge safety oversight |
| What is in it | public highway bridges over 20 ft |
| How collected | inspections every 2 or 4 years (Item 91) |
| Who labels | state and local inspectors' ratings |
| Known limits | ratings subjective; culverts left out here |

::: notes
Gebru et al. (2021): datasheets for datasets ask who, why, what, how, and with what limits. Every
lecture that used the NBI (1, 3, 6, 13, 26, today) inherits these answers.
:::

## Vote: what should IDOT require?

Before the flag list guides preventive work, the model must show …

- **A.** equal flag rates in every SVI quartile.
- **B.** equal catch rates, within their intervals, every cycle.
- **C.** calibration in every quartile and in Cook County.
- **D.** good overall accuracy; groups are not its job.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction. A (parity) would cut flags in the high quartile, where a flag is help. B fits a
decision whose harm is a miss; it is met today within noise, and should be checked every cycle. C
would expose Cook's calibration, which swings between cycles; its over-prediction this cycle spends
money but harms no bridge. D is what produced the
questions. B, with C reported, is defensible; A misreads what a flag means here.
:::

## Unbiased, fair, and just

| Goal | Asks | Here |
|:-----|:-----|:-----|
| Unbiased | is the estimate right on average? | is p calibrated? |
| Fair | are groups treated alike? | equal catch rates |
| Just | does each get what they need? | more for the worse off? |

::: notes
CS 441's three goals (after Perona). They pull apart: a just program might inspect vulnerable
counties more, which is unequal treatment by design. Equal outcome or equal opportunity is a choice
society makes attribute by attribute; the model's builder should state which one the program wants.
:::

## Where it lives today

| Practice | What it is |
|:---------|:-----------|
| Model cards | on Hugging Face's model pages (Qwen, Lectures 19 and 25) |
| Datasheets | dataset cards, datasheets in public data portals |
| Group testing | NIST's face recognition tests, by demographic group |
| Risk frameworks | NIST AI RMF 1.0: measure and manage bias |

::: notes
Grother et al. (2019, NISTIR 8280) measured false matches by demographic group across face
recognition algorithms: disaggregated evaluation at national scale. The NIST AI Risk Management
Framework 1.0 (2023) lists harmful bias among the risks to measure and manage.
:::

## Exit question

A city says its new pavement-repair model is "fair: it is equally accurate in every ward". What
would you ask?

**In the notebook** (`cee_l27_fairness.ipynb`): the join, the metrics, county-resampled intervals,
the identity, and per-group thresholds.

::: notes
Any two of: accurate by which measure, and is the harm here a miss or a false alarm?; with what
interval, resampling what?; the same in last year's data?; how many repairs were needed in each ward
(base rates)?; are wards the right groups, or do they hide neighbourhoods?; what do the labels
measure: complaints, inspections, or condition?
:::

## Things to remember

:::::: {.columns}
::: {.column}
- Bias enters at labels, data, proxies, thresholds.
- Leaving the group out does not keep it out.
- Evaluate each group, with intervals, every cycle.
- Know which error is the harm.
- Fairness criteria conflict: choose, and say so.

**Next:** AI as infrastructure.
:::
::: {.column}
![Thumbnails of today's figures: rates by quartile, Cook County, the identity, and the owner-by-quartile cells.](cee_l27_fairness/remember.png)
:::
::::::

::: notes
CS 441's take-aways, on CEE data: bias is the expected outcome unless countered; be wary of
decisions about people; model cards and intersectional evaluation disclose limits; unbiased, fair,
and just conflict. Some slides follow the structure of Derek Hoiem's CS 441 lecture on data bias and
fair ML.
:::

## References

:::::: {.columns}
::: {.column}
- Mitchell et al. (2019). Model cards. *FAT\*.*
- Gebru et al. (2021). Datasheets. *CACM* 64(12).
- Hardt et al. (2016). Equality of opportunity. *NeurIPS.*
- Chouldechova (2017). *Big Data* 5(2).
- Kleinberg et al. (2017). *ITCS.*
- Obermeyer et al. (2019). *Science* 366.
:::
::: {.column}
- CDC/ATSDR (2024). SVI 2022 Database, Illinois.
- FHWA. National Bridge Inventory, 2014-2024.
- Grother et al. (2019). NISTIR 8280.
- NIST (2023). AI RMF 1.0.
- Zhang et al. (2018). Adversarial debiasing. *AIES.*
:::
::::::
