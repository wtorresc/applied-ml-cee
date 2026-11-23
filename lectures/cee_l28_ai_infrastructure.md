---
title: AI as Infrastructure
subtitle: "Lecture 28 · A data centre asks to connect in Illinois: what does it draw, what does it emit, and who signs?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 9–27

On your own, no notes, two minutes:

1. Lecture 9: how did we decide a reading was a glitch, not a burst?
2. Lecture 26: what did a 7% discount rate do to the bridge policy?
3. Lecture 27: what made a false alarm cheap and a miss costly?

**Today:** AI's own footprint, and the engineer's name on the design.

::: notes
Give the answers before moving on.
1. A reading far from the robust fit, judged by the median absolute deviation, not the mean. Today
   that rule cleans an hour of grid data.
2. It made the future cheap: at a poor-condition cost of 0.10, it never replaced. Today a choice of
   factor makes the same load look three times cleaner or dirtier.
3. A flag brought resources, so a false alarm spent money while a miss left a bridge unwatched.
   Today the costs land on the grid, the river, and the air.
:::

## A data centre asks to connect

| U.S. data centres (LBNL, 2024) | Electricity | Share of U.S. use |
|:-------------------------------|------------:|------------------:|
| 2023 | 176 TWh | 4.4% |
| 2028, projected | 325 to 580 TWh | 6.7% to 12% |

::: notes
Shehabi et al. (2024), the Lawrence Berkeley National Laboratory report for DOE. Our question: a
hypothetical 300 MW campus asks to connect in Illinois. The utility's engineers, the regulator, and
the engineers who design the campus each need the same three numbers: energy, emissions, and water.
CS 441 asks what one query or one training run costs; a civil engineer asks what the grid and the
river carry.
:::

## Your turn: one campus, one year

A 300 MW campus draws its full load all year (8,760 hours).

1. How many TWh a year? What share of Illinois's 177.7 TWh of 2023 generation?
2. At 472 lb CO2 per MWh, how many tonnes of CO2?

::: notes
Two minutes, with a calculator. 300 MW × 8,760 h = 2.63 TWh, 1.5% of Illinois's generation. At 472
lb/MWh (Illinois's average over all its plants in 2023) that is 0.56 million tonnes. Keep the
number; the lecture asks whether 472 was the right factor.
:::

## The chain of infrastructure

![The chain: a model (training, queries), GPU servers, the data centre, the grid, and the power plants with their rivers and air.](cee_l28_ai_infrastructure/picture_1.png)

::: notes
The one picture, three frames. Frame 1: AI is a load on physical systems that civil and
environmental engineers design and permit: buildings, substations, transmission, generation, cooling
water.
:::

## Each link multiplies

![The same chain with what each link adds: hours of work; watts, measured, not nameplate; times PUE (cooling and losses) plus WUE, water on site; times CO2 per MWh, which factor; and the fuel mix, which sets CO2 and cooling water.](cee_l28_ai_infrastructure/picture_2.png)

::: notes
Frame 2: the accounting. Energy = power × time; the facility's energy = PUE × the IT energy; CO2 =
energy × an emission factor; water = IT energy × WUE (on site) + energy × the water the power plants
consume per MWh. Every factor is a choice an engineer makes and must defend.
:::

## And someone answers for it

![The same chain, with a band beneath: who answers are the utility's engineers, the regulator, and the engineer of record who signs the design.](cee_l28_ai_infrastructure/picture_3.png)

::: notes
Frame 3. CS 441: responsibility requires control and knowledge. The utility controls the connection;
the regulator controls the approval; the engineer of record controls, and signs, the design.
:::

## The accounting, in four lines

$$E_{\text{grid}} = P \cdot t \qquad E_{\text{IT}} = E_{\text{grid}} / \text{PUE}$$

$$\text{CO}_2 = E_{\text{grid}} \cdot \text{EF} \qquad W = E_{\text{IT}} \cdot \text{WUE} + E_{\text{grid}} \cdot \text{EWIF}$$

::: notes
PUE (power usage effectiveness): facility energy over IT energy; we assume 1.2. WUE (water usage
effectiveness): litres evaporated on site per kWh of IT energy; it depends on the cooling design, so
we sweep it from 0 to 2 L/kWh. EF: lb (or kg) of CO2 per MWh, the factor in question. EWIF: the
water the power plants consume per MWh, from the fuel mix and Macknick et al. (2012).
:::

## Measure the watts, not the label

![The course's GPU (RTX 3090 Ti) sampled every half second: about 15 W idle; 437 W while training a ResNet-18, just under its 450 W limit; 103 W while a 2-billion-parameter language model generates text.](cee_l28_ai_infrastructure/power.png)

::: notes
Build and verify: the multiplier we can measure ourselves. Training kept the GPU busy 98.8% of the
time (1,434 images a second) at 437 W. Generation kept it busy 13.7% of the time (14 tokens a
second; the library ran its slower reference kernels here) at 103 W: the nameplate would overstate
that energy fourfold. Your laptop's GPU or CPU has the same distinction: measure under the work.
:::

## This course's own footprint

![GPU energy of one full rebuild of the course's figures, by lecture: L21 1.02 kWh, L22 0.55, L20 0.47, L16 0.33, L17 0.27, L25 0.21, L19 0.18, L18 0.10; 3.1 kWh in all.](cee_l28_ai_infrastructure/ledger.png)

::: notes
8.7 GPU-hours (523 minutes, each script's own record of its last run) per rebuild of every GPU
lecture's figures. At the measured watts they come to 3.13 kWh; the runs did not overlap on the GPU.
That is 0.67 to 1.86 kg of CO2, depending on the factor, less than CS 441's 10-mile drive (4.0 kg).
The busy watts come from a loop that kept the GPU 98.8% busy, and L19 mixes fine-tuning with
generation but is counted at the busy power, so the ledger is an upper estimate; L24 runs on the CPU
and is left out. Development reran each script several times, and the rest of the workstation (its
processors, memory, fans) is not counted: we cannot read its meters. This is the estimate homework 4
asks of every team (stretch goal S5).
:::

## Your turn: which factor?

The same 2.63 TWh a year in Illinois. Which emission factor belongs in the filing?

1. Illinois's average over all its plants (472 lb/MWh).
2. The average of the grid that serves the campus (PJM or MISO).
3. What the grid emits for one more MWh, when the campus is added.

::: notes
Two minutes. Ask what question each factor answers. Illinois's plants run largely on
nuclear; the campus is served by a regional grid; and the campus is new load.
:::

## One load, five factors

![Five emission factors for the same load, lb CO2 per MWh: Illinois, all plants (2023), 472; PJM average (2024), 719; MISO average (2024), 920; PJM marginal (2024), 764, between 674 and 878 under other cleanings of the data; MISO marginal (2024), 1,312, between 1,228 and 1,312.](cee_l28_ai_infrastructure/factors.png)

::: notes
Break it: the assumption "one emission factor" fails. Illinois's own average is low because its
plants are heavily nuclear (54.9% of their 2023 generation), and a campus does not draw from
Illinois's plants alone. It draws from PJM (northern Illinois) or MISO (downstate), whose 2024
averages are 719 and 920, computed hour by hour from EIA-930 generation by fuel and eGRID's 2023 CO2
rates by fuel. The marginal factors answer the question a new load asks: 764 in PJM and 1,312 in
MISO; the brackets show how far other defensible cleanings of the same data move them.
:::

## The grid changes by hour and season

![Average CO2 per MWh by month and local hour, 2024. PJM: lighter, mostly between about 590 and 870 lb/MWh. MISO: darker, highest in January and July.](cee_l28_ai_infrastructure/heatmap.png)

::: notes
In MISO the monthly average runs from 757 lb/MWh to 1,081, the hourly average across the day from
863 to 969; in PJM, months from 629 to 808, hours from 683 to 735. For the average factor, the
months differ more than the hours. A moved load, though, changes the marginal emissions, which swing
across the day as well (the notebook measures them by hour and by month), so a flexible training run
should be scheduled on the marginal factor, not the average. Hours of the year range from 677 to
1,119 in MISO (5th to 95th percentile).
:::

## Marginal: what one more MWh emits

![Hour-to-hour change in CO2 against change in demand, 2024: 7,940 hourly changes in PJM, 8,783 in MISO. PJM: slope 764 lb/MWh against an average of 719. MISO: slope 1,312 against 920.](cee_l28_ai_infrastructure/marginal.png)

::: notes
Siler-Evans et al. (2012): regress the change in emissions on the change in load, hour to hour. When
demand rises, the plants that follow it are gas and coal, not nuclear or wind. In MISO, where coal
still follows load, the slope exceeds the average by 43%, and by 33% to 43% under every cleaning we
tried: the robust finding. PJM's is not settled. Its fuel columns miss its own net generation by more
than 5% in 473 hours (MISO's never do); without those hours the slope is 764, 6% over the average,
but other defensible cleanings give 674 to 878, from 6% under the average to 22% over. The intervals
that resample whole weeks, 731 to 790 (PJM) and 1,289 to 1,335 (MISO), cover sampling only, not that
choice. Lecture 9's rule removed one PJM glitch (demand falling 38.6 GW for an hour on 21 November):
two hourly changes, which moved the slope from 721 to 764.
:::

## The campus, by factor and by water

![Left: the 300 MW campus's CO2 a year, million tonnes: 0.56 at Illinois's average, 0.86 at PJM's, 1.10 at MISO's, 0.91 at PJM's marginal, 1.56 at MISO's marginal. Right: water a year, billion litres: 4.2 at the power plants (PJM's mix) plus 0 to 4.4 on site as WUE goes from 0 to 2 L/kWh.](cee_l28_ai_infrastructure/datacentre.png)

::: notes
CO2 spans 0.56 to 1.56 million tonnes a year for the same campus: the factor is not a detail. Water:
the power plants consume 418 gallons per MWh of PJM's mix (1.58 L/kWh; 360 and 1.36 for MISO), 4.2
billion litres a year, as much as the campus would evaporate itself at a WUE near 2. A dry-cooled
campus (WUE near 0) still drinks through its grid. Cooling-tower medians from Macknick et al.
(2012); hydropower's reservoir evaporation left out.
:::

## Vote: what should the regulator require?

A data-centre filing in Illinois must report its CO2 with …

- **A.** Illinois's average over all its plants.
- **B.** its grid's annual average (PJM or MISO).
- **C.** its grid's marginal factor, with an interval.
- **D.** nothing: emissions are the utility's business.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction. A answers a different question (what Illinois's plants emit) and understates a new
load by up to a factor of 2.8. B describes the grid the campus joins, but not what it adds. C
answers "what changes if we connect it", with its uncertainty, which in PJM includes how the data
are cleaned (674 to 878 lb/MWh); it is the defensible choice for a new load; report B beside it. D
leaves the largest number out of the filing.
:::

## Who answers for the design?

| CS 441: responsibility requires | In licensure |
|:--------------------------------|:-------------|
| Control | the work done under the engineer's direction |
| Knowledge | the engineer's detailed professional knowledge of it |
| A name on it | the seal of the engineer of record |

::: notes
CS 441 (after Coeckelbergh): we hold someone responsible when they had control and knowledge.
Illinois's Professional Engineering Practice Act of 1989 (225 ILCS 325, Section 4) ties "direct
supervision/responsible charge" to the engineer's control of the work, or detailed professional
knowledge of it (paraphrased; read the current text at ilga.gov). A model's output becomes the
engineer's responsibility when the engineer seals it: the agent of Lecture 25, the controller of
Lecture 26, and the classifier of Lecture 27 all needed a person who could explain them.
:::

## Where it lives today

| Today | What engineers do |
|:------|:------------------|
| Grid planning for data centres | load forecasts, interconnection studies |
| Emissions accounting | average and marginal factors, by hour |
| Water permits | cooling design, withdrawal and consumption |
| AI in engineering practice | tools inside workflows; the engineer seals |

::: notes
CS 441's environmental slides count prompts and training runs; the same numbers, multiplied by
millions of users, arrive at substations, interconnection queues, and water permits. eGRID (EPA) and
EIA-930 are the public data an engineer can check a filing against.
:::

## 28 lectures, one thread

![The course's six units and 28 lectures: learning from engineering data (L1-L8); uncertainty, risk, and decisions (L9-L12); deep learning for sensing (L13-L16); foundation models and time series (L17-L20); structure and physics (L21-L24); agents, decisions, and the profession (L25-L28).](cee_l28_ai_infrastructure/course_map.png)

::: notes
CS 441's conclusion, on our data: the same pipeline (data, an encoder, a decoder, a prediction, a
loss) ran through bridges, concrete, rivers, pipes, photos of cracks, laser scans, and the grid.
:::

## Go-to methods, on CEE data

| You have | Start with |
|:---------|:-----------|
| A small table | nearest neighbors, linear models (L2, L5, L6) |
| A larger table | boosted trees (L8) |
| Images, scans | a pretrained network, probe or fine-tune (L15-L18) |
| Text | retrieval and a language model, checked (L19, L25) |
| Physics you trust | a solver, or a model that obeys it (L23, L24) |
| Decisions over time | value iteration if P is known (L26) |

::: notes
CS 441's "go-to approaches", in our terms. And CS 441's standing advice: it is all about the data.
:::

## Always, before you trust it

| Habit | Lectures |
|:------|:---------|
| Split by time, place, or asset, not at random | 3, 6, 20 |
| A baseline above, a noise floor below | every lecture |
| Intervals, and calibration | 10, 11, 27 |
| Checks between a model and a report | 25 |
| Each group evaluated separately | 27 |
| The engineer signs, and can explain | 28 |

::: notes
The course's own rules, each learned on CEE data. Bias, variance, leakage, and drift are not ML
curiosities: they become a wrong pressure, a missed bridge, a flooded road.
:::

## Exit question

A vendor says its AI-designed cooling plant "cut the data centre's footprint 30%". What would you
ask before you seal the design?

**In the notebook** (`cee_l28_ai_infrastructure.ipynb`): the hourly grid, the marginal factor, the
campus, and your own GPU's watts.

::: notes
Any two of: 30% of what, energy, CO2, or water?; at which emission factor, average or marginal, and
for which grid?; on site only, or with the power plants' water?; measured or nameplate power?;
against which baseline design?; and can the engineer of record explain the design well enough to
take responsibility for it?
:::

## Things to remember

:::::: {.columns}
::: {.column}
- AI is a load on grids and rivers.
- Measure watts under the work.
- One load, many factors: say which, and why.
- New load: marginal, with an interval.
- The engineer who seals it answers for it.

**Thank you.**
:::
::: {.column}
![Thumbnails of today's figures: the GPU's power, the five factors, the marginal regression, and the campus.](cee_l28_ai_infrastructure/remember.png)
:::
::::::

::: notes
CS 441's take-aways: our work has personal and societal impact; ask who benefits, who is harmed, who
is responsible, and who owns the data; and use the tools that exist, such as emissions calculators,
model cards, and datasheets. Some slides follow the structure of Derek Hoiem's CS 441 lectures on
the ethics and impact of AI and on the course's conclusion.
:::

## References

:::::: {.columns}
::: {.column}
- Shehabi et al. (2024). U.S. data center energy. LBNL.
- U.S. EPA (2025). eGRID2023, rev. 2.
- U.S. EIA (2025). Form EIA-930, 2024 data.
- Siler-Evans et al. (2012). *Environ. Sci. Technol.* 46(9).
- Macknick et al. (2012). *Environ. Res. Lett.* 7(4).
:::
::: {.column}
- Illinois Professional Engineering Practice Act of 1989, 225 ILCS 325.
- Coeckelbergh (2020). *AI Ethics.* MIT Press.
:::
::::::
