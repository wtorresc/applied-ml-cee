---
title: 1-D Signals, CNNs, and LSTMs
subtitle: "Lecture 20 · Will the river flood tomorrow? Is the bridge still the bridge?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 8–19

On your own, no notes, two minutes:

1. Lecture 15: a 3 × 3 filter slid over a tile's pixels. What would a filter that slides over
   days of rain compute?
2. Lecture 8: the forest scored R² +0.16 on random folds, 0.00 with whole sites held out. What
   will random *days* do to a river forecast?
3. Lecture 19: a report's embedding lands near reports that mean the same. What would the
   embedding of a river's flow record land near?

**Today:** tomorrow's flow from the last two weeks, and a bridge's frequencies.

::: notes
Give the answers before moving on.
1. A weighted sum of the last few days of rain: the flow it adds today. Hydrologists have used
   that filter since 1932 (the unit hydrograph); today it is the first model, and a 1-D CNN
   learns many of them.
2. Usually inflate it: neighbouring days share their weather and their river, and a random split
   puts them on both sides. Today's break-it slide tests that on a forecaster that sees today's
   flow and on one that sees rain alone, against random whole years, which leave no neighbour
   in training.
3. Rivers that behave alike: snowmelt rivers near snowmelt rivers, flashy ones near flashy ones.
   Today a pretrained forecaster embeds all 671 CAMELS rivers, and the groups line up with
   climate it was never shown.
:::

## 1-D signals, three CEE records

![Three records stacked. A river's daily flow at Bismarck, Illinois, over water year 2013, low most of the year with a 30 mm/day flood in April. A railway bridge's vertical deck acceleration over 52 seconds at 825.8 values a second: quiet, a burst while a train crosses from about 11 to 22 seconds, then a fading ring. The bridge's 13th natural frequency, one value a day over 15 months: flat near 6.32 Hz, a jump in the January 2019 frost, a step up after the 2019 retrofit.](cee_l20_time_series/series.png)

::: notes
CS 441's list: stock prices, vibrations, popularity, temperature, audio. Ours: a river (USGS
03338780, North Fork Vermilion River near Bismarck, IL, 692 km², 33 miles from campus;
CAMELS, Newman et al., 2015), a steel railway bridge (KW51, Leuven; Maes & Lombaert, 2021) at
825.8 samples a second, and the same bridge's 13th natural frequency, identified every hour by
its operator. Three sampling rates about eight orders of magnitude apart. What they share: order
matters, neighbours are alike, and a record can be any length. Ask: what would you predict from
each, and from how much of its past?
:::

## Trends at several time scales

![Left: 25 water years of daily flow on a log axis, from 0.01 to 50 mm/day, with its 365-day mean wandering between 0.2 and 2.6 mm/day: wet years and droughts. Right: spring 2013 in mm/day: the daily flow with two floods, its Gaussian smoothing with sigma 3 days, a rounded hump, and its day-to-day change, the difference filter minus one, one, spiking up as a flood rises and down as it falls.](cee_l20_time_series/scales.png)

::: notes
Two operations from CS 441's lecture, both filters. Smoothing (a Gaussian, σ = 3 days, or the
365-day mean) shows the slow part: wet years and droughts, the season (June 5.2 times
September on average). Differencing ([−1, 1]) shows the fast part: the change from yesterday,
up to +20.3 mm/day in a day in April 2013. A model can be given either, or both: CS 441's
"reframe in terms of differences from recent days".
:::

## Will it flood tomorrow?

:::::: {.columns}
::: {.column}
- North Fork Vermilion at Bismarck, IL: 692 km².
- Alert when tomorrow beats **12.0 mm/day**: the 1-in-100-days flow.
- Data: 25 water years, weather and flow.
:::
::: {.column}
| Tomorrow | Floods | Doesn't |
|:---------|:-------|:--------|
| Alert | roads closed in time | a needless closure |
| No alert | cars in the water | nothing |
:::
::::::

::: notes
The decision: the county closes low-water crossings and moves equipment a day ahead. The
threshold is the training years' 99th percentile of daily flow, 12.0 mm/day over the
basin (about 96 m³/s). A missed flood risks lives at crossings; a false alarm costs a
closure and trust in the next alert. Data: CAMELS (Newman et al., 2015; CC BY 4.0): Daymet
weather averaged over the basin (rain, temperatures, radiation, humidity, day length) and USGS
daily flow, divided by the area to give mm/day. Periods in time order: train water years
1990–2004, validate 2005–2008, test 2009–2014, scored once.
:::

## Your turn: same rain, two floods?

| Storm | Rain, two days | Flow the day before | Month |
|:------|---------------:|--------------------:|:------|
| A | 53 mm | 0.69 mm/day | April |
| B | 66 mm | 0.008 mm/day | August |

::: notes
Ask: which storm raises the river more, and by how much? Two minutes alone, then a neighbour.
Most students say B: more rain. Keep the answers; the
reveal comes in three steps (the filter, the LSTM, the drought). Storm A: 10–11 April 2013, a
wet spring. Storm B: 26–27 August 2012, the end of the 2012 drought. The answer: A peaked at
14.5 mm/day; B at 0.47, about thirty times less from more rain. Ask: what did the river
know that the rain gauge did not?
:::

## Seven days, fourteen numbers

![The week to 11 April 2013: rain bars above, almost none until 5 mm on the 9th and 13 on the 10th, then 40 mm on the 11th; the river's daily flow below, steady near 0.7 mm/day, 1.7 on the 10th, then 14.5 on the 11th; the seven days are shaded as one window; at 12 April, a large question mark.](cee_l20_time_series/window_1.png)

::: notes
CS 441's homework 1: predict a city's temperature from the previous days. The same move here:
cut the record into windows of the last k days (rain and flow) and predict the next day. Each
window becomes one row of a table, and every method from Unit 1 applies. Ask: how would you
weigh yesterday's rain against last week's?
:::

## A weight per day: the forecast

![The same week with a weight beside each day: the linear model's weights on rain, largest on the last day, and on flow, largest on today's flow; their weighted sum gives the forecast, marked at the next day, against the observed flow, a star.](cee_l20_time_series/window_2.png)

::: notes
Linear regression on the window (k = 7, fitted on 1990–2008): forecast
$\hat q_{t+1} = b + \sum_k a_k\, P_{t-k} + \sum_k c_k\, q_{t-k}$. Each weight times its day, plus
the intercept, gives 14.2 mm/day; the river ran 13.1. Persistence (tomorrow = today) says
14.50. The weights on rain are the river's response to a millimetre; the weight on today's flow
(0.87) is how much of today's flow stays. The same arithmetic, with learned weights, slid along
the whole record: that is a 1-D filter.
:::

## Windows: a forest on the changes

![Validation NSE against window length, 1 to 30 days, for three models, each predicting either tomorrow's flow or the change from today. The random forest on changes is highest from two days on, near 0.68; the linear model is near 0.65 and identical for both targets; 10-nearest neighbours on flow falls from 0.65 at one day to 0.33 at 30 days, while on changes it stays between 0.60 and 0.66; persistence is a dotted line at 0.56.](cee_l20_time_series/windows.png)

::: notes
NSE (Nash & Sutcliffe, 1970) = 1 − squared error / variance of the observations: 1 perfect, 0
no better than the mean. A forest's score moves with its seed (up to 0.012 here), as much as
its settings differ, so each forest setting is scored on three seeds. Chosen on validation:
forest, on changes, 14-day window (validation 0.686; next, the 2-day window, 0.686: close);
refitted on 1990–2008 and scored once on the test years: 0.66, against persistence 0.55.
Three lessons from earlier lectures: the linear model gives exactly the same forecasts for both
targets, because today's flow is an input (moving it across the equals sign changes nothing);
k-NN on the flow falls from 0.65 to 0.33 as the window grows, while on the change, from the
same inputs, it stays between 0.60 and 0.66; trees cannot extrapolate a level but can a change. Given
tomorrow's rain too (a perfect weather forecast), the forest reaches 0.83.
:::

## Assume: rain adds flow the same way

:::::: {.columns}
::: {.column}
One filter $w$ on rain, the same every day:

$$\hat q_t = b + \sum_{k=0}^{59} w_k\, P_{t-k}$$

a 1-D convolution: Sherman's unit hydrograph.
:::
::: {.column}
![The filter's 60 weights by days since the rain: 0.11 on the day of the rain and 0.11 the next, falling to near zero by three weeks; they sum to 0.42.](cee_l20_time_series/uh_weights.png)
:::
::::::

::: notes
Sherman (1932): the flow a basin gives from one unit of rain, shifted and added for every day's
rain. Fitted by ridge (penalty chosen on validation) from rain alone, no flow: the weights peak
at lag 1 and sum to 0.42: about 42% of a millimetre of rain leaves as flow. The same sliding sum as
Lecture 15's filter, along one axis; the next slide runs it on a year it never saw.
:::

## The filter has no memory

![June 2012 to June 2013, observed flow against the filter's flow: the filter makes floods out of the late-summer 2012 storms, 66 mm of rain among them, that the river never ran, and reaches only part of the April 2013 flood after 53 mm. Test NSE 0.32.](cee_l20_time_series/uh_year.png)

::: notes
Test NSE 0.32. The attempt, first reveal: the filter gives storm B 7.4 mm/day and storm A 5.9. It must:
more rain, more flow, every day of the year. The assumption fails because the basin keeps
water: wet soil sends rain to the river, dry soil drinks it. A model needs memory.
:::

## SARIMA: seasons first, then noise

:::::: {.columns}
::: {.column}
Log monthly flow $y$: its change from a year before, recent errors:

$$(1 - \phi B)(1 - B^{12})\, y_t = (1 + \theta B)(1 + \Theta B^{12})\, \varepsilon_t$$

$B$: one month back.
:::
::: {.column}
![Monthly mean flow on a log axis, 2007 to 2014. After October 2008 SARIMA's forecast 72 months ahead repeats one seasonal shape inside a wide 90% band; its one-month-ahead forecasts follow each month's ups and downs closely.](cee_l20_time_series/sarima.png)
:::
::::::

::: notes
statsmodels SARIMAX, (1, 0, 1)(0, 1, 1)₁₂ on log monthly flow, fitted on water years 1990–2008.
Seventy-two months ahead it is the climatology: log RMSE 0.87 against 0.86 for each month's
training mean (the seasonal MA coefficient -0.998 cancels the seasonal difference). One
month ahead it beats it: 0.76 (last month alone: 1.03). The 90% band covered 99% of the
72 months: honest, and wide. CS 441's example is coat sales in December; ours is that the
river's season repeats and its storms do not.
:::

## Trees and ARIMA, or deep nets?

| Practitioners weigh | Trees, (S)ARIMA | Deep networks |
|:--------------------|:----------------|:--------------|
| Accuracy | often enough | often better with many series |
| Explain a forecast | yes | hard |
| Stable, controllable | yes | needs care |
| Expert knowledge in | easy | hard |
| Cost to run and watch | low | higher |

::: notes
CS 441's verdict: deep forecasters (TFT, Informer, N-BEATS) win benchmarks, but trees and
statistical models are more practical: interpretable, stable, controllable, cheap to monitor.
Hydrology is the counterexample to keep in mind: Kratzert et al. (2019) trained one LSTM on 531
basins; over the 447 basins every benchmark model covered, its median NSE was 0.76 against 0.60
for a calibrated conceptual model (SAC-SMA). Google's flood forecasts, in over 80 countries, run
on LSTMs (Nearing et al., 2024). The difference: thousands of similar series to learn from. Hold this; we vote at
the end.
:::

## Any record is a sum of sines

![Four seconds of the bridge ringing after a train, grey, rebuilt in blue from its largest Fourier terms: one sine catches the main swing, 27% of the variance; four sines, 52%; sixteen, 77%; sixty-four of 1,651, 93%.](cee_l20_time_series/fourier.png)

::: notes
Fourier's claim (1807), CS 441's next section: any record is a weighted sum of sines and
cosines of different frequencies. Here the four seconds after a train left KW51 (825.8 samples
a second, 3,303 samples, so 1,651 frequencies, multiples of 0.25 Hz). The largest single
term (4.25 Hz) explains 27% of the variance; 4, 52%; 16, 77%; 64, 93%; all 1,651 rebuild it exactly. A
bridge is the friendliest case: after the load leaves, it rings at a few natural frequencies,
each a sine that decays.
:::

## The spectrum finds the bridge's modes

![The amplitude spectrum of 29 seconds of ringing, 0 to 12 Hz, log scale, with sharp peaks; dashed lines mark the seven natural frequencies the operator identified in the same hour, 1.89 to 6.30 Hz; six of them sit on a peak.](cee_l20_time_series/spectrum.png)

::: notes
The amplitude of each sine, over 29 seconds of free vibration (Hann window, 0.034 Hz between
frequencies). The operator identifies the modes every hour from ambient vibration (Maes &
Lombaert, 2021). Six of their seven modes that hour have a
peak in our spectrum within 0.031 Hz; the seventh (5.32 Hz) is weak at this sensor. The
tallest peak, 4.11 Hz, matches their 4.106 Hz. Sampling at 825.8 Hz shows up to 413 Hz
(Nyquist). Peaks above 7 Hz are modes the operator does not track.
:::

## Spectrogram: frequencies over time

![Top: the deck strain, flat, then a row of humps while the train crosses, from about 11 to 22 seconds, then flat. Below: the spectrogram, 0 to 20 Hz over 52 seconds, power in dB: dark before the train, bright across all frequencies while it crosses, then a few bright horizontal lines that fade, near 4 Hz and above.](cee_l20_time_series/spectrogram.png)

::: notes
The spectrum of each 4-second window, slid every half second: time on one axis, frequency on
the other, power as colour (dB). While the train is on the bridge (from the strain gauge, a
different sensor) the response is forced: broad, at the train's own rhythm. After it leaves,
the bridge rings at its natural frequencies, the horizontal lines. Longer windows separate
frequencies better and blur time: 4 s gives 0.25 Hz. CS 441 converts audio to a mel
spectrogram because human hearing is logarithmic in frequency; a bridge does not hear, so we
keep linear frequency and use dB for the power.
:::

## A train, then the bridge rings

![The spectrogram as a landscape: time and frequency on the floor, power as height. A broad hill rises while the train crosses, then long narrow ridges run on along time at the bridge's frequencies, slowly sinking as the ringing fades.](cee_l20_time_series/spectrogram3d.png)

::: notes
The same numbers as the last slide, smoothed lightly and shown as height (8 Hz and below). The
hill is forced vibration; each ridge is one mode, and its slope down along time is that mode's
damping. A monitoring system reads exactly these ridges, every hour, to track the bridge's
frequencies (the third record of the opening slide).
:::

## Audio's two recipes, on our records

| CS 441, audio | The recipe | On CEE records |
|:--------------|:-----------|:---------------|
| Sound classes | spectrogram → vision CNN | bridge spectrograms → Lecture 15's ResNet |
| Speech to text | CNN features → LSTM over time | weather → 1-D CNN, LSTM → river flow |
| Foundation models | pretrained, used as is | Chronos-2 on the river, today |

::: notes
CS 441's audio models: classify a sound by turning it into a (mel) spectrogram and running an
image CNN on it, augmented by shifts and masks; recognize speech as Deep Speech does (Hannun
et al., 2014): a network extracts features from each window of the spectrogram, a recurrent
network (e.g. an LSTM) runs over time, and each step outputs character probabilities. Our records use the same two building blocks along time:
a 1-D convolution (the filter, learned) and an LSTM (the memory). The first row is the next
step for KW51 (vehicle and damage classes from spectrograms are an active SHM topic); the
second is our river.
:::

## 1-D CNN: Lecture 15's filter, along time

:::::: {.columns}
::: {.column}
$$z_{t} = b + \sum_{c} \sum_{k=0}^{K-1} w_{c,k}\, x_{c,\,t-k}$$

16 filters, 31 days long; then dilated filters: 285 days in view.
:::
::: {.column}
| Layer | Sees | Weights |
|:------|:-----|--------:|
| 16 filters, 6 inputs × 31 days | 31 days | 2,992 |
| 7 dilated layers, 3 taps, 1–64 apart | +254 days | 20,192 |
| Read-out of the last day | | 33 |
:::
::::::

::: notes
The same weight sharing as Lecture 15, along one axis: one filter applied at every day. Causal:
the output at day t sees only days up to t. Dilation (WaveNet, van den Oord et al., 2016)
spaces a filter's taps 1, 2, 4, ..., 64 days apart, so seven small layers see 285 days with
23,217 weights in all. Inputs: the six Daymet variables, standardized on training days; target:
the day's flow, from weather alone (no flow input). Trained with Adam at the rate a bracketing
search kept on validation, 0.003 (a worse rate on each side), and stopped 600 gradient steps
after its best validation check, checked every 5 steps: every network in this lecture gets that
same budget.
:::

## An LSTM keeps a store

![One LSTM step. Along the top runs the store, c: yesterday's store is multiplied by the forget gate, the input gate times a candidate is added, giving today's store. Below, four gates (forget, input, candidate, output) each read yesterday's hidden state and today's weather. The output gate times tanh of the store gives today's hidden state, h.](cee_l20_time_series/lstm_cell.png)

::: notes
Hochreiter & Schmidhuber (1997). In symbols: c_t = f_t ⊙ c_{t−1} + i_t ⊙ g_t and
h_t = o_t ⊙ tanh(c_t), each gate σ(W[h_{t−1}, x_t] + b), g with tanh; the next slide's code is
these two lines. The cell state c is a store: each day the forget gate keeps a
share of it (f between 0 and 1), the input gate adds a share of a new candidate, and the output
gate decides how much shows. Like a basin's soil and groundwater: it keeps water from earlier
rain and releases it later. Because c is updated by addition, gradients flow back over long
spans (the residual idea of Lecture 15, in time). 64 cells over 6 inputs: 4 × 64 × (64 + 6 +
2) = 18,432 weights, plus a read-out of 65: 18,497.
:::

## Your turn: predict the check

```python
for x_t in x:                      # 365 days, 6 inputs each
    z = W_ih @ x_t + W_hh @ h + b  # four gates at once
    i, f = sigmoid(z[:n]), sigmoid(z[n:2*n])
    g, o = tanh(z[2*n:3*n]), sigmoid(z[3*n:])
    c = f * c + i * g              # keep some, add some
    h = o * tanh(c)                # show some
```

Against PyTorch's `nn.LSTM`, same weights, float64: how close?

::: notes
Write a number. The loop is the two equations of the last slide, with PyTorch's gate order
(input, forget, candidate, output) and its two bias vectors added. The check runs one real
365-day window through both.
:::

## Agrees with PyTorch to round-off

| Quantity, one 365-day window | Max abs difference |
|:-----------------------------|-------------------:|
| Hidden states, all days, 64 cells | 2.7e-16 |
| Last cell state, 64 cells | 5.3e-15 |
| Weights | 18,497 |

::: notes
Float64 round-off over 365 steps: both run in float64 on the CPU, with the trained network's
weights (it trained in float32 on the GPU). Students who predicted "exactly zero": the order of
additions differs, so the last bit can differ.
:::

## Weather alone: networks beat the filter

![Water year 2013 at Bismarck: the observed flow against three models that see only the weather: the unit hydrograph, the 1-D CNN, and the LSTM. The networks follow the year more closely than the filter; the 1-D CNN overshoots several winter rises; all three miss part of the April peak.](cee_l20_time_series/hydrograph.png)

::: notes
From weather alone (no flow input), test water years 2009–2014, five seeds each: LSTM median
NSE 0.68 (0.61–0.74), 1-D CNN 0.63 (0.55–0.63), the filter 0.32. A run takes 8 to 14 seconds
on one GPU: 15 years of one basin is a small dataset. Each network's Adam rate was bracketed on
validation (the LSTM kept 0.03, the CNN 0.003), and each run stopped 600 steps after its best
validation check (the LSTMs' best checks at steps 80 to 530). The test years are drier (they
include the 2012 drought), and the LSTMs score lower on them than on validation (median 0.68
against 0.77).
:::

## The LSTM's response depends on the day

![Flow added per millimetre of rain, by days since the rain, 0 to 30. The filter's weights, dotted, the same every day, peaking near 0.11 and fading by three weeks. The LSTM on a wet day, 11 April 2013: 0.22 for today's rain, then falling, rain from the whole past month still counting a little. The LSTM on a dry day, 27 August 2012: 0.37 for today's rain, then a swing up and down over four days, near zero after a week.](cee_l20_time_series/response.png)

::: notes
The gradient of the LSTM's simulated flow with respect to each past day's rain: its own unit
hydrograph, computed for one day. Today's rain counts on both days, 0.22 and 0.37 mm/day per
mm; the rain of the 29 days before adds 0.65 on 11 April 2013 and nets to −0.08 on 27 August
2012, after a swing over its first four days. The filter's weights sum to 0.42 on every day of
the year. Second reveal, the attempt's answer: the same network, with the same weights, passes
a wet basin's past month of rain to the river and not a dry one's; what it carries from day to
day (its store) differs. These are one network's gradients (the median seed): read the shape,
not the digits.
:::

## Your turn: predict four failures

1. Score the forest on random days instead of later years.
2. Storm B: the river was lower than on any training day.
3. Train with one year of record instead of fifteen.
4. Alarm on the bridge's frequency through a winter.

Write what happens to each, and why.

::: notes
Two minutes, alone. Each of the next four slides reveals one.
:::

## Random splits: a lottery, not a leak

| The forest, 14-day windows | Later years | Random days | Random years |
|:-------------|------:|------------------:|------------------:|
| With today's flow | 0.66 | 0.66, 0.54 to 0.76 | 0.68, 0.51 to 0.72 |
| Rain only, no flow | 0.33 | 0.43, 0.38 to 0.47 | 0.42, 0.35 to 0.50 |

::: notes
The same model and settings; only the split changes. Ten draws of random test days from
1990–2014, as many as the test years have; ten draws of 6 random whole water years of the 25,
which leave no neighbour of a test day in training; and the later years, 2009–2014. Each cell:
the median, then the range of the ten. With today's flow as an input, the random splits' medians
sit next to the later years' 0.66; what the split changes is which floods land in the test set:
one draw reports 0.54, another 0.76. The same forest on rain alone (to tomorrow's), with no flow
input, simulates the river: every random-day draw scores above the later years, 0.38 to 0.47
against 0.33. A leak, like Lecture 8's sites? Then random whole years would score lower than
random days; they score the same (0.42 against 0.43). The gap is the later years themselves:
drier (the 2012 drought) and harder for this model, and a random split mixes them in with
easier years. The rule stands: a forecast is used on days after its training, so test on later
years, and report how much the score moves.
:::

## The drought no training day saw

| Storm B, August 2012: peak flow | mm/day |
|:--------------------------------|-------:|
| Observed | 0.47 |
| Filter on rain | 7.4 |
| LSTM, weather alone | 6.2 |
| Forest, with today's flow | 0.42 |
| LSTM, with today's flow | 1.14 |

::: notes
Third reveal. The flow before storm B, 0.0085 mm/day, is below every training day (lowest:
0.0088). The models that see only the weather make a flood of it, 7.4 and 6.2 mm/day, over ten
times the river's 0.47; those that see today's flow stay far closer: the forest 0.42 (a little
short), the LSTM 1.14. Both trained on the same 15 years, 1990–2004; peaks read over the storm's two days and the
day after. Lecture 14's lesson in time: a model asked about conditions it never saw answers
confidently. A drought and a flood are exactly where an alert matters.
:::

## One year of record: borrow a model

![Test NSE of tomorrow's flow against years of training record, 1 to 15, log axis. The LSTM from scratch, its median of three seeds with their range, scores 0.54 with one year, 0.58 with two, 0.65 with five, and 0.64 with fifteen; the forest rises from 0.56 to 0.65, close to the LSTM at every length; Chronos-2, never trained on this river, is a flat line at 0.63, above both with one or two years.](cee_l20_time_series/data_size.png)

::: notes
Lecture 15 with time series. Every model forecasts tomorrow from the past flow and rain, with
the same validation and test years. Each length gets the same training budget, counted in
gradient steps, with early stopping, and the LSTM's rate is bracketed again at each length; the
forest keeps the setting chosen on 15 years, and the inputs are standardized with the 15 years'
statistics at every length. The LSTM (three seeds) scores 0.54, 0.58, 0.65, 0.64 with 1, 2, 5,
15 years (at 15 its seeds span 0.63 to 0.66); the forest 0.56, 0.60, 0.62, 0.65: close at every
length (gaps under 0.03). Chronos-2, a pretrained forecaster used as is, scores 0.63 with
none: with one year, above both (0.56 and 0.54). CS 441's rule (Lecture 17, 2026): with very
little data, probe a pretrained model; with moderate data, fine-tune it. A new gauge has one
year.
:::

## Frost hides the retrofit

![Top: KW51's mode 13 by day over 15 months, with a band of the pre-retrofit mean plus or minus 3 standard deviations, 0.13 Hz each way; in the January 2019 frost the frequency jumps to 6.6 Hz, outside; after the 2019 retrofit it settles 0.1 Hz higher, inside the band. Bottom left: frequency against bridge temperature: flat and slightly falling above 0 degrees, climbing steeply below 0, with the hinge fit; after the retrofit, a separate, higher line. Bottom right: the residual from the temperature fit, flat near zero before the retrofit and a clean step to 0.1 Hz after it, outside its band.](cee_l20_time_series/alarm.png)

::: notes
KW51 was retrofitted from May to September 2019 (Maes & Lombaert, 2021). Mode 13 rose 1.5%
after the works; on the 6 days the bridge was below 0 °C it rose 3.3%: a frozen structure is
stiffer, a known effect in bridge monitoring (Peeters & De Roeck, 2001, on the Z24 bridge). Both
alarms take the same reference: the 174 pre-retrofit days with a temperature reading, a winter
included. On the raw frequency, mean ± 3 SD is ±0.13 Hz: it fires on 6 of those days (5 of
them frozen) and on none of the 108 days after the retrofit. Regress on temperature with a bend
at 0 °C (Lecture 5) and alarm on the residual, ±0.064 Hz: 4 of the 174, and all 108 after the
retrofit. With the autumn weeks as the reference instead, each with its own band, both fire on
12 of the 165 later healthy days; but the residual's band then rests on the 9 autumn days with a
temperature reading, and at the raw band's width the residual alarm fires on 7. What the
correction surely does is make the retrofit visible. The confounder is the season, again.
:::

## Chronos-2: no training, close behind

![The April 2013 flood, forecast one day ahead. Black: observed. Chronos-2's median, dashed, with its 10 to 90% band shaded, rises a day after each jump in flow; the forest, trained on 15 years, is also a day late, and overshoots the day after each peak.](cee_l20_time_series/chronos_event.png)

::: notes
Chronos-2 (Ansari et al., 2025; Amazon, Apache-2.0, 119 million weights): a transformer
(Lecture 17) pretrained on many public time series, given the last 512 days of flow and rain
(context chosen on validation) and asked for tomorrow's quantiles. Test NSE: flow alone 0.62,
with rain 0.63, with tomorrow's rain 0.71. Trained here on the same 15 years, 1990–2004: the
forest 0.65 and 0.81, the LSTM 0.64 and 0.85 (three seeds). Its band covered 81% of these 47
days. Like the forest, it is a day late on a rise it cannot see coming
without tomorrow's rain. Its training corpora list no CAMELS or USGS series, but one of them
does not list its datasets: contamination cannot be ruled out (Lecture 3's warning).
:::

## Embeddings group 671 rivers

![A map of the 671 CAMELS basins across the United States, each a marker coloured and shaped by one of five groups found by k-means on Chronos-2's embedding of its flow record: the snowiest group, downward triangles, lies almost entirely in the western mountains; the driest, circles, centres on the southern Plains and Texas; humid basins of the Northeast and the Pacific Northwest share the diamonds; the Appalachians and the Southeast the upward triangles; the home basin, circled, sits in the Midwest and South group, squares.](cee_l20_time_series/basin_map.png)

::: notes
Lecture 19's move, on rivers: stop training and use a pretrained model's embeddings. Each basin's flow
over 2048 days (to September 2004), embedded by Chronos-2: the mean of its 768-number patch
tokens. k-means (Lecture 4) on 20 principal components finds five groups; they line up with
climate, which the model never saw. Median snow fraction by group: 0.04, 0.06, 0.08,
0.19, 0.48; median aridity: 1.48, 0.85, 0.77, 0.61, 1.29. The home basin is
in group 2. A linear probe (ridge), each of the 18 water
regions held out in turn, reads snow fraction from the embedding with R² 0.72 (four flow
signatures a hydrologist would compute: 0.40) and aridity with 0.53 (0.07).
:::

## Day-ahead alerts at Bismarck

![Horizontal bars for seven forecasters on the six test years: floods caught out of the flood days, and false alarms. Persistence, the forest, the LSTM, and Chronos-2, from past flow and rain; then the forest, the LSTM, and Chronos-2 given tomorrow's rain as well, which catch more floods.](cee_l20_time_series/decide.png)

::: notes
Test water years 2009–2014: 10 days above 12.0 mm/day. Floods caught / false alarms:
persistence 4 / 6; forest 4 / 1; LSTM 5 / 0; Chronos-2 2 / 1. Given tomorrow's rain: forest
7 / 1; LSTM 7 / 4; Chronos-2 4 / 1. The forest and the LSTMs trained on the same 15 years; the
LSTMs are their median seeds' runs; tomorrow's rain is one more input to each, rain only. A
storm's rain reaches this river the same day (the filter's weight on today's rain, 0.105, is
next to its peak, 0.109), and every forecaster given tomorrow's rain catches more floods: the
alert needs the weather forecast. In service that rain is a forecast, not the
observation used here, so these counts are an upper bound. 10 flood days is a small sample:
a difference of one or two is noise.
:::

## Who answers for a missed flood?

| Question | A gauge reading | A model's alert |
|:---------|:----------------|:----------------|
| Issued by | the weather service, by rule | an agency, a vendor, a county |
| A miss | a known limit of the rule | whose model, whose threshold? |
| Weakest | where there is no gauge | where training had no such year |
| Gains most | gauged, wealthier basins | ungauged ones, if it transfers |

::: notes
The social and ethical note. In the US the National Weather Service issues flood warnings; a
county acting on a vendor's model inherits a decision with no clear author. Google's model
extended reliable warnings to ungauged rivers in over 80 countries (Nearing et al., 2024),
where people have had none: a real gain in equity. Its errors fall where data are scarcest,
the same places. Ask: who must re-check the threshold after a drought like 2012?
:::

## Vote: what runs the alert next year?

The county wants a day-ahead alert at Bismarck.

- **A.** Persistence: alert if today is above the threshold.
- **B.** The forest on 14-day windows, with the rain forecast.
- **C.** An LSTM trained here, with the rain forecast.
- **D.** Chronos-2 as is: no training, no upkeep.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: individual vote, discussion, revote, then the explanation; never skip it.
B, with a check. Trained on the same 15 years (1990–2004) and given tomorrow's rain, the forest
caught 7 flood days with 1 false alarm and the LSTM 7 with 4 (last slide): close, on 10 flood
days, and the forest is cheaper to run, explain, and monitor (CS 441's table). D is the
right start at a new gauge with one year of record. A catches only floods already under way
(4 of 10 flood days) and raises the most false alarms (6). C becomes the
answer at scale: many basins, many years, one model (Kratzert; Nearing). Either way: re-check
the threshold after a drought, and log every alert against what happened.
:::

## Where 1-D models live today

| Today | What it is | Lecture |
|:------|:-----------|:--------|
| Google Flood Hub | LSTMs on thousands of gauges | here |
| Chronos-2, TimesFM, Moirai | pretrained transformers, used as is | 17, here |
| Whisper and speech models | spectrogram encoder + transformer | 17 |
| Bridge monitoring | hourly modes, alarms on residuals | here |

::: notes
CS 441 ends with a table of audio foundation models; ours is for time series. Google Flood Hub
(Nearing et al., 2024) runs LSTMs trained on thousands of gauges. Pretrained forecasters:
Chronos-2 (Ansari et al., 2025), TimesFM (Das et al., 2024), Moirai (Woo et al., 2024). Speech
recognition moved from Deep Speech's recurrent layers to transformers on spectrograms (Whisper;
Radford et al., 2023), the architecture of Lecture 17. One network for many rivers needs many rivers: with
our 15 Illinois basins, one LSTM for all does not beat fifteen separate ones (median NSE 0.62
against 0.66, better at 7 of 15; the same three seeds on both sides, each side's rate bracketed
on validation).
:::

## Exit question

A vendor sells a river forecaster: "NSE 0.92, tested on a random 20% of days from 30 years."
Name two things you would ask before believing it, and the number you would want instead.

**In the notebook** (`cee_l20_time_series.ipynb`): windows, the unit hydrograph, SARIMA, the
bridge's spectrum, an LSTM from scratch, and Chronos-2.

::: notes
Answer: the split first: one random draw can land anywhere (today 0.54 to 0.76), and random days
can put a test day's neighbours in training and mix hard years with easy ones; ask for later
years held out, and for a drought or flood year the training lacked. Then
the inputs: did it use tomorrow's observed rain? That is a perfect weather forecast. The number
to want: hits and false alarms at the county's threshold on held-out years, with the rain
forecast it will have in service. Collect the answers; they seed the next warm-up.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- Windows turn a record into a table.
- A filter on time is a 1-D convolution.
- Sines and spectrograms read vibration.
- An LSTM's store is its memory.
- Few years? Borrow a pretrained model.

**Next:** graphs, for networks of pipes and roads.
:::
::: {.column}
![Four thumbnails of the lecture's key figures: validation NSE against window length for three models; the LSTM's response to past rain on a wet and a dry day against the filter's; the bridge's spectrogram during and after a train; test NSE against years of record, with Chronos-2's flat line.](cee_l20_time_series/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Addor et al. (2017). *HESS*.
- Ansari et al. (2025). arXiv.
- Box & Jenkins (1970). Holden-Day.
- Das et al. (2024). *ICML*.
- Hannun et al. (2014). arXiv.
- Hochreiter & Schmidhuber (1997).
- Kratzert et al. (2019). *HESS*.
- Maes & Lombaert (2021). *JBE*.
:::
::: {.column}
- Nash & Sutcliffe (1970). *J. Hydrol.*
- Nearing et al. (2024). *Nature*.
- Newman et al. (2015). *HESS*.
- Peeters & De Roeck (2001). *EESD*.
- Radford et al. (2023). *ICML*.
- Sherman (1932). *Eng. News-Rec.*
- van den Oord et al. (2016). arXiv.
- Woo et al. (2024). *ICML*.
:::
::::::
