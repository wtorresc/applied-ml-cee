---
title: Decisions Over Time
subtitle: "Lecture 26 · Markov decision processes and reinforcement learning: when to replace a bridge, when to switch a signal"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 6–25

On your own, no notes, two minutes:

1. Lecture 6: a miss costs 20 false alarms. Where does the flagging threshold go?
2. Lecture 13: what did the logistic regression predict about bridge decks?
3. Lecture 25: what stood between the agent's answer and a report?

**Today:** decisions whose consequences arrive years, or minutes, later.

::: notes
Give the answers before moving on.
1. Flag when p > 1 / (1 + 20), about 0.048: the threshold that minimizes the expected cost of one
   decision. Today the decision repeats, and today's action changes the state the next one faces.
2. The chance that a deck loses a rating point by its next inspection. Today those chances become a
   transition matrix, and the question becomes what to do about them.
3. Five checks and an engineer. Today's learned controller needs a check of its own.
:::

## When should Illinois replace a bridge?

![Illinois's 21,971 highway bridges in 2024 (culverts left out) by lowest rating (deck, superstructure, substructure): most are rated 5 to 7; 2,309 are poor (4 or less), highlighted.](cee_l26_reinforcement_learning/ratings.png)

::: notes
The National Bridge Inventory again (Lectures 1, 3, 6, and 13's national decks): every Illinois
highway bridge, inspected every two years, rated 0 to 9 for its deck, superstructure, and
substructure. FHWA calls a bridge poor when its lowest rating is 4 or less. A replacement is
expensive; a poor bridge costs its users (weight limits, detours, risk). The decision repeats every
inspection, and today's choice changes the bridge the next inspector sees.
:::

## Your turn: replace now or wait?

| A poor bridge: replace now, or in two years? | 2% a year | 7% a year |
|:----------------------------------------------|----------:|----------:|
| γ, the weight of a cost two years away | 0.961 | 0.873 |
| Replace now: 1, plus 0.10 for this poor step | 1.10 | 1.10 |
| Wait: 0.10 now, then 1.10 in two years | ? | ? |

::: notes
Two minutes, with a calculator. The convention is the MDP's, used all lecture: a two-year step
that starts poor costs 0.10, whatever is done about it, so replacing a poor bridge now costs 1.10.
Waiting costs 0.10 now, plus 1.10 in two years (the replacement, and that step starts poor too),
worth γ × 1.10 today: 0.10 + 0.9612 × 1.10 = 1.157 at 2%, more than 1.10, so replace now; 0.10 +
0.8734 × 1.10 = 1.061 at 7%, so wait. The same bridge, the same costs, opposite answers: the
discount rate decides. And this compares only two plans; waiting four years, or replacing the new
bridge sooner in its own life, are plans too. The rest of the lecture compares all of them at
once. CS 441 asks the same question of a game: 10 points now and 10 three steps later, at γ = 1/2,
are worth 10 + 10/8 = 11.25.
:::

## An MDP: state, action, cost

![The loop: the agency (agent) sees the bridge's state, its lowest rating; it chooses an action, do nothing or replace; the bridge and its next inspection (environment) return a cost and the next state, two years later.](cee_l26_reinforcement_learning/picture_1.png)

::: notes
The one picture, three frames. A Markov decision process: states (the lowest rating, 3 or worse
merged up to 9), actions (do nothing, replace), costs, and transition probabilities P(s' | s, a):
the chance of each rating two years later. "Markov": the next state depends only on the current
state and action, not on how the bridge got there. CS 441 states it with rewards; an agency
minimizes costs, which are rewards with the sign flipped.
:::

## Later costs count less

![The same loop, with a row of bars: a cost k steps ahead weighs γ to the k, with γ = 0.961 per two years: 1.00 now, 0.96, 0.92, 0.89, 0.85, 0.82 over ten years.](cee_l26_reinforcement_learning/picture_2.png)

::: notes
Frame 2: the objective is the expected discounted sum of costs, E[Σ γ^k c_k]. Discounting makes the
infinite sum finite and says a dollar next cycle is worth less than a dollar now. The weights shown
are OMB's 2% a year, compounded over a two-year step.
:::

## A policy, and its value

![The same loop and discount bars; the agency box is now labelled with a policy, π(rating) gives the action, and a value, V(rating), the expected discounted cost of following π from that state.](cee_l26_reinforcement_learning/picture_3.png)

::: notes
Frame 3: a policy maps each state to an action; its value V(s) is the expected discounted cost of
following it from s. The goal: the policy with the smallest value in every state. CS 441's problem
statement in our terms: state (what the agent knows), action, policy, reward (here, minus cost),
expected discounted return.
:::

## Two years of Illinois bridges

![Left, the do-nothing transition matrix: rows the rating now, columns the rating two years later, in percent; most mass on the diagonal (91% at 4, 88% at 6, 72% at 8), the rest mostly one or two ratings lower. Right, after a replacement: 63% of new bridges are first rated 9, 32% are rated 8.](cee_l26_reinforcement_learning/transitions.png)

::: notes
Counted, not assumed. 61,891 pairs of a bridge's records in consecutive files, 2014 to 2024, whose
inspections were 18 to 30 months apart: a bridge inspected every four years would otherwise count
four years of change as one step. Pairs with recorded work (Item 106, 665 of them) are not "do
nothing". A rating that rose with no recorded work (2,443 pairs, 4.0% of the rest) is unrecorded
work or two inspectors disagreeing; those pairs are left out: 58,783 remain. The matching
disagreements downward stay in, a small bias toward decay: counting the rises as no change instead
moves the critical cost of being poor (two slides on) from 0.071 to 0.070. A replacement leads to
the ratings that 2,191 new bridges received at their first inspection. Rating 9 rarely lasts: 9% of
nines are still nines two years later. Madanat et al. (1995) estimate such matrices with an ordered
model, which shares strength across sparse ratings; with 58,783 pairs, counting is enough.
:::

## Your turn: which discount rate?

| OMB Circular A-94 | Real rate | γ, two years |
|:------------------|----------:|-------------:|
| App. C (2026): cost-effectiveness | 2.0% | 0.961 |
| 1992 base case: benefit-cost | 7% | 0.873 |

::: notes
Which fits a maintenance policy, and what does each say about the future? CS 441 asks what discount
factor suits Atari (0.99 per frame). Ours come from federal guidance. Appendix C (March 2026) gives
Treasury borrowing rates for cost-effectiveness analysis: choosing the cheapest way to deliver a
fixed service, which a maintenance policy is. The 1992 Circular, reinstated in 2025, sets 7% for
benefit-cost analysis of public investment, the return capital would earn elsewhere. At 7%, a cost
20 years away weighs 0.26; at 2%, 0.67. A high rate makes the future cheap.
:::

## Bellman: cost now plus what follows

```python
V = np.zeros(n_states)
while True:
    Q = C + gamma * np.einsum("ast,t->sa", P, V)  # all states, actions
    V_new = Q.min(axis=1)
    if np.abs(V_new - V).max() < 1e-10: break
    V = V_new
```

$$V(s) = \min_a \Big[\, c(s,a) + \gamma \sum_{s'} P(s' \mid s, a)\, V(s') \Big]$$

$$Q(s,a) = c(s,a) + \gamma \sum_{s'} P(s' \mid s, a)\, \min_{a'} Q(s', a')$$

::: notes
The value of a state is the best action's cost now plus the discounted value of where it leads. Q(s,
a) is the same quantity for one action; the policy picks the action with the smallest Q. Value
iteration applies the right-hand side until nothing moves: the derive step. It needs P, which we
counted. CS 441 writes the same equation with rewards and a max.
:::

## Value iteration converges like γ^k

![Left, each rating's value by sweep: all rise from zero and level off (rating 3 or worse and rating 4 at 1.89 replacements, 6 at 1.03, 9 at 0.81). Right, the largest change per sweep on a log scale, falling along a line parallel to γ^k, below 1e-10 after 510 sweeps.](cee_l26_reinforcement_learning/value_iteration.png)

::: notes
Each sweep shrinks the largest error by at least γ: the Bellman update is a contraction, so value
iteration converges from any start, to one answer. 510 sweeps at γ = 0.961 reach changes under
1e-10. The values are in replacements: a new bridge (rating 9) will cost its owner 0.81 of a
replacement, discounted, over all its future; a poor one, 1.89.
:::

## Your turn: predict the policy

A poor bridge costs 0.10 per two years. At which ratings does the optimal policy replace?

1. At 2% a year.
2. At 7% a year.
3. With no cost for being poor.

::: notes
Write a rating, or "never", for each. Use the attempt: at 2% replacing now beat waiting two years;
at 7% it did not.
:::

## Replace when poor, if poor costs enough

![The optimal action by lowest rating and by the cost of two years poor. At 2% a year, replace at ratings 4 and below once that cost exceeds 0.071; at 7%, only above 0.163. Below those lines, never replace; at a cost of 0.10 (the marker), 2% replaces and 7% does not.](cee_l26_reinforcement_learning/policy_map.png)

::: notes
At 2%, with 0.10 per two years poor: replace at 4 and 3. At 7%: never; the bridge stays poor
forever, at 0.10 every two years, worth 0.79 today, less than a replacement. With no cost for being
poor, no rate ever replaces: the cheapest policy lets every bridge decay. The cost of being poor is
a judgement (user delays, detours, risk), not data. The policy is only as good as that judgement:
rewards are hard to write, a point CS 441 makes about chatbots.
:::

## Two surfaces, one boundary

![Two surfaces over the lowest rating and the cost of two years poor, at 2%: Q for doing nothing (blue) and for replacing (orange). Doing nothing is cheaper at good ratings; at poor ratings its surface rises with the cost and crosses the replace surface at 0.071.](cee_l26_reinforcement_learning/q_surfaces.png)

::: notes
Q(s, do nothing) and Q(s, replace) as surfaces. The policy takes the lower surface; where they cross
is the line on the policy map. The replace surface is nearly flat: after a replacement, every bridge
starts over the same way. At rating 4 and a cost of 0.10, the two Q values are 1.917 (do nothing)
and 1.890 (replace): close, so a small change in the costs flips the decision.
:::

## Q-learning: learn Q by trying

```python
for t in range(2_000_000):                 # simulated inspections
    a = random_action() if rng.random() < 0.2 else Q[s].argmin()
    s2 = simulate(s, a)                    # the world: we never read P
    n[s, a] += 1
    Q[s, a] += n[s, a] ** -0.7 * (C[s, a] + gamma * Q[s2].min() - Q[s, a])
    s = s2
```

$$Q(s,a) \leftarrow Q(s,a) + \alpha \Big[\, c + \gamma \min_{a'} Q(s', a') - Q(s,a) \Big]$$

::: notes
No transition matrix: only experience, one inspection at a time. The bracket is the temporal-
difference error: a target from one step of hindsight (the cost now plus the estimate at the next
state) minus the current estimate. CS 441: "hindsight is 20/20". The step size shrinks with each
pair's visits; exploration tries a random action one time in five, so replacing is tried even at
good ratings.
:::

## Q-learning finds the same policy

![Q-learning's largest error against value iteration's Q, on log axes, falling from 1.15 after a thousand simulated inspections to 0.02 after two million; points are green where the greedy policy matches value iteration's, orange where it does not; a dotted line marks the last step at which it differs.](cee_l26_reinforcement_learning/q_learning.png)

::: notes
Build and verify: Q-learning never saw P, value iteration did, and they agree. Checked after every
step, the greedy policy last differs from value iteration's at step 676,020 and matches it for good
after that. Before, it flickers almost only at ratings 3 and 4 (on 181,094 and 81,654 steps; a few
dozen at 5 and 9), where the two Q values are close (1.917 against 1.890). The largest error falls
to 0.0379 by a million and 0.0201 by two million; the rarest state-action pair was visited 3,315
times.
Q-learning converges when every pair is visited infinitely often and the step sizes shrink slowly
enough (Watkins and Dayan, 1992).
:::

## Too many states for a table

![One intersection in SUMO at the busiest moment (minute 13) of a held-out hour under the trained DQN: four single-lane approaches of 300 m, the 160 m nearest the junction drawn, each car orange if stopped and blue if moving; 21 vehicles on the approaches, 9 of them stopped.](cee_l26_reinforcement_learning/intersection.png)

::: notes
CS 441's Breakout has a screen for a state; a signal has queues. One intersection in SUMO, a
microscopic simulator in which each car follows the one ahead: four approaches, through traffic
only. Every five seconds the controller picks a green, north-south or east-west; a switch costs a 3
s yellow and a 2 s all-red. Its state: each approach's stopped and total vehicles, the green, and
how long it has run. Ten numbers, eight of them counts up to about 40: a table over them is out of
reach, so a network estimates Q. At the busiest moment of a held-out hour (minute 13), 21 vehicles
were on the approaches and 9 were stopped; with the exit roads, 29 were in the network.
:::

## Webster's plan: a first baseline

![Left, the assumed demand by quarter hour: north-south 600, 700, 400, 300 vehicles per hour per approach; east-west 250, 300, 500, 600. Right, Webster's greens and change intervals: one plan for the hour (16 s and 13 s), and one per quarter hour that follows the demand.](cee_l26_reinforcement_learning/webster.png)

::: notes
The baseline engineers use: Webster (1958). The cycle C0 = (1.5 L + 5) / (1 − Y), with L the lost
time (10 s: two changes) and Y the sum of the critical flow ratios q / s; the greens share the rest
by flow ratio. The saturation flow s is measured in the simulator: a standing queue of 39 cars
discharged at 1.9 s headways, 1,895 vehicles per hour of green. One plan for the hour uses its mean
flows (Y = 0.482, a 38.6 s cycle). The plans per 15 minutes know the schedule in advance. The
formula counts every yellow and all-red as lost time and the displayed green as effective green,
so its cycle is a starting point, not the best fixed plan: we will tune it. The demand is assumed,
not counted.
:::

## A deep Q-network

| Part | Setting | Why |
|:-----|:--------|:----|
| Q-network | 10 inputs, 2 × 64 ReLU, 2 outputs | one Q per green |
| Reward | minus the vehicles stopped, each second | delay |
| γ per five-second decision | 0.99 | about 8.3 minutes ahead |
| Exploration | ε from 1 to 0.05 over 30,000 decisions | try both greens |
| Replay buffer | 50,000 decisions, batches of 64 | break the correlation |
| Target network | copied every 500 decisions | a target that holds still |

::: notes
Mnih et al. (2015), as CS 441 presents it, scaled down. The loss is Q-learning's TD error on a batch
drawn at random from past decisions; the target uses a copy of the network that is refreshed only
every 500 decisions. The reward sums to minus the vehicle-seconds of waiting, counted on the
approaches; a car that cannot even enter a full approach is not in it, which matters only when
queues reach 300 m. Each run trains on 200 simulated hours (a decision every five seconds, ten when
the green changes), three runs per method, one CPU thread each. A decision with a change of green
lasts ten seconds but is discounted by one γ, a simplification. A training hour ends at 3,600 s
with its queues still there: a time limit, not a terminal state, so every target bootstraps from
the next state, the hour's last too (Pardo et al., 2018).
:::

## Your turn: learn from each step?

CS 441 on learning from each step of one game at a time: "may not be efficient", "may be
unstable".

Predict: the same network, learning from each decision once, as it happens: no replay, no
target network. Better or worse than Webster's plans? Than the DQN?

::: notes
Two minutes, written. The textbook reason (Mnih et al., 2015): consecutive decisions are nearly the
same state, so updates from one at a time chase the last few seconds of traffic; replay draws them
from many hours, and a target network holds the target still.
:::

## Replay and a target net learn more

![Training curves: waiting in each simulated hour, in vehicle-hours, a ten-hour moving average over three runs per method, against Webster's plans (dotted) and actuated control (dashed) on the held-out hours. All three fall below the hourly plan's level during training; over the last 20 hours the DQN averages 4.11, the online network 4.94, PPO 4.98.](cee_l26_reinforcement_learning/curves.png)

::: notes
Same network, same 200 hours. With replay and a target network: 4.11 vehicle-hours of waiting per
hour over the last 20 training hours; online, each decision once: 4.94. The online network does
learn, so the drawback is efficiency more than collapse here. The comparison changes three things
at once: the online network has no replay, no target network, and a batch of one decision instead
of 64, at the same Adam rate; which of them matters most was not tested. Exploration is still on
during training (ε = 0.05 for the DQNs; PPO samples its policy), so these curves are not the
held-out scores that follow.
:::

## PPO: improve, but not too far

![PPO's objective, min(r A, clip(r, 0.8, 1.2) A), against the probability ratio r. For an action better than expected (advantage above zero) the gain stops growing past r = 1.2; for a worse one (below zero) the loss stops shrinking below r = 0.8. The unclipped objective is dashed.](cee_l26_reinforcement_learning/ppo_clip.png)

::: notes
Schulman et al. (2017), as in CS 441. The objective in full: maximize the mean over decisions of
min(r_t A_t, clip(r_t, 1 − 0.2, 1 + 0.2) A_t), where r_t = π_new(a_t | s_t) / π_old(a_t | s_t). The
policy outputs a probability for each green; the critic predicts the value of the state; the
advantage A_t says how much better the action was than the critic expected (estimated over the
following decisions, GAE). The ratio compares the new policy with the one that collected the data:
past 1 ± 0.2 the objective stops paying, so one batch has no incentive to drag the policy far. An
entropy bonus keeps it from becoming too sure too early. Here the actor and the critic are separate
networks: in a trial run with one shared body, PPO learned to hold one green. A likely reason, not
measured: the critic's large early errors dominate the shared gradients.
:::

## Actuated control is the bar

![Mean delay per vehicle over ten held-out hours: Webster for the hour 31.9 s; Webster per 15 minutes 18.7 s; the same plans with every green × 1.4, tuned on training hours, 17.7 s; SUMO's actuated control 16.3 s; DQN 17.1 s; DQN online 22.1 s; PPO 19.6 s; dots are the three trained runs of each method.](cee_l26_reinforcement_learning/results.png)

::: notes
Ten demand draws never seen in training, every vehicle counted, including any still waiting when the
network had to clear. Two baselines an engineer tries first. A fixed plan tuned like the DQN:
Webster's formula is a starting point, so one factor on its 15-minute greens was chosen on ten
training hours, ×1.4 of 1.0 to 1.6: 17.7 s. And SUMO's own actuated control, untuned: each green
runs 5 to 50 s, extended while its detectors see cars; no schedule, no training: 16.3 s. The DQN's
three runs (16.5 to 17.8 s) beat Webster per 15 minutes on 29 of 30 run-hours and the tuned plan on
25 of 30, but beat actuated control on only 2 of 30: every run trails it, by 0.2 to 1.5 s. The
bracket: the hourly plan (31.9 s) above; the spread of the three DQN runs (16.5 to 17.8 s) below.
PPO (19.0 to 19.9 s) is slightly behind Webster per 15 minutes (18.7 s), worse on 23 of 30
run-hours, and left 61 vehicles waiting at the end of the 30 runs' clearing time; PPO is run
greedily here, by its most likely green. The online network: 22.1 s, one run at 28.3 s.
:::

## The DQN follows the demand

![North-south's share of green by five minutes: the hourly plan stays near 55%; the plan per 15 minutes steps from 71% to 33%; the DQN goes from 69% to about 33%, actuated control from 69% to about 35%, in the direction of the demand's own share (dotted), which falls from 71% to 33%.](cee_l26_reinforcement_learning/adapt.png)

::: notes
Nobody told the DQN the schedule: it reads only the queues, and its split follows the demand's,
within 6 points in every quarter hour. Actuated control, which reads only its detectors, follows it
too. Webster's 15-minute plans follow the demand more closely, but only because they were computed
from it.
:::

## DQN or PPO?

| | DQN | PPO |
|:--|:--|:--|
| Learns | Q for each action | a policy and a value |
| Actions | discrete, one output each | discrete or continuous |
| Data | off-policy: replays old decisions | on-policy: each batch, then discarded |
| Exploration | ε-greedy | sampling its own policy |
| Here: delay; training | 17.1 s; about 10 min a run | 19.6 s; about 6 min a run |

::: notes
CS 441's comparison, with our numbers. PPO's on-policy data is used for ten epochs and then thrown
away, while the DQN replays every decision many times; whether that explains the gap here was not
tested. On Atari, Schulman et al. report PPO ahead of A2C, another policy method, in most games
(CS 441's slide). Training times are on one CPU thread a run, nine runs at once on a shared
workstation.
:::

## Four ways to learn a policy

| CS 441's approach | Learns | Here |
|:------------------|:-------|:-----|
| Behavioural cloning | the expert's action, given the state | what Illinois did, by rating |
| Q-learning | the value of each state and action | Q-learning; the DQN |
| Policy learning | the policy, and a value to judge it | PPO |
| Model-based | the world's transitions; then plan | value iteration on counted P |

::: notes
CS 441's four approaches, each on today's problems. The first three are model-free: they never
estimate P. Value iteration on the inventory's transition matrix is model-based: count the world,
then plan. When the model can be counted, as with bridges, planning is exact and cheap; when it
cannot (a signal's queues in continuous time), learn from a simulator.
:::

## What Illinois did, by rating

![By lowest rating, the share of bridges with no rated record in any later file (replaced, removed, renumbered, or dropped from the rated bridges): 16.9% at 3 or worse, 7.1% at 4, 1.4% at 5, near zero above; and the share with recorded work within two years, under 3% everywhere. Shaded: the ratings where the MDP at 2% replaces.](cee_l26_reinforcement_learning/agency.png)

::: notes
Behavioural cloning would learn this: act at 3 and 4, rarely above. The MDP agrees on where (at 4
and below), not on how often: it replaces every poor bridge at once, while 7% of Illinois's rating-4
bridges leave the rated inventory by the next file. Budgets may explain the gap; the MDP has no budget. A
clone of the agency inherits its budget and its habits; the MDP inherits our cost judgement.
:::

## Your turn: predict three failures

1. Is a bridge's rating Markov? Does a rating-6 bridge drop faster when it is old?
2. If it does, does the replacement policy change?
3. The side street's detector cabinet loses power: both east-west detectors read zero. What does
   the DQN do? What would you add?

::: notes
Break it: each question tests an assumption. The MDP assumes the rating is the whole state; the DQN
assumes its sensors tell the truth. Two minutes, written.
:::

## Age matters: the rating is not Markov

![The share of bridges whose lowest rating dropped within two years, by age band, for ratings 4 to 8, with 95% Wilson intervals. Every line rises from the youngest band (0-20 years) to 40-60 years: at rating 6, from 7.1% to 14.2%.](cee_l26_reinforcement_learning/markov.png)

::: notes
A Markov chain on the rating alone predicts flat lines; the data say a 40-to-60-year-old bridge at
rating 6 drops twice as often as one under 20 (14.2% against 7.1% in two years). The rating does not
carry the whole state. CS 441's DQN fed four stacked frames for the same reason: one frame does not
say which way the ball moves. The fix is the same: add what is missing to the state. With age in the
state (four 20-year bands), a rating-6 bridge under 20 reaches poor in 44.3 years, not the 36.0 the
rating-only chain predicts. The intervals treat a bridge's pairs as independent, so they are
somewhat too narrow.
:::

## The policy survives the missing state

| At 2%, poor costing 0.10 | Replaces at |
|:-------------------------|:------------|
| Rating only | 4 and below |
| Rating and age, 0-20 years | 4 and below |
| Rating and age, over 60 years | 4 and below |
| Extra cost of ignoring age | 0.00% |

::: notes
The forecast is wrong, and the decision is not. Replacing pays only once a bridge is poor, and a
poor bridge costs the same at 4 as at 3, so how fast it drops does not change the choice; above 4,
no age band drops fast enough to make an early replacement pay. So the rating-only policy costs
nothing extra in the age-aware world. That is a finding about this decision, not a licence: a policy
that acted earlier (preventive work at 5 or 6) would depend on age. Pontis, as Golabi and Shepard
(1997) describe it, models deterioration as a Markov chain on condition states.
:::

## Dead detectors starve the side street

![The side street's stopped vehicles over a held-out hour, under the first trained DQN. With working detectors the queue stays short. With both east-west detectors reading zero, this DQN never serves the side street, and its queue grows until both approaches are full. With a fault check, five minutes of zero readings switch control to Webster's hourly plan and the queue clears.](cee_l26_reinforcement_learning/stuck.png)

::: notes
The DQN trusts its inputs. With the side street's detectors dead, two of the three trained DQNs
never turned the side street green: none of its 8,236 vehicles over the ten hours got through. The
third served all of them, with a mean delay of 640.6 s per vehicle. Over the three, delay rises
from 17.1 s to at least 921.9 s per vehicle, and 549 vehicles an hour never get through (counted at
the end of the clearing time, so a lower bound). The fix is the one signal controllers already
carry: a detector that reads no vehicle for 60 decisions in a row (five minutes or a little more;
the first reading is at t = 0) is declared faulty (here between 295 and 315 s), and control falls
back to a fixed plan: 35.9 s, worse than the DQN's 17.1 s, far better than 921.9 s. Lecture 25
again: a learned component behind a check.
:::

## Rewards get what they pay for

| Reward | What the optimizer did |
|:-------|:-----------------------|
| Agency cost only, nothing for poor bridges | never replace |
| The same at 7% a year, poor at 0.10 | never replace |
| People's approval of a chatbot's answer | agree with the user (sycophancy) |

::: notes
CS 441: it is hard to force a model to earn a reward the way you want it to. The optimizer is not
wrong in any row; the reward is. The first two rows are today's bridge MDP. The third is RLHF's
known failure: people prefer answers that agree with them, so a reward model learned from their
choices rewards agreement (Sharma et al., 2024). The engineer's job is the reward, and checking what
the optimizer does with it.
:::

## Where RL lives today

| Where | What runs |
|:------|:----------|
| Bridge management (AASHTOWare BrM, once Pontis) | Markov deterioration, optimized actions |
| Chat models: RLHF | a reward learned from people's choices, PPO |
| Signal control in the field | actuated, or adaptive (SCOOT, SCATS); RL mostly simulated |
| Games and robots | DQN, PPO, and their successors |

::: notes
Pontis (Golabi and Shepard, 1997) brought Markov deterioration models and optimized maintenance to
US bridge networks; its successor is AASHTOWare BrM. It models deterioration as a Markov chain on
condition states, the assumption tested today. RLHF (Ouyang et al., 2022): people rank a model's
answers, a reward model learns their preferences, and PPO tunes the language model to it. Signal
control: many field signals are actuated, extending each green while detectors see cars; the
adaptive systems some cities run (SCOOT, SCATS) retune designed plans. Fixed and actuated control
are the baselines RL studies compare against, and RL has beaten them mostly in simulation (Wei et
al., 2019).
:::

## Vote: what should the city install?

- **A.** Actuated control, as SUMO ships it.
- **B.** Fixed plans per 15 minutes, tuned and retimed by an engineer.
- **C.** The DQN.
- **D.** The DQN, with a detector-fault check that falls back to a fixed plan.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: vote, discuss, revote, then explain. The evidence, in simulation with assumed
demand. A: 16.3 s per vehicle, with no schedule and no training. B: 17.7 s tuned (18.7 s as
Webster's formula gives it), and it needs the demand schedule, which changes: with the same quarter
hours in reverse order, the 15-minute plans reach 216.0 s tuned and 324.0 s untuned, until an
engineer retimes them (18.1 s). C: 17.1 s, and 17.1 s on the reversed demand too, but behind A in
every run, and at least 921.9 s when the side street's detectors die. D: C's delay with the fault
covered; our test fell back to the hourly plan (35.9 s). A is the bar: no DQN run beat it, on the
demand it trained on or the reversed one (16.5 s). The DQN beats Webster's plans, and the tuned
plan in two runs of three (the third ties it), and it does not need the schedule; before any
purchase it must beat actuated control, in a field trial with the fallback in place. Actuated control relies on its detectors too; how it fails when they die was not
tested here.
:::

## When to use reinforcement learning

| You have | Reach for |
|:---------|:----------|
| A model you can count (P) and few states | value iteration: no learning |
| A signal with detectors | actuated control first: the bar to beat |
| A simulator, not P; discrete actions | Q-learning, a DQN |
| Continuous actions, or a stochastic policy | PPO |
| Records of what experts did | behavioural cloning (their habits too) |
| A controller in the field | a fault check and a fixed fallback |

::: notes
The first row is the reminder: if you can write P down, plan; nothing needs learning. The second is
today's result: a learned controller is worth its training only if it beats what the field already
runs; here SUMO's untuned actuated control beat every trained DQN. Learning from a simulator
inherits the simulator's assumptions (today, assumed demand on one intersection). And a learned
controller fails in ways a fixed plan cannot: it trusts its sensors.
:::

## Exit question

A vendor's RL signal controller "cut delay 30% in simulation". What would you ask before the city
buys it?

**In the notebook** (`cee_l26_reinforcement_learning.ipynb`): value iteration, Q-learning, the
intersection in SUMO, a DQN, and PPO.

::: notes
Any two of: 30% against which baseline (a stale plan, a tuned one, or actuated control?); on demand
it was trained on, or held out, or with the peak moved?; how many training runs, and their
spread?; what happens when a detector fails?; was the simulator calibrated to the corridor's
counts?; what is the fallback, and who decides when it takes over?
:::

## Things to remember

:::::: {.columns}
::: {.column}
- RL: decisions whose costs come later.
- Bellman: cost now, plus discounted cost next.
- Known P: value iteration. Experience: Q-learning.
- DQN: replay and a target net; PPO: clipped steps.
- Beat the field's baseline; test reward and state.

**Next:** fairness and accountability.
:::
::: {.column}
![Thumbnails of today's figures: the policy map, Q-learning's error, the held-out delays of every controller, and the dead detectors.](cee_l26_reinforcement_learning/remember.png)
:::
::::::

::: notes
CS 441's take-aways, on CEE problems: reinforcement learning applies when a sequence of actions
reaches the goal; Q-learning predicts the long-term reward of a state and action; PPO improves the
policy directly; and rewards are hard to specify, whether for a chatbot or for a bridge program.
Some slides follow the structure of Derek Hoiem's CS 441 lecture on reinforcement learning.
:::

## References

:::::: {.columns}
::: {.column}
- Bellman (1957). *Dynamic Programming.* Princeton.
- Watkins & Dayan (1992). Q-learning. *Mach. Learn.* 8.
- Mnih et al. (2015). *Nature* 518.
- Schulman et al. (2017). PPO. arXiv:1707.06347.
- Ouyang et al. (2022). RLHF. *NeurIPS.*
- Sharma et al. (2024). Sycophancy. *ICLR.*
:::
::: {.column}
- Madanat et al. (1995). *J. Infrastruct. Syst.* 1(2).
- Golabi & Shepard (1997). Pontis. *Interfaces* 27(1).
- OMB (2026). Circular A-94, App. C.
- Webster (1958). Road Res. Tech. Paper 39.
- Lopez et al. (2018). SUMO. *IEEE ITSC.*
- Wei et al. (2019). arXiv:1904.08117.
- Pardo et al. (2018). *ICML.*
:::
::::::
