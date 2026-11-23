---
title: Language Models and Retrieval
subtitle: "Lecture 19 · Can a model answer a crew's question from the manual?"
author: Wilfredo Torres Calderon
lang: en-US
---

## Warm-up: three questions, Lectures 4–18

On your own, no notes, two minutes:

1. Lecture 4: two vectors point the same way; one is twice as long. Their cosine similarity?
2. Lecture 17: BERT filled in a hidden word from both sides. What changes if it may look
   only to the left?
3. Lecture 18: CLIP screened tiles with no labels. What did it still need labels for?

**Today:** a model that predicts the next word, asked about the manual for signs and signals.

::: notes
Give the answers before moving on.
1. 1: cosine ignores length. Today's retrieval ranks the manual's sections by the cosine
   between a question's embedding and each section's.
2. It can then generate text, one word after another: that is GPT. The change is one mask in
   attention (today's picture).
3. To set the threshold: the 90% recall target needed labelled validation photos. Today the
   same split shows up for text: a prompt with no examples, then a few, then labels.
:::

## Two jobs: can a model do either?

| Job | Data | A wrong answer costs |
|:----|:-----|:---------------------|
| Code each severe-injury report by event type | 13,572 labelled construction reports, 2015–2022 | a miscounted hazard trend |
| Answer crews' questions: how high, how far, how long | the MUTCD: 954 sections, 400,027 words | a sign at the wrong height |

::: notes
The question for today: can a language model do either job, and what must a person check?
Two public texts. OSHA's Severe Injury Reports (Lecture 17): every report has a narrative and
an OIICS event code, coded by OSHA staff; we keep construction (NAICS 23) and the four event
types with over 1,000 construction reports each (the other types are 4% of them). The
Manual on Uniform Traffic Control Devices, 11th Edition (FHWA, December 2023): the national
standard for signs, signals, and markings, adopted in federal regulation (23 CFR 655,
Subpart F); states adopt it, some with their own supplements. A crew's question ("how high
must this sign be?") has an answer in the manual, and a wrong one is a liability.
:::

## Your turn: finish the report

> An employee was descending an 8-foot ladder. When he was on the fourth rung, the ladder …

- Write the next word, and how sure you are, as a probability.
- Then the next three words.

::: notes
Two minutes, alone. The report is a real test-year one (2023–2025): drawn at random among the
1,165 construction fall reports with "fell" between words 9 and 25, cut just
before that word. The whole report: "An employee was descending an 8-foot ladder. When he was on the fourth rung, the ladder fell down and the employee fell on top of it, resulting in a broken arm." Keep the guesses: the next slides show
what GPT-2 guesses, before and after it reads 20,000 reports.
:::

## One token at a time, looking back

![Left: a grid with the report's GPT-2 tokens on both axes, from the start token through "the ladder"; only cells on or below the diagonal are shaded (each token sees itself and earlier tokens), and the last row is dark: the position that predicts the next word. Right: two empty panels with question marks, for the next-token probabilities of GPT-2 as released and after 20,000 reports.](cee_l19_language_models/picture_1.png)

::: notes
The one picture. GPT is Lecture 17's transformer with one change: in attention, position t may
look only at positions 1 to t (the shaded triangle). The last row reads the whole prefix and
outputs a probability for every token of the vocabulary (50,257 for GPT-2) as the next one.
The tokens are GPT-2's byte-pair pieces (Lecture 17); · marks a leading space. Ask for the
class's guesses for the next word, and their probabilities.
:::

## One token at a time, looking back

![The same grid; the right panels now show the five likeliest next tokens. GPT-2 as released: ·was, ·fell, ·broke, ·snapped, ·collapsed with probabilities 0.09, 0.08, 0.06, 0.03, 0.03. After 20,000 reports: ·slipped, ·fell, ·tipped, ·dropped, ·became with probabilities 0.31, 0.20, 0.07, 0.04, 0.03.](cee_l19_language_models/picture_2.png)

::: notes
The report's next word is "fell". GPT-2 as released gives it 8% (rank 2);
after one pass over 20,000 training-year reports, 20% (rank 2). Compare with the class's
probabilities: a person who has read injury reports guesses like the fine-tuned model. Nothing
here was labelled: the report's own next word is the target, as BERT's hidden word was.
:::

## One objective, four ways to use it

| Model (year) | Parameters | How it is used |
|:-------------|:-----------|:---------------|
| GPT-1 (2018) | 117 million | fine-tuned for each task |
| GPT-2 (2019) | 124 million to 1.5 billion | zero-shot: the task in words |
| GPT-3 (2020) | 175 billion | few-shot: examples in the prompt |
| ChatGPT (2022) | not published | tuned on instructions and human preferences |
| Today | GPT-2, 124 million; Qwen3.5, 0.75 billion | all four, on the reports and the manual |

::: notes
CS 441's arc, in one table: every model here is trained the same way, to predict the next
token of web text; what changed is the size and how it is used. GPT-1 was fine-tuned per task;
GPT-2 showed tasks can be stated in words (zero-shot); GPT-3 that a few examples in the prompt
work without any training (in-context learning); ChatGPT added supervised fine-tuning on
demonstrations and reinforcement learning from human feedback. Today: GPT-2's smallest model
(fine-tuning), and Qwen3.5-0.8B (Qwen Team, 2026; Apache-2.0) in its two releases: the base
model, pretrained only, as a size reference, and the chat model (zero- and few-shot, and
reading excerpts). Both Qwen releases' text models have 752,393,024 parameters (the releases
also ship a vision encoder, unused here), small enough for a laptop; every model here runs in
float32.
:::

## A text is a chain of next words

![The chain rule and the loss above bars for each token of the attempt's sentence through "fell": the surprise in bits, -log2 of the probability the model gave that token, for GPT-2 as released (grey) and after 20,000 reports (blue). Surprise is high at the start and at words specific to the report, and falls after fine-tuning for 21 of the 22 tokens; "fell" drops from 3.7 to 2.3 bits.](cee_l19_language_models/surprise.png)

::: notes
Assumption: each token can be predicted from the tokens before it, and the training text is
like the text we will see. The chain rule makes this exact for any order; left to right is a
choice. Training minimizes the average surprise, the cross-entropy of Lecture 6 with the next
token as the label, at every position at once: the mask lets all T predictions train in one
pass. Perplexity = e^loss, the number of equally likely choices the surprise amounts to
(CS 441 writes 2^entropy in bits; the same quantity). Our tokens and theirs differ, so we will
compare models in bits per character.
:::

## Your turn: predict the check

```python
ids = tok(start + text)                     # 22 tokens after <start>
logits = model(ids).logits                  # (T, V)
ours = cross_entropy(logits[:-1], ids[1:])  # position t scores token t+1
hf = model(ids, labels=ids).loss            # Hugging Face's own loss
bug = cross_entropy(logits, ids)            # the shift forgotten
ids2 = ids, last token changed to " crane"  # do earlier logits move?
```

How close are `ours` and `hf`? Is `bug` higher or lower? Do earlier positions' logits move?

::: notes
Two minutes, written. Then the next slide. The text is the attempt's sentence through "fell".
:::

## Same loss; the mask holds

| Check, the attempt's sentence | Value |
|:------------------------------|------:|
| Our loss, nats per token | 3.7595 |
| Hugging Face's | 3.7595 |
| Difference | 1.7e-07 |
| The shift forgotten | 10.15 |
| Last token changed: largest change in any earlier logit | 0 |
| … in the last position's logits | 23.46 |

::: notes
Ours is computed in float64 from the same logits, Hugging Face's in float32: they agree to
1.7e-07. Forgetting the shift scores each position on the token it already sees,
which a next-token model rarely predicts again: the loss jumps to 10.15. Changing the last token
leaves every earlier position's logits exactly as they were (largest change 0): the mask holds, so no position
has seen its own answer.
:::

## 400 reports beat 6× the parameters

![Test bits per character (2,000 reports from 2023–2025) against training-year reports seen by GPT-2 in one pass: 1.08 as released, then 0.82, 0.77, 0.73, 0.71, and 0.68 after 20,000 reports. Dashed line: token frequencies alone, 1.78. Dash-dot line: Qwen3.5-0.8B-Base, pretrained only, as released, 1.03.](cee_l19_language_models/bits.png)

::: notes
Bracket: token frequencies alone (a unigram model, Lecture 17's counts) give 1.78 bits per
character; GPT-2 as released 1.08. One pass over 20,000 reports (minutes on one GPU, five
scorings of the test reports included; AdamW, rate 5e-5, gradients clipped, nothing tuned)
brings it to 0.68. Qwen3.5-0.8B-Base, a pretrained model with 752,393,024 parameters against
GPT-2's 124,439,808, scores 1.03 as released; its chat-tuned release, after the same start
token, scores 1.29, so the reference is the base model, the fair one for raw text. Scaling laws (Kaplan et al., 2020;
CS 441) say loss falls predictably with parameters and data, measured on text like the
training text. On a narrow kind of text, a little of it beats a lot of size: after 400
reports (25 steps, seconds) GPT-2 scores 0.82, below Qwen's 1.03. Training cut-off: Qwen3.5
(released February 2026) may have read these 2023–2025 reports, and the 2023 MUTCD, in its
training text; GPT-2 (2019) cannot have. 0.4% of the training reports were cut at 128 tokens.
:::

## Fluent is not true

| Model | Continues "… fourth rung, the ladder" |
|:------|:------------------------------------|
| GPT-2 as released | was about to fall. "I was just trying to get up and get out of the way," said the employee, who asked not to |
| After 20,000 reports | slipped and he fell to the ground, breaking his left femur. |
| The real report | fell down and the employee fell on top of it, resulting in a broken arm. |

::: notes
Greedy decoding: the likeliest token, every step. The fine-tuned model writes a fluent
report and stops by itself (it learned where reports end), but the details are not this
report's: in the real report the ladder fell and the employee broke an arm. GPT-2
as released writes news copy, with a quote from the employee. A language model is a model of
text, not of events: what it writes must be checked against a source.
:::

## A chat is one long document

```text
<|im_start|>system
You answer questions about the MUTCD (11th Edition, 2023), the U.S. manual
for traffic signs, signals, and markings. Answer in one or two sentences
with the number and the section. If the MUTCD does not cover the question,
say: Not in the MUTCD.<|im_end|>
<|im_start|>user
Posted speed limits must be multiples of what?<|im_end|>
<|im_start|>assistant
<think>

</think>

```

::: notes
What an instruct model reads (the system line wrapped for the slide): the system's
instruction, the user's turn, and a cue for the assistant's turn, marked by special tokens.
The model predicts the next token, exactly as GPT-2 did. Qwen3.5 can "think" first, writing a
chain of thought between the think tags; we switch that off (enable_thinking=False), and the
template writes the block empty, so the answer comes next. CS 441 (GPT-3 to ChatGPT):
supervised fine-tuning on written demonstrations of good answers, then reinforcement learning
from human feedback (Ouyang et al., 2022), teach the model to continue such documents
helpfully. Everything we do next is text placed in this document: examples, excerpts,
instructions.
:::

## Examples go in the prompt

```text
system:    You code OSHA severe-injury reports by event type.
           Answer with the digit only.
user:      Event types: 2 Transportation incidents, 4 Falls, ...
           Report: <a training-year report>   Event type:
assistant: 4
           ... k examples per type, in random order
user:      Event types: ...  Report: <the report to code>  Event type:
assistant: (empty think block) ?   P(2), P(4), P(5), P(6): largest wins
```

::: notes
The prompt in outline (the real one lists the four OIICS division names in full, one per line,
in every user turn). In-context learning (GPT-3, Brown et al., 2020): no weights change; the examples sit in the
prompt. With k = 0 the model has only the type names. We read the model's probabilities for
the four digits rather than its free text, so every report gets one of the four answers; with
thinking off, the digit comes right after the empty think block. The same examples train a
logistic probe on BGE sentence embeddings (Lecture 18's probe, on text), so the two methods
see the same labels. Thirty examples per type make a prompt of 11,230 to 11,872 tokens; the
examples are read once and their cache reused for every report (the notebook shows how, and
checks it against each prompt read alone). Tixier et al. (2016) extracted injury precursors
from such reports with hand-written rules; today the model reads the categories from examples.
:::

## Labels pay more in a probe

![Balanced accuracy on 1,000 test-year construction reports against labelled reports per event type. Qwen3.5-0.8B with examples in the prompt: 0.45 with none, median 0.70 with 1 per type, 0.73 with 3, 0.80 with 10, 0.81 with 30. Probe on BGE embeddings: median 0.66 with 1, 0.75 with 3, 0.82 with 10, 0.84 with 30, 0.87 with 100, and 0.91 with all 13,572 training reports. Chance: 0.25.](cee_l19_language_models/coding.png)

::: notes
Balanced accuracy: the mean of the four types' recalls, so answering "falls" every time scores
0.25 although falls are 44% of the test reports. Every method sees the same 1,000 test reports
(2023–2025) and, at k = 1, 3, 10, and 30, the very same examples (three draws; the band and
bars show the range). Compare them draw by draw. With one example per type they are close: the
prompt is ahead in two of the three draws. From three per type on, the probe is ahead in every
draw, by 0.015 to 0.068: small gaps, but the same sign nine times out of nine. The prompt
reaches 0.81 with 30 per type, a prompt of 11,230 to 11,872 tokens; the probe keeps paying, to
0.91 with all 13,572 labels, which no prompt can hold (as examples they would make
1,278,161 tokens, against Qwen3.5's context of 262,144). The probe also costs one small
embedding per report (BGE, 109,482,240 parameters) instead of a pass of a
0.75-billion-parameter model. The model always answered with one of the four digits (its top
token was a digit for every report).
:::

## Retrieve, then read

:::::: {.columns}
::: {.column}
| Rank | Section | Cosine |
|:----:|:--------|-------:|
| 1 | 2A.15 | 0.811 |
| 2 | 2A.15 | 0.725 |
| 3 | 2N.06 | 0.721 |
| 4 | 2A.15 | 0.711 |
| 5 | 2C.73 | 0.694 |
:::
::: {.column}
**Question:** What is the minimum height to the bottom of a warning sign on a freeway?

**Answer, from these five:** 2A.15 Mounting Height: the minimum height, measured vertically from the bottom of the sign to the elevation of the near edge of the pavement, of signs installed on freeways shall be 7 feet.
:::
::::::

::: notes
Retrieval-augmented generation (Lewis et al., 2020). The manual is cut into 4,032 overlapping
150-word excerpts, each headed by its section number and title; BGE embeds each once. A question
is embedded the same way (with BGE's query prefix), the five excerpts with the largest cosine
(Lecture 4's nearest neighbours) are pasted into the prompt above the question, and the model
answers from them. The answer is in Section 2A.15, Paragraph 11 (Standard): 7 feet, in the
excerpt at rank 1, and the model names the section, as the instruction asks. The sentence is
the model's own: Paragraph 4's wording (the rural rule, 5 feet) with Paragraph 11's 7 feet for
freeways.
:::

## The manual labels its own test set

![Recall against k, the number of sections returned, on the manual's 221 cross-references: BGE 0.65 at 1, 0.85 at 5, 0.95 at 20; TF-IDF 0.52, 0.79, 0.94. A star: our 16 questions, a right section among the five excerpts, 0.94.](cee_l19_language_models/recall.png)

::: notes
Labels for free: the manual points to itself. Section 1C.02 says that Section 1D.11
"contains additional information about crashworthiness". Each such description is a query; the cited section is the right
answer; the citing section is excluded. 221 references to 174 sections. TF-IDF matches words; BGE
matches meaning and wins most at k = 1. For our 16 questions a right section was among the five
excerpts for 15 (five excerpts can come from fewer than five sections), and the sentence that
holds the rule for 13. That is the ceiling for the answers, by section: an excerpt not
retrieved cannot be read.
:::

## Your turn: predict four failures

1. Code the reports with no examples in the prompt.
2. Draw another set of examples.
3. Ask the 22 MUTCD questions from memory, 6 of them about things the manual does not cover.
4. Answer from the five excerpts when the manual has several similar rules.

Write what happens to each.

::: notes
Two minutes, alone. Then four reveals.
:::

## No examples: two types vanish

![Bars for the four event types: the test reports' own codes (grey), 65 transportation, 439 falls, 116 exposure, 380 contact; Qwen's answers with no examples (orange): 0, 455, 545, 0; with 3 per type (blue): 23, 448, 131, 398.](cee_l19_language_models/zero_shot.png)

::: notes
First reveal. With only the type names, Qwen3.5-0.8B never answers transportation or contact:
of the 380 contact reports it put 294 under exposure and 86 under falls, and of the 65
transportation reports 49 under exposure. OSHA's coders follow the OIICS manual's rules; the
model only has the names. Is it the list? The notebook rotates the order of the four types
(each listed first once, second once, and so on): contact gets 0 answers listed last, as here,
8 listed first, 731 listed second, and 335 listed third; transportation never gets more than
19 in any order. Language models are known to favour some answer labels and positions over
others (Zhao et al., 2021). Three examples per type bring contact back (398 answers in the
first draw, for 380 reports); transportation stays short (23, for 65).
:::

## Other examples, another score

| Balanced accuracy, 1,000 test reports | Draw 1 | Draw 2 | Draw 3 |
|:--------------------------------------|-------:|-------:|-------:|
| Prompt, 1 example per type | 0.73 | 0.70 | 0.62 |
| Prompt, 3 per type | 0.73 | 0.73 | 0.63 |
| Probe, the same 1 per type | 0.75 | 0.66 | 0.44 |
| Probe, the same 3 per type | 0.78 | 0.75 | 0.68 |

::: notes
Second reveal. Same test reports, same model; only the drawn examples differ. With one example
per type the prompt scores from 0.62 to 0.73 over the three draws, the probe from 0.44 to
0.75: for the probe, which one label per type you happened to draw decides the score; the
prompt moves less, and its three-example draws still range from 0.63 to 0.73. With 30 per
type the prompt's draws range from 0.78 to 0.82. Which reports are the examples, and their
order, is a hyperparameter: choose them on validation reports, never on the test, and report
the spread.
:::

## From memory, it invents numbers

| Asked from memory | Qwen3.5-0.8B answered |
|:------------------|:----------------------|
| Minimum height to the bottom of a warning sign on a freeway? | "… 12 inches (305 mm) from the ground. This is found in Section 11.1.1 of the MUTCD …" |
| Maximum live-load deflection for a steel bridge girder? | "… generally limited to 1/200 of the span length (0.5%) … Not in the MUTCD." |
| How far from the travel lane should a post-mounted sign be set back? | "Not in the MUTCD. … it only requires them to be placed at least 10 feet from the edge of the travel lane." |

::: notes
Third reveal (answers shortened; the notebook prints them whole). From memory the model
answered none of the 16 answerable questions right: it got
5 wrong (two of them only strings of "1." and ones) and declined 11. It declined all 6 that the
MUTCD does not cover (concrete, asphalt, girders: other manuals' business), by the list of
declining phrasings set after a pilot run; by the list fixed before it, 5, and the design-life
answer ("does not explicitly define") counts as invented. Either way, 9 of its 17 declines (9
of 16) still state a number of their own: "not in the MUTCD", and then a rule it made up,
like the girder's 1/200 of the span, or a 10-foot offset where the manual says 12 feet
(Section 2A.16). Three answers cite sections in a numbering the manual does not use (11.1.1,
10.1.1.1). Nothing in a made-up number signals that it is made up. The right answer to the
first row is 7 feet (Section 2A.15); to the second, "not in the MUTCD". Training cut-off: the
2023 MUTCD may have been in the model's training text; if it was, the model did not recall it.
:::

## Right section, wrong sentence

| Question | Rule's sentence at rank | Answer with five excerpts | The manual |
|:---------------|:----------:|:--------------------------|:-----------|
| Walking speed for pedestrian clearance? | 4 | "3 feet per second … where an extended push button press function has been installed" | 3.5 ft/s (Guidance) |
| Stop line before a crosswalk? | not retrieved | "Section 3B.19: 20 to 50 feet in advance of the nearest crosswalk line" | 4 feet (3B.19) |
| Post-mounted sign's offset? | 4 | "Where an expressway median is 12 feet or less in width, …" | 12 feet (2A.16) |

::: notes
Fourth reveal. Each answer had a right section among its excerpts (at ranks 1, 1, and 2) and
answered from the wrong sentence. The MUTCD marks each paragraph Standard (shall), Guidance
(should), Option (may), or Support, and keeps several similar rules in one section. Section
4I.06 gives 3.5 ft/s as Guidance, in the excerpt at rank 4; its 3 ft/s is for the walk interval
and the clearance together, and the push button belongs to an Option of up to 4 ft/s: the
answer stitched two of them. For the stop line, the 4-foot sentence (3B.19, Paragraph 13) was
in none of the five excerpts: the excerpt of 3B.19 at rank 1 begins after it, so the model read
the section's rule for yield lines at uncontrolled multi-lane crossings (20 to 50 feet). That
one is a retrieval miss inside the right section, not a misreading. The setback's "12 feet"
comes from a sentence about expressway medians in Section 2E.20, ranked first, while the rule's
own sentence sat at rank 4: the right number from the wrong rule, which a number test scores as
right. Two more did the same: the Yield Here To Pedestrians distance came from the yield-line
rule, and the Crossbuck strip from Section 8B.04's rule for the support, as the blade's rule
in 8B.03 was not retrieved. A reader checking the excerpt and its section catches all of them.
:::

## Excerpts: 13 numbers right, 5 cited

![Outcomes for the 16 questions the manual answers and the 6 it does not. With five excerpts: 5 with the number and its section, 8 with the number but not the section, 1 declined, 2 wrong; all 6 unanswerable ones declined. From memory: 11 declined and 5 wrong; all 6 unanswerable ones declined.](cee_l19_language_models/scorecard.png)

::: notes
The decisive figure. The same model and instruction ("answer with the number and the section;
if the MUTCD does not cover it, say so"); only the excerpts differ. A number test alone would
count 13 of 16 right with excerpts; asking for the section that holds the rule too leaves 5.
Of the 8 with the number but not the section, 5 copy the rule's own words without naming its
section, and 3 took the right number from another rule (the last slide). Retrieval put a right
section among the five for 15 of the 16, and the rule's own sentence for 13; 9 of the 11
answers that fell short had that sentence among their excerpts: here the 0.75-billion-parameter
reader failed more often than the retrieval. Bracket: from memory, 0 of 16; retrieval's
ceiling, 15 of 16 by section (13 with the rule's sentence). The prompt, the questions, and the
excerpts were fixed before any answer was scored; the scorer's section rule and its list of
ways to decline were set after a pilot run of this model (the from-memory counts are given
both ways). 22 questions are few, so read the counts, not percentages. Larger models may read
better; measure before trusting that.
:::

## Who signs the answer?

| Do not use it alone when | Because, today |
|:-------------------------|:---------------|
| A crew sets a height, offset, or time from it | with excerpts, 2 of 16 wrong and 3 right numbers from the wrong rule |
| A section holds several similar rules | it read the stop line's 20 to 50 feet from the yield-line rule |
| The state has a supplement | the excerpts are the national manual only |
| Its codes feed a hazard trend | with no examples it never answered "contact"; the test has 380 contact reports |

::: notes
When not to use it: wherever an answer becomes a dimension in the field without a person
reading the source. States adopt the national MUTCD, some with a supplement or their own
edition; a retriever over the national text cannot know the state's rule. Impact
(accountability): a sign set at the wrong height is the agency's liability and the engineer's
responsibility, not the vendor's; "the assistant said so" is not a citation. ABET's
professional responsibility (SO 4): the engineer answers for the tools used. Hazard trends
built on machine-coded reports inherit the coder's biases, here two types that vanish.
:::

## Vote: what should the crews get?

- **A.** A chat model that answers from memory.
- **B.** Retrieval plus the model: an answer, with the excerpt shown.
- **C.** Retrieval only: the five excerpts, no generated answer.
- **D.** Nothing new: the PDF and its search box.

Vote alone. Discuss with a neighbor for two minutes. Vote again.

::: notes
Peer instruction: vote, discuss two minutes, revote, then explain. The evidence: A answered 0
of 16 from memory, and 9 of its 17 declines still gave a made-up number; it is out. B: 13 of
16 right numbers, only 5 with the section that holds the rule, 2 wrong, 1 declined, none
invented; the rule's own sentence was among the excerpts for 9 of the 11 that fell short, so
showing the excerpt and its section lets the crew catch most of them. C: the right section
among five for 15
of 16; no generated errors, but the crew reads five excerpts. D: the baseline a crew already
has. A defensible answer: B with the excerpt and section number always shown, or C where a
wrong number is costly; never A. The vote is about what the person checks, not about the model
alone.
:::

## Where language models live

| Today | What it is | Lecture |
|:------|:-----------|:--------|
| ChatGPT, Claude, Gemini | next-token models tuned on instructions and human preferences | today |
| Reasoning models (o1, DeepSeek-R1) | long chains of thought, trained by reinforcement learning | 26 |
| Assistants with file search | retrieval, then an answer citing the source | today |
| Coding assistants, agents | models that call tools and check the results | 25 |
| Embedding search | vectors like BGE's in a vector database | 4, today |

::: notes
The chat assistants are next-token models like today's, scaled up (Kaplan et al., 2020) and
tuned on instructions and preferences (Ouyang et al., 2022). Reasoning models write a chain of
thought before answering (Wei et al., 2022) and are trained to do so by reinforcement learning
(DeepSeek-AI, 2025; Lecture 26). Agents call tools, a search or a calculator, and read the
results back into the document (Lecture 25). File search in today's assistants is retrieval
plus generation, as here; the embeddings sit in a vector index such as Lecture 4's FAISS.
:::

## Exit question

A vendor's assistant "answers any question about the MUTCD." Name the two numbers you would
ask for before your crews use it, and how you would measure them.

**In the notebook** (`cee_l19_language_models.ipynb`): GPT-2's loss by hand, fine-tuning on
the reports, prompts against probes, and retrieval over the manual.

::: notes
A good answer: (1) how often the right section is among what it retrieves, on questions with
known answers (like the manual's cross-references, or a set the agency writes); (2) how often
it answers wrong, and how often it invents an answer to a question the manual does not cover,
on questions the vendor has not seen. Both on the agency's own questions, with the counts.
:::

## Things to remember

:::::: {.columns}
::: {.column}
- A language model predicts the next token; a chat is a document.
- Score text in bits per character across tokenizers.
- Examples in a prompt help; labels for a probe help more.
- Retrieval sets the ceiling; the reader can still misread.
- Show the excerpt; a person signs the answer.

**Next:** 1-D signals: rainfall, runoff, and vibration.
:::
::: {.column}
![Thumbnails of today's four figures: the causal mask and next-token probabilities, bits per character against reports seen, balanced accuracy against labelled reports per type, and the outcomes from memory and with excerpts.](cee_l19_language_models/remember.png)
:::
::::::

## References

:::::: {.columns}
::: {.column}
- Radford et al. (2019). GPT-2. OpenAI.
- Brown et al. (2020). *NeurIPS*.
- Kaplan et al. (2020). Scaling laws. arXiv.
- Lewis et al. (2020). *NeurIPS*.
- Zhao et al. (2021). *ICML*.
- Ouyang et al. (2022). *NeurIPS*.
- Wei et al. (2022). *NeurIPS*.
:::
::: {.column}
- Xiao et al. (2024). C-Pack (BGE). *SIGIR*.
- Qwen Team (2026). Qwen3.5. Qwen blog.
- DeepSeek-AI (2025). DeepSeek-R1. arXiv.
- Tixier et al. (2016). *Autom. Constr.*
- FHWA (2023). MUTCD, 11th Edition.
- OSHA. Severe Injury Reports.
:::
::::::
