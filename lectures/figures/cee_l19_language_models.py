"""Figures and numbers for CEE Lecture 19: language models, embeddings, and retrieval.

Run (from ``lectures/``)::

    pixi run -e aml-env python figures/cee_l19_language_models.py \
        --out build/fig/cee_l19_language_models

Writes the PNG figures and ``numbers.json`` (every number quoted on the slides). The lecture
follows the GPT half of CS 441's "Foundation Models: CLIP and GPT" (Fall 2026): generative
pretraining, zero- and few-shot use, chat models, and tools, on two CEE texts:

- OSHA's Severe Injury Reports (Lecture 17's loader; training years 2015-2022, test 2023-2025).
  GPT-2 (Radford et al., 2019; 124M parameters, MIT) predicts the next token of a report, before
  and after one pass over 20,000 training-year reports, scored in bits per character so that
  models with different tokenizers compare; Qwen3.5-0.8B-Base (Qwen Team, 2026; Apache-2.0), a
  pretrained model six times its size, is the reference. Qwen3.5-0.8B, the same model tuned for
  chat (Apache-2.0), codes the event type of construction reports from a prompt with 0, 1, 3,
  10, or 30 examples per type, against a logistic probe on BGE embeddings (Xiao et al., 2024;
  MIT) given the same examples.
- The MUTCD, 11th Edition (FHWA, 2023; public domain): sections retrieved by BGE and TF-IDF for
  the manual's own cross-references, and 22 questions (16 it answers, 6 it does not) answered
  by Qwen3.5-0.8B from memory and from the five retrieved excerpts. The prompt, the questions,
  and the five excerpts were fixed before any answer was scored. The scorer was revised after a
  pilot run of this model (2026-11-19): an answer must cite a section that holds the rule, not
  only give the number, and "does not define" and similar phrasings count as declining. The
  outcomes are also counted under the list of declining phrasings fixed before the pilot
  (``DECLINE_PILOT``), so the slides can give both counts. Retrieval is scored twice: by
  section (an excerpt of a section that holds the rule) and by sentence (an excerpt that holds
  the rule's own words).

Bits per character: every model scores each report after the same start token,
``<|endoftext|>`` (GPT-2's BOS, and the token both Qwen3.5 releases separate documents with),
named rather than read from each tokenizer: the chat model's EOS is ``<|im_end|>``, a chat
turn's end.

Models: every one runs in float32 (a T4 has no native bfloat16), with TF32 off. Qwen3.5 is a
vision-language release; ``AutoModelForCausalLM`` loads its text model only. The pinned
revisions ship no ``generation_config.json``, so every generation setting is passed: greedy, no
repetition penalty, stop at ``<|im_end|>`` or ``<|endoftext|>``. Thinking is off
(``enable_thinking=False``: the template writes an empty think block). Each coding prompt is
read in two parts: the system turn and the examples, the same for all 1,000 reports, run once
and cached; then the reports' turns, up to 32 at a time (fewer when the cache is long) on copies
of that cache, padded on the right (a report never sees the padding after it). The first
reports of every setting are checked against their prompt read alone, unbatched and unpadded.
A prompt longer than 2,048 tokens is read in pieces of that length, each continuing the cache.

Training cut-off: Qwen3.5 was released in February 2026 and its training cut-off is not
published, so its training text may include the 2023-2025 test reports and the 2023 MUTCD;
GPT-2 (2019) cannot have seen either.

Data: ``scripts/fetch_osha_sir.sh`` and ``scripts/fetch_mutcd.sh``; the MUTCD's text comes from
poppler's ``pdftotext``. Models from the Hugging Face Hub at pinned revisions.
"""

from __future__ import annotations

import copy
import math
import os
import re
import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.nn import functional as F  # noqa: N812 (PyTorch's convention)
from torch.nn.attention import SDPBackend, sdpa_kernel

import aml_style
import title_art
from common import DATA_DIR, parse_out_dir, require, seed_all, write_numbers
from osha import TRAIN_YEARS, osha

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # read when huggingface_hub is imported
import transformers

SEED = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPT2 = "openai-community/gpt2"  # Radford et al. (2019); 124M parameters; MIT
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
QWEN_BASE = "Qwen/Qwen3.5-0.8B-Base"  # Qwen Team (2026); pretrained only; Apache-2.0
QWEN_BASE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
QWEN = "Qwen/Qwen3.5-0.8B"  # the same model tuned for chat and instructions; Apache-2.0
QWEN_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
QWEN_NAME = "Qwen3.5-0.8B"
BGE = "BAAI/bge-base-en-v1.5"  # Xiao et al. (2024); MIT
BGE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
BGE_QUERY = "Represent this sentence for searching relevant passages: "  # BGE's query prefix
MUTCD_PDF = DATA_DIR / "raw" / "mutcd" / "mutcd11thedition.pdf"
CHAT_TURN_END, DOC_END = "<|im_end|>", "<|endoftext|>"  # Qwen's two stop tokens
DOC_START = DOC_END  # every model scores a report after this one token (``doc_start``)
USER_TURN = "<|im_start|>user\n"  # where a chat's user turn starts

# Next-token prediction on the reports
LM_TRAIN = 20_000  # training-year reports for one pass of fine-tuning
LM_TEST = 2_000  # test-year reports scored
TRAIN_TOKENS = 128  # fine-tuning truncates longer reports (the share is reported)
BATCH = 16
LR = 5e-5  # AdamW; a usual rate for fine-tuning GPT-2, fixed in advance, not tuned
EVAL_AT = (0, 400, 1_200, 4_000, 10_000, 20_000)  # reports seen when the test is scored
GENERATE = 30  # tokens of greedy continuation
QWEN_SCORE_BATCH = 4  # reports per pass when Qwen scores bits per character (248,320 logits)

# Event coding: the OIICS event divisions with over 1,000 construction reports (the other
# divisions are about 4% of them, too few to score: a few test reports each)
EVENTS = {
    "2": "Transportation incidents",
    "4": "Falls, slips, trips",
    "5": "Exposure to harmful substances or environments",
    "6": "Contact with objects and equipment",
}
EVENT_SYSTEM = "You code OSHA severe-injury reports by event type. Answer with the digit only."
ICL_TEST = 1_000  # test-year construction reports, the same for every method
SHOTS = (0, 1, 3, 10, 30)  # examples per type in Qwen's prompt (30: about 11,000 tokens)
PROBE_SHOTS = (1, 3, 10, 30, 100)  # labelled examples per type for the BGE probe
SHOT_SEEDS = (0, 1, 2)  # draws of the examples
CODE_BATCH = 32  # at most this many reports per forward pass, on copies of the examples' cache
CACHE_TOKENS = 49_152  # and at most this many cached tokens per pass: memory on a shared GPU
PREFILL = 2_048  # a long prompt is read in pieces of this many tokens, which bounds attention
CHECK_REPORTS = 4  # per setting: the cached, batched pass against each prompt read alone
SHOW_K = 3  # the examples per type set against none in the zero-shot figure

# Retrieval over the MUTCD
CHUNK_WORDS, CHUNK_STRIDE = 150, 100  # overlapping windows within a section
TOP_K = 5  # excerpts given to the model
MAX_ANSWER = 80  # tokens
REPETITION_PENALTY = 1.0  # none: plain greedy decoding (Qwen's model card also sets 1.0)
QA_SYSTEM = (
    "You answer questions about the MUTCD (11th Edition, 2023), the U.S. manual for traffic "
    "signs, signals, and markings. Answer in one or two sentences with the number and the "
    "section. If the MUTCD does not cover the question, say: Not in the MUTCD."
)
DECLINE = re.compile(
    r"not in the mutcd|does not (?:explicitly |directly )?"
    r"(?:cover|address|specify|provide|define|contain|include|mention|state)",
    re.IGNORECASE,
)
# The declining phrasings fixed before the pilot run (2026-11-19), which ``DECLINE`` widened
DECLINE_PILOT = re.compile(
    r"not in the mutcd|does not (?:cover|address|specify|provide)", re.IGNORECASE
)
ODD_SECTION = re.compile(r"Section (\d+(?:\.\d+){2,})")  # a numbering the manual does not use
SECTION_ID = re.compile(r"\b\d[A-Z]\.\d{2}\b")  # the manual's numbering: 2A.16


@dataclass(frozen=True)
class Question:
    """A question for the MUTCD, the manual's sentence that answers it, and the answer's test.

    ``evidence`` is a phrase of the manual (whitespace-normalized); every section containing it
    counts as a correct retrieval, and as a correct section to cite. ``answer`` holds regular
    expressions that a correct answer matches, all of them. Questions the manual does not
    answer have neither.
    """

    text: str
    evidence: str | None = None
    answer: tuple[str, ...] = ()


FT = r"[- ]?(feet|foot|ft)\b"
QUESTIONS = (
    Question(
        "How far from the edge of the travel lane should a post-mounted sign be set back?",
        "minimum lateral offset should be 12 feet",
        (r"\b12" + FT,),
    ),
    Question(
        "What is the minimum height to the bottom of a warning sign on a freeway?",
        "warning signs, and regulatory signs on freeways and expressways shall be installed "
        "with a minimum height of 7 feet",
        (r"\b7" + FT,),
    ),
    Question(
        "How low can a sign be mounted if it sits 30 feet or more from the edge of the road?",
        "30 feet or more from the edge of the traveled way may be installed with a minimum "
        "height of 5 feet",
        (r"\b5" + FT,),
    ),
    Question(
        "How long should the yellow change interval of a traffic signal last?",
        "minimum duration of 3 seconds",
        (r"\b3\b", r"\b6 seconds\b"),
    ),
    Question(
        "What walking speed is used to time the pedestrian clearance interval at a signal?",
        "walking speed of 3.5 feet per second",
        (r"\b3\.5" + FT,),
    ),
    Question(
        "How long are the dashes and the gaps of a broken lane line?",
        "10-foot line segments and 30-foot gaps",
        (r"\b10" + FT, r"\b30" + FT),
    ),
    Question(
        "How wide is a normal longitudinal pavement marking line?",
        "Normal line—4 to 6 inches wide",
        (r"\b4\b", r"\b6 inch"),
    ),
    Question(
        "Posted speed limits must be multiples of what?",
        "speed limits displayed shall be in multiples of 5 mph",
        (r"\b5 ?mph\b",),
    ),
    Question(
        "How tall must traffic cones be on a freeway or at night?",
        "cones shall be a minimum of 28 inches in height",
        (r"\b28 inch",),
    ),
    Question(
        "What is the minimum height of traffic cones on a low-speed road in daytime?",
        "cones shall be not less than 18 inches in height",
        (r"\b18 inch",),
    ),
    Question(
        "What ground clearance is needed under a sign larger than 50 square feet on multiple "
        "breakaway posts?",
        "installed on multiple breakaway posts, the clearance from the ground to the bottom of "
        "the sign shall be at least 7 feet",
        (r"\b7" + FT,),
    ),
    Question(
        "What is the minimum size of a diamond-shaped warning sign on a multi-lane road with a "
        "speed limit above 35 mph?",
        "speed limit is higher than 35 mph shall be 36 x 36 inches",
        (r"\b36 ?(x|×|by) ?36\b",),
    ),
    Question(
        "How far ahead of a crosswalk should Yield Here To Pedestrians signs be placed on a "
        "multi-lane approach?",
        "20 to 50 feet in advance of the nearest edge of the crosswalk",
        (r"\b20\b", r"\b50" + FT),
    ),
    Question(
        "How wide is the reflective strip on the back of each Crossbuck blade at a passive "
        "grade crossing?",
        "retroreflective white material not less than 2 inches in width shall be used on the "
        "back of each blade",
        (r"\b2 inch",),
    ),
    Question(
        "How far in advance of the crosswalk should a stop line be placed at a controlled "
        "intersection?",
        "stop and yield lines should be placed a minimum of 4 feet in advance of the nearest "
        "crosswalk line",
        (r"\b4" + FT,),
    ),
    Question(
        "What diameters are allowed for vehicular signal indications?",
        "three nominal diameter sizes for vehicular signal indications: 4 inches, 8 inches, "
        "and 12 inches",
        (r"\b4\b", r"\b8\b", r"\b12 inch"),
    ),
    Question("What concrete compressive strength is required for a sign post foundation?"),
    Question("What asphalt binder grade should be used for a resurfacing overlay?"),
    Question("What is the maximum live-load deflection allowed for a steel bridge girder?"),
    Question("What is the design life of a traffic signal controller cabinet?"),
    Question("What pavement thickness is required for a bus-only lane?"),
    Question("What slump should the concrete for a curb ramp have?"),
)


# ---------------------------------------------------------------- the models


def gpt2() -> tuple[transformers.PreTrainedTokenizerBase, transformers.PreTrainedModel]:
    """GPT-2 (124M) and its tokenizer; padding reuses the end-of-text token, masked out."""
    tok = transformers.AutoTokenizer.from_pretrained(GPT2, revision=GPT2_REVISION)
    tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(GPT2, revision=GPT2_REVISION)
    return tok, model.to(DEVICE).eval()


def qwen(
    name: str, revision: str
) -> tuple[transformers.PreTrainedTokenizerBase, transformers.PreTrainedModel]:
    """A Qwen3.5 text model in float32, and its tokenizer.

    Args:
        name: ``QWEN_BASE`` or ``QWEN``.
        revision: Its pinned revision.

    Returns:
        The tokenizer and the model, in eval mode on ``DEVICE``.
    """
    tok = transformers.AutoTokenizer.from_pretrained(name, revision=revision)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=torch.float32
    )
    assert type(model).__name__ == "Qwen3_5ForCausalLM"  # the text model, no vision encoder
    return tok, model.to(DEVICE).eval()


def doc_start(tok: transformers.PreTrainedTokenizerBase) -> str:
    """The token every document starts after, for every model here: ``DOC_START``.

    ``<|endoftext|>`` is GPT-2's BOS, and both Qwen3.5 releases (no BOS) separate documents
    with it. It is named, not read from the tokenizer: the chat model's EOS is ``<|im_end|>``,
    a chat turn's end, and scoring after it would give the two Qwen releases different starts.

    Args:
        tok: GPT-2's tokenizer or either Qwen3.5 release's.

    Returns:
        The token's text, checked to be one special token for this tokenizer.
    """
    s = DOC_START
    assert s in tok.all_special_tokens and len(tok.encode(s, add_special_tokens=False)) == 1, s
    return s


def context_length(model: transformers.PreTrainedModel) -> int:
    """The longest input the model's positions cover (GPT-2: 1,024; Qwen3.5: 262,144)."""
    return int(model.config.get_text_config().max_position_embeddings)


def parameters(model: transformers.PreTrainedModel) -> int:
    """Parameters, tied input and output embeddings counted once."""
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- next-token prediction


def nll_nats(
    tok: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
    texts: list[str],
    batch: int = 16,
) -> float:
    """Total negative log-likelihood (nats) of the texts, each token given the ones before it.

    Each text is prefixed with the document start (``doc_start``) so that its first token is
    predicted too; padding is on the right and masked. Logits are float32.
    """
    tok.padding_side = "right"
    start = doc_start(tok)
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(
                [start + t for t in texts[i : i + batch]],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(DEVICE)
            assert enc["input_ids"].shape[1] <= context_length(model)
            logits = model(**enc).logits[:, :-1].float()  # (B, T-1, V): position t predicts t+1
            target = enc["input_ids"][:, 1:].masked_fill(enc["attention_mask"][:, 1:] == 0, -100)
            # Per-token losses summed in float64: a summing reduction on the GPU adds with
            # atomics, whose order (and last digits) change from run to run
            nll = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")  # (B, T-1)
            total += float(nll.double().sum())
    return total


def bits_per_char(
    tok: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
    texts: list[str],
    batch: int = 16,
) -> float:
    """Bits per character: comparable across tokenizers, unlike per-token perplexity."""
    model.eval()
    return nll_nats(tok, model, texts, batch) / math.log(2) / sum(map(len, texts))


def unigram_bits_per_char(
    tok: transformers.PreTrainedTokenizerBase, train: list[str], test: list[str]
) -> float:
    """The naive baseline: GPT-2's tokens, each predicted by its training frequency alone.

    Add-one smoothing over the whole vocabulary, so unseen tokens get a finite cost.
    """
    counts = np.ones(len(tok))
    for ids in tok(train)["input_ids"]:
        np.add.at(counts, ids, 1)
    logp = np.log(counts / counts.sum())
    nll = -sum(logp[ids].sum() for ids in tok(test)["input_ids"])
    return float(nll / math.log(2) / sum(map(len, test)))


def fine_tune(
    tok: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
    train: list[str],
    test: list[str],
) -> dict[str, object]:
    """One pass of next-token training over the reports; the test is scored at ``EVAL_AT``.

    Each report is ``start + text + end-of-text``, truncated at ``TRAIN_TOKENS``; AdamW at
    ``LR`` with PyTorch's default betas, eps, and weight decay (0.01) stated; gradients clipped
    at norm 1. Training uses PyTorch's plain ("math") attention, whose backward pass is
    deterministic, so a rerun reproduces the curve.
    """
    assert all(n % BATCH == 0 for n in EVAL_AT) and len(train) == EVAL_AT[-1]
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    curve = {0: bits_per_char(tok, model, test)}
    tok.padding_side = "right"
    start = doc_start(tok)
    t0, truncated = time.perf_counter(), 0
    for i in range(0, len(train), BATCH):
        model.train()  # dropout on
        texts = [start + t + tok.eos_token for t in train[i : i + BATCH]]
        truncated += sum(len(ids) > TRAIN_TOKENS for ids in tok(texts)["input_ids"])
        enc = tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=TRAIN_TOKENS
        ).to(DEVICE)
        labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
        with sdpa_kernel(SDPBackend.MATH):  # the fused kernels' backward is not deterministic
            loss = model(**enc, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (seen := i + BATCH) in EVAL_AT:
            curve[seen] = bits_per_char(tok, model, test)
            print(f"  {seen:,} reports: {curve[seen]:.3f} bits/char", flush=True)
    model.eval()
    return {
        "bits_per_char": {str(k): v for k, v in curve.items()},
        "seconds": time.perf_counter() - t0,
        "truncated_share": truncated / len(train),
    }


def attempt_report(test: pd.DataFrame) -> dict[str, object]:
    """A test-year construction fall report, cut just before the word "fell".

    Drawn at random (seed ``SEED``) among reports with "fell" after word 8 and before word 25,
    so the prefix carries some context and the word is early.
    """
    pool = test[test["naics"].str.startswith("23") & test["event"].str.startswith("4")]
    words_before = pool["text"].str.extract(r"^(.*?)\bfell\b", expand=False).str.split().str.len()
    pool = pool[words_before.between(8, 24)]
    row = pool.iloc[int(np.random.default_rng(SEED).integers(len(pool)))]
    cut = re.search(r" fell\b", row["text"]).start()
    prefix = row["text"][:cut]
    return {
        "prefix": prefix,
        "report": row["text"],
        "candidates": len(pool),
        "prefix_tail": " ".join(prefix.split()[-4:]),
        "report_tail": row["text"][cut:].strip(),
    }


def show(tok: transformers.PreTrainedTokenizerBase, ids: list[int]) -> list[str]:
    """Tokens as text, with a leading space shown as a middle dot and the start as <start>."""
    start = tok.convert_tokens_to_ids(doc_start(tok))
    return ["<start>" if i == start else tok.decode([i]).replace(" ", "·") for i in ids]


def next_token(
    tok: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
    prefix: str,
    word: str = " fell",
) -> dict[str, object]:
    """The five most likely next tokens after the prefix, and the probability of ``word``."""
    ids = tok(doc_start(tok) + prefix, return_tensors="pt")["input_ids"].to(DEVICE)
    with torch.no_grad():
        p = F.softmax(model(input_ids=ids).logits[0, -1].double(), dim=-1)
    top = torch.topk(p, 5)
    w = tok.encode(word)
    assert len(w) == 1, w
    return {
        "top": show(tok, top.indices.tolist()),
        "p": top.values.tolist(),
        "p_word": float(p[w[0]]),
        "rank_word": int((p > p[w[0]]).sum()) + 1,
    }


def surprise(
    tok: transformers.PreTrainedTokenizerBase, model: transformers.PreTrainedModel, text: str
) -> dict[str, object]:
    """Bits of surprise, -log2 p(token | tokens before), for each token of the text."""
    ids = tok(doc_start(tok) + text, return_tensors="pt")["input_ids"].to(DEVICE)
    with torch.no_grad():
        logp = F.log_softmax(model(input_ids=ids).logits[0, :-1].double(), dim=-1)  # (T-1, V)
    bits = -logp.gather(1, ids[0, 1:, None])[:, 0] / math.log(2)
    return {
        "tokens": show(tok, ids[0, 1:].tolist()),
        "bits": bits.tolist(),
        "last": float(bits[-1]),
    }


def generate(
    tok: transformers.PreTrainedTokenizerBase, model: transformers.PreTrainedModel, prefix: str
) -> str:
    """Greedy continuation of the prefix: the most likely token, every step.

    Whitespace is collapsed to single spaces, for display.
    """
    enc = tok(doc_start(tok) + prefix, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=GENERATE, do_sample=False, pad_token_id=tok.eos_token_id
        )
    return " ".join(
        tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).split()
    )


def loss_check(
    tok: transformers.PreTrainedTokenizerBase, model: transformers.PreTrainedModel, text: str
) -> dict[str, float]:
    """Our next-token loss against Hugging Face's, the no-shift bug, and the causal mask.

    The loss: position t's logits score token t+1. Without the shift each position is scored on
    the token it already sees. The mask: changing the last token must leave every earlier
    position's logits unchanged.
    """
    ids = tok(doc_start(tok) + text, return_tensors="pt")["input_ids"].to(DEVICE)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].double()  # (T, V)
        ours = F.cross_entropy(logits[:-1], ids[0, 1:])
        hf = model(input_ids=ids, labels=ids).loss.double()
        no_shift = F.cross_entropy(logits, ids[0])
        other = ids.clone()
        other[0, -1] = tok.encode(" crane")[0]
        changed = model(input_ids=other).logits[0].double()
    return {
        "ours": float(ours),
        "hf": float(hf),
        "diff": abs(float(ours - hf)),
        "no_shift": float(no_shift),
        "earlier_max_change": float((changed[:-1] - logits[:-1]).abs().max()),
        "last_max_change": float((changed[-1] - logits[-1]).abs().max()),
        "tokens": int(ids.shape[1] - 1),
    }


# ---------------------------------------------------------------- event coding


def construction(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construction reports (NAICS 23) of the four event types: training pool and test sample."""
    c = df[df["naics"].str.startswith("23")].assign(type=lambda d: d["event"].str[0])
    c = c[c["type"].isin(EVENTS)]
    pool = c[c["year"].between(*TRAIN_YEARS)].reset_index(drop=True)
    test = c[c["year"] >= 2023].sample(ICL_TEST, random_state=SEED).reset_index(drop=True)
    return pool, test


def draw(pool: pd.DataFrame, k: int, seed: int) -> np.ndarray:
    """Row positions of k examples per type from the pool, in a shuffled order."""
    rng = np.random.default_rng((SEED, k, seed))
    rows = [rng.choice(np.flatnonzero(pool["type"] == t), k, replace=False) for t in EVENTS]
    return rng.permutation(np.concatenate(rows))


def event_turn(text: str, order: tuple[str, ...] = tuple(EVENTS)) -> str:
    """The user's turn: the four types (in the given order), the report, and the cue."""
    options = "\n".join(f"{k}: {EVENTS[k]}" for k in order)
    return f"Event types:\n{options}\n\nReport: {text}\nEvent type:"


def event_prompt(
    tok: transformers.PreTrainedTokenizerBase,
    examples: list[tuple[str, str]],
    text: str,
    order: tuple[str, ...] = tuple(EVENTS),
) -> str:
    """The chat as one document: system, each example as a question and its answer, the report.

    Thinking is off: the template ends the document with an empty think block.
    """
    msgs = [{"role": "system", "content": EVENT_SYSTEM}]
    for t, d in examples:
        msgs += [
            {"role": "user", "content": event_turn(t, order)},
            {"role": "assistant", "content": d},
        ]
    msgs.append({"role": "user", "content": event_turn(text, order)})
    return tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )


def head_and_tails(prompts: list[str]) -> tuple[str, list[str]]:
    """Split prompts into their shared head (system and examples) and each report's turn.

    Args:
        prompts: Full prompts from ``event_prompt`` with the same examples.

    Returns:
        The head, and each prompt's remainder (its user turn and the assistant's cue).
    """
    head = prompts[0][: prompts[0].rindex(USER_TURN)]
    assert all(p.startswith(head) and p.rindex(USER_TURN) == len(head) for p in prompts)
    return head, [p[len(head) :] for p in prompts]


def digit_ids(tok: transformers.PreTrainedTokenizerBase) -> list[int]:
    """The four event digits' token ids, checked to be read as the answer token.

    Qwen's tokenizer splits digits one per token, so "4" is one token alone, after the
    assistant cue's newline in an example, and after the empty think block's two newlines.

    Args:
        tok: Qwen's tokenizer.

    Returns:
        The ids of "2", "4", "5", "6".
    """
    ids = [tok.convert_tokens_to_ids(d) for d in EVENTS]
    for d, i in zip(EVENTS, ids, strict=True):
        assert tok.encode(d, add_special_tokens=False) == [i]
        for before in ("\n", "</think>\n\n"):
            assert tok.encode(before + d, add_special_tokens=False)[-1] == i, (before, d)
    return ids


def prefill(
    model: transformers.PreTrainedModel, ids: torch.Tensor
) -> tuple[transformers.Cache, torch.Tensor]:
    """Read one prompt in pieces of ``PREFILL`` tokens, each continuing the cache of the last.

    A prompt this short or shorter is one pass. Longer ones are read in pieces so that no
    attention matrix spans the whole prompt (11,000 tokens squared would need gigabytes).

    Args:
        model: Qwen.
        ids: (1, T) token ids.

    Returns:
        The cache after the last token, and that position's logits (V,).
    """
    cache = None
    for i in range(0, ids.shape[1], PREFILL):
        out = model(
            input_ids=ids[:, i : i + PREFILL],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = out.past_key_values
    return cache, out.logits[0, -1]


def digit_logprobs(
    tok: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
    prompts: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Log-probabilities of the four digits as each prompt's next token, and whether one leads.

    "Leads": a digit is the likeliest of all tokens. The head (system turn and examples) is
    read once; batches of reports' turns then run on copies of its cache, padded on the right,
    each read at its own last token. A batch holds ``CODE_BATCH`` reports, fewer when the head
    is long (at most ``CACHE_TOKENS`` cached tokens per batch).

    Args:
        tok: Qwen's tokenizer.
        model: Qwen.
        prompts: Full prompts with the same examples, from ``event_prompt``.

    Returns:
        (n, 4) log-probabilities and (n,) booleans.
    """
    head, tails = head_and_tails(prompts)
    ids = digit_ids(tok)
    tok.padding_side = "right"
    head_ids = tok(head, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    whole = tok(prompts[0], add_special_tokens=False)["input_ids"]  # the split keeps the tokens
    assert whole == head_ids[0].tolist() + tok(tails[0], add_special_tokens=False)["input_ids"]
    batch = max(1, min(CODE_BATCH, CACHE_TOKENS // head_ids.shape[1]))
    lp, top = [], []
    with torch.no_grad():
        cache, _ = prefill(model, head_ids)
        for i in range(0, len(tails), batch):
            part = tails[i : i + batch]
            enc = tok(part, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
            n = head_ids.shape[1] + enc["input_ids"].shape[1]
            assert n <= context_length(model), n
            c = copy.deepcopy(cache)
            c.reorder_cache(torch.zeros(len(part), dtype=torch.long, device=DEVICE))  # batch rows
            h = model.model(input_ids=enc["input_ids"], past_key_values=c, use_cache=True)
            last = enc["attention_mask"].sum(1) - 1  # each report's own last token
            h = h.last_hidden_state[torch.arange(len(part), device=DEVICE), last]  # (B, H)
            logp = F.log_softmax(model.lm_head(h).float(), dim=-1)  # (B, V)
            lp.append(logp[:, ids].cpu().numpy())
            top.append(np.isin(logp.argmax(-1).cpu().numpy(), ids))
    out = np.concatenate(lp)
    assert out.shape == (len(prompts), len(EVENTS)), out.shape
    return out, np.concatenate(top)


def alone_logprobs(
    tok: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
    prompts: list[str],
) -> np.ndarray:
    """The four digits' log-probabilities with each prompt read alone: no batch, no padding.

    No shared cache either: one pass up to ``PREFILL`` tokens, pieces beyond.
    """
    ids = digit_ids(tok)
    out = []
    with torch.no_grad():
        for p in prompts:
            enc = tok(p, return_tensors="pt", add_special_tokens=False).to(DEVICE)
            _, logits = prefill(model, enc["input_ids"])
            out.append(F.log_softmax(logits.float(), dim=-1)[ids].cpu().numpy())
    return np.array(out)


def event_coding(
    tok: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
    pool: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, object]:
    """Qwen codes the test reports with 0, 1, 3, 10, or 30 examples per type in the prompt."""
    labels = np.array(list(EVENTS))
    y = test["type"].to_numpy()
    out: dict[str, object] = {}
    checks = []
    for k in SHOTS:
        runs = []
        for seed in SHOT_SEEDS if k else (0,):
            t0 = time.perf_counter()
            ex = pool.iloc[draw(pool, k, seed)] if k else pool.iloc[[]]
            examples = list(zip(ex["text"], ex["type"], strict=True))
            prompts = [event_prompt(tok, examples, t) for t in test["text"]]
            lp, top = digit_logprobs(tok, model, prompts)
            alone = alone_logprobs(tok, model, prompts[:CHECK_REPORTS])
            checks.append(
                {
                    "max_abs_diff": float(np.abs(alone - lp[:CHECK_REPORTS]).max()),
                    "same_answers": bool((alone.argmax(1) == lp[:CHECK_REPORTS].argmax(1)).all()),
                }
            )
            pred = labels[lp.argmax(1)]
            runs.append(
                {
                    "accuracy": float(np.mean(pred == y)),
                    "balanced": float(balanced_accuracy_score(y, pred)),
                    "digit_on_top": float(top.mean()),
                    "predicted": {t: int((pred == t).sum()) for t in EVENTS},
                    "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
                    "prompt_tokens": len(tok(prompts[0], add_special_tokens=False)["input_ids"]),
                    "seconds": time.perf_counter() - t0,
                }
            )
            print(
                f"  {k} per type, draw {seed}: balanced {runs[-1]['balanced']:.3f} "
                f"({runs[-1]['seconds']:.0f} s)",
                flush=True,
            )
        out[str(k)] = runs
    out["cache_check"] = {
        "max_abs_diff": max(c["max_abs_diff"] for c in checks),
        "same_answers": all(c["same_answers"] for c in checks),
        "prompts": CHECK_REPORTS * len(checks),
    }
    ex = pool.iloc[draw(pool, 1, 0)]
    first = list(zip(ex["text"], ex["type"], strict=True))[:1]
    out["example_prompt"] = event_prompt(tok, first, "<the report to code>")
    # "No prompt can hold" every labelled report: all 13,572 as examples, against the context
    every = list(zip(pool["text"], pool["type"], strict=True))
    out["all_examples_tokens"] = len(
        tok(event_prompt(tok, every, test["text"][0]), add_special_tokens=False)["input_ids"]
    )
    out["context_tokens"] = context_length(model)
    return out


def list_order(
    tok: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
    test: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """With no examples, the answers when the four types are listed in each of four orders.

    Each type is listed first once, second once, and so on (a Latin square); the digits stay
    with their names. The first order is the slides'.
    """
    labels = np.array(list(EVENTS))
    out = {}
    for r in range(len(labels)):
        order = tuple(labels[r:]) + tuple(labels[:r])
        lp, _ = digit_logprobs(tok, model, [event_prompt(tok, [], t, order) for t in test["text"]])
        pred = labels[lp.argmax(1)]
        out["".join(order)] = {t: int((pred == t).sum()) for t in EVENTS}
        print(f"  listed {', '.join(order)}: {out[''.join(order)]}", flush=True)
    return out


def embed(
    tok: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
    texts: list[str],
    prefix: str = "",
    batch: int = 128,
) -> np.ndarray:
    """BGE sentence embeddings: the [CLS] vector, L2-normalized (BGE's recipe). (n, 768)."""
    tok.padding_side = "right"
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(
                [prefix + t for t in texts[i : i + batch]],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(DEVICE)
            out.append(F.normalize(model(**enc).last_hidden_state[:, 0], dim=1).cpu().numpy())
    e = np.concatenate(out)
    assert e.shape == (len(texts), model.config.hidden_size), e.shape
    return e


def probe_coding(
    pool_emb: np.ndarray, test_emb: np.ndarray, pool: pd.DataFrame, test: pd.DataFrame
) -> dict[str, object]:
    """A logistic probe on BGE embeddings with k labelled reports per type, and with all.

    Draws use the same seeds as the prompts, so k = 1, 3, 10, and 30 see the very same
    examples. L2 penalty (``l1_ratio=0``), C = 1, and balanced class weights throughout (fixed,
    not tuned: k = 1 leaves nothing to tune on).
    """
    y = test["type"].to_numpy()

    def fit(rows: np.ndarray) -> dict[str, float]:
        clf = LogisticRegression(
            C=1.0, l1_ratio=0.0, solver="lbfgs", class_weight="balanced", max_iter=5000
        )
        pred = clf.fit(pool_emb[rows], pool["type"].to_numpy()[rows]).predict(test_emb)
        return {
            "accuracy": float(np.mean(pred == y)),
            "balanced": float(balanced_accuracy_score(y, pred)),
        }

    out: dict[str, object] = {
        str(k): [fit(draw(pool, k, s)) for s in SHOT_SEEDS] for k in PROBE_SHOTS
    }
    out["all"] = fit(np.arange(len(pool))) | {"reports": len(pool)}
    return out


def spread(values: list[float]) -> dict[str, float]:
    """Median, minimum, and maximum over the draws."""
    return {"median": float(np.median(values)), "min": min(values), "max": max(values)}


def coding_summary(icl: dict[str, object], probe: dict[str, object]) -> dict[str, object]:
    """Each setting's spread over draws, and the probe against the prompt draw by draw.

    ``paired`` holds, for each k both methods ran, the probe's balanced accuracy minus the
    prompt's on the same drawn examples, one value per draw.
    """
    out = {}
    for name, res, ks in (("prompt", icl, SHOTS), ("probe", probe, PROBE_SHOTS)):
        for k in ks:
            out[f"{name}_{k}"] = {
                m: spread([r[m] for r in res[str(k)]]) for m in ("balanced", "accuracy")
            }
    both = [k for k in SHOTS if k in PROBE_SHOTS]
    out["paired"] = {
        str(k): [
            b["balanced"] - a["balanced"] for a, b in zip(icl[str(k)], probe[str(k)], strict=True)
        ]
        for k in both
    }
    return out


# ---------------------------------------------------------------- retrieval over the MUTCD

FURNITURE = re.compile(
    r"\s*(Page \d+\s+MUTCD 11th Edition|MUTCD 11th Edition\s+Page \d+|Sect\. .* December 2023"
    r"|December 2023\s+Sect\.)"
)
LEADER = re.compile(r"\.{4,}")  # dot leaders, as in a list of figures
BODY_PAGE = re.compile(r"\bPage \d+\b")
FRONT_PAGE = re.compile(r"\bPage [A-Z]+-\d")
HEADING = re.compile(r"^Section (\d[A-Z]\.\d{2}) (.+)$")
CROSS_REF = re.compile(
    r"Sections? (\d[A-Z]\.\d{2}) (?:contains?|provides?|describes?|includes?|has) ([^.]{15,160})\."
)


def mutcd_sections() -> dict[str, tuple[str, str]]:
    """The manual's sections, in order: number -> (title, whitespace-normalized text).

    Only numbered body pages are read ("Page 1" onwards: the cover and the table of contents,
    numbered "TC-1" and so on, are dropped), without their headers and footers. A heading is a
    line that starts "Section 2A.16 ..." with a short title.
    """
    if shutil.which("pdftotext") is None:
        raise FileNotFoundError("pdftotext (poppler) is needed to read the MUTCD")
    pdf = require(MUTCD_PDF, "scripts/fetch_mutcd.sh")
    text = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True, text=True
    ).stdout
    pages = [p for p in text.split("\f") if BODY_PAGE.search(p) and not FRONT_PAGE.search(p)]
    secs: dict[str, tuple[str, list[str]]] = {}
    cur = None
    for line in "\n".join(pages).splitlines():
        if FURNITURE.match(line) or LEADER.search(line):
            continue
        m = HEADING.match(line.strip())
        if m and len(m.group(2)) < 90:
            cur = m.group(1)
            secs.setdefault(cur, (m.group(2).strip(), []))
        elif cur:
            secs[cur][1].append(line.strip())
    return {k: (t, " ".join(" ".join(b).split())) for k, (t, b) in secs.items()}


def pdftotext_version() -> str:
    """The first line of ``pdftotext -v`` ("pdftotext version 26.01.0").

    The MUTCD's text, and so its section and excerpt counts, depend on poppler's version, which
    comes from the system, not from pixi; numbers.json records it.

    Returns:
        The version line.

    Raises:
        FileNotFoundError: If pdftotext is not installed.
    """
    if shutil.which("pdftotext") is None:
        raise FileNotFoundError("pdftotext (poppler) is needed to read the MUTCD")
    out = subprocess.run(["pdftotext", "-v"], check=True, capture_output=True, text=True)
    return (out.stderr or out.stdout).splitlines()[0].strip()


def chunks(secs: dict[str, tuple[str, str]]) -> tuple[list[str], np.ndarray]:
    """Overlapping windows of each section, headed by its number and title.

    Returns the chunk texts and the index of each section's first chunk (every section has
    at least one), for a section's score = its best chunk's.
    """
    texts, starts = [], []
    for k, (title, body) in secs.items():
        starts.append(len(texts))
        w = body.split()
        last = max(1, len(w) - (CHUNK_WORDS - CHUNK_STRIDE))
        texts.extend(
            f"Section {k} {title}: " + " ".join(w[s : s + CHUNK_WORDS])
            for s in range(0, last, CHUNK_STRIDE)
        )
    return texts, np.array(starts)


def recall_at(
    s_sections: np.ndarray, targets: np.ndarray, citing: np.ndarray, ks: range
) -> list[float]:
    """Share of cross-references whose cited section ranks in the top k (citing one excluded)."""
    s = s_sections.copy()
    s[np.arange(len(s)), citing] = -np.inf
    rank = (s > s[np.arange(len(s)), targets][:, None]).sum(1)
    return [float(np.mean(rank < k)) for k in ks]


def retrieval(
    secs: dict[str, tuple[str, str]],
    texts: list[str],
    starts: np.ndarray,
    *,
    doc_emb: np.ndarray,
    query_emb: np.ndarray,
    refs: list[tuple[str, str, str]],
) -> dict[str, object]:
    """Recall@k of BGE and TF-IDF on the manual's cross-references."""
    ids = list(secs)
    pos = {k: i for i, k in enumerate(ids)}
    targets = np.array([pos[r[1]] for r in refs])
    citing = np.array([pos[r[0]] for r in refs])
    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(texts)
    s_tfidf = (tfidf.transform([r[2] for r in refs]) @ tfidf.transform(texts).T).toarray()
    ks = range(1, 21)
    return {
        "k": list(ks),
        "bge": recall_at(
            np.maximum.reduceat(query_emb @ doc_emb.T, starts, axis=1), targets, citing, ks
        ),
        "tfidf": recall_at(np.maximum.reduceat(s_tfidf, starts, axis=1), targets, citing, ks),
        "refs": len(refs),
        "cited_sections": len(set(targets.tolist())),
        "examples": [{"citing": c, "cited": t, "query": q} for c, t, q in refs[:8]],
    }


def cross_refs(secs: dict[str, tuple[str, str]]) -> list[tuple[str, str, str]]:
    """(citing section, cited section, description) for "Section X contains ..." sentences."""
    return [
        (k, m.group(1), m.group(2))
        for k, (_, body) in secs.items()
        for m in CROSS_REF.finditer(body)
        if m.group(1) in secs and m.group(1) != k
    ]


def answer(
    tok: transformers.PreTrainedTokenizerBase, model: transformers.PreTrainedModel, user: str
) -> dict[str, object]:
    """Qwen's answer to one user turn under ``QA_SYSTEM``: greedy, every setting passed.

    The pinned revision has no generation config, so the call names the decoding (greedy, no
    repetition penalty) and both stop tokens: the end of a chat turn and the end of a document.

    Returns:
        The answer (whitespace collapsed), its length in tokens, and whether it stopped by
        itself before ``MAX_ANSWER`` tokens.
    """
    msgs = [{"role": "system", "content": QA_SYSTEM}, {"role": "user", "content": user}]
    enc = tok.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    ).to(DEVICE)
    stop = [tok.convert_tokens_to_ids(t) for t in (CHAT_TURN_END, DOC_END)]
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_ANSWER,
            do_sample=False,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=stop,
            pad_token_id=tok.convert_tokens_to_ids(DOC_END),
        )
    new = out[0, enc["input_ids"].shape[1] :].tolist()
    return {
        "text": " ".join(tok.decode(new, skip_special_tokens=True).split()),
        "tokens": len(new),
        "stopped": new[-1] in stop,
    }


NUMBER = re.compile(r"\d+(?:[./]\d+)*")  # 7, 3.5, 1/200, 10.1.1.1


def added_numbers(text: str, question: str) -> list[str]:
    """Numbers in an answer that neither the question nor the instruction gave it.

    Section numbers of the manual's form (2A.16) are left out; invented ones (11.1.1) count.
    """
    given = set(NUMBER.findall(question)) | set(NUMBER.findall(QA_SYSTEM))
    return sorted(set(NUMBER.findall(SECTION_ID.sub(" ", text))) - given)


def quotes_rule(q: Question, text: str) -> bool:
    """Whether the answer holds the words of the rule itself (``q.evidence``).

    Case, spacing, and markdown stars are ignored. An answer can give the right number from
    another rule (a yield line's distance for a sign's); this tells the two apart where the
    answer copies the manual, which the 0.8B model mostly does.
    """
    if q.evidence is None:
        return False
    return " ".join(q.evidence.lower().split()) in " ".join(text.replace("*", "").lower().split())


def outcome(q: Question, text: str, gold: list[str], decline: re.Pattern[str] = DECLINE) -> str:
    """The answer's score, by its number and by the section it cites.

    Args:
        q: The question.
        text: The answer.
        gold: The sections that hold the rule.
        decline: The phrasings that count as declining (``DECLINE``, or ``DECLINE_PILOT``).

    Returns:
        If the manual answers it: declined; wrong (a number missing); correct (every number,
        and a cited section that holds the rule); or "number only" (every number, but no
        section cited or only sections that do not hold the rule). If not: declined or
        invented.
    """
    if decline.search(text):
        return "declined"
    if q.evidence is None:
        return "invented"
    if not all(re.search(p, text, re.IGNORECASE) for p in q.answer):
        return "wrong"
    return "correct" if set(SECTION_ID.findall(text)) & set(gold) else "number only"


def question_answering(
    tok: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
    secs: dict[str, tuple[str, str]],
    *,
    texts: list[str],
    starts: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, object]]:
    """Each question answered from memory and from the top five excerpts (BGE).

    Retrieval is recorded by section (``hit``, ``gold_rank``: an excerpt of a section that
    holds the rule) and by sentence (``evidence_ranks``: the ranks of the excerpts that hold
    the rule's own words, ``Question.evidence``).
    """
    sec_of = np.repeat(list(secs), np.diff([*starts, len(texts)]))
    out = []
    for q, s in zip(QUESTIONS, scores, strict=True):
        gold = sorted(k for k, (_, body) in secs.items() if q.evidence and q.evidence in body)
        assert q.evidence is None or gold, q.evidence
        top = np.argsort(-s)[:TOP_K]
        context = "\n\n".join(texts[j] for j in top)
        closed = answer(tok, model, q.text)
        rag = answer(tok, model, f"Excerpts from the MUTCD:\n\n{context}\n\nQuestion: {q.text}")
        row: dict[str, object] = {
            "question": q.text,
            "answerable": q.evidence is not None,
            "gold": gold,
            "retrieved": [str(sec_of[j]) for j in top],
            "chunks": [int(j) for j in top],
            "scores": [float(s[j]) for j in top],
            "hit": bool(set(sec_of[top]) & set(gold)),
            "gold_rank": next((r + 1 for r, j in enumerate(top) if sec_of[j] in gold), None),
            "evidence_ranks": [
                r + 1 for r, j in enumerate(top) if q.evidence and q.evidence in texts[j]
            ],
        }
        for mode, a in (("closed", closed), ("rag", rag)):
            row[mode] = a["text"]
            row[f"{mode}_outcome"] = outcome(q, a["text"], gold)
            row[f"{mode}_cited"] = SECTION_ID.findall(a["text"])
            row[f"{mode}_quotes_rule"] = quotes_rule(q, a["text"])
            row[f"{mode}_added"] = added_numbers(a["text"], q.text)
            row[f"{mode}_tokens"] = a["tokens"]
            row[f"{mode}_stopped"] = a["stopped"]
        out.append(row)
        print(f"  [{len(out)}] {row['closed_outcome']} / {row['rag_outcome']}", flush=True)
    return out


KINDS = {
    "answerable": ("correct", "number only", "declined", "wrong"),
    "unanswerable": ("declined", "invented"),
}


def tally(
    qa: list[dict[str, object]], decline: re.Pattern[str] = DECLINE
) -> dict[str, dict[str, int]]:
    """Counts of each outcome, from memory and with excerpts, answerable or not.

    Two more counts for each: declines that still give a number of their own ("not in the
    MUTCD, but it requires 10 feet"), and right numbers without the section whose text holds
    the rule's own words (the others took the number from another rule, or paraphrased).

    Args:
        qa: The answers, from ``question_answering``.
        decline: The phrasings that count as declining. Each answer is scored again with
            them; with ``DECLINE`` the scores must be the recorded ones.

    Returns:
        For each mode and kind of question, the count of each outcome, and the two extras.
    """
    by_text = {q.text: q for q in QUESTIONS}
    out = {}
    for mode in ("closed", "rag"):
        scored = [outcome(by_text[r["question"]], r[mode], r["gold"], decline) for r in qa]
        if decline is DECLINE:
            assert scored == [r[f"{mode}_outcome"] for r in qa], mode
        for kind, kinds in KINDS.items():
            rows = [
                (o, r)
                for o, r in zip(scored, qa, strict=True)
                if r["answerable"] is (kind == "answerable")
            ]
            out[f"{mode}_{kind}"] = {k: sum(o == k for o, _ in rows) for k in kinds}
            out[f"{mode}_{kind}_extra"] = {
                "declined_with_numbers": sum(
                    o == "declined" and bool(r[f"{mode}_added"]) for o, r in rows
                ),
                "number_only_quotes_rule": sum(
                    o == "number only" and r[f"{mode}_quotes_rule"] for o, r in rows
                ),
            }
    return out


# ---------------------------------------------------------------- figures


def picture(tok_ids: list[str], nxt: dict[str, dict[str, object]], out_dir: Path) -> None:
    """The one picture: the causal mask over the prefix, then the next-token distributions.

    Frame 1 shows the mask with question marks; frame 2 adds GPT-2's five likeliest next tokens,
    as released and after the reports.
    """
    n = len(tok_ids)
    for frame in (1, 2):
        fig, axes = plt.subplots(
            1, 3, figsize=aml_style.FULL, gridspec_kw={"width_ratios": [1.3, 1, 1]}
        )
        mask = np.tril(np.ones((n, n)))
        mask[-1, :] *= 2  # the last row: the position that predicts the next word
        axes[0].imshow(mask, cmap=ListedColormap(["white", "#d6e4f2", aml_style.PALETTE[0]]))
        axes[0].set_xticks([])
        axes[0].set_yticks(range(n), tok_ids, fontsize=8)
        axes[0].set_xlabel("the tokens it may look at", fontsize=9)
        axes[0].set_xticks(np.arange(n + 1) - 0.5, minor=True)
        axes[0].set_yticks(np.arange(n + 1) - 0.5, minor=True)
        axes[0].grid(which="minor", color="white", lw=0.6)
        axes[0].tick_params(which="minor", length=0)
        axes[0].set_title("each token sees only earlier ones", fontsize=10)
        for ax, (name, d) in zip(axes[1:], nxt.items(), strict=True):
            ax.set_title(name, fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_xlabel("probability of the next token", fontsize=9)
            if frame == 1:
                ax.text(0.5, 0.5, "?", fontsize=40, ha="center", va="center", color=aml_style.MUTED)
                ax.set_yticks([])
                continue
            y = np.arange(5)[::-1]
            ax.barh(y, d["p"], color=aml_style.PALETTE[0])
            ax.set_yticks(y, d["top"], fontsize=10)
            for yy, p in zip(y, d["p"], strict=True):
                ax.text(p + 0.02, yy, f"{p:.2f}", va="center", fontsize=9)
        fig.tight_layout()
        aml_style.save(fig, out_dir, f"picture_{frame}")


def surprise_figure(s: dict[str, dict[str, object]], out_dir: Path) -> None:
    """Bits of surprise per token of the attempt's sentence, before and after the reports."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    names = list(s)
    tokens = s[names[0]]["tokens"]
    x = np.arange(len(tokens))
    for j, (name, col) in enumerate(
        zip(names, (aml_style.MUTED, aml_style.PALETTE[0]), strict=True)
    ):
        ax.bar(x + (j - 0.5) * 0.4, s[name]["bits"], 0.4, color=col, label=name)
    ax.set_xticks(x, tokens, rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("surprise, bits (-log2 p)")
    ax.set_title(
        r"$p(x_1,\dots,x_T)=\prod_t\, p(x_t\mid x_{<t})$,    "
        r"loss $=-\frac{1}{T}\sum_t \log p(x_t\mid x_{<t})$",
        fontsize=13,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=10)
    aml_style.save(fig, out_dir, "surprise")


def bits_figure(
    ft: dict[str, object], unigram: float, base_bpc: float, ratio: float, out_dir: Path
) -> None:
    """Test bits per character against reports seen in fine-tuning, with two references.

    Args:
        ft: ``fine_tune``'s output.
        unigram: The token-frequency baseline.
        base_bpc: Qwen3.5-0.8B-Base as released.
        ratio: Its parameters over GPT-2's.
        out_dir: Where the figure goes.
    """
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    seen = [int(k) for k in ft["bits_per_char"]]
    xs = [max(k, 150) for k in seen]  # 0 sits at the left edge of the log axis
    ax.plot(
        xs,
        list(ft["bits_per_char"].values()),
        marker="o",
        lw=2,
        color=aml_style.PALETTE[0],
        label="GPT-2 (124M), fine-tuned on the reports",
    )
    ax.axhline(
        unigram,
        ls="--",
        lw=2,
        color=aml_style.MUTED,
        label="token frequencies alone (naive baseline)",
    )
    ax.axhline(
        base_bpc,
        ls="-.",
        lw=2,
        color=aml_style.PALETTE[1],
        label=f"{QWEN_NAME}-Base ({ratio:.0f}× the parameters), as released",
    )
    ax.set_xscale("log")
    ax.set_xticks(xs, ["0"] + [f"{k:,}" for k in seen[1:]])
    ax.minorticks_off()
    ax.set_xlabel("training-year reports seen (one pass)")
    ax.set_ylabel("test bits per character")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    aml_style.save(fig, out_dir, "bits")


def coding_figure(icl: dict[str, object], probe: dict[str, object], out_dir: Path) -> None:
    """Balanced accuracy against labelled examples per type: prompts and probes."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    ks = list(PROBE_SHOTS)
    vals = [[r["balanced"] for r in probe[str(k)]] for k in ks]
    ax.fill_between(
        ks, [min(v) for v in vals], [max(v) for v in vals], color=aml_style.PALETTE[0], alpha=0.15
    )
    ax.plot(
        ks,
        [float(np.median(v)) for v in vals],
        marker="o",
        lw=2,
        color=aml_style.PALETTE[0],
        label=f"probe on BGE embeddings ({len(SHOT_SEEDS)} draws)",
    )
    ax.plot(
        [400],
        [probe["all"]["balanced"]],
        marker="s",
        ms=9,
        ls="",
        color=aml_style.NAVY,
        label=f"probe, all {probe['all']['reports']:,} training reports",
    )
    xs = [max(k, 0.5) for k in SHOTS]
    v = [[r["balanced"] for r in icl[str(k)]] for k in SHOTS]
    med = [float(np.median(x)) for x in v]
    ax.errorbar(
        xs,
        med,
        yerr=[np.subtract(med, [min(x) for x in v]), np.subtract([max(x) for x in v], med)],
        marker="D",
        ms=8,
        lw=2,
        ls="--",
        capsize=4,
        color=aml_style.PALETTE[1],
        label=f"{QWEN_NAME}, examples in the prompt",
    )
    ax.axhline(
        1 / len(EVENTS),
        ls=":",
        color=aml_style.MUTED,
        lw=1.5,
        label="chance (also: always answer one type)",
    )
    ax.set_xscale("log")
    ax.set_xticks([0.5, *ks, 400], ["0", *map(str, ks), "all"])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlabel("labelled reports per event type")
    ax.set_ylabel("balanced accuracy, test")
    ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=2)  # under the chance line
    aml_style.save(fig, out_dir, "coding")


def zero_shot_figure(icl: dict[str, object], test: pd.DataFrame, out_dir: Path) -> None:
    """How the test reports' types are spread, and how the prompts' answers are spread."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    x = np.arange(len(EVENTS))
    true = [int((test["type"] == t).sum()) for t in EVENTS]
    series = [
        ("the reports' own codes", true, aml_style.MUTED, ""),
        ("no examples", list(icl["0"][0]["predicted"].values()), aml_style.PALETTE[1], "//"),
        (
            f"{SHOW_K} per type (first draw)",
            list(icl[str(SHOW_K)][0]["predicted"].values()),
            aml_style.PALETTE[0],
            "..",
        ),
    ]
    for j, (name, vals, col, hatch) in enumerate(series):
        bars = ax.bar(x + (j - 1) * 0.27, vals, 0.27, color=col, hatch=hatch, label=name)
        ax.bar_label(bars, fontsize=8, padding=1)
    ax.set_xticks(x, [textwrap.fill(f"{k}: {v}", 24) for k, v in EVENTS.items()], fontsize=9)
    ax.set_ylabel(f"of {len(test):,} test reports")
    ax.legend(frameon=False, fontsize=10)
    aml_style.save(fig, out_dir, "zero_shot")


def recall_figure(r: dict[str, object], hits: float, n: int, out_dir: Path) -> None:
    """Recall@k on the cross-references, and the hand-written questions' hit rate at 5."""
    fig, ax = plt.subplots(figsize=aml_style.FULL)
    ax.plot(
        r["k"],
        r["bge"],
        lw=2,
        marker="o",
        ms=4,
        color=aml_style.PALETTE[0],
        label="BGE embeddings (cosine)",
    )
    ax.plot(
        r["k"],
        r["tfidf"],
        lw=2,
        marker="s",
        ms=4,
        ls="--",
        color=aml_style.MUTED,
        label="TF-IDF words (Lecture 4)",
    )
    ax.plot(
        [TOP_K],
        [hits],
        marker="*",
        ms=16,
        ls="",
        color=aml_style.PALETTE[1],
        label=f"our {n} questions: right section in the {TOP_K} excerpts",
    )
    ax.set_xlabel("sections returned (k)")
    ax.set_ylabel("cited section among them")
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    aml_style.save(fig, out_dir, "recall")


def scorecard_figure(t: dict[str, dict[str, int]], out_dir: Path) -> None:
    """The decisive figure: outcomes from memory and with excerpts, by kind of question."""
    colors = {
        "correct": aml_style.PALETTE[2],
        "number only": aml_style.PALETTE[3],
        "declined": "#bfbfbf",
        "wrong": aml_style.PALETTE[1],
        "invented": aml_style.PALETTE[1],
    }
    hatch = {"correct": "", "number only": "..", "declined": "", "wrong": "//", "invented": "//"}
    sizes = {kind: sum(t[f"closed_{kind}"].values()) for kind in KINDS}
    fig, axes = plt.subplots(
        1,
        2,
        figsize=aml_style.FULL,
        gridspec_kw={"width_ratios": [sizes["answerable"], sizes["unanswerable"] + 2]},
        sharey=True,
    )
    rows = (("rag", f"with {TOP_K} excerpts"), ("closed", "from memory"))
    titles = (
        f"{sizes['answerable']} questions the manual answers",
        f"{sizes['unanswerable']} it does not",
    )
    for ax, kind, title in zip(axes, KINDS, titles, strict=True):
        for y, (mode, _) in enumerate(rows):
            left = 0
            for k, n in t[f"{mode}_{kind}"].items():
                if n:
                    ax.barh(y, n, left=left, color=colors[k], hatch=hatch[k], edgecolor="white")
                    ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=11)
                left += n
        ax.set_yticks(range(len(rows)), [r[1] for r in rows])
        ax.set_xlim(0, sizes[kind])
        ax.set_xticks([])
        ax.set_title(title, fontsize=11)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[k], hatch=hatch[k], ec="white")
        for k in ("correct", "number only", "declined", "wrong")
    ]
    fig.legend(
        handles,
        ["number and section", "number, not the section", "declined", "wrong or invented"],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    aml_style.save(fig, out_dir, "scorecard")


def title_art_figure(out_dir: Path) -> None:
    """The title art: a line of words, each a dim light, ending in the bright next one."""
    c = title_art.Canvas(seed=SEED)
    top = c.top
    c.ground(("#0b0d14", "#11141f", "#1c1420", "#08070b"), grain=0.08)
    c.glaze(c.polygon(title_art.stepped(0.04, 0.05, 0.60, 0.40, 5), soft=4), "#3d1d14", 0.85)
    c.glaze(c.polygon(title_art.stepped(0.55, 0.05, 0.40, 0.26, 3), soft=4), "#2a1610", 0.85)
    xs = np.linspace(0.12, 0.70, 7)
    for j, x in enumerate(xs):
        c.glow(c.disc((x, top * 0.66), 0.016, soft=2), "#9cc3e6", 0.12 + 0.05 * j, spread=8)
    c.glow(
        c.lines([[(xs[-1], top * 0.66), (0.84, top * 0.70)]], 1.6, soft=1.0),
        "#c8d8ea",
        0.35,
        spread=6,
    )
    c.glow(c.disc((0.86, top * 0.71), 0.04, soft=3), "#eaa548", 0.6, spread=22)
    c.save_title_art(
        out_dir,
        "Abstract painting on a dark ground: stepped rust forms at the bottom; a row of seven "
        "small pale-blue lights, each a little brighter than the one before, leads by a thin line "
        "to a large golden light: the words so far, and the next word a language model predicts.",
    )


# ---------------------------------------------------------------- main


def language_modelling(df: pd.DataFrame, numbers: dict[str, object], out_dir: Path) -> None:
    """GPT-2 on the reports: the attempt, the loss check, fine-tuning; Qwen's references."""
    train_years = df[df["year"].between(*TRAIN_YEARS)]
    test_years = df[df["year"] >= 2023]
    train = train_years["text"].sample(LM_TRAIN, random_state=SEED).tolist()
    test = test_years["text"].sample(LM_TEST, random_state=SEED).tolist()
    att = attempt_report(test_years)
    tok, base = gpt2()
    model = copy.deepcopy(base)
    numbers["attempt"] = att
    numbers["check"] = loss_check(tok, base, att["prefix"] + " fell")
    numbers["fine_tune"] = ft = fine_tune(tok, model, train, test)
    names = {"GPT-2 as released": base, f"after {LM_TRAIN:,} reports": model}
    nxt = {n: next_token(tok, m, att["prefix"]) for n, m in names.items()}
    sur = {n: surprise(tok, m, att["prefix"] + " fell") for n, m in names.items()}
    numbers["next_token"] = nxt
    numbers["surprise"] = sur
    before, after = (np.array(sur[n]["bits"]) for n in names)
    numbers["surprise_lower"] = int(np.sum(after < before))  # tokens less surprising after
    numbers["generated"] = {n: generate(tok, m, att["prefix"]) for n, m in names.items()}
    numbers["unigram_bits_per_char"] = unigram = unigram_bits_per_char(tok, train, test)
    prefix_ids = tok(doc_start(tok) + att["prefix"])["input_ids"]
    picture(show(tok, prefix_ids), nxt, out_dir)
    surprise_figure(sur, out_dir)
    numbers["lm"] = {
        "train_reports": LM_TRAIN,
        "test_reports": LM_TEST,
        "lr": LR,
        "batch": BATCH,
        "steps": {str(k): k // BATCH for k in EVAL_AT},  # optimizer steps at each evaluation
    }
    params = {"gpt2": parameters(base)}
    vocab = {"gpt2": len(tok)}
    n_chars = sum(map(len, test))
    tokens = {"gpt2": sum(map(len, tok(test)["input_ids"]))}
    del base, model
    torch.cuda.empty_cache()
    # The size reference: the pretrained Qwen3.5 (no chat tuning), and the chat model too
    t0 = time.perf_counter()
    for key, (name, rev) in {"qwen_base": (QWEN_BASE, QWEN_BASE_REVISION),
                             "qwen": (QWEN, QWEN_REVISION)}.items():  # fmt: skip
        q_tok, q_model = qwen(name, rev)
        numbers[f"{key}_bits_per_char"] = bits_per_char(q_tok, q_model, test, QWEN_SCORE_BATCH)
        params[key] = parameters(q_model)
        vocab[key] = len(q_tok)
        tokens[key] = sum(map(len, q_tok(test, add_special_tokens=False)["input_ids"]))
        del q_model
        torch.cuda.empty_cache()
    numbers["qwen_scoring_seconds"] = time.perf_counter() - t0
    numbers["parameters"] = params
    numbers["parameters_billions"] = {k: v / 1e9 for k, v in params.items()}
    numbers["parameters_millions"] = {k: v / 1e6 for k, v in params.items()}
    numbers["vocab"] = vocab
    numbers["test_tokens"] = tokens | {"characters": n_chars}
    numbers["base_ratio"] = ratio = params["qwen_base"] / params["gpt2"]
    bits_figure(ft, unigram, numbers["qwen_base_bits_per_char"], ratio, out_dir)
    msgs = [
        {"role": "system", "content": QA_SYSTEM},
        {"role": "user", "content": QUESTIONS[7].text},
    ]
    numbers["chat_document"] = q_tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )


def embeddings(
    df: pd.DataFrame, numbers: dict[str, object]
) -> tuple[
    pd.DataFrame, pd.DataFrame, dict[str, tuple[str, str]], list[str], np.ndarray, np.ndarray
]:
    """BGE: the probes on the construction reports, and retrieval over the MUTCD.

    Returns:
        The coding pool and test reports, the manual's sections, its excerpts and their
        sections' first indices, and the 22 questions' cosines with every excerpt.
    """
    pool, test = construction(df)
    built = df[df["naics"].str.startswith("23")]
    numbers["coding_data"] = {
        "other_types_share": float(np.mean(~built["event"].str[0].isin(EVENTS))),
        "pool": len(pool),
        "test": len(test),
        "test_counts": {t: int((test["type"] == t).sum()) for t in EVENTS},
        "majority_share": float(test["type"].value_counts(normalize=True).iloc[0]),
        "chance": 1 / len(EVENTS),  # balanced accuracy of any constant answer
        "median_words": float(test["text"].str.split().str.len().median()),
    }
    b_tok = transformers.AutoTokenizer.from_pretrained(BGE, revision=BGE_REVISION)
    bge = transformers.AutoModel.from_pretrained(BGE, revision=BGE_REVISION).to(DEVICE).eval()
    numbers["probe"] = probe_coding(
        embed(b_tok, bge, pool["text"].tolist()),
        embed(b_tok, bge, test["text"].tolist()),
        pool,
        test,
    )
    secs = mutcd_sections()
    texts, starts = chunks(secs)
    refs = cross_refs(secs)
    doc_emb = embed(b_tok, bge, texts)
    numbers["retrieval"] = retrieval(
        secs,
        texts,
        starts,
        doc_emb=doc_emb,
        query_emb=embed(b_tok, bge, [r[2] for r in refs], BGE_QUERY),
        refs=refs,
    )
    q_scores = embed(b_tok, bge, [q.text for q in QUESTIONS], BGE_QUERY) @ doc_emb.T
    numbers["mutcd"] = {
        "sections": len(secs),
        "chunks": len(texts),
        "words": sum(len(b.split()) for _, b in secs.values()),
        "chunk_words": CHUNK_WORDS,
        "chunk_stride": CHUNK_STRIDE,
        "pdftotext": pdftotext_version(),
    }
    numbers["bge_parameters"] = parameters(bge)
    del bge
    torch.cuda.empty_cache()
    return pool, test, secs, texts, starts, q_scores


def main() -> None:
    """Generate every figure and number for CEE Lecture 19."""
    out_dir = parse_out_dir(__doc__ or "")
    aml_style.apply()
    seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False  # float32 means float32 (PyTorch's default)
    torch.backends.cudnn.allow_tf32 = False  # Qwen3.5's convolution too (cuDNN's default is on)
    transformers.logging.set_verbosity_error()
    t0 = time.perf_counter()
    numbers: dict[str, object] = {}
    df = osha()
    language_modelling(df, numbers, out_dir)
    print(f"next-token prediction in {time.perf_counter() - t0:.0f} s", flush=True)
    pool, test, secs, texts, starts, q_scores = embeddings(df, numbers)
    print(f"embeddings in {time.perf_counter() - t0:.0f} s", flush=True)

    q_tok, q32 = qwen(QWEN, QWEN_REVISION)
    numbers["coding"] = icl = event_coding(q_tok, q32, pool, test)
    numbers["list_order"] = order = list_order(q_tok, q32, test)
    assert order["".join(EVENTS)] == icl["0"][0]["predicted"]  # the slides' order: same prompts
    print(f"event coding in {time.perf_counter() - t0:.0f} s", flush=True)
    numbers["qa"] = qa = question_answering(
        q_tok, q32, secs, texts=texts, starts=starts, scores=q_scores
    )
    del q32
    torch.cuda.empty_cache()
    numbers["tally"] = t = tally(qa)
    numbers["tally_pilot_declines"] = tally(qa, DECLINE_PILOT)  # both counts on the slides
    answerable = [r for r in qa if r["answerable"]]
    numbers["qa_hits"] = sum(r["hit"] for r in answerable)
    numbers["qa_hit_rate"] = numbers["qa_hits"] / len(answerable)
    # Retrieval by sentence: the rule's own words among the five excerpts, for every answerable
    # question and for the answers with excerpts that fell short of correct
    numbers["qa_evidence_hits"] = sum(bool(r["evidence_ranks"]) for r in answerable)
    short = [r for r in answerable if r["rag_outcome"] != "correct"]
    numbers["qa_short"] = {
        "answers": len(short),
        "with_section": sum(r["hit"] for r in short),
        "with_evidence": sum(bool(r["evidence_ranks"]) for r in short),
    }
    numbers["qa_odd_sections"] = sorted(m for r in qa for m in ODD_SECTION.findall(r["closed"]))
    numbers["doc_start"] = DOC_START
    print(f"question answering in {time.perf_counter() - t0:.0f} s", flush=True)

    numbers["coding_summary"] = coding_summary(icl, numbers["probe"])
    coding_figure(icl, numbers["probe"], out_dir)
    zero_shot_figure(icl, test, out_dir)
    recall_figure(numbers["retrieval"], numbers["qa_hit_rate"], len(answerable), out_dir)
    scorecard_figure(t, out_dir)
    title_art_figure(out_dir)
    aml_style.thumbnails(out_dir, ("picture_2", "bits", "coding", "scorecard"))
    numbers["minutes"] = (time.perf_counter() - t0) / 60
    write_numbers(numbers, out_dir)
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
