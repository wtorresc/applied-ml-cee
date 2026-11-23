"""Applied Machine Learning for CEE — Lecture 19: Language Models and Retrieval.

Marimo source notebook. Edit with::

    pixi run -e aml-env marimo edit notebooks/cee_l19_language_models.py

Export to Jupyter for sharing::

    pixi run -e aml-env export-ipynb

Data: OSHA's Severe Injury Reports (U.S. government data, public): osha.gov refuses scripted
downloads, so save the ZIP with a browser and run ``bash scripts/fetch_osha_sir.sh <saved zip>``.
The MUTCD, 11th Edition (FHWA, 2023; a U.S. government work, public domain) from
``bash scripts/fetch_mutcd.sh``; its text is read with poppler's ``pdftotext``. GPT-2 (MIT),
Qwen3.5-0.8B-Base and Qwen3.5-0.8B (Apache-2.0), and BGE base v1.5 (MIT) come from the Hugging
Face Hub at pinned revisions on the first run.

Set ``L19_SETTING=reduced`` (the default: a laptop CPU) or ``L19_SETTING=slides`` (the slides'
full run, on a GPU) to choose the setting (see the header cell).
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Applied Machine Learning for CEE — Lecture 19: Language Models and Retrieval

    **Slides:** *Applied Machine Learning for CEE*, Lecture 19 (`lectures/cee_l19_language_models`, PowerPoint).

    **Credits:** some material inspired by Derek Hoiem's *CS 441 — Applied Machine Learning* (University of Illinois Urbana-Champaign, Fall 2026).

    **Reading:** Forsyth, *Applied Machine Learning* (Springer 2019), §13.1.3 (Markov chain models of text: unigrams, n-grams, and smoothing), §16.1.1–16.1.2 (the softmax and the cross-entropy loss), §16.4.2 (dropout), §6.3.1 and §6.3.6 (the cosine distance and TF-IDF), and §1.2 (nearest neighbours).

    **Data:**
    - OSHA Severe Injury Reports (U.S. Occupational Safety and Health Administration; U.S.
      government data, public): Lecture 17's reports, January 2015 to November 2025, each with
      the employer's description of the incident and an OIICS event code; federal OSHA
      jurisdiction only. osha.gov refuses scripted downloads: save the ZIP with a browser, then
      `bash scripts/fetch_osha_sir.sh <the saved zip>`. Read in place, never extracted.
    - The MUTCD, *Manual on Uniform Traffic Control Devices*, 11th Edition (FHWA, December
      2023; a work of the U.S. federal government, public domain), about 24 MB:
      `bash scripts/fetch_mutcd.sh`. Its text is read with poppler's `pdftotext -layout`, which
      must be installed (`pdftotext -v`).
    - GPT-2 (`openai-community/gpt2`, Radford et al., 2019; MIT), the 124-million-parameter
      model, about 0.5 GB.
    - Qwen3.5-0.8B-Base (`Qwen/Qwen3.5-0.8B-Base`, Qwen Team, 2026; Apache-2.0), pretrained
      only, and Qwen3.5-0.8B (`Qwen/Qwen3.5-0.8B`), the same model tuned for chat: about 1.7 GB
      each. Both are vision-language releases; we load their text model only.
    - BGE base v1.5 (`BAAI/bge-base-en-v1.5`, Xiao et al., 2024; MIT), about 0.4 GB.

    The four models come from the Hugging Face Hub at pinned revisions on the first run. Every
    one runs in float32 (a free Colab T4 has no native bfloat16).

    This notebook follows the lecture's eight steps, §0 to §7: **warm-up → the two jobs and the
    attempt → one picture → derive → build and verify → break it → decide → check yourself**.
    Every number on the slides is recomputed here and checked against the slides' run
    (`numbers.json`, read by the notebook, never typed in).

    **Two settings**, chosen by the environment variable `L19_SETTING` (whatever the device):

    - `reduced` (the default): for a laptop CPU. GPT-2 reads 800 training
      reports instead of 20,000 and is scored on 200 test reports instead of 2,000; Qwen codes
      the first 50 test reports with one draw of 0, 1, 3, and 10 examples per type; the probe
      skips its fit on all 13,572 reports. Everything else is the slides' computation: the
      data, the manual, retrieval, and the 22 questions. What does not depend on the device
      (parsing, counting, TF-IDF, the attempt, the prompts' lengths) is checked exactly; the
      rest is printed beside the slides' numbers.
      Measured on 8 CPU threads of a machine busy with other jobs (November 20, 2026): 30
      minutes, and 12 GB of memory at its peak (the four models stay loaded), so a laptop with
      16 GB. It also runs on a GPU, untimed there (a shared GPU ran out of memory).
    - `slides`: the slides' full run, about 40 minutes on the slides' GPU (an RTX 3090 Ti,
      shared with other jobs; November 19, 2026), where it reproduced every checked number of
      the slides' run. On another GPU the last digits move, and those numbers are printed
      beside the slides'.

    **Your tasks** (from the slides): finish an injury report, compute GPT-2's loss by hand and
    match Hugging Face's, fine-tune GPT-2 on the reports, code reports with examples in a prompt
    and with a probe, retrieve excerpts from the manual (ask it your own question), reorder the
    event types in the prompt, then decide what the crews should get.
    """)
    return


@app.cell
def _():
    import copy
    import itertools
    import json
    import math
    import os
    import random
    import re
    import shutil
    import subprocess
    import time
    import zipfile
    from dataclasses import dataclass
    from pathlib import Path

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # read when huggingface_hub loads
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F  # noqa: N812 - the PyTorch convention
    from matplotlib.colors import ListedColormap
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )
    from transformers.utils import logging as hf_logging

    SEED = 19
    os.environ["PYTHONHASHSEED"] = "0"  # for subprocesses; this one needs it before Python starts

    def seed_all(seed: int) -> None:
        """Seed Python's, NumPy's, and PyTorch's (CPU and every GPU) global generators.

        Args:
            seed: The seed for every generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    seed_all(SEED)
    # As the slides' figure script: deterministic kernels wherever PyTorch has one, and float32
    # that is float32 (no TF32 in matrix products or in Qwen3.5's convolution)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hf_logging.set_verbosity_error()  # quiet: Qwen3.5's optional fast kernels are not installed
    hf_logging.disable_progress_bar()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GPU = DEVICE.type == "cuda"

    # The setting, independent of the device: "reduced" (default) or "slides" (the header cell)
    SETTING = os.environ.get("L19_SETTING", "reduced").strip().lower()
    if SETTING not in {"reduced", "slides"}:
        raise ValueError(f"L19_SETTING must be 'reduced' or 'slides', not {SETTING!r}")
    SLIDES = SETTING == "slides"
    # The slides' digits of what ran on a GPU are checked only on the slides' GPU, in "slides"
    REFERENCE_GPU = SLIDES and GPU and "3090 Ti" in torch.cuda.get_device_name(0)

    def find_root() -> Path:
        """The repository, found from its root, ``notebooks/``, or ``ipynb/`` (Jupyter's cwd).

        Returns:
            The folder that holds ``data/``.

        Raises:
            FileNotFoundError: If no candidate holds the OSHA reports.
        """
        rel = Path("data") / "raw" / "osha_sir" / "January2015toNovember2025.zip"
        bases = [Path.cwd(), Path.cwd().parent]
        if "__file__" in globals():
            bases.insert(0, Path(__file__).resolve().parents[1])
        for base in bases:
            if (base / rel).exists():
                return base
        raise FileNotFoundError(
            f"{rel} not found from {Path.cwd()}: save the ZIP from osha.gov, then run "
            "`bash scripts/fetch_osha_sir.sh <saved zip>` from the repository root"
        )

    ROOT = find_root()
    DATA = ROOT / "data"
    OSHA_ZIP = DATA / "raw" / "osha_sir" / "January2015toNovember2025.zip"
    MUTCD_PDF = DATA / "raw" / "mutcd" / "mutcd11thedition.pdf"
    TRAIN_YEARS = (2015, 2022)  # Lecture 17's: learn from these years, test on 2023-2025

    # The slides' numbers (the figure script's numbers.json): every reference value below is
    # read from it, never typed. "slides" needs it; "reduced" compares where it can.
    NUMBERS = ROOT / "lectures" / "build" / "fig" / "cee_l19_language_models" / "numbers.json"
    REF = json.loads(NUMBERS.read_text(encoding="utf-8")) if NUMBERS.exists() else None
    if SLIDES and REF is None:
        raise FileNotFoundError(
            f"{NUMBERS} not found: build the slides' figures first "
            "(`make build/fig/cee_l19_language_models/.stamp` in lectures/)"
        )

    def ref(path: str) -> object:
        """The slides' value at ``path`` in numbers.json, e.g. ``"coding/3/0/balanced"``.

        Args:
            path: Keys and list positions, separated by "/".

        Returns:
            The value, or None when numbers.json is not there.
        """
        if REF is None:
            return None
        node = REF
        for key in path.split("/"):
            node = node[int(key)] if isinstance(node, list) else node[key]
        return node

    def near(x: float, path: str, tol: float) -> bool:
        """Whether ``x`` is within ``tol`` of the slides' value at ``path`` (relative above 1).

        Args:
            x: The value computed here.
            path: Its key in numbers.json, as for ``ref``.
            tol: The tolerance.

        Returns:
            True if they agree, or if numbers.json is not there to compare with.
        """
        want = ref(path)
        return want is None or abs(x - want) <= tol * max(1.0, abs(want))

    # Tolerances. Parsing, counting, TF-IDF, and the unigram baseline run on the CPU and are
    # exact; so is our loss against PyTorch's on the same logits (float64).
    EXACT = 1e-9
    LOSS_TOL = 1e-12
    # Against Hugging Face's loss, computed in float32 from its own forward pass: float32
    # round-off on a GPU (the slides: 1.7e-07); on a CPU its threads also split the sums
    # differently between the two passes (1.5e-05 measured on 16 threads)
    HF_TOL = 1e-6 if GPU else 1e-4
    # What ran on the slides' GPU (an RTX 3090 Ti) is checked to these tolerances there. Elsewhere
    # GPU kernels and CPU threads change the last digits, and those numbers are printed beside
    # the slides' rather than asserted (except where the check says otherwise).
    FWD_TOL = 1e-5 if GPU else 1e-3  # models as released: probabilities, bits (any setting)
    TUNE_TOL = 0.005  # GPT-2 after fine-tuning, on the slides' GPU (it keeps the slides' digits)
    COS_TOL = 1e-4 if GPU else 1e-3  # BGE cosines (any setting)
    RECALL_TOL = 0.005  # recall@k over 221 references: 1 of them
    PROBE_TOL = 1e-6  # probes on BGE embeddings
    LOGP_TOL = 1e-3  # Qwen's digit log-probabilities: cached and batched against each prompt alone

    # The models, at the slides' revisions
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
    CHAT_TURN_END, DOC_END = "<|im_end|>", "<|endoftext|>"  # Qwen's two stop tokens
    DOC_START = DOC_END  # every model scores a report after this one token (doc_start)
    USER_TURN = "<|im_start|>user\n"  # where a chat's user turn starts

    # Next-token prediction on the reports ("reduced": the first reports of the slides' samples)
    LM_TRAIN = 20_000 if SLIDES else 800  # training-year reports for one pass of fine-tuning
    LM_TEST = 2_000 if SLIDES else 200  # test-year reports scored
    TRAIN_TOKENS = 128  # fine-tuning truncates longer reports
    BATCH = 16
    LR = 5e-5  # AdamW; a usual rate for fine-tuning GPT-2, fixed in advance, not tuned
    EVAL_AT = (0, 400, 1_200, 4_000, 10_000, 20_000) if SLIDES else (0, 400, 800)
    GENERATE = 30  # tokens of greedy continuation
    # Reports per pass when Qwen scores bits per character (248,320 logits per token): one at a
    # time in "reduced" (the same bits to 6 decimals, 1.4 GB less memory on a CPU)
    QWEN_SCORE_BATCH = 4 if SLIDES else 1

    # Event coding: the OIICS event divisions with over 1,000 construction reports
    EVENTS = {
        "2": "Transportation incidents",
        "4": "Falls, slips, trips",
        "5": "Exposure to harmful substances or environments",
        "6": "Contact with objects and equipment",
    }
    EVENT_SYSTEM = "You code OSHA severe-injury reports by event type. Answer with the digit only."
    ICL_TEST = 1_000  # test-year construction reports, the same for every method
    SHOTS = (0, 1, 3, 10, 30)  # examples per type in Qwen's prompt (the slides')
    PROBE_SHOTS = (1, 3, 10, 30, 100)  # labelled examples per type for the BGE probe
    SHOT_SEEDS = (0, 1, 2)  # draws of the examples
    # "reduced": Qwen codes the first 50 test reports, one draw, up to 10 examples per type
    CODING_REPORTS = ICL_TEST if SLIDES else 50
    CODING_SHOTS = SHOTS if SLIDES else (0, 1, 3, 10)
    CODING_SEEDS = SHOT_SEEDS if SLIDES else (0,)
    CODE_BATCH = 32  # at most this many reports per pass, on copies of the examples' cache
    CACHE_TOKENS = 49_152  # and at most this many cached tokens per pass
    PREFILL = 2_048  # a long prompt is read in pieces of this many tokens
    CHECK_REPORTS = 4 if SLIDES else 2  # per setting: cached and batched against reading alone
    SHOW_K = 3  # the examples per type set against none in the zero-shot figure

    # Retrieval over the MUTCD
    CHUNK_WORDS, CHUNK_STRIDE = 150, 100  # overlapping windows within a section
    TOP_K = 5  # excerpts given to the model
    MAX_ANSWER = 80  # tokens
    # The pinned Qwen3.5 revisions ship no generation_config.json: every setting is passed.
    # Greedy with no repetition penalty (1.0, as Qwen's model card sets it too).
    REPETITION_PENALTY = 1.0
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
    # The declining phrasings fixed before the pilot run, which DECLINE widened (§5 gives both)
    DECLINE_PILOT = re.compile(
        r"not in the mutcd|does not (?:cover|address|specify|provide)", re.IGNORECASE
    )
    # The manual's page furniture, headings, and its own cross-references
    FURNITURE = re.compile(
        r"\s*(Page \d+\s+MUTCD 11th Edition|MUTCD 11th Edition\s+Page \d+|Sect\. .* December 2023"
        r"|December 2023\s+Sect\.)"
    )
    LEADER = re.compile(r"\.{4,}")  # dot leaders, as in a list of figures
    BODY_PAGE = re.compile(r"\bPage \d+\b")
    FRONT_PAGE = re.compile(r"\bPage [A-Z]+-\d")
    HEADING = re.compile(r"^Section (\d[A-Z]\.\d{2}) (.+)$")
    CROSS_REF = re.compile(
        r"Sections? (\d[A-Z]\.\d{2}) (?:contains?|provides?|describes?|includes?|has) "
        r"([^.]{15,160})\."
    )
    SECTION_ID = re.compile(r"\b\d[A-Z]\.\d{2}\b")  # the manual's numbering: 2A.16

    @dataclass(frozen=True)
    class Question:
        """A question for the MUTCD, the manual's sentence that answers it, and the answer's test.

        ``evidence`` is a phrase of the manual (whitespace-normalized); every section containing
        it counts as a correct retrieval. ``answer`` holds regular expressions that a correct
        answer matches, all of them. Questions the manual does not answer have neither.
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
            "What ground clearance is needed under a sign larger than 50 square feet on "
            "multiple breakaway posts?",
            "installed on multiple breakaway posts, the clearance from the ground to the bottom "
            "of the sign shall be at least 7 feet",
            (r"\b7" + FT,),
        ),
        Question(
            "What is the minimum size of a diamond-shaped warning sign on a multi-lane road with "
            "a speed limit above 35 mph?",
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
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Okabe-Ito
    MUTED = "#595959"

    def shown(x: float, text: str) -> bool:
        """Whether ``x`` reads as the slide's ``text`` at the slide's precision ("0.66").

        Args:
            x: The unrounded value.
            text: The slide's digits.

        Returns:
            True if ``x`` rounds to ``text``.
        """
        decimals = len(text.split(".")[1]) if "." in text else 0
        return f"{x:.{decimals}f}" == text

    def require(path: Path, hint: str) -> Path:
        """``path`` if it exists; otherwise say which script writes it.

        Args:
            path: A file under ``data/``.
            hint: How to get it.

        Returns:
            ``path``.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: {hint}")
        return path

    T_START = time.perf_counter()
    print(f"device: {DEVICE}; setting: {SETTING}" + (" (the slides' GPU)" if REFERENCE_GPU else ""))
    return (
        AutoModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        BATCH,
        BGE,
        BGE_QUERY,
        BGE_REVISION,
        BODY_PAGE,
        CACHE_TOKENS,
        CHAT_TURN_END,
        CHECK_REPORTS,
        CHUNK_STRIDE,
        CHUNK_WORDS,
        CODE_BATCH,
        CODING_REPORTS,
        CODING_SEEDS,
        CODING_SHOTS,
        COS_TOL,
        CROSS_REF,
        DECLINE,
        DECLINE_PILOT,
        DEVICE,
        DOC_END,
        DOC_START,
        EVAL_AT,
        EVENTS,
        EVENT_SYSTEM,
        EXACT,
        F,
        FRONT_PAGE,
        FURNITURE,
        FWD_TOL,
        GENERATE,
        GPT2,
        GPT2_REVISION,
        GPU,
        HEADING,
        HF_TOL,
        ICL_TEST,
        LEADER,
        LM_TEST,
        LM_TRAIN,
        LOGP_TOL,
        LOSS_TOL,
        LR,
        ListedColormap,
        LogisticRegression,
        MAX_ANSWER,
        MUTCD_PDF,
        MUTED,
        OSHA_ZIP,
        PALETTE,
        PREFILL,
        PROBE_SHOTS,
        PROBE_TOL,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        QA_SYSTEM,
        QUESTIONS,
        QWEN,
        QWEN_BASE,
        QWEN_BASE_REVISION,
        QWEN_NAME,
        QWEN_REVISION,
        QWEN_SCORE_BATCH,
        Question,
        RECALL_TOL,
        REFERENCE_GPU,
        REPETITION_PENALTY,
        SDPBackend,
        SECTION_ID,
        SEED,
        SETTING,
        SHOTS,
        SHOT_SEEDS,
        SHOW_K,
        SLIDES,
        TOP_K,
        TRAIN_TOKENS,
        TRAIN_YEARS,
        TUNE_TOL,
        T_START,
        TfidfVectorizer,
        USER_TURN,
        balanced_accuracy_score,
        confusion_matrix,
        copy,
        itertools,
        math,
        np,
        pd,
        plt,
        re,
        near,
        ref,
        require,
        sdpa_kernel,
        shown,
        shutil,
        subprocess,
        time,
        torch,
        zipfile,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Warm-up: three questions, Lectures 4–18

    On your own, no notes, two minutes:

    1. Lecture 4: two vectors point the same way; one is twice as long. Their cosine similarity?
    2. Lecture 17: BERT filled in a hidden word from both sides. What changes if it may look
       only to the left?
    3. Lecture 18: CLIP screened tiles with no labels. What did it still need labels for?

    **Today:** a model that predicts the next word, asked about the manual for signs and signals.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers": mo.md(r"""
        1. 1: the cosine ignores length (checked below). Today's retrieval ranks the manual's
           excerpts by the cosine between a question's embedding and each excerpt's.
        2. It can then generate text, one word after another: that is GPT. The change is one
           mask in attention (today's picture, §2).
        3. To set the threshold: Lecture 18's 90% recall target needed labelled validation
           photos. Today the same ladder shows up for text: a prompt with no examples, then a
           few, then labels for a probe.
        """)
        }
    )
    return


@app.cell
def _(np):
    _v = np.array([3.0, 4.0, 12.0])
    _cos = float(_v @ (2 * _v) / (np.linalg.norm(_v) * np.linalg.norm(2 * _v)))
    print(f"cosine of v and 2v: {_cos}")
    assert abs(_cos - 1) < 1e-12
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Two jobs, and the attempt

    Two public texts, two jobs a civil engineer might hand to a language model:

    - **Code each severe-injury report by event type.** OSHA's Severe Injury Reports (Lecture 17)
      carry a narrative and an OIICS event code, coded by OSHA staff. Hazard trends are counted
      from the codes.
    - **Answer crews' questions from the manual.** The MUTCD is the national standard for signs,
      signals, and markings (23 CFR 655, Subpart F); states adopt it, some with supplements. "How
      high must this sign be?" has an answer in the manual, and a wrong answer is a liability.

    First the reports: Lecture 17's loader, then the construction reports (NAICS 23) of the four
    event types with over 1,000 construction reports each.
    """)
    return


@app.cell
def _(EVENTS, ICL_TEST, OSHA_ZIP, SEED, TRAIN_YEARS, np, pd, require, zipfile):
    def osha() -> pd.DataFrame:
        """The reports: event year, industry code, OIICS event code, and the narrative.

        Read from the published ZIP (latin-1 CSV), never extracted to disk.

        Returns:
            One row per report with a narrative: year, naics, text, event, lower.
        """
        hint = "save the ZIP from osha.gov, then `bash scripts/fetch_osha_sir.sh <saved zip>`"
        with (
            zipfile.ZipFile(require(OSHA_ZIP, hint)) as zf,
            zf.open("January2015toNovember2025.csv") as fh,
        ):
            df = pd.read_csv(fh, encoding="latin-1", low_memory=False, dtype={"Primary NAICS": str})
        out = pd.DataFrame(
            {
                "year": pd.to_datetime(df["EventDate"], format="%m/%d/%Y").dt.year,
                "naics": df["Primary NAICS"].fillna(""),
                "text": df["Final Narrative"]
                .fillna("")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip(),
                # OIICS event codes: the first digit is the division (4: falls, slips, trips)
                "event": df["Event"].astype("Int64").astype(str),
            }
        )
        out = out[out["text"].str.len() > 0].reset_index(drop=True)
        out["lower"] = out["text"].str.lower()
        return out

    def construction(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Construction reports (NAICS 23) of the four event types: training pool and test sample.

        Args:
            df: The reports, from ``osha()``.

        Returns:
            The training-year pool (2015-2022) and 1,000 test-year reports (2023-2025), each with
            a ``type`` column, the event code's first digit.
        """
        c = df[df["naics"].str.startswith("23")].assign(type=lambda d: d["event"].str[0])
        c = c[c["type"].isin(list(EVENTS))]
        pool = c[c["year"].between(*TRAIN_YEARS)].reset_index(drop=True)
        test = c[c["year"] >= 2023].sample(ICL_TEST, random_state=SEED).reset_index(drop=True)
        return pool, test

    def draw(pool: pd.DataFrame, k: int, seed: int) -> np.ndarray:
        """Row positions of k examples per type from the pool, in a shuffled order.

        Args:
            pool: The training-year construction reports.
            k: Examples per type.
            seed: The draw (0, 1, or 2).

        Returns:
            4k row positions; the prompts and the probes use the same ones.
        """
        rng = np.random.default_rng((SEED, k, seed))
        rows = [rng.choice(np.flatnonzero(pool["type"] == t), k, replace=False) for t in EVENTS]
        return rng.permutation(np.concatenate(rows))

    return construction, draw, osha


@app.cell
def _(EVENTS, TRAIN_YEARS, construction, np, osha, shown):
    osha_df = osha()
    coding_pool, coding_test = construction(osha_df)
    _built = osha_df[osha_df["naics"].str.startswith("23")]
    _divisions = _built["event"].str[0].value_counts()
    _other_share = float(np.mean(~_built["event"].str[0].isin(list(EVENTS))))
    print(
        f"{len(osha_df):,} reports, {osha_df['year'].min()}-{osha_df['year'].max()}; "
        f"construction: {len(_built):,}; by OIICS division: {_divisions.to_dict()}"
    )
    print(
        f"coding pool (2015-2022): {len(coding_pool):,}; test sample (2023-2025): "
        f"{len(coding_test):,}; other divisions {100 * _other_share:.1f}% of construction"
    )
    assert len(osha_df) == 105_996 and (osha_df["year"].min(), osha_df["year"].max()) == (
        2015,
        2025,
    )
    assert osha_df["year"].between(TRAIN_YEARS[0], 2025).all()
    # "the four event types with over 1,000 construction reports each": exactly these four
    assert set(_divisions[_divisions > 1_000].index) == set(EVENTS)
    assert shown(100 * _other_share, "4")  # "the other types are 4% of them"
    assert len(coding_pool) == 13_572 and len(coding_test) == 1_000  # "13,572 labelled reports"
    assert coding_pool["year"].between(*TRAIN_YEARS).all() and (coding_test["year"] >= 2023).all()
    return coding_pool, coding_test, osha_df


@app.cell
def _(
    BODY_PAGE,
    CHUNK_STRIDE,
    CHUNK_WORDS,
    CROSS_REF,
    FRONT_PAGE,
    FURNITURE,
    HEADING,
    LEADER,
    MUTCD_PDF,
    np,
    re,
    require,
    shutil,
    subprocess,
):
    def mutcd_sections() -> dict[str, tuple[str, str]]:
        """The manual's sections, in order: number -> (title, whitespace-normalized text).

        Only numbered body pages are read ("Page 1" onwards: the cover and the table of
        contents, numbered "TC-1" and so on, are dropped), without their headers and footers. A
        heading is a line that starts "Section 2A.16 ..." with a short title.

        Returns:
            The sections, keyed by number ("2A.16").

        Raises:
            FileNotFoundError: If pdftotext is not installed or the PDF is missing.
        """
        if shutil.which("pdftotext") is None:
            raise FileNotFoundError("pdftotext (poppler) is needed to read the MUTCD")
        pdf = require(MUTCD_PDF, "run `bash scripts/fetch_mutcd.sh`")
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

    def make_chunks(secs: dict[str, tuple[str, str]]) -> tuple[list[str], np.ndarray]:
        """Overlapping 150-word windows of each section, headed by its number and title.

        Args:
            secs: The sections, from ``mutcd_sections()``.

        Returns:
            The chunk texts, and the index of each section's first chunk (every section has at
            least one), so that a section's score is its best chunk's.
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

    def cross_refs(secs: dict[str, tuple[str, str]]) -> list[tuple[str, str, str]]:
        """The manual's own cross-references: "Section X contains <description>." sentences.

        Args:
            secs: The sections.

        Returns:
            (citing section, cited section, description), the description being the query.
        """
        return [
            (k, m.group(1), m.group(2))
            for k, (_, body) in secs.items()
            for m in CROSS_REF.finditer(body)
            if m.group(1) in secs and m.group(1) != k
        ]

    def paragraph_label(body: str, phrase: str) -> tuple[str, str]:
        """The MUTCD's label (Standard, Guidance, Option, Support) and paragraph of a phrase.

        Args:
            body: A section's text.
            phrase: Words of one of its paragraphs.

        Returns:
            The label and the two-digit paragraph number in force where the phrase starts.
        """
        i = body.index(phrase)
        label = re.findall(r"\b(Standard|Guidance|Option|Support):", body[:i])[-1]
        number = re.findall(r"\b(\d{2}) (?=[A-Z])", body[: i + 1])[-1]  # "03 For ..."
        return label, number

    return cross_refs, make_chunks, mutcd_sections, paragraph_label


@app.cell
def _(SECTION_ID, cross_refs, make_chunks, mutcd_sections, np, ref, subprocess, time):
    _t0 = time.perf_counter()
    secs = mutcd_sections()
    chunk_texts, chunk_starts = make_chunks(secs)
    chunk_section = np.repeat(list(secs), np.diff([*chunk_starts, len(chunk_texts)]))
    refs = cross_refs(secs)
    mutcd_words = sum(len(_b.split()) for _, _b in secs.values())
    print(
        f"MUTCD read in {time.perf_counter() - _t0:.0f} s: {len(secs)} sections, "
        f"{mutcd_words:,} words, {len(chunk_texts):,} excerpts; {len(refs)} cross-references to "
        f"{len({_r[1] for _r in refs})} sections"
    )
    assert chunk_section.shape == (len(chunk_texts),)
    assert all(SECTION_ID.fullmatch(_k) for _k in secs)
    # The slides: "954 sections, 400,027 words"; "4,032 overlapping 150-word excerpts";
    # "221 references to 174 sections". The text depends on poppler's version: another one
    # can cut the manual differently, so a mismatch names both versions.
    _v = subprocess.run(["pdftotext", "-v"], check=True, capture_output=True, text=True)
    _poppler = (_v.stderr or _v.stdout).splitlines()[0].strip()
    if ref("mutcd") is not None:
        _ours = (len(secs), mutcd_words, len(chunk_texts))
        _want = tuple(ref(f"mutcd/{_k}") for _k in ("sections", "words", "chunks"))
        assert _ours == _want, (
            f"MUTCD parsed as {_ours}, the slides {_want}: {_poppler} here, "
            f"{ref('mutcd/pdftotext')} for the slides"
        )
        _ret = ref("retrieval")
        _cited = len({_c[1] for _c in refs})
        assert (len(refs), _cited) == (_ret["refs"], _ret["cited_sections"])
    return chunk_section, chunk_starts, chunk_texts, mutcd_words, refs, secs


@app.cell
def _(coding_pool, mo, mutcd_words, secs):
    mo.md(f"""
    ### Two jobs: can a model do either?

    | Job | Data | A wrong answer costs |
    |:----|:-----|:---------------------|
    | Code each severe-injury report by event type | {len(coding_pool):,} labelled construction reports, 2015–2022 | a miscounted hazard trend |
    | Answer crews' questions: how high, how far, how long | the MUTCD: {len(secs)} sections, {mutcd_words:,} words | a sign at the wrong height |

    The question for today: can a language model do either job, and what must a person check?
    """)
    return


@app.cell
def _(SEED, TRAIN_YEARS, np, osha_df, re):
    def attempt_report() -> dict[str, object]:
        """A test-year construction fall report, cut just before the word "fell".

        Drawn at random (seed ``SEED``) among reports with "fell" after word 8 and before word
        26, so the prefix carries some context and the word is early.

        Returns:
            The prefix, the whole report, and the number of candidate reports.
        """
        test = osha_df[osha_df["year"] > TRAIN_YEARS[1]]
        pool = test[test["naics"].str.startswith("23") & test["event"].str.startswith("4")]
        words_before = (
            pool["text"].str.extract(r"^(.*?)\bfell\b", expand=False).str.split().str.len()
        )
        pool = pool[words_before.between(8, 24)]
        row = pool.iloc[int(np.random.default_rng(SEED).integers(len(pool)))]
        cut = re.search(r" fell\b", row["text"]).start()
        return {"prefix": row["text"][:cut], "report": row["text"], "candidates": len(pool)}

    attempt = attempt_report()
    print(attempt)
    # The notes: "drawn at random among the 1,165 construction fall reports with 'fell' between
    # words 9 and 25", and the whole report
    assert attempt["candidates"] == 1_165
    assert attempt["report"] == (
        "An employee was descending an 8-foot ladder. When he was on the fourth rung, the ladder "
        "fell down and the employee fell on top of it, resulting in a broken arm."
    )
    assert attempt["prefix"] == (
        "An employee was descending an 8-foot ladder. When he was on the fourth rung, the ladder"
    )
    assert 9 <= len(attempt["prefix"].split()) + 1 <= 25  # "fell" is word 17
    return (attempt,)


@app.cell
def _(attempt, mo):
    mo.md(f"""
    ### Your turn: finish the report

    > {attempt["prefix"]} …

    - Write the next word, and how sure you are, as a probability.
    - Then the next three words.

    The report is a real test-year one (2023–2025), drawn at random among the
    {attempt["candidates"]:,} construction fall reports with "fell" between words 9 and 25, and
    cut just before that word. Keep your guesses: GPT-2 guesses next, before and after it reads
    20,000 reports.
    """)
    return


@app.cell
def _(attempt, mo):
    mo.accordion({"The whole report": mo.md(f"> {attempt['report']}")})
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. One picture: one token at a time, looking back

    GPT is Lecture 17's transformer with one change: in attention, position $t$ may look only at
    positions $1, \dots, t$ (the shaded triangle below). The last row reads the whole prefix and
    outputs a probability for every token of the vocabulary as the next one. The tokens are
    GPT-2's byte pairs (Lecture 17); · marks a leading space, and `<start>` is the end-of-text
    token GPT-2 reads as the start of a document.

    | Model (year) | Parameters | How it is used |
    |:-------------|:-----------|:---------------|
    | GPT-1 (2018) | 117 million | fine-tuned for each task |
    | GPT-2 (2019) | 124 million to 1.5 billion | zero-shot: the task in words |
    | GPT-3 (2020) | 175 billion | few-shot: examples in the prompt |
    | ChatGPT (2022) | not published | tuned on instructions and human preferences |
    | Today | GPT-2, 124 million; Qwen3.5, 0.75 billion | all four, on the reports and the manual |

    Every model in the table is trained the same way, to predict the next token of web text;
    what changed is the size and how it is used (the GPT-1 to GPT-3 sizes are the papers'; ours
    are counted below and in §4).
    """)
    return


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    DEVICE,
    DOC_START,
    F,
    GPT2,
    GPT2_REVISION,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    math,
    ref,
    torch,
):
    gpt2_tok = AutoTokenizer.from_pretrained(GPT2, revision=GPT2_REVISION)
    gpt2_tok.pad_token = gpt2_tok.eos_token  # padding reuses end-of-text; it is always masked
    gpt2_base = AutoModelForCausalLM.from_pretrained(GPT2, revision=GPT2_REVISION).to(DEVICE).eval()

    def doc_start(tok: PreTrainedTokenizerBase) -> str:
        """The token every document starts after, for every model here: ``<|endoftext|>``.

        It is GPT-2's BOS, and both Qwen3.5 releases (no BOS) separate documents with it. It is
        named, not read from the tokenizer: the chat model's EOS is ``<|im_end|>``, a chat
        turn's end, and scoring after it would give the two Qwen releases different starts.

        Args:
            tok: GPT-2's tokenizer or either Qwen3.5 release's.

        Returns:
            The token's text, checked to be one special token for this tokenizer.
        """
        s = DOC_START
        assert s in tok.all_special_tokens and len(tok.encode(s, add_special_tokens=False)) == 1
        return s

    def show_tokens(tok: PreTrainedTokenizerBase, ids: list[int]) -> list[str]:
        """Tokens as text, with a leading space shown as a middle dot and the start as <start>."""
        start = tok.convert_tokens_to_ids(doc_start(tok))
        return ["<start>" if i == start else tok.decode([i]).replace(" ", "·") for i in ids]

    def next_token(
        tok: PreTrainedTokenizerBase, model: PreTrainedModel, prefix: str, word: str = " fell"
    ) -> dict[str, object]:
        """The five most likely next tokens after the prefix, and the probability of ``word``.

        Args:
            tok: The model's tokenizer.
            model: A causal language model.
            prefix: The text so far (the document start is prepended).
            word: A one-token continuation to look up.

        Returns:
            top (tokens), p (their probabilities), p_word, and rank_word.
        """
        ids = tok(doc_start(tok) + prefix, return_tensors="pt")["input_ids"].to(DEVICE)
        with torch.no_grad():
            p = F.softmax(model(input_ids=ids).logits[0, -1].double(), dim=-1)
        top = torch.topk(p, 5)
        w = tok.encode(word)
        assert len(w) == 1, w
        return {
            "top": show_tokens(tok, top.indices.tolist()),
            "p": top.values.tolist(),
            "p_word": float(p[w[0]]),
            "rank_word": int((p > p[w[0]]).sum()) + 1,
        }

    def surprise(
        tok: PreTrainedTokenizerBase, model: PreTrainedModel, text: str
    ) -> dict[str, object]:
        """Bits of surprise, -log2 p(token | tokens before), for each token of the text."""
        ids = tok(doc_start(tok) + text, return_tensors="pt")["input_ids"].to(DEVICE)
        with torch.no_grad():
            logp = F.log_softmax(model(input_ids=ids).logits[0, :-1].double(), dim=-1)  # (T-1, V)
        bits = -logp.gather(1, ids[0, 1:, None])[:, 0] / math.log(2)
        return {"tokens": show_tokens(tok, ids[0, 1:].tolist()), "bits": bits.tolist()}

    gpt2_params = sum(_p.numel() for _p in gpt2_base.parameters())  # tied embeddings once
    print(f"GPT-2: {gpt2_params:,} parameters, vocabulary {gpt2_base.config.vocab_size:,}")
    # "Today: GPT-2, 124 million"; the notes: "a probability for every token of the vocabulary
    # (50,257 for GPT-2)"
    assert gpt2_base.config.vocab_size == len(gpt2_tok) == 50_257  # GPT-2's byte pairs
    assert ref("parameters/gpt2") in (None, gpt2_params)
    return doc_start, gpt2_base, gpt2_params, gpt2_tok, next_token, show_tokens, surprise


@app.cell
def _(ListedColormap, MUTED, PALETTE, np, plt):
    def draw_picture(tokens: list[str], panels: dict[str, dict | None]) -> plt.Figure:
        """The causal mask over the prefix, then each model's five likeliest next tokens.

        Args:
            tokens: The prefix's tokens, ``<start>`` first.
            panels: Model name -> ``next_token`` output, or None for a question mark.

        Returns:
            The figure.
        """
        n = len(tokens)
        fig, axes = plt.subplots(
            1,
            1 + len(panels),
            figsize=(11, 4.6),
            gridspec_kw={"width_ratios": [1.3] + [1] * len(panels)},
            layout="constrained",
        )
        mask = np.tril(np.ones((n, n)))
        mask[-1, :] *= 2  # the last row: the position that predicts the next word
        axes[0].imshow(mask, cmap=ListedColormap(["white", "#d6e4f2", PALETTE[0]]))
        axes[0].set_xticks([])
        axes[0].set_yticks(range(n), tokens, fontsize=8)
        axes[0].set_xticks(np.arange(n + 1) - 0.5, minor=True)
        axes[0].set_yticks(np.arange(n + 1) - 0.5, minor=True)
        axes[0].grid(which="minor", color="white", lw=0.6)
        axes[0].tick_params(which="minor", length=0)
        axes[0].set_xlabel("the tokens it may look at", fontsize=9)
        axes[0].set_title("each token sees only earlier ones", fontsize=10)
        for ax, (name, d) in zip(axes[1:], panels.items(), strict=True):
            ax.set_title(name, fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_xlabel("probability of the next token", fontsize=9)
            if d is None:
                ax.text(0.5, 0.5, "?", fontsize=40, ha="center", va="center", color=MUTED)
                ax.set_yticks([])
                continue
            y = np.arange(5)[::-1]
            ax.barh(y, d["p"], color=PALETTE[0])
            ax.set_yticks(y, d["top"], fontsize=10)
            for yy, p in zip(y, d["p"], strict=True):
                ax.text(p + 0.02, yy, f"{p:.2f}", va="center", fontsize=9)
        return fig

    return (draw_picture,)


@app.cell
def _(
    FWD_TOL,
    GPU,
    LM_TRAIN,
    attempt,
    doc_start,
    draw_picture,
    gpt2_base,
    gpt2_tok,
    next_token,
    np,
    ref,
    show_tokens,
):
    next_released = next_token(gpt2_tok, gpt2_base, attempt["prefix"])
    _ids = gpt2_tok(doc_start(gpt2_tok) + attempt["prefix"])["input_ids"]
    prefix_tokens = show_tokens(gpt2_tok, _ids)
    print(next_released)
    # The picture and its alt text: the five tokens and their probabilities, and "fell"'s rank
    _want = ref("next_token/GPT-2 as released")
    if _want is not None:
        assert next_released["top"] == _want["top"]
        assert next_released["rank_word"] == _want["rank_word"] == 2
        assert np.abs(np.array(next_released["p"]) - _want["p"]).max() < FWD_TOL
        assert abs(next_released["p_word"] - _want["p_word"]) < FWD_TOL
        if GPU:  # the slide's two decimals
            assert [f"{_p:.2f}" for _p in next_released["p"]] == [f"{_p:.2f}" for _p in _want["p"]]
    assert prefix_tokens[0] == "<start>" and prefix_tokens[-2:] == ["·the", "·ladder"]
    draw_picture(
        prefix_tokens, {"GPT-2 as released": next_released, f"after {LM_TRAIN:,} reports": None}
    )
    return next_released, prefix_tokens


@app.cell
def _(mo, next_released):
    mo.md(f"""
    The report's next word is "fell". GPT-2 as released gives it
    {100 * next_released["p_word"]:.0f}% (rank {next_released["rank_word"]}). Compare with your
    probability. The right panel fills in §4, after GPT-2 reads 20,000 training-year reports.
    Nothing here was labelled: the report's own next word is the target, as BERT's hidden word
    was in Lecture 17.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Derive

    **Assumption:** each token can be predicted from the tokens before it, and the training text
    is like the text we will see.

    **The chain rule.** For any distribution over sequences, by the definition of conditional
    probability applied $T-1$ times,

    $$p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t}) .$$

    This is exact for any order; left to right is a choice, and it is what lets the model
    generate one token after another.

    **The loss.** Training minimizes the average surprise, Lecture 6's cross-entropy with the
    next token as the label, at every position at once:

    $$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p(x_t \mid x_{<t}) .$$

    The mask lets all $T$ predictions train in one pass: one forward pass gives every position's
    prediction, each from its prefix only (checked below against $T$ separate passes).

    **Perplexity and bits.** Perplexity $= e^{\mathcal{L}}$ (the loss in nats), the number of
    equally likely choices the surprise amounts to; in bits, $2^{\mathcal{L}/\ln 2}$, the same
    number. **Bits per character** divide the total surprise by the number of characters
    instead of tokens, $\sum_t -\log_2 p(x_t \mid x_{<t}) \,/\, n_{\text{chars}}$, so models that
    cut the same text into different tokens compare.
    """)
    return


@app.cell
def _(DEVICE, F, FWD_TOL, attempt, doc_start, gpt2_base, gpt2_tok, math, np, ref, surprise, torch):
    _text = attempt["prefix"] + " fell"
    surprise_released = surprise(gpt2_tok, gpt2_base, _text)
    _bits = np.array(surprise_released["bits"])
    _loss = float(_bits.mean() * math.log(2))  # nats per token
    _ppl, _ppl_bits = math.exp(_loss), 2 ** float(_bits.mean())
    _bpc = float(_bits.sum() / len(_text))
    print(
        f"{len(_bits)} tokens; loss {_loss:.4f} nats/token; perplexity {_ppl:.1f} "
        f"(2^bits: {_ppl_bits:.1f}); {_bpc:.3f} bits per character"
    )
    # "22 tokens after <start>"; the alt text: "fell" costs 3.7 bits as released
    assert len(_bits) == 22 and surprise_released["tokens"][-1] == "·fell"
    _want = ref("surprise/GPT-2 as released")
    if _want is not None:
        assert surprise_released["tokens"] == _want["tokens"]
        assert np.abs(_bits - _want["bits"]).max() < FWD_TOL * 10  # bits: up to about 20
    # The notes: perplexity is e to the loss, and 2 to the bits: the same quantity
    assert abs(_ppl - _ppl_bits) < 1e-9 * _ppl
    # The alt text: "surprise is high at the start and at words specific to the report": the
    # three most surprising tokens are ·descending, ·fourth, ·8, and the first three tokens are
    # above the median
    _order = np.argsort(-_bits)
    assert [surprise_released["tokens"][_i] for _i in _order[:3]] == [
        "·descending",
        "·fourth",
        "·8",
    ]
    assert _bits[:3].min() > np.median(_bits)

    # One pass gives every position's prediction, each from its prefix only: compare with one
    # forward pass per prefix
    _ids = gpt2_tok(doc_start(gpt2_tok) + _text, return_tensors="pt")["input_ids"].to(DEVICE)
    with torch.no_grad():
        _full = F.log_softmax(gpt2_base(input_ids=_ids).logits[0].double(), dim=-1)
        _alone = torch.stack(
            [gpt2_base(input_ids=_ids[:, : _t + 1]).logits[0, -1] for _t in range(_ids.shape[1])]
        )
        _alone = F.log_softmax(_alone.double(), dim=-1)
    one_pass_diff = float((_full - _alone).abs().max())
    print(
        f"one pass against {_ids.shape[1]} passes: largest log-probability difference {one_pass_diff:.1e}"
    )
    assert _full.shape == _alone.shape == (_ids.shape[1], 50_257)
    assert one_pass_diff < 1e-3  # float32 round-off: different matrix shapes, same answer
    return (surprise_released,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Your turn: predict the check

    ```python
    ids = tok(start + text)                     # 22 tokens after <start>
    logits = model(ids).logits                  # (T, V)
    ours = cross_entropy(logits[:-1], ids[1:])  # position t scores token t+1
    hf = model(ids, labels=ids).loss            # Hugging Face's own loss
    bug = cross_entropy(logits, ids)            # the shift forgotten
    ids2 = ids, last token changed to " crane"  # do earlier logits move?
    ```

    The text is the attempt's sentence through "fell". How close are `ours` and `hf`? Is `bug`
    higher or lower? Do earlier positions' logits move? Write your answers, then run the check.
    Our loss is written twice: from scratch in NumPy (log-sum-exp, then the target's
    log-probability), and with PyTorch's `cross_entropy`.
    """)
    return


@app.cell
def _(
    DEVICE,
    EXACT,
    F,
    FWD_TOL,
    GPU,
    HF_TOL,
    LOSS_TOL,
    attempt,
    doc_start,
    gpt2_base,
    gpt2_tok,
    np,
    ref,
    surprise_released,
    torch,
):
    _text = doc_start(gpt2_tok) + attempt["prefix"] + " fell"
    _ids = gpt2_tok(_text, return_tensors="pt")["input_ids"].to(DEVICE)
    with torch.no_grad():
        _logits = gpt2_base(input_ids=_ids).logits[0].double()  # (T, V)
        _hf = gpt2_base(input_ids=_ids, labels=_ids).loss.double()  # shifts inside, float32
        _other = _ids.clone()
        _other[0, -1] = gpt2_tok.encode(" crane")[0]
        _changed = gpt2_base(input_ids=_other).logits[0].double()
    assert _logits.shape == (23, 50_257)
    # From scratch in NumPy: log-softmax by log-sum-exp, then each target's log-probability
    _z, _y = _logits.cpu().numpy(), _ids[0].cpu().numpy()
    _m = _z.max(axis=1, keepdims=True)
    _logp = _z - (_m + np.log(np.exp(_z - _m).sum(axis=1, keepdims=True)))
    _ours_np = float(-_logp[np.arange(len(_y) - 1), _y[1:]].mean())  # position t scores t+1
    _ours = float(F.cross_entropy(_logits[:-1], _ids[0, 1:]))
    loss_check = {
        "ours": _ours,
        "hf": float(_hf),
        "diff": abs(_ours - float(_hf)),
        "no_shift": float(F.cross_entropy(_logits, _ids[0])),
        "earlier_max_change": float((_changed[:-1] - _logits[:-1]).abs().max()),
        "last_max_change": float((_changed[-1] - _logits[-1]).abs().max()),
        # the shift forgotten scores each position on the token it already sees: how often is
        # that token the model's first guess?
        "repeat_top": int((_z.argmax(axis=1) == _y).sum()),
    }
    print(loss_check, f"NumPy against PyTorch: {abs(_ours_np - _ours):.1e}")
    assert abs(_ours_np - _ours) < LOSS_TOL
    assert loss_check["diff"] < HF_TOL
    # The same loss as the surprise bars: their mean, in nats
    assert abs(np.mean(surprise_released["bits"]) * np.log(2) - _ours) < LOSS_TOL + EXACT
    # The table: our loss, Hugging Face's, the difference, the shift forgotten, the mask
    _want = ref("check")
    if _want is not None:
        for _k in ("ours", "hf", "no_shift", "last_max_change"):
            assert abs(loss_check[_k] - _want[_k]) < FWD_TOL * max(1, _want[_k]), _k
        if GPU:  # the table's digits
            for _k, _d in (("ours", 4), ("hf", 4), ("no_shift", 2), ("last_max_change", 2)):
                assert f"{loss_check[_k]:.{_d}f}" == f"{_want[_k]:.{_d}f}", _k
            assert f"{loss_check['diff']:.1e}" == f"{_want['diff']:.1e}"
    assert loss_check["earlier_max_change"] < FWD_TOL and loss_check["last_max_change"] > 1
    if GPU:
        assert loss_check["earlier_max_change"] == 0.0  # "exactly as they were"
    # "which a next-token model rarely predicts again": never its first guess here
    assert loss_check["repeat_top"] == 0
    return (loss_check,)


@app.cell
def _(loss_check, mo):
    mo.md(f"""
    ### Same loss; the mask holds

    | Check, the attempt's sentence | Value |
    |:------------------------------|------:|
    | Our loss, nats per token | {loss_check["ours"]:.4f} |
    | Hugging Face's | {loss_check["hf"]:.4f} |
    | Difference | {loss_check["diff"]:.1e} |
    | The shift forgotten | {loss_check["no_shift"]:.2f} |
    | Last token changed: largest change in any earlier logit | {loss_check["earlier_max_change"]:.0f} |
    | … in the last position's logits | {loss_check["last_max_change"]:.2f} |

    Ours is computed in float64 from the same logits, Hugging Face's in float32: they agree to
    {loss_check["diff"]:.1e}. Forgetting the shift scores each position on the token it already
    sees, which a next-token model rarely predicts again (it is the first guess at
    {loss_check["repeat_top"]} of the 23 positions here): the loss jumps to
    {loss_check["no_shift"]:.2f}. Changing the last token leaves every earlier position's logits
    as they were (largest change {loss_check["earlier_max_change"]:.0f}): the mask holds, so no
    position has seen its own answer.
    """)
    return


@app.cell
def _(LM_TEST, LM_TRAIN, mo):
    mo.md(rf"""
    ## 4. Build and verify

    ### Fine-tuning: 400 reports beat 6× the parameters

    One pass of next-token training over {LM_TRAIN:,} training-year reports (2015–2022), scored
    on {LM_TEST:,} test-year reports (2023–2025) in bits per character (the slides: 20,000 and
    2,000). Each report is `<start> + text + end-of-text`, cut at 128 tokens; AdamW at rate
    $5 \times 10^{{-5}}$, batches of 16, gradients clipped at norm 1, nothing tuned. Dropout is
    on while training (`model.train()`) and off while scoring (`model.eval()`); training uses
    PyTorch's plain attention, whose backward pass is deterministic. Two references: token
    frequencies alone (a unigram model with add-one smoothing, Forsyth §13.1.3), and
    Qwen3.5-0.8B-Base as released, a pretrained model with 6 times the parameters (and its chat
    release, for comparison).
    """)
    return


@app.cell
def _(EXACT, LM_TEST, LM_TRAIN, SEED, SLIDES, TRAIN_YEARS, gpt2_tok, math, near, np, osha_df):
    # The slides' samples (20,000 and 2,000 reports); "reduced" takes the first of each
    lm_train = (
        osha_df.loc[osha_df["year"].between(*TRAIN_YEARS), "text"]
        .sample(20_000, random_state=SEED)
        .tolist()[:LM_TRAIN]
    )
    lm_test = (
        osha_df.loc[osha_df["year"] >= 2023, "text"].sample(2_000, random_state=SEED).tolist()
    )[:LM_TEST]

    def unigram_bits_per_char(tok: object, train: list[str], test: list[str]) -> float:
        """The naive baseline: GPT-2's tokens, each predicted by its training frequency alone.

        Add-one smoothing over the whole vocabulary, so unseen tokens get a finite cost.

        Args:
            tok: GPT-2's tokenizer.
            train: Training reports (the counts).
            test: Test reports (the score).

        Returns:
            Test bits per character.
        """
        counts = np.ones(len(tok))
        for ids in tok(train)["input_ids"]:
            np.add.at(counts, ids, 1)
        logp = np.log(counts / counts.sum())
        nll = -sum(logp[ids].sum() for ids in tok(test)["input_ids"])
        return float(nll / math.log(2) / sum(map(len, test)))

    unigram_bpc = unigram_bits_per_char(gpt2_tok, lm_train, lm_test)
    print(
        f"{len(lm_train):,} training reports, {len(lm_test):,} test; unigram {unigram_bpc:.4f} bits/char"
    )
    if SLIDES:  # counted on the CPU: exact
        assert near(unigram_bpc, "unigram_bits_per_char", EXACT)
    return lm_test, lm_train, unigram_bpc


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    CHAT_TURN_END,
    DEVICE,
    F,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    QWEN,
    QWEN_REVISION,
    doc_start,
    math,
    torch,
):
    def context_length(model: PreTrainedModel) -> int:
        """The longest input the model's positions cover (GPT-2: 1,024; Qwen3.5: 262,144)."""
        return int(model.config.get_text_config().max_position_embeddings)

    def nll_nats(
        tok: PreTrainedTokenizerBase, model: PreTrainedModel, texts: list[str], batch: int = 16
    ) -> float:
        """Total negative log-likelihood (nats) of the texts, each token given the ones before.

        Each text is prefixed with the document start (``doc_start``) so that its first token is
        predicted too; padding is on the right and masked. Logits are float32.

        Args:
            tok: The model's tokenizer.
            model: A causal language model.
            texts: The texts.
            batch: Texts per forward pass.

        Returns:
            The total, in nats.
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
                logits = model(**enc).logits[:, :-1].float()  # (B, T-1, V): t predicts t+1
                target = enc["input_ids"][:, 1:].masked_fill(
                    enc["attention_mask"][:, 1:] == 0, -100
                )
                # Per-token losses summed in float64: a summing reduction on the GPU adds with
                # atomics, whose order (and last digits) change from run to run
                nll = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
                total += float(nll.double().sum())
        return total

    def bits_per_char(
        tok: PreTrainedTokenizerBase, model: PreTrainedModel, texts: list[str], batch: int = 16
    ) -> float:
        """Bits per character: comparable across tokenizers, unlike per-token perplexity."""
        model.eval()
        return nll_nats(tok, model, texts, batch) / math.log(2) / sum(map(len, texts))

    def load_qwen(name: str, revision: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        """A Qwen3.5 text model in float32, on the device, and its tokenizer.

        Qwen3.5 is released as a vision-language model; ``AutoModelForCausalLM`` builds its
        text model alone (``Qwen3_5ForCausalLM``) from the same weights.

        Args:
            name: The base model or the chat model.
            revision: Its pinned revision.

        Returns:
            The tokenizer and the model, in eval mode.
        """
        tok = AutoTokenizer.from_pretrained(name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.float32)
        assert type(model).__name__ == "Qwen3_5ForCausalLM"
        return tok, model.to(DEVICE).eval()

    qwen_tok = AutoTokenizer.from_pretrained(QWEN, revision=QWEN_REVISION)
    # The chat model's tokenizer: no BOS; a chat turn ends with <|im_end|>, its EOS
    assert qwen_tok.bos_token is None and qwen_tok.eos_token == CHAT_TURN_END
    return bits_per_char, context_length, load_qwen, qwen_tok


@app.cell
def _(
    BATCH,
    DEVICE,
    EVAL_AT,
    GPU,
    LR,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    SDPBackend,
    SEED,
    TRAIN_TOKENS,
    bits_per_char,
    copy,
    doc_start,
    gpt2_base,
    gpt2_tok,
    lm_test,
    lm_train,
    sdpa_kernel,
    time,
    torch,
):
    def fine_tune(
        tok: PreTrainedTokenizerBase, model: PreTrainedModel, train: list[str], test: list[str]
    ) -> dict[str, object]:
        """One pass of next-token training over the reports; the test is scored at ``EVAL_AT``.

        Each report is ``start + text + end-of-text``, truncated at ``TRAIN_TOKENS``; AdamW at
        ``LR`` (PyTorch's AdamW with its weight decay 0.01, stated), gradients clipped at norm 1.

        Args:
            tok: GPT-2's tokenizer.
            model: The model to train, in place.
            train: Training reports, in order.
            test: Test reports.

        Returns:
            Bits per character at each count of reports seen, the seconds (all, and the
            training steps alone, at each evaluation), and the share of reports truncated.
        """
        assert all(n % BATCH == 0 for n in EVAL_AT) and len(train) == EVAL_AT[-1]
        opt = torch.optim.AdamW(
            model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )
        curve = {0: bits_per_char(tok, model, test)}
        tok.padding_side = "right"
        t0, truncated, train_s, train_at = time.perf_counter(), 0, 0.0, {}
        for i in range(0, len(train), BATCH):
            t_step = time.perf_counter()
            model.train()  # dropout on
            texts = [doc_start(tok) + t + tok.eos_token for t in train[i : i + BATCH]]
            truncated += sum(len(ids) > TRAIN_TOKENS for ids in tok(texts)["input_ids"])
            enc = tok(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=TRAIN_TOKENS
            ).to(DEVICE)
            labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
            with sdpa_kernel(SDPBackend.MATH):  # fused kernels' backward is not deterministic
                loss = model(**enc, labels=labels).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if (seen := i + BATCH) in EVAL_AT:
                if GPU:
                    torch.cuda.synchronize()  # the clock waits for the GPU
                train_s += time.perf_counter() - t_step
                train_at[seen] = train_s
                curve[seen] = bits_per_char(tok, model, test)
                print(f"  {seen:,} reports: {curve[seen]:.3f} bits/char", flush=True)
            else:
                train_s += time.perf_counter() - t_step
        model.eval()
        return {
            "bits_per_char": curve,
            "seconds": time.perf_counter() - t0,
            "train_seconds": train_at,
            "truncated_share": truncated / len(train),
        }

    gpt2_tuned = copy.deepcopy(gpt2_base)
    torch.manual_seed(SEED)  # the dropout masks: the figure script's generator state here
    tune = fine_tune(gpt2_tok, gpt2_tuned, lm_train, lm_test)
    print(
        f"one pass in {tune['seconds']:.0f} s ({tune['train_seconds'][EVAL_AT[-1]]:.0f} s of "
        f"training steps); {100 * tune['truncated_share']:.2f}% of reports cut at "
        f"{TRAIN_TOKENS} tokens"
    )
    return gpt2_tuned, tune


@app.cell
def _(
    DOC_START,
    EVAL_AT,
    EXACT,
    FWD_TOL,
    GPU,
    LR,
    QWEN,
    QWEN_BASE,
    QWEN_BASE_REVISION,
    QWEN_REVISION,
    QWEN_SCORE_BATCH,
    REFERENCE_GPU,
    SLIDES,
    TUNE_TOL,
    bits_per_char,
    gpt2_params,
    gpt2_tok,
    itertools,
    lm_test,
    load_qwen,
    near,
    ref,
    torch,
    tune,
    unigram_bpc,
):
    # The size reference: Qwen3.5-0.8B pretrained only (the base model), and the chat model, each
    # scoring every report after the same start token as the slides (numbers.json records it)
    assert ref("doc_start") in (None, DOC_START)
    qwen_bpc, qwen_params, _tokens = {}, {}, {}
    for _key, (_name, _rev) in {"base": (QWEN_BASE, QWEN_BASE_REVISION),
                                "chat": (QWEN, QWEN_REVISION)}.items():  # fmt: skip
        _tok, _model = load_qwen(_name, _rev)
        qwen_bpc[_key] = bits_per_char(_tok, _model, lm_test, QWEN_SCORE_BATCH)
        qwen_params[_key] = sum(_p.numel() for _p in _model.parameters())  # tied embeddings once
        _tokens[_key] = sum(map(len, _tok(lm_test, add_special_tokens=False)["input_ids"]))
        del _model
        if GPU:
            torch.cuda.empty_cache()
    _tokens["gpt2"] = sum(map(len, gpt2_tok(lm_test)["input_ids"]))
    print(
        f"Qwen3.5-0.8B-Base {qwen_bpc['base']:.4f} bits/char, the chat model {qwen_bpc['chat']:.4f}; "
        f"{qwen_params['base']:,} parameters ({qwen_params['base'] / gpt2_params:.1f}x GPT-2's); "
        f"test tokens: {_tokens}"
    )
    # Per-token scores would compare different units: the tokenizers cut the same reports into
    # different numbers of tokens (Qwen's vocabulary is five times GPT-2's)
    assert _tokens["gpt2"] != _tokens["base"] == _tokens["chat"]
    assert qwen_params["base"] == qwen_params["chat"]  # the same network, tuned for chat
    assert ref("parameters/qwen_base") in (None, qwen_params["base"])
    _curve = tune["bits_per_char"]
    assert list(_curve) == list(EVAL_AT)
    if SLIDES:  # the slides' 2,000 test reports
        assert near(_curve[0], "fine_tune/bits_per_char/0", FWD_TOL)  # GPT-2 as released
        assert near(qwen_bpc["base"], "qwen_base_bits_per_char", FWD_TOL)
        assert near(qwen_bpc["chat"], "qwen_bits_per_char", FWD_TOL)
        assert near(tune["truncated_share"], "fine_tune/truncated_share", EXACT)
    if REFERENCE_GPU:  # the curve after fine-tuning, as on the slides
        assert all(near(_curve[_k], f"fine_tune/bits_per_char/{_k}", TUNE_TOL) for _k in EVAL_AT)
        assert all(f"{_curve[_k]:.2f}" == f"{ref(f'fine_tune/bits_per_char/{_k}'):.2f}"
                   for _k in EVAL_AT)  # fmt: skip
    # The curve falls at every evaluation, and every point is below the unigram bracket
    _v = list(_curve.values())
    assert all(_a > _b for _a, _b in itertools.pairwise(_v)) and max(_v) < unigram_bpc
    assert LR == 5e-5
    return qwen_bpc, qwen_params


@app.cell
def _(
    EVAL_AT, MUTED, PALETTE, QWEN_NAME, gpt2_params, plt, qwen_bpc, qwen_params, tune, unigram_bpc
):
    _seen = list(EVAL_AT)
    _xs = [max(_k, 150) for _k in _seen]  # 0 sits at the left edge of the log axis
    _fig, _ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    _ax.plot(_xs, list(tune["bits_per_char"].values()), marker="o", lw=2, color=PALETTE[0],
             label="GPT-2 (124M), fine-tuned on the reports")  # fmt: skip
    _ax.axhline(unigram_bpc, ls="--", lw=2, color=MUTED, label="token frequencies alone")
    _ax.axhline(qwen_bpc["base"], ls="-.", lw=2, color=PALETTE[1],
                label=f"{QWEN_NAME}-Base ({qwen_params['base'] / gpt2_params:.0f}× the "
                "parameters), as released")  # fmt: skip
    _ax.set_xscale("log")
    _ax.set_xticks(_xs, ["0"] + [f"{_k:,}" for _k in _seen[1:]])
    _ax.minorticks_off()
    _ax.set_xlabel("training-year reports seen (one pass)")
    _ax.set_ylabel("test bits per character")
    _ax.set_ylim(0, None)
    _ax.legend(frameon=False, fontsize=9, loc="lower left")
    _fig
    return


@app.cell
def _(EVAL_AT, LM_TRAIN, gpt2_params, mo, qwen_bpc, qwen_params, tune, unigram_bpc):
    _c = tune["bits_per_char"]
    # The title: 400 reports (25 steps) take GPT-2 below the base model six times its size
    _ratio = qwen_params["base"] / gpt2_params
    assert f"{_ratio:.0f}" == "6" and 400 // 16 == 25 and 400 in EVAL_AT
    assert _c[400] < qwen_bpc["base"] < qwen_bpc["chat"]  # and the base is the fairer reference
    mo.md(f"""
    Token frequencies alone give {unigram_bpc:.2f} bits per character; GPT-2 as released
    {_c[0]:.2f}. One pass over {LM_TRAIN:,} reports ({tune["seconds"]:.0f} seconds here,
    evaluations included) brings it to {_c[LM_TRAIN]:.2f}. Qwen3.5-0.8B-Base, {qwen_params["base"]:,}
    parameters ({_ratio:.1f} times GPT-2's), scores {qwen_bpc["base"]:.2f} as released; its chat
    release, after the same start token, {qwen_bpc["chat"]:.2f}, so the base model is the fair
    reference for raw text. Scaling
    laws (Kaplan et al., 2020) say loss falls predictably with parameters and data, measured on
    text like the training text. On a narrow kind of text, a little of it beats a lot of size:
    after 400 reports (25 steps) GPT-2 is already below Qwen ({_c[400]:.2f} against
    {qwen_bpc["base"]:.2f}).

    **Training cut-off.** Qwen3.5 was released in February 2026, and its training cut-off is not
    published: its training text may include these 2023–2025 test reports (OSHA publishes them
    online) and the 2023 MUTCD. GPT-2 (2019) cannot have seen either. A model that has read the
    test set scores better than it should, so here the comparison, if anything, favours Qwen.
    """)
    return


@app.cell
def _(
    FWD_TOL,
    LM_TRAIN,
    REFERENCE_GPU,
    TUNE_TOL,
    attempt,
    draw_picture,
    gpt2_tok,
    gpt2_tuned,
    near,
    next_released,
    next_token,
    np,
    prefix_tokens,
    ref,
    surprise,
    surprise_released,
):
    next_tuned = next_token(gpt2_tok, gpt2_tuned, attempt["prefix"])
    surprise_tuned = surprise(gpt2_tok, gpt2_tuned, attempt["prefix"] + " fell")
    _before, _after = np.array(surprise_released["bits"]), np.array(surprise_tuned["bits"])
    tokens_lower = int((_after < _before).sum())
    print(next_tuned)
    print(f"less surprising after fine-tuning: {tokens_lower} of {len(_after)} tokens; "
          f'"fell" {_before[-1]:.2f} -> {_after[-1]:.2f} bits')  # fmt: skip
    _name = f"after {LM_TRAIN:,} reports"
    if REFERENCE_GPU:
        # The picture's right panel and the surprise bars, as on the slides (their alt text)
        _want = ref(f"next_token/{_name}")
        assert next_tuned["top"] == _want["top"] and next_tuned["rank_word"] == _want["rank_word"]
        assert np.abs(np.array(next_tuned["p"]) - _want["p"]).max() < TUNE_TOL
        assert [f"{_p:.2f}" for _p in next_tuned["p"]] == [f"{_p:.2f}" for _p in _want["p"]]
        assert f"{100 * next_tuned['p_word']:.0f}" == f"{100 * _want['p_word']:.0f}"
        assert tokens_lower == ref("surprise_lower")
        assert abs(_after[-1] - ref(f"surprise/{_name}/last")) < 10 * TUNE_TOL
        assert f"{_after[-1]:.1f}" == f"{ref(f'surprise/{_name}/last'):.1f}"
    assert near(_before[-1], "surprise/GPT-2 as released/last", FWD_TOL)
    assert next_tuned["p_word"] > next_released["p_word"]  # "fell" more likely after the reports
    draw_picture(prefix_tokens, {"GPT-2 as released": next_released, _name: next_tuned})
    return surprise_tuned, tokens_lower


@app.cell
def _(MUTED, PALETTE, np, plt, surprise_released, surprise_tuned):
    _x = np.arange(len(surprise_released["tokens"]))
    _fig, _ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    _ax.bar(_x - 0.2, surprise_released["bits"], 0.4, color=MUTED, label="GPT-2 as released")
    _ax.bar(_x + 0.2, surprise_tuned["bits"], 0.4, color=PALETTE[0], hatch="//",
            label="after 20,000 reports")  # fmt: skip
    _ax.set_xticks(_x, surprise_released["tokens"], rotation=60, ha="right", fontsize=9)
    _ax.set_ylabel("surprise, bits (-log2 p)")
    _ax.set_title(
        r"$p(x_1,\dots,x_T)=\prod_t\, p(x_t\mid x_{<t})$,    "
        r"loss $=-\frac{1}{T}\sum_t \log p(x_t\mid x_{<t})$",
        fontsize=12,
        loc="left",
    )
    _ax.legend(frameon=False, fontsize=9)
    _fig
    return


@app.cell
def _(LM_TRAIN, mo, next_tuned, tokens_lower):
    mo.md(f"""
    ### The picture, after {LM_TRAIN:,} reports

    After one pass over the training-year reports, "fell" gets
    {100 * next_tuned["p_word"]:.0f}% (rank {next_tuned["rank_word"]}), and
    "{next_tuned["top"][0].strip("·")}" leads. A person who has read injury reports guesses like
    the fine-tuned model. Over the attempt's sentence, {tokens_lower} of the 22 tokens are less
    surprising after fine-tuning (the bars above).
    """)
    return


@app.cell
def _(
    DEVICE,
    GENERATE,
    GPU,
    LM_TRAIN,
    REFERENCE_GPU,
    attempt,
    doc_start,
    gpt2_base,
    gpt2_tok,
    gpt2_tuned,
    ref,
    torch,
):
    def greedy_hf(model: object, prefix: str) -> list[int]:
        """Hugging Face's greedy continuation: the new token ids, end-of-text included."""
        enc = gpt2_tok(doc_start(gpt2_tok) + prefix, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=GENERATE, do_sample=False, pad_token_id=gpt2_tok.eos_token_id
            )
        return out[0, enc["input_ids"].shape[1] :].tolist()

    def greedy_ours(model: object, prefix: str) -> list[int]:
        """Greedy from scratch: the most likely token, every step, until end-of-text."""
        ids = gpt2_tok(doc_start(gpt2_tok) + prefix, return_tensors="pt")["input_ids"].to(DEVICE)
        new = []
        with torch.no_grad():
            for _ in range(GENERATE):
                nxt = int(model(input_ids=ids).logits[0, -1].argmax())
                new.append(nxt)
                if nxt == gpt2_tok.eos_token_id:
                    break
                ids = torch.cat([ids, torch.tensor([[nxt]], device=DEVICE)], dim=1)
        return new

    generated = {}
    _tuned_name = f"after {LM_TRAIN:,} reports"
    for _name, _model in (("GPT-2 as released", gpt2_base), (_tuned_name, gpt2_tuned)):
        _ids = greedy_hf(_model, attempt["prefix"])
        assert _ids == greedy_ours(_model, attempt["prefix"]), _name  # the same greedy decoding
        generated[_name] = {
            "text": " ".join(gpt2_tok.decode(_ids, skip_special_tokens=True).split()),
            "stopped": gpt2_tok.eos_token_id in _ids,  # ended the report by itself
        }
        print(f"{_name}: {generated[_name]}")
    _released, _tuned = generated["GPT-2 as released"], generated[_tuned_name]
    if GPU:
        # The table's first row; the notes: the released model writes news copy with a quote
        # from the employee and runs to the 30-token limit
        assert ref("generated/GPT-2 as released") in (None, _released["text"])
        assert not _released["stopped"] and "said the employee" in _released["text"]
    if REFERENCE_GPU:
        # The second row; the notes: the fine-tuned model "stops by itself", with details that
        # are not this report's (the real ladder fell and the employee broke an arm)
        assert _tuned["text"] == ref(f"generated/{_tuned_name}")
        assert _tuned["stopped"] and "femur" in _tuned["text"]
    assert "femur" not in attempt["report"] and "arm" in attempt["report"]
    return (generated,)


@app.cell
def _(attempt, generated, mo):
    _tail = attempt["report"][len(attempt["prefix"]) :].strip()
    _name = next(_k for _k in generated if _k != "GPT-2 as released")
    _stops = "stops by itself" if generated[_name]["stopped"] else "runs to the 30-token limit"
    mo.md(f"""
    ### Fluent is not true

    | Model | Continues "… fourth rung, the ladder" |
    |:------|:------------------------------------|
    | GPT-2 as released | {generated["GPT-2 as released"]["text"]} |
    | {_name.capitalize()} | {generated[_name]["text"]} |
    | The real report | {_tail} |

    Greedy decoding: the likeliest token, every step (Hugging Face's `generate` and our loop give
    the same tokens). The fine-tuned model writes a report and {_stops}, but the details are
    not this report's. A language model is a model of text, not of events: what it writes must
    be checked against a source.
    """)
    return


@app.cell
def _(QA_SYSTEM, QUESTIONS, mo, qwen_tok, ref):
    _msgs = [
        {"role": "system", "content": QA_SYSTEM},
        {"role": "user", "content": QUESTIONS[7].text},
    ]
    chat_document = qwen_tok.apply_chat_template(
        _msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    # The slide shows this document, the system line wrapped; thinking off is the empty block
    assert ref("chat_document") in (None, chat_document)
    assert chat_document.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    _on = qwen_tok.apply_chat_template(
        _msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True
    )
    assert _on.endswith("<|im_start|>assistant\n<think>\n")  # thinking on: the block left open
    _doc = "\n    ".join(chat_document.strip().splitlines())  # mo.md strips a common indent
    mo.md(f"""
    ### A chat is one long document

    ```text
    {_doc}
    ```

    What an instruct model reads: the system's instruction, the user's turn, and a cue for the
    assistant's turn, marked by special tokens. The model predicts the next token, exactly as
    GPT-2 did. Qwen3.5 can "think" first, writing a chain of thought between `<think>` and
    `</think>`; we switch that off (`enable_thinking=False`), and the template writes the block
    empty, so the answer comes next. Supervised fine-tuning on written demonstrations, then
    reinforcement learning from human feedback (Ouyang et al., 2022), teach a model to continue
    such documents helpfully. Everything we do next is text placed in this document: examples,
    excerpts, instructions.
    """)
    return


@app.cell
def _(
    CACHE_TOKENS,
    CODE_BATCH,
    DEVICE,
    EVENTS,
    EVENT_SYSTEM,
    F,
    PREFILL,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    USER_TURN,
    context_length,
    copy,
    np,
    torch,
):
    def event_turn(text: str, order: tuple[str, ...] = tuple(EVENTS)) -> str:
        """The user's turn: the four types (in the given order), the report, and the cue."""
        options = "\n".join(f"{k}: {EVENTS[k]}" for k in order)
        return f"Event types:\n{options}\n\nReport: {text}\nEvent type:"

    def event_prompt(
        tok: PreTrainedTokenizerBase,
        examples: list[tuple[str, str]],
        text: str,
        order: tuple[str, ...] = tuple(EVENTS),
    ) -> str:
        """The chat as one document: system, each example as a question and its answer, then
        the report to code. Thinking is off: the template ends with an empty think block.

        Args:
            tok: Qwen's tokenizer.
            examples: (report, its type) pairs.
            text: The report to code.
            order: The order in which the event types are listed.

        Returns:
            The prompt, ending with the assistant's cue.
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

    def digit_ids(tok: PreTrainedTokenizerBase) -> list[int]:
        """The four event digits' token ids, checked to be read as the answer token.

        Qwen's tokenizer splits digits one per token, so "4" is one token alone, after the
        assistant cue's newline in an example, and after the empty think block's newlines.

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

    def prefill(model: PreTrainedModel, ids: torch.Tensor) -> tuple[object, torch.Tensor]:
        """Read one prompt in pieces of ``PREFILL`` tokens, each continuing the last's cache.

        A prompt this short or shorter is one pass; longer ones are read in pieces so that no
        attention matrix spans the whole prompt (11,000 tokens squared would need gigabytes).

        Args:
            model: Qwen.
            ids: (1, T) token ids.

        Returns:
            The cache after the last token, and that position's logits (V,).
        """
        cache = None
        for i in range(0, ids.shape[1], PREFILL):
            out = model(input_ids=ids[:, i : i + PREFILL], past_key_values=cache,
                        use_cache=True, logits_to_keep=1)  # fmt: skip
            cache = out.past_key_values
        return cache, out.logits[0, -1]

    def digit_logprobs(
        tok: PreTrainedTokenizerBase, model: PreTrainedModel, prompts: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Log-probabilities of the four digits as each prompt's next token, and whether one
        of the four is the likeliest of all tokens.

        The head (system turn and examples, the same for every report) is read once; batches
        of reports' turns run on copies of its cache, padded on the right (a report never sees
        the padding after it), each read at its own last token. A batch holds ``CODE_BATCH``
        reports, fewer when the head is long (``CACHE_TOKENS`` cached tokens at most).

        Args:
            tok: Qwen's tokenizer.
            model: Qwen.
            prompts: Full prompts with the same examples, from ``event_prompt``.

        Returns:
            (n, 4) log-probabilities and (n,) booleans.
        """
        head = prompts[0][: prompts[0].rindex(USER_TURN)]
        assert all(_p.startswith(head) and _p.rindex(USER_TURN) == len(head) for _p in prompts)
        tails = [_p[len(head) :] for _p in prompts]
        ids = digit_ids(tok)
        tok.padding_side = "right"
        head_ids = tok(head, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
        whole = tok(prompts[0], add_special_tokens=False)["input_ids"]  # the split keeps tokens
        assert whole == head_ids[0].tolist() + tok(tails[0], add_special_tokens=False)["input_ids"]
        batch = max(1, min(CODE_BATCH, CACHE_TOKENS // head_ids.shape[1]))
        lp, top = [], []
        with torch.no_grad():
            cache, _ = prefill(model, head_ids)
            for i in range(0, len(tails), batch):
                part = tails[i : i + batch]
                enc = tok(part, return_tensors="pt", padding=True, add_special_tokens=False)
                enc = enc.to(DEVICE)
                assert head_ids.shape[1] + enc["input_ids"].shape[1] <= context_length(model)
                c = copy.deepcopy(cache)  # the cache is updated in place: each batch its own
                c.reorder_cache(torch.zeros(len(part), dtype=torch.long, device=DEVICE))  # rows
                h = model.model(input_ids=enc["input_ids"], past_key_values=c, use_cache=True)
                last = enc["attention_mask"].sum(1) - 1  # each report's own last token
                h = h.last_hidden_state[torch.arange(len(part), device=DEVICE), last]  # (B, H)
                logp = F.log_softmax(model.lm_head(h).float(), dim=-1)  # (B, V)
                lp.append(logp[:, ids].cpu().numpy())
                top.append(np.isin(logp.argmax(-1).cpu().numpy(), ids))
        out = np.concatenate(lp)
        assert out.shape == (len(prompts), len(EVENTS))
        return out, np.concatenate(top)

    def alone_logprobs(
        tok: PreTrainedTokenizerBase, model: PreTrainedModel, prompts: list[str]
    ) -> np.ndarray:
        """The four digits' log-probabilities with each prompt read alone: no batch, no padding,
        no shared cache (one pass up to ``PREFILL`` tokens, pieces beyond).

        Args:
            tok: Qwen's tokenizer.
            model: Qwen.
            prompts: Full prompts.

        Returns:
            (n, 4) log-probabilities.
        """
        ids = digit_ids(tok)
        out = []
        with torch.no_grad():
            for p in prompts:
                enc = tok(p, return_tensors="pt", add_special_tokens=False).to(DEVICE)
                _, logits = prefill(model, enc["input_ids"])
                out.append(F.log_softmax(logits.float(), dim=-1)[ids].cpu().numpy())
        return np.array(out)

    return alone_logprobs, digit_logprobs, event_prompt, prefill


@app.cell
def _(coding_pool, draw, event_prompt, mo, qwen_tok, ref):
    _ex = coding_pool.iloc[draw(coding_pool, 1, 0)]
    _one = list(zip(_ex["text"], _ex["type"], strict=True))[:1]
    example_prompt = event_prompt(qwen_tok, _one, "<the report to code>")
    assert example_prompt.count("<|im_start|>user") == 2 and _one[0][1] == "4"
    assert ref("coding/example_prompt") in (None, example_prompt)
    # The examples' answers are bare ("assistant\n4"); only the last turn gets the empty block
    assert "<|im_start|>assistant\n4<|im_end|>" in example_prompt
    _doc = "\n    ".join(example_prompt.strip().splitlines())  # mo.md strips a common indent
    mo.md(f"""
    ### Examples go in the prompt

    In-context learning (GPT-3, Brown et al., 2020): no weights change; the examples sit in the
    prompt, each as a user turn and the assistant's answer. With $k = 0$ the model has only the
    type names. We read the model's probabilities for the four digits as the next token rather
    than its free text, so every report gets one of the four answers. The same examples train a
    logistic probe on BGE sentence embeddings (Lecture 18's probe, on text), so both methods see
    the same labels. The real prompt with one example (the first example of the first draw); the
    template writes the earlier answers without a think block, and the last turn with an empty
    one:

    ```text
    {_doc}
    ```
    """)
    return


@app.cell
def _(
    CHECK_REPORTS,
    CODING_REPORTS,
    CODING_SEEDS,
    CODING_SHOTS,
    EVENTS,
    QWEN,
    QWEN_REVISION,
    alone_logprobs,
    balanced_accuracy_score,
    coding_pool,
    coding_test,
    confusion_matrix,
    context_length,
    digit_logprobs,
    draw,
    event_prompt,
    load_qwen,
    np,
    time,
):
    qwen_chat_tok, qwen_chat = load_qwen(QWEN, QWEN_REVISION)  # kept for §4's answers and §5
    _test = coding_test.iloc[:CODING_REPORTS]
    _y = _test["type"].to_numpy()
    _labels = np.array(list(EVENTS))
    _t0 = time.perf_counter()
    coding, _checks = {}, []
    for _k in CODING_SHOTS:
        _runs = []
        for _seed in CODING_SEEDS if _k else (0,):
            _ex = coding_pool.iloc[draw(coding_pool, _k, _seed)] if _k else coding_pool.iloc[[]]
            _examples = list(zip(_ex["text"], _ex["type"], strict=True))
            _prompts = [event_prompt(qwen_chat_tok, _examples, _t) for _t in _test["text"]]
            _lp, _top = digit_logprobs(qwen_chat_tok, qwen_chat, _prompts)
            # The cached, batched, padded pass against the first prompts each read alone
            _alone = alone_logprobs(qwen_chat_tok, qwen_chat, _prompts[:CHECK_REPORTS])
            _checks.append((float(np.abs(_alone - _lp[:CHECK_REPORTS]).max()),
                            bool((_alone.argmax(1) == _lp[:CHECK_REPORTS].argmax(1)).all())))  # fmt: skip
            _pred = _labels[_lp.argmax(1)]
            _runs.append(
                {
                    "pred": _pred,
                    "accuracy": float(np.mean(_pred == _y)),
                    "balanced": float(balanced_accuracy_score(_y, _pred)),
                    "digit_on_top": float(_top.mean()),
                    "predicted": {_t: int((_pred == _t).sum()) for _t in EVENTS},
                    "confusion": confusion_matrix(_y, _pred, labels=_labels).tolist(),
                    "prompt_tokens": len(
                        qwen_chat_tok(_prompts[0], add_special_tokens=False)["input_ids"]
                    ),
                }
            )
            print(
                f"  {_k} per type, draw {_seed}: balanced {_runs[-1]['balanced']:.3f}", flush=True
            )
        coding[_k] = _runs
    coding_seconds = time.perf_counter() - _t0
    cache_check = {"max_abs_diff": max(_c[0] for _c in _checks),
                   "same_answers": all(_c[1] for _c in _checks), "prompts": CHECK_REPORTS * len(_checks)}  # fmt: skip
    # "No prompt can hold" every labelled report: all 13,572 as examples, against the context
    _every = list(zip(coding_pool["text"], coding_pool["type"], strict=True))
    all_examples_tokens = len(qwen_chat_tok(event_prompt(qwen_chat_tok, _every, _test["text"].iloc[0]),
                                            add_special_tokens=False)["input_ids"])  # fmt: skip
    context_tokens = context_length(qwen_chat)
    print(
        f"{sum(map(len, coding.values()))} settings x {len(_test):,} reports in "
        f"{coding_seconds / 60:.1f} min; cache check {cache_check}; every example in one prompt: "
        f"{all_examples_tokens:,} tokens, against a context of {context_tokens:,}"
    )
    return (
        all_examples_tokens,
        cache_check,
        coding,
        context_tokens,
        qwen_chat,
        qwen_chat_tok,
    )


@app.cell
def _(
    CODING_REPORTS,
    EVENTS,
    LOGP_TOL,
    REFERENCE_GPU,
    SLIDES,
    all_examples_tokens,
    balanced_accuracy_score,
    cache_check,
    coding,
    coding_test,
    context_tokens,
    near,
    ref,
):
    # The cached, batched, right-padded pass gives each prompt's answer as reading it alone
    assert cache_check["same_answers"] and cache_check["max_abs_diff"] < LOGP_TOL
    # The prompts' lengths do not depend on the device: the first report's prompt per draw
    for _k, _runs in coding.items():
        for _i, _r in enumerate(_runs):
            assert ref(f"coding/{_k}/{_i}/prompt_tokens") in (None, _r["prompt_tokens"]), (_k, _i)
    # "No prompt can hold" all 13,572 labelled reports: as examples, against the context
    assert all_examples_tokens > context_tokens == 262_144  # Qwen3.5's native context
    assert ref("coding/all_examples_tokens") in (None, all_examples_tokens)
    # Answering "falls" every time scores 0.25 balanced, although falls are 44% of the reports
    _y = coding_test["type"].to_numpy()
    assert balanced_accuracy_score(_y, ["4"] * len(_y)) == 0.25
    assert f"{100 * float((_y == '4').mean()):.0f}" == "44"
    assert ref("coding_data/test_counts") in (
        None, {_t: int((coding_test["type"] == _t).sum()) for _t in EVENTS}
    )  # fmt: skip
    # The notes: "the model always answered with one of the four digits"
    assert all(_r["digit_on_top"] == 1.0 for _runs in coding.values() for _r in _runs)
    if REFERENCE_GPU:  # the slides' run, setting by setting
        for _k, _runs in coding.items():
            for _i, _r in enumerate(_runs):
                assert near(_r["balanced"], f"coding/{_k}/{_i}/balanced", 1e-9), (_k, _i)
                assert _r["predicted"] == ref(f"coding/{_k}/{_i}/predicted"), (_k, _i)
    elif SLIDES:
        print("not the slides' GPU: balanced accuracies beside the slides'",
              {_k: [(round(_r["balanced"], 3), round(ref(f"coding/{_k}/{_i}/balanced"), 3))
                    for _i, _r in enumerate(_runs)] for _k, _runs in coding.items()})  # fmt: skip
    else:
        print(f"reduced: the first {CODING_REPORTS} test reports, one draw; balanced accuracy",
              {_k: round(_runs[0]["balanced"], 3) for _k, _runs in coding.items()})  # fmt: skip
    return


@app.cell
def _(AutoModel, AutoTokenizer, BGE, BGE_REVISION, DEVICE, F, np, torch):
    bge_tok = AutoTokenizer.from_pretrained(BGE, revision=BGE_REVISION)
    bge = AutoModel.from_pretrained(BGE, revision=BGE_REVISION).to(DEVICE).eval()

    def embed(
        tok: object, model: object, texts: list[str], prefix: str = "", batch: int = 128
    ) -> np.ndarray:
        """BGE sentence embeddings: the [CLS] vector, L2-normalized (BGE's recipe).

        Args:
            tok: BGE's tokenizer.
            model: BGE.
            texts: The texts.
            prefix: BGE's query prefix for questions; none for passages.
            batch: Texts per pass (the slides' 128; padding depends on it).

        Returns:
            (n, 768) unit vectors.
        """
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

    bge_params = sum(_p.numel() for _p in bge.parameters())
    print(f"BGE base: {bge_params:,} parameters")
    return bge, bge_params, bge_tok, embed


@app.cell
def _(
    CODING_REPORTS,
    LogisticRegression,
    PROBE_SHOTS,
    PROBE_TOL,
    REFERENCE_GPU,
    SHOT_SEEDS,
    SLIDES,
    balanced_accuracy_score,
    bge,
    bge_tok,
    coding_pool,
    coding_test,
    draw,
    embed,
    near,
    np,
):
    # "slides": every training report embedded (the probe on all 13,572); "reduced": only the
    # reports the draws use, and the test reports Qwen codes
    _draws = {_k: [draw(coding_pool, _k, _s) for _s in SHOT_SEEDS] for _k in PROBE_SHOTS}
    _rows = (np.arange(len(coding_pool)) if SLIDES
             else np.unique(np.concatenate([_r for _v in _draws.values() for _r in _v])))  # fmt: skip
    _pool_emb = np.zeros((len(coding_pool), 768), dtype=np.float32)
    _pool_emb[_rows] = embed(bge_tok, bge, coding_pool["text"].iloc[_rows].tolist())
    _test = coding_test.iloc[:CODING_REPORTS]
    _test_emb = embed(bge_tok, bge, _test["text"].tolist())
    _y = _test["type"].to_numpy()
    _y_pool = coding_pool["type"].to_numpy()

    def fit_probe(rows: np.ndarray) -> dict[str, object]:
        """A logistic probe on BGE embeddings of the given training rows, scored on the test.

        C = 1 and balanced class weights throughout (fixed, not tuned: with one example per
        type there is nothing to tune on); L2 penalty (``l1_ratio=0``) and L-BFGS stated.

        Args:
            rows: Positions in the training pool.

        Returns:
            The test predictions, accuracy, and balanced accuracy.
        """
        clf = LogisticRegression(
            C=1.0, l1_ratio=0.0, solver="lbfgs", class_weight="balanced", max_iter=5000
        )
        pred = clf.fit(_pool_emb[rows], _y_pool[rows]).predict(_test_emb)
        return {
            "pred": pred,
            "accuracy": float(np.mean(pred == _y)),
            "balanced": float(balanced_accuracy_score(_y, pred)),
        }

    # Draws with the prompts' seeds: at k = 1, 3, 10, and 30 the very same examples
    probe = {_k: [fit_probe(_r) for _r in _v] for _k, _v in _draws.items()}
    if SLIDES:
        probe["all"] = fit_probe(np.arange(len(coding_pool)))
    for _k, _v in probe.items():
        _b = [_r["balanced"] for _r in (_v if isinstance(_v, list) else [_v])]
        print(f"probe, {_k} per type: balanced {np.round(_b, 3).tolist()}")
    if REFERENCE_GPU:
        # The coding figure and the draws' table: every probe as on the slides
        for _k in PROBE_SHOTS:
            for _i, _r in enumerate(probe[_k]):
                assert near(_r["balanced"], f"probe/{_k}/{_i}/balanced", PROBE_TOL), (_k, _i)
        assert near(probe["all"]["balanced"], "probe/all/balanced", PROBE_TOL)
    return (probe,)


@app.cell
def _(
    CODING_REPORTS,
    CODING_SHOTS,
    MUTED,
    PALETTE,
    PROBE_SHOTS,
    QWEN_NAME,
    coding,
    coding_pool,
    np,
    plt,
    probe,
):
    _fig, _ax = plt.subplots(figsize=(9, 4), layout="constrained")
    _vals = [[_r["balanced"] for _r in probe[_k]] for _k in PROBE_SHOTS]
    _ax.fill_between(PROBE_SHOTS, [min(_v) for _v in _vals], [max(_v) for _v in _vals],
                     color=PALETTE[0], alpha=0.15)  # fmt: skip
    _ax.plot(PROBE_SHOTS, [np.median(_v) for _v in _vals], marker="o", lw=2, color=PALETTE[0],
             label=f"probe on BGE embeddings ({len(probe[1])} draw(s))")  # fmt: skip
    if "all" in probe:
        _ax.plot([400], [probe["all"]["balanced"]], marker="s", ms=9, ls="", color="#1B3A5C",
                 label=f"probe, all {len(coding_pool):,} training reports")  # fmt: skip
    _v = [[_r["balanced"] for _r in coding[_k]] for _k in CODING_SHOTS]
    _med = [float(np.median(_x)) for _x in _v]
    _ax.errorbar([max(_k, 0.5) for _k in CODING_SHOTS], _med,
                 yerr=[np.subtract(_med, [min(_x) for _x in _v]),
                       np.subtract([max(_x) for _x in _v], _med)],
                 marker="D", ms=8, lw=2, capsize=4, color=PALETTE[1], ls="--",
                 label=f"{QWEN_NAME}, examples in the prompt")  # fmt: skip
    _ax.axhline(0.25, ls=":", color=MUTED, lw=1.5, label="chance (also: always one type)")
    _ax.set_xscale("log")
    _ax.set_xticks([0.5, *PROBE_SHOTS, 400], ["0", *map(str, PROBE_SHOTS), "all"])
    _ax.minorticks_off()
    _ax.set_ylim(0, 1)
    _ax.set_xlabel("labelled reports per event type")
    _ax.set_ylabel(f"balanced accuracy, {CODING_REPORTS:,} test reports")
    _ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=2)
    _fig
    return


@app.cell
def _(CODING_SHOTS, REFERENCE_GPU, bge_params, coding, mo, np, probe, qwen_params, ref):
    # "one small embedding per report instead of a pass of a 0.75-billion-parameter model"
    assert bge_params < qwen_params["chat"] / 5 and f"{qwen_params['chat'] / 1e9:.2f}" == "0.75"
    assert ref("bge_parameters") in (None, bge_params)
    # Prompt and probe on the very same examples, draw by draw: probe minus prompt
    _both = [_k for _k in CODING_SHOTS if _k in probe]
    # "reduced" draws once for the prompt and three times for the probe: pair the first ones
    _paired = {_k: [_b["balanced"] - _a["balanced"]
                    for _a, _b in zip(coding[_k], probe[_k], strict=False)]
               for _k in _both}  # fmt: skip
    _pm = {_k: float(np.median([_r["balanced"] for _r in coding[_k]])) for _k in CODING_SHOTS}
    _bm = {
        _k: float(np.median([_r["balanced"] for _r in probe[_k]])) for _k in probe if _k != "all"
    }
    print(
        "probe minus prompt, draw by draw:",
        {_k: np.round(_v, 3).tolist() for _k, _v in _paired.items()},
    )
    if REFERENCE_GPU:
        # The notes: with 1 per type the prompt is ahead in 2 of 3 draws ("close"); from 3 per
        # type on the probe is ahead in every draw, by 0.015 to 0.068
        assert all(np.allclose(_paired[_k], ref(f"coding_summary/paired/{_k}"), atol=1e-9)
                   for _k in _both)  # fmt: skip
        assert sum(_d < 0 for _d in _paired[1]) == 2
        _later = [_d for _k in (3, 10, 30) for _d in _paired[_k]]
        assert min(_later) > 0 and (f"{min(_later):.3f}", f"{max(_later):.3f}") == (
            "0.015",
            "0.068",
        )
    _all = (
        f", and {probe['all']['balanced']:.2f} with all the training reports"
        if "all" in probe
        else ""
    )
    mo.md(f"""
    ### Labels pay more in a probe

    Balanced accuracy is the mean of the four types' recalls, so answering "falls" every time
    scores 0.25. Every method sees the same test reports (2023–2025) and the very same drawn
    examples. Prompt medians: {", ".join(f"{_pm[_k]:.2f} with {_k}" for _k in CODING_SHOTS)}
    examples per type. Probe medians: {", ".join(f"{_bm[_k]:.2f} with {_k}" for _k in _bm)}{_all}.
    Draw by draw, probe minus prompt:
    {"; ".join(f"{_k} per type: " + ", ".join(f"{_d:+.3f}" for _d in _paired[_k]) for _k in _both)}.
    Where the signs are mixed, call the two close; a run of same-signed small gaps says the probe
    is ahead, by that much. The prompt also runs out of room: with every labelled report as an
    example it would not fit the model's context (below), while the probe takes them all, and it
    costs one small embedding per report instead of a pass of a 0.75-billion-parameter model.
    """)
    return


@app.cell
def _(
    BGE_QUERY,
    COS_TOL,
    EXACT,
    GPU,
    QUESTIONS,
    RECALL_TOL,
    TfidfVectorizer,
    bge,
    bge_tok,
    chunk_starts,
    chunk_texts,
    embed,
    near,
    np,
    ref,
    refs,
    secs,
):
    def recall_at(
        s_sections: np.ndarray, targets: np.ndarray, citing: np.ndarray, ks: range
    ) -> list[float]:
        """Share of references whose cited section ranks in the top k (the citing one excluded).

        Args:
            s_sections: (references, sections) scores.
            targets: Each reference's cited section (column).
            citing: Each reference's citing section, excluded from the ranking.
            ks: The k values.

        Returns:
            Recall at each k.
        """
        s = s_sections.copy()
        s[np.arange(len(s)), citing] = -np.inf
        rank = (s > s[np.arange(len(s)), targets][:, None]).sum(1)
        return [float(np.mean(rank < k)) for k in ks]

    doc_emb = embed(bge_tok, bge, chunk_texts)
    _query_emb = embed(bge_tok, bge, [_r[2] for _r in refs], BGE_QUERY)
    q_scores = embed(bge_tok, bge, [_q.text for _q in QUESTIONS], BGE_QUERY) @ doc_emb.T
    assert q_scores.shape == (len(QUESTIONS), len(chunk_texts))
    _pos = {_k: _i for _i, _k in enumerate(secs)}
    _targets = np.array([_pos[_r[1]] for _r in refs])
    _citing = np.array([_pos[_r[0]] for _r in refs])
    # TF-IDF as scikit-learn computes it: sublinear tf, smoothed idf, rows L2-normalized
    _tfidf = TfidfVectorizer(
        stop_words="english", sublinear_tf=True, use_idf=True, smooth_idf=True, norm="l2"
    ).fit(chunk_texts)
    _s_tfidf = (
        _tfidf.transform([_r[2] for _r in refs]) @ _tfidf.transform(chunk_texts).T
    ).toarray()
    _ks = range(1, 21)
    recall = {
        "k": list(_ks),
        "bge": recall_at(np.maximum.reduceat(_query_emb @ doc_emb.T, chunk_starts, axis=1),
                         _targets, _citing, _ks),
        "tfidf": recall_at(np.maximum.reduceat(_s_tfidf, chunk_starts, axis=1), _targets,
                           _citing, _ks),
    }  # fmt: skip
    print({_m: [round(recall[_m][_k - 1], 3) for _k in (1, 5, 20)] for _m in ("bge", "tfidf")})
    # The recall figure and its alt text: TF-IDF runs on the CPU (exact), BGE on the device
    if ref("retrieval") is not None:
        assert np.abs(np.array(recall["tfidf"]) - ref("retrieval/tfidf")).max() < EXACT
        assert np.abs(np.array(recall["bge"]) - ref("retrieval/bge")).max() <= RECALL_TOL
        if GPU:
            assert [f"{_x:.2f}" for _x in recall["bge"]] == [
                f"{_x:.2f}" for _x in ref("retrieval/bge")
            ]
    # The notes: "BGE ... wins most at k = 1": the gap is largest there, and BGE never loses
    _gap = np.subtract(recall["bge"], recall["tfidf"])
    assert int(np.argmax(_gap)) == 0 and (_gap >= 0).all()
    # The notes' example: Section 1C.02 says that Section 1D.11 "contains additional
    # information about crashworthiness"
    assert ("1C.02", "1D.11", "additional information about crashworthiness") in refs
    assert near(float(q_scores[0].max()), "qa/0/scores/0", COS_TOL)
    return doc_emb, q_scores, recall


@app.cell
def _(
    CHAT_TURN_END,
    DECLINE,
    DEVICE,
    DOC_END,
    MAX_ANSWER,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    QA_SYSTEM,
    QUESTIONS,
    Question,
    REPETITION_PENALTY,
    SECTION_ID,
    TOP_K,
    chunk_section,
    chunk_texts,
    np,
    q_scores,
    qwen_chat,
    qwen_chat_tok,
    re,
    secs,
    time,
    torch,
):
    def answer(
        tok: PreTrainedTokenizerBase, model: PreTrainedModel, user: str
    ) -> dict[str, object]:
        """Qwen's greedy answer to one user turn under QA_SYSTEM, every setting passed.

        The pinned revision ships no generation config, so the call names the decoding (greedy,
        no repetition penalty) and both stop tokens: the end of a chat turn, and the end of a
        document. Thinking is off.

        Args:
            tok: Qwen's tokenizer.
            model: Qwen.
            user: The user's turn.

        Returns:
            The answer (whitespace collapsed), its token ids, the prompt's ids, and whether it
            stopped by itself before ``MAX_ANSWER`` tokens.
        """
        msgs = [{"role": "system", "content": QA_SYSTEM}, {"role": "user", "content": user}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, enable_thinking=False).to(DEVICE)  # fmt: skip
        stop = [tok.convert_tokens_to_ids(_t) for _t in (CHAT_TURN_END, DOC_END)]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_ANSWER, do_sample=False,
                                 repetition_penalty=REPETITION_PENALTY, eos_token_id=stop,
                                 pad_token_id=tok.convert_tokens_to_ids(DOC_END))  # fmt: skip
        new = out[0, enc["input_ids"].shape[1] :].tolist()
        return {
            "text": " ".join(tok.decode(new, skip_special_tokens=True).split()),
            "ids": new,
            "prompt": enc["input_ids"],
            "stopped": new[-1] in stop,
        }

    def outcome(q: Question, text: str, gold: list[str], decline: re.Pattern = DECLINE) -> str:
        """The answer's score, by its number and by the section it cites.

        Args:
            q: The question.
            text: The answer.
            gold: The sections that hold the rule.
            decline: The phrasings that count as declining (DECLINE, or DECLINE_PILOT).

        Returns:
            If the manual answers it: "declined"; "wrong" (a number missing); "correct" (every
            number, and a cited section that holds the rule); or "number only" (every number,
            but no section cited or only sections that do not hold the rule). If not:
            "declined" or "invented".
        """
        if decline.search(text):
            return "declined"
        if q.evidence is None:
            return "invented"
        if not all(re.search(_p, text, re.IGNORECASE) for _p in q.answer):
            return "wrong"
        return "correct" if set(SECTION_ID.findall(text)) & set(gold) else "number only"

    _t0 = time.perf_counter()
    qa = []
    for _q, _s in zip(QUESTIONS, q_scores, strict=True):
        _gold = sorted(_k for _k, (_, _b) in secs.items() if _q.evidence and _q.evidence in _b)
        assert _q.evidence is None or _gold, _q.evidence
        _top = np.argsort(-_s)[:TOP_K]
        _context = "\n\n".join(chunk_texts[_j] for _j in _top)
        _closed = answer(qwen_chat_tok, qwen_chat, _q.text)
        _rag = answer(qwen_chat_tok, qwen_chat,
                      f"Excerpts from the MUTCD:\n\n{_context}\n\nQuestion: {_q.text}")  # fmt: skip
        _row = {
            "question": _q.text,
            "answerable": _q.evidence is not None,
            "gold": _gold,
            "retrieved": [str(chunk_section[_j]) for _j in _top],
            "chunks": [int(_j) for _j in _top],
            "scores": [float(_s[_j]) for _j in _top],
            "hit": bool(set(chunk_section[_top]) & set(_gold)),
            "gold_rank": next(
                (_r + 1 for _r, _j in enumerate(_top) if chunk_section[_j] in _gold), None
            ),
            # Retrieval by sentence: the ranks of the excerpts that hold the rule's own words
            "evidence_ranks": [
                _r + 1
                for _r, _j in enumerate(_top)
                if _q.evidence and _q.evidence in chunk_texts[_j]
            ],
        }
        for _mode, _a in (("closed", _closed), ("rag", _rag)):
            _row[_mode] = _a["text"]
            _row[f"{_mode}_outcome"] = outcome(_q, _a["text"], _gold)
            _row[f"{_mode}_cited"] = SECTION_ID.findall(_a["text"])
            _row[f"{_mode}_tokens"] = len(_a["ids"])
            _row[f"{_mode}_stopped"] = _a["stopped"]
        qa.append(_row)
        if len(qa) == 1:
            _first = _rag
        print(f"  [{len(qa)}] {_row['closed_outcome']} / {_row['rag_outcome']}", flush=True)
    qa_seconds = time.perf_counter() - _t0

    # The decoding, from scratch, on the first question's answer with excerpts: read the prompt
    # once, then feed back the likeliest token, one at a time, until a stop token
    _stop = {qwen_chat_tok.convert_tokens_to_ids(_t) for _t in (CHAT_TURN_END, DOC_END)}
    _ours = []
    with torch.no_grad():
        _out = qwen_chat(input_ids=_first["prompt"], use_cache=True, logits_to_keep=1)
        for _ in range(MAX_ANSWER):
            _nxt = int(_out.logits[0, -1].argmax())
            _ours.append(_nxt)
            if _nxt in _stop:
                break
            _out = qwen_chat(input_ids=torch.tensor([[_nxt]], device=DEVICE),
                             past_key_values=_out.past_key_values, use_cache=True)  # fmt: skip
    decoding_check = {"same_tokens": _ours == _first["ids"], "steps": len(_ours)}
    print(f"22 questions twice in {qa_seconds:.0f} s; decoding from scratch: {decoding_check}")
    assert decoding_check["same_tokens"]
    return decoding_check, outcome, qa


@app.cell
def _(QUESTIONS, REFERENCE_GPU, outcome, qa, ref):
    # The scorer, on the slides' own answers: it runs on any device, so it is checked anywhere
    if ref("qa") is not None:
        for _i, (_q, _w) in enumerate(zip(QUESTIONS, ref("qa"), strict=True)):
            for _mode in ("closed", "rag"):
                assert outcome(_q, _w[_mode], _w["gold"]) == _w[f"{_mode}_outcome"], (_i, _mode)
    # This run, question by question, against the slides' (from memory, with five excerpts)
    _ours = [(_r["closed_outcome"], _r["rag_outcome"]) for _r in qa]
    _want = [(_w["closed_outcome"], _w["rag_outcome"]) for _w in ref("qa")] if ref("qa") else _ours
    qa_matches = sum(_a == _b for _a, _b in zip(_ours, _want, strict=True))
    print(f"{qa_matches} of {len(qa)} questions have both outcomes as on the slides")
    if REFERENCE_GPU:  # float32 and greedy: the same answers, word for word, on the slides' GPU
        assert all(_r[_m] == _w[_m] for _r, _w in zip(qa, ref("qa"), strict=True)
                   for _m in ("closed", "rag"))  # fmt: skip
    else:
        print("answers can differ off the slides' GPU (another order of float32 sums can flip a "
              "near-tie); outcomes that differ are listed, not asserted:",
              [(_i, _a, _b) for _i, (_a, _b) in enumerate(zip(_ours, _want, strict=True))
               if _a != _b])  # fmt: skip
    return (qa_matches,)


@app.cell
def _(COS_TOL, GPU, QUESTIONS, REFERENCE_GPU, mo, np, paragraph_label, qa, ref, secs):
    _r = qa[1]  # the slide's question: the height of a warning sign on a freeway
    _want = ref("qa/1")
    if _want is not None:
        assert np.abs(np.array(_r["scores"]) - _want["scores"]).max() < COS_TOL
        if GPU:  # the five excerpts, their order, and the slide's cosines
            assert _r["retrieved"] == _want["retrieved"] and _r["gold_rank"] == _want["gold_rank"]
            assert [f"{_s:.3f}" for _s in _r["scores"]] == [f"{_s:.3f}" for _s in _want["scores"]]
    if REFERENCE_GPU:  # the slide's answer, word for word
        assert _r["rag"] == _want["rag"] and _r["rag_outcome"] == "correct"
    # The notes: "The answer is in Section 2A.15, Paragraph 11 (Standard): 7 feet, in the excerpt
    # at rank 1"
    assert _r["gold"] == ["2A.15"] and _r["gold_rank"] == 1
    assert paragraph_label(secs["2A.15"][1], QUESTIONS[1].evidence) == ("Standard", "11")
    # The slides' answer is the model's own sentence: Paragraph 4's wording (the rural rule, 5
    # feet) with Paragraph 11's 7 feet for freeways; no section holds it
    _wording = (
        "the minimum height, measured vertically from the bottom of the sign to the elevation of "
        "the near edge of the pavement, of signs installed"
    )
    assert paragraph_label(secs["2A.15"][1], _wording) == ("Standard", "04")
    if _want is not None:
        _sentence = _want["rag"].split(": ", 1)[1]
        assert _sentence.startswith(_wording) and "on freeways shall be 7 feet" in _sentence
        assert not any(_sentence.rstrip(".") in _b for _, _b in secs.values())
    _rows = "\n    ".join(  # mo.md strips a common indent: keep it on every row
        f"| {_i + 1} | {_s} | {_c:.3f} |"
        for _i, (_s, _c) in enumerate(zip(_r["retrieved"], _r["scores"], strict=True))
    )
    mo.md(f"""
    ### Retrieve, then read

    **Question:** {_r["question"]}

    | Rank | Section | Cosine |
    |:----:|:--------|-------:|
    {_rows}

    **Answer, from these five:** {_r["rag"]}

    Retrieval-augmented generation (Lewis et al., 2020). The manual is cut into overlapping
    150-word excerpts, each headed by its section number and title; BGE embeds each once. A
    question is embedded the same way (with BGE's query prefix), the five excerpts with the
    largest cosine (Lecture 4's nearest neighbours) are pasted into the prompt above the
    question, and the model answers from them. The answer is in Section 2A.15
    "{secs["2A.15"][0]}", Paragraph 11 (Standard): 7 feet, in the excerpt at rank 1. The
    slides' answer is the model's own sentence, though: Paragraph 4's wording (the rural rule,
    5 feet) with Paragraph 11's 7 feet for freeways.
    """)
    return


@app.cell
def _(MUTED, PALETTE, TOP_K, plt, qa, recall):
    _hits = sum(_r["hit"] for _r in qa if _r["answerable"]) / sum(_r["answerable"] for _r in qa)
    _fig, _ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    _ax.plot(recall["k"], recall["bge"], lw=2, marker="o", ms=4, color=PALETTE[0],
             label="BGE embeddings (cosine)")  # fmt: skip
    _ax.plot(recall["k"], recall["tfidf"], lw=2, marker="s", ms=4, color=MUTED, ls="--",
             label="TF-IDF words (Lecture 4)")  # fmt: skip
    _n = sum(_r["answerable"] for _r in qa)
    _ax.plot([TOP_K], [_hits], marker="*", ms=16, ls="", color=PALETTE[1],
             label=f"our {_n} questions, BGE, top {TOP_K} excerpts")  # fmt: skip
    _ax.set_xlabel("sections returned (k)")
    _ax.set_ylabel("cited section among them")
    _ax.set_ylim(0, 1.02)
    _ax.set_xticks([1, 5, 10, 15, 20])
    _ax.legend(frameon=False, fontsize=9, loc="lower right")
    _fig
    return


@app.cell
def _(GPU, mo, qa, recall, ref, refs):
    _ans = [_r for _r in qa if _r["answerable"]]
    qa_hits = sum(_r["hit"] for _r in _ans)
    if GPU:  # the star: the right section among the five excerpts (BGE, on the device)
        assert ref("qa_hits") in (None, qa_hits)
        _sent = sum(bool(_r["evidence_ranks"]) for _r in _ans)
        assert ref("qa_evidence_hits") in (None, _sent)
    # The star counts the right section among the five excerpts given to the model (they come
    # from at most five sections); the curves count sections. On these questions the five
    # excerpts covered:
    _distinct = [len(set(_r["retrieved"])) for _r in _ans]
    print(f"distinct sections among each question's five excerpts: {_distinct}")
    mo.md(f"""
    ### The manual labels its own test set

    Labels for free: the manual points to itself. Section 1C.02 says that Section 1D.11 "contains
    additional information about crashworthiness". Each such description is a query; the cited
    section is the right answer; the citing section is excluded: {len(refs)} references to
    {len({_r[1] for _r in refs})} sections. TF-IDF matches words; BGE matches meaning and wins
    most at $k = 1$ ({recall["bge"][0]:.2f} against {recall["tfidf"][0]:.2f}). At $k = 5$,
    {recall["bge"][4]:.2f} against {recall["tfidf"][4]:.2f}. For our {len(_ans)} questions the
    right section was among the five excerpts for {qa_hits}, and the sentence that holds the rule
    for {sum(bool(_r["evidence_ranks"]) for _r in _ans)}. An excerpt not retrieved cannot be
    read: retrieval sets the ceiling for the answers, counted here by section; §5 counts it by
    the sentence too. (The star counts excerpts, which came from
    {min(_distinct)} to {max(_distinct)} distinct sections per question; the curves count
    sections.)
    """)
    return (qa_hits,)


@app.cell
def _(mo):
    my_question = mo.ui.text(
        value="How high above the pavement must a traffic signal be mounted?",
        label="Ask the manual (retrieval only: the five excerpts, no generated answer)",
        full_width=True,
    )
    my_question
    return (my_question,)


@app.cell
def _(
    BGE_QUERY,
    TOP_K,
    bge,
    bge_tok,
    chunk_section,
    chunk_texts,
    doc_emb,
    embed,
    mo,
    my_question,
    np,
    secs,
):
    _s = (embed(bge_tok, bge, [my_question.value], BGE_QUERY) @ doc_emb.T)[0]
    _top = np.argsort(-_s)[:TOP_K]
    _rows = "\n    ".join(
        f"| {_i + 1} | {chunk_section[_j]} {secs[chunk_section[_j]][0]} | {_s[_j]:.3f} | "
        f"{chunk_texts[_j].split(': ', 1)[1][:160].replace('|', '/')} … |"
        for _i, _j in enumerate(_top)
    )
    mo.md(f"""
    ### Your turn: ask the manual

    | Rank | Section | Cosine | Excerpt (first words) |
    |:----:|:--------|-------:|:----------------------|
    {_rows}

    Change the question above. Read the excerpts yourself: is the answer there? This is option C
    of the vote in §6.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Break it

    ### Your turn: predict four failures

    1. Code the reports with no examples in the prompt.
    2. Draw another set of examples.
    3. Ask the 22 MUTCD questions from memory, 6 of them about things the manual does not cover.
    4. Answer from the five excerpts when the manual has several similar rules.

    Write what happens to each, then run the reveals.
    """)
    return


@app.cell
def _(
    CODING_REPORTS,
    EVENTS,
    MUTED,
    PALETTE,
    REFERENCE_GPU,
    SHOW_K,
    coding,
    coding_test,
    confusion_matrix,
    np,
    plt,
    ref,
):
    _test = coding_test.iloc[:CODING_REPORTS]
    _y = _test["type"].to_numpy()
    _codes = list(EVENTS)
    _cm = confusion_matrix(_y, coding[0][0]["pred"], labels=_codes)  # rows: the reports' codes
    zero_shot = {
        "confusion": _cm,
        "answered": dict(zip(_codes, _cm.sum(axis=0).tolist(), strict=True)),
        "contact_to": dict(zip(_codes, _cm[3].tolist(), strict=True)),  # contact reports' answers
        "transport_to": dict(zip(_codes, _cm[0].tolist(), strict=True)),
    }
    print("rows: the reports' codes 2, 4, 5, 6; columns: Qwen's answers with no examples")
    print(_cm)
    if REFERENCE_GPU:
        # The notes: no transportation or contact answers; of 380 contact reports, 294 under
        # exposure and 86 under falls; of 65 transportation reports, 49 under exposure
        assert _cm.tolist() == ref("coding/0/0/confusion")
        assert zero_shot["answered"]["2"] == zero_shot["answered"]["6"] == 0
    _true = [int((_y == _t).sum()) for _t in _codes]
    _fig, _ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    _x = np.arange(len(_codes))
    _series = (
        ("the reports' own codes", _true, MUTED, ""),
        ("no examples", list(coding[0][0]["predicted"].values()), PALETTE[1], "//"),
        (f"{SHOW_K} per type (first draw)", list(coding[SHOW_K][0]["predicted"].values()),
         PALETTE[0], ".."),
    )  # fmt: skip
    for _j, (_name, _vals, _col, _hatch) in enumerate(_series):
        _bars = _ax.bar(_x + (_j - 1) * 0.27, _vals, 0.27, color=_col, hatch=_hatch, label=_name)
        _ax.bar_label(_bars, fontsize=8, padding=1)
    _ax.set_xticks(_x, [f"{_k}: {_v}"[:28] for _k, _v in EVENTS.items()], fontsize=9)
    _ax.set_ylabel(f"of {CODING_REPORTS:,} test reports")
    _ax.legend(frameon=False, fontsize=9)
    _fig
    return (zero_shot,)


@app.cell
def _(SHOW_K, coding, mo, zero_shot):
    _p0, _p3 = coding[0][0]["predicted"], coding[SHOW_K][0]["predicted"]
    _n = {_t: sum(_row) for _t, _row in zip("2456", zero_shot["confusion"].tolist(), strict=True)}
    mo.md(f"""
    ### Reveal 1: no examples, two types vanish

    With only the type names, Qwen3.5-0.8B answers transportation {_p0["2"]} times (for
    {_n["2"]} such reports) and contact {_p0["6"]} times (for {_n["6"]}). Of the contact
    reports it put {zero_shot["contact_to"]["5"]} under exposure and
    {zero_shot["contact_to"]["4"]} under falls; of the transportation reports,
    {zero_shot["transport_to"]["5"]} under exposure. OSHA's coders follow the OIICS manual's
    rules; the model only has the names. {SHOW_K} examples per type bring contact back
    ({_p3["6"]} answers in the first draw), while transportation stays short ({_p3["2"]}).

    Is it the list? Language models are known to favour some answer labels and positions over
    others (Zhao et al., 2021). The next cell tests it: the same zero-shot prompt with the four
    types listed in each of four orders.
    """)
    return


@app.cell
def _(
    CODING_REPORTS,
    EVENTS,
    REFERENCE_GPU,
    coding,
    coding_test,
    digit_logprobs,
    event_prompt,
    np,
    qwen_chat,
    qwen_chat_tok,
    ref,
):
    _test = coding_test.iloc[:CODING_REPORTS]
    _codes = list(EVENTS)
    _labels = np.array(_codes)
    # Each type listed first once, second once, and so on (a Latin square); the digits stay
    # with their names. Rotation 0 is the slides' order.
    reorder = {}
    for _r in range(len(_codes)):
        _order = tuple(_codes[_r:] + _codes[:_r])
        _prompts = [event_prompt(qwen_chat_tok, [], _t, _order) for _t in _test["text"]]
        _lp, _ = digit_logprobs(qwen_chat_tok, qwen_chat, _prompts)
        _pred = _labels[_lp.argmax(1)]
        reorder[_order] = {_t: int((_pred == _t).sum()) for _t in _codes}
        print(f"  listed {', '.join(_order)}: answers {reorder[_order]}", flush=True)
    # The slides' order gives the same prompts as §4's zero-shot run, so the same answers
    assert reorder[tuple(_codes)] == coding[0][0]["predicted"]
    # Answers by the position in which their type was listed (rows: the four orders)
    by_position = np.array([[_c[_o[_p]] for _p in range(4)] for _o, _c in reorder.items()])
    print(f"answers by listed position (rows: the four orders):\n{by_position}")
    if REFERENCE_GPU:  # the slides' run, order by order
        assert {"".join(_o): _c for _o, _c in reorder.items()} == ref("list_order")
    return by_position, reorder


@app.cell
def _(CODING_REPORTS, EVENTS, PALETTE, by_position, coding_test, mo, np, plt, reorder):
    _codes = list(EVENTS)
    _fig, _ax = plt.subplots(figsize=(9, 3.4), layout="constrained")
    _m = np.array([[_c[_t] for _t in _codes] for _c in reorder.values()])
    _ax.imshow(_m, cmap="Blues", vmin=0, vmax=CODING_REPORTS)
    for _i, _o in enumerate(reorder):
        for _j, _t in enumerate(_codes):
            _place = ("first", "second", "third", "last")[_o.index(_t)]
            _ax.text(_j, _i, f"{_m[_i, _j]} ({_place})", ha="center", va="center", fontsize=10,
                     color="white" if _m[_i, _j] > CODING_REPORTS / 2 else "black")  # fmt: skip
    _ax.set_xticks(range(4), [f"answered {_t}" for _t in _codes])
    _ax.set_yticks(range(4), [f"listed {', '.join(_o)}" for _o in reorder])
    _ax.tick_params(length=0)
    _ax.set_title(f"Qwen's zero-shot answers, {CODING_REPORTS:,} reports, four orders of the list",
                  fontsize=10, color=PALETTE[0])  # fmt: skip
    _pos = by_position.sum(axis=0) / by_position.sum()
    _true = {_t: int((coding_test["type"].iloc[:CODING_REPORTS] == _t).sum()) for _t in _codes}
    _shares = ", ".join(f"{_t}: {100 * _true[_t] / CODING_REPORTS:.0f}%" for _t in _codes)
    _by_place = {_t: [_c[_t] for _o, _c in sorted(reorder.items(), key=lambda _kv: _kv[0].index(_t))]
                 for _t in _codes}  # fmt: skip
    mo.vstack([
        _fig,
        mo.md(f"""
    ### The answers follow the position

    Over the four orders, {100 * _pos[0]:.0f}% of the answers go to the type listed first,
    {100 * _pos[1]:.0f}% to the second, {100 * _pos[2]:.0f}% to the third, and
    {100 * _pos[3]:.0f}% to the last (each type's share of the reports: {_shares}). Contact,
    listed last in the slides' order, gets {", ".join(map(str, _by_place["6"]))} answers listed
    first, second, third, and last; transportation {", ".join(map(str, _by_place["2"]))}. So with
    no examples the order of a list in the prompt is part of the classifier
    ({"here the middle of the list wins" if _pos[1] + _pos[2] > 0.5 else "here the ends win"}).
    Another reason to choose prompts on validation reports, and to report the spread.
    """),
    ])  # fmt: skip
    return


@app.cell
def _(CODING_SHOTS, REFERENCE_GPU, coding, mo, probe, ref):
    _rows = [(f"Prompt, {_k} per type", coding[_k]) for _k in CODING_SHOTS if _k] + [
        (f"Probe, the same {_k} per type", probe[_k]) for _k in (1, 3)
    ]
    _width = max(len(_rs) for _, _rs in _rows)
    _table = "\n    ".join(
        f"| {_n} | "
        + " | ".join(f"{_r['balanced']:.2f}" for _r in _rs)
        + " |" * (_width - len(_rs) + 1)
        for _n, _rs in _rows
    )
    _range = {_n: (min(_r["balanced"] for _r in _rs), max(_r["balanced"] for _r in _rs))
              for _n, _rs in _rows}  # fmt: skip
    if REFERENCE_GPU:
        # The slide's table: 1 and 3 per type, prompt and probe, three draws each
        for _k in (1, 3):
            for _name, _src in (("coding", coding), ("probe", probe)):
                _want = [f"{_r['balanced']:.2f}" for _r in ref(f"{_name}/{_k}")]
                assert [f"{_r['balanced']:.2f}" for _r in _src[_k]] == _want, (_name, _k)
    _header = " | ".join(f"Draw {_i + 1}" for _i in range(_width))
    mo.md(f"""
    ### Reveal 2: other examples, another score

    | Balanced accuracy | {_header} |
    |:------------------|{"-------:|" * _width}
    {_table}

    Same test reports, same model; only the drawn examples differ. With one example per type the
    prompt ranges over {_range["Prompt, 1 per type"][0]:.2f}–{_range["Prompt, 1 per type"][1]:.2f}
    and the probe over {_range["Probe, the same 1 per type"][0]:.2f}–{_range["Probe, the same 1 per type"][1]:.2f}:
    with one label per type, which label you happened to pick moves the probe's score most.
    Which reports are the examples, and their order, is a hyperparameter: choose them on
    validation reports, never on the test, and report the spread.
    """)
    return


@app.cell
def _(
    DECLINE_PILOT, QA_SYSTEM, QUESTIONS, REFERENCE_GPU, SECTION_ID, mo, outcome, qa, re, ref, secs
):
    def added_numbers(text: str, question: str) -> list[str]:
        """Numbers in an answer that neither the question nor the instruction gave it.

        Section numbers of the manual's form (2A.16) are left out; invented ones (11.1.1) count.

        Args:
            text: The answer.
            question: The question.

        Returns:
            The numbers, sorted.
        """
        _num = re.compile(r"\d+(?:[./]\d+)*")
        given = set(_num.findall(question)) | set(_num.findall(QA_SYSTEM))
        return sorted(set(_num.findall(SECTION_ID.sub(" ", text))) - given)

    _ans = [_r for _r in qa if _r["answerable"]]
    _not = [_r for _r in qa if not _r["answerable"]]
    _count = {_o: sum(_r["closed_outcome"] == _o for _r in _ans)
              for _o in ("correct", "number only", "declined", "wrong")}  # fmt: skip
    _count_not = {
        _o: sum(_r["closed_outcome"] == _o for _r in _not) for _o in ("declined", "invented")
    }
    # Declines that still state a number of their own ("not in the MUTCD", then a rule)
    _declines = [_r for _r in qa if _r["closed_outcome"] == "declined"]
    _with_number = [_r for _r in _declines if added_numbers(_r["closed"], _r["question"])]
    # Section numbers cited in a form the manual does not use (it numbers them like 2A.15)
    _odd = [_m for _r in qa for _m in re.findall(r"Section (\d+(?:\.\d+){2,})", _r["closed"])]
    # The same answers under the list of declining phrasings fixed before the pilot run
    _pilot = [
        outcome(_q, _r["closed"], _r["gold"], DECLINE_PILOT)
        for _q, _r in zip(QUESTIONS, qa, strict=True)
    ]
    _pilot_not = {
        _o: sum(_p == _o for _p, _r in zip(_pilot, qa, strict=True) if not _r["answerable"])
        for _o in ("declined", "invented")
    }
    _pilot_declines = [_r for _p, _r in zip(_pilot, qa, strict=True) if _p == "declined"]
    _pilot_number = [_r for _r in _pilot_declines if added_numbers(_r["closed"], _r["question"])]
    print(f"from memory: {_count}; not in the manual: {_count_not} ({_pilot_not} by the list "
          f"fixed before the pilot); declines with a number: {len(_with_number)} of "
          f"{len(_declines)} ({len(_pilot_number)} of {len(_pilot_declines)}); sections cited: "
          f"{_odd}")  # fmt: skip
    _rows = (1, 18, 0)  # the slide's three rows: wrong; declined but invented; declined, a rule
    if REFERENCE_GPU:
        # The notes, from the slides' tally: none right, 5 wrong (two only "1." and ones), 11
        # declined; all 6 not covered declined (5 and 1 invented by the list fixed before the
        # pilot); 9 of 17 declines with a number of their own (9 of 16); three cite 11.1.1 or
        # 10.1.1.1
        _t, _tp = ref("tally"), ref("tally_pilot_declines")
        assert _count == _t["closed_answerable"] and _count_not == _t["closed_unanswerable"]
        assert _pilot_not == _tp["closed_unanswerable"]
        for _n, _ds, _tt in ((_with_number, _declines, _t), (_pilot_number, _pilot_declines, _tp)):
            _kinds = ("answerable", "unanswerable")
            assert len(_ds) == sum(_tt[f"closed_{_k}"]["declined"] for _k in _kinds)
            assert len(_n) == sum(
                _tt[f"closed_{_k}_extra"]["declined_with_numbers"] for _k in _kinds
            )
        assert sorted(_odd) == ref("qa_odd_sections")
        assert not any(SECTION_ID.fullmatch(_c) or _c in secs for _c in _odd)
        _ones = [_r["closed"] for _r in _ans if _r["closed_outcome"] == "wrong"
                 and not re.sub(r"[1.\s]", "", _r["closed"])]  # fmt: skip
        assert len(_ones) == 2
        # The rows: 12 inches where the manual says 7 feet (2A.15); 1/200 of the span, then
        # "Not in the MUTCD"; a 10-foot offset where the manual says 12 feet (2A.16)
        assert "12 inches" in qa[1]["closed"] and qa[1]["gold"] == ["2A.15"]
        assert "1/200" in qa[18]["closed"] and qa[18]["closed_outcome"] == "declined"
        assert "10 feet" in qa[0]["closed"] and qa[0]["closed_outcome"] == "declined"
        assert "12 feet" in secs["2A.16"][1] and qa[0]["gold"] == ["2A.16"]
    _table = "\n    ".join(f"| {qa[_i]['question']} | {qa[_i]['closed']} |" for _i in _rows)
    mo.md(f"""
    ### Reveal 3: from memory, it invents numbers

    | Asked from memory | Qwen3.5-0.8B answered |
    |:------------------|:----------------------|
    {_table}

    From memory the model answered {_count["correct"]} of the {len(_ans)} answerable questions
    right, got {_count["wrong"]} wrong, and declined {_count["declined"]}. Of the {len(_not)}
    questions the MUTCD does not cover (concrete, asphalt, girders: other manuals' business) it
    declined {_count_not["declined"]}. That count uses the list of declining phrasings set after
    a pilot run of this model ("does not define" counts); by the list fixed before it,
    {_pilot_not["declined"]} declined and {_pilot_not["invented"]} invented. Either way,
    {len(_with_number)} of its {len(_declines)} declines ({len(_pilot_number)} of
    {len(_pilot_declines)} by the earlier list) still state a number of their own: "not in the
    MUTCD", and then a rule it made up. Section
    numbers it cited: {", ".join(_odd) or "none"}; the manual numbers its sections like 2A.15.
    Nothing in a made-up number signals that it is made up. The right answer to the first row is
    7 feet (Section 2A.15); to the second, "not in the MUTCD"; to the third, 12 feet (2A.16).
    """)
    return


@app.cell
def _(QUESTIONS, REFERENCE_GPU, chunk_texts, mo, paragraph_label, qa, ref, secs):
    _walk, _stop, _setback, _yield, _cross = qa[4], qa[14], qa[0], qa[12], qa[13]
    # The manual: Section 4I.06 gives 3.5 ft/s as Guidance; 3 ft/s is for the walk interval and
    # the clearance together; the extended push button belongs to an Option of up to 4 ft/s
    _b = secs["4I.06"][1]
    assert paragraph_label(_b, "walking speed of 3.5 feet per second")[0] == "Guidance"
    assert paragraph_label(_b, "walking speed of 3 feet per second")[0] == "Guidance"
    assert "total of the walk interval and pedestrian clearance time" in _b
    assert paragraph_label(_b, "A walking speed of up to 4 feet per second")[0] == "Option"
    assert "extended push button press function" in _b
    # The stop line: 4 feet (3B.19, Paragraph 13); 20 to 50 feet is its rule for yield and stop
    # lines at an uncontrolled multi-lane approach (Paragraph 15)
    _b3 = secs["3B.19"][1]
    assert paragraph_label(_b3, QUESTIONS[14].evidence)[1] == "13"
    assert paragraph_label(_b3, "20 to 50 feet in advance of the nearest crosswalk line")[1] == "15"
    assert "uncontrolled multi-lane approach" in _b3
    # The setback's "12 feet" sentence is about expressway medians, in Section 2E.20
    _median = "Where an expressway median is 12 feet or less in width"
    assert [_k for _k, (_, _t) in secs.items() if _median in _t] == ["2E.20"]
    # Yield Here To Pedestrians: 2B.19's sentence is about the signs; the answer quotes 3B.19's
    # yield-line rule. The Crossbuck strip: 8B.04's rule for the support; 8B.03 not retrieved
    assert QUESTIONS[12].evidence in secs["2B.19"][1] and QUESTIONS[12].evidence not in _b3
    assert "shall be used on each Crossbuck support" in secs["8B.04"][1]
    # The stop line's rule (3B.19, Paragraph 13) is in none of the five excerpts: the excerpt of
    # 3B.19 that was retrieved begins after it, so the model never saw the 4 feet
    if _stop["retrieved"][0] == "3B.19":
        _head = f"Section 3B.19 {secs['3B.19'][0]}: "  # each excerpt's heading
        _words = chunk_texts[_stop["chunks"][0]].removeprefix(_head)
        assert _b3.index(_words) > _b3.index(QUESTIONS[14].evidence)
    _rank = {
        _n: ", ".join(map(str, _r["evidence_ranks"])) or "not retrieved"
        for _n, _r in (("walk", _walk), ("stop", _stop), ("setback", _setback))
    }
    if REFERENCE_GPU:
        # The slide's three rows: a right section among the excerpts (ranks 1, 1, 2) and the
        # rule's own sentence at rank 4, nowhere, and rank 4; the answers
        for _i in (4, 14, 0):
            assert qa[_i]["gold_rank"] == ref(f"qa/{_i}/gold_rank")
            assert qa[_i]["evidence_ranks"] == ref(f"qa/{_i}/evidence_ranks")
        assert _stop["evidence_ranks"] == [] and _stop["retrieved"][0] == "3B.19"
        assert _walk["rag_outcome"] == _stop["rag_outcome"] == "wrong"
        assert "3 feet per second" in _walk["rag"] and "extended push button" in _walk["rag"]
        assert "20 to 50 feet" in _stop["rag"] and "3B.19" in _stop["rag"]
        assert _median in _setback["rag"] and _setback["retrieved"][0] == "2E.20"
        # The two more "number only" answers from another rule
        assert _yield["rag_outcome"] == _cross["rag_outcome"] == "number only"
        assert "yield (stop) line should be placed 20 to 50 feet" in _yield["rag"]
        assert not _cross["hit"] and "Crossbuck support" in _cross["rag"]
    mo.md(f"""
    ### Reveal 4: right section, wrong sentence

    | Question | Right section's rank | Rule's sentence, rank | Answer with five excerpts | The manual |
    |:---------|:--------------------:|:---------------------:|:--------------------------|:-----------|
    | Walking speed for pedestrian clearance? | {_walk["gold_rank"]} | {_rank["walk"]} | {_walk["rag"]} | 3.5 ft/s (Guidance); 3 ft/s is for walk plus clearance |
    | Stop line before a crosswalk? | {_stop["gold_rank"]} | {_rank["stop"]} | {_stop["rag"]} | 4 feet; 20 to 50 is for an uncontrolled multi-lane approach |
    | Post-mounted sign's offset? | {_setback["gold_rank"]} | {_rank["setback"]} | {_setback["rag"]} | 12 feet, in another rule (Section 2A.16) |

    Each had a right section among its excerpts, and answered from the wrong sentence. The
    MUTCD marks each paragraph Standard (shall), Guidance (should), Option (may), or Support,
    and keeps several similar rules in one section. Section 4I.06 gives 3.5 ft/s as Guidance;
    its 3 ft/s is for the walk interval and the clearance together, and the push button belongs
    to an Option of up to 4 ft/s. For the stop line, the 4-foot sentence (3B.19, Paragraph 13)
    was in none of the five excerpts: the retrieved excerpt of 3B.19 begins after it, so the
    model read that section's rule for yield lines at uncontrolled multi-lane crossings. That
    one is a retrieval miss inside the right section, not a misreading. The setback's "12 feet"
    comes from a sentence about expressway medians in Section 2E.20: the right number from the
    wrong rule, which a number test scores as right. Two more did the same: the Yield Here To
    Pedestrians distance came from the yield-line rule, and the Crossbuck strip from Section
    8B.04's rule for the support (8B.03, the blade's, was not retrieved).
    """)
    return


@app.cell
def _(MUTED, PALETTE, QUESTIONS, REFERENCE_GPU, TOP_K, mo, plt, qa, ref):
    _kinds = {
        "answerable": ("correct", "number only", "declined", "wrong"),
        "unanswerable": ("declined", "invented"),
    }

    def tally(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
        """Counts of each outcome, from memory and with excerpts, answerable or not.

        Args:
            rows: The answers, from the question-answering cell.

        Returns:
            For each mode and kind of question, the count of each outcome.
        """
        out = {}
        for mode in ("closed", "rag"):
            for kind, kinds in _kinds.items():
                sub = [r for r in rows if r["answerable"] is (kind == "answerable")]
                out[f"{mode}_{kind}"] = {
                    k: sum(r[f"{mode}_outcome"] == k for r in sub) for k in kinds
                }
        return out

    def quotes_rule(question: object, text: str) -> bool:
        """Whether the answer holds the words of the rule itself (case, spacing, stars aside).

        Args:
            question: The question, with its ``evidence``.
            text: The answer.

        Returns:
            True if the rule's words are in the answer.
        """
        if question.evidence is None:
            return False
        return " ".join(question.evidence.lower().split()) in " ".join(
            text.replace("*", "").lower().split()
        )

    scorecard = tally(qa)
    print(scorecard)
    _ans = [_r for _r in qa if _r["answerable"]]
    _short = [_r for _r in _ans if _r["rag_outcome"] != "correct"]  # fell short with excerpts
    _number_only = [(_q, _r) for _q, _r in zip(QUESTIONS, qa, strict=True)
                    if _r["rag_outcome"] == "number only"]  # fmt: skip
    _copies = sum(quotes_rule(_q, _r["rag"]) for _q, _r in _number_only)
    # Retrieval by section and by sentence: a right section among the five excerpts, and the
    # rule's own words among them, for every question and for the answers that fell short
    _by = {
        "section": (sum(_r["hit"] for _r in _ans), sum(_r["hit"] for _r in _short)),
        "sentence": (
            sum(bool(_r["evidence_ranks"]) for _r in _ans),
            sum(bool(_r["evidence_ranks"]) for _r in _short),
        ),
    }
    print(f"retrieval of the {len(_ans)} questions and the {len(_short)} short answers: {_by}")
    _reader = _by["sentence"][1] > len(_short) - _by["sentence"][1]  # more misreadings than misses
    if REFERENCE_GPU:
        # The alt text, the title, and the notes, from the slides' tally: with five excerpts,
        # the number and the section, the number only, declined, wrong; from memory; the right
        # numbers that copy the rule's words; retrieval by section and by sentence
        assert scorecard == {_k: ref(f"tally/{_k}") for _k in scorecard}
        assert _copies == ref("tally/rag_answerable_extra/number_only_quotes_rule")
        assert len(_short) == ref("qa_short/answers")
        assert _by["section"] == (ref("qa_hits"), ref("qa_short/with_section"))
        assert _by["sentence"] == (ref("qa_evidence_hits"), ref("qa_short/with_evidence"))
    _colors = {"correct": PALETTE[2], "number only": PALETTE[3], "declined": "#bfbfbf",
               "wrong": PALETTE[1], "invented": PALETTE[1]}  # fmt: skip
    _hatch = {"correct": "", "number only": "..", "declined": "", "wrong": "//", "invented": "//"}
    _sizes = {_k: sum(scorecard[f"closed_{_k}"].values()) for _k in _kinds}
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True, layout="constrained",
                               gridspec_kw={"width_ratios": [_sizes["answerable"],
                                                             _sizes["unanswerable"] + 2]})  # fmt: skip
    for _ax, _kind in zip(_axes, _kinds, strict=True):
        for _y, _mode in enumerate(("rag", "closed")):
            _left = 0
            for _k, _n in scorecard[f"{_mode}_{_kind}"].items():
                if _n:
                    _ax.barh(_y, _n, left=_left, color=_colors[_k], hatch=_hatch[_k],
                             edgecolor="white", label=_k if _y == 0 else None)  # fmt: skip
                    _ax.text(_left + _n / 2, _y, str(_n), ha="center", va="center", fontsize=10)
                _left += _n
        _ax.set_yticks([0, 1], [f"with {TOP_K} excerpts", "from memory"])
        _ax.set_xlim(0, _sizes[_kind])
        _ax.set_xticks([])
        _ax.set_title(f"{_sizes[_kind]} questions the manual " + ("answers" if _kind == "answerable"
                      else "does not answer"), fontsize=11)  # fmt: skip
        for _side in ("top", "right", "bottom"):
            _ax.spines[_side].set_visible(False)
        _ax.spines["left"].set_color(MUTED)
    _axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0))
    _ra = scorecard["rag_answerable"]
    mo.vstack([
        mo.md(f"""
    ## 6. Decide

    ### Excerpts: {_ra["correct"] + _ra["number only"]} numbers right, {_ra["correct"]} cited
    """),
        _fig,
        mo.md(f"""
    The same model and instruction ("answer with the number and the section; if the MUTCD does
    not cover it, say so"); only the excerpts differ. A number test alone counts
    {_ra["correct"] + _ra["number only"]} of {len(_ans)} right with excerpts; asking for a section
    that holds the rule leaves {_ra["correct"]}. Of the {len(_number_only)} with the number but not
    the section, {_copies} copy the rule's own words without naming its section, and the others
    took the number from another rule (Reveal 4). Retrieval put a right section among the five
    for {_by["section"][0]} of {len(_ans)} questions, and the rule's own sentence for
    {_by["sentence"][0]}; {_by["sentence"][1]} of the {len(_short)} answers that fell short had
    that sentence among their excerpts, so the {"reader" if _reader else "retrieval"} failed
    more often than the {"retrieval" if _reader else "reader"}.
    Bracket: from memory, {scorecard["closed_answerable"]["correct"]} of {len(_ans)}; retrieval's
    ceiling, {_by["section"][0]} of {len(_ans)} by section ({_by["sentence"][0]} with the rule's
    sentence). The prompt, the questions, and the excerpts were fixed before any answer was
    scored; the scorer's section rule and its ways to decline were set after a pilot run of this
    model (Reveal 3 gives the counts both ways). 22 questions are few: read the counts, not
    percentages.
    """),
    ])  # fmt: skip
    return (scorecard,)


@app.cell
def _(REPETITION_PENALTY, decoding_check, mo, scorecard, zero_shot):
    _ra = scorecard["rag_answerable"]
    _n6 = int(zero_shot["confusion"][3].sum())  # contact reports in the coded test set
    mo.md(f"""
    ### Who signs the answer?

    | Do not use it alone when | Because, today |
    |:-------------------------|:---------------|
    | A crew sets a height, offset, or time from it | with excerpts, {_ra["wrong"]} of 16 wrong and {_ra["number only"]} without the section that holds the rule |
    | A section holds several similar rules | it read the stop line's distance from the yield-line rule |
    | The state has a supplement | the excerpts are the national manual only |
    | Its codes feed a hazard trend | with no examples, {zero_shot["answered"]["6"]} contact answers for {_n6} contact reports |

    Wherever an answer becomes a dimension in the field, a person reads the source. States adopt
    the national MUTCD, some with a supplement or their own edition; a retriever over the national
    text cannot know the state's rule. Accountability: a sign set at the wrong height is the
    agency's liability and the engineer's responsibility, not the vendor's; "the assistant said
    so" is not a citation (ABET's professional responsibility, SO 4). Hazard trends built on
    machine-coded reports inherit the coder's biases, here types that vanish.

    A note on the decoding: the answers are plain greedy (repetition penalty
    {REPETITION_PENALTY}, named in the call: the pinned release ships no generation settings), and
    our step-by-step decoding reproduces Hugging Face's ({decoding_check["steps"]} tokens of the
    first answer with excerpts).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Vote: what should the crews get?

    - **A.** A chat model that answers from memory.
    - **B.** Retrieval plus the model: an answer, with the excerpt shown.
    - **C.** Retrieval only: the five excerpts, no generated answer.
    - **D.** Nothing new: the PDF and its search box.

    Vote alone. Discuss with a neighbour for two minutes. Vote again.
    """)
    return


@app.cell
def _(mo, qa, qa_hits, scorecard):
    _c, _r = scorecard["closed_answerable"], scorecard["rag_answerable"]
    _short = [_q for _q in qa if _q["answerable"] and _q["rag_outcome"] != "correct"]
    _seen = sum(bool(_q["evidence_ranks"]) for _q in _short)  # the rule's sentence retrieved
    mo.accordion(
        {
            "Discussion": mo.md(f"""
        A answered {_c["correct"]} of 16 from memory, and its declines often came with a made-up
        number (Reveal 3); it is out. B: {_r["correct"] + _r["number only"]} of 16 right numbers,
        only {_r["correct"]} with a section that holds the rule, {_r["wrong"]} wrong,
        {_r["declined"]} declined, none invented; the rule's own sentence was among the excerpts
        for {_seen} of the {len(_short)} that fell short, so showing the excerpt and its section
        lets the crew catch most of them. C: the right section among five for
        {qa_hits} of 16; no generated errors, but the crew reads five excerpts. D: the baseline a
        crew already has. A defensible answer: B with the excerpt and section number always
        shown, or C where a wrong number is costly; never A. The vote is about what the person
        checks, not about the model alone.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where language models live

    | Today | What it is | Lecture |
    |:------|:-----------|:--------|
    | ChatGPT, Claude, Gemini | next-token models tuned on instructions and human preferences | today |
    | Reasoning models (o1, DeepSeek-R1) | long chains of thought, trained by reinforcement learning | 26 |
    | Assistants with file search | retrieval, then an answer citing the source | today |
    | Coding assistants, agents | models that call tools and check the results | 25 |
    | Embedding search | vectors like BGE's in a vector database | 4, today |

    The chat assistants are next-token models like today's, scaled up (Kaplan et al., 2020) and
    tuned on instructions and preferences (Ouyang et al., 2022). Reasoning models write a chain of
    thought before answering (Wei et al., 2022) and are trained to do so by reinforcement learning
    (DeepSeek-AI, 2025). Agents call tools, a search or a calculator, and read the results back
    into the document. File search in today's assistants is retrieval plus generation, as here;
    the embeddings sit in a vector index such as Lecture 4's FAISS.

    ## 7. Check yourself

    ### Exit question

    A vendor's assistant "answers any question about the MUTCD." Name the two numbers you would
    ask for before your crews use it, and how you would measure them.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answer": mo.md(r"""
        (1) How often the right section is among what it retrieves, on questions with known
        answers (like the manual's cross-references, or a set the agency writes); (2) how often
        it answers wrong, and how often it invents an answer to a question the manual does not
        cover, on questions the vendor has not seen. Both on the agency's own questions, with the
        counts.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Gotchas

    - **Shift by one.** Position $t$'s logits score token $t+1$. Hugging Face shifts inside
      `model(ids, labels=ids)`; a hand-written loss must shift itself (forgetting it gave 10.15
      nats instead of 3.76).
    - **GPT-2 has no padding token.** We reuse end-of-text and mask it out of the labels (`-100`)
      and the attention; the end-of-text appended to each training report is a real target, the
      padding is not.
    - **Read shared examples once.** Every coding prompt starts with the same system turn and
      examples: read them once, keep their cache, and run each report's turn on a copy (the cache
      is updated in place). Pad on the right and read each report at its own last token: a
      causal model never sees the padding after a token. The check reads the first prompts of
      every setting alone, with no batch, padding, or shared cache.
    - **Precision is part of the method.** In bfloat16 a batch's padding changes the rounding of
      every product, enough to move close answers. In float32, with TF32 off, the cached and
      batched pass agrees with each prompt read alone to about $10^{-5}$ (§4's check). A free
      Colab T4 has no native bfloat16 either.
    - **Bits per character, not perplexity per token,** when tokenizers differ: GPT-2 and Qwen
      cut the same reports into different numbers of tokens (Qwen's vocabulary is five times
      GPT-2's).
    - **Name every generation setting.** The pinned Qwen3.5 releases ship no
      `generation_config.json`, and the model's own config stops only at `<|endoftext|>`: without
      `eos_token_id` for `<|im_end|>` too, it would write the next turns of the chat itself.
      Name the decoding (greedy, repetition penalty 1.0) and turn thinking off in the template
      (`enable_thinking=False`), or a model may reason for hundreds of tokens before answering.
    - **Sums on a GPU can change from run to run.** A summing reduction adds with atomics in any
      order, and fused attention's backward pass is not deterministic: we score per token and
      sum in float64, and fine-tune GPT-2 with PyTorch's plain attention.
    - **`model.train()` turns dropout on.** Score with `model.eval()` and under `torch.no_grad()`.
    - **The order of a list in a prompt is a hyperparameter**: with no examples, the answers
      follow where each event type sits in the list (§5).
    - **Score the reader and the retriever separately, by the sentence.** A wrong answer whose
      excerpts hold the rule's own sentence is a reading failure; one whose excerpts miss it is a
      retrieval failure, even when they come from the right section (the stop line, §5).

    ## Going deeper (Forsyth)

    - §13.1.3: Markov chain models of text. A unigram model is a chain of order 0 (our 1.78-bit
      baseline); bigram and trigram models condition on one or two earlier items, and Forsyth
      warns that most n-grams are never seen, so their counts need smoothing (our unigram adds
      one to every count). A transformer conditions on the whole prefix and shares strength
      across contexts through learned vectors instead of counts. Forsyth's items are letters and
      words; ours are byte-pair tokens.
    - §16.1.1–16.1.2: the softmax over units' outputs and the log-loss (cross-entropy) it is
      trained with. The next-token head is that classifier with 50,257 classes, applied at every
      position at once.
    - §16.4.2: dropout, on while training and off at test time, with the $1/(1-p)$ reweighting
      during training that PyTorch applies; here it is on during fine-tuning and off whenever the
      test is scored.
    - §6.3.6: TF-IDF. Forsyth's score is $c_{ij} \log(N/N_i)$; scikit-learn's, as used here, is
      $(1 + \log c_{ij})\,(\log\frac{1+N}{1+N_i} + 1)$ with each row then L2-normalized: different
      numbers, the same idea.
    - §6.3.1 and §1.2: the cosine between document vectors, and nearest neighbours. Retrieval is
      a 5-nearest-neighbour search over the excerpts' embeddings, by the cosine (Forsyth calls
      the cosine itself the cosine distance).

    **Exercises.** (1) Build a bigram model of GPT-2's tokens on the 20,000 training reports with
    add-one smoothing (§13.1.3) and score the 2,000 test reports in bits per character: where
    does it fall between the unigram (1.78) and GPT-2 as released (1.08)? (2) Keep the three
    examples per type of the first draw but shuffle their order in the prompt five times: how
    far does the balanced accuracy move, against the spread over draws?

    ## Things to remember

    - A language model predicts the next token; a chat is a document.
    - Score text in bits per character across tokenizers.
    - Examples in a prompt help; labels for a probe help more.
    - Retrieval sets the ceiling; the reader can still misread.
    - Show the excerpt; a person signs the answer.

    **Next:** 1-D signals: rainfall, runoff, and vibration.
    """)
    return


if __name__ == "__main__":
    app.run()
