#!/usr/bin/env python3
"""
MOOD RING - three frontier NLP abilities, zero training, one laptop.

Three pretrained models do three different jobs here, and none of them were
trained by us. That is the whole point: transfer learning means somebody else
paid the pretraining bill and we get the abilities for the price of a download.

  1. SENTIMENT   distilbert-base-uncased-finetuned-sst-2-english  (~255 MB)
  2. ZERO-SHOT   facebook/bart-large-mnli                         (~1.6 GB)
  3. FILL-MASK   distilbert-base-uncased                          (~265 MB)

HONESTY ABOUT THE FIRST RUN
  The first run DOWNLOADS these weights - roughly 2.1 GB in total, from
  ~255 MB for the smallest to ~1.6 GB for bart-large-mnli, which is by far
  the big one. Expect the zero-shot section to pause for a while on a cold
  cache. Files land in ~/.cache/huggingface/hub (override with HF_HOME).
  Every run after that is fully offline - nothing is uploaded, ever, and no
  API key is involved. The sections are ordered smallest-model-first so you
  see real output before the 1.6 GB download starts.

USAGE
  python mood_ring.py           run the three demos, then an interactive REPL
  python mood_ring.py --json    run the three demos, write results.json, exit

Each section is independently fault tolerant: if a model fails to load (say
you go offline halfway through), that section prints a one-line skip and the
rest of the program carries on.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import sys

try:
    import transformers
    from transformers import pipeline
except ImportError:
    sys.exit(
        "mood_ring needs the transformers library (and a torch backend).\n"
        "  install:  pip install transformers torch\n"
        "  then run: python mood_ring.py"
    )

transformers.logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = pathlib.Path(__file__).resolve().parent

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
FILL_MASK_MODEL = "distilbert-base-uncased"

BAR_WIDTH = 24

SENTIMENT_TEXTS = [
    # clearly positive - the easy case, and the model should be near-certain
    "This is hands down the best sandwich I have ever eaten.",
    # dry negative - no insults, no swearing, just resignation
    "Sure, the meeting could have been an email.",
    # upset - unambiguous, high intensity
    "I waited two hours and nobody even told me the flight was cancelled.",
    # genuinely ambivalent - this is the interesting one. The model has exactly
    # two labels available, so it CANNOT say "mixed"; forcing a side is the
    # whole point of including it. Note what the score does NOT do: it does not
    # drop to reflect the hedge. Confidence here is not calibrated doubt.
    "Cautiously optimistic about the new release.",
]

ZERO_SHOT_LABELS = ["cooking", "technology", "sports", "finance"]
ZERO_SHOT_TEXTS = [
    "Sear the steak in a hot pan, then finish it in the oven with butter and thyme.",
    "He tore his ACL in the third quarter and is out for the season.",
    "Interest rates on the thirty-year mortgage climbed again this quarter.",
]

FILL_MASK_PROMPT = "The best way to learn machine learning is to [MASK] things."


# ---------------------------------------------------------------- formatting


def bar(score: float, width: int = BAR_WIDTH) -> str:
    """A █ bar over a ░ track, so 0.02 still looks like something."""
    filled = int(round(max(0.0, min(1.0, score)) * width))
    return "█" * filled + "░" * (width - filled)


def pct(score: float) -> str:
    return f"{score * 100:5.1f}%"


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")
    print("─" * 72)


def loading(model: str, size: str) -> None:
    print(f"  loading {model} ({size})")
    print("  first run downloads it to ~/.cache/huggingface; after that it is cached.")
    sys.stdout.flush()


def skip(section: str, exc: Exception) -> None:
    print(f"  [skipped] {section}: {type(exc).__name__}: {exc}".replace("\n", " ")[:200])


# ------------------------------------------------------------------ sections


def run_sentiment() -> tuple[object | None, list[dict]]:
    """Smallest model, so it goes first: real output before the big download."""
    rule("1 · SENTIMENT — a model fine-tuned on movie reviews, pointed at anything")
    loading(SENTIMENT_MODEL, "~255 MB")
    try:
        clf = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    except Exception as exc:  # offline, corrupt cache, no backend...
        skip("sentiment", exc)
        return None, []

    captured: list[dict] = []
    try:
        print()
        for text in SENTIMENT_TEXTS:
            result = clf(text)[0]
            label, score = result["label"], float(result["score"])
            print(f"  [{label:<8}] {bar(score)} {pct(score)}  “{text}”")
            captured.append({"text": text, "label": label, "score": score})
        print()
        print("  Only two labels exist, so ambivalence has nowhere to go. The last")
        print("  sentence hedges out loud and the model still commits, hard. Forcing a")
        print("  side is the interesting part — the confidence is not a measure of doubt.")
    except Exception as exc:
        skip("sentiment demo", exc)

    return clf, captured


def run_zero_shot() -> list[dict]:
    rule("2 · ZERO-SHOT — labels invented at runtime, no training of any kind")
    loading(ZERO_SHOT_MODEL, "~1.6 GB — the big one; this is the long pause")
    try:
        clf = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
    except Exception as exc:
        skip("zero-shot", exc)
        return []

    captured: list[dict] = []
    try:
        print(f"\n  labels: {', '.join(ZERO_SHOT_LABELS)}\n")
        for text in ZERO_SHOT_TEXTS:
            result = clf(text, candidate_labels=ZERO_SHOT_LABELS)
            label = result["labels"][0]
            score = float(result["scores"][0])
            print(f"  “{text}”")
            print(f"  → [{label:<10}] {bar(score)} {pct(score)}\n")
            captured.append({"text": text, "top_label": label, "score": score})
        print("  Nobody fine-tuned this model on cooking, sports or finance. The labels")
        print("  were made up a second ago — it reasons about what the WORDS mean.")
    except Exception as exc:
        skip("zero-shot demo", exc)

    return captured


def run_fill_mask() -> dict:
    rule("3 · FILL-MASK — the pretraining objective itself, run in the open")
    loading(FILL_MASK_MODEL, "~265 MB")
    try:
        unmasker = pipeline("fill-mask", model=FILL_MASK_MODEL, top_k=5)
    except Exception as exc:
        skip("fill-mask", exc)
        return {}

    captured: dict = {}
    try:
        print(f"\n  “{FILL_MASK_PROMPT}”\n")
        candidates = []
        for guess in unmasker(FILL_MASK_PROMPT):
            token = guess["token_str"].strip()
            score = float(guess["score"])
            print(f"  {token:<12} {bar(score)} {pct(score)}")
            candidates.append({"token": token, "score": score})
        print()
        print("  This guessing game, played over the whole internet, IS pretraining.")
        print("  Grammar, topic, tone, world knowledge — the other two abilities you")
        print("  just watched came out of nothing more exotic than this.")
        captured = {"prompt": FILL_MASK_PROMPT, "candidates": candidates}
    except Exception as exc:
        skip("fill-mask demo", exc)

    return captured


# ---------------------------------------------------------------------- repl


def repl(clf) -> None:
    rule("REPL — type a sentence, get a reading. Blank line or Ctrl-D to leave.")
    if clf is None:
        print("  sentiment model unavailable, so there is nothing to type at. Bye.")
        return
    print()
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  bye.")
            return
        if not text:
            print("  bye.")
            return
        try:
            result = clf(text)[0]
            label, score = result["label"], float(result["score"])
            face = ":)" if label.upper().startswith("POS") else ":("
            print(f"  {face} {label:<8} {bar(score)} {pct(score)}\n")
        except Exception as exc:
            skip("reading", exc)


# ---------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three pretrained models, three NLP abilities, zero training."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="run the three demo sections, write results.json, skip the REPL",
    )
    args = parser.parse_args()

    print("\n\033[1mMOOD RING\033[0m  ·  three frontier abilities, zero training, one laptop")
    print(f"transformers {transformers.__version__}")
    print("First run downloads ~2.1 GB of weights to ~/.cache/huggingface. Offline after.")

    clf, sentiment = run_sentiment()
    zero_shot = run_zero_shot()
    fill_mask = run_fill_mask()

    if args.json:
        payload = {
            "run_date": _dt.date.today().isoformat(),
            "transformers_version": transformers.__version__,
            "models": {
                "sentiment": SENTIMENT_MODEL,
                "zero_shot": ZERO_SHOT_MODEL,
                "fill_mask": FILL_MASK_MODEL,
            },
            "sentiment": sentiment,
            "zero_shot": zero_shot,
            "fill_mask": fill_mask,
        }
        out = HERE / "results.json"
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
        return

    repl(clf)


if __name__ == "__main__":
    main()
