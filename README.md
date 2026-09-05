# MOOD RING

**Three frontier NLP abilities, zero training, one laptop.** →
**[mood-ring-theta.vercel.app](https://mood-ring-theta.vercel.app)**

Three pretrained models do three different jobs here, and none of them were trained by us.
That is the whole point: transfer learning means somebody else paid the pretraining bill,
and the abilities cost the price of a download.

| # | task | model | first-run download |
| - | ---- | ----- | ------------------ |
| 1 | sentiment | `distilbert-base-uncased-finetuned-sst-2-english` | ~255 MB |
| 2 | zero-shot classification | `facebook/bart-large-mnli` | ~1.6 GB |
| 3 | fill-mask | `distilbert-base-uncased` | ~265 MB |

Sections run smallest-model-first, so you see real output on screen before the 1.6 GB
download starts. Each section is independently fault tolerant — if a model fails to load
(you go offline halfway through), that section prints a one-line skip and the rest carries on.

## Install and run

```
pip install transformers torch

python mood_ring.py          # three demos, then an interactive REPL
python mood_ring.py --json   # three demos, writes results.json, no REPL
```

CPU is fine. No API key, no account, nothing leaves your machine.

**About the first run:** it downloads roughly **2.1 GB** of weights into
`~/.cache/huggingface/hub` (override with `HF_HOME`). Expect the zero-shot section to pause
for a while on a cold cache. Every run after that is fully offline.

## What it does

**Sentiment** — four sentences: clearly positive, dry negative, upset, and one genuine
hedge. The hedge is the interesting one. `"Cautiously optimistic about the new release."`
comes back **POSITIVE at 99.3%** — the model has exactly two labels, cannot answer "mixed",
and does not get less certain about being forced to pick. Confidence is not calibrated doubt.

**Zero-shot** — the labels `["cooking", "technology", "sports", "finance"]` were typed into a
list a second before the run. Nothing was fine-tuned on them. The model reasons about what
the labels *mean*: each becomes a sentence to test for entailment, and the winner is whichever
the text most implies.

**Fill-mask** — `"The best way to learn machine learning is to [MASK] things."` Top 5 guesses
with their probabilities. This guessing game, played over the whole internet, **is**
pretraining — it is where the other two abilities came from.

**REPL** — a `> ` prompt for live sentiment on anything you type. Blank line, Ctrl-D or
Ctrl-C all exit cleanly.

## Repo

```
mood_ring.py     the whole thing: three pipelines, a REPL, and --json capture
results.json     captured output from a real run (2026-09-05, transformers 5.16.1)
site/            the showcase site — a single standalone index.html
```

The site embeds `results.json` verbatim; every classification it displays is real captured
output, and the provenance block names all three models, the transformers version and the run
date. See [`site/README.md`](site/README.md) for deploy and refresh instructions.
