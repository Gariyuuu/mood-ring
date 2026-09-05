# MOOD RING — showcase site

A single self-contained `index.html`. No build step at serve time, no framework, no bundler,
no CDN scripts — all CSS and JS are inline, the captured run is inlined as a JS object, and
the full text of `mood_ring.py` is inlined in the code block. The one external request the
page makes is to Google Fonts (Inter Tight / Inter / IBM Plex Mono); every family has a real
system fallback stack, so the page is fully legible offline.

## Deploy

```
cd site && vercel --prod
```

Static, zero config — Vercel serves `index.html` as-is. Any static host works the same way
(`python3 -m http.server`, GitHub Pages, S3, Netlify drop).

## Where the content comes from

Every classification on the page is captured output from a real run of `../mood_ring.py` —
the labels, the confidences, the top-5 mask predictions, the run date and the transformers
version. None of it is hand-typed or illustrative. The provenance block under the hero names
all three models, the transformers version, and the date of the run it is showing:

| section    | model                                            | first-run download |
| ---------- | ------------------------------------------------ | ------------------ |
| sentiment  | `distilbert-base-uncased-finetuned-sst-2-english` | ~255 MB            |
| zero-shot  | `facebook/bart-large-mnli`                        | ~1.6 GB            |
| fill-mask  | `distilbert-base-uncased`                         | ~265 MB            |

The page says the same thing its final section says: the first run downloads roughly 2.1 GB
of weights into `~/.cache/huggingface/hub`, and every run after that is offline.

## Refreshing the showcase

From the repository root:

```
python mood_ring.py --json      # re-runs the three demo sections, rewrites results.json
```

Then update the two embedded blocks in `site/index.html` so the page cannot drift from the
run it claims to show:

1. the `const RESULTS = { … }` object literal in the page's `<script>` — replace it with the
   new contents of `results.json`;
2. the `<pre id="code">` block — replace it with the HTML-escaped text of `mood_ring.py`
   (`&`, `<` and `>` only).

Everything else on the page — the sentiment cards, the zero-shot rows and their highlighted
label, the ranked mask candidates, the provenance line and the footer — is rendered from
`RESULTS` at load time, so those two edits are the whole update.

## Accessibility and behaviour notes

- Light and dark are both defined; the toggle in the header persists to `localStorage`
  (`mood-theme`), and with no stored choice the page follows `prefers-color-scheme`.
- Confidence bars animate their width when scrolled into view. Under
  `prefers-reduced-motion: reduce` the transitions are removed and the bars simply appear
  at their final width — no value is ever hidden by the animation.
- Every bar carries `role="img"` with its label and percentage in `aria-label`, and the
  numeric percentage is always printed next to it in text.
