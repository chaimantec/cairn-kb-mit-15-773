# KB build — MIT 15.773 (Hands-On Deep Learning, Spring 2024)

Catalog id `5935cc5f-a533-4ded-a08f-6fac61ded777`. Course site:
<https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/>. Decks are linked from
`lists/lecture-notes/` as `mit15_773_s24_lecNN.pdf`, and their numbering does **not** follow the
11 videos: there are decks 01, 02, 02-backprop-example, 03a, 03b, 04–10, 10.5 and 11. Work out
which deck(s) each video covers from the transcript before reading anything. Video 1 covers deck 01
only (it ends on deck 01's last slide; deck 02 opens video 2). Videos 1–5 are built (see BUILD_LOG.md);
video 5 presented deck 05 whole (slides 1–66), so deck 06 (Embeddings) should open video 6. Confirm
from the transcript.

Images are opted in: every figure-bearing slide, rendered per Step 1c, never a slide with an OCW notice.

This file holds **open work only**. When a run is committed and pushed, its finished lines move
to [`BUILD_LOG.md`](BUILD_LOG.md).

## Transcripts
- [ ] 07 DL for Natural Language – Transformers — video IeF7aATDaw4
- [ ] 08 DL for Natural Language – Transformers, Self-Supervised Learning — video v-lHsawHyaI
- [ ] 09 Generative AI – LLMs and RAG — video KGDe1QvfKJ8
- [ ] 10 Generative AI – Adapting LLMs with PEFT — video d-tngNnaG4U
- [ ] 11 Generative AI – Text-to-Image Models — video NQBhhRG-Pe4

## Lecture 6 run (2026-10-14)

Video 6 (`LqFc0z-pQTg`, 78 min) presents deck 06 (`mit15_773_s24_lec06.pdf`, 70 pages, 1:1 by
corner spans; page 70 is OCW's end page): it opens on slide 2 ("Until now, we encoded input text
strings with one-hot vectors") and ends in the Colab (GloVe, then trained-from-scratch
embeddings, 72%). Text-layer OCW notices: slides 14, 15, 17, 18, 19 (Cohere embedding quizzes) and
47 (Chollet's one-hot vs embedding figures). Slide 19 carries the Cohere notice too. Slides 48–51 (GloVe visualizations) carry no notice.

- [x] Transcript 06 — video LqFc0z-pQTg → raw/transcripts/06-word-embeddings.md (+ original/)
- [x] Slides 06 → raw/slides/06-word-embeddings.md (Sonnet, two agents; slide 19's notice found by the agent)
- [x] 06 figure check on rendered images (37, 45, 48, 59, 63, 67, 68 agree; slide 37's last handwritten label marked hard to read)
- [ ] raw/images/06-word-embeddings/ — render, embed in slide file and wiki (rendered 37, 49, 51, 55, 59, 61, 67, 68; slide file embedded; wiki pending)
- [ ] wiki/06-word-embeddings.md
- [ ] Topic pages for lecture 6 (word embeddings; extend text-vectorization, bag-of-words, NLP, keras, representation-learning)
- [ ] INDEX.md — lecture 6 entries, coverage and image sentences
- [ ] kb.json, BUILD_LOG.md lecture 6 record, AGENTS.md image row, LICENSE.md attribution, sources.md
- [ ] verify_kb.py clean (with check_math), review section read
- [ ] Commit and push
- [ ] Move this section to BUILD_LOG.md
