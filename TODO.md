# KB build — MIT 15.773 (Hands-On Deep Learning, Spring 2024)

Catalog id `5935cc5f-a533-4ded-a08f-6fac61ded777`. Course site:
<https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/>. Decks are linked from
`lists/lecture-notes/` as `mit15_773_s24_lecNN.pdf`, and their numbering does **not** follow the
11 videos: there are decks 01, 02, 02-backprop-example, 03a, 03b, 04–10, 10.5 and 11. Work out
which deck(s) each video covers from the transcript before reading anything. Video 1 covers deck 01
only (it ends on deck 01's last slide; deck 02 opens video 2). Videos 1–4 are built (see BUILD_LOG.md);
video 4 ended on deck 04's slide 81, so deck 05 (its page 1 is a roadmap of the NLP lectures) should
open video 5. Confirm from the transcript.

Images are opted in: every figure-bearing slide, rendered per Step 1c, never a slide with an OCW notice.

This file holds **open work only**. When a run is committed and pushed, its finished lines move
to [`BUILD_LOG.md`](BUILD_LOG.md).

## Transcripts
- [ ] 06 DL for Natural Language – Embeddings — video LqFc0z-pQTg
- [ ] 07 DL for Natural Language – Transformers — video IeF7aATDaw4
- [ ] 08 DL for Natural Language – Transformers, Self-Supervised Learning — video v-lHsawHyaI
- [ ] 09 Generative AI – LLMs and RAG — video KGDe1QvfKJ8
- [ ] 10 Generative AI – Adapting LLMs with PEFT — video d-tngNnaG4U
- [ ] 11 Generative AI – Text-to-Image Models — video NQBhhRG-Pe4

## Lecture 5 run (2026-10-14)

Video 5 presents deck 05 (`mit15_773_s24_lec05.pdf`, 67 pages, 1:1) whole: slides 1–66, then the
Colab from ≈45:53. Text-layer OCW notices on slides 15 and 17. Byte-pair encoding is deferred to
deck 09 (slides 58–60); dropout is finally explained (≈1:11:38–1:14:44) with no deck-05 slide.

- [x] Transcript 05 — video duBLxHjaecQ → raw/transcripts/05-text-vectorization-and-bag-of-words.md (+ original/)
- [x] Slides 05 → raw/slides/05-text-vectorization-and-bag-of-words.md (Sonnet, two agents)
- [x] 05 figure check on rendered images (21, 37, 40, 50, 56, 65 all agree; slide 37 note on hidden text)
- [x] raw/images/05-text-vectorization-and-bag-of-words/ — render, embed in slide file and wiki
- [x] wiki/05-text-vectorization-and-bag-of-words.md
- [x] Topic pages for lecture 5 (new NLP pages; extend dropout, keras, overfitting, multiclass); correct pages that defer dropout
- [x] INDEX.md — lecture 5 entries, coverage and image sentences
- [x] kb.json, BUILD_LOG.md lecture 5 record, AGENTS.md image row, LICENSE.md attribution, sources.md
- [x] verify_kb.py clean (with check_math), review section read
- [x] Commit and push
- [ ] Move this section to BUILD_LOG.md
