# KB build — MIT 15.773 (Hands-On Deep Learning, Spring 2024)

Catalog id `5935cc5f-a533-4ded-a08f-6fac61ded777`. Course site:
<https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/>. Decks are linked from
`lists/lecture-notes/` as `mit15_773_s24_lecNN.pdf`, and their numbering does **not** follow the
11 videos: there are decks 01, 02, 02-backprop-example, 03a, 03b, 04–10, 10.5 and 11. Work out
which deck(s) each video covers from the transcript before reading anything. Video 1 covers deck 01
only (it ends on deck 01's last slide; deck 02 opens video 2). Videos 1–7 are built (see BUILD_LOG.md);
video 7 presented deck 07 to its slide 76 and ended in the Colab before training the slot-filling
model, so video 8 probably opens by finishing that Colab before deck 08 (Transformers II). Confirm
from the transcript. Deck 08 reuses deck 07's slot-filling rasters (see BUILD_LOG.md, lecture 7).

Images are opted in: every figure-bearing slide, rendered per Step 1c, never a slide with an OCW notice.

This file holds **open work only**. When a run is committed and pushed, its finished lines move
to [`BUILD_LOG.md`](BUILD_LOG.md).

## Lecture 7 run (2026-10-14)

Video 7 (`IeF7aATDaw4`, 76 min) presents deck 07 (`mit15_773_s24_lec07.pdf`, 78 pages; page 78 is
OCW's end page): it opens on slide 2 (transformers across domains) and covers slides 2–76, with
the Colab from ≈1:02:34, a return to slides 72–76 at ≈1:08:01, and the Colab again to the end
(the model is set up but not trained). Numbering by corner spans: pages 2–9 print 2–9, pages 10–77
print N+1 (a hidden slide); headings use the PDF page. Text-layer OCW notices: slides 23, 51, 64,
65, 66, 67, 68, 69 ("Attention is All You Need" figures, Vaswani et al.); 65–69 share one raster.

- [x] 07 DL for Natural Language – Transformers — video IeF7aATDaw4 → raw/transcripts/07-transformers.md (+ original/)
- [x] Slides 07 → raw/slides/07-transformers.md (Sonnet, two agents; slide 17's hidden "see" noted)
- [x] 07 figure check on rendered images (33, 42, 51, 53, 61, 64, 75 and slide 1; slide 59's "sit" → "sat" added, slide 51's line description cut back to what the render settles)
- [x] raw/images/07-transformers/ — render, embed in slide file and wiki (rendered 33, 35, 43, 47, 49, 53, 75; slide file by script, wiki hand-placed with hand-written captions)
- [x] wiki/07-transformers.md
- [x] Topic pages for lecture 7 (new: transformers; extended word-embeddings, NLP, multiclass, text-vectorization, keras, representation-learning, CNNs; lecture 6 links forward)
- [x] INDEX.md — lecture 7 entries, coverage and image sentences
- [x] kb.json, BUILD_LOG.md lecture 7 record, AGENTS.md image row, LICENSE.md attribution, sources.md
- [x] verify_kb.py clean (with check_math and --github), review section read (slide file's math in Figure lines made plain text, `\hat{w}_N` → `\hat w_N` for github.com)
- [ ] Commit and push
- [ ] Move this section to BUILD_LOG.md

## Transcripts
- [ ] 08 DL for Natural Language – Transformers, Self-Supervised Learning — video v-lHsawHyaI
- [ ] 09 Generative AI – LLMs and RAG — video KGDe1QvfKJ8
- [ ] 10 Generative AI – Adapting LLMs with PEFT — video d-tngNnaG4U
- [ ] 11 Generative AI – Text-to-Image Models — video NQBhhRG-Pe4
