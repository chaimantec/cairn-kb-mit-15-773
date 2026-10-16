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

## Lecture 8 run (2026-10-16)

Video 8 (`v-lHsawHyaI`, 76 min) presents deck 08 (`mit15_773_s24_lec08.pdf`, 99 pages; page 99 is
OCW's end page) from slide 2 to slide 93, after finishing lecture 7's Colab (≈3:53–5:25) and with an
iPad derivation in place of slides 15–19 (≈7:46–24:08) and the Hugging Face Colab at slide 87. Slides
94–98 (contrastive learning, "time permitting") are not presented. Numbering: every printed number
equals its page; 38 pages print none. Text-layer OCW notices: 28, 41, 50, 51, 56, 79, 80, 81, 82, 83,
90, 93, 96, 97, 98 (81's names a BERT figure but shows the course's slide-38 diagram).

- [x] 08 DL for Natural Language – Transformers, Self-Supervised Learning — video v-lHsawHyaI → raw/transcripts/08-transformers-and-self-supervised-learning.md (+ original/)
- [x] Slides 08 → raw/slides/08-transformers-and-self-supervised-learning.md (Sonnet, three agents)
- [x] 08 figure check on rendered images (18, 20, 24, 29, 47, 69, 80, 81, 86, 91, 93 crop)
- [x] raw/images/08-transformers-and-self-supervised-learning/ — render, embed in slide file (19 slides)
- [ ] wiki/08-transformers-and-self-supervised-learning.md, with images hand-placed
- [ ] Topic pages for lecture 8
- [ ] INDEX.md — lecture 8 entries, coverage and image sentences
- [ ] kb.json, BUILD_LOG.md lecture 8 record, AGENTS.md image row, LICENSE.md attribution, sources.md
- [ ] verify_kb.py clean (with check_math and --github), review section read
- [ ] Commit and push
- [ ] Move this section to BUILD_LOG.md

## Transcripts
- [ ] 09 Generative AI – LLMs and RAG — video KGDe1QvfKJ8
- [ ] 10 Generative AI – Adapting LLMs with PEFT — video d-tngNnaG4U
- [ ] 11 Generative AI – Text-to-Image Models — video NQBhhRG-Pe4
