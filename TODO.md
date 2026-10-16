# KB build — MIT 15.773 (Hands-On Deep Learning, Spring 2024)

Catalog id `5935cc5f-a533-4ded-a08f-6fac61ded777`. Course site:
<https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/>. Decks are linked from
`lists/lecture-notes/` as `mit15_773_s24_lecNN.pdf`, and their numbering does **not** follow the
11 videos: there are decks 01, 02, 02-backprop-example, 03a, 03b, 04–10, 10.5 and 11. Work out
which deck(s) each video covers from the transcript before reading anything. Video 1 covers deck 01
only (it ends on deck 01's last slide; deck 02 opens video 2). Videos 1–8 are built (see BUILD_LOG.md);
video 8 presented deck 08 to its slide 93 and skipped its contrastive-learning section (94–98). Deck 09
(LLMs 1/2) opens with a review of masking, so video 9 probably starts there; confirm from the transcript,
and work out how videos 9–10 split decks 09, 10 and 10.5. Deck 09 shares rasters with deck 08 on its
pages 2–4 (deck 08's masking diagram, 70–71 and 76), 26 (encoder pieces of 14, 21, 22) and 37 (the use
cases, 30–32): check those on renders before rendering.

Images are opted in: every figure-bearing slide, rendered per Step 1c, never a slide with an OCW notice.

This file holds **open work only**. When a run is committed and pushed, its finished lines move
to [`BUILD_LOG.md`](BUILD_LOG.md).

## Transcripts
- [ ] 10 Generative AI – Adapting LLMs with PEFT — video d-tngNnaG4U
- [ ] 11 Generative AI – Text-to-Image Models — video NQBhhRG-Pe4

## Lecture 9 run (2026-10-16)

Video 9 (`KGDe1QvfKJ8`, 75 min) presents deck 09 (`mit15_773_s24_lec09.pdf`, 68 pages; page 68 is
OCW's end page) from slide 2 to slide 66, ending on the token-visualizer demo; slide 67 (a sidebar) is
not mentioned. RAG, in the catalog title, is deck 10's slides 76–82, so it falls to video 10.
Numbering: every printed number equals its page; 1–4, 44, 45 and 68 print none. Text-layer OCW
notices: 27, 29, 41, 67.

- [x] 09 Generative AI – LLMs and RAG — video KGDe1QvfKJ8 → raw/transcripts/09-large-language-models.md (+ original/)
- [x] Slides 09 → raw/slides/09-large-language-models.md (Sonnet, two agents)
- [x] 09 figure check on rendered images (19, 26, 39, 43, 49, 52, 54, 61, 64)
- [x] raw/images/09-large-language-models/ — render, embed in slide file (10 slides)
- [ ] wiki/09-large-language-models.md, with images hand-placed
- [ ] Topic pages for lecture 9
- [ ] INDEX.md — lecture 9 entries, coverage and image sentences
- [ ] kb.json, BUILD_LOG.md lecture 9 record, AGENTS.md image row, LICENSE.md attribution, sources.md
- [ ] verify_kb.py clean (with check_math and --github), review section read
- [ ] Commit and push
- [ ] Move this section to BUILD_LOG.md
