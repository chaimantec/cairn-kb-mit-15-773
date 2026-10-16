# KB build — MIT 15.773 (Hands-On Deep Learning, Spring 2024)

Catalog id `5935cc5f-a533-4ded-a08f-6fac61ded777`. Course site:
<https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/>. Decks are linked from
`lists/lecture-notes/` as `mit15_773_s24_lecNN.pdf`, and their numbering does **not** follow the
11 videos: there are decks 01, 02, 02-backprop-example, 03a, 03b, 04–10, 10.5 and 11. Work out
which deck(s) each video covers from the transcript before reading anything. Video 1 covers deck 01 only
(it ends on deck 01's last slide; deck 02 opens video 2). Videos 1–9 are built (see BUILD_LOG.md);
video 9 presented deck 09 to its slide 66 and did not reach the RAG of its catalog title. Deck 10
("LLMs (2/2)", 115 pages) opens by reviewing video 9 (slide 2) and correcting its "Dense (ReLU)" layer to
linear activations (slide 3), so video 10 probably starts there; confirm from the transcript. Deck 10 runs
through prompting, RAG (slides 76–82) and fine-tuning, and deck 10.5 ("Finetuning") repeats some of its
fine-tuning slides: work out how videos 10 and 11 split decks 10, 10.5 and 11 before reading any of them.
Lecture 9's wiki page, the large-language-models page and the deck 09 slide file cite deck 10's slide 3 by
URL; once deck 10 is transcribed, point those citations at its slide file.

Images are opted in: every figure-bearing slide, rendered per Step 1c, never a slide with an OCW notice.

This file holds **open work only**. When a run is committed and pushed, its finished lines move
to [`BUILD_LOG.md`](BUILD_LOG.md).

## Lecture 10 run (2026-10-16)

Video 10 (`d-tngNnaG4U`, 77 min) presents deck 10 (`mit15_773_s24_lec10.pdf`, 115 pages; 115 is OCW's
end page) from slide 2 (the review) and slide 3 (the ReLU correction) to slide 112 (the LoRA Colab,
deferred to a video walkthrough). Video 11 opens on deck 11, so no video presents deck 10.5
(`mit15_773_s24_lec105.pdf`, 33 pages), a variant of deck 10's slides 82–114; it is lecture 10's
companion deck. Numbering: slide N = PDF page N in both. Deck 10 prints numbers only on 2, 3, 67 and
76–114, and from 76 on they run one ahead (a hidden slide between 67 and 76). Text-layer OCW notices,
deck 10: 59, 70–73, 78, 80, 87–89, 92, 93, 95, 113 (114 is a CC BY credit); deck 10.5: none, though it
reuses deck 10's excluded rasters.

- [x] 10 Generative AI – Adapting LLMs with PEFT — video d-tngNnaG4U → raw/transcripts/10-adapting-llms.md (+ original/)
- [x] Slides 10 → raw/slides/10-adapting-llms.md (Sonnet, three agents); deck 10.5 → raw/slides/10-finetuning.md (one agent)
- [x] 10 figure check on rendered images (3, 20, 31, 33, 39, 58, 77, 85, 103, 109, 111, 114; crops of 98 and 100)
- [x] raw/images/10-adapting-llms/ — render, embed in slide file (13 slides; none from deck 10.5)
- [ ] wiki/10-adapting-llms.md, with images hand-placed
- [ ] Topic pages for lecture 10; repoint deck 10 slide 3 citations (lecture 9 page, large-language-models, deck 09 slide file) at the new slide file
- [ ] INDEX.md — lecture 10 entries, coverage and image sentences
- [ ] kb.json, BUILD_LOG.md lecture 10 record, AGENTS.md image row, LICENSE.md attribution, sources.md, SEE_ALSO.md
- [ ] verify_kb.py clean (with check_math and --github), review section read
- [ ] Commit and push
- [ ] Move this section to BUILD_LOG.md

## Transcripts
- [ ] 11 Generative AI – Text-to-Image Models — video NQBhhRG-Pe4
