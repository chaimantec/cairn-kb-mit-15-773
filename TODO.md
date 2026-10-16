# KB build — MIT 15.773 (Hands-On Deep Learning, Spring 2024)

Catalog id `5935cc5f-a533-4ded-a08f-6fac61ded777`. Course site:
<https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/>. Decks are linked from
`lists/lecture-notes/` as `mit15_773_s24_lecNN.pdf`, and their numbering does **not** follow the
11 videos: there are decks 01, 02, 02-backprop-example, 03a, 03b, 04–10, 10.5 and 11. Work out
which deck(s) each video covers from the transcript before reading anything. Video 1 covers deck 01 only
(it ends on deck 01's last slide; deck 02 opens video 2). Videos 1–10 are built (see BUILD_LOG.md);
video 10 presented deck 10's slides 2–112, and deck 10.5 is lecture 10's unpresented companion deck.
Video 11 ("Generative AI – Text-to-Image Models", 75 min) opens on deck 11 ("Lecture 11: Diffusion
Models for Image Generation", 90 pages; page 90 is OCW's end page) with the ChatGPT-generated classroom
image of its slide 2, and calls itself "the last lecture of the class". Confirm from the transcript which
slide it ends on, and check deck 11's credits (slide 5 credits a Sora video to OpenAI) for OCW notices.

Images are opted in: every figure-bearing slide, rendered per Step 1c, never a slide with an OCW notice.

This file holds **open work only**. When a run is committed and pushed, its finished lines move
to [`BUILD_LOG.md`](BUILD_LOG.md).

## Lecture 11 run (2026-10-16)

Video 11 (`NQBhhRG-Pe4`, 75 min) presents deck 11 (`mit15_773_s24_lec11.pdf`, 90 pages; 90 is OCW's end
page) from slide 2 (the ChatGPT classroom image, 0:00) to slide 89 (the lightning tour of Hugging Face
computer-vision pipelines, ≈1:09:49 to the end). Slide 1 is the title page. The captions label the speaker
only `PROFESSOR:`; he refers to "when I did the quick lightning tour of the Hugging Face ecosystem for
language" (1:04:07), which lecture 8's captions attribute to Rama Ramakrishnan. Numbering: slide N = PDF
page N. Pages 2, 3 and 6–48 print their own number, 49–89 print N+1 (a hidden slide between 48 and 49), and
1, 4, 5 and 90 print none. Text-layer OCW notices (22): 9, 38–41, 47, 48, 50, 51, 59, 62–64, 68, 70, 73,
82–85, 87, 88. Raster hashes: the Labrador photo appears only on noticed pages; page 60 shares only a black
shape mask with them; no figure is shared with an earlier deck.

- [x] 11 Generative AI – Text-to-Image Models — video NQBhhRG-Pe4 → raw/transcripts/11-diffusion-models.md (+ original/)
- [x] Slides 11 → raw/slides/11-diffusion-models.md (Sonnet, three agents: 1–30, 31–60, 61–90)
- [x] 11 figure check on rendered images (20, 34, 37, 38 on a crop, 41, 64, 70, 73, 81, 82; renders of 2 and 72)
- [x] raw/images/11-diffusion-models/ — render, embed in slide file (16 slides)
- [ ] wiki/11-diffusion-models.md, with images placed
- [ ] Topic pages for lecture 11 (new and extended)
- [ ] INDEX.md — lecture 11 entries, coverage and image sentences (11 of 11)
- [ ] kb.json, BUILD_LOG.md lecture 11 record, AGENTS.md image row, LICENSE.md attribution, sources.md (deck 11 no longer provisional)
- [ ] verify_kb.py clean (with check_math and --github), review section read
- [ ] Commit and push
- [ ] Move this section to BUILD_LOG.md
