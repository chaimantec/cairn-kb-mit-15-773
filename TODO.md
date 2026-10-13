# KB build — MIT 15.773 (Hands-On Deep Learning, Spring 2024)

Catalog id `5935cc5f-a533-4ded-a08f-6fac61ded777`. Course site:
<https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/>. Decks are linked from
`lists/lecture-notes/` as `mit15_773_s24_lecNN.pdf`, and their numbering does **not** follow the
11 videos: there are decks 01, 02, 02-backprop-example, 03a, 03b, 04–10, 10.5 and 11. Work out
which deck(s) each video covers from the transcript before reading anything. Video 1 covers deck 01
only (it ends on deck 01's last slide; deck 02 opens video 2).

Images are opted in: every figure-bearing slide, rendered per Step 1c, never a slide with an OCW notice.

## Transcripts
- [x] 01 Introduction to Neural Networks and Deep Learning — video kyQ0CRkYhy4 (OCW human captions; light copy-edit, verbatim in original/)
- [x] 02 Training Deep NNs (cont.); Intro to Keras/TF; Tabular Data — video 3Lr8nVUDCpk (OCW human captions; light copy-edit; covers deck 02 slides 2–94 only)
- [x] 03 DL for Computer Vision – CNNs from Scratch — video 8QuyDcMIdRc (OCW human captions; light copy-edit; covers deck 3A slides 2–43 and deck 3B slides 2–28; no CNNs — those are deck 04)
- [x] 04 DL for Computer Vision – Transfer Learning, Fine-Tuning; HuggingFace — video 8xh7Y0pBrCE (OCW human captions; light copy-edit, inline; covers deck 04 slides 2–81: CNNs from scratch, then transfer learning) → raw/transcripts/04-cnns-and-transfer-learning.md
- [ ] 05 DL for Natural Language – The Basics — video duBLxHjaecQ
- [ ] 06 DL for Natural Language – Embeddings — video LqFc0z-pQTg
- [ ] 07 DL for Natural Language – Transformers — video IeF7aATDaw4
- [ ] 08 DL for Natural Language – Transformers, Self-Supervised Learning — video v-lHsawHyaI
- [ ] 09 Generative AI – LLMs and RAG — video KGDe1QvfKJ8
- [ ] 10 Generative AI – Adapting LLMs with PEFT — video d-tngNnaG4U
- [ ] 11 Generative AI – Text-to-Image Models — video NQBhhRG-Pe4

## Crawl
- [x] Fetch course site index: https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/
- [x] Download linked PDFs and slides (14 decks, 11 OCW transcript PDFs; gitignored)
- [x] Write sources.md

## Slides (raw/slides/)
- [x] 01 Introduction — mit15_773_s24_lec01.pdf (110 pages; slide N = page N; read by Sonnet; figure check on 6 rendered pages, 1 corrected)
- [x] 02 Training Deep NNs — mit15_773_s24_lec02.pdf (95 pages; slide N = page N, pages 50–94 print N+1; read by Sonnet in three parts; figure check on 8 rendered pages incl. 2 backprop, 1 corrected)
- [x] 02 Backprop example — mit15_773_s24_lec02_backprop_example.pdf (14 handwritten pages; not presented in the video, pointed to at 59:40)
- [x] 03 deck 3A Keras/TF, training — mit15_773_s24_lec03a.pdf (44 pages; 1:1) → raw/slides/03-keras-tensorflow.md
- [x] 03 deck 3B Vision basics — mit15_773_s24_lec03b.pdf (29 pages; hidden slide 16, pages 16–28 print N+1 where printed; 8 OCW notices) → raw/slides/03-vision-basics.md
- [x] 03 figure check on rendered images (3A 10, 15, 17, 18; 3B 6 (scratch crop), 16, 18 — all agreed)
- [x] 04 CNNs and transfer learning — mit15_773_s24_lec04.pdf (84 pages; 1:1; 14 OCW notices) → raw/slides/04-cnns-and-transfer-learning.md (Sonnet, two agents in parallel)
- [x] 04 figure check on rendered images (21, 31, 40, 47, 75, 80; slide 40 corrected)

## Wiki
- [x] wiki/01-introduction.md
- [x] Topic pages (cross-lecture concepts) — ai-ml-dl-genai, representation-learning, neural-network-anatomy, activation-functions (lecture 1 only; extend as lectures land)
- [x] INDEX.md table of contents
- [x] wiki/02-training-deep-nns.md
- [x] Topic pages for lecture 2 (loss functions, gradient descent, backprop, SGD, Keras; extend neural-network-anatomy, activation-functions)
- [x] INDEX.md — lecture 2 entries and coverage sentence
- [x] wiki/03-keras-and-vision-basics.md
- [x] Topic pages for lecture 3 — new: overfitting-and-regularization, tensors, data-preprocessing, images-and-vision-tasks, multiclass-classification; extended: keras, training-neural-networks, gradient-descent, loss-functions, activation-functions, neural-network-anatomy
- [x] INDEX.md — lecture 3 entries and coverage sentence
- [ ] wiki/04-cnns-and-transfer-learning.md
- [ ] Topic pages for lecture 4 — new: convolutional-neural-networks, transfer-learning, data-augmentation; extend the pages it touches
- [ ] Correct the earlier pages that defer CNNs or dropout to lecture 4
- [ ] INDEX.md — lecture 4 entries and coverage sentence

## Images
- [x] raw/images/01-introduction/ — rank, pick, render, embed in the wiki and slide files
- [x] INDEX.md image rules; AGENTS.md Images conventions and table row; BUILD_LOG.md; LICENSE.md attribution
- [x] raw/images/02-training-deep-nns/ — deck 02 as slide-N, backprop deck as backprop-N; embed
- [x] INDEX.md image sentence; AGENTS.md table row; BUILD_LOG.md; LICENSE.md attribution (lecture 2)
- [x] raw/images/03-keras-and-vision-basics/ — deck 3A as 3a-N, deck 3B as 3b-N; embed
- [x] INDEX.md image sentence; AGENTS.md table row; BUILD_LOG.md; LICENSE.md attribution (lecture 3)
- [x] raw/images/04-cnns-and-transfer-learning/ — 16 renders; embedded in the slide file
- [ ] Embed lecture 4 images in the wiki page; INDEX.md image sentence; AGENTS.md table row; BUILD_LOG.md; LICENSE.md attribution

## Publish
- [x] kb.json — coverage, materials.method, provenance caveats
- [x] BUILD_LOG.md — lecture 1 record
- [x] SEE_ALSO.md, if a sibling KB is genuinely relevant
- [x] verify_kb.py clean, and its review section read
- [x] Commit and push
- [x] PATCH kbUrl onto the catalog entry (https://github.com/chaimantec/cairn-kb-mit-15-773)
- [x] Lecture 2: kb.json, BUILD_LOG.md, verify_kb.py clean, commit and push
- [x] Lecture 3: kb.json, BUILD_LOG.md, verify_kb.py clean, commit and push
- [ ] Lecture 4: kb.json, BUILD_LOG.md, verify_kb.py clean, commit and push
- [ ] Move finished sections from TODO.md to BUILD_LOG.md; TODO.md keeps only open work
