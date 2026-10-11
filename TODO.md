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
- [ ] 02 Training Deep NNs (cont.); Intro to Keras/TF; Tabular Data — video 3Lr8nVUDCpk
- [ ] 03 DL for Computer Vision – CNNs from Scratch — video 8QuyDcMIdRc
- [ ] 04 DL for Computer Vision – Transfer Learning, Fine-Tuning; HuggingFace — video 8xh7Y0pBrCE
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

## Wiki
- [x] wiki/01-introduction.md
- [x] Topic pages (cross-lecture concepts) — ai-ml-dl-genai, representation-learning, neural-network-anatomy, activation-functions (lecture 1 only; extend as lectures land)
- [x] INDEX.md table of contents

## Images
- [x] raw/images/01-introduction/ — rank, pick, render, embed in the wiki and slide files
- [x] INDEX.md image rules; AGENTS.md Images conventions and table row; BUILD_LOG.md; LICENSE.md attribution

## Publish
- [x] kb.json — coverage, materials.method, provenance caveats
- [x] BUILD_LOG.md — lecture 1 record
- [x] SEE_ALSO.md, if a sibling KB is genuinely relevant
- [x] verify_kb.py clean, and its review section read
- [ ] Commit and push
- [ ] PATCH kbUrl onto the catalog entry
