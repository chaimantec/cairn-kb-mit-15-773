# Build log — 15.773 knowledge base

The per-lecture record of how this knowledge base was built and checked: each
deck's licence notices and lecture pointers, each transcript's edit and checks,
what each figure audit found, each deck's printed slips, and which slides were
rendered as images and why the others were not. It is a build record, **not
needed to answer questions**: every fact a learner needs is on the wiki page,
slide file or transcript it concerns. The rules every build follows are in
[`AGENTS.md`](AGENTS.md).

## Lecture 1 — Introduction to Neural Networks and Deep Learning

Built 2026-10-11. Video `kyQ0CRkYhy4` (57 min); deck `mit15_773_s24_lec01.pdf`, 110 pages.

**Video ↔ deck.** The recording covers deck 01 only: it opens at slide 10 and ends at slide 109.
Slides 1–9 (prerequisites, grading, the instructor, why HODL) are not in the video, and slide 110 is
OCW's end page. The catalog's "…; Training Deep NNs" is not reached; deck 02 opens video 2.

**Numbering.** Slide N = PDF page N. 94 pages print their own number. `slide_number_map.py`
reported a non-1:1 deck with 19 absent numbers and 8 ambiguous pages, but the raw spans show two
different problems. (1) Stale static numbers, visibly printed (confirmed on page crops): pages 13–18
print 20–25, page 21 prints 29, page 31 prints 39, page 79 prints 87, pages 81–85 print 89–93, and
page 88 prints 96. (2) Split spans: "1"+"0" on page 10, "4"+"0"/"4"+"1" on 40–41, "6"+"1" on 61,
and "10"+"0", "1"+"02" … on 100–109. `--verify` therefore fails with 30 phantom mismatches; a plain
heading-sequence check (110 headings, 1..110 in order) passes. The Sonnet agent reported that the
stale numbers were "not visible"; the crops show they are.

**OCW notices (15):** slides 11, 12 (Minsky Family photo), 22, 23, 28, 41, 43 (puppy photo, © unknown),
24 (Sebastian Raschka figure and six bird photos), 35 (Apple, Al-Masry, Abhijit Ramesh, Springer
Nature, unknown), 37, 38 (Swarovski Optik), 44 (nielsr on Hugging Face), 49, 50 (petergyang on
Twitter), 87 (He, Zhang, Ren, Sun / arXiv; printed without "All rights reserved"). The agent looked
at every page and found none beyond the text layer's. Slide 32 carries a CC BY-SA credit (Glossier.ca,
Wikimedia Commons), not an exclusion.

**Transcript edit.** OCW human captions. Restorations: "GP" → GPA (27:00, 49:33, 50:19); "drama,
non-drama" → Rama, non-Rama (14:42, with note); "camera and the iPhone" → "camera in the iPhone"
(13:56); a comma at 4:40; "gettinig" → "getting" (42:32). `[Ed:]` notes at 3:53 (garbled "incomplete
underlier"), 14:42, 21:35 ("breast cancer brush"), 32:26 (spoken illustrative weights), 43:17 ("it
breaks up", unclear) and 53:31 (spoken "22.6" vs slide 104's $y = 0.226$). Checks: 73 timestamps
identical, number inventory identical, no paragraph ratio outside 0.72–1.10.

**Figure check.** Done in the parent on rendered images instead of with a second agent (the user
asked for no subagents beyond the deck read). Pages checked: 51, 59, 80, 93, 94 and 100. Five
agreed; slide 59's sigmoid was described as crossing 0.5 near x = 0.5 and passing 0.4 at x = 0, and
was corrected (it crosses 0.5 at x = 0). Slides 100–104's numbers were also checked by arithmetic:
hidden outputs 1.87, 3.03 and −0.06 → 0, output 0.226, all consistent with slide 100's weights.

**Printed slips.** Slides 105 and 107 print the third hidden node as `max(0.5+0.2x1−0.1x2)`, without
"0,", and with unbalanced parentheses. Slide 18's asterisk has no footnote. Slide 44's caption has
"laying".

**Images: 45 of 110 pages.** Rendered: 13, 14, 18, 19, 27, 32, 40, 42, 45, 46, 51, 56, 59, 61, 62,
64, 66, 67, 70, 71, 73, 75–85, 89–94, 98–100, 103, 104, 108, 109. Not rendered:
the 15 OCW-noticed slides; build steps or repeats superseded by a rendered slide (39 → 40, 53 → 51,
63 → 64, 68–69 → 67/70, 72 → 71, 74 → 73, 88 → 82, 101–102 → 100/103, 105 and 106, whose network
and formulas are on 100–104 and 61); tables the slide file reproduces exactly (21, 58); single-box
text diagrams (47, 48, 52); logistics slides 1–9 (slide 2 is a Kaggle screenshot); and text-only
slides. 4.87 MB, 28 PNG and 17 JPEG. All 45 also appear in the wiki page, which cites nearly every
slide it discusses.

## Lecture 2 — Training Deep Neural Networks

Built 2026-10-11. Video `3Lr8nVUDCpk` (78 min); decks `mit15_773_s24_lec02.pdf` (95 pages) and
`mit15_773_s24_lec02_backprop_example.pdf` (14 handwritten pages).

**Video ↔ deck.** The recording covers deck 02 only. It opens at slide 2 (recap), reaches slide 94
(summary of the training flow) at ≈1:12:46, and takes questions to the end. The backprop example is
pointed to at 59:40 and 1:11:14 (slide 87) but not presented. It is transcribed with this lecture
because the lecture assigns it. Deck 03a (Keras/TF introduction, overfitting, regularization,
Colab) is not reached: the lecturer defers regularization at 1:15:03. Deck 03a's slide 2 recaps
deck 02's slide 94, which is consistent with it opening video 3; confirm that when building
lecture 3.

**Numbering.** Deck 02: slide N = PDF page N. Pages 2–49 print their own number, pages 50–94 print
N+1 (a hidden slide 50 was never exported), and pages 1 and 95 print none. The raw corner spans are
clean on this deck, with no split spans or stale numbers. The corner spans "23)" and "(h)" on pages
23–24 are code text. Heading check 1..95 passes. Backprop example: each page carries a handwritten
number equal to its PDF page. Its text layer is OCR of the handwriting and unusable.

**OCW notices (1):** slide 75 (gradient descent figures © Jeremy Watt/neonwatty on GitHub). The
agents found no drawn notices. Unmarked third-party content that was rendered: slides 55–58 and 69
(plots from the mlrefined page that slide 75's notice credits to Jeremy Watt; OCW marks only 75),
66 (public-domain Cauchy lithograph, Wikimedia Commons), 93 (arXiv 1712.09913) and 94 (Chollet,
*Deep Learning with Python* 2nd ed., Figure 2.26). Rendering 55–58 and 69 follows the
notice-only rule and was flagged to the user for review.

**Transcription.** Sonnet. The first two agents were killed by a session limit after deck 02 slides
1–20 and backprop pages 1–8. Both had appended as they went, so nothing was lost. Fresh top-up agents
did 21–58 and 59–95 in parallel into scratch files, which the parent spliced, and backprop pages
9–14.

**Transcript edit.** OCW human captions, `STUDENT:` labels. Restorations: "characters library" →
Keras (17:08); "LLM" → lm and "GNN" → glm (20:11); "never used" → use (56:35); "backdrop" →
backprop (1:11:14). Two speaker labels were corrected: 34:45 was captioned as the lecturer, and
1:10:26 ran on from his "Yes.". Five punctuation/case fixes. `[Ed:]` notes at 14:48 ("h4"; the code
says `h`), 28:36 ("if the loss is 1, it's 0" inverts probability and loss), 40:13 ("the data" for
the derivative), 44:55 ("w1, 75 billion"), 46:27 (garbled dictation of slide 74) and 53:29 ("9:31"
is the time of day). Checks: 101 timestamps identical, number inventory identical, paragraph word
ratios 0.994–1.006.

**Figure check.** Done in the parent on 1400px renders. Deck 02 pages 4 (table, cell by cell), 33,
42, 55, 69 and 79, and backprop pages 7 and 12. All agreed except slide 52's minimum, which was
described as about −0.3; the plot shows it just below zero (−0.17 by the formula). It was corrected.
Slide 78's subscript slip was confirmed on a crop. Slide 33's trained values equal lecture 1's
slide 100. Slide 69's iterates (2.5 → 0.95 → 0.64 → 0.40) and slides 43 and 45's tables (base-2
logarithms) were checked by arithmetic.

**Printed slips.** Slide 78 writes $model(x_i)$ for $model(x^i)$. Slide 45's caption loses its
closing parenthesis at the slide edge. Slide 42's x-axis prints "1" for a clipped −1. Slides 15–28
use curly quotes in the Keras code; the slide file normalizes them inside code fences and says so.
Backprop page 7 writes $\partial a_2/\partial w_1$ for $\partial a_2/\partial w_2$.

**Images: 18.** Deck 02, 15 of 95 pages: 2, 10, 27, 33, 46, 47, 55–58, 66, 69, 79, 93, 94. Not
rendered: 75 (OCW notice); build steps or repeats superseded by a rendered slide (8, 9, 12–26 →
10/27; 42–45 → 46; 52–54 → 55; 68 → 69); tables the slide file reproduces exactly (4, 7, 61–64);
and text-only slides. Backprop example, 3 of 14 pages, saved as `backprop-N`: 2, 8, 12. Pages 1, 9
and 10 are superseded, and the rest are handwritten equations transcribed in full. 1.7 MB in total.
17 of the 18 are embedded in the wiki lecture page (deck 02's 15 plus backprop 8 and 12); backprop
page 2 is in the slide file only.

**Wiki.** A lecture page and five new topic pages (training-neural-networks, loss-functions,
gradient-descent, backpropagation, keras). activation-functions and neural-network-anatomy were
extended. Lecture 1's page and activation-functions had promised softmax and ReLU's properties for
"the next lecture". Lecture 2 defers softmax again and gives ReLU's gradient only in Q&A, so both
sentences were corrected.
