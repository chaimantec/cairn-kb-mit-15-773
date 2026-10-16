# Build log — 15.773 knowledge base

The per-lecture record of how this knowledge base was built and checked: each
deck's licence notices and lecture pointers, each transcript's edit and checks,
what each figure audit found, each deck's printed slips, and which slides were
rendered as images and why the others were not. It is a build record, **not
needed to answer questions**: every fact a learner needs is on the wiki page,
slide file or transcript it concerns. The rules every build follows are in
[`AGENTS.md`](AGENTS.md).

One section per run: lectures 1, 2 and 3 (built 2026-10-11), lecture 4 (2026-10-14), lecture 5
(2026-10-14), lecture 6 (2026-10-14), lecture 7 (2026-10-14), lecture 8 (2026-10-16), lecture 9 (2026-10-16), lecture 10 (2026-10-16), and at the end the ticked build checklists of those runs,
moved from `TODO.md`.

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

## Lecture 3 — Keras and TensorFlow, Training the Heart-Disease Model, and Computer Vision Basics

Built 2026-10-11. Video `8QuyDcMIdRc` (77 min); decks `mit15_773_s24_lec03a.pdf` (deck 3A, 44 pages)
and `mit15_773_s24_lec03b.pdf` (deck 3B, 29 pages).

**Video ↔ deck.** The recording presents two decks. Deck 3A runs from 0:45 (slide 2, the recap that
repeats deck 02's slide 94) to the Colab pointer and instructions (slides 41–43, ≈27:52–29:25). This
confirms lecture 2's note that deck 3A opens video 3. The heart-disease Colab, which is not in any
deck, runs 29:25–58:08. Deck 3B runs from 58:08 (slide 2) to its Colab pointer (slide 28, ≈1:13:36),
and the video ends at 1:16:41 partway into the Fashion-MNIST Colab, before any model is trained. The
catalog's "Building Convolutional Neural Networks from Scratch" is not reached. Deck 04's page 2 text
recaps Fashion MNIST with a single-hidden-layer network ("85% +"), consistent with deck 04 opening
video 4; confirm when building lecture 4. Neither Colab notebook is published on OCW (`sources.md`
has none), so the wiki describes the code from the narration only.

**Numbering.** Both decks: slide N = PDF page N. Deck 3A is 1:1 (pages 2–43 print their own number;
1 and 44 none). Deck 3B has a hidden slide 16: pages 2–7 and 9–15 print their own number, pages
16–19 and 27–28 print N+1, and pages 1, 8, 20–26 and 29 print none. `slide_number_map.py` read deck
3B correctly this time, apart from labelling the pages that print nothing as inferred. The raw corner
spans are clean on both decks, with no split spans or stale numbers. Heading checks 1..44 and 1..29
pass. Each deck's last page is OCW's end page.

**OCW notices (8, all in deck 3B):** slides 3 and 4 (grayscale intensity matrix © fast.ai), 6 (© unknown,
the RGB matrices and photograph), 8 (© unknown, dog and cat photographs), 9–12 (sheep image © Nirmala
Murali on Medium). Deck 3A has none. The agents found no drawn notices. Deck 3B's slide 14 carries a
CC BY-SA credit (Yuzamei, Wikimedia Commons), not an exclusion. Unmarked third-party content in deck
3A: slide 2 (Chollet Figure 2.26, page 61), slides 24–25 ("Image credit: fast.ai"; they reuse deck
3B's excluded fast.ai matrix and RGB matrices), slide 32 (Chollet, page 70), and slide 43 (Colab
screenshots).

**Transcription.** Sonnet, one agent per deck in parallel. Neither was interrupted. One correction:
the deck 3B agent identified slide 6's photograph as "the MIT Great Dome", which the slide does not
print. It was rewritten to the printed filename, `killian.jpeg`, and the lecturer's "a picture of
Killian Court".

**Transcript edit.** OCW human captions, with `AUDIENCE:` labels. One restoration: "by sending a
random seed" → setting (31:42), beside `set_random_seed`. `[Ed:]` notes at 11:32 ("the training data"
for the held-out error), 15:20 ("Teras and KensorFlow", self-corrected), 21:42 ("9 values. 0 … 9"),
45:41 (193 against slide 10's 194), 49:35 ("crossed the int", unclear), 55:50 (the proxy relation
stated backwards; 56:36 has it the right way round) and 1:16:41 ("9:55" is the time of day). Checks: 100 timestamps
identical, number inventory identical, paragraph word ratios 1.000 throughout once the notes are
removed.

**Figure check.** Done in the parent on 1400px renders: deck 3A slides 10, 15, 17 and 18, and deck 3B
slides 16 and 18, all agreed. Deck 3B slide 6's three intensity tables, which carry a notice and are
not rendered, were checked on a scratch crop that was not committed. All ten rows of the blue table
and the visible rows of red and green agreed. The two agents also cross-checked each other without
knowing it: deck 3A's slide 25 shows the same RGB matrices, and both read the same opening rows.

**Printed slips.** Deck 3A: slide 10 gives the training set as 194, while the lecturer says 193
twice in the Colab (both make 7 batches); slide 17's x-axis label wraps as "Iteration" / "s";
slide 36 prints curly quotes in the code; slide 40 prints "adam" with mismatched quotes; slide 43's
asterisk has no footnote. Deck 3B: slide 14 promises "over 90%", while the lecturer says "close to
90%".

**Images: 13.** Deck 3A, 9 of 44 pages as `3a-N`: 6, 9, 10, 15, 17, 18, 32, 36, 43. Not rendered:
slide 2, the same figure as deck 02's slide 94 and linked to that render instead; build steps
superseded by a rendered slide (5, 7, 8 → 6 and 9; 12–14 → 15; 31 → 32); slides 24–25, which reuse
deck 3B's excluded figures; and text-only slides. Deck 3B, 4 of 29 pages as `3b-N`: 14, 16, 17, 18.
Not rendered: the eight noticed slides (3, 4, 6, 8–12); 15 → 16; the tables on 19–27, which the slide
file reproduces exactly; and dividers and text slides. 1.41 MB. All 13 are in the wiki lecture page.
They were placed by `embed_slide_images.py` run once per deck on that deck's half of the page. 3a-6
and 3b-18 were then moved below the displayed formulas their anchor sentences introduce, and every
caption was hand-written.

**Wiki.** A lecture page and five new topic pages (overfitting-and-regularization, tensors,
data-preprocessing, images-and-vision-tasks, multiclass-classification). keras, training-neural-networks,
gradient-descent, loss-functions, activation-functions and neural-network-anatomy were extended.
Sentences in lectures 1–2, activation-functions, keras, loss-functions and gradient-descent that
deferred softmax, the Keras introduction or epochs and batches "for later" now point at lecture 3.

## Lecture 4 — Convolutional Neural Networks and Transfer Learning

Built 2026-10-14. Video `8xh7Y0pBrCE` (76 min); deck `mit15_773_s24_lec04.pdf`, 84 pages.

**Video ↔ deck.** The recording presents deck 04 alone. It opens at slide 2 (the Fashion MNIST
recap, 0:46) and reaches slide 81 ("Back to Colab") at ≈1:10:54, then runs the transfer-learning
Colab and the webcam demo to the end (1:15:32). The appendix (slides 82–83) is pointed to at 27:18
and not presented; slide 1 is the title and 84 OCW's end page. This confirms lecture 3's note that
deck 04 opens video 4. The catalog title ("Transfer Learning and Fine-Tuning; Intro to HuggingFace")
understates the recording, whose first hour builds CNNs from scratch. Material outside the deck:
the fast.ai convolution spreadsheet (12:29–17:13), a board example (39:04–40:37) the camera does not
show, the Notability explanation of pooling (could not be shown; a walkthrough was promised), and
data augmentation, in the Colab only (1:03:11–1:06:15). The lecture assumes a Fashion-MNIST
walkthrough video posted the day before, not on OCW.

**Numbering.** Slide N = PDF page N, 1:1. `slide_number_map.py` read it correctly. Pages 2–77 and
81–83 print their own number. Pages 1, 78–80 (hub screenshots) and 84 print none. The raw corner
spans are clean, with no split spans or stale numbers; the only extra span in the corner region was
a "-1" belonging to slide 12's filter figure. Heading check 1..84 passes.

**OCW notices (14):** 41 (Lee, Grosse, Ranganath, Ng), 52–55 (Alshuwaier, Areshey, Poon; CC
BY-NC-ND, still "excluded from our Creative Commons license"), 58 (© unknown, handbag and shoe
photos), 66 (© unknown, ImageNet graphic), 67–69 (Zeiler and Fergus/Springer), 70–73 (He, Zhang,
Ren, Sun/arXiv). The agents found no drawn notices. Slide 2 carries a CC BY-SA credit (Yuzamei,
Wikimedia Commons), not an exclusion. **Slide 77 prints no notice but redraws the ResNet 34 diagram
of 70–73**, so it is treated as excluded. Unmarked third-party content: slides 30–32
("Image source: deeplearning.ai"), 40 (a photograph of Fig 14-2 of *Hands-On Machine Learning*,
linked on Amazon), 47 (CS231n notes), 78–80 (hub screenshots). The Colab links on slides 56, 59 and
81 point to Google Drive notebooks through PDF link annotations; their targets are not recorded,
following lecture 3.

**Transcription.** Sonnet, two agents in parallel (pages 1–42 and 43–84) writing scratch files that
the parent spliced under a hand-written header. Neither was interrupted. The parent normalized
`<span style="color:red">` markup to plain text with a parenthetical note, and replaced the
recorded Colab URLs with "target not printed".

**Transcript edit.** OCW human captions, `AUDIENCE:` labels and one `SIRI:`. Done inline in the
parent as a scripted list, since the parent had already read the whole transcript to map it onto
the deck. Restorations: "that tree" → three (13:16), "image nest" → image-ness (1:00:50). `[Ed:]`
notes at 0:46 ("did exploit" for did not), 2:17 and 15:40 (unclear), 9:21 ("three numbers" for a
3-by-3 patch), 28:09 and 58:32 (times of day), 45:15 ("MNIST" for Fashion MNIST), 54:38 ("structured
data … best game in town", probably "unstructured"), 58:32 ("224 to 24"), 1:07:49 ("dimension data
set") and 1:12:26 ("7 by 7 by 224, 2,048", self-corrected). Checks: 98 timestamps identical, number
inventory identical, paragraph word ratios 0.993–1.000 once the notes are removed.

**Figure check.** Done in the parent on 1400px renders: slides 21, 31, 40, 47, 75 and 80. Five agreed.
Slide 40's description of the dashed projections was muddled (it put a dashed rectangle on layer 2)
and was rewritten. Arithmetic checks: slides 15–21's output (−30 → 0 for the first three rows, 30
for the last two), slide 44's max pooling (43, 109, 105, 35), slide 45's averages (22.25 printed as
22.2, 45.5, 70.75, 21), slide 47's max pooling (6, 8, 3, 4), and slide 70's ResNet 34 box counts
(6 + 8 + 12 + 6 convolutions, plus the first convolution and the fc layer, make 34). The lecturer's
spoken 43, 109, 105, 35, 22.2 and 45.5 agree with the slides.

**Printed slips.** Slide 8 prints "rather learn it separately". Slide 41 prints "Lee at al". Slide
45 prints 22.2 for 22.25. Slide 2 says "85% +" where the lecturer says the high 80s and 88%. Slide
26's title drops slide 25's italics; slides 72–73 and 75 drop the asterisk on "ResNet 34"; slide
70's title overflows its band. The deck pictures ResNet 34, while the Colab's narration gives 23
million parameters and a 7 × 7 × 2,048 output without naming the variant; the wiki says only that.

**Images: 16 of 84 pages.** Rendered: 14, 18, 19, 21, 24, 29, 31, 32, 40, 44, 45, 47, 75, 78, 79, 80.
Not rendered: the 14 noticed slides and slide 77 (reuses the excluded ResNet diagram); build steps
superseded by a rendered slide (11, 12, 23 → 14 and 24; 15–17 → 18; 20 → 21; 25–28 → 29; 30 → 18 and
31; 46 → 45); slide 2, whose mosaic is lecture 3's `3b-14.jpg` and is linked to it; and text-only
slides and dividers. 1.71 MB, 8 PNG and 8 JPEG. All 16 are in the wiki lecture page: 14 placed by
hand and 78–79 by `embed_slide_images.py`, all with hand-written captions. The convolution steps
were written as bold-led paragraphs rather than a numbered list, so that images 18 and 19 could sit
between the steps without breaking a list.

**Wiki.** A lecture page and three new topic pages (convolutional-neural-networks, transfer-learning,
data-augmentation). images-and-vision-tasks, representation-learning, overfitting-and-regularization,
tensors, keras and neural-network-anatomy were extended. Sentences in lectures 1 and 3,
images-and-vision-tasks and overfitting-and-regularization that deferred CNNs or dropout to lecture 4
were corrected: lecture 4 uses a dropout layer but defers its explanation again, to "early next week"
(≈1:12:26). The lecture page hand-checks the Fashion-MNIST CNN's 302,000 parameters (302,026 with
2 × 2 kernels, no padding and pooling rounding down), labelled as the page's own arithmetic.
`SEE_ALSO.md`'s 6.7960 entry was updated to that KB's current coverage (lectures 1–19) and now
points to its convolution, transfer-learning and data-augmentation pages.

## Lecture 5 — Text Vectorization and the Bag-of-Words Model

Built 2026-10-14. Video `duBLxHjaecQ` (77 min); deck `mit15_773_s24_lec05.pdf`, 67 pages.

**Video ↔ deck.** The recording presents deck 05 whole: slide 1 (the roadmap of the NLP lectures,
0:00) to slide 66 (Colab, ≈45:08), then runs the Colab to the end (≈45:53–1:16:18). Slide 67 is
OCW's end page. This confirms lecture 4's note that deck 05 opens video 5. The catalog title
("Deep Learning for Natural Language – The Basics") is accurate. Material outside the deck: the
Gemini 1.5 Pro aside (≈6:57–9:19); the song classifier drawn on the board (slide 64 says
"Blackboard"; slide 65 is the drawing); dropout, explained at ≈1:11:38–1:14:44 with no deck-05
slide (its slide is deck 3A's 18); the whole Colab. Byte-pair encoding is promised "at the end"
(≈23:19) and deferred (≈1:14:44); its slides are deck 09's 58–60, found by a text-layer grep of all
decks.

**Numbering.** Slide N = PDF page N, 1:1. `slide_number_map.py` read it correctly and `--verify`
passes. Pages 1, 15–17, 19–21 and 67 print no number; the corner spans are otherwise clean.

**OCW notices (2):** 15 ("Various AI logos © respective copyright holders", with the LMSYS Chatbot
Arena leaderboard) and 17 (Einstein GPT © Salesforce). The agents found no drawn notices. Unmarked
third-party content: slide 3 (Google autocomplete screenshots), 4 (a ChatGPT exchange), 16 (the
Y Combinator W23 Generative AI Landscape, "Created by your friends at Truewind"), and 63 (eight
lines of song lyrics). The Colab link on slide 66 is a PDF link annotation; its target is not
recorded.

**Transcription.** Sonnet, two agents in parallel (pages 1–33 and 34–67) writing scratch files that
the parent spliced under a hand-written header. The second agent was stopped by an output content
filter after writing slides 34–50, almost certainly on slide 63's lyrics, which its prompt had
said it could copy from the text layer. A third agent wrote 51–67 with the instruction to describe
the lyrics and not reproduce them. Nothing was lost: the first two agents appended chunk by chunk.
The parent changed slide 1's figure line to "no" (a logo only), and added notes on text the
rendered page hides: slide 37's table header ("Token, Integer, Encoding"), its `<UNK> 0*` row and
"aardvark", and the $f(x, w)$ inside slides 22–25's box. Both exist only in the text layer.

**Transcript edit.** OCW human captions, `AUDIENCE:` labels and `[MUTED]`, done inline in the parent
as a scripted list. Restorations (11 replacements): "routed" → routing (4:36, slide 7); "marketing,
copy sales, emails" → marketing copy, sales emails (5:23, slide 10); "the cats are on the mat" → the
cat sat on the mat (32:45); "order them" → OR them (33:31, slide 57); "text vectorization array" and
"textualization layer" → text vectorization layer (45:53, 46:40); "singlular" (41:59); three
"one hot-encoded" hyphenations (54:25); "Jeff" → Geoff Hinton (1:13:58, as in lecture 3's captions).
`[Ed:]` notes at 8:33 ("Kalimpong", the language's name as captioned and not in the deck), 20:11,
27:09 ("zero-hot"), 27:54, 34:18 ("two 2s" for two 1s), 37:25 ("the other 50,000" for about
450,000), 59:51 ("D-E-A"), 1:01:24 (twice: "Keras' write input"; "a 50 max_tokens long vector"),
1:03:44 (baseline "around 50%" against 55% at 53:38), 1:06:51 ("are going to show up", where the sense
needs "are not") and 1:16:18 ("Keras" for KerasTuner). Checks: 99 timestamps identical, number
inventory identical once notes are removed, paragraph word ratios 1.000–1.007.

**Figure check.** Done in the parent on 1400px renders: slides 21, 37, 40, 50, 56 and 65. All six
agree with the slide file. Slide 37 first looked wrong, because the text layer prints a header and
an `<UNK>` row that the agent omitted, but the render shows them hidden; the agent had described
the visible page correctly. Slide 56's count and multi-hot rows were checked cell by cell (1s under
cat, mat, on and sat; 2 and 1 under "the"). The lecturer's 40,008 parameters (5,000 × 8 + 8) are
consistent with the narration.

**Printed slips.** Slides 19–21 print "Hand-crated". Slide 29 lists "eaten" twice. Slides 49, 51
and 52 call the sentence's table 6 x 100 and slides 53–55 and 57 call it 8 x 100. Slide 7's box
says "LLM" where slides 8–14 say "NLP". Spoken: the rock baseline is 55% at 53:38 and "around 50%"
at 1:03:44.

**Images: 7 of 67 pages.** Rendered: 21, 27, 43, 47, 50, 56, 65 (0.71 MB; 3 PNG, 4 JPEG). Not
rendered: the two noticed slides (15, 17); slides 3, 4 and 16, unmarked third-party screenshots and
logo graphics of the kind OCW excluded on 15 and 17, whose text the slide file transcribes; slide 63
(copyrighted lyrics, described only); the single-box diagrams 6–14 and 22–25; build steps
superseded by a rendered slide (19–20 → 21; 28, 34, 35, 44 → 27; 29–32 and 36–42, whose pipeline
icon is 27's and whose vectors are 43's; 46, 48, 49 → 47; 51, 52 → 50); slides 30, 32, 36 and 37,
examples and tables the slide file reproduces exactly; and text-only slides and dividers. All seven
are in the wiki lecture page, hand-placed with hand-written captions; slides 21, 27, 43 and 56 also
appear on the new topic pages.

**Wiki.** A lecture page and three new topic pages (natural-language-processing,
text-vectorization, bag-of-words). Extended: overfitting-and-regularization (dropout explained; a
short input keeps parameters down), keras (layers for text), multiclass-classification (softmax
rather than the maximum), data-preprocessing, training-neural-networks (hyperparameter choice,
KerasTuner), loss-functions, activation-functions and representation-learning. Sentences in
lectures 1, 3 and 4 and INDEX.md that left dropout unexplained now point to lecture 5. The lecture
page totals both song classifiers' parameters (40,035 and 160,035) as its own arithmetic and names
its assumptions. `sources.md`'s deck table had kept decks 3A, 3B and 04 "provisional" after lectures
3 and 4 confirmed them; those rows and the paragraph above them were corrected with deck 05's.

## Lecture 6 — Word Embeddings

Built 2026-10-14. Video `LqFc0z-pQTg` (78 min); deck `mit15_773_s24_lec06.pdf`, 70 pages.

**Video ↔ deck.** The recording presents deck 06 from slide 2 (the recap of one-hot vectors, 0:49)
to slide 68 (the overall flow in Keras, ≈55:09), then runs the Colab to the end (≈55:09–1:17:33).
Slide 1 is the roadmap, not discussed; slide 69 is the Colab divider and 70 OCW's end page. This
confirms lecture 5's note that deck 06 opens video 6. The catalog title ("Deep Learning for Natural
Language – Embeddings") is accurate. Material outside the deck: the iPad drawing of two
co-occurrence rows, apple and banana (≈29:28–34:11); the worked "deep"/"learning" example on the
board (≈37:19–39:45); the double-descent aside (≈41:17); and the whole Colab (frozen GloVe 63%,
GloVe trained further almost 69%, 64-dimensional embeddings from scratch 72%). The lecturer does
not say which split the accuracies are on; the wiki page says so.

**Numbering.** Slide N = PDF page N, 1:1. The corner spans were clean (no split or stale numbers),
`slide_number_map.py` read them correctly and `--verify` passes. Pages 1 and 70 print no number.

**OCW notices (6):** 14, 15, 17, 18 and 19 ("Embeddings quizzes © Cohere") and 47 ("One-hot word
vectors and word embeddings figures © François Chollet"). The parent's text-layer scan found all
six, but the brief to the first agent listed only 14, 15, 17 and 18, because the list was retyped
by hand; the agent found slide 19's typed notice itself. `AGENTS.md` now says to copy the script's
page list. The agents found no drawn notices. Unmarked third-party content: slides 48–51, a plot of
GloVe vectors for ten male/female word pairs that the deck does not credit (rendered; see Images).
The Colab link on slide 69 is a PDF link annotation; its target is not recorded.

**Transcription.** Sonnet, two agents in parallel (pages 1–35 and 36–70), reading 1400px PNGs the
parent pre-rendered into the scratchpad rather than PDF pages, and appending to scratch files that
the parent spliced under a hand-written header. Both finished. Parent fixes: slide 2's footnote,
whose text layer ends in a stray backslash, escaped so it does not become a markdown line break;
the agent's note about slide 19's notice removed; slide 37's last handwritten matrix label, which
the agent read as "book", marked hard to read. Hidden text noted by the agents: slides 12–13 hold
later bullets in the text layer, slide 16's text layer says "geometric distance" and "semantic
distance" where the render says "relationship", and slide 53's text layer stops at "fairly gen".

**Transcript edit.** OCW human captions with `AUDIENCE:` labels, done inline in the parent as a
scripted list. Restorations (8): "So FR" → So far (0:00, slide 2); "just do or" → OR (2:21, as
lecture 5); "inferrential" (12:23); "deep end learning" → deep and learning (23:20); "bi, by" → bi,
bj (25:38, slides 41–43); "B2" → b2 (38:06); "incoming centers" → sentences (51:18, slide 58); "GloVe
assist" → GloVe as is (1:16:00). `[Ed:]` notes at 3:54 ("the SD"), 4:42 (a student's name), 21:03
("We went with word", meaning GloVe), 23:20 ("under embedding"), 1:12:53 and 1:16:47 (unclear).
Checks: 101 timestamps identical, number inventory identical once notes are removed, paragraph word
ratios 1.000–1.007. No restored proper noun needed a deck check beyond John Firth (slide 28) and
GloVe/word2vec, which the captions already spell correctly.

**Figure check.** Done in the parent on 1400px renders: slides 37, 45, 48, 59, 63, 67 and 68. All
agree with the slide file. Slide 45's two equations are exact; slide 48's twenty word positions
were read against the axes; slide 59's indices (2 9 7 14 17; 23 62 5 0 0) and slide 63's row
colours match. The one uncertainty is slide 37's handwritten label, now marked as such. The
lecturer's parameter counts (500,835 total, 835 trainable) check by arithmetic: $5{,}000 \times 100$
plus $100 \times 8 + 8$ plus $8 \times 3 + 3$.

**Printed slips.** Slide 59 gives "love" the index 62; slides 62–68 give it 9 in the same sentence.
Slides 22 and 23 print "the the". Spoken: the training set is "roughly 49,000" songs at 55:55 and
"50,000" later, a rounding.

**Images: 8 of 70 pages.** Rendered: 37, 49, 51, 55, 59, 61, 67, 68 (0.8 MB; 1 PNG, 7 JPEG). Not
rendered: the six noticed slides (14, 15, 17, 18, 19, 47); slide 3, whose handwritten one-hot table
is the same embedded raster as lecture 5's slide 50 (found by hashing the decks' images), and slide
2, its build step; build steps superseded by a rendered slide (48, 50 → 49, 51; 56, 60 → 55; 62–65
→ 68, which assembles 62, 63, 64 and 67); slides 57–58, code screenshots the slide file reproduces
exactly; and text-only slides and dividers. Slides 49 and 51 are the uncredited GloVe plot, rendered
under the lecture 2 and 4 precedent for unmarked academic figures, with `LICENSE.md` saying the deck
does not credit it. All eight are in the wiki lecture page, hand-placed with hand-written captions;
slides 37 and 51 also appear on the new word-embeddings topic page.

**Wiki.** A lecture page and one new topic page (word-embeddings). Extended: text-vectorization
(integer output and padding), bag-of-words (lecture 6's results beside lecture 5's), keras (a
"Layers for embeddings" section), natural-language-processing (stand-alone and contextual
embeddings; coverage now lectures 5 and 6), representation-learning, transfer-learning (pretrained
word embeddings), gradient-descent (fitting a model that is not a network) and
overfitting-and-regularization (noise without a test set); lecture 5's page now links forward. The
lecture page checks the frozen model's 835 trainable parameters and counts the from-scratch model
(320,547) as its own arithmetic, naming its assumptions. `sources.md`'s row for deck 06 is no longer
provisional.

## Lecture 7 — Transformers

Built 2026-10-14. Video `IeF7aATDaw4` (76 min); deck `mit15_773_s24_lec07.pdf`, 78 pages.

**Video ↔ deck.** The recording presents deck 07 from slide 2 (transformers across domains, 0:00) to
slide 71 (≈1:01:01), opens the Colab at ≈1:02:34, goes back to slides 72–76 at ≈1:08:01 ("I forgot
about doing this"), and returns to the Colab at ≈1:11:06 until the end. Slide 1 is the roadmap,
calling the lecture "Transformers (1/2)"; slide 77 is the Colab divider and 78 OCW's end page. The
catalog title ("Deep Learning for Natural Language – Transformers") is accurate. Material outside the
deck: three iPad drawings, the dot product as lengths times cosine (≈28:42–31:00, behind slide 40,
which prints only "iPad"), concatenate and project (≈41:44–44:47), and the feed-forward ReLU layer four
times as wide (≈45:34–46:20); and the Colab, which sets up the ATIS slot-filling model (888-token
vocabulary, 125 output tags, 512-dimensional embeddings, five heads, `dense_dim` 64, about 5.3
million encoder parameters) but stops before training, at 9:45 by the lecturer's clock.

**Numbering.** Slide N = PDF page N. The corner spans are clean, one span per page at the same
position: pages 2–9 print 2–9 and pages 10–77 print 11–78, so a hidden slide 10 was not exported.
`slide_number_map.py` read this correctly (one absent number, 10); the headings were checked with a
plain 1..78 sequence check. Pages 1 and 78 print none.

**OCW notices (8):** 23 (the paper's title block and Figure 1, the transformer architecture), 51 (two
attention heads on "The Law will never be perfect …") and 64–69 (the paper's encoder block; 65–69
share one raster), all "Attention is All You Need" © Vaswani, Shazeer, Parmar, Uszkoreit, Jones,
Gomez, Kaiser, Polosukhin. The script's page list went to the agents verbatim; they found no drawn
notices. Unmarked third-party content: slides 21–22, screenshots of Google search results from
Google's BERT blog post (not rendered, following lecture 5's slides 3, 4 and 16). The raster that
slides 53–56 and 62–63 share is a picture of the course's own slide-52 diagram, checked on the
render, not a copy of the paper's figure. The Colab link on slide 77 is a PDF link annotation; its
target is not recorded.

**Transcription.** Sonnet, two agents in parallel (pages 1–39 and 40–78), reading 1400px PNGs the
parent pre-rendered into the scratchpad, appending to scratch files that the parent spliced under a
hand-written header. Both finished. Parent fixes: slide 17's empty Word cell noted as holding a hidden
"see" in the text layer; slide 26's "rendering artefacts" remark on three textured cells reworded;
slide 40's placeholder note replaced with what the video shows; slide 59 now says its table renames
"sit" to "sat" (the agent wrote "every other value is unchanged"); slide 51's line-by-line account of
the attention panels cut back to what the render settles.

**Transcript edit.** OCW human captions with `AUDIENCE:` labels, done inline in the parent as a
scripted list. Restorations (12 strings): "Bert" → BERT (13:55, twice; 14:40; slides 21–22's URL);
"glove" → GloVe (11:36, 47:05, 50:58, 1:11:06; slides 16–17); "these links" → lengths (29:28); "the
eighth test data set" → ATIS (1:02:34; slide 7); a full stop (17:02); two unbalanced quotations (4:38,
50:58). `[Ed:]` notes (11) at 26:19 ("added as H2"), 31:00 ("Input."), 48:38 ("dot reduction"), 52:31
(spoken "0.571", "1.33.9"), 53:18 ("0.681"), 57:55 (the self-contradicting punctuation sentence),
1:03:20 ("raw box", presumably Dropbox; "top view"), 1:07:14 ("uncompared", the garbled answer to 125
vs 123) and 1:08:01 ("0, 0" for the slot label O). Checks: 99 timestamps identical, number inventory
identical once notes are removed, paragraph word ratios 0.992–1.000. Restored names checked against
the deck: GloVe and ATIS print there; BERT only in slides 21–22's URL.

**Figure check.** Done in the parent on 1400px renders: slides 1, 33, 42, 51, 53, 61, 64 and 75. All
agree except the two fixes above (59's renamed word, found on 61's render; 51's over-specific line
description). The positional-encoding tables of slides 58–61 also match the text layer cell for cell,
and their sums check by arithmetic.

**Printed slips.** Slide 58 lists "sit" where slides 59–61 have "sat" with the same numbers. Slide 17's
first row shows no word (the text layer holds "see"). Slides 72–76 cut "B-toloc.city_name" at the
right edge. The slides and narration attach a 123-way softmax per word; the Colab's output vocabulary
has 125 entries. Spoken: "53 million — sorry, 5.3 million"; a dataset query read as "8:38 AM" and
then "8:30 AM".

**Images: 7 of 78 pages.** Rendered: 33, 35, 43, 47, 49, 53, 75 (0.75 MB; 5 PNG, 2 JPEG). Not
rendered: the eight noticed slides (23, 51, 64–69); slides 21–22 (Google screenshots); slide 11, a
list of slot labels transcribed in the slide file; build steps and repeats superseded by a rendered
slide (26–32 → 33; 34 → 35; 36, 41, 42 → 43; 44–46 → 47; 48 → 49; 52, 54–56, 62, 63 → 53; 72–74, 76 →
75); the positional-encoding tables (58–61), reproduced exactly; and text-only slides and dividers.
All seven are in the wiki lecture page, hand-placed with hand-written captions; slides 43 and 53 also
appear on the new transformers topic page. **For lecture 8:** deck 08 reuses deck 07's rasters on
its slot-filling pages (07's 73–76) and its page-43 images, by hash; check the repeats before
rendering.

**Wiki.** A lecture page and one new topic page (transformers). Extended: word-embeddings
(contextual embeddings by self-attention), natural-language-processing (the encoder and ATIS slot
filling; coverage now lectures 5–7), multiclass-classification (the softmax in attention and per word),
text-vectorization (standardization off for labels), keras (a "Layers for transformers" section),
representation-learning, convolutional-neural-networks; lecture 6's page now links forward. The
lecture page counts the encoder's 5,319,232 parameters as its own arithmetic, under named
assumptions (query, key and value projections of 512 per head, which lecture 8 introduces; a 512 →
64 → 512 feed-forward block; two layer normalizations), matching the lecture's "5.3 million", and
the whole model's 5,871,037, which the lecture does not report. `sources.md`'s row for deck 07 is no
longer provisional.

## Lecture 8 — Transformers (2/2) and Self-Supervised Learning

Built 2026-10-16. Video `v-lHsawHyaI` (76 min); deck `mit15_773_s24_lec08.pdf`, 99 pages.

**Video ↔ deck.** The recording reviews slides 2–11 (0:00–3:53), then spends ≈3:53–5:25 on lecture 7's
Colab, reporting the trained result without re-running it (10 epochs, batch size 64, 99% test accuracy,
93% on the non-O slots). It resumes at slides 12–13 and moves to an iPad at ≈7:46. The iPad covers
self-attention as softmax of X X-transpose times X on "I love HODL", then the tunable version with AK,
AQ and AV, and the paper's formula, shown from the paper itself at ≈22:31. That runs to ≈24:08 in
place of slides 15–19, "from last year", which the lecturer says to read afterwards. Slides 20–93
follow in order, with the Hugging Face Colab at slide 87 (≈59:53–1:06:53), and the lecture ends on
slide 93 at 1:16:13. The "Contrastive Learning (time permitting)" section, slides 94–98, is not
reached. Slide 1 is the roadmap ("Lecture 8: Transformers (2/2)"); 99 is OCW's end page. The catalog
title is accurate. Material outside the deck: the iPad drawings and both Colabs. Slide 15 points to a
handout, "HODL-SP24-Section-A-The Self-Attention Layer.pdf", that is not on OCW.

**Numbering.** Slide N = PDF page N. Every page that prints a corner number prints its own page
number (one span each, no splits); 38 pages print none. `slide_number_map.py` reported 1:1 correctly,
and `--verify` passed on the finished file.

**OCW notices (15):** 28 ("Attention is All You Need"), 41 and 56 (ResNet-34, He et al.), 50 ("ImageNet
training image © unknown", the Zeiler and Fergus feature grids), 51 (Lee et al.), 79–83 (BERT paper and
figures, Devlin et al.; 79 also credits Jay Alammar's BERT cartoon, CC BY-NC-SA), 90 (Vision Transformer,
Dosovitskiy et al.), 93 (desk chair, Wayfair LLC), 96–98 (SimCLR dog images, Chen et al.). The script's
list went to the agents verbatim; they found no drawn notices. **Slide 81's notice names a "BERT
figure", but the page shows only the course's own `<CLS>` diagram from slide 38 under a new title**,
checked on the render. Its large rasters sit at the same positions as slide 38's but are re-encoded.
The hashes that slides 35–38 share with 81 and 93 belong to solid black shape masks (extracted and
viewed), not to a figure. So 81 keeps its notice and is not rendered, and slide 38, which has no
notice, is rendered. Unmarked third-party content: 84 (Sentence-Transformers logo) and 86 (Hugging
Face Hub screenshot), not rendered, following lectures 5 and 7. Slide 91 prints "Figure license: CC0
1.0" for the TabTransformer figure, which is rendered and attributed. The "Huggingface Colab" link on
slide 87 is a PDF link annotation; its target is not recorded.

**Transcription.** Sonnet, three agents in parallel (pages 1–33, 34–66, 67–99), reading 1400px PNGs
pre-rendered into the scratchpad and appending to scratch files that the parent spliced under a
hand-written header. All three finished in about two minutes each. Parent fixes: "w_1 to w_6", written as
math, lost both symbols on slides 26 and 28, and slide 29's A^K lost its opening dollar sign, to an
unquoted shell heredoc (lesson in `AGENTS.md`); slide 20's claim that its sentence is cut off at the right edge was
wrong; slide 93's review text, which the agent quoted while calling it unreliable, was confirmed on a
4× crop and given in full; slide 81's entry notes the notice-content mismatch; footnote asterisks were
escaped and slide 82's heading puts `<CLS>` in backticks so github.com does not swallow it as a tag.

**Transcript edit.** OCW human captions with `AUDIENCE:` labels, done inline in the parent as a
scripted list. Restorations (12 strings): GPT3 → GPT-3 (2:21, 35:52) and GPT4 → GPT-4 (1:15:26), as
lectures 2, 3 and 7's captions spell them; "transfer" → transpose (10:08); "attention rates" →
attention heads (27:17); a stray apostrophe (48:59); six missing full stops. `[Ed:]` notes (8) at
8:38 ("W1s, Ws-2, W3"), 18:40 ("by 1", presumably V), 19:28 ("50%"), 30:23 ("under 100"), 36:38
("normalized", presumably optimized), 41:17 ("connects", presumably converts), 57:36 (BERT's
feed-forward size: said 4,096 for H = 768; slide 80's footnote gives 3072, and 4096 for H = 1024) and
1:13:08 ("correct", presumably collect). Checks: 99 timestamps identical, number inventory identical
once notes are removed, paragraph word ratios 1.000 throughout. No proper noun was restored.

**Figure check.** Done in the parent on renders and crops: slides 18 (the superscript K and Q on
bold A confirmed), 20 (the cut-off claim, wrong), 24, 29 (the list), 47 (outline colours confirmed), 69, 80 (the
BERT sizes and footnote 3, confirmed), 81 (notice over the course's diagram), 86 (525,416 confirmed), 91
and a 4× crop of 93.

**Printed slips.** Slide 70 "We then use train"; slide 72 a doubled closing quote; slide 92 "turns out
to very useful"; slide 91 "The Tab Transformer: :"; slide 51 "Lee at al"; slide 81's notice; slide 29
cuts "B-toloc.city_name" and slides 77–78 "management" at the right edge. Spoken: "4,096" for
BERT-Base's feed-forward layer; "125 possibilities" per word (lecture 7's slides say 123; its Colab has
125); "100 examples" and "200 examples" for handbags and shoes.

**Images: 19 of 99 pages.** Rendered: 8, 21, 23, 25, 27, 30, 35, 38, 45, 47, 60, 62, 63, 69, 70, 71,
74, 77, 91 (2.2 MB; 11 PNG, 8 JPEG). Not rendered: the fifteen noticed slides; 84 and 86
(third-party logo and screenshot); repeats of earlier lectures' rendered figures, found by raster
hash and checked by description: 10 (lecture 7's 75), 13 (lecture 7's 43), 29 (lecture 7's 76 plus a
transcribed list) and 42, 43, 57 (lecture 4's 75); build steps and repeats superseded by a rendered
slide (4–7 → 8; 14 → 21; 22, 24, 26 → 27; 31–32 → 30; 34, 36–37 → 35; 46 → 47; 58–59 → 60; 61 → 62;
75 → 63; 76 → 71; 78 → 77); 9, 11 and 33, single-box encoder drawings; text-only slides and dividers.
The wiki lecture page hand-places 17 of the 19 with hand-written captions (not 62 and 70, which sit in
the slide file beside 63 and 71). Lecture 8's 27 is also on the transformers topic page, 47 on
representation learning, 63 on transfer learning, and 69 and 77 on the new self-supervised-learning
page.

**Wiki.** A lecture page and one new topic page (self-supervised learning). Extended: transformers
(matrix form, keys, queries and values, residual connections and layer normalization, `<CLS>`, BERT,
beyond text; its "lecture 7 only" sentence rewritten), transfer learning (why it works, the general
recipe, pretrained language models), representation learning, natural language processing (coverage
now lectures 5–8), data preprocessing (layer normalization), images and vision tasks (patches), data
augmentation (contrastive learning, slides only), Keras, word embeddings, convolutional neural
networks; lecture 7's page now links forward in four places, including the trained result. The
lecture page notes, without recomputing, that lecture 7's parameter count assumed 512-to-512 value
matrices, while lecture 8 says value matrices are often smaller. `sources.md`'s row for deck 08 is no
longer provisional.

## Lecture 9 — Large Language Models (1/2)

Built 2026-10-16. Video `KGDe1QvfKJ8` (75 min); deck `mit15_773_s24_lec09.pdf`, 68 pages.

**Video ↔ deck.** The recording presents slides 2–66 in order, from the masking review (0:00) to the
token-visualizer demo of slide 66 (≈1:10:54–1:14:30). Slide 1 is the roadmap ("Lecture 9: Large Language
Models (1/2)"), slide 67, a sidebar on coding BPE, is not mentioned, and 68 is OCW's end page. At ≈10:58
the lecturer "unhides" the loss slide. It is slide 11 of the export, so the hidden slide did not shift the
numbering. Material outside the deck: the iPad derivation of temperature (slide 56 says only "iPad";
≈51:21–56:01), the OpenAI Playground demo (slide 57) and the token visualizer (slide 66). The catalog
title's "Retrieval Augmented Generation" is not reached. RAG is deck 10's slides 76–82, so it falls to
video 10, and sources.md now gives deck 10 as "10 (provisional)" rather than "9–10". Deck 10's slide 3 is
a correction of this lecture: the dense layer before the softmax has linear activations, not ReLU. The
lecture 9 page, the large-language-models page and the slide file's header carry it, cited to deck 10.

**Numbering.** Slide N = PDF page N. Every page that prints a number prints its own, one span each;
1–4, 44, 45 and 68 print none. `slide_number_map.py` reported 1:1, and `--verify` passed on the finished
file.

**OCW notices (4):** 27 and 29 (figures from "Attention is All You Need"), 41 and 67 ("GPT Tokenizer, Byte
Pair Encoding (BPE) related images, screenshots © Andrej Karpathy"). The agents found no drawn notices.
Slide 41's notice credits Karpathy, but its upper screenshot is the header of Jay Mody's blog post; the
notice covers the page. Slide 28, between the two "Attention" pages, is plain text. Unmarked third-party
content: slide 39 (the GPT-3 paper's table of training data) and 47–50 (OpenAI Playground screenshots),
transcribed and not rendered, following lectures 5, 7 and 8. Hyperlinks whose targets are not printed:
slide 61's "Source" and slide 67's two "Link"s.

**Shared rasters.** As the TODO predicted, slides 2–4 share deck 08's masking rasters (70, 71, 76). Slide 2
is lecture 8's 71 under a review title; 3 and 4 are new (prefix tables). Slide 26 shares encoder pieces with
deck 08's 14, 21 and 22 but is new (causal heads). Slide 37 shares 08's 30–32 and is 08's 30 with
generation ticked and no footnote. Slide 59's pipeline raster is lecture 5's (28, 34; rendered there as
27). Slide 29's paper figure also appears in deck 07 (65–69); all of those carry notices.

**Transcription.** Sonnet, two agents in parallel (pages 1–34, 35–68), reading 1400px PNGs pre-rendered
into the scratchpad and appending with quoted heredocs; each took about two and a half minutes. No `$`
gaps. Agent B read slides 61–64's BPE strings off the text layer, which the rules forbid for tables; the
renders of 61 and 64 confirmed them, missing commas included. Parent fixes: slide 11's formula moved to a
`$$` block, with `\lt`/`\gt` for the end marker and the marker in backticks in prose; slide 26's arrow
count (five, not four); slide 54's claim that the page prints the 0.9 sum (it does not); slide 49's
"invited = 3.19%" format noted; the figure lines of slides 13 and 26 de-mathed, since math inside italics
does not render on github.com.

**Transcript edit.** OCW human captions with `AUDIENCE:` labels, done inline in the parent as a scripted
list of 19 changes. Restorations (12): "expert prediction" → next word prediction (14:52, 17:12, 25:45);
"mass prediction" → mask prediction (25:45); "feature words" → future words (18:02); "what elements do"
→ what LLMs do (45:58); "GPT 3.5" → GPT-3.5 (40:30, 42:52, 58:24); "wordpiece" → WordPiece (1:03:10); a
stray "$" before "30 billion sentences" (28:51); "96 blocks,." (28:05). `[Ed:]` notes (7) at 15:39
(slide 28 says the Causal Encoder is called the decoder), 18:48 ("puzzle", presumably causal), 38:07
("night" at 0.3, where slide 44 gives 0.3 to "rainy"), 46:44 ("red", where slide 52's last bar is
"wet"), 54:29 ("add up to 0"), 56:01 ("1.52", perhaps 1.5, 2) and 1:10:54 ("And", presumably "ands").
Checks: 96 timestamps identical, number inventory (153) identical once notes are removed, paragraph word
ratios 0.99–1.01. Both restored proper nouns, WordPiece and GPT-3.5, are printed in the deck.

**Figure check.** Done in the parent on renders: slides 19, 26, 39 (the GPT-3 table), 43 (the
spell-check squiggles under "Softmax" and "ReLU" are on the slide), 49, 52, 54, 61 and 64. Fixes as above;
the rest agreed.

**Printed slips.** Slide 11's factor 1/7 and its seven terms ending in `<E>`, against five output words;
slide 43's "TCE\*" with no footnote; slide 61's starting corpus missing commas ("_o,n,_t,h,e_m"), and the
1st-merge corpus "e_,m" on 62–64; slide 41's notice credit; the "Dense (ReLU)" box, corrected by deck 10.
Spoken: "night … 0.3"; "the transformer encoder … also referred to … as the transformer decoder"; "add up
to 0"; Fashion MNIST's "10 digits".

**Images: 10 of 68 pages.** Rendered: 3, 10, 16, 19, 20, 26, 36, 43, 52, 54 (1.1 MB; 6 PNG, 4 JPEG). Not
rendered: the four noticed slides (27, 29, 41, 67); 39 and 47–50 (unmarked third-party, transcribed);
repeats of earlier lectures' rendered figures: 2 (lecture 8's 71), 37 (lecture 8's 30) and 59 (lecture
5's 27); build steps superseded by a rendered slide (4 → 3; 5–9, 12, 14 and 30 → 10; 15 → 16; 18 → 19
and 20; 31 and 34 → 36 and 43); 11 (slide 10 plus a transcribed formula); 32, 33, 35 and 44 (probability
tables transcribed exactly); 21–25 (word columns with a bracket and one arrow); 13 (a single-box drawing);
the BPE slides 61–65 (text and tables); text-only slides and dividers. The wiki lecture page hand-places all
10 with hand-written captions. Slides 19 and 36 are also on the new large-language-models page, 52 and 54
on decoding-and-sampling, and 26 on transformers.

**Wiki.** A lecture page and two new topic pages (large-language-models, decoding-and-sampling). Extended:
transformers (causal self-attention and the decoder), self-supervised learning (next word prediction; its
"does not cover lecture 9 yet" sentence replaced), text vectorization (byte pair encoding), natural
language processing (sequence generation; coverage now lectures 5–9), multi-class classification (a
softmax per position, temperature), loss functions (cross-entropy averaged over positions), and AI, ML, DL
and GenAI (a link). Lectures 5, 7 and 8 now link forward to lecture 9 where they deferred byte-pair
encoding or the causal transformer. SEE_ALSO's CS224N entry no longer says this KB covers language only as
far as lecture 5, a sentence stale since lecture 6. Anchors of the new section links were checked by
script.

## Lecture 10 — Large Language Models (2/2): Instruction Tuning, Prompting, RAG and Fine-Tuning

Built 2026-10-16. Video `d-tngNnaG4U` (77 min); deck `mit15_773_s24_lec10.pdf`, 115 pages, and the
companion deck `mit15_773_s24_lec105.pdf` ("Lecture 10.5: Finetuning"), 33 pages.

**Video ↔ deck.** The recording presents deck 10's slides 2–112 in order, from the review of lecture 9 and
the ReLU correction (0:00–≈1:32) to the LoRA Colab slide (≈1:16:35), which the lecturer defers to a video
walkthrough ("9:54, time does not permit"). Slides 113 (effort-benefit curve) and 114 (topics covered) are
not reached; slide 1 is the roadmap and 115 OCW's end page. Video 11 opens on deck 11 (diffusion), so no
video presents deck 10.5. It is a variant of deck 10's fine-tuning section (slides 82–114). 26 of its 33
pages match deck 10 nearly word for word by the text layer, and raster hashes confirm most. It is
transcribed as lecture 10's companion deck, `raw/slides/10-finetuning.md`, with a page map to deck 10 in its
header. The catalog title's "PEFT" is never said or printed. Material outside the deck: the RAG Colab
(slide 81; ≈41:50–1:02:40), described from the narration.

**The published deck is later than the one shown in class.** Numbers the lecturer reads off the screen
differ from the PDF. For memory he gives a "3 to 4" multiplier ("should really be 1 to 6"), 420–560 GB and
6 to 7 GPUs; the PDF has ~3-6, 420-840GB and 6 to 11 (slide 97). He gives 7 GPUs and 8,000 days; the PDF
has 11 GPUs and 5,261 days (slide 98). He gives an optimizer state of "2x" and a total of 560 GB; the PDF
has 1-4x and 420-840GB (slide 102). For LoRA he gives "16,192 parameters, 0.02%"; slide 109 has 4 x 8192 =
32,768, 0.05%. He also says the fine-tuning diagram still shows a ReLU; slide 83 prints a plain Dense box.
Deck 10.5 is an intermediate version. It already has the corrected memory numbers, but its LoRA slides use
8096 and "2 x 8096 = 16,192 … 0.02%" (slides 21, 24–25), its pipeline keeps "Dense (ReLU)" (slide 3), and
its list of updated weights has seven items (slide 5). Its slide 8 names the marketing-copy model, "Alpaca,
running off of Llama 7B". The lesson went into AGENTS.md in general form.

**Numbering.** Slide N = PDF page N in both decks. Deck 10 prints numbers only on 2, 3, 67 and 76–114.
2, 3 and 67 print their own, and 76–114 print N+1, so a hidden slide sits somewhere in 68–75 (which print
nothing). `slide_number_map.py` reported printed 76 absent and pages 68–75 ambiguous. The headings were
checked with a plain 1..N sequence check (115 and 33, both in order). In deck 10.5 every printed number
equals its page.

**OCW notices.** Deck 10 has 14: 59 (Wayfair LLC), 70–73 (prompt examples from the GPT-3 paper, Tom B.
Brown et al./ArXiv), 78 (benjedwards on Twitter), 80 (Sebastian Raschka's RAG figure), 87, 88, 92 (rag &
bone), 89, 93 (Urban Outfitters), 95 (Llama 2 screenshots, Meta) and 113 (Karpathy's effort-benefit curve).
114 carries a CC BY credit (Minaee et al., "Large Language Models: A Survey"). It is not an exclusion, and
it is rendered. Deck 10.5 prints no notice anywhere, typed or drawn. Its pages 7, 8 and 30 (rag & bone),
9 and 29 (Urban Outfitters), 11 and 12 (Llama 2) and 31 (Karpathy) reuse deck 10's excluded material, and
its page 32 has the survey flow chart without the credit line. Deck 10's slide 96 repeats 95's Llama 2
screenshot without a notice. None of these is rendered. Unmarked third-party content, transcribed and not
rendered: 5–6 (a GPT-3 continuation from Twitter; the continuation is described, not copied), 9–10, 42,
44–45, 48–50, 63–64 (GPT-3, InstructGPT and ChatGPT screenshots), 15 (the InstructGPT paper's title block),
55 (the lecturer's LinkedIn post), 60–61 (Playground), 65–67 (paper figure, table and first page) and 91 (a
dataset screenshot). Hyperlinks whose targets are not printed: 81 (Colab), 109 (LoRA paper), 114 (survey),
and deck 10.5's 28.

**Shared rasters.** Deck 10 shares only PowerPoint shape masks (168×76, 60×59, used across every deck) with
earlier decks; no figure is a repeat by hash. A pixel diff shows slide 2 close to, but not identical with,
lecture 9's slide 10. It is superseded by slide 3 and not rendered.

**Transcription.** Sonnet, four agents in parallel: deck 10 pages 1–38, 39–76 and 77–115, and deck 10.5
1–33. They read 1400px PNGs pre-rendered into the scratchpad and appended with quoted heredocs; each took
2.5–3.5 minutes. No `$` gaps. Parent fixes: slide 31's formula subscript ($r_p - r_O$, as printed, not
$r_o$); slide 111's cell "Roughly zero by computing these 'just in time'" (both agents dropped "these");
slide 114's unverified "(added by the lecturer)"; slide 113's curve values (re-read off the render); dollar
amounts escaped as `\$` (check_math passes them); quotes removed before inline math on slides 30 and 106
and deck 10.5's 22, which github.com would not render. The agent's claim that "2 x 108" on slides 100–101
has no superscript was checked on a crop: true, it is printed flat, where slide 98's $10^{12}$ is
superscripted.

**Transcript edit.** OCW human captions, done inline in the parent as a scripted list of 18 changes.
Restorations (5): "And the reason, So" (3:55); "very Creative" (9:21); "Chat-GPT" → ChatGPT (21:45,
27:05); "a new and updated element" → LLM (1:05:47). `[Ed:]` notes (13) at 1:32 (slide 4's "GPT and
GPT-2"), 26:19 (garbled), 45:44 ("GPT-3-turbo"), 52:38 ("0.934" for 0.9934), 56:32 ("1,600 something" for
16,385), 1:02:40 (the ReLU the published slide no longer has), 1:09:39, 1:10:26 (two), 1:12:43, 1:13:29
(two) and 1:15:49 (the published deck's numbers). Left as captioned: "embedding-ada-002", "Carhart",
"LORA" (as slide 109 prints it), "a Sloany". Checks: 100 timestamps identical, number inventory (416)
identical once notes are removed, paragraph word ratios 1.00–1.007.

**Figure check.** In the parent on renders: deck 10's slides 3, 20, 31, 33, 39, 58, 77, 85, 103, 109,
111 and 114, plus crops of the printed powers of ten on 98 and 100; deck 10.5's 3 and 25. Fixes as above;
the rest agreed. Deck 10.5's seven-item list (slide 5) and its 8096 numbers were confirmed against the text
layer and render.

**Printed slips.** Slides 100–101's "2 x 108" for 2×10^8; slide 109's stray "8096" label among 8192s, and
"LORA"; slide 110's "ΔA^v" and its unfinished last bullet; slide 111's "can be comfortably be finetuned" and
footnote-less asterisks; slide 57's "base (aka foundation or base)"; slide 60's "in being described"; "Open
AI" on 28–29; slide 55's attached document dated January 10, 2023. Deck 10.5: "Bit first…" (slide 6), "Lets
get better at Fake Reviews" (slide 10), "8096x8096" (slide 21). Spoken: see the deck-version paragraph
above, and the `[Ed:]` notes.

**Images: 13 of 115 pages.** Rendered: 3, 20, 31, 33, 39, 40, 52, 58, 77, 84, 85, 109, 114 (1.24 MB; 10
PNG, 3 JPEG). None from deck 10.5. Not rendered: the 14 noticed slides and 96; unmarked third-party
screenshots and paper figures (above); 2 (lecture 9's pipeline, superseded by 3); build steps and repeats
superseded by a rendered slide (16, 46, 47, 51 → 40; 30, 32 → 31; 35–38 → 39; 53, 54 → 52; 68, 74, 75,
82 → 58; 83, 104 → 85; 108 → 109); 106 (two number tables, transcribed exactly); 14 and 110
(single-shape drawings); text-only slides. The wiki lecture page hand-places all 13 with hand-written
captions. Slides 39 and 52 are also on the instruction-tuning page, 58, 85 and 109 on adapting-llms, and 77
on retrieval-augmented-generation.

**Wiki.** A lecture page and three new topic pages: instruction-tuning, adapting-llms (the ladder,
prompting, in-context learning, fine-tuning, Llama-2-70b's cost, LoRA) and retrieval-augmented-generation.
Extended: large-language-models (the correction, now cited to lecture 10; GPT/GPT-2/GPT-3 scale and
in-context learning; base model to assistant; context window and hallucination; its "second lecture is not
yet in this knowledge base" sentence replaced), transfer-learning (fine-tuning LLMs), loss-functions (the
Reward Model's loss), decoding-and-sampling (sampling several answers to rank), NLP (coverage now lectures
5–10), word-embeddings, transformers and self-supervised-learning ("where this comes up"). Lecture 9's page
and deck 09's slide file now cite deck 10's slide 3 and RAG through the new slide file and lecture page. The
lecture page works two checks the lecturer invites or the slide prints, labelled as its own: the Reward
Model loss's behaviour, and 32,768 = 2 × 2 × 8192 ≈ 0.05% of 64M. SEE_ALSO's CS224N entry now points to
its post-training, prompting, instruction-finetuning, RLHF and LoRA pages, each confirmed tracked in that
repo. Four plain-text "×" in the new pages were rewritten in words; verify_kb's math-warning list is capped
and did not show them.

## Build checklists, lectures 1–4

The ticked lines of `TODO.md` for the runs above, moved here verbatim on 2026-10-14 once lecture 4
was committed and pushed.

### Transcripts
- [x] 01 Introduction to Neural Networks and Deep Learning — video kyQ0CRkYhy4 (OCW human captions; light copy-edit, verbatim in original/)
- [x] 02 Training Deep NNs (cont.); Intro to Keras/TF; Tabular Data — video 3Lr8nVUDCpk (OCW human captions; light copy-edit; covers deck 02 slides 2–94 only)
- [x] 03 DL for Computer Vision – CNNs from Scratch — video 8QuyDcMIdRc (OCW human captions; light copy-edit; covers deck 3A slides 2–43 and deck 3B slides 2–28; no CNNs — those are deck 04)
- [x] 04 DL for Computer Vision – Transfer Learning, Fine-Tuning; HuggingFace — video 8xh7Y0pBrCE (OCW human captions; light copy-edit, inline; covers deck 04 slides 2–81: CNNs from scratch, then transfer learning) → raw/transcripts/04-cnns-and-transfer-learning.md

### Crawl
- [x] Fetch course site index: https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/
- [x] Download linked PDFs and slides (14 decks, 11 OCW transcript PDFs; gitignored)
- [x] Write sources.md

### Slides (raw/slides/)
- [x] 01 Introduction — mit15_773_s24_lec01.pdf (110 pages; slide N = page N; read by Sonnet; figure check on 6 rendered pages, 1 corrected)
- [x] 02 Training Deep NNs — mit15_773_s24_lec02.pdf (95 pages; slide N = page N, pages 50–94 print N+1; read by Sonnet in three parts; figure check on 8 rendered pages incl. 2 backprop, 1 corrected)
- [x] 02 Backprop example — mit15_773_s24_lec02_backprop_example.pdf (14 handwritten pages; not presented in the video, pointed to at 59:40)
- [x] 03 deck 3A Keras/TF, training — mit15_773_s24_lec03a.pdf (44 pages; 1:1) → raw/slides/03-keras-tensorflow.md
- [x] 03 deck 3B Vision basics — mit15_773_s24_lec03b.pdf (29 pages; hidden slide 16, pages 16–28 print N+1 where printed; 8 OCW notices) → raw/slides/03-vision-basics.md
- [x] 03 figure check on rendered images (3A 10, 15, 17, 18; 3B 6 (scratch crop), 16, 18 — all agreed)
- [x] 04 CNNs and transfer learning — mit15_773_s24_lec04.pdf (84 pages; 1:1; 14 OCW notices) → raw/slides/04-cnns-and-transfer-learning.md (Sonnet, two agents in parallel)
- [x] 04 figure check on rendered images (21, 31, 40, 47, 75, 80; slide 40 corrected)

### Wiki
- [x] wiki/01-introduction.md
- [x] Topic pages (cross-lecture concepts) — ai-ml-dl-genai, representation-learning, neural-network-anatomy, activation-functions (lecture 1 only; extend as lectures land)
- [x] INDEX.md table of contents
- [x] wiki/02-training-deep-nns.md
- [x] Topic pages for lecture 2 (loss functions, gradient descent, backprop, SGD, Keras; extend neural-network-anatomy, activation-functions)
- [x] INDEX.md — lecture 2 entries and coverage sentence
- [x] wiki/03-keras-and-vision-basics.md
- [x] Topic pages for lecture 3 — new: overfitting-and-regularization, tensors, data-preprocessing, images-and-vision-tasks, multiclass-classification; extended: keras, training-neural-networks, gradient-descent, loss-functions, activation-functions, neural-network-anatomy
- [x] INDEX.md — lecture 3 entries and coverage sentence
- [x] wiki/04-cnns-and-transfer-learning.md
- [x] Topic pages for lecture 4 — new: convolutional-neural-networks, transfer-learning, data-augmentation; extend the pages it touches
- [x] Correct the earlier pages that defer CNNs or dropout to lecture 4
- [x] INDEX.md — lecture 4 entries and coverage sentence

### Images
- [x] raw/images/01-introduction/ — rank, pick, render, embed in the wiki and slide files
- [x] INDEX.md image rules; AGENTS.md Images conventions and table row; BUILD_LOG.md; LICENSE.md attribution
- [x] raw/images/02-training-deep-nns/ — deck 02 as slide-N, backprop deck as backprop-N; embed
- [x] INDEX.md image sentence; AGENTS.md table row; BUILD_LOG.md; LICENSE.md attribution (lecture 2)
- [x] raw/images/03-keras-and-vision-basics/ — deck 3A as 3a-N, deck 3B as 3b-N; embed
- [x] INDEX.md image sentence; AGENTS.md table row; BUILD_LOG.md; LICENSE.md attribution (lecture 3)
- [x] raw/images/04-cnns-and-transfer-learning/ — 16 renders; embedded in the slide file
- [x] Embed lecture 4 images in the wiki page; INDEX.md image sentence; AGENTS.md table row; BUILD_LOG.md; LICENSE.md attribution

### Publish
- [x] kb.json — coverage, materials.method, provenance caveats
- [x] BUILD_LOG.md — lecture 1 record
- [x] SEE_ALSO.md, if a sibling KB is genuinely relevant
- [x] verify_kb.py clean, and its review section read
- [x] Commit and push
- [x] PATCH kbUrl onto the catalog entry (https://github.com/chaimantec/cairn-kb-mit-15-773)
- [x] Lecture 2: kb.json, BUILD_LOG.md, verify_kb.py clean, commit and push
- [x] Lecture 3: kb.json, BUILD_LOG.md, verify_kb.py clean, commit and push
- [x] Lecture 4: kb.json, BUILD_LOG.md, verify_kb.py clean, commit and push
- [x] Move finished sections from TODO.md to BUILD_LOG.md; TODO.md keeps only open work

## Build checklist, lecture 5

The ticked lines of `TODO.md` for the lecture 5 run, moved here verbatim on 2026-10-14 once it was
committed and pushed.

### Lecture 5 run (2026-10-14)

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
- [x] Move this section to BUILD_LOG.md

## Build checklist, lecture 6

The ticked lines of `TODO.md` for the lecture 6 run, moved here verbatim on 2026-10-14 once it was
committed and pushed.

### Lecture 6 run (2026-10-14)

Video 6 (`LqFc0z-pQTg`, 78 min) presents deck 06 (`mit15_773_s24_lec06.pdf`, 70 pages, 1:1 by
corner spans; page 70 is OCW's end page): it opens on slide 2 ("Until now, we encoded input text
strings with one-hot vectors") and ends in the Colab (GloVe, then trained-from-scratch
embeddings, 72%). Text-layer OCW notices: slides 14, 15, 17, 18, 19 (Cohere embedding quizzes) and
47 (Chollet's one-hot vs embedding figures). Slide 19 carries the Cohere notice too. Slides 48–51 (GloVe visualizations) carry no notice.

- [x] Transcript 06 — video LqFc0z-pQTg → raw/transcripts/06-word-embeddings.md (+ original/)
- [x] Slides 06 → raw/slides/06-word-embeddings.md (Sonnet, two agents; slide 19's notice found by the agent)
- [x] 06 figure check on rendered images (37, 45, 48, 59, 63, 67, 68 agree; slide 37's last handwritten label marked hard to read)
- [x] raw/images/06-word-embeddings/ — render, embed in slide file and wiki (rendered 37, 49, 51, 55, 59, 61, 67, 68; embedded in the slide file and the wiki)
- [x] wiki/06-word-embeddings.md
- [x] Topic pages for lecture 6 (new: word-embeddings; extended text-vectorization, bag-of-words, NLP, keras, representation-learning, transfer-learning, gradient-descent, overfitting)
- [x] INDEX.md — lecture 6 entries, coverage and image sentences
- [x] kb.json, BUILD_LOG.md lecture 6 record, AGENTS.md image row, LICENSE.md attribution, sources.md
- [x] verify_kb.py clean (with check_math and --github), review section read
- [x] Commit and push
- [x] Move this section to BUILD_LOG.md

## Build checklist, lecture 7

The ticked lines of `TODO.md` for the lecture 7 run, moved here verbatim on 2026-10-14 once it was
committed and pushed.

### Lecture 7 run (2026-10-14)

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
- [x] Commit and push
- [x] Move this section to BUILD_LOG.md

## Build checklist, lecture 8

The ticked lines of `TODO.md` for the lecture 8 run, moved here verbatim on 2026-10-16 once it was
committed and pushed.

### Lecture 8 run (2026-10-16)

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
- [x] wiki/08-transformers-and-self-supervised-learning.md, with images hand-placed (17 of 19, hand-written captions)
- [x] Topic pages for lecture 8 (new: self-supervised-learning; extended transformers, transfer-learning, representation-learning, NLP, data-preprocessing, images, data-augmentation, keras, word-embeddings, CNNs; lecture 7 links forward)
- [x] INDEX.md — lecture 8 entries, coverage and image sentences
- [x] kb.json, BUILD_LOG.md lecture 8 record, AGENTS.md image row, LICENSE.md attribution, sources.md
- [x] verify_kb.py clean (with check_math and --github), review section read (three formulas on transformers.md repaired after a Python-escape slip; one BUILD_LOG line de-mathed)
- [x] Commit and push
- [x] Move this section to BUILD_LOG.md

## Build checklist, lecture 9

The ticked lines of `TODO.md` for the lecture 9 run, moved here verbatim on 2026-10-16 once it was
committed and pushed.

### Lecture 9 run (2026-10-16)

Video 9 (`KGDe1QvfKJ8`, 75 min) presents deck 09 (`mit15_773_s24_lec09.pdf`, 68 pages; page 68 is
OCW's end page) from slide 2 to slide 66, ending on the token-visualizer demo; slide 67 (a sidebar) is
not mentioned. RAG, in the catalog title, is deck 10's slides 76–82, so it falls to video 10.
Numbering: every printed number equals its page; 1–4, 44, 45 and 68 print none. Text-layer OCW
notices: 27, 29, 41, 67.

- [x] 09 Generative AI – LLMs and RAG — video KGDe1QvfKJ8 → raw/transcripts/09-large-language-models.md (+ original/)
- [x] Slides 09 → raw/slides/09-large-language-models.md (Sonnet, two agents)
- [x] 09 figure check on rendered images (19, 26, 39, 43, 49, 52, 54, 61, 64)
- [x] raw/images/09-large-language-models/ — render, embed in slide file (10 slides)
- [x] wiki/09-large-language-models.md, with images hand-placed (10 of 10, hand-written captions)
- [x] Topic pages for lecture 9 (new: large-language-models, decoding-and-sampling; extended transformers, self-supervised-learning, text-vectorization, NLP, multiclass-classification, loss-functions, ai-ml-dl-genai; lectures 5, 7, 8 link forward; deck 10's ReLU correction carried)
- [x] INDEX.md — lecture 9 entries, coverage and image sentences
- [x] kb.json, BUILD_LOG.md lecture 9 record, AGENTS.md image row and two lessons, LICENSE.md attribution, sources.md, SEE_ALSO.md
- [x] verify_kb.py clean (with check_math and --github), review section read (two italic-math figure lines in the slide file fixed; anchors checked by script)
- [x] Commit and push
- [x] Move this section to BUILD_LOG.md

## Build checklist, lecture 10

The ticked lines of `TODO.md` for the lecture 10 run, moved here verbatim on 2026-10-16 once it was
committed and pushed.

### Lecture 10 run (2026-10-16)

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
- [x] wiki/10-adapting-llms.md, with images hand-placed (13 of 13, hand-written captions)
- [x] Topic pages for lecture 10 (new: instruction-tuning, adapting-llms, retrieval-augmented-generation; extended large-language-models, transfer-learning, loss-functions, decoding-and-sampling, NLP, word-embeddings, transformers, self-supervised-learning); deck 10 slide 3 citations repointed (lecture 9 page, large-language-models, deck 09 slide file)
- [x] INDEX.md — lecture 10 entries, coverage and image sentences
- [x] kb.json, BUILD_LOG.md lecture 10 record, AGENTS.md image row and two lessons, LICENSE.md attribution, sources.md, SEE_ALSO.md (CS224N post-training pages)
- [x] verify_kb.py clean (with check_math and --github), review section read (quotes before inline math and plain-text × fixed; 44 anchor links checked by script)
- [x] Commit and push
- [x] Move this section to BUILD_LOG.md
