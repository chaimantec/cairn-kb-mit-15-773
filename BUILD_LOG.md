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
