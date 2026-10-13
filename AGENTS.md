# How this knowledge base is organized

This repo is the knowledge base for **15.773 — Hands-On Deep Learning**. It is read
by Cairn's in-extension AI chat, which starts at `INDEX.md`, fetches files over
raw.githubusercontent.com and follows relative markdown links.

The chat never reads this file. It is for whoever builds the KB, and Claude Code
loads it into every step of every build agent, so keep it to the rules every
lecture's build follows. The per-lecture record (licence notices, audit findings,
printed slips, which slides were rendered and why) goes in
[`BUILD_LOG.md`](BUILD_LOG.md); anything the chat needs goes in `INDEX.md`.

## Layout

| Path                | Contents                                                      |
| ------------------- | ------------------------------------------------------------- |
| `INDEX.md`          | Entry point. Course summary + annotated table of contents.    |
| `wiki/`             | Durable pages: one per lecture, plus cross-lecture topics.    |
| `raw/slides/`       | Full text of every slide, numbered. Slide N = PDF page N.     |
| `raw/transcripts/`  | Copy-edited transcripts with `[MM:SS]` paragraph marks — the ones to read. |
| `raw/transcripts/original/` | The unedited OCW caption text, for reference.        |
| `raw/pdfs/`         | Decks downloaded from OCW. **Gitignored**; cite the OCW URL.  |
| `raw/images/`       | Rendered figure slides, if any. Linked from the pages using them. |
| `sources.md`        | Crawl inventory: source URL → local file, fetch date.         |
| `TODO.md`           | Build tracker. Unchecked boxes are outstanding work.          |
| `BUILD_LOG.md`      | Per-lecture build record. Not needed to answer questions.     |
| `LICENSE.md`        | CC BY-NC-SA 4.0, following OCW; exclusions; image attribution. |
| `kb.json`           | Machine-readable coverage and provenance.                     |

## This course

- **Source.** MIT OpenCourseWare, <https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/>,
  CC BY-NC-SA 4.0. Lecturer Rama Ramakrishnan; the lecture 1 title slide credits "Farias, Ramakrishnan".
- **Decks do not follow the videos.** OCW lists 14 decks (01, 02, 02 backprop example, 03a, 03b,
  04–10, 10.5, 11) against 11 videos. Before reading a deck, find which slides a video actually
  covers: compare the transcript's first and last minutes with deck titles and the text layer.
  Name files by the **video** number (`01-…`), and say in the transcript header which deck slides
  the recording spans. Catalog video titles misstate coverage in both directions: video 1's
  mentions "Training Deep NNs", which it never reaches, and video 4's omits the CNNs that fill its
  first hour.
- **A lecture can own a companion deck it never presents** (lecture 2's handwritten backprop
  example). Give it its own slide file, `NN-<companion>.md`, with its own `## Slide 1..N`
  headings. Render its pages into the lecture's single image directory with a distinct prefix
  (`backprop-N`, via `embed_slide_images.py --prefix`), because `verify_kb.py` keys slide files and
  image directories on the two-digit prefix. In the wiki, cite it as "page N of the …" so the embed
  script's `slide N` anchoring cannot misplace it, and hand-place its images.
- **A video can present two decks** (lecture 3: decks 3A and 3B). Give each its own slide file
  `NN-<deck-slug>.md`, render both into the lecture's one image directory with per-deck prefixes
  (`3a-N`, `3b-N`), and cite "deck 3A, slide N". The embed script anchors on any "slide N", so write
  the wiki page in one part per deck, split it at the boundary, run `embed_slide_images.py --wiki`
  on each part with its deck's prefix, and rejoin. Keep cross-deck slide citations out of the other
  deck's part, and check the result: an anchor sentence ending in a colon puts the image between it
  and its displayed formula.
- **An exclusion follows the image, not the page.** A figure OCW marks as excluded in one deck can
  reappear without a notice in another (lecture 3's deck 3A reuses deck 3B's excluded pixel and RGB
  matrices). Compare figure descriptions across a lecture's decks, and do not render the unmarked
  copy.
- **The Colab notebooks are not on OCW.** Describe a lecture's code from the narration, naming only
  the calls and arguments the lecturer says, and say so on the page. The decks' "Colab" links carry
  their targets only in the PDF's link annotations: write that the target is not printed, and do not
  record it.
- **Slide N = PDF page N** in every heading and citation, unless a deck's header says otherwise.
  These decks print PowerPoint numbers bottom-right, but some pages carry visibly printed **stale
  static numbers** from an older version of the deck (lecture 1's page 13 prints "20"), and a
  hidden slide makes every later printed number run ahead of the page (lecture 2, from page 50). Two-digit
  and three-digit numbers are also often split across two text spans ("1"+"0", "10"+"0").
  `slide_number_map.py` misreads both, so derive the mapping from raw span positions, check the
  outliers on page crops, and verify headings with a plain `1..N` sequence check rather than
  `--verify`. Each deck's last page is OCW's end page, which is not lecture content.
- **OCW notices.** Excluded third-party material carries "© … All rights reserved. This content is
  excluded from our Creative Commons license". Each such slide gets an `*OCW notice: …*` line in the
  slide file and is **never rendered**. A notice may lack "All rights reserved" (lecture 1, slide 87)
  and is still an exclusion. A CC BY-SA credit (lecture 1, slide 32) is not an exclusion; render it
  and attribute it in `LICENSE.md`.

## Transcripts

The YouTube captions on these videos are OCW's **human-made** captions: punctuated, with speaker
labels (`RAMA RAMAKRISHNAN:`, `AUDIENCE:`) and sound tags. So the edit is light, done inline: a short
list of restorations checked against the slide file, plus `[Ed: …]` notes where the captions are
garbled or the speaker misstates a printed number. Every change is listed in the file's header. The
verbatim text stays in `raw/transcripts/original/`. Verify by script against `original/`: the
`[MM:SS]` sequence must be identical, the number inventory identical once `[Ed: …]` notes are
removed, and every paragraph's word ratio within 0.72–1.10 (with notes removed).

## Slides

Each deck is transcribed from page images by a Sonnet agent. The parent first decides the numbering
and lists the OCW notices from the text layer, and hands both to the agent. The agent must look
for notices on every page, because some are drawn rather than typed. Each slide entry ends with
`*Figure: yes — …*` or `*Figure: no*`, which is what image selection reads. Figure descriptions are
checked against **rendered images** (1400px renders are cheap), never against full PDF page images
in the parent's context.

### Images

Every figure-bearing slide is rendered at 1400px, keeping the smaller of JPEG q85 and PNG, as
`raw/images/NN-<slug>/slide-N.{jpg,png}` with N the PDF page (a per-deck prefix replaces `slide`
when a lecture has more than one deck). Not rendered: slides with an OCW
notice, build steps or repeats superseded by a rendered slide, tables the slide file reproduces
exactly, single-box text diagrams, and course-logistics slides. `extract_slide_images.py` selects
by *printed* number, which is unreliable on these decks (see above), so render by PDF page with a
short PyMuPDF script using the same defaults. Then run `embed_slide_images.py`. Hand-placed wiki
images must **copy the filename from disk**, because the extension varies per slide. After
embedding, rewrite the default captions of build-step slides that share a title, since otherwise
several identical captions sit in a row. Images are linked relatively; the chat gets a renderable
URL by reading the path. `grep -o 'raw/images/[^)]*' wiki/NN-*.md` lists a lecture's images.
What was rendered and why goes in `BUILD_LOG.md`.

| Lecture | Images | Deck |
| --- | --- | --- |
| 1 Introduction to Neural Networks and Deep Learning | 45 of 110 pages | rendered from `mit15_773_s24_lec01.pdf` |
| 2 Training Deep Neural Networks | 18 | 15 of `mit15_773_s24_lec02.pdf`'s 95 pages as `slide-N`, and 3 of the backprop example's 14 as `backprop-N` |
| 3 Keras and TensorFlow, Training the Heart-Disease Model, and Computer Vision Basics | 13 | 9 of `mit15_773_s24_lec03a.pdf`'s 44 pages as `3a-N`, and 4 of `mit15_773_s24_lec03b.pdf`'s 29 as `3b-N` |
| 4 Convolutional Neural Networks and Transfer Learning | 16 of 84 pages | rendered from `mit15_773_s24_lec04.pdf` |

## Conventions

- **INDEX.md is the front door.** The chat reads it first on every conversation.
  Every wiki page must appear there with a one-line description of what it holds.
  An unindexed page is effectively invisible.
- **Relative links only** (`[gradient descent](gradient-descent.md)`,
  `[slides](../raw/pdfs/lecture03.pdf)`). Absolute GitHub URLs break when the
  repo is renamed or forked.
- **Cite everything.** Claims trace back to a transcript timestamp or a slide.
- **Never invent course content.** If the transcript is unclear at some point,
  say so on the page. Do not fill the gap from outside knowledge — the chat
  presents these pages as authoritative material from this course. Arithmetic the lecturer
  invites (a parameter count to hand-check) may be worked on the page, labelled as the page's own
  and naming every assumption the lecture does not state.
- **Prose over fragments.** The chat quotes these pages to learners; bullet
  fragments quote badly.
- **Math in LaTeX**: `$...$` inline, `$$...$$` displayed. Renders both in
  Cairn's chat and on github.com. Never inside a code fence — that shows the
  source instead of the formula. Define each symbol on first use, and follow the
  course's own notation rather than a tidier one, since the learner is comparing
  the page against the lecturer's board.
- **Transcripts stay verbatim.** Captions spell notation out as speech ("theta
  sub j"); reconstructing it as $\theta_j$ belongs in `wiki/`, not in
  `raw/transcripts/`.

## Rebuilding

Built and updated by the `cairn-kb` skill. To add newly released lectures,
append entries to `TODO.md` and re-run the skill — it only does unchecked work.
Write each lecture's details into `BUILD_LOG.md`, and add here only a lesson
that applies to every lecture, stated in general form.
