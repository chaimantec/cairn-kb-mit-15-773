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
| `raw/transcripts/`  | Lecture transcripts with `[MM:SS]` paragraph marks.           |
| `raw/pdfs/`         | Slides and handouts downloaded from the course website.       |
| `raw/images/`       | Rendered figure slides, if any. Linked from the pages using them. |
| `sources.md`        | Crawl inventory: source URL → local file, fetch date.         |
| `TODO.md`           | Build tracker. Unchecked boxes are outstanding work.          |
| `BUILD_LOG.md`      | Per-lecture build record. Not needed to answer questions.     |

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
  presents these pages as authoritative material from this course.
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
