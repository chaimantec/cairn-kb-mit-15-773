# 15.773 — Hands-On Deep Learning (MIT, Spring 2024)

MIT Sloan's hands-on introduction to deep learning, taught by **Rama Ramakrishnan** (the lecture 1
title slide credits "Farias, Ramakrishnan") and published on
[MIT OpenCourseWare](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/). Over
11 lectures it goes from neural-network basics and training, through computer vision (CNNs,
transfer learning) and natural language (embeddings, transformers), to generative AI (LLMs, RAG,
parameter-efficient fine-tuning, text-to-image diffusion). It deliberately "skips the math" and
focuses on concepts and on coding models in Keras/TensorFlow.

**Coverage: lecture 1 of 11.** This knowledge base is being built lecture by lecture. Only lecture 1
has a wiki page, transcript and slide transcription so far. Do not cite it as covering lectures
2–11. For anything after lecture 1, say that the KB does not cover it yet.

## Contents

### Lectures

- [Lecture 1 — Introduction to Neural Networks and Deep Learning](wiki/01-introduction.md): how AI,
  machine learning, deep learning and generative AI relate (rules → learning from examples →
  learned representations → generating unstructured data); "put DL behind every sensor"; then a
  neural network built from logistic regression: weights and biases, neurons, layers, activation
  functions (sigmoid, linear, ReLU), designing a DNN, counting the 13 parameters of a 2-3-1 network,
  and a worked forward pass. Training is *not* covered here despite the video's catalog title.
  Also summarises the deck-only course logistics (prerequisites, grading, course philosophy).

### Topics

- [AI, machine learning, deep learning and generative AI](wiki/ai-ml-dl-genai.md): the nested
  circles the course uses, what bottleneck each layer removed, and X-to-Y multimodality.
- [Representation learning](wiki/representation-learning.md): why raw unstructured data needs a
  representation, hand-made features versus learned ones, and hidden layers as the learned
  representation.
- [Anatomy of a neural network](wiki/neural-network-anatomy.md): the vocabulary (weight, bias,
  neuron, layer, dense, feedforward, architecture), what you design versus what training finds,
  parameter counting, and the forward pass with numbers.
- [Activation functions](wiki/activation-functions.md): sigmoid, linear and ReLU with formulas and
  plots, and the rule of thumb: ReLU for hidden layers, output activation dictated by the output.

## Raw materials

- [`raw/transcripts/`](raw/transcripts/): one transcript per lecture, in ~45-second paragraphs each
  opening with an `[MM:SS]` timestamp. **Read and quote these.** They are OCW's human-made captions
  with a light copy-edit, and every change is listed in the file's header. The unedited caption text
  is in [`raw/transcripts/original/`](raw/transcripts/original/), for reference only.
- [`raw/slides/`](raw/slides/): every slide of each lecture's deck, transcribed in order under
  `## Slide N` headings, with equations, tables and figure descriptions. **Slide N is PDF page N.**
  Some lecture 1 pages also print a stale number in the corner (page 13 prints "20"); the slide
  file lists them. Cite the heading number.
- [`raw/images/`](raw/images/): rendered figure slides (see below).
- [`sources.md`](sources.md): every course document on the OCW site with its canonical URL. The
  slide PDFs are not in this repo; link to them at their OCW URL.
- [`LICENSE.md`](LICENSE.md): CC BY-NC-SA 4.0, following OCW, with the attribution of every image.
- [`SEE_ALSO.md`](SEE_ALSO.md): sibling knowledge bases worth consulting, with their repo URLs to
  pass as `kb`.

## Images

**Lecture 1 has images. No other lecture does yet.** 45 of its figure slides are rendered as whole
slides and shown inline, both in [the lecture page](wiki/01-introduction.md) and under the matching
`## Slide N` heading in [its slide file](raw/slides/01-introduction.md). Slides whose content OCW
excludes from its licence (marked `*OCW notice: …*`) are never rendered.

- **Use an image URL you have actually read in a file.** Never construct one from the pattern, and
  never assume a slide has an image because a neighbouring one does. The extension varies (`.jpg`
  or `.png`), and many slides are deliberately not rendered.
- To show an image, read its relative path (for example
  `raw/images/01-introduction/slide-100.png`) and use the URL that comes back.
- For numbers, prefer the transcription to the picture. Show one image, not a gallery, and keep the
  slide citation beside it.
