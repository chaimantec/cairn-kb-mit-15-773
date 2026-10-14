# Licence and attribution

This repository is a study aid compiled from **MIT 15.773 Hands-On Deep Learning, Spring
2024**, as published on MIT OpenCourseWare:
<https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/>.

Instructor: Rama Ramakrishnan. The lecture 1 title slide credits "Farias, Ramakrishnan".

## The short version

Everything in this repository is offered under **Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International** (`CC BY-NC-SA 4.0`),
<https://creativecommons.org/licenses/by-nc-sa/4.0/>. That is the licence MIT OpenCourseWare
publishes this course under. You may reuse it non-commercially, with attribution, and must share
adaptations under the same licence.

**Except** the third-party figures that OCW itself excludes from its licence (next section).
None of those images is reproduced in this repository.

## What comes from the course, and on what terms

- `raw/transcripts/` — adapted from OCW's published caption files for the lecture videos.
  `raw/transcripts/original/` holds the caption text regrouped into timestamped paragraphs. The
  files directly in `raw/transcripts/` add a light copy-edit with every change marked.
- `raw/slides/` — a slide-by-slide transcription of the lecture decks, with figures described
  in prose.
- `raw/images/` — whole-slide renders of deck pages, unmodified apart from scaling.

These are adaptations of CC BY-NC-SA material and carry the same licence. Attribution:
*MIT 15.773 Hands-On Deep Learning, Spring 2024, MIT OpenCourseWare, <https://ocw.mit.edu>,
licensed CC BY-NC-SA 4.0.* OCW's citation and terms of use are at <https://ocw.mit.edu/terms>.

The wiki prose in `wiki/` is original writing that summarises and cites the above. It is offered
under the same licence.

## Third-party content that is excluded

OCW marks some figures in the decks with a notice such as *"© unknown. All rights reserved.
This content is excluded from our Creative Commons license"*. OCW includes them under its own
fair-use determination (<https://ocw.mit.edu/help/faq-fair-use/>), and that determination does
not pass to anyone who copies the deck. So a slide carrying such a notice is **transcribed in
prose only and never rendered** into `raw/images/`. Its slide-file entry carries an
`*OCW notice: …*` line naming the rights holder.

## Rendered images

Each image in `raw/images/NN-<slug>/slide-N.jpg` (or `.png`) is a whole-page render of page N
of that lecture's OCW deck, at 1400px wide. Where a lecture has more than one deck, a prefix names
the deck in place of `slide` (`backprop-N`, `3a-N`, `3b-N`). Attribution for every one is the course line above,
plus the deck:

- **Lecture 1**, `raw/images/01-introduction/` (45 images), from
  [`mit15_773_s24_lec01.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec01.pdf).
  Slide 32 (`slide-32.jpg`) contains an artificial neural network figure by Glossier.ca, licensed
  CC BY-SA, from Wikimedia Commons (<https://en.wikipedia.org/wiki/Artificial_neural_network>), as
  credited on the slide. That figure remains under its own licence. Slide 45's temple image was
  generated with Stable Diffusion from the prompt printed on the slide (source cited there:
  mpost.io). OCW did not mark it as excluded. None of the 15 slides that carry an OCW exclusion
  notice (11, 12, 22–24, 28, 35, 37, 38, 41, 43, 44, 49, 50, 87) is rendered.
- **Lecture 2**, `raw/images/02-training-deep-nns/` (18 images): `slide-N` files from
  [`mit15_773_s24_lec02.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec02.pdf), and `backprop-N` files (pages 2, 8 and 12) from the
  lecturer's handwritten [`mit15_773_s24_lec02_backprop_example.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec02_backprop_example.pdf). Third-party material
  on the rendered slides, as credited there: slides 55–58 and 69 reproduce plots from the mlrefined
  blog page cited on each slide (kenndanielso.github.io/mlrefined, "6.4 Gradient descent"). OCW
  marks only slide 75, whose notice credits that page's gradient-descent figures to Jeremy
  Watt/neonwatty on GitHub, as excluded. Slide 66 is a lithograph of Augustin-Louis Cauchy, in the
  public domain via Wikimedia Commons, as the slide states. Slide 93 reproduces a loss-landscape
  visualization from arXiv 1712.09913, the paper the slide cites. Slide 94 reproduces Figure 2.26
  from page 61 of the course textbook, named on slide 2 of the next deck as François Chollet's
  *Deep Learning with Python*, 2nd edition. Those figures remain their authors'. Slide 75, the
  deck's only slide with an OCW exclusion notice, is not rendered.
- **Lecture 3**, `raw/images/03-keras-and-vision-basics/` (13 images): `3a-N` files (pages 6, 9, 10,
  15, 17, 18, 32, 36 and 43) from
  [`mit15_773_s24_lec03a.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec03a.pdf),
  and `3b-N` files (pages 14, 16, 17 and 18) from
  [`mit15_773_s24_lec03b.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec03b.pdf).
  Third-party material on the rendered slides, as credited there: deck 3A's slide 32 reproduces the
  Keras/TensorFlow/hardware figure from page 70 of the course textbook (François Chollet, *Deep
  Learning with Python*, 2nd edition). Deck 3A's slide 43 shows screenshots of the Google Colab
  interface. Deck 3B's slide 14 contains "Sample of the Fashion MNIST images by Yuzamei", from
  Wikimedia Commons under CC BY-SA, as credited on the slide; it remains under its own licence. None
  of deck 3B's eight slides with an OCW exclusion notice (3, 4, 6 and 8–12: fast.ai, Nirmala Murali on
  Medium, and unknown rights holders) is rendered. Deck 3A's slides 24–25 are not rendered either.
  They print no notice, but they reuse the fast.ai pixel matrix and the RGB matrices that deck 3B
  marks as excluded.
- **Lecture 4**, `raw/images/04-cnns-and-transfer-learning/` (16 images: pages 14, 18, 19, 21, 24, 29,
  31, 32, 40, 44, 45, 47, 75, 78, 79 and 80) from
  [`mit15_773_s24_lec04.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf).
  Third-party material on the rendered slides, as credited there: slides 31 and 32 carry figures
  credited "Image source: deeplearning.ai". Slide 40 is a photograph of Fig 14-2 from the book the
  slide links on Amazon, *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow*. Slide
  47 reproduces a figure from the CS231n notes it cites (cs231n.github.io/convolutional-networks).
  Slides 78–80 are screenshots of the TensorFlow Hub, PyTorch Hub and Hugging Face Hub web pages.
  OCW marks none of these as excluded; they remain their owners'. None of the deck's fourteen
  slides with an OCW exclusion notice is rendered: 41 (Honglak Lee, Roger Grosse, Rajesh Ranganath,
  Andrew Y. Ng), 52–55 (Faisal Alshuwaier, Ali Areshey, Josiah Poon, CC BY-NC-ND), 58 and 66
  (unknown rights holders), 67–69 (Matthew D. Zeiler and Rob Fergus/Springer) and 70–73 (Kaiming He,
  Xiangyu Zhang, Shaoqing Ren, Jian Sun/arXiv). Slide 77 is not rendered either. It prints no notice,
  but it redraws the ResNet 34 diagram that slides 70–73 mark as excluded. Slide 2's Fashion MNIST
  mosaic is the CC BY-SA image already rendered for lecture 3 (deck 3B, slide 14) and is not
  rendered again.
- **Lecture 5**, `raw/images/05-text-vectorization-and-bag-of-words/` (7 images: pages 21, 27, 43,
  47, 50, 56 and 65) from
  [`mit15_773_s24_lec05.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec05.pdf).
  The rendered slides are the course's own diagrams; slide 21 quotes Frederick Jelinek. Neither of
  the deck's two slides with an OCW exclusion notice is rendered: 15 ("various AI logos",
  respective copyright holders, with the LMSYS Chatbot Arena leaderboard) and 17 (Einstein GPT,
  Salesforce). Slides 3, 4 and 16 are not rendered either. They print no notice, but they are
  third-party screenshots and logo graphics of the same kind (Google autocomplete, a ChatGPT
  exchange, and the Y Combinator W23 Generative AI Landscape credited to Truewind). Slide 63's song
  lyrics are copyrighted and are neither rendered nor reproduced in the slide file.
