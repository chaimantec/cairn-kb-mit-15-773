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
- **Lecture 6**, `raw/images/06-word-embeddings/` (8 images: pages 37, 49, 51, 55, 59, 61, 67 and
  68) from
  [`mit15_773_s24_lec06.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec06.pdf).
  Slides 49 and 51 reproduce a two-dimensional plot of GloVe word vectors for ten male/female word
  pairs that the deck does not credit; the deck cites the GloVe paper
  (<https://nlp.stanford.edu/pubs/glove.pdf>) on its GloVe slides, and the plot appears to be one of
  the GloVe project's own visualizations. OCW does not mark it as excluded; it remains its authors'.
  The other rendered slides are the course's own diagrams and handwritten drawings. None of the
  deck's six slides with an OCW exclusion notice is rendered: 14, 15, 17, 18 and 19 (embedding
  quizzes, Cohere) and 47 (one-hot word vector and word embedding figures, François Chollet).
  Slide 3 is not rendered either: its handwritten one-hot table is the image already rendered for
  lecture 5 (slide 50 of that deck).
- **Lecture 7**, `raw/images/07-transformers/` (7 images: pages 33, 35, 43, 47, 49, 53 and 75) from
  [`mit15_773_s24_lec07.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec07.pdf).
  The rendered slides are the course's own diagrams; slide 53 reuses a picture of the course's own
  attention diagram from slide 52. None of the deck's eight slides with an OCW exclusion notice is
  rendered: 23, 51 and 64–69, figures from "Attention Is All You Need" (Ashish Vaswani, Noam
  Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia
  Polosukhin). Slides 21–22 are not rendered either. They print no notice, but they are screenshots
  of Google search results from Google's blog post on BERT
  (<https://blog.google/products/search/search-language-understanding-bert/>).
- **Lecture 8**, `raw/images/08-transformers-and-self-supervised-learning/` (19 images: pages 8, 21,
  23, 25, 27, 30, 35, 38, 45, 47, 60, 62, 63, 69, 70, 71, 74, 77 and 91) from
  [`mit15_773_s24_lec08.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf).
  Third-party material on the rendered slides, as credited there: slide 91 reproduces Figure 1 of the
  TabTransformer paper (<http://arxiv.org/abs/2012.06678>), which the slide marks "Figure license: CC0
  1.0". Slides 71 and 74 quote MIT Sloan's mission statement. The other rendered slides are the
  course's own diagrams. None of the deck's fifteen slides with an OCW exclusion notice is rendered:
  28 ("Attention Is All You Need", Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
  Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin), 41 and 56 (ResNet-34, Kaiming He, Xiangyu
  Zhang, Shaoqing Ren, Jian Sun/arXiv), 50 (an ImageNet training image, unknown rights holder), 51
  (Honglak Lee, Roger Grosse, Rajesh Ranganath, Andrew Y. Ng), 79–83 (the BERT paper and figures,
  Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova/ArXiv; slide 79 also credits a BERT
  cartoon to Jay Alammar under CC BY-NC-SA), 90 (the Vision Transformer figure, Alexey Dosovitskiy and
  co-authors/ArXiv), 93 (a desk-chair image, Wayfair LLC) and 96–98 (dog images, Ting Chen, Simon
  Kornblith, Mohammad Norouzi, Geoffrey Hinton/ArXiv). Slide 81's notice names a BERT figure, but the
  page shows the course's own diagram from slide 38, which carries no notice and is rendered from
  slide 38. Slides 84 and 86 are not rendered either: they print no notice, but they are the
  Sentence-Transformers logo and a screenshot of the Hugging Face Hub. Slides 42, 43 and 57 repeat the
  network diagram already rendered for lecture 4 (slide 75 of that deck), and slides 10, 13 and 29
  repeat lecture 7's slides 75, 43 and 76; none is rendered again.
- **Lecture 9**, `raw/images/09-large-language-models/` (10 images: pages 3, 10, 16, 19, 20, 26, 36,
  43, 52 and 54) from
  [`mit15_773_s24_lec09.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec09.pdf).
  The rendered slides are the course's own diagrams; slide 3 quotes MIT Sloan's mission statement.
  None of the deck's four slides with an OCW exclusion notice is rendered: 27 and 29 (figures from
  "Attention Is All You Need", Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin) and 41 and 67 (GPT tokenizer and byte pair encoding
  images and screenshots, credited to Andrej Karpathy; slide 41's upper screenshot is of Jay Mody's blog
  post "GPT in 60 Lines of NumPy"). Slides 39 and 47–50 are not rendered either: they print no notice,
  but slide 39 reproduces the GPT-3 paper's table of training data (<http://arxiv.org/abs/2005.14165>)
  and slides 47–50 are screenshots of the OpenAI Playground; their slide entries transcribe them. Slides
  2 and 37 repeat lecture 8's slides 71 and 30, and slide 59 repeats lecture 5's slide 27; none is
  rendered again.
- **Lecture 10**, `raw/images/10-adapting-llms/` (13 images: pages 3, 20, 31, 33, 39, 40, 52, 58, 77, 84,
  85, 109 and 114) from
  [`mit15_773_s24_lec10.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf).
  Twelve are the course's own diagrams. Slide 114 is the "How LLMs Are Built?" flow chart from "Large
  Language Models: A Survey" by Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard
  Socher, Xavier Amatriain and Jianfeng Gao, which the slide credits under CC BY, with the course's red
  outlines added. None of the deck's fourteen slides with an OCW exclusion notice is rendered: 59
  (Wayfair LLC), 70–73 (prompt examples from the GPT-3 paper, Tom B. Brown et al./ArXiv), 78 (a ChatGPT
  conversation, benjedwards on Twitter), 80 (a retrieval-augmented generation figure, Sebastian Raschka),
  87, 88 and 92 (rag & bone), 89 and 93 (Urban Outfitters), 95 (Llama 2 screenshots, Meta) and 113 (an
  effort-benefit curve, Andrej Karpathy on Twitter). Slide 96 repeats slide 95's Llama 2 screenshot
  without a notice and is not rendered either. Unmarked third-party material is transcribed and not
  rendered: a GPT-3 completion posted on Twitter (5–6), GPT-3, InstructGPT and ChatGPT screenshots (9–10,
  42, 44–45, 48–50, 63–64), the InstructGPT paper's title block (15), a LinkedIn post (55), OpenAI
  Playground screenshots (60–61), a figure from the "Let's think step by step" paper (65), a table from
  the "Take a deep breath" paper (66), the first page of a prompt-engineering survey (67), and an
  instruction-tuning dataset of product descriptions and reviews (91). The companion deck
  [`mit15_773_s24_lec105.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec105.pdf)
  has no images here: it prints no exclusion notices, but its rag & bone, Urban Outfitters, Llama 2 and
  effort-benefit pages reuse the material deck 10 marks as excluded, and its other figures repeat deck
  10's.
- **Lecture 11**, `raw/images/11-diffusion-models/` (16 images: pages 2, 3, 5, 11, 14, 15, 16, 20, 24, 28,
  34, 35, 37, 72, 77 and 81) from
  [`mit15_773_s24_lec11.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf).
  Third-party material on the rendered slides, as credited there. Slide 2 is a ChatGPT conversation whose
  image was generated with DALL-E 3 from the instructor's prompt ("OpenAI (2024). ChatGPT (DALL-E 3)").
  Slide 3's image was generated with Midjourney from a prompt by Alex Serban on MSPowerUser ("Midjourney
  (2024). Midjourney (V6)", <https://mspoweruser.com/best-midjourney-prompts/>). Slide 5 is a frame of a
  video generated with Sora from a prompt by Brooks, Peebles et al. at OpenAI ("OpenAI (2024). ChatGPT Plus
  (SORA)"). OCW credits these AI-generated images and does not mark them as excluded. The MIT Dome
  photograph on slides 11, 14–16, 20, 24, 28, 34, 35 and 81 is by Muzammil Soorma on Unsplash, as the
  deck credits it on most of the pages that show it. The Rotunda at UVA on slide 11 is in the public domain,
  from Wikimedia Commons. Slide 37 carries the title block of "Denoising Diffusion Probabilistic Models"
  (Jonathan Ho, Ajay Jain, Pieter Abbeel, <https://arxiv.org/abs/2006.11239>) beside the course's own
  diagram. None of the deck's 22 slides with an OCW exclusion notice is rendered: 9 (a Google Image Search
  grid of college buildings, with credits to John Simpson Architects, Michael Hickey, Getty Images, the
  University of South Carolina, Rex Harris, iStock and others), 38–41 (U-Net figures, LMB, University of
  Freiburg), 47, 48, 50 and 51 (a photo of a Labrador retriever, © unknown), 59, 62–64 and 68 (the Labrador
  and train photos, © source unknown, and a blue jay photo, © Timber Press), 70 and 73 (CLIP paper figures,
  Alec Radford et al./ArXiv), 82–85 (the latent diffusion figure, Robin Rombach et al./ArXiv), 87 (protein
  images, Namruta Anand/The New York Times Company) and 88 (the Hugging Face Diffusers logo and page).
  Unmarked third-party material is transcribed and not rendered: the title blocks of the first diffusion
  paper (36) and the CLIP paper (57), and a screenshot of lexica.art (86).
