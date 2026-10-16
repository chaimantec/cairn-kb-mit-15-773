# 15.773 — Hands-On Deep Learning (MIT, Spring 2024)

MIT Sloan's hands-on introduction to deep learning, taught by **Rama Ramakrishnan** (the lecture 1
title slide credits "Farias, Ramakrishnan") and published on
[MIT OpenCourseWare](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/). Over
11 lectures it goes from neural-network basics and training, through computer vision (CNNs,
transfer learning) and natural language (embeddings, transformers), to generative AI (LLMs, RAG,
parameter-efficient fine-tuning, text-to-image diffusion). It deliberately "skips the math" and
focuses on concepts and on coding models in Keras/TensorFlow.

**Coverage: lectures 1–9 of 11.** This knowledge base is being built lecture by lecture. Only
lectures 1 to 9 have wiki pages, transcripts and slide transcriptions so far. Do not cite it as
covering lectures 10–11. On language it covers lectures 5 (text vectorization and the Bag of Words
model), 6 (word embeddings and GloVe), 7 (the transformer encoder: self-attention, multi-head
attention, positional encoding, applied to slot filling), 8 (the encoder completed with key, query
and value matrices, residual connections and layer normalization; the `<CLS>` token; self-supervised
learning and BERT; Hugging Face pipelines; the Vision Transformer and TabTransformer) and 9 (the first
of two lectures on large language models: next word prediction, causal self-attention and the
decoder, text generation and GPT-3, decoding and sampling, byte pair encoding). Retrieval-augmented
generation is in the title of video 9 but not in the lecture; it belongs to lecture 10. For anything
after lecture 9, including adapting LLMs (prompting, retrieval-augmented generation, fine-tuning),
parameter-efficient fine-tuning and text-to-image models, say that the KB does not cover it yet.

## Contents

### Lectures

- [Lecture 1 — Introduction to Neural Networks and Deep Learning](wiki/01-introduction.md): how AI,
  machine learning, deep learning and generative AI relate (rules → learning from examples →
  learned representations → generating unstructured data); "put DL behind every sensor"; then a
  neural network built from logistic regression: weights and biases, neurons, layers, activation
  functions (sigmoid, linear, ReLU), designing a DNN, counting the 13 parameters of a 2-3-1 network,
  and a worked forward pass. Training is *not* covered here despite the video's catalog title.
  Also summarises the deck-only course logistics (prerequisites, grading, course philosophy).
- [Lecture 2 — Training Deep Neural Networks](wiki/02-training-deep-nns.md): the heart-disease
  case study (Cleveland Clinic data, 29 one-hot inputs, 16 ReLU units, sigmoid output, 497
  parameters) and its four-line Keras model; then training: loss functions (mean squared error,
  binary cross-entropy built from −log p), gradient descent on a toy quartic and in many variables,
  the learning rate, local minima and saddle points, backpropagation with the handwritten worked
  example, weight initialization, stochastic (minibatch) gradient descent and Adam, and the
  training loop. The fuller Keras/TensorFlow introduction, the Colab and regularization are *not*
  in this recording, despite the catalog title; they open lecture 3.
- [Lecture 3 — Keras and TensorFlow, Training the Heart-Disease Model, and Computer Vision
  Basics](wiki/03-keras-and-vision-basics.md): two decks. Deck 3A: epochs and batches (194 samples in
  batches of 32 make 7), underfitting and overfitting, early stopping (dropout only previewed),
  tensors and their ranks, what TensorFlow and Keras provide, the functional API, then the
  heart-disease Colab (baseline 72.6%, one-hot encoding, standardization after the train/test split,
  `compile`, `fit` for 300 epochs with a 20% validation split, loss and accuracy curves, 83.6% test
  accuracy). Deck 3B: images as 0–255 pixel tables and RGB channels, five vision tasks (classification
  to instance segmentation), Fashion MNIST, flattening 28 × 28 to 784, softmax, and matching the
  Keras cross-entropy loss to the label encoding. Convolutional networks are *not* reached, despite
  the catalog title.
- [Lecture 4 — Convolutional Neural Networks and Transfer Learning](wiki/04-cnns-and-transfer-learning.md):
  why flattening an image fails (2.7 billion weights for one phone photo into 100 neurons), the
  convolutional filter and the convolution operation worked on a 7 × 7 example, the fast.ai
  spreadsheet demo, convolutional layers and colour images (filters as deep as the input), learning
  the filters as weights, later layers seeing more of the image and learning hierarchical features, max and average
  pooling, the basic CNN architecture, and a Fashion-MNIST CNN in Colab (90.5%, with a hand check of
  its 302,000 parameters). Then a handbags-vs-shoes classifier from under 100 images per class (87%
  from scratch), data augmentation (Colab only), transfer learning with ImageNet and a headless
  ResNet (100% on the test set, and a live webcam demo), fine-tuning, and the TensorFlow, PyTorch
  and Hugging Face model hubs. Dropout is used but not explained until lecture 5. The catalog title
  ("Transfer Learning and Fine-Tuning; Intro to HuggingFace") understates the CNN half.
- [Lecture 5 — Text Vectorization and the Bag-of-Words Model](wiki/05-text-vectorization-and-bag-of-words.md):
  the start of the NLP sequence. Why text matters and what "text in, text out" covers (classification,
  extraction, summarization, generation, code, question-answering, call centers), the arc of NLP from
  hand-crafted rules through statistics and RNNs to transformers, and text as "fancy regression".
  Then text vectorization in four steps, STIE (standardize, tokenize, index, encode), the vocabulary,
  the `<UNK>` token and one-hot vectors; "the cat sat on the mat" as a 6 × 100 table; count and
  multi-hot encoding, which make the Bag of Words model, and its shortcomings. In Colab, Keras's text
  vectorization layer and a song-genre classifier (hip-hop, rock or pop) with eight ReLUs and a
  softmax: 72% on the test set against a 55% baseline, 75% with bigrams. Dropout is finally
  explained. Byte-pair encoding is deferred to the LLM lectures and explained in lecture 9.
- [Lecture 6 — Word Embeddings](wiki/06-word-embeddings.md): why one-hot vectors fail (as long as the
  vocabulary; any two are $\sqrt{2}$ apart), what word vectors should do (close for related words,
  directions for relationships; the apple and cow quizzes), and why context needs transformers
  (stand-alone versus contextual embeddings). Then GloVe built from scratch: "You shall know a word
  by the company it keeps", the word-word co-occurrence matrix from Wikipedia's sentences, the model
  $\log(X_{ij}) = b_i + b_j + w_i^T w_j$ fitted by least squares and stochastic gradient descent, the
  embedding length as a hyperparameter, t-SNE plots and word vector algebra,
  $(\text{brother} - \text{man}) + \text{woman} = \text{sister}$, and pretrained versus your own
  embeddings. In Keras, `TextVectorization` with
  integer output and padding, the `Embedding` lookup table and `GlobalAveragePooling1D`; in Colab, the
  song-genre classifier on frozen GloVe (63%, 835 of 500,835 parameters trainable), GloVe trained
  further (almost 69%) and 64-dimensional embeddings from scratch (72%).
- [Lecture 7 — Transformers](wiki/07-transformers.md): transformers across domains, and the task that
  motivates them: turning a travel query into a structured query by tagging every word of ATIS queries
  with one of 123 slots (B-, I- and O labels), a 123-way softmax per word. The three requirements
  (output as long as the input, context, order); BERT's effect on Google search. Self-attention built
  step by step on "The train slowly left the station": weights on the other words, the weighted
  average $\hat w_6$, dot products as relatedness, the softmax that makes them weights. Multi-head
  attention (concatenate and project), feed-forward ReLU layers, why self-attention ignores order,
  positional encoding worked on "cat sat mat", the transformer encoder and stacking. Slot filling with
  an encoder, and the Colab set-up (888-token vocabulary, 512-dimensional embeddings, five heads,
  about 5.3 million encoder parameters, hand-checked on the page); the model is not trained in this
  lecture. Attention's learned projections, residual connections and layer normalization are left for
  lecture 8.
- [Lecture 8 — Transformers (2/2) and Self-Supervised Learning](wiki/08-transformers-and-self-supervised-learning.md):
  lecture 7's slot-filling model trained (99% test accuracy, 93% on the non-O slots); self-attention
  as matrix multiplication, $\text{softmax}(XX^T)X$, drawn on an iPad; making it tunable with key,
  query and value matrices $A^K$, $A^Q$, $A^V$, each head with its own, up to the paper's
  $\text{softmax}(QK^T/\sqrt{d_k})\thinspace V$; residual connections; layer normalization; the
  complete, stackable encoder and what backpropagation optimizes. Then sequence classification,
  labeling and generation, and the `<CLS>` token; neural networks as representation learners and the
  transfer-learning recipe; self-supervised learning by masking, which gives BERT (BERT-Base 12
  layers, 768, 12 heads; BERT-Large 24, 1024, 16); the Hugging Face Hub and a Colab of `pipeline`
  sentiment, named-entity and question-answering demos; and transformers beyond text: the Vision
  Transformer, the TabTransformer and a multi-modal classifier. The deck's contrastive-learning slides
  (94–98) are not presented and are summarized from the slides alone.
- [Lecture 9 — Large Language Models (1/2): Next Word Prediction, Decoding and Tokenization](wiki/09-large-language-models.md):
  next word prediction as a second self-supervised task; why a plain encoder cheats by seeing the next
  word; causal (masked) self-attention, which zeroes the weights of future words; the Transformer Causal
  Encoder, also called the Transformer Decoder; generating text one word at a time; autoregressive
  language models and GPT-3 (96 layers of 96 heads, about 30 billion sentences); decoding: greedy
  decoding versus random sampling, the head and the tail, GPT-3.5 going "off the rails", top-K and top-P
  (nucleus) sampling, temperature (with the formula) and the OpenAI Playground demo; and byte pair
  encoding worked through on "the cat sat on the mat", with the token visualizer. RAG, in the video's
  catalog title, is not in this lecture.

### Topics

- [AI, machine learning, deep learning and generative AI](wiki/ai-ml-dl-genai.md): the nested
  circles the course uses, what bottleneck each layer removed, and X-to-Y multimodality.
- [Representation learning](wiki/representation-learning.md): why raw unstructured data needs a
  representation, hand-made features versus learned ones, hidden layers as the learned
  representation, the feature hierarchy in a CNN, reusing a pre-trained network's
  representation, word embeddings as the learned representation of words, and a deep network as
  many encoders whose representations capture general knowledge.
- [Anatomy of a neural network](wiki/neural-network-anatomy.md): the vocabulary (weight, bias,
  neuron, layer, dense, feedforward, architecture), what you design versus what training finds,
  parameter counting, the forward pass with numbers, and the convolutional filter as a neuron.
- [Activation functions](wiki/activation-functions.md): sigmoid, linear and ReLU with formulas and
  plots, the rule of thumb (ReLU for hidden layers, output activation dictated by the output), the
  softmax, and why ReLU's gradient of 1 keeps gradients flowing.
- [Training a neural network](wiki/training-neural-networks.md): what training means, its pieces
  (data, loss, optimizer), the training loop, the course's checklist and defaults, training,
  validation and test sets, weight initialization, and what the weights are worth.
- [Loss functions](wiki/loss-functions.md): what a loss measures, mean squared error, binary
  cross-entropy derived step by step, categorical cross-entropy and the Keras loss for each label
  encoding, the output-layer/loss table, and why accuracy and loss can disagree.
- [Gradient descent](wiki/gradient-descent.md): the derivative's three cases, the update rule, the
  learning rate, gradients in many variables, local minima and saddle points, stochastic
  (minibatch) gradient descent, epochs, batches and batch size, and fitting models that are not
  neural networks, such as GloVe.
- [Overfitting and regularization](wiki/overfitting-and-regularization.md): training versus validation
  error, the sweet spot, why deep networks need regularization, the validation and test sets, early
  stopping in practice, dropout (previewed in lecture 3, used in lecture 4, explained in lecture 5),
  the remedies for little data, why a longer input overfits, and why noise matters even without a
  test set.
- [Tensors](wiki/tensors.md): ranks 0 to 5 with the course's examples (scalar, vector, table, image,
  video), thinking of a tensor as a list of lower-rank tensors, and the shapes convolutional and
  pooling layers produce.
- [Preparing data for a network](wiki/data-preprocessing.md): class balance and a baseline model,
  one-hot encoding, standardization (and layer normalization, its counterpart inside a transformer),
  splitting before standardizing, NumPy arrays, and random seeds.
- [Images and computer-vision tasks](wiki/images-and-vision-tasks.md): pixels and 0–255 intensities,
  colour channels, classification, localization, object detection, semantic and instance
  segmentation, flattening an image into a vector and why that falls short, loading colour
  images from one folder per class, and the Vision Transformer's patches.
- [Convolutional neural networks](wiki/convolutional-neural-networks.md): the three problems with
  flattening, filters and the convolution operation, layers and tensor shapes, colour images,
  learned filters, the feature hierarchy, max and average pooling, the basic CNN architecture, and
  the course's Keras CNN.
- [Transfer learning](wiki/transfer-learning.md): the two research trends behind it, ImageNet and
  ResNet, headless ResNet as a feature extractor, fine-tuning, the model hubs, the
  handbags-vs-shoes results, pretrained word embeddings (GloVe) as the same idea for text, why it
  works and the general recipe, and pretrained language models (BERT, Hugging Face pipelines).
- [Data augmentation](wiki/data-augmentation.md): perturbing images (flips, small rotations, zooms)
  to stretch a small data set without changing what they mean, the Keras layers for it, and augmented
  pairs in contrastive learning.
- [Multi-class classification and softmax](wiki/multiclass-classification.md): why ten sigmoids fail,
  the softmax formula, why softmax rather than the maximum, its role in GPT-4, a softmax per position
  in a language model and temperature, sparse versus one-hot labels and the matching Keras loss, and
  the table of output layers and losses.
- [Backpropagation](wiki/backpropagation.md): computational graphs and the chain rule, why it is
  fast, GPUs, and the lecturer's handwritten worked example.
- [Keras](wiki/keras.md): what TensorFlow and Keras each provide, why not PyTorch, the three APIs,
  defining a model with `keras.Input`, `Dense` layers and `keras.Model`, training it with
  `summary`, `compile`, `fit` and `evaluate`, the image layers (`Conv2D`, `MaxPool2D`,
  `Rescaling`, augmentation) and `include_top=False`, the text vectorization layer (`adapt`,
  `get_vocabulary`, `max_tokens`, `ngrams`), `Dropout` and KerasTuner, and the layers for
  embeddings (`output_mode="int"` with `output_sequence_length`, `Embedding`,
  `GlobalAveragePooling1D`, freezing a layer), and lecture 7's transformer encoder and positional
  embedding layer, trained in lecture 8.
- [Natural language processing](wiki/natural-language-processing.md): why text matters, the range
  of "text in, text out" applications, how to represent text, from one-hot vectors to stand-alone
  and contextual embeddings, the transformer encoder and ATIS slot filling, sequence classification,
  labeling and generation, BERT and Hugging Face pipelines, causal language models for generation, the
  arc of NLP progress from rules to transformers, and the course's NLP roadmap (lectures 5–10).
- [Text vectorization](wiki/text-vectorization.md): standardize, tokenize, index, encode (STIE);
  stop words and stemming; the vocabulary and the `<UNK>` token; one-hot vectors and the vocabulary
  size; choosing `max_tokens`; Keras's text vectorization layer, including integer output,
  padding and truncation; and byte pair encoding, the subword tokenization LLMs use instead.
- [The Bag of Words model](wiki/bag-of-words.md): why a sentence's table is not flattened, count
  versus multi-hot encoding, what the bag throws away, bigrams and n-grams, and the song-genre
  results, with lecture 6's embedding models alongside.
- [Word embeddings](wiki/word-embeddings.md): what is wrong with one-hot vectors, what embeddings
  should do, GloVe (the co-occurrence matrix, the model, the least-squares fit), word vector algebra,
  the embedding length, pretrained versus your own, the Keras layers, and stand-alone versus
  contextual embeddings, which lecture 7 computes by self-attention.
- [Transformers](wiki/transformers.md): self-attention as a softmax-weighted average of embeddings
  and as matrix multiplication, key, query and value matrices and the paper's attention formula,
  multi-head attention, feed-forward layers, positional encoding, residual connections and layer
  normalization, the transformer encoder and stacking, slot filling, the `<CLS>` token, BERT, causal
  self-attention and the decoder, and transformers for images, tables and multi-modal data.
- [Self-supervised learning](wiki/self-supervised-learning.md): why text needs labels it does not
  have, masking ("fill in the blanks"), why it learns a representation, masked learning as sequence
  labeling with a transformer, BERT, next word prediction (the task behind GPT), and contrastive
  learning for images (from lecture 8's slides only).
- [Large language models](wiki/large-language-models.md): next word prediction, why the plain encoder
  cheats, causal self-attention and the Transformer Causal Encoder (decoder), generating text one word
  at a time, autoregressive language models, GPT-3's size and training data, what an LLM's answer is
  and is not, and tokens rather than words.
- [Decoding and sampling](wiki/decoding-and-sampling.md): choosing the next token from the softmax:
  greedy decoding versus random sampling and when to use each, the head and the tail of the
  distribution, top-K and top-P (nucleus) sampling with the slides' worked numbers, temperature and its
  formula, and the OpenAI Playground's settings.

## Raw materials

- [`raw/transcripts/`](raw/transcripts/): one transcript per lecture, in ~45-second paragraphs each
  opening with an `[MM:SS]` timestamp. **Read and quote these.** They are OCW's human-made captions
  with a light copy-edit, and every change is listed in the file's header. The unedited caption text
  is in [`raw/transcripts/original/`](raw/transcripts/original/), for reference only.
- [`raw/slides/`](raw/slides/): every slide of each lecture's deck, transcribed in order under
  `## Slide N` headings, with equations, tables and figure descriptions. **Slide N is PDF page N.**
  Some lecture 1 pages also print a stale number in the corner (page 13 prints "20"), and from page
  50 on lecture 2's deck prints a number one higher than the page; each slide file says so. Cite the
  heading number. Lecture 2 also has a second, handwritten deck, the backprop worked example, in
  its own file, [`raw/slides/02-backprop-example.md`](raw/slides/02-backprop-example.md); cite it as
  "page N of the backprop example". Lecture 3's video covers two decks, each with its own file:
  deck 3A in [`raw/slides/03-keras-tensorflow.md`](raw/slides/03-keras-tensorflow.md) and deck 3B in
  [`raw/slides/03-vision-basics.md`](raw/slides/03-vision-basics.md). Cite them as "deck 3A, slide N"
  and "deck 3B, slide N". From page 16 on, deck 3B prints a number one higher than the page where it
  prints one. The decks of lectures 4, 5, 6, 8 and 9 print their own page number wherever they print one.
  From page 10 on, lecture 7's deck prints a number one higher than the page.
  The Colab notebooks of lectures 3 to 8 are not published on OCW; the KB knows them only from
  the lecturer's narration.
- [`raw/images/`](raw/images/): rendered figure slides (see below).
- [`sources.md`](sources.md): every course document on the OCW site with its canonical URL. The
  slide PDFs are not in this repo; link to them at their OCW URL.
- [`LICENSE.md`](LICENSE.md): CC BY-NC-SA 4.0, following OCW, with the attribution of every image.
- [`SEE_ALSO.md`](SEE_ALSO.md): sibling knowledge bases worth consulting, with their repo URLs to
  pass as `kb`.

## Images

**Lectures 1 to 9 have images. No other lecture does yet.** Their figure slides are rendered
as whole slides and shown inline, both in the lecture pages ([lecture 1](wiki/01-introduction.md),
[lecture 2](wiki/02-training-deep-nns.md), [lecture 3](wiki/03-keras-and-vision-basics.md),
[lecture 4](wiki/04-cnns-and-transfer-learning.md),
[lecture 5](wiki/05-text-vectorization-and-bag-of-words.md),
[lecture 6](wiki/06-word-embeddings.md), [lecture 7](wiki/07-transformers.md),
[lecture 8](wiki/08-transformers-and-self-supervised-learning.md),
[lecture 9](wiki/09-large-language-models.md)) and under the
matching `## Slide N` heading in
each slide file. Lecture 1 has 45. Lecture 2 has 18: 15 from its main deck and 3 pages of the
handwritten backprop example, which are named `backprop-N` rather than `slide-N`. Lecture 3 has 13 in
one directory: 9 from deck 3A named `3a-N` and 4 from deck 3B named `3b-N`. Lecture 4 has 16.
Lecture 5 has 7, mostly hand-drawn diagrams (the one-hot table, count and multi-hot encoding, the
song classifier); its third-party screenshots and logo graphics (slides 3, 4, 15–17) are not
rendered. Lecture 6 has 8: the co-occurrence matrix, the GloVe word-pair plot (twice, with its two
arrows), and the Keras diagrams (the layer split, padding, the Embedding table, average pooling and
the overall flow). Its Cohere embedding quizzes (slides 14–19) and Chollet's one-hot and embedding
figure (slide 47) are excluded by OCW and described in prose only. Lecture 7 has 7: self-attention as
a weighted sum and as softmax weights, the stand-alone and contextual vectors of "station", the pull
of "train" on "station", two attention heads concatenated and projected, the encoder stack before
positional encoding, and the slot-filling pipeline. Its figures from "Attention Is All You Need"
(slides 23, 51 and 64–69), including the paper's transformer architecture diagram, are excluded by
OCW, and its Google search screenshots (slides 21–22) are not rendered; their slide entries describe
them in prose. Lecture 8 has 19, all the course's own diagrams except the TabTransformer figure (slide
91, CC0): positional input embeddings, attention heads with their own matrices, the residual
connection, layer normalization, the stackable encoder, the three use cases, a sentence embedding and
the `<CLS>` token, representations and encoders, chopping off and fine-tuning a network, masking (three
slides), extracting an encoder, and masked learning as sequence labeling. Its figures from "Attention Is All You Need" and the BERT,
Vision Transformer and SimCLR papers, the ResNet-34 diagram, the learned-feature grids and a product
photo (slides 28, 41, 50, 51, 56, 79–83, 90, 93, 96–98) are excluded by OCW, and its Hugging Face Hub
screenshot (slide 86) is not rendered. For lecture 8's slot-filling and self-attention review slides
(10, 13, 29), use lecture 7's images, which they repeat. Lecture 9 has 10, all the course's own diagrams:
next word prediction, the next-word-prediction pipeline, the problem of seeing the next word, causal
self-attention (two slides), the Transformer Causal Encoder, generating text, decoding, and the top-K and
top-P bar charts. Its figures from "Attention Is All You Need" (slides 27 and 29) and its screenshots
about building GPT and BPE (slides 41 and 67) are excluded by OCW; the GPT-3 paper's training-data table
(slide 39) and the OpenAI Playground screenshots (slides 47–50) are not rendered, and their slide entries
transcribe them. For lecture 9's masking and use-case review slides (2 and 37), use lecture 8's images of
slides 71 and 30, which they repeat. The byte pair encoding slides (61–65) are text and tables, with no
image. Slides
whose content OCW excludes from its licence (marked `*OCW notice: …*`) are never rendered, and
neither are slides that reuse an excluded figure without a notice of their own: deck 3A's slides
24–25 and lecture 4's slide 77, whose ResNet diagram slides 70–73 exclude. So lecture 4 has no image
of a CNN architecture diagram, ResNet, ImageNet or the learned-feature visualizations; their slide
entries describe them in prose.

- **Use an image URL you have actually read in a file.** Never construct one from the pattern, and
  never assume a slide has an image because a neighbouring one does. The extension varies (`.jpg`
  or `.png`), and many slides are deliberately not rendered.
- To show an image, read its relative path (for example
  `raw/images/01-introduction/slide-100.png`) and use the URL that comes back.
- For numbers, prefer the transcription to the picture. Show one image, not a gallery, and keep the
  slide citation beside it.
