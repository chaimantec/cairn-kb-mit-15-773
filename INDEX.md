# 15.773 — Hands-On Deep Learning (MIT, Spring 2024)

MIT Sloan's hands-on introduction to deep learning, taught by **Rama Ramakrishnan** (the lecture 1
title slide credits "Farias, Ramakrishnan") and published on
[MIT OpenCourseWare](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/). Over
11 lectures it goes from neural-network basics and training, through computer vision (CNNs,
transfer learning) and natural language (embeddings, transformers), to generative AI (LLMs, RAG,
parameter-efficient fine-tuning, text-to-image diffusion). It deliberately "skips the math" and
focuses on concepts and on coding models in Keras/TensorFlow.

**Coverage: lectures 1–4 of 11.** This knowledge base is being built lecture by lecture. Only
lectures 1 to 4 have wiki pages, transcripts and slide transcriptions so far. Do not cite it as
covering lectures 5–11. For anything after lecture 4, including everything on language (text
vectorization, embeddings, transformers) and generative AI, say that the KB does not cover it yet.

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
  and Hugging Face model hubs. Dropout is used but not explained. The catalog title ("Transfer
  Learning and Fine-Tuning; Intro to HuggingFace") understates the CNN half.

### Topics

- [AI, machine learning, deep learning and generative AI](wiki/ai-ml-dl-genai.md): the nested
  circles the course uses, what bottleneck each layer removed, and X-to-Y multimodality.
- [Representation learning](wiki/representation-learning.md): why raw unstructured data needs a
  representation, hand-made features versus learned ones, hidden layers as the learned
  representation, the feature hierarchy in a CNN, and reusing a pre-trained network's
  representation.
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
  (minibatch) gradient descent, and epochs, batches and batch size.
- [Overfitting and regularization](wiki/overfitting-and-regularization.md): training versus validation
  error, the sweet spot, why deep networks need regularization, the validation and test sets, early
  stopping in practice, dropout (previewed, then used without explanation), and the remedies for
  little data.
- [Tensors](wiki/tensors.md): ranks 0 to 5 with the course's examples (scalar, vector, table, image,
  video), thinking of a tensor as a list of lower-rank tensors, and the shapes convolutional and
  pooling layers produce.
- [Preparing data for a network](wiki/data-preprocessing.md): class balance and a baseline model,
  one-hot encoding, standardization, splitting before standardizing, NumPy arrays, and random seeds.
- [Images and computer-vision tasks](wiki/images-and-vision-tasks.md): pixels and 0–255 intensities,
  colour channels, classification, localization, object detection, semantic and instance
  segmentation, flattening an image into a vector and why that falls short, and loading colour
  images from one folder per class.
- [Convolutional neural networks](wiki/convolutional-neural-networks.md): the three problems with
  flattening, filters and the convolution operation, layers and tensor shapes, colour images,
  learned filters, the feature hierarchy, max and average pooling, the basic CNN architecture, and
  the course's Keras CNN.
- [Transfer learning](wiki/transfer-learning.md): the two research trends behind it, ImageNet and
  ResNet, headless ResNet as a feature extractor, fine-tuning, the model hubs, and the
  handbags-vs-shoes results.
- [Data augmentation](wiki/data-augmentation.md): perturbing images (flips, small rotations, zooms)
  to stretch a small data set without changing what they mean, and the Keras layers for it.
- [Multi-class classification and softmax](wiki/multiclass-classification.md): why ten sigmoids fail,
  the softmax formula, its role in GPT-4, sparse versus one-hot labels and the matching Keras loss,
  and the table of output layers and losses.
- [Backpropagation](wiki/backpropagation.md): computational graphs and the chain rule, why it is
  fast, GPUs, and the lecturer's handwritten worked example.
- [Keras](wiki/keras.md): what TensorFlow and Keras each provide, why not PyTorch, the three APIs,
  defining a model with `keras.Input`, `Dense` layers and `keras.Model`, training it with
  `summary`, `compile`, `fit` and `evaluate`, and the image layers (`Conv2D`, `MaxPool2D`,
  `Rescaling`, augmentation) and `include_top=False`.

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
  prints one. Lecture 4's deck prints its own page number wherever it prints one. The Colab
  notebooks of lectures 3 and 4 are not published on OCW; the KB knows them only from the
  lecturer's narration.
- [`raw/images/`](raw/images/): rendered figure slides (see below).
- [`sources.md`](sources.md): every course document on the OCW site with its canonical URL. The
  slide PDFs are not in this repo; link to them at their OCW URL.
- [`LICENSE.md`](LICENSE.md): CC BY-NC-SA 4.0, following OCW, with the attribution of every image.
- [`SEE_ALSO.md`](SEE_ALSO.md): sibling knowledge bases worth consulting, with their repo URLs to
  pass as `kb`.

## Images

**Lectures 1, 2, 3 and 4 have images. No other lecture does yet.** Their figure slides are rendered
as whole slides and shown inline, both in the lecture pages ([lecture 1](wiki/01-introduction.md),
[lecture 2](wiki/02-training-deep-nns.md), [lecture 3](wiki/03-keras-and-vision-basics.md),
[lecture 4](wiki/04-cnns-and-transfer-learning.md)) and under the matching `## Slide N` heading in
each slide file. Lecture 1 has 45. Lecture 2 has 18: 15 from its main deck and 3 pages of the
handwritten backprop example, which are named `backprop-N` rather than `slide-N`. Lecture 3 has 13 in
one directory: 9 from deck 3A named `3a-N` and 4 from deck 3B named `3b-N`. Lecture 4 has 16. Slides
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
