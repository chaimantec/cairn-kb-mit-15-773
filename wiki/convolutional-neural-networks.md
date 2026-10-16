# Convolutional neural networks

A **convolutional neural network** (CNN) is the course's main tool for images. It replaces the
"flatten the image and feed it to a dense layer" approach of lecture 3 with layers designed for the
fact that the input is an image. Everything on this page comes from
[lecture 4](04-cnns-and-transfer-learning.md) (deck 04; ≈0:46–55:25) unless it says otherwise.

## Why not just flatten the image?

Flattening a $28 \times 28$ Fashion-MNIST image into 784 numbers works: one hidden layer gets into the
high 80s on accuracy ([lecture 3](03-keras-and-vision-basics.md); lecture 4, slide 2). But the network
never learns that its input is an image, and three things go wrong (slides 3–9; ≈1:32–6:14).

- **Too many parameters.** A $3{,}024 \times 3{,}024$ colour photo is about 27 million numbers, and connecting
  it to a single 100-neuron dense layer takes about 2.7 billion weights. That is computationally
  demanding, very data-hungry, and invites overfitting.
- **Lost spatial adjacency.** Neighbouring pixels belong together, for example to the same t-shirt,
  and a flat vector ignores what is nearby.
- **No learning once and reusing.** A vertical line at the top left and one at the bottom right are
  learned separately, though "a vertical line is a vertical line". Detecting a pattern wherever it
  appears is called **translation invariance**.

Convolutional layers were developed to address these shortcomings (slide 9).

## Filters and the convolution operation

A **convolutional filter** is a small square matrix of numbers, and a **convolutional layer** is one
or more filters (slides 11–12). With well-chosen numbers a filter detects a feature: rows of
1 1 1 / 0 0 0 / −1 −1 −1 detect horizontal lines, and 1 0 −1 in every row detects vertical lines
(slide 14).

Applying a filter to an image is the **convolution operation** (slides 15–21; ≈8:33–11:43):

1. Overlay the filter on the top-left patch of the image.
2. Multiply matching elements and add them up.
3. Pass the sum through a ReLU, $\max(0, \text{sum})$. This is the output's top-left cell. Strictly,
   the ReLU is not part of the convolution, but the course includes it for convenience (slide 18).
4. Slide the window one step right and repeat; at the end of a row, go to the start of the next row;
   stop at the bottom-right corner.

On the lecture's $7 \times 7$ example a $3 \times 3$ filter gives a $5 \times 5$ output. The first patch gives
$0 \cdot 1 + 0 \cdot 1 + 0 \cdot 1 + 5 \cdot 0 + 5 \cdot 0 + 5 \cdot 0 + 10 \cdot (-1) + 10 \cdot (-1) + 10 \cdot (-1) = -30$,
which the ReLU turns into 0 (slides 17–18). The horizontal-line filter responds only where a patch
is brighter on top than below. That is what makes it an **edge** detector, as the lecturer's
spreadsheet demonstration shows on a handwritten 3 (slide 22; ≈12:29–16:25).

![Slide 21 — The Convolution Operation](../raw/images/04-cnns-and-transfer-learning/slide-21.jpg)

*Lecture 4, slide 21 — sliding a 3 × 3 filter over a 7 × 7 image gives a 5 × 5 output; this filter fires (30) only where brightness falls from top to bottom.*

Two refinements are named and set aside: **padding** the image's border so the filter can go past
the edge, and a **stride** of more than one step. The default of no padding and one step "works
well, amazingly well, almost all the time" (≈11:43).

## Layers, depth and colour

Each filter in a layer is a **specialist** for one feature, but you do not assign the
specialities. The network learns them, so there is no human bottleneck (slides 23–24; ≈17:13). The
filters run in parallel on the same input, and their outputs stack into a [tensor](tensors.md): two
filters on the $7 \times 7$ image give $5 \times 5 \times 2$, or equally $2 \times 5 \times 5$ (slides 25–29).

For a colour image, such as $6 \times 6 \times 3$, the filter is given **the same depth as the input**, $3 \times 3 \times 3$, so that the three channel values at a pixel are treated as one thing. You still multiply the 27
pairs, add and apply the ReLU, and each filter still produces a single 2D table, here $4 \times 4$ (slides
30–31; ≈19:33–22:38). With $f$ filters the layer's output is $4 \times 4 \times f$ (slide 32). Textbook
chapter 8.1 has more (slide 33).

## The big idea: filters are learned

Filters used to be designed by hand by computer-vision researchers. The breakthrough was to treat
the numbers in a filter as **weights** and learn them from data with backpropagation, like every
other weight (slides 34–36; ≈25:45–26:32). That works because a filter is a neuron, a special one
connected only to a small region of the input and sharing its weights across every window. So
layers, loss functions and gradient descent all apply unchanged (slides 37–38 and appendix slide
83; see [anatomy of a neural network](neural-network-anatomy.md)). It was a turning point (slide
39). The lecturer dates the unlock to AlexNet in the 2012 ImageNet competition (≈28:09–28:56).

## Stacking layers builds a hierarchy

Each succeeding convolutional layer sees more of the original image, because its input is the
previous layer's output (slide 40; ≈28:56–29:43). So a deep stack learns increasingly complex
features: lines, then edges and parts, then whole faces (slide 41; ≈29:43–30:29). This hierarchy is
why a network trained on a large image collection is worth reusing; see
[transfer learning](transfer-learning.md) and [representation learning](representation-learning.md).

## Pooling layers

A **pooling layer**, also called a down-sampling or subsampling layer, shrinks the tensor coming out
of a convolutional layer, forcing the network to summarize. It has **no weights** (slide 43;
≈30:29–31:15). **Max pooling** keeps the largest number in each $2 \times 2$ box. **Average pooling** keeps
the average. On slide 44's $4 \times 4$ grid, max pooling gives 43, 109 / 105, 35, and average pooling
gives 22.2, 45.5 / 70.75, 21. $2 \times 2$ pooling removes 75% of the entries (slide 46). It acts on each
depth slice separately: $224 \times 224 \times 64$ becomes $112 \times 112 \times 64$ (slide 47).

![Slide 44 — 2x2 max pooling](../raw/images/04-cnns-and-transfer-learning/slide-44.png)

*Lecture 4, slide 44 — 2 × 2 max pooling keeps the largest number in each colour-coded box.*

Max pooling works like an **"OR" condition**: if a feature lit up anywhere in the region, the
output says so. It is a feature detector that moves the network to a higher level of abstraction
(slides 48–50; ≈33:37–35:56).

## The architecture of a basic CNN

A basic CNN is a series of **convolutional blocks**, each typically one or two convolutional layers
followed by a pooling layer. Then the final tensor is **flattened** and sent through zero or more
dense layers to the output layer (slides 52–55; ≈38:19–43:42). Through the blocks the tensor gets
**smaller in height and width but deeper**. When ResNet finishes with a $224 \times 224 \times 3$ image, its
final tensor is $7 \times 7 \times 2{,}048$ (≈39:04–40:37). The depth is a design choice, the number of filters per
layer, and later layers get more filters because the ways to combine a few atomic features are
combinatorial, "10 kinds of atoms. How many molecules can I make from it?" (≈40:37–44:30).

How many layers and filters to use is typically trial and error: "start simple. And if it works,
stop working on it. If it doesn't, add more layers". In practice it is usually better still to start
from a pre-trained network (≈49:08–49:55).

## In Keras

The course's Keras CNN for Fashion MNIST, as narrated (lecture 4 ≈45:15–52:15; the notebook is not
on OCW):

- Grayscale images get an explicit channel dimension, $28 \times 28 \times 1$, because convolutional layers
  expect one.
- `layers.Conv2D` with 32 filters, `kernel_size` 2 by 2 and ReLU, then `MaxPool2D`, repeated as a
  second block. Then flatten, a dense layer of 256 and a 10-way softmax.
- About 302,000 parameters. Trained with Adam for 10 epochs with a 20% validation split, it reaches
  **90.5%** on the test set, against the high 80s for the flattened network.

The same pattern with a single sigmoid output, and a `Rescaling` layer in place of dividing by 255,
classifies colour photos of handbags and shoes at 87% from fewer than 100 training images
(≈58:32–1:03:11). See [Keras](keras.md).

## Where this comes up

- [Lecture 3](03-keras-and-vision-basics.md): images as pixel tables, and the flattened network
  this replaces.
- [Lecture 4](04-cnns-and-transfer-learning.md): everything above, plus the Colab walkthroughs.
- [Lecture 7](07-transformers.md): a transformer's attention heads are compared to a convolutional
  layer's filters, each learning its own pattern, and stacked encoders to the hierarchy of layers
  (slide 49; ≈37:55, ≈1:02:34). Transformers now also do well on computer-vision problems (≈0:46).
- [Lecture 8](08-transformers-and-self-supervised-learning.md): ResNet, the Residual Net, named for the residual connections that transformers
  also use (≈31:12); more attention heads compared to more filters (≈28:04); and the Vision
  Transformer, which classifies an image from its patches with a transformer.
- See also [images and computer-vision tasks](images-and-vision-tasks.md) and
  [transformers](transformers.md).
