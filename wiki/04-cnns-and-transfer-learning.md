# Lecture 4 — Convolutional Neural Networks and Transfer Learning

**Lecturer:** Rama Ramakrishnan ·
**Video:** [youtube.com/watch?v=8xh7Y0pBrCE](https://www.youtube.com/watch?v=8xh7Y0pBrCE) (76 min) ·
**Slides:**
[`mit15_773_s24_lec04.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)
(84 pages; transcribed in [`raw/slides/04-cnns-and-transfer-learning.md`](../raw/slides/04-cnns-and-transfer-learning.md)) ·
**Transcript:** [`raw/transcripts/04-cnns-and-transfer-learning.md`](../raw/transcripts/04-cnns-and-transfer-learning.md)

## What this lecture establishes

The lecturer promises at the start to take the class "from not knowing anything about convolutions
… to actually knowing how convolutional networks work, and actually to build one, and demo it in
class" (≈0:00). Flattening an image into a long vector throws away what makes it an image, and a
dense layer on a phone photo would need billions of weights. A **convolutional filter** is a small
square matrix of numbers. Slid across an image, multiplying and adding at each position, it lights
up wherever a feature such as a horizontal line appears. A **convolutional layer** is a set of such
filters, and the big idea is that their numbers are not designed by hand but **learned** by
backpropagation like any other weight, because a filter is just a special neuron. Stacked layers
see more and more of the image and learn increasingly complex features. **Pooling layers** shrink
the tensor between them, and a basic **CNN** is a series of convolutional blocks followed by a
flatten, dense layers and an output layer. In Colab, a two-block CNN lifts Fashion MNIST from the
high 80s to 90.5%.

The second half turns to colour images and a deliberately tiny data set: about 100 web-scraped
photos each of handbags and shoes. A CNN built from scratch scores 87% on the test set. **Data
augmentation** is introduced in the Colab as a way to stretch a small data set. Then comes
**transfer learning**: take a network that someone has already trained on ImageNet's millions of
everyday images, here ResNet, cut off its last layer, and use what comes out as a "smart
representation" of each image for a small network of your own. That model reaches 100% on the test
set and correctly classifies a shoe and a handbag held up to the webcam in class (≈1:13:14–1:15:32).
Fine-tuning the whole network end to end is described and left to homework 1.

The catalog titles this video "Deep Learning for Computer Vision – Transfer Learning and
Fine-Tuning; Intro to HuggingFace". The recording covers more than that: it builds convolutional
networks from scratch first, and the Hugging Face Hub appears only as one of three places to find
pre-trained models (slide 80). It presents deck 04 from slide 2 to slide 81. The two appendix
slides (82–83) are pointed to but not presented (≈27:18). The lecture assumes the class has watched
a video walkthrough, posted the day before and not part of OCW, in which a single-hidden-layer
network was trained on Fashion MNIST (≈0:00–0:46). Lecture 3 ended before any model was trained.

The Colab notebooks are not published on OCW and are not in this knowledge base. Everything said
below about the code follows the lecturer's narration, and only the functions and arguments he names
are given. The fast.ai spreadsheet and the handwritten Notability explanation of pooling that he
mentions are not in it either.

## Part 1 — Convolutional networks from scratch

### Why flattening an image falls short

In the walkthrough, a network with one hidden layer reached an accuracy "in the high 80s" on
Fashion MNIST. Slide 2 says "85% +", and later in the lecture he puts it at 88% (≈48:22, ≈53:05).
That network did not know its input was an image: it took the $28 \times 28$ table, concatenated the
rows into one long vector and sent it in. That is the clue for how to do better (slide 2;
≈0:46–1:32).

Flattening an image and feeding it to a dense layer causes three problems (slides 3–9;
≈1:32–6:14).

The first is **too many parameters**. A colour photo from the lecturer's older iPhone is $3{,}024 \times 3{,}024$ pixels, roughly 9 million, and with three channels that is about 27 million numbers between
0 and 255. Connect it to a single dense layer of 100 neurons and that one connection needs about
27 million × 100 ≈ **2.7 billion** weights, ignoring the biases (slides 4–5; ≈3:08–3:55). "Do you
think we can actually get 2.7 billion images to train any of these things so that we don't
overfit?" The approach is computationally demanding, very data-hungry, and increases the risk of
overfitting (slide 6; ≈3:55–4:41).

The second is that we **lose local spatial adjacency**, the relationships between neighbouring
pixels that define the features of an image (slide 7). A student named it first: flattening loses
"the proximity of one pixel to other ones that would be around it". If a pixel sits in the middle
of a t-shirt, knowing that the pixels around it belong to the same t-shirt would help
(≈1:32–2:17).

The third is that we **don't learn once and reuse**. If an image has a vertical line at the top
left and another at the bottom right, the flattened approach learns to detect each separately,
"which doesn't make any sense. A vertical line is a vertical line." The network should "learn it
once and use it again and again" (slide 8; ≈4:41–5:28). He calls this **translation invariance**:
"Translation is math speak for move stuff around" (≈5:28).

Asked whether the problem comes from compressing the image, he says no: it would happen with any
picture, compressed or not (≈5:28–6:14). Convolutional layers were developed to address exactly
these shortcomings (slide 9; ≈6:14).

### Convolutional filters

A **convolutional filter** is "nothing but a small matrix of numbers", a small square matrix, and a
**convolutional layer** is composed of one or more filters (slides 11–12; ≈7:02). If the numbers
are chosen carefully and the filter is "applied" to an image, it can detect features: lines,
curves, gradations in colour, circles (slide 13; ≈7:47). Slide 14 claims that the filter with rows
1 1 1 / 0 0 0 / −1 −1 −1 detects horizontal lines, and that the filter whose every row is
1 0 −1 detects vertical lines. The lecture then demonstrates the claim.

![Slide 14 — The Convolutional Filter](../raw/images/04-cnns-and-transfer-learning/slide-14.png)

*Slide 14 — the two filters used throughout the lecture: rows 1 1 1 / 0 0 0 / −1 −1 −1 for horizontal lines, and 1 0 −1 in every row for vertical lines. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)*

### The convolution operation

Applying a filter to an image is called the **convolution operation** (slides 15–21;
≈8:33–11:43). The worked example uses a $7 \times 7$ grayscale image whose rows are constant across the
columns, reading 0, 5, 10, 15, 10, 5 and 0 from top to bottom, and the horizontal-line filter.

**Overlay.** Place the filter on the top-left $3 \times 3$ patch of the image, so that each of the
nine filter numbers sits on top of an image number (slide 16; ≈9:21).

**Multiply matching elements and add up** (slide 17). For the first patch, whose rows are 0 0 0,
5 5 5 and 10 10 10:

$$0 \cdot 1 + 0 \cdot 1 + 0 \cdot 1 + 5 \cdot 0 + 5 \cdot 0 + 5 \cdot 0 + 10 \cdot (-1) + 10 \cdot (-1) + 10 \cdot (-1) = -30$$

**Run the result through a ReLU**: $\max(0, -30) = 0$. This is the top-left cell of the output
(slide 18; ≈10:09). Slide 18's footnote adds that the ReLU is not strictly part of the convolution
operation but is included for convenience. Here "all that effort came to nothing because it's 0.
That's OK."

![Slide 18 — the first output cell](../raw/images/04-cnns-and-transfer-learning/slide-18.png)

*Slide 18 — the filter laid over the top-left 3 × 3 patch (red box) gives −30, and the ReLU turns it into the output's first cell, 0. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)*

**Slide** the window one step to the right and repeat, to get the second number of the output
(slide 19; ≈10:09–10:55).

![Slide 19 — sliding the window](../raw/images/04-cnns-and-transfer-learning/slide-19.png)

*Slide 19 — the window moves one column to the right and the same multiply, add and ReLU gives the second output cell. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)*

**Continue row by row.** At the end of the first row, move to the start of the second row and
carry on, until the window reaches the bottom-right corner (slides 20–21; ≈10:55).

The finished output is $5 \times 5$, because a $3 \times 3$ window fits in five positions across a 7-wide image
and five down a 7-tall one. Its first three rows are all 0 and its last two all 30 (slide 21):

| | | | | |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 | 0 |
| 30 | 30 | 30 | 30 | 30 |
| 30 | 30 | 30 | 30 | 30 |

Because the filter's top row is all 1s, its middle row 0s and its bottom row −1s, each output cell
is the sum of the patch's top row minus the sum of its bottom row. It is positive only where the
patch is brighter at the top than at the bottom, as in the lower half of this image where the rows
fall from 15 to 10 to 5 to 0.

![Slide 21 — The Convolution Operation](../raw/images/04-cnns-and-transfer-learning/slide-21.jpg)

*Slide 21 — the finished 5 × 5 output: the filter responds (30) only in the lower half of the image, where brightness falls from top to bottom. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)*

A student asked what happens when the filter does not fit the remaining columns exactly. You stop
when the filter's right edge meets the image's right edge. There are two nuances he names and then
sets aside: you can **pad** the image's borders so that the filter can go outside the image, and
you can move two steps at a time instead of one, which is called the **stride**. "This basic
default approach works well, amazingly well, almost all the time" (≈10:55–12:29).

### The spreadsheet demonstration

To show that the filters live up to their claims, the lecturer switches to an Excel spreadsheet
"courtesy of the fast.ai people" (slide 22; ≈12:29–17:13). It holds a table of numbers between 0
and 1, the pixel values divided by 255, and conditional formatting colours the high values red, so
that a handwritten **3** pops out. Applying the horizontal-line filter with a cell formula that
multiplies, adds and then takes $\max(0, \cdot)$, the ReLU, gives an output in which only the
horizontal strokes of the 3 light up. The vertical-line filter lights up only the vertical stroke
(≈13:16–14:53).

Looking at why a cell lights up shows that the filter is finding **edges**: places where one row
of the patch is bright and the row below it is dark. The 1s multiply the bright values and the −1s
the dark ones, so "the small values are going to get very small. The big values are going to get
very big, and the overall thing is going to be emphasized" (≈14:53–16:25). He encourages working
through the spreadsheet after class. Slide 22 also points, as optional practice, to
<https://setosa.io/ev/image-kernels/>, where you can type in your own filter numbers and see what
they detect (≈16:25). "By choosing the numbers in a filter carefully and applying this operation,
different features can be detected" (≈16:25–17:13).

### Convolutional layers

A convolutional layer is one or more filters, and each filter can be thought of as a
**specialist** for one feature: a horizontal line, an arc, a vertical line (slides 23–24;
≈17:13). Real images have lots going on, so you probably want lots of filters. But you do not
decide up front what each filter specializes in: "You will let the system figure out what it wants
to figure out. So there is no human bottleneck in doing this", which there was before deep learning
(≈17:13–17:59).

![Slide 24 — Convolutional Layers](../raw/images/04-cnns-and-transfer-learning/slide-24.png)

*Slide 24 — a convolutional layer is a stack of filters, each a specialist for one feature. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)*

With two filters, they simply run in parallel on the same input: each does the convolution
operation and produces its own output (slides 25–27; ≈17:59). On the $7 \times 7$ image, the
horizontal-line filter gives the $5 \times 5$ output above, and the vertical-line filter gives a $5 \times 5$
output of all zeros, since the image has no vertical edges. These two $5 \times 5$ matrices form a
**tensor** of shape $5 \times 5 \times 2$, or equally $2 \times 5 \times 5$. Which order you use "ends up being a matter of
convention" (slides 28–29; ≈18:44). "So now you begin to see why we care about tensors." With 103
filters the result would be $5 \times 5 \times 103$ (≈18:44). See [tensors](tensors.md).

![Slide 29 — Applying a Convolutional Layer to an image](../raw/images/04-cnns-and-transfer-learning/slide-29.png)

*Slide 29 — a layer with two filters turns the 7 × 7 image into two 5 × 5 outputs, a tensor of shape 5 × 5 × 2. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)*

### Colour images

A colour image is a rank-3 tensor: for example $6 \times 6 \times 3$, three tables of numbers for red, green
and blue (slide 30; ≈19:33). Applying a $3 \times 3$ filter to each channel separately is the wrong
answer. A student first suggested that the lines would sit in the same place in each channel. The
lecturer's point is that, done independently, "the network has not been informed that these things
are all part of the same underlying concept": the three numbers at a pixel location belong
together (≈20:21–21:07).

So the filter is made rank-3 as well, with the **same depth as the input**. A $3 \times 3$ filter becomes
a $3 \times 3 \times 3$ cube. There are now 27 numbers in the image patch and 27 in the filter, but you still
match them up, multiply, add and apply the ReLU, and the other aspects of the operation are
unchanged (slide 31; ≈21:07–21:53). If the input is 10 deep, the filter is 10 deep (≈22:38). On a $6 \times 6 \times 3$ input, one $3 \times 3 \times 3$ filter gives a $4 \times 4$ output. The lecturer tried to get ChatGPT to draw
this and "it just completely bombed", so the picture is from deeplearning.ai (≈21:53).

![Slide 31 — Applying a Convolutional Filter to a color image](../raw/images/04-cnns-and-transfer-learning/slide-31.jpg)

*Slide 31 — on a 6 × 6 × 3 colour image the filter becomes 3 × 3 × 3, matching the input's depth, and still produces a single 4 × 4 table. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)*

However deep the input and filter, what comes out of one filter is always a 2D table, because each
position gives a single number. With several filters you get several tables, packaged into a
tensor: two $3 \times 3 \times 3$ filters give $4 \times 4 \times 2$, ten give $4 \times 4 \times 10$, and $f$ filters give $4 \times 4 \times f$ (slide 32; ≈24:11–24:57).

![Slide 32 — Applying a Convolutional Layer to a color image](../raw/images/04-cnns-and-transfer-learning/slide-32.jpg)

*Slide 32 — two 3 × 3 × 3 filters each give a 4 × 4 table, stacked into a 4 × 4 × 2 tensor; f filters give 4 × 4 × f. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)*

Asked whether one might first combine the channels into an intermediate layer instead of making
the filter deeper, he says the research literature has many such variations, but "this basic
approach is extremely powerful" and good enough for practical use (≈23:26). Chapter 8.1 of the
textbook has more detail and intuition (slide 33). "This convolutional stuff, it grows in the
telling": revisit it a few times and it becomes "part of your muscle memory" (≈24:57–25:45).

### The big idea: learn the filters

Where do the numbers in a filter come from (slide 34; ≈25:45)? Traditionally they were **designed
by hand**. Computer-vision researchers invested "prodigious amounts of time, and effort, and
talent" to find the right filters for each application, for example the features to extract from
an MRI image to predict a stroke (slide 35; ≈25:45–26:32).

As the field learned to train deep networks with many parameters, with ReLU, stochastic gradient
descent, GPUs and backprop, a big idea emerged: **think of the numbers in the filter as weights and
learn them from the data with backpropagation, just like all the other weights** (slide 36;
≈26:32). "Blindingly obvious in hindsight. I'm sure it was not obvious in foresight. This was the
breakthrough." It works because a convolutional filter is just a neuron, "albeit a special one",
whose arithmetic is even simpler than a regular neuron's (slide 37; ≈27:18). So the whole machinery
of neurons, layers, loss functions and gradient descent applies, and "we don't have to invent any
new stuff to make it work" (slide 38; ≈27:18–28:09).

The appendix slide he points to explains why (slide 83; not presented). A traditional neuron in the
first hidden layer is connected to **all** the pixels of the input image, while a convolutional
filter neuron is connected only to the pixels in a small **region**. Sliding the filter across the
image can be thought of as "a different filter for each window *but with the same weights*". Its
benefits answer the three problems above: it preserves local adjacency, needs far fewer
parameters, and gives translation invariance, detecting the same pattern wherever it appears. See
[anatomy of a neural network](neural-network-anatomy.md).

This turned out to be a turning point for computer vision (slide 39). The massive unlock came in
2012, when a system called **AlexNet** that used this approach "crushed the competition" in the
**ImageNet** competition. As the lecturer recalls it, the previous best error rate was 26% and
AlexNet's was 16%, a 10-point improvement in a field used to gains of 0.5% or 1% a year
(≈28:09–28:56). These figures are spoken only; the deck does not print them.

### Later layers see more, and learn more complex features

With every succeeding convolutional layer, each filter implicitly sees more of the input image
(slide 40; ≈28:56–29:43). A cell in the first layer might see only the top of a house's chimney.
But the next layer's input is the first layer's output, so one of its cells gathers information
from a whole square of first-layer cells, each of which covers part of the original picture. "With
every additional layer, you're seeing more, and more, and more of the image", and the network
builds a **hierarchical** understanding of it, which is "a very key part of the unlock" (≈29:43).

![Slide 40 — Later conv layers "see" more of the original input than the earlier layers](../raw/images/04-cnns-and-transfer-learning/slide-40.jpg)

*Slide 40 — a point in convolutional layer 2 depends on a patch of layer 1, which in turn covers a larger region of the input image. The figure is Fig 14-2 of the Hands-On Machine Learning book the slide credits. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)*

So a network with many convolutional layers learns increasingly complex features: "lines =>
edges, circles => faces!" (slide 41, from Lee et al., 2009; its figure is excluded from OCW's
licence and not reproduced here). In a face-detection network visualized layer by layer, the first
layer learns lines and edges, the second puts lines together into edges and parts ("this looks like
3/4 of somebody's ears"), and the next assembles whole faces. "Can you imagine the researchers who
did this work? … I mean, the goosebumps" (≈29:43–30:29). See
[representation learning](representation-learning.md).

### Pooling layers

**Pooling layers**, also called down-sampling or subsampling layers, reduce the size of the tensor
coming out of a convolutional layer (slide 43; ≈30:29–31:15). Making it smaller "will force the
network to try to summarize and learn what's going on". A pooling layer has **no weights to learn**.
It is a simple arithmetic operation (≈31:15).

In **max pooling**, superimpose a $2 \times 2$ window on the top-left of the table and keep the biggest of
its four numbers, then move the window to the next $2 \times 2$ box, and so on (slide 44; ≈31:15–32:03).
In **average pooling**, keep the average of each box instead (slide 45; ≈32:03). On slide 44's $4 \times 4$ example:

| | | | |
| --- | --- | --- | --- |
| 23 | 14 | 0 | 109 |
| 9 | 43 | 58 | 15 |
| 105 | 85 | 35 | 25 |
| 24 | 69 | 17 | 7 |

the four $2 \times 2$ boxes give a max-pooled output of 43, 109 / 105, 35 and an average-pooled output of
22.2, 45.5 / 70.75, 21. The first average is exactly 22.25, which the slide prints as 22.2.

![Slide 44 — 2x2 max pooling](../raw/images/04-cnns-and-transfer-learning/slide-44.png)

*Slide 44 — max pooling keeps the largest number in each colour-coded 2 × 2 box: 43, 109, 105 and 35. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)*

![Slide 45 — 2x2 average pooling](../raw/images/04-cnns-and-transfer-learning/slide-45.png)

*Slide 45 — average pooling on the same grid keeps each box's average: 22.2 (exactly 22.25), 45.5, 70.75 and 21. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)*

$2 \times 2$ pooling reduces the number of entries by 75%, and the output is fed to the next layer as
usual (slide 46; ≈32:48). The pattern is: a convolutional layer, then a pooling layer to shrink its
output, then the next convolutional layer.

Pooling works on **every slice of the tensor separately**. A tensor of $224 \times 224 \times 64$ is 64 tables
of $224 \times 224$. Pooling shrinks each table to $112 \times 112$, so the result is $112 \times 112 \times 64$: "each table
shrinks due to pooling, but the number of tables does not change" (slide 47; ≈36:41–37:28). The
slide's figure comes from Stanford's CS231n course notes, which he recommends.

![Slide 47 — Pooling Layers](../raw/images/04-cnns-and-transfer-learning/slide-47.jpg)

*Slide 47 — pooling acts on each depth slice separately: 224 × 224 × 64 becomes 112 × 112 × 64, and on one slice 2 × 2 max pooling turns a 4 × 4 grid into 6, 8 / 3, 4. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)*

**The intuition.** Max pooling acts like an **"OR" condition**: if a feature exists anywhere in its
input, max pooling picks it up, so it acts like a **feature detector** (slide 48;
≈33:37–34:22). A high number coming out of a convolutional layer means some feature lit up. Max
pooling asks of each region only whether anything lit up there. Over a whole picture, you can step
back and say that nothing lit up at the top left, something did at the top right, and so on, so
"you're operating at a higher level of abstraction" (≈34:22). The lecturer's handwritten
explanation of this was meant for his iPad, which would not connect, so he promised a recorded
walkthrough instead (slide 49; ≈32:48, ≈34:22).

A student asked whether pooling loses the spatial information. Not in the way that matters: the
network already knows that, say, an eye-like circle lit up in the top left, and it is no longer
looking at pixels. This proceeds slowly and incrementally, which is why networks are big
(≈34:22–35:08). Because successive convolutional layers see more and more of the original image,
the max-pooling layers that follow them detect whether a feature exists in more and more of the
original image too (slide 50; ≈35:08). By the seventh, eighth and ninth layers the network has, in
effect, "tagged all the features of that image at various resolutions" (≈35:56).

Asked whether to pre-process the input, for example turning a colour video into black-and-white
frames, or to add more layers: if there is important domain knowledge that has to be true, encode it
in the input so the network does not waste capacity learning it. If you are not sure, let the
network learn whatever helps it predict (≈35:56–36:41).

### The architecture of a basic CNN

A basic CNN runs the input through a series of **convolutional blocks**, each typically one or two
convolutional layers followed by a pooling layer (slides 52–53; ≈38:19). The slides' diagram of
such a network, from towardsdatascience.com, is excluded from OCW's licence and not reproduced
here. Further into the network, the blocks get **smaller in height and width, because of pooling,
but deeper** (slide 54; ≈38:19–39:04). "We have empirically figured out that actually that model of
reducing the size, the height and the width, but then making it deeper, tends to work really well
in practice."

He gives a numerical example on the board, which the recording does not show: a colour picture
comes in as $224 \times 224 \times 3$, and when ResNet, a famous network used later in the lecture, has finished
its convolution and pooling, its final tensor is **$7 \times 7 \times 2{,}048$**: much smaller in height and
width, much deeper (≈39:04–40:37).

Why deeper? The number of atomic features worth detecting, such as lines, curves and gradations in
colour, is probably not large. But the ways they combine to depict real-world things are
combinatorial: "I have 10 kinds of atoms. How many molecules can I make from it?" So later layers
get more filters, each able to pick up some combination of what comes in (≈40:37–41:23). The depth
is **your design choice**. When you add a convolutional layer you choose how many filters it has,
just as you choose the number of neurons in a hidden layer, and you design it so the later layers
are deeper (≈43:42–44:30).

The network is not told what to learn, only to minimize the loss. Once a network is accurate you
can introspect and see what it learned, sometimes lines, then shapes, then faces, and sometimes
nothing understandable (≈41:23–42:09). He points to the visualization paper shown later (slides
67–69) as one of the first to do this (≈42:09).

At the end, the final tensor is **flattened** into a long vector and sent through zero or more
dense ("fully connected") hidden layers to the output layer, such as a softmax that classifies the
image as a car, truck, van or bicycle (slide 55; ≈42:55–43:42). "So now we are back in familiar
territory." See [convolutional neural networks](convolutional-neural-networks.md).

### Colab: a CNN for Fashion MNIST

The first notebook revisits the Fashion-MNIST Colab from the walkthrough (slide 56;
≈44:30–55:25). After the usual imports and random seed, it loads the data and the plotting helpers
as before, and divides the pixel values by 255 to normalize them to the range 0 to 1. The training
set is 60,000 images of $28 \times 28$ (≈44:30–45:15).

A convolutional layer expects a channel dimension. A colour image has three channels and a
grayscale image only one, so instead of $28 \times 28$ the layer is told to expect **$28 \times 28 \times 1$**: "It's
the same thing conceptually, but that's the format that it expects." The lecturer expands the
array's dimension with a call the captions render as `expand_dimension`, and the training set
becomes $60{,}000 \times 28 \times 28 \times 1$ (≈45:15–46:01).

The model, as he narrates it (≈46:01–48:22):

- The input is `keras.Input` with shape $28 \times 28 \times 1$.
- The first **convolutional block** is a `layers.Conv2D` layer with 32 filters,
  `kernel_size` 2 by 2 ("the simplest size"), ReLU activation and a name, followed by a
  `MaxPool2D` layer. There is also a `Conv1D` and a `Conv3D`, but for images `Conv2D` is all you
  need.
- The second block is a copy of the first. He keeps 32 filters because it is a simple problem,
  though "you can totally go to 64" to make it deeper.
- Then the tensor is flattened, passed through a dense layer of 256 neurons, and ends in a softmax
  with 10 outputs, one per clothing category. Keras is then told to "take this input and the
  output. String them up together. Define a model for me."

"The new concepts we are seeing here are Conv2D for the convolutional layer, and then MaxPool2D
for the max pooling layer. That's it." A student asked why every layer's output is called `x`. It
is a notational convenience. Each layer is fed `x` and its output is called `x` again, and Keras
tracks everything under the hood. Numbering them x1, x2, x3 would mean renumbering everything when
a layer is inserted (≈49:55–50:41). See [Keras](keras.md).

`model.summary` reports about **302,000 parameters**, and he invites the class to hand-calculate
it (≈50:41). The following check is this page's arithmetic, not the lecture's. It assumes, as in
the slides' worked example, no padding and a step of one, and that each $2 \times 2$ pooling halves the
height and width, rounding down. Neither assumption is stated for the Colab. Each filter is a
neuron with one weight per number it covers plus a bias.

- The first `Conv2D` has 32 filters of $2 \times 2 \times 1$, so $32 \times (4 + 1) = 160$ parameters. The
  $28 \times 28$ input gives $27 \times 27 \times 32$, which pooling shrinks to $13 \times 13 \times 32$.
- The second `Conv2D` has 32 filters of $2 \times 2 \times 32$, so $32 \times (128 + 1) = 4{,}128$. It gives
  $12 \times 12 \times 32$, which pooling shrinks to $6 \times 6 \times 32$, flattened to 1,152 numbers.
- The dense layer of 256 has $1{,}152 \times 256 + 256 = 295{,}168$, and the softmax of 10 has
  $256 \times 10 + 10 = 2{,}570$.

The total is 302,026, in line with the summary's 302,000.

It is compiled as before with Adam and trained for 10 epochs with a 20% validation split.
Convolutional networks have more going on, so training is a bit slower (≈50:41). The training loss
falls nicely, the validation loss flattens out around the eighth epoch, and the accuracy is in the
90s. On the test set: **90.5%** (≈52:15).

If you are not impressed by going from 88 to 90, think in terms of the error that is left: about
12% remained, and the CNN knocked 2 points off it. He believes the state of the art on this data
set is 97%, and invites the class to try different filters and reach the mid 90s (≈52:15–53:05).
Ten epochs is plenty, since it is already beginning to overfit (≈53:52). A student asked how epochs
relate to batches. An epoch is one pass through the data, and within it the number of batches is
the number of training examples divided by the batch size, rounded up (≈53:52; see
[gradient descent](gradient-descent.md)).

Three questions along the way shaped how to use this:

- **CNN or flatten?** The flattened network did work, but with images "it's typically a good idea to
  just start with the CNN straight off the bat … You're not giving up anything." On how many layers:
  "start simple. And if it works, stop working on it. If it doesn't, add more layers" (≈48:22–49:08).
- **Is the architecture just trial and error?** Typically yes, but rather than designing from
  scratch, "it's much better to just download a pre-trained model and just adapt it for your
  particular problem. That is actually the norm." He builds from scratch so that it "should not be a
  black box to you" (≈49:08–49:55).
- **What about tasks other than classification, such as semantic segmentation?** The input side
  stays a stack of convolutional layers, but if every pixel must be classified, the output must have
  the same dimensions as the input, which needs a different architecture, such as one called U-Net, not covered here (lecture 11 explains
  it: [Part 7](11-diffusion-models.md#part-7--the-u-net)). He planned an optional walkthrough on using
  the Hugging Face Hub for such tasks (≈51:30–52:15), which lecture 11 gives at its end
  ([Part 16](11-diffusion-models.md#part-16--a-lightning-tour-of-hugging-face-computer-vision-pipelines)). See [images and computer-vision tasks](images-and-vision-tasks.md).

Asked whether neural networks always beat other machine learning, he says they are very good for
unstructured data like images. For structured data, like the heart-disease problem, methods such as
gradient boosting (XGBoost) sometimes work really well, so he would try both rather than assume the
deep network wins (≈54:38). The Colab also has an optional section on how to improve a model once
it is built (≈54:38).

## Part 2 — Colour images and transfer learning

### The handbags-and-shoes problem

To "kick it up a notch", the lecturer web-scraped roughly 100 colour images each of handbags and
shoes: can about 200 examples train a good classifier, when Fashion MNIST's 60,000 images already
overfit within a few epochs (slides 57–58; ≈55:25–56:11)? Slide 58 promises to classify "your shoe
or handbag in class with high accuracy". Its sample photographs are excluded from OCW's licence.

The second notebook first builds a CNN from scratch (slide 59; ≈56:11). It downloads the images
from a Dropbox folder, unzips them, and splits them into training, validation and test folders,
each with a `handbags` and a `shoes` subfolder (≈56:11–57:46). When working with images you can
create one folder per class and "just point Keras at it. It'll automatically figure out those are
the labels" (≈57:00–57:46). A Keras utility, not named in the narration, reads the JPEGs, converts
them to tensors, resizes them all to the same size, since web-scraped images come in different
shapes and sizes, and batches them, here with a batch size of 32. It finds 98 training images in
two classes, 49 validation and 38 test, "so less than 100 examples in the training set"
(≈57:46). Each image is resized to $224 \times 224 \times 3$ because ResNet, used later, expects that size
(≈58:32).

Instead of dividing by 255 by hand, the model now uses a `Rescaling` layer, "graduating to the
higher levels of Keras". Then come two convolutional blocks of 32 filters, as for Fashion MNIST,
max pooling and a flatten. The output is a single sigmoid, since this is a binary classification
problem (≈58:32–59:17). It has about 101,000 parameters and is compiled with binary cross-entropy,
Adam and accuracy, then trained for 20 epochs (≈59:17–1:00:05).

The results: training accuracy approaches 100%, but validation accuracy hovers in the 80s, with
some overfitting from around the tenth epoch. On the test set the model scores **87%**. That is
"actually pretty good, given that we only have 100 examples", for a model built entirely from
scratch (≈1:02:25–1:03:11). Can we do better (slide 60)?

### Data augmentation

The Colab has a section on **data augmentation**, which is not in the deck (≈1:03:11–1:06:15).
Rotate an image of a handbag by 10 degrees and it is still a handbag. Zooming in or out slightly
or cropping it slightly does not change its meaning either. So you can perturb each image slightly
and add the result to the training data as a new example: "It's an unbelievable free lunch,
frankly." The same kind of technique works for text, covered later in the course. When you do not
have much data, it artificially bolsters the amount you have (≈1:03:11–1:03:58).

Keras predefines data-augmentation layers. The example randomly flips a picture horizontally,
randomly rotates it by 0.1, and randomly zooms in and out a little. The lecturer forgets whether
the 0.1 is in degrees or radians and points to the documentation. Each perturbation is applied only
at random, so some pictures get perturbed in some ways, which gives the variety (≈1:04:44). Running
it repeatedly on one handbag shows it slightly rotated one way, then the other, with a little zoom.

Augmentation must not change the meaning of a picture. With satellite data, be careful with flips.
With everyday images, horizontal flips are fine but vertical ones are not: "How many times will you
have an upside down dog picture that you need to classify?" The augmentation layers go right after
the input, and the rest of the model stays unchanged (≈1:05:30). The augmented model is in the
Colab but was not trained in class. Augmentation "figures prominently in homework 1"
(≈1:06:15). See [data augmentation](data-augmentation.md).

### Transfer learning

Transfer learning takes advantage of two research trends (slides 61–63; ≈1:00:05–1:01:37). First,
researchers have designed architectures matched to different **types** of data. Slide 62's table
gives residual connections for all types, convolutional layers for images, and transformers for
sequences such as natural language, audio, video and gene sequences. Transformers are the subject
from the next week. Second, they have used those architectures to train high-performance networks
on vast real-world data sets, and "thankfully, they have made them publicly available": "Numerous
pretrained models are available!" (slide 63).

**Transfer learning** is customizing such a pre-trained network to your problem rather than
designing and training a network from scratch (slide 64; ≈1:01:37). Handbags and shoes are
everyday objects, so look for a network trained on images of everyday objects, as opposed to
specialized images such as MRIs or X-rays (slide 65). "Of course, the first thing you should
probably do is to see if anybody has built the specific thing you want … on GitHub". **ImageNet** is
a database of millions of images of everyday objects in **1,000 categories**, such as furniture,
animals and automobiles, so look for networks trained on ImageNet (slide 66; ≈1:01:37–1:02:25).
Slide 66's graphic gives 1.2 million training and 100,000 test images. It is excluded from OCW's
licence.

### What an ImageNet-trained network knows

A network that does well on ImageNet has developed "a smart, hierarchical representation of these
objects across its layers" (slides 67–69; ≈1:06:15–1:07:02). The slides show visualizations from
Zeiler and Fergus's paper, which are excluded from OCW's licence. Layer 1 picks up gradations of
colour and line-like patterns, layer 2 edges, and layer 3 interesting honeycomb shapes. Going
further, he points to one feature that is "already picking up the shape of a human torso" and
another that picks up "what looks like a Labrador retriever".
This paper was one of the first to visualize what such networks learn, and he recommends it to
anyone curious how the pictures are made (≈1:07:02).

### ResNet, and cutting off its head

**ResNet** is a family of networks trained on ImageNet that did very well in its associated
competition (slide 70; ≈1:07:49). Its learned weights and biases should therefore embody knowledge
of lines, shapes, curves and the other characteristics of the millions of everyday images it was
trained on. The slides draw the 34-layer version, ResNet 34, from the paper they cite, arXiv 1512.03385
(slides 70–73; the figure is excluded from OCW's licence).

But ResNet cannot be used as is: it was designed to classify an image into 1,000 categories, and
this problem has two (slide 71; ≈1:08:37). So "we do just a little bit of surgery". Take ResNet and
**stop just before the final layer**, the fully connected layer with 1,000 outputs (slide 72). What
comes out of this **"headless" ResNet** should be a smart representation of the image, as if the
image had been tagged with lots of general tags that can be used any way you wish (slide 73;
≈1:08:37–1:09:22).

Run all the images through headless ResNet. The tensor that comes out should carry a rich
understanding of each image: its objects, its features. Treat it as the input to a small network of
your own, with one hidden layer and a sigmoid output for handbag or shoe (slides 74–75;
≈1:09:22). Since the inputs to the hidden layer are no longer raw pixels but this higher level of
abstraction, it may learn to tell handbags from shoes "with hardly any examples".

![Slide 75 — a small network on top of headless ResNet](../raw/images/04-cnns-and-transfer-learning/slide-75.png)

*Slide 75 — the output of headless ResNet becomes the input layer of a small network with one ReLU hidden layer and a sigmoid output, handbag or shoe. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)*

**Fine-tuning.** You can get fancier and connect headless ResNet directly to the small network,
training the whole thing **end to end**. This is called **fine-tuning**, and you must start the
training from the weights and biases that came with ResNet, "because that is the crown jewel that's
been learned", not from scratch. The class does this in homework 1 (slide 77;
≈1:09:22–1:10:09). See [transfer learning](transfer-learning.md).

A student asked whether publicly known weights make a model prone to adversarial attacks. There is
"some adversarial risk", and he offered to discuss it offline (≈1:10:09–1:10:54).

### Where to find pre-trained models

Pre-trained models are available all over the internet: on the TensorFlow Hub, the PyTorch Hub and
the Hugging Face Hub (slides 78–80; ≈1:10:09). When the lecturer checked the Hugging Face Hub on
February 13, the day before the lecture, it listed over half a million models: 505,594, according
to slide 80's screenshot. "I think last year it was 50,000 when I taught the course."

![Slide 78 — the TensorFlow Hub](../raw/images/04-cnns-and-transfer-learning/slide-78.jpg)

*Slide 78 — the TensorFlow Hub, "a repository of trained machine learning models", with a short example that loads one as a Keras layer. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)*

![Slide 79 — the PyTorch Hub](../raw/images/04-cnns-and-transfer-learning/slide-79.jpg)

*Slide 79 — the PyTorch Hub, whose models can be filtered by audio, generative, NLP, scriptable and vision. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)*

![Slide 80 — the Hugging Face Hub](../raw/images/04-cnns-and-transfer-learning/slide-80.jpg)

*Slide 80 — the Hugging Face models page on February 13, 2024, listing 505,594 models. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec04.pdf)*

### Colab: transfer learning with ResNet

Back in the notebook (slide 81; ≈1:10:54), ResNet comes packaged and ready to download. The call
passes `include_top=False`, which tells Keras not to include the very final layer: "Just give me
everything up to but not including that". "Top" because people think of the network as running
from bottom to top, where the lecturer draws it left to right. The downloaded model has **23
million parameters**. The narration does not name which ResNet variant the notebook downloads
(≈1:10:54–1:11:39).

All the data is run through headless ResNet **once**, and what comes out of the penultimate layer is
stored. Each of the 98 training images is now a **$7 \times 7 \times 2{,}048$** tensor (≈1:11:39–1:12:26).

The new model takes that tensor as its input, flattens it immediately, and passes it through a
dense layer of 256 ReLU neurons, a `Dropout` layer and a sigmoid output (≈1:12:26). Dropout is
used here without explanation, deferred again to "early next week": "Don't worry about this detail
for the moment." [Lecture 5](05-text-vectorization-and-bag-of-words.md) explains it (≈1:11:38–1:14:44). It is trained for 10 epochs and finishes almost at once, because ResNet's outputs
were computed once up front and are not recomputed every epoch (≈1:13:14). The training curves
show "something bad" in the tenth epoch, so perhaps it should have stopped at nine. On the test set
it scores **100%** accuracy: "It's unbelievable" (≈1:13:14–1:14:01).

### The live demo

For "the moment of truth", a code snippet captures an image from the laptop's webcam and the model
predicts its class (≈1:14:01). A student offered a handbag, which looked rather like a backpack ("This
feels like an adversarial example"). So the lecturer tried a shoe first, "a pretty big shoe", and
the model said: "It's a shoe!" Then the bag: "It's a handbag!" "I swear, every time I do the demo,
I age a few years" (≈1:14:01–1:15:32).

## Printed slips

- Slide 8 prints "rather learn it separately" (missing "than").
- Slide 41 prints "Lee at al" for "Lee et al".
- Slide 45 prints the first average-pooling result as 22.2. The average of 23, 14, 9 and 43 is
  22.25.
- Slide 2 puts the single-hidden-layer network at "85% +". The lecturer says "the high 80s" and
  later 88% (≈0:46, ≈48:22, ≈53:05).

## Related pages

- [Convolutional neural networks](convolutional-neural-networks.md): filters, the convolution
  operation, layers, colour images, pooling, the basic architecture and its Keras layers.
- [Transfer learning](transfer-learning.md): pre-trained networks, ImageNet, headless ResNet,
  fine-tuning and the model hubs.
- [Data augmentation](data-augmentation.md): perturbing images to stretch a small data set.
- [Images and computer-vision tasks](images-and-vision-tasks.md): pixels, channels and the vision
  tasks; loading colour images from folders.
- [Representation learning](representation-learning.md): hierarchical features, and a pre-trained
  network's output as a learned representation.
- [Tensors](tensors.md): filter outputs stacked into tensors, and channels.
- [Keras](keras.md): `Conv2D`, `MaxPool2D`, `Rescaling`, augmentation layers and `include_top`.
- [Overfitting and regularization](overfitting-and-regularization.md): augmentation as a remedy for
  little data, and where dropout stands.
- [Anatomy of a neural network](neural-network-anatomy.md): why a filter is a neuron.
- [Lecture 3](03-keras-and-vision-basics.md): images as pixel tables, Fashion MNIST and the softmax,
  which this lecture builds on.
