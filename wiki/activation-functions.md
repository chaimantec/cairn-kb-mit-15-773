# Activation functions

An **activation function** is the nonlinear function applied inside each neuron after its linear
part. It "receives a single number and outputs a single number": scalar in, scalar out
([lecture 1](01-introduction.md) ≈41:45; slide 89). The linear part may combine thousands of
inputs, but the activation only transforms the one number that results. It has to be nonlinear,
otherwise stacking layers achieves nothing (≈33:58).

## The three introduced in lecture 1

- **Sigmoid**: $\sigma(a) = \frac{1}{1+e^{-a}}$ (slide 90). It maps any number into $(0, 1)$. It is
  near 0 for very negative inputs, near 1 for very positive ones, and changes steeply around 0.
  It is the activation of logistic regression's output, and the one to use when the output is a
  single probability (≈41:45–42:32, 45:37).
- **Linear**: $f(a) = a$ (slide 91). It passes the number straight through. It is included "just for
  completeness" (≈42:32–43:17).
- **ReLU**, the rectified linear unit: $g(a) = \max(0, a)$ (slide 92). Positive inputs pass unchanged
  and negative inputs become 0. The lecturer calls it "the hero of deep learning" and notes that
  many long-time practitioners count it among the key reasons for deep learning's success
  (≈43:17–44:03). He promised its properties for the next class. Lecture 2 gives one of them, its
  gradient, below.

![Slide 93: three node drawings — a plain + node, a + node followed by max(0, x), and a + node followed by the sigmoid — mapped by a big arrow to three circle icons: a diagonal line (Linear Activation), a hinge (ReLU Activation) and an S-curve (Sigmoid Activation).](../raw/images/01-introduction/slide-93.png)

*Lecture 1, slide 93 — the visual shorthand used in later diagrams: each neuron is drawn as a circle with an icon of its activation.*

## Others

Tanh, leaky ReLU, GELU and swish also exist. Researchers keep proposing variants, "a menagerie of
activation functions" ([lecture 1](01-introduction.md) ≈44:48).

## How to choose

- **Hidden layers**: use ReLU. Empirically it does "an amazing job", and "if you just stick to the
  ReLU for your hidden layers, you can basically get anything done" (≈44:48–45:37).
- **Output layer**: there is no free choice, because the kind of output dictates it. A single
  probability needs a sigmoid, as in lecture 2's heart-disease model. Several probabilities that
  must sum to 1 need a **softmax**, one output unit per class (≈45:37). Lecture 1 promised softmax
  for the next class, lecture 2 deferred it again (lecture 2 ≈17:54), and lecture 3 covers it, below.

## Softmax, for several probabilities that sum to 1

The activations above take one number and return one number. The **softmax** works on a whole layer
instead: it "takes in $n$ arbitrary numbers and converts them to $n$ probabilities"
([lecture 3](03-keras-and-vision-basics.md), deck 3B, slide 18; ≈1:09:00–1:09:47). Given the outputs
$a_1, \ldots, a_{10}$ of ten linear units, the $i$-th probability is

$$\frac{e^{a_i}}{e^{a_1} + e^{a_2} + \ldots + e^{a_{10}}}$$

Each exponential is positive, and dividing by their total makes the ten probabilities add up to 1. Ten
separate sigmoids would also give ten probabilities, but they need not add up to 1 (deck 3B, slide 17;
≈1:08:15–1:09:00). Lecture 3's summary: a single number takes a linear output, a single probability a
sigmoid, several numbers a stack of linear units, and several probabilities that must add up to 1 a
softmax (deck 3B, slide 19; ≈1:10:33). See
[multi-class classification and softmax](multiclass-classification.md).

## ReLU and the flow of gradients

Training runs gradients backwards through every activation function (see
[backpropagation](backpropagation.md)), and an activation must neither "squish" the gradient nor
make it explode ([lecture 2](02-training-deep-nns.md) ≈1:15:03–1:15:48). The ReLU passes this test
well. For positive inputs its gradient is exactly 1, so whatever gradient arrives "just gets
multiplied by 1 and gets pushed out the other side". It "preserves the gradient while injecting
almost the minimum amount of non-linearity", which is one reason it is so popular (≈1:15:48–1:16:34).
Lecture 2 also cites "a mountain of empirical evidence" for ReLU as the hidden-layer default
(≈7:50).

## In Keras

The activation is an argument of each layer, as in `keras.layers.Dense(16, activation="relu")`.
The name is apt. Taking the largest output directly would not work in training, because the maximum
"is not very friendly for differentiation" and backpropagation needs well-behaved gradients; the
softmax is "a soft version of it" ([lecture 5](05-text-vectorization-and-bag-of-words.md) ≈43:35–45:08).

Custom activation functions can be defined. Asked what Keras uses when none is given, the lecturer
was not sure (lecture 2 ≈17:08–18:39). See [Keras](keras.md).

## Where this comes up

- [Lecture 1](01-introduction.md): definition, sigmoid, linear, ReLU, and the rule of thumb.
- [Lecture 2](02-training-deep-nns.md): ReLU hidden units and a sigmoid output for heart-disease
  prediction, and why the ReLU's gradient makes it a good default.
- [Lecture 3](03-keras-and-vision-basics.md): softmax for Fashion MNIST's ten classes, and 256 ReLU
  hidden units.
- [Lecture 5](05-text-vectorization-and-bag-of-words.md): eight ReLU hidden units and a three-way softmax for song genres,
  and softmax as a differentiable maximum.
