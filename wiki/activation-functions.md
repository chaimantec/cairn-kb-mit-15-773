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
  (≈43:17–44:03). He defers its properties to the next lecture.

![Slide 93: three node drawings — a plain + node, a + node followed by max(0, x), and a + node followed by the sigmoid — mapped by a big arrow to three circle icons: a diagonal line (Linear Activation), a hinge (ReLU Activation) and an S-curve (Sigmoid Activation).](../raw/images/01-introduction/slide-93.png)

*Lecture 1, slide 93 — the visual shorthand used in later diagrams: each neuron is drawn as a circle with an icon of its activation.*

## Others

Tanh, leaky ReLU, GELU and swish also exist. Researchers keep proposing variants, "a menagerie of
activation functions" ([lecture 1](01-introduction.md) ≈44:48).

## How to choose

- **Hidden layers**: use ReLU. Empirically it does "an amazing job", and "if you just stick to the
  ReLU for your hidden layers, you can basically get anything done" (≈44:48–45:37).
- **Output layer**: there is no free choice, because the kind of output dictates it. A single
  probability needs a sigmoid. Several probabilities that must sum to 1 need a **softmax**, which
  the lecture 2 deck introduces (≈45:37).

## Where this comes up

- [Lecture 1](01-introduction.md): definition, sigmoid, linear, ReLU, and the rule of thumb.
