# Training a neural network

**Training** a network means finding values for all its weights and biases that make its
predictions as close as possible to the actual values in the data. That is exactly what fitting a
linear or logistic regression does. A deep network is just "a very complex function with lots of
parameters" ([lecture 2](02-training-deep-nns.md) ≈19:25–20:58; slides 30–33). Regression hides the
optimization under the hood. For neural networks the course opens it up, because it matters.

## The pieces

Training needs three things beyond the network's architecture, which must be fixed first because
without it nothing can compute an output from an input (lecture 2 ≈21:43):

- **Data**: inputs $x^i$ with actual outputs $y^i$, for data points $i = 1, \ldots, n$.
- **A loss function** that measures the discrepancy between the actual values and the model's
  predictions $model(x^i)$. It is small when the predictions are close and exactly zero for a perfect
  model. The loss must suit the kind of output: mean squared error for a number, binary
  cross-entropy for a probability. See [loss functions](loss-functions.md).
- **An optimizer** that changes the weights to lower the loss. The course's optimizers are all
  variants of gradient descent, "SGD and its siblings", with **Adam** as the default. See
  [gradient descent](gradient-descent.md).

The weights are the only variables. The data $x^i$ and $y^i$ are fixed, so once the network's
formula is substituted into the loss, the loss is "just a 'good old' function" of the weights and
gradient descent applies to it directly (lecture 2 ≈55:48–56:35; slides 79–81).

## The loop

Lecture 2 sums up the whole process in one picture (≈1:12:46–1:14:17; slide 94). Run a minibatch of
inputs through the layers to get predictions. Compare them with the true targets in the loss
function to get a loss score. The optimizer computes the gradient of the loss with respect to every
weight, using [backpropagation](backpropagation.md), and updates each weight by

$$w \leftarrow w - \alpha \nabla g(w)$$

where $g$ stands for the loss, $\nabla g(w)$ is its gradient and $\alpha$ is the learning rate. Then
repeat with the next minibatch. "This is how our little network is going to get built for heart
disease prediction, this is how GPT was built. And this is how AlphaFold was built. And AlphaGo."

![Slide 94: block diagram of the training loop — input X through two layers with weights to predictions Y', which with true targets Y feed the loss function and loss score; the optimizer turns the loss score into a weight update for each layer.](../raw/images/02-training-deep-nns/slide-94.jpg)

*Lecture 2, slide 94 — the training loop, from page 61 of the course textbook.*

## Starting the weights: initialization

The weights start out random, but not arbitrarily random (lecture 2 ≈1:01:56–1:02:44).
**Initialization schemes** pick starting values so that gradients stay well-behaved. A bad start in a
big network makes the gradients of the earlier layers blow up or shrink to nothing: the **exploding
gradient** and **vanishing gradient** problems. Keras builds good schemes in as defaults. The
lecturer believed Keras uses "HE" or "Xavier Glorot" initialization, and advised accepting the
default. Every weight is initialized at once before training starts, and the scheme takes into
account the sizes of the layers on either side (≈1:12:46).

## When to stop, and what the minimum means

Training stops when you run out of iterations, time or compute, or when the gradient is close to
zero. That point is rarely the global minimum, and it need not be. A decent solution is usually good
enough, and the best possible fit to the training data risks **overfitting**, because new data has
its own loss whose minimum lies elsewhere (lecture 2 ≈50:22–51:54, ≈1:03:29–1:04:15). See
[gradient descent](gradient-descent.md#local-minima-and-saddle-points).

## What training produces

The output of training is the weights, "the crown jewel", the result of a lot of money, time and
smartness. When Meta made "the weights of Llama 2" available, that is what it published (lecture 2
≈54:15–55:03).

## Where this comes up

- [Lecture 1](01-introduction.md): assumes a trained network (slide 100) and defers training.
- [Lecture 2](02-training-deep-nns.md): the whole training story, from loss functions to SGD and the
  training loop.
