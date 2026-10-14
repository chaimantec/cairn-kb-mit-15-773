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

In practice the data are fed in **batches**, and one pass through all of them is an **epoch**. Each
batch gets its own weight update, so an epoch has as many updates as batches
([lecture 3](03-keras-and-vision-basics.md) ≈3:03–9:15; deck 3A, slides 4–10). See
[gradient descent](gradient-descent.md#epochs-and-batches).

## The checklist

Lecture 3 condenses creating and training a network into a checklist (deck 3A, slide 19;
≈14:34–15:20):

1. Get the data ready. See [preparing data for a network](data-preprocessing.md).
2. Design, or "lay out", the network: the number of hidden layers, the number of neurons in each, and
   the right output layer for the type of output.
3. Pick a loss function that suits the type of output, and an optimizer from the many flavours of SGD,
   with a "good" learning rate.
4. Decide on a regularization strategy.
5. Set it up in Keras/TensorFlow and start training.

The defaults: ReLU for hidden layers, **Adam** as the optimizer "because it's really good", and early
stopping or dropout as the regularization (≈14:34–15:20). For the heart-disease model the checklist
reads 1 hidden layer of 16 ReLU neurons, a sigmoid output, binary cross-entropy, Adam and early
stopping (deck 3A, slides 38–40). The output layer and the loss come as a pair, which deck 3B's slide 27
tabulates; see [multi-class classification and softmax](multiclass-classification.md#output-layers-at-a-glance).

## Training, validation and test sets

Training a real model uses three sets of data (lecture 3 ≈36:16–37:52, ≈47:17–51:09):

- The **test set** is split off first and locked away, and used once with the final model to estimate
  how it will do when deployed.
- The **validation set** is taken from the training data before training starts. The model is evaluated
  on it at the end of every epoch, but it never contributes to a gradient. It is what reveals
  overfitting and decides when to stop. See
  [overfitting and regularization](overfitting-and-regularization.md).
- The rest is the **training set**, which is divided into batches.

Training involves randomness: the initial weights, the shuffle into batches, the splits, and dropout.
Setting a random seed makes a run reproducible, mostly; bitwise reproducibility is not guaranteed
(≈30:57–32:27).

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
[gradient descent](gradient-descent.md#local-minima-and-saddle-points). Lecture 3 turns this into a
procedure, **early stopping**: watch the loss on a validation set and stop when it stops improving (deck
3A, slide 17). A run of 300 epochs on the heart-disease model shows it in practice; see
[overfitting and regularization](overfitting-and-regularization.md#early-stopping).

How many neurons, then? "It tends to be trial and error", which is how the lecturer chose the eight
hidden ReLUs of lecture 5's song classifier. To be more systematic, try many values with the Python
package KerasTuner. Choosing the number of neurons, the activation, the learning rate and the rest
this way is called **hyperparameter optimization** ([lecture 5](05-text-vectorization-and-bag-of-words.md) ≈45:08,
≈1:15:30–1:16:18).

## What training produces

The output of training is the weights, "the crown jewel", the result of a lot of money, time and
smartness. When Meta made "the weights of Llama 2" available, that is what it published (lecture 2
≈54:15–55:03).

## Where this comes up

- [Lecture 1](01-introduction.md): assumes a trained network (slide 100) and defers training.
- [Lecture 2](02-training-deep-nns.md): the whole training story, from loss functions to SGD and the
  training loop.
- [Lecture 3](03-keras-and-vision-basics.md): epochs and batches, the checklist, the three data sets, and
  training the heart-disease model in Keras (83.6% test accuracy against a 72.6% baseline).
- [Lecture 5](05-text-vectorization-and-bag-of-words.md): choosing hyperparameters by trial and error or with KerasTuner.
