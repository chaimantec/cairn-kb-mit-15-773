# Gradient descent

**Gradient descent** minimizes a function by repeatedly nudging its variables in the direction that
lowers it, guided by the derivative. It is how neural networks are trained: the function is the
loss and the variables are the weights. "This is the basic intuition behind how GPT-4 was built"
([lecture 2](02-training-deep-nns.md) ≈38:38). Decades of heavier optimization methods are largely
unused; what is used is this algorithm "with some flavors on top". It was invented in 1847 by Cauchy
(≈40:59; slide 66).

## One variable

Setting the derivative to zero and solving fails for complicated functions, so use what the
derivative *tells* you instead (≈35:33–37:52; slides 52–58). The derivative of $g(w)$ at a point is
the change in $g(w)$ for a small increase in $w$:

| If the derivative at $w$ is … | What it means | To minimize, do this |
| --- | --- | --- |
| Positive | Increasing $w$ slightly increases $g(w)$ | Reduce $w$ slightly |
| Negative | Increasing $w$ slightly decreases $g(w)$ | Increase $w$ slightly |
| ~0 | Changing $w$ slightly won't change $g(w)$ | Stop |

The algorithm (slides 59–64): start at some $w$; compute the derivative; act on the table; unless you
have hit the maximum number of iterations or run out of time, repeat. The whole table collapses into
one update rule,

$$w \leftarrow w - \alpha \frac{dg(w)}{dw}$$

where $\alpha$ is the learning rate. A positive derivative lowers $w$, a negative one raises it, and
a near-zero derivative barely moves it.

Lecture 2's example is $g(w) = \frac{1}{50}\left(w^4 + w^2 + 10w\right)$. Starting at $w = 2.5$ with
$\alpha = 1$, gradient descent reaches the minimum in "four or five iterations", taking smaller steps
as the slope flattens (≈42:30–43:16; slides 68–69).

![Slide 69: plot of g(w) for w from −3 to 3 with gradient descent iterates, circles on the w axis and crosses on the curve, starting green at w = 2.5 and ending red at the minimum near w = −1.25.](../raw/images/02-training-deep-nns/slide-69.jpg)

*Lecture 2, slide 69 — gradient descent on g(w) from w = 2.5 with α = 1, green at the start and red at the minimum.*

## The learning rate

$\alpha$ keeps each step small, because the derivative describes the function only near the current
point: "if you take a big step, all bets are off" (≈41:44; slide 67). Typical values are small, such
as 0.1, 0.01 or 0.001, found by trial and error, and published papers' appendices are read closely
for them because "the learning rate is part of the IP" (≈42:30). The step size reflects confidence:
if the function looks like it keeps going down in this direction, a big step is safe; if not, a small
one avoids backtracking (≈52:41).

## Many variables: the gradient

A network's loss has one variable per parameter, 175 billion for GPT-3 (≈43:16). For a function of
several variables, take the **partial derivative** with respect to each, treating the others as
constants, and stack them into a vector, the **gradient**, written with nabla (≈44:06–45:41; slides
70–72). For $g(w_1, w_2) = w_1^2 + w_2^2 + 2$:

$$\nabla g = \left[\frac{\partial g}{\partial w_1}, \frac{\partial g}{\partial w_2}\right] = [2w_1, 2w_2]$$

Gradient descent updates every coordinate with its own partial derivative, which in vector form is
(slides 73–74)

$$w \leftarrow w - \alpha \nabla g(w)$$

and it is the same formula whether $w$ has 2 entries or 175 billion (≈47:14).

## Local minima and saddle points

Gradient descent stops where the gradient is near zero, which may be a **local minimum** rather than
the global one, or a **saddle point** where the function flattens and then keeps falling (≈48:00–50:22;
slide 76). In practice "it has not mattered" (≈50:22–51:54). For complex networks a decent solution is
good enough, and the very best fit to the training data risks overfitting. In a space of a billion
dimensions, a point where *every* direction slopes upward is very unlikely, so you probably end near a
saddle point, "and it turns out, it's good enough". And a global minimum of the *training* loss is not
the minimum of the loss on future data (≈1:03:29–1:04:15).

## Stochastic gradient descent

For a large dataset, computing the gradient over all $n$ data points at every step is expensive and
may not even fit in memory. **Stochastic gradient descent (SGD)** instead computes each step's loss
and gradient from a small random **minibatch**, typically 32 or 64 points, even out of billions
(lecture 2 ≈1:05:03–1:06:38; slides 89–91). Strictly, SGD uses one point and this is minibatch
gradient descent, but the field says SGD.

The minibatch gradient only approximates the true one, yet SGD "works extremely well in practice",
for reasons research has not fully settled. The approximation can even help it **escape local
minima** of the full loss (≈1:07:24–1:08:10, ≈1:12:00; slide 92). Each minibatch gets one update,
and the next minibatch starts from the updated weights (≈1:09:41, ≈1:14:17). SGD has many variants;
the course's default is **Adam** (≈1:08:10), "because it's really good"
([lecture 3](03-keras-and-vision-basics.md) ≈14:34–15:20). Lecture 2 promises epochs and batch sizes
for the Colab (≈1:09:41), and lecture 3 covers them, below.

The bowls and curves used to teach this are idealized. A visualization of a real network's loss
landscape is all hills, valleys and cracks (slide 93).

## Epochs and batches

An **epoch** is one pass through all the training data (lecture 3 ≈3:03; deck 3A, slides 4–9). In
gradient descent the whole training set goes through the network, the gradient is computed from all of
it, and the weights are updated **once**, at the end of the epoch, by

$$w \leftarrow w - \alpha \frac{dLoss(w)}{dw}$$

with $w$ a weight, $\alpha$ the learning rate and $\frac{dLoss(w)}{dw}$ the derivative of the loss
(deck 3A, slide 6). In SGD the training set is divided into minibatches, which the course calls simply
**batches**. For each batch in turn, its samples go through the whole network, the gradient is computed
on that batch, and the weights are updated, so the next batch already sees the new weights (deck 3A,
slides 7–9; ≈3:50–5:25). One epoch therefore contains as many updates as there are batches (≈9:15).

![Deck 3A, slide 9: a row of batches 1 to N across the training set, with an arrow from each batch down to its own copy of the weight-update formula, arranged as a staircase.](../raw/images/03-keras-and-vision-basics/3a-9.png)

*Lecture 3, deck 3A, slide 9 — in SGD every batch gets its own weight update.*

The number of batches in an epoch is the training-set size divided by the batch size, rounded up
(deck 3A, slide 10; ≈7:44–8:30). With 194 training samples and batches of 32 that is 7: six batches of
32 and a last one of 2. Batches need not be equal in size. The batch size is a choice. 32 is a good
default; 64 and 128 are worth trying, and researchers' advice is "the biggest batch size that doesn't
make your machine die". Sizes like 32 and 64 suit parallel hardware (≈7:44, ≈43:18–44:03). In Keras's
training output, each epoch reports the batches it ran, such as "7/7" (≈45:41).

How many epochs? Twenty to 30 is a good starting point, and the validation loss says whether to go on.
See [overfitting and regularization](overfitting-and-regularization.md) (≈44:03, ≈53:30–54:17).

## Where this comes up

- [Lecture 2](02-training-deep-nns.md): the derivative, the algorithm, learning rate, multi-variable
  gradients, local minima and SGD.
- [Lecture 3](03-keras-and-vision-basics.md): epochs, batches, and batch size in practice.
- See also [backpropagation](backpropagation.md), which computes the gradient, and
  [training a neural network](training-neural-networks.md), which puts the pieces together.
