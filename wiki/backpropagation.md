# Backpropagation

**Backpropagation**, "backprop" to anyone "hip and cool" in deep learning, is an efficient way to
compute the gradient of the loss function with respect to every weight in a network
([lecture 2](02-training-deep-nns.md) ≈56:35; slide 82). [Gradient descent](gradient-descent.md)
needs that gradient at every step. With a billion weights and millions of data points, computing it
naively would be prohibitive.

## The idea

Backprop exploits the network's layer-by-layer structure (≈57:22–58:10; slides 82–83). The
computation is organized as a **computational graph**. It starts at the very end, with the gradient
of the loss with respect to the output, then steps back one layer at a time, each time computing the
gradient with respect to the previous layer's output. It is "a way to organize the **chain rule**" of
calculus to fit the layered architecture (≈1:11:14).

Going backwards has two advantages:

- **Nothing is computed twice.** Going forwards, many paths would recompute the same intermediate
  quantities "again and again and again". Going backwards, each is computed once and reused
  (≈1:10:26–1:11:14).
- **The work becomes a sequence of matrix multiplications**, when layers have more than one neuron.

It runs once per gradient calculation, not in a loop (≈1:10:26).

## Why GPUs matter

Graphics processing units were invented to accelerate video games, whose core operation is matrix
multiplication. Applying them to deep learning's gradient calculations was the turning point:
"backprop + GPUs" means fast loss-function gradients, and "if this thing were not true, this class
would not exist" (lecture 2 ≈58:10–59:40; slides 84–86). GPUs were first used for deep learning in
the mid-2000s, by the lecturer's recollection, and burst onto the world stage when AlexNet won a
famous computer vision competition in 2012 (≈1:02:44–1:03:29).

## A worked example

The lecturer's handwritten example (the backprop example deck, pages 1–13) uses the smallest
possible network: two inputs $x_1$, $x_2$, weights $w_1$, $w_2$, a bias $b$, no hidden layer, and a
squared-error loss for one data point with actual value $y$:

$$\text{Loss} = (b + w_1 x_1 + w_2 x_2 - y)^2$$

Introduce intermediate values $a_1 = w_1 x_1$, $a_2 = w_2 x_2$ and $\hat{y} = b + a_1 + a_2$, so that
$\text{Loss} = (\hat{y} - y)^2$. Each step has an easy **local derivative**:
$\frac{\partial \text{Loss}}{\partial \hat{y}} = 2(\hat{y} - y)$; $\frac{\partial \hat{y}}{\partial b}$,
$\frac{\partial \hat{y}}{\partial a_1}$ and $\frac{\partial \hat{y}}{\partial a_2}$ are all 1;
$\frac{\partial a_1}{\partial w_1} = x_1$ and $\frac{\partial a_2}{\partial w_2} = x_2$ (pages 4–6).

Draw the equations as a graph, with the local derivatives on the edges. To get the derivative of the
loss with respect to any parameter, "start from the loss and travel backwards to the parameter,
multiplying the partial derivatives as you go" (pages 8–11). For $w_1$:

$$\frac{\partial \text{Loss}}{\partial w_1} = \frac{\partial \text{Loss}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial a_1} \cdot \frac{\partial a_1}{\partial w_1} = 2(\hat{y} - y) \cdot 1 \cdot x_1$$

which matches differentiating the loss directly (pages 3, 7 and 12–13).

![Page 12 of the backprop example: the computational graph with every local derivative on a backward magenta arrow and the path from Loss back through y-hat and a1 to w1 highlighted in yellow.](../raw/images/02-training-deep-nns/backprop-12.jpg)

*Backprop example, page 12 — multiply the derivatives along the yellow path from Loss back to w1.*

The efficiency shows even here. $\frac{\partial \text{Loss}}{\partial \hat{y}}$ appears on the path to
every parameter, and backprop computes it once (page 13).

## Gradients flowing through activations

During backprop the gradient passes backwards through every activation function, which should
neither "squish" it nor make it explode. For positive inputs the ReLU's gradient is exactly 1, so the
gradient passes through unharmed. That is one reason ReLU is the default for hidden layers
([lecture 2](02-training-deep-nns.md) ≈1:15:03–1:16:34). Badly initialized weights can make
gradients explode or vanish in the early layers, which is why initialization schemes matter
(≈1:01:56–1:02:44). See [activation functions](activation-functions.md) and
[training a neural network](training-neural-networks.md).

## Where this comes up

- [Lecture 2](02-training-deep-nns.md): what backprop is and why it is fast; the worked example in
  [`raw/slides/02-backprop-example.md`](../raw/slides/02-backprop-example.md). The slides cite
  sections 2.4.3 and 2.4.4 of the textbook for more detail (slide 85).
