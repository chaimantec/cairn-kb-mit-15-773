# Anatomy of a neural network

The parts of a feedforward neural network and the words 15.773 uses for them, built up from
logistic regression as in [lecture 1](01-introduction.md) (≈24:41–56:00).

## Starting point: logistic regression as a network

Logistic regression takes inputs $x_1, \dots, x_k$, forms a linear combination plus an intercept,
and passes it through the sigmoid $\frac{1}{1+e^{-a}}$ to get a probability (slides 56–57). Drawn
as a network, each input sits in a circle. It flows along an arrow and is multiplied by its
coefficient. A "+" node adds everything plus the intercept, and a sigmoid node outputs the
probability (slides 61–63). In the course's interview example, with GPA 3.8 and experience 1.2:

$$\frac{1}{1+e^{-(0.4+0.2\cdot 3.8+0.5\cdot 1.2)}} = \frac{1}{1+e^{-1.76}} = 0.85$$

([lecture 1](01-introduction.md) ≈28:31; slide 62).

## The vocabulary

| Term | Meaning in the course |
| --- | --- |
| **Weight** | A coefficient: the multiplier on a value flowing along an edge (slide 64). |
| **Bias** | An intercept: the constant added at a node (slide 64). |
| **Neuron** / **node** | A linear function followed by a nonlinear activation function. Many numbers in, one number out (slide 79; ≈48:46). |
| **Activation function** | The nonlinear function inside each neuron, scalar in and scalar out (slides 82, 89). See [activation functions](activation-functions.md). |
| **Layer** | A vertical stack of neurons (slide 81). |
| **Input "layer"** | The inputs themselves. In quotes because it does no computation (slide 83). |
| **Hidden layer** | Any layer between input and output (slide 83). |
| **Output layer** | The final layer that produces the prediction (slide 83). |
| **Dense** / **fully connected** | A layer whose every neuron feeds every neuron of the next layer (slide 84). |
| **Feedforward** | Data flows only from left to right, unlike recurrent networks (slide 108). |
| **Architecture** | The arrangement of neurons into layers, the activation functions and the connections between layers (slide 108). |

The whole set of weights and biases is sometimes loosely called "the weights". Knowing the weights
and the architecture is enough to reconstruct a model (≈29:16–30:04).

So logistic regression is a neural network with **no hidden layers** (slide 85). A **deep** network
has many hidden layers (slides 86–87).

![Slide 83: the two-hidden-layer network with dashed boxes labelled Input "Layer", Hidden Layer, Hidden Layer and Output Layer.](../raw/images/01-introduction/slide-83.jpg)

*Lecture 1, slide 83 — input "layer", hidden layers and output layer.*

## Why the activation must be nonlinear

Stacking linear functions alone would be "a total waste of time", so each neuron applies a
**nonlinear** activation after its linear part ([lecture 1](01-introduction.md) ≈33:58). Lecture 1
asks students to take this on faith and defers the reason.

## What you design, and what is given

The input and output are fixed by the problem. You choose the **number of hidden layers**, the
**number of neurons** in each, and the **activation functions**. The output layer's activation must
match the kind of output: a sigmoid for one probability, a softmax for several probabilities that
sum to 1 ([lecture 1](01-introduction.md) ≈45:37–46:27; slides 94–96). The weights and biases are not
chosen. Training finds them, using data, a loss function and an optimizer (≈48:00), which
[lecture 2](02-training-deep-nns.md) covers. See [training a neural network](training-neural-networks.md).

Lecture 2 adds the meta-rule for these choices: start with the simplest network you can think of,
and only make it more complicated if it is not good enough (≈0:45–1:33). For heart-disease
prediction that meant one hidden layer. The lecturer tried 4, 8 and 16 neurons and kept 16, because
more started to overfit (≈5:30–6:15). Categorical inputs are one-hot-encoded, one input per level,
so that dataset's 13 variables become 29 input nodes (≈7:04; slide 7). Every layer is fully
connected to the next by default for structured data (≈9:23).

## Counting parameters

Count weights and biases separately. For a dense layer from $m$ neurons to $n$ neurons there are
$m \times n$ weights and $n$ biases, one per receiving neuron. Lecture 1's 2-3-1 network, with two
inputs, three ReLU hidden neurons and one sigmoid output, has $2 \times 3 + 3 \times 1 = 9$ weights
and $3 + 1 = 4$ biases, so **13 parameters** (≈50:19–51:59; slides 98–99). Forgetting the biases is
the common mistake. The lecturer recommends counting by hand for the first few complex networks you
build.

Lecture 2's heart-disease network, with 29 inputs, 16 hidden units and one output, has
$29 \times 16 + 16 + 16 \times 1 + 1 = 497$ parameters (≈9:23–10:55; slide 10). Because the
weights between two fully connected layers of $A$ and $B$ units number about $A \times B$,
parameters "explode" as layers widen, which has to be watched to prevent overfitting (≈10:55).

## The forward pass

Prediction runs the inputs through the network layer by layer. With lecture 1's trained 2-3-1
network (slide 100), the hidden neurons compute

$$a_1 = \max(0, -0.3 + 0.5x_1 + 0.1x_2), \quad a_2 = \max(0, 0.2 - 0.1x_1 + 0.3x_2), \quad a_3 = \max(0, 0.5 + 0.2x_1 - 0.1x_2)$$

and the output neuron computes

$$y = \frac{1}{1+e^{-(0.05-0.2a_1-0.3a_2-0.15a_3)}}$$

For $x_1 = 2.3$ and $x_2 = 10.2$ this gives $a_1 = 1.87$, $a_2 = 3.03$ and $a_3 = 0$ (ReLU clips
$-0.06$ to 0), then $y = 0.226$ (slides 101–104). Written as a single formula, even this small
network is far more complex than logistic regression. That complexity is where its power comes from
(slides 105–107).

## Where this comes up

- [Lecture 1](01-introduction.md): the whole construction, terminology, parameter counting and the
  worked forward pass.
- [Lecture 2](02-training-deep-nns.md): designing the heart-disease network, its 497 parameters,
  and writing it in [Keras](keras.md).
