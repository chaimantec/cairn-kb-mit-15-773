# Multi-class classification and softmax

In **multi-class classification** the output is one of several categories, such as the 10 kinds of
clothing in Fashion MNIST ([lecture 3](03-keras-and-vision-basics.md) ≈1:07:27; deck 3B, slide 17).
The network has to say which category is right, so it outputs one probability per category, and since
exactly one category is right the probabilities must add up to 1. The **softmax** layer does that. The
lecturer flags it as "actually how GPT-4 works" (≈1:08:15).

## Why not ten sigmoids?

Ten outputs are no problem in themselves (deck 3B, slide 17; ≈1:08:15–1:09:00). Ten linear units give
10 numbers, and ten sigmoid units give 10 probabilities. But ten independent sigmoids need not add up to
1, as a student points out, and only one category can be true. "If not for that, we can just use 10
sigmoids."

![Deck 3B, slide 17: three panels. A column of linear units under "We know how to output 10 numbers", a column of sigmoid units under "We know how to output 10 probabilities", and the question "How do we output 10 probabilities that sum to 1.0?"](../raw/images/03-keras-and-vision-basics/3b-17.png)

*Lecture 3, deck 3B, slide 17 — linear units and sigmoid units each fall short of probabilities that sum to 1.*

## The softmax function

Let the final layer be 10 linear units with outputs $a_1, a_2, \ldots, a_{10}$. Softmax runs each
through the exponential function and divides by the total (deck 3B, slide 18; ≈1:09:00–1:09:47). The
$i$-th output is

$$\frac{e^{a_i}}{e^{a_1} + e^{a_2} + \ldots + e^{a_{10}}}$$

Exponentiating makes every number positive, whatever its sign. Dividing each positive number by the
total of all of them makes the results add up to 1, "there's no other option". So softmax "takes in $n$
arbitrary numbers and converts them to $n$ probabilities". The slide writes the first, second and last
outputs out in full, in the course's notation.

![Deck 3B, slide 18: linear units output a1, a2 to a10 into a box labelled softmax, which outputs e to the a1 over the sum of e to the a1 through e to the a10, and likewise for a2 and a10.](../raw/images/03-keras-and-vision-basics/3b-18.jpg)

*Lecture 3, deck 3B, slide 18 — softmax exponentiates each output and divides by the total.*

Why not train the network to output exactly 1 for the right class and 0 for the rest? "You can't
force the network to give you 1's or 0's. It's going to produce what it's going to produce." The best
you can do is to tame its output into the range you want (≈1:10:33–1:11:19).

## Why softmax, and not the maximum?

In [lecture 5](05-text-vectorization-and-bag-of-words.md) a student asks why a classifier cannot just output one number
per class and take the largest (≈43:35–45:08). The maximum of a bunch of numbers "is not very friendly
for differentiation". The network's output goes through a loss such as cross-entropy and then through
backpropagation, which "is just differentiation" and needs everything in it to have well-behaved
gradients. The max function does not, "which is why we have a soft version of it, softmax". It is
the same reason the course minimizes cross-entropy rather than maximizing accuracy directly.

## Softmax and language models

When GPT-4 emits each word ("it's actually a token, but we'll get to that later"), it does so through a
softmax with one output for every possible word, which the lecturer describes as a 52,000-way softmax.
It picks the most probable word and emits it. "This notion of a softmax is actually very powerful. But
we'll come back to that later" (≈1:09:47–1:10:33).

## Softmax inside a transformer, and a softmax per word

[Lecture 7](07-transformers.md) uses the softmax twice more. Inside self-attention it turns dot
products, which can be negative, into weights that are non-negative and add up to 1 — "When was the
last time you had to take a bunch of numbers which could be anything, and then somehow make sure that
they are going to be positive, non-negative, and they add up to 1?" (lecture 7, slides 41–43;
≈31:47–32:32; see [transformers](transformers.md)). And for slot filling, where every word of a query
needs its own label, a separate 123-way softmax sits on each of the transformer's outputs, one
classification per word (lecture 7, slides 13 and 75; ≈9:17–10:05).

[Lecture 9](09-large-language-models.md) puts a softmax on every position of a language model, each over the whole vocabulary
of about 50,000 words, and generates text from the last one (slides 10 and 31–32). It also adds a
parameter, **temperature** $T$, that divides every input of the softmax before exponentiating. As $T$
approaches 0 the largest input's probability goes to 1, which makes sampling greedy, and as $T$ grows the
probabilities flatten towards equal (≈51:21–56:01). See
[decoding and sampling](decoding-and-sampling.md#temperature).

## Encoding the labels, and choosing the loss

A multi-class label can arrive in two forms (deck 3B, slides 20–22; ≈1:11:19). **Sparse encoding**
stores the class as one integer: for Fashion MNIST, T-shirt/top 0, trouser 1, pullover 2, dress 3,
coat 4, sandal 5, shirt 6, sneaker 7, bag 8, ankle boot 9. **One-hot encoding** stores ten 0/1 entries
with a single 1 in the class's position. A binary label is simply 1 or 0.

The Keras cross-entropy loss must match the encoding (deck 3B, slides 23–26; ≈1:12:04–1:12:51):
`binary_crossentropy` for 1/0 labels, `sparse_categorical_crossentropy` for integer labels, and
`categorical_crossentropy` for one-hot labels. "These are all equivalent things"; the choice depends
only on how the data happens to be encoded. Fashion MNIST comes sparse encoded. Lecture 5's song genres are one-hot encoded with
pandas' `get_dummies`, so its three-way softmax is trained with plain `categorical_crossentropy`
([lecture 5](05-text-vectorization-and-bag-of-words.md) ≈54:25, ≈1:02:10). Categorical
cross-entropy generalizes binary cross-entropy with "roughly the same" intuition, and the lecture does
not give its formula. See [loss functions](loss-functions.md).

## Output layers at a glance

| Output variable | Output layer | Loss function |
| --- | --- | --- |
| Single number (regression) | one linear unit | Mean squared error |
| Single probability (binary classification) | one sigmoid unit | Binary cross-entropy |
| Vector of $n$ numbers (regression with multiple outputs) | stack of linear units | Mean squared error |
| Vector of $n$ probabilities that add up to 1 (multi-class) | softmax | Categorical cross-entropy |

This is deck 3B's slide 27 (≈1:10:33, ≈1:12:51). For a softmax, use categorical or sparse categorical
cross-entropy to match the labels.

## Where this comes up

- [Lecture 1](01-introduction.md) and [lecture 2](02-training-deep-nns.md): softmax promised as the
  output for several probabilities that must sum to 1.
- [Lecture 3](03-keras-and-vision-basics.md): softmax for Fashion MNIST, the GPT-4 connection, and the
  three cross-entropy losses.
- [Lecture 5](05-text-vectorization-and-bag-of-words.md): a three-way softmax for hip-hop, rock and pop, and why
  softmax rather than the maximum.
- [Lecture 7](07-transformers.md): the softmax that makes attention weights, and a softmax per word for
  slot filling.
- [Lecture 9](09-large-language-models.md): a softmax per position over the vocabulary, and temperature.
- See also [activation functions](activation-functions.md) and [decoding and sampling](decoding-and-sampling.md).
