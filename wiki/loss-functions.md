# Loss functions

A **loss function** quantifies the discrepancy between a model's predictions and the actual values.
It is small when the predictions are close, exactly zero for a perfect model, and it is what
training minimizes ([lecture 2](02-training-deep-nns.md) ≈21:43–23:15; slides 34–37). The lecturer
prefers "discrepancy" to "error" on purpose. "Predicted minus actual" is too narrow, much research
creativity goes into defining new losses, and many deep learning breakthroughs came from a clever
one. "There are hundreds of loss functions."

Two rules frame the choice. The loss is defined **per data point** and then averaged over all the
data (≈33:12–33:59). And it must be **matched to the kind of output** the model produces (slide 39).
Linear regression's "sum of squared errors" is a loss function under another name (slide 38).

Throughout, $n$ is the number of data points, $y^i$ is the actual value of the $i$-th data point and
$model(x^i)$ is the model's prediction for its inputs $x^i$. The superscript $i$ is an index, not a
power.

## Mean squared error: numerical outputs

When the output is a number, such as next week's demand for a product, the difference between
prediction and actual is a reasonable discrepancy. Squaring it and averaging gives the **mean
squared error** (lecture 2 ≈23:15–24:47; slide 40):

$$\frac{1}{n}\sum_{i=1}^{n}\left(y^i - model(x^i)\right)^2$$

## Binary cross-entropy: a probability for a yes/no outcome

When the model outputs a probability, from a sigmoid, and the actual value is 0 or 1, the loss must
compare a fraction with a label (lecture 2 ≈24:47–31:40; slides 41–49):

- If $y = 1$, the loss should be near 0 when the predicted probability is near 1 and very large when
  it is near 0. $-\log(\text{predicted probability})$ has that shape.
- If $y = 0$, the reverse: $-\log(1 - \text{predicted probability})$.

![Slide 46: two loss curves side by side; for y = 1 the loss falls from high values to 0 as the predicted probability goes to 1, and for y = 0 it rises from 0 and shoots up as the predicted probability approaches 1.](../raw/images/02-training-deep-nns/slide-46.jpg)

*Lecture 2, slide 46 — the two cases: −log(p) for y = 1, −log(1 − p) for y = 0.*

The two cases combine into one expression with no if-then, since for $y^i = 1$ the second term
vanishes and for $y^i = 0$ the first does. Averaged over the data, that is the **binary
cross-entropy loss**:

$$\frac{1}{n}\sum_{i=1}^{n} -y^i \log\left(model(x^i)\right) - (1 - y^i)\log\left(1 - model(x^i)\right)$$

The tables on slides 43 and 45 show how hard it punishes confident mistakes: for a patient who has
the disease, a predicted probability of 1/2 costs 1.0 and 1/1000 costs 9.97. For one who does not, a
predicted probability of 0.999999 costs 19.93. (The tables use base-2 logarithms, which neither the
slide nor the lecture names.)

Why a logarithm? A linear penalty would also slope the right way, but a steep, curved loss punishes
bad mistakes harder and works better with gradient descent (≈27:04–27:49). The log is easy to work
with, "has good gradients", is well-behaved mathematically and is empirically very effective. It is
"not the only game in town" (≈34:45). If you see a big network doing binary classification,
"chances are they're using this thing" (≈32:25).

## Beyond the symmetric case

Binary cross-entropy treats both kinds of error alike. To penalize one kind more, for instance
overestimates more than underestimates, other losses exist. The lecturer points to the **pinball
loss** (lecture 2 ≈31:40–32:25). Lecture 3 confirms that more complex loss functions can account for
the asymmetry between false positives and false negatives ([lecture 3](03-keras-and-vision-basics.md)
≈58:08).

## Categorical cross-entropy: several classes

When the output is one of several classes, the network ends in a softmax that produces one probability
per class, and the loss is **categorical cross-entropy**, a generalization of binary cross-entropy with
"roughly the same" intuition (lecture 3 ≈1:12:51). The lecture does not give its formula. In Keras the
loss has to match how the labels are encoded (deck 3B, slides 20–26; ≈1:11:19–1:12:51):

| Labels | Keras loss |
| --- | --- |
| Binary, 1 or 0 | `binary_crossentropy` |
| Multi-class, integers (sparse encoded) | `sparse_categorical_crossentropy` |
| Multi-class, one-hot encoded | `categorical_crossentropy` |

"These are all equivalent things"; the choice depends only on the encoding the data arrives in. See
[multi-class classification and softmax](multiclass-classification.md).

## Matching the loss to the output layer

Deck 3B's slide 27 sums up which loss goes with which output layer (lecture 3 ≈1:12:51):

| Output variable | Output layer | Loss function |
| --- | --- | --- |
| Single number (regression) | one linear unit | Mean squared error |
| Single probability (binary classification) | one sigmoid unit | Binary cross-entropy |
| Vector of $n$ numbers (regression with multiple outputs) | stack of linear units | Mean squared error |
| Vector of $n$ probabilities that add up to 1 (multi-class) | softmax | Categorical cross-entropy |

## Loss versus the metric you care about

A loss is chosen partly because it is mathematically convenient. What you usually care about in
practice is closer to accuracy or error rate, so report and plot that too (lecture 3 ≈51:56). The two can
disagree. Accuracy is discrete: a predicted probability that moves from 0.49 to 0.51 barely changes the
binary cross-entropy but flips the prediction from wrong to right (≈55:04–55:50). So validation accuracy
can keep improving while the validation loss rises. The lecturer calls binary cross-entropy "a beautiful
proxy, but an imperfect proxy for the thing we actually care about in the real world, which is error
rate and accuracy" (≈56:36).

## Where this comes up

- [Lecture 2](02-training-deep-nns.md): what a loss is, MSE, and binary cross-entropy built step by
  step for the heart-disease model.
- [Lecture 3](03-keras-and-vision-basics.md): categorical cross-entropy and the Keras losses for each
  label encoding, the output-layer/loss table, and loss versus accuracy.
