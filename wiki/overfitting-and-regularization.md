# Overfitting and regularization

A model **overfits** when it fits the idiosyncrasies of its training data instead of the patterns that
generalize. It **underfits** when it is too simple to capture the richness of the data.
**Regularization** is the family of methods for handling the risk of overfitting, so that the model
fits "just right" ([lecture 3](03-keras-and-vision-basics.md) ≈10:45–13:04; deck 3A, slides 11–16).
The course treats this as background from earlier machine-learning courses and recalls it quickly.

## Training error versus validation error

As a model grows more complex (in a regression, add interaction terms, logarithms "and this and
that"), its error on the training data keeps going down (lecture 3 ≈10:45–11:32; deck 3A, slides
12–13). Its error on data held out from training goes down at first, bottoms out, and then climbs
again (deck 3A, slide 14). The climbing part is overfitting, the part before the minimum is
underfitting, and the minimum is the "sweet spot" to aim for (deck 3A, slide 15; ≈11:32–12:18).

![Deck 3A, slide 15: error against model complexity. The training-set curve falls steadily towards zero; the validation-set curve falls to a minimum marked "Sweet spot" and then rises, with the falling part labelled underfitting and the rising part overfitting.](../raw/images/03-keras-and-vision-basics/3a-15.png)

*Lecture 3, deck 3A, slide 15 — underfitting, the sweet spot and overfitting, read off the validation curve.*

## Why neural networks need it

To learn smart representations of complex, unstructured data, a network needs large "capacity": many
layers and many neurons in each (lecture 3 ≈12:18–13:04; deck 3A, slide 16). "GPT-3, for example,
has 96 layers, if I recall." But more layers mean more parameters, more parameters a more complex
model, and so a greater chance of overfitting. Lecture 2 makes the same point from the other side: the
global minimum of the training loss is not the goal, because new data has its own loss whose minimum
lies elsewhere ([lecture 2](02-training-deep-nns.md) ≈50:22–51:54). Many regularization methods have
been developed. The course uses two.

## The validation set

Both methods rely on a **validation set**, part of the training data set aside before training starts
(lecture 3 ≈13:04–13:49, ≈48:49–50:22). It is the same set in every epoch. The model is evaluated on it
at the end of each epoch, but it is never used to calculate gradients, only loss and accuracy: "That's
what keeps you honest" (≈15:20–16:06). It is also where, later, different hyperparameter values are
compared (≈47:17).

The validation set is not the **test set**. The test set is split off at the very beginning and used
once, with the final model, for a realistic estimate of how it will do once deployed: "we open the
safe, take out the test set, and use it just once with your final, final model" (≈47:17–48:03). See
[preparing data for a network](data-preprocessing.md).

## Early stopping

"Stop the training early before the training loss is minimized by monitoring the loss on a validation
dataset" (deck 3A, slide 17; lecture 3 ≈13:04–14:34). Keep training and keep track of the validation
loss. When it flattens out and starts to climb, stop. Geoff Hinton has called early stopping a
"beautiful free lunch".

![Deck 3A, slide 17: error against iterations. The training-set curve keeps falling; the validation-set curve falls to a minimum and rises again, and a vertical dashed line labelled "early stopping" marks where it turns up.](../raw/images/03-keras-and-vision-basics/3a-17.png)

*Lecture 3, deck 3A, slide 17 — stop where the validation error turns up, not where the training error bottoms out.*

In practice (lecture 3 ≈51:09–55:04), plot the training and validation loss against the epoch. For
the heart-disease model trained for 300 epochs, the validation loss flattened after about 50 epochs and
then "maybe gently starts to rise". A purist would retrain for about 50 epochs. Because the validation
*accuracy* still improved at the end, the lecturer chose to live with the slight overfitting. He trusts
accuracy over binary cross-entropy here, as the measure closer to real-world deployment.

If the validation loss is still falling at the end of a run, there is more room: Keras's `fit` resumes
where it left off, so run 10 more epochs at a time until the validation loss starts to climb, then back
off. Keras **callbacks**, which the course comes to later, automate this: stop when the validation loss
stops improving, or save the best model as training goes (≈53:30–55:04).

## Dropout

**Dropout** randomly zeroes out the outputs of some of the nodes in a hidden layer, typically 50%, and
is implemented as a "dropout layer" in Keras (deck 3A, slide 18). Lecture 3 only names it, calls it "a
very, very clever strategy", and defers it to lecture 4, the first time it is used (≈14:34). Dropout is
also a source of randomness in training, which matters for reproducibility (≈31:42).

## Where this comes up

- [Lecture 2](02-training-deep-nns.md): why the training loss's global minimum is not the goal.
- [Lecture 3](03-keras-and-vision-basics.md): underfitting and overfitting, the validation set, early
  stopping in the heart-disease Colab, and dropout previewed.
- See also [training a neural network](training-neural-networks.md).
