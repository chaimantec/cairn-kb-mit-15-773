# Data augmentation

**Data augmentation** creates extra training examples by perturbing the ones you have in ways that
do not change their meaning. It is the course's way to "artificially bolster the amount of data you
have" when you do not have much ([lecture 4](04-cnns-and-transfer-learning.md) ≈1:03:11–1:03:58).
It appears in the handbags-and-shoes Colab, not in the slides.

## The idea

Rotate a picture of a handbag by 10 degrees and it is still a handbag. The same goes for zooming in
or out slightly, or cropping slightly. So you can take any image, perturb it slightly, and add the
result to the training data as a new example: "It's an unbelievable free lunch, frankly"
(≈1:03:11–1:03:58). The lecturer says the same kinds of techniques work for text, which the course
covers later.

## The rule: keep the meaning

Make sure a perturbation does not change what the picture means. With everyday images, horizontal
flips are fine but vertical flips are not: "How many times will you have an upside down dog picture
that you need to classify?" With satellite data, be careful with flips (≈1:05:30).

## In Keras

Keras predefines data-augmentation layers. The course's example, as narrated (≈1:04:44–1:06:15;
the notebook is not on OCW), randomly flips a picture horizontally, randomly rotates it by 0.1, and
randomly zooms in or out a little. The lecturer does not remember whether the 0.1 is in degrees or
radians and refers to the documentation. Each perturbation is applied only at random, so different
pictures get perturbed in different ways. The augmentation layers go **right after the input**, and
the rest of the model stays unchanged.

The augmented model is in the Colab but was not trained in class. Data augmentation "figures
prominently in homework 1" (≈1:06:15).

## How it relates to other remedies

Augmentation is one answer to overfitting on a small data set; see
[overfitting and regularization](overfitting-and-regularization.md). The other answer lecture 4
gives is to start from a network already trained on far more data; see
[transfer learning](transfer-learning.md).

Augmentation also drives **contrastive learning**, the image counterpart of self-supervised learning in
the last, unpresented section of [lecture 8](08-transformers-and-self-supervised-learning.md)'s deck (slides 94–98). Two augmented versions of one
image are passed through the same network, which is trained to make their representations agree. Slide
97 shows the augmentations: crops, flips, colour distortion, rotation, cutout, noise, blur and Sobel
filtering. See [self-supervised learning](self-supervised-learning.md#images-contrastive-learning).

## Where this comes up

- [Lecture 4](04-cnns-and-transfer-learning.md): the handbags-and-shoes Colab.
- [Lecture 8](08-transformers-and-self-supervised-learning.md): augmented pairs in contrastive learning (slides only).
