# Representation learning

A **representation** is the form in which data is handed to a model. 15.773 treats it as the
central idea of deep learning: "the whole name of the game is the representations"
([lecture 1](01-introduction.md) ≈9:18).

## Why raw unstructured data needs a representation

A colour image is stored as three tables of pixel intensities (red, green and blue, each 0–255), but
those numbers have **no intrinsic meaning**. A blue value of 251 could be sky, water or blue paint
([lecture 1](01-introduction.md) ≈6:59–7:45; slide 23). A model given raw pixels has nothing
meaningful to work with. Structured data does not have this problem: its columns already mean
something, such as age or cholesterol (slide 21).

## The old way: hand-made features

Before deep learning, researchers built representations by hand. This is **feature engineering**
or feature extraction. For bird photos you might measure beak length, wingspan and primary colour,
which turns images into a table (≈8:31; slide 24). With a good representation, even linear or
logistic regression performs well (slide 25). But designing one was expert work: a computer-vision
PhD could take four years to build the representation for one narrow problem (≈9:18–10:03). This
**human bottleneck** limited where machine learning could be used (slide 26).

## The deep learning way: learn it

Deep learning **learns representations automatically** from raw input. The raw data goes in, a
stack of learned transformations builds the representation, and a linear or logistic regression at
the end makes the prediction ([lecture 1](01-introduction.md) ≈10:03–10:51; slides 27–29).

Mechanically, the representation is what the hidden layers compute. Starting from logistic
regression, you insert linear functions between the input and the output, pass each through a
nonlinear activation, and repeat. Each transformation gives the network more capacity to learn a
useful representation of the input before the final prediction (≈30:51–36:15; slides 66–78). In the
lecture's words, a neural network is "repeatedly transformed inputs which are finally fed to a
linear or logistic regression model" (≈36:15). Whether the learned representation is actually good
depends on training.

![Slide 77: "Before" — inputs feeding logistic regression directly; "After" — the inputs pass through two stages of three neurons each before the same logistic-regression output.](../raw/images/01-introduction/slide-77.jpg)

*Lecture 1, slide 77 — the hidden layers are where the learned representation lives. The input and output are unchanged.*

## Where this comes up

- [Lecture 1](01-introduction.md): why representations matter, and how hidden layers learn them.
