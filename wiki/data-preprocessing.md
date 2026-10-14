# Preparing data for a network

A neural network takes numbers, ideally in a small range, and its evaluation has to be kept honest.
Lecture 3's heart-disease Colab works through the steps that get tabular data there
([lecture 3](03-keras-and-vision-basics.md) ≈33:13–40:11). The lecturer calls it "boring
preprocessing", and "machine learning 101", which is why he goes through it fast. The notebook is not
published on OCW. This page follows his narration of it.

## Look at the data and set a baseline

The heart-disease data, made available by the textbook's author François Chollet as a CSV file, loads
into a Pandas dataframe of 303 rows and 14 columns (≈33:13). For any binary classification problem,
first check whether the classes are balanced. Here 72.6% of the patients do not have heart disease and
27.4% do (≈33:13–33:58).

That gives a **baseline model**: predict the majority class for everyone. It is right 72.6% of the time,
"and so any fancy model we build, it's got to do better than this" (≈33:58–34:43). The trained network
later scores 83.6% on the test set (≈57:22). For Fashion MNIST's ten balanced classes the same baseline
gets 10% (≈1:04:16–1:05:07). In [lecture 5](05-text-vectorization-and-bag-of-words.md)'s song-genre data, rock is 55%
of the songs, so always guessing rock is right 55% of the time (≈53:38).

## Categorical inputs: one-hot encoding

A neural network, unlike a decision tree, cannot handle categorical inputs directly, so every
categorical variable has to be made numeric (≈34:43). The standard way is **one-hot encoding**: one 0/1
column per value. Pandas's `get_dummies` does the whole table at once. The variable `thal`, with three
values (fixed, normal, reversible), becomes three columns, `thal_fixed`, `thal_normal` and
`thal_reversible` (≈35:29). This is how the heart-disease data's 13 input columns become the
network's 29 inputs ([lecture 2](02-training-deep-nns.md), slides 3–10; lecture 3 ≈39:25).

## Numeric inputs: standardization

Networks work best when the numeric inputs all lie in a relatively small range (≈35:29). The standard
practice is to **standardize** each numeric variable: subtract its mean and divide by its standard
deviation (≈36:16). Afterwards the values are "all very smallish — 0, 1, minus 1, around that range",
which is "kind of ideal for neural network training" (≈37:52).

## Split first

Before standardizing, split the data into a **training set** and a **test set**, here 80% and 20%: 242
and 61 patients (≈36:16–37:52). Only then compute the means and standard deviations and standardize
the columns (≈37:52). The order matters. If the standardization used the whole data set, it would
carry information about the test set into the modelling, and "the test set is not really kept away from
anything" (≈37:04). Lock the test set away and use it once at the end, to estimate how the model will do
when deployed.

During training, part of the training set is further held out as a **validation set**, 20% in the
Colab, for early stopping and tuning (≈44:03–44:50). See
[overfitting and regularization](overfitting-and-regularization.md).

## Hand Keras arrays, with x and y apart

Keras takes NumPy arrays, so each dataframe is converted with `to_numpy` (≈38:39). The target column is
then removed into its own vector, because the inputs $x$ produce the predictions that are compared with
the actual $y$. For the heart-disease data that leaves 242 training rows by 29 input columns, matching
the 29 inputs of the network, with vectors of 242 and 61 targets (≈39:25–40:11).

## Labels for multi-class problems

Labels for several classes can be **sparse encoded**, as integers, or **one-hot encoded**, and the Keras
loss has to match the encoding (deck 3B, slides 20–26). Lecture 5 one-hot encodes its genre labels with
pandas' `get_dummies` (≈54:25). See
[multi-class classification and softmax](multiclass-classification.md).

## Randomness and reproducibility

Splitting is random, and so are the initial weights, the shuffle that divides the training data into
batches, and dropout. To let someone else reproduce your results, set a random seed with Keras's
`set_random_seed`. Even then "bitwise reproducibility" is not guaranteed, because of hardware and device
drivers (≈30:57–32:27).

Asked whether rerunning the same notebook gives the same accuracy, the lecturer says roughly, with
minute differences from hardware and device drivers; on his own machine it also depends on what else
has run in the notebook ([lecture 5](05-text-vectorization-and-bag-of-words.md) ≈1:09:13–1:10:00).

The preprocessing done here in Pandas can also be done inside Keras, which has preprocessing layers for
it (≈57:22).

## Text

Text has to be turned into numbers before a network can use it, by standardizing, tokenizing,
indexing and encoding it. See [text vectorization](text-vectorization.md) and
[the Bag of Words model](bag-of-words.md).

## Where this comes up

- [Lecture 2](02-training-deep-nns.md): the heart-disease data, and one-hot encoding to 29 inputs.
- [Lecture 3](03-keras-and-vision-basics.md): the heart-disease Colab's preprocessing, and label
  encodings for Fashion MNIST.
- [Lecture 5](05-text-vectorization-and-bag-of-words.md): the song-genre baseline, one-hot labels with `get_dummies`, and
  text vectorization.
