# Keras

**Keras** is the library the course uses to build and train networks. Its appeal is how little code a
network takes: lecture 2's heart-disease model is four lines, and the lecturer calls the result "a
beautiful example of the power of abstraction" ([lecture 2](02-training-deep-nns.md) ≈16:20–19:25).
Lecture 2 shows how to *define* a model. [Lecture 3](03-keras-and-vision-basics.md) introduces Keras
and TensorFlow properly and trains that model in a Colab notebook.

## Keras on top of TensorFlow

**TensorFlow** is the library underneath. Its name comes from tensors "flowing from input to output"
through a network (lecture 3 ≈22:27–23:16; see [tensors](tensors.md)). It provides four things (deck
3A, slides 28–31; ≈23:16–24:48): automatic calculation of the gradients of arbitrarily complicated loss
functions ("You don't have to use the chain rule"), a library of optimizers including SGD and its
variants, automatic distribution of the computation across servers, and automatic adaptation of the
code to parallel hardware, GPUs and TPUs. A TPU, a Tensor Processing Unit, is "Google's GPU".

**Keras** sits on top of TensorFlow and adds conveniences (deck 3A, slides 32–33; ≈24:48): predefined
layers, flexible ways to specify architectures, easy ways to preprocess data, train models and report
metrics, and pretrained models to download and customize.

![Deck 3A, slide 32: a stack of three layers of boxes, Keras on top, TensorFlow in the middle, and CPU, GPU and TPU side by side at the bottom.](../raw/images/03-keras-and-vision-basics/3a-32.jpg)

*Lecture 3, deck 3A, slide 32 — Keras on TensorFlow on the hardware, from page 70 of the textbook.*

The course chose TensorFlow and Keras over PyTorch, "a fantastic framework as well", to stay accessible
to students without much programming background. PyTorch demands more object-oriented programming
(lecture 3 ≈25:33).

Keras has three ways to build models, the **sequential**, **functional** and **subclassing** APIs. The
course uses the **functional API** almost exclusively. The heart-disease model below is an example,
and section 7.2.2 of the textbook explains the API in detail (deck 3A, slide 34; ≈25:33–26:20). The
lecturer finds it "basically all I need", "a beautifully designed LEGO-block environment". Tutorials and
Colabs are at tensorflow.org and keras.io (slide 35).

## Defining a model, layer by layer

The model is lecture 2's heart-disease network: 29 inputs, one hidden layer of 16 ReLU units and one
sigmoid output (slides 13–28):

```python
input = keras.Input(shape=29)
h = keras.layers.Dense(16, activation="relu")(input)
output = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(input, output)
```

Layers are written left to right, as the network is drawn (slide 12):

- **`keras.Input(shape=29)`** is the input layer. It takes a *shape* because inputs can be matrices,
  3D cubes or 4D tensors as well as vectors (≈12:28).
- **`keras.layers.Dense(units, activation=...)`** is a fully connected layer. "Dense" means connected
  to every unit of the layers before and after (≈13:13–14:03).
- **Calling a layer on a previous output**, as in `(input)` or `(h)`, connects it to that layer,
  which it otherwise knows nothing about. The result gets a name so the next layer can use it
  (≈14:03–15:34).
- **`keras.Model(input, output)`** wraps everything from the input to the output into a model object
  that can be trained, evaluated and used for prediction (≈16:20).

![Slide 27: the heart-disease network drawn as input layer x1 to x29, a hidden layer of 16 ReLU units and a sigmoid output, with each Keras line under the layer it defines and the model line boxed.](../raw/images/02-training-deep-nns/slide-27.png)

*Lecture 2, slide 27 — each line of Keras sits under the layer it defines.*

## Training a model: summary, compile, fit, evaluate

Lecture 3 trains the heart-disease model in a Colab notebook that is not published on OCW. The calls
below are the ones the lecturer names, in the order he uses them (≈40:11–57:22).

- **`model.summary`**: run it as soon as a model is built. For each layer it shows the shape of the
  tensor coming in and going out and the layer's number of parameters, here 497 in all, which is worth
  checking by hand: $29 \times 16 + 16$ for the hidden layer and $16 \times 1 + 1$ for the output
  (≈40:57–41:42). Layers can be given names, which is optional and cosmetic (≈40:11).
- **`plot_model`** draws the network, which helps as networks grow (≈41:42).
- **Compile** with three choices: the loss (`binary_crossentropy` here), the optimizer (Adam) and the
  metrics to report (accuracy) (≈41:42–42:29). Metrics are only reported, never optimized, so any will
  do, and Keras offers "this dizzying list" (≈48:03). Compiling reorganizes the model for parallel and
  distributed computation (≈42:29–43:18).
- **`model.fit`** trains. It takes the x tensor, the y tensor, the number of epochs, the batch size,
  `verbose=1` for descriptive output, and `validation_split`, the fraction of the training data to hold
  out for validation, here 20% (≈44:50). Its output has a pair of lines per epoch: the batches run
  ("7/7"), the time, the training loss and accuracy, and the validation loss and accuracy (≈45:41–46:30).
  Writing `history = model.fit(...)` keeps these values in the dictionary `history.history` for
  plotting (≈46:30–47:17). Running `fit` again continues from where it left off rather than starting
  over (≈54:17).
- **Callbacks**, to come later in the course, can stop training when the validation loss stops
  improving or save the best model (≈54:17–55:04).
- **`model.evaluate`** on the test inputs and targets measures the final model once. The heart-disease
  network scored 83.6% accuracy against a 72.6% baseline (≈56:36–57:22).

Other helpers he mentions: `set_random_seed` for reproducibility (≈31:42–32:27), Keras preprocessing
layers that can replace preprocessing done in Pandas (≈57:22), and standard data sets such as Fashion
MNIST, which a `load_data` command loads already split into training and test sets (≈1:13:36–1:14:22).
For a multi-class output, pick the cross-entropy loss that matches the label encoding:
`sparse_categorical_crossentropy` for integer labels and `categorical_crossentropy` for one-hot ones
(≈1:12:04–1:12:51). See [multi-class classification and softmax](multiclass-classification.md).

## Answers from lecture 2

- **Custom activation functions** can be defined (≈17:08).
- **Multi-class outputs** use one unit per class with a softmax, which lecture 2 defers (≈17:54) and
  lecture 3 covers.
- **Deep networks** are built with a loop rather than typed out layer by layer (≈18:39–19:25).
- **The default activation** of a `Dense` layer was asked about. The lecturer was not sure, guessed
  ReLU, and said he would double-check (≈17:54–18:39). The lecture does not settle it. The slides always pass
  `activation=` explicitly.
- **Initialization schemes** are built in as defaults, and the lecturer advises keeping them
  (≈1:01:56).
- **Printed quotes.** Slides 15–28 print curly quotes in the code. Use straight quotes when typing it.

## Where this comes up

- [Lecture 2](02-training-deep-nns.md): the four-line heart-disease model.
- [Lecture 3](03-keras-and-vision-basics.md): TensorFlow and Keras, the functional API, and training the
  heart-disease model in a Colab.
- See also [anatomy of a neural network](neural-network-anatomy.md) for the design choices the code
  expresses.
