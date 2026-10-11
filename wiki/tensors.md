# Tensors

A **tensor** is the generalization of a number, a vector and a table to any number of dimensions
([lecture 3](03-keras-and-vision-basics.md) ≈16:56–18:32; deck 3A, slides 21–25). Everything that
flows through a neural network, from the input to the output, is a tensor, which is where TensorFlow
gets its name (≈22:27–23:16). The lecturer warns that the idea is "simultaneously kind of easy to
understand and also slippery", and recommends chapter 2.2 of the textbook (deck 3A, slide 27;
≈21:42–22:27).

## Rank

Every tensor has a **rank**, its number of dimensions (deck 3A, slides 22–25):

| Rank | Also called | Example on the slides |
| --- | --- | --- |
| 0 | scalar | 42 |
| 1 | vector | (42, 23.4, 11.2) |
| 2 | matrix, table | a grayscale image's table of pixel values |
| 3 | "cube", a bunch of tables | the red, green and blue tables of a colour image |

A tensor can be simpler than a matrix: a single number is a tensor too (≈16:56). A student's answer
that a tensor is "a matrix, but generalizable with higher dimensions" was "correct, but incomplete" for
that reason.

## Examples from the course

- A **grayscale image** is a table of numbers, a tensor of rank 2 (≈18:32). A **colour image** is three
  tables, one per colour channel, so a tensor of rank 3 (≈1:00:26–1:01:12).
- A **colour video** is a stream of colour images. It is a tensor of rank 4 whose first dimension says
  which frame it is (deck 3A, slide 26 asks for this example; ≈18:32–19:19).
- A **list of videos** is a tensor of rank 5 (≈20:54).
- A **data set** is a tensor one rank higher than its examples. Fashion MNIST's training input is a
  tensor of rank 3: 60,000 images, each a 28 × 28 table, "just a list of rank 2 tensors"
  (≈1:14:22).
- Keras's `keras.Input` takes a *shape* rather than a size, because inputs can be matrices, cubes or
  4D tensors as well as vectors ([lecture 2](02-training-deep-nns.md) ≈12:28).

## How to think about one

The lecturer always thinks of a tensor as a list: an open square bracket, a bunch of things, a closed
square bracket (≈20:06). Its first dimension indexes the list, and each element is a tensor one rank
lower. So "a tensor of rank, say, 10, is just a list of rank 9 tensors", and you can always make a
more complicated tensor by taking a list of simpler ones (≈20:54–21:42). A tensor of shape 7 by 3 has
7 rows and 3 columns (≈20:06).

He remarks that the tensor concept "is actually something Einstein came up with" (≈21:42). If it feels as though you
"kind of understood but not quite, you're not alone… You have to pay the price and go through the
crucible" (≈22:27).

## Where this comes up

- [Lecture 2](02-training-deep-nns.md): why `keras.Input` takes a shape.
- [Lecture 3](03-keras-and-vision-basics.md): ranks 0 to 5, TensorFlow, and images as tensors.
- See also [images and computer-vision tasks](images-and-vision-tasks.md) and [Keras](keras.md).
