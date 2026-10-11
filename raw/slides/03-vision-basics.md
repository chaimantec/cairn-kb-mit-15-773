---
title: Lecture 3 — Deep Learning for Computer Vision: The Basics (slide deck 3B)
lecture: 3
deck: 3B (mit15_773_s24_lec03b.pdf)
slides: 29
source_pdf: https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec03b.pdf
note: Slide N = PDF page N (N = 1..29). Pages 2-7 and 9-15 print their own page number; pages 16-19 print N+1 (page 16 prints "17" ... page 19 prints "20") and pages 27-28 print N+1 ("28", "29"), because a hidden slide numbered 16 was never exported; pages 20-26 print no number; pages 1, 8 and 29 print none. Cite by PDF page. Page 29 is OCW's appended end page, not lecture content.
---

# Lecture 3 — Deep Learning for Computer Vision: The Basics (deck 3B): slide-by-slide

Text and figures of all 29 slides of
[`mit15_773_s24_lec03b.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec03b.pdf)
(MIT 15.773 Hands-On Deep Learning, Spring 2024; lecturer Rama Ramakrishnan), transcribed from the deck. This is the second of the two decks that lecture 3's video covers; the first is deck 3A, [`03-keras-tensorflow.md`](03-keras-tensorflow.md). Cite these slides as "deck 3B, slide N", where N is the PDF page number; slide 29 is OCW's appended end page. Diagrams, plots and photographs are described in prose since the KB is read as text.

Slides marked with an `*OCW notice: …*` line carry third-party material excluded from the Creative Commons licence; they are described in prose and never rendered as images. Code is reproduced as printed, with any curly quotes normalized to straight quotes inside code fences.

**Images.** Four figure slides carry a 1400px render under their heading, saved as `3b-N`: 14, 16, 17 and 18. The eight with an OCW exclusion notice (3, 4, 6 and 8–12) are never rendered. Slide 15 is superseded by 16, and the tables on 19–27 are reproduced exactly below.

## Numbering: slide numbers vs PDF pages

Headings use the PDF page number. Pages 2–7 and 9–15 print their own page number in the bottom-right corner. Pages 16–19 print the PDF page number plus one (PDF page 16 prints "17" ... PDF page 19 prints "20"), because a hidden slide numbered 16 was never exported from the original deck. Pages 20–26 print no number. Pages 27 and 28 again print the PDF page number plus one ("28" and "29"). Pages 1, 8 and 29 print no number. A reader who opens PDF page 17 therefore sees "18" printed there: cite and look up by PDF page.

## Contents

| Slides | Section |
| ------ | ------- |
| 1 | Title |
| 2–6 | Representing images digitally: grayscale pixel-intensity matrix (a 28x28 handwritten 5); colour images as three 0–255 channel matrices (Red, Green, Blue) |
| 7–12 | Key tasks in computer vision: image classification, classification and localization, object detection, semantic segmentation, instance segmentation |
| 13–14 | Image classification section; motivating application: Fashion MNIST (70,000 images, 10 clothing categories, goal over 90% accuracy) |
| 15–16 | A simple NN for grayscale clothing images: 28x28 input, flattened to 784 inputs, 256 hidden units, open question of the 10-way output |
| 17–19 | Multi-class classification: the softmax layer; summary table of output layers |
| 20–26 | Refresher on encodings (one-hot for binary, sparse and one-hot for multi-class) and matching Keras losses: `binary_crossentropy`, `sparse_categorical_crossentropy`, `categorical_crossentropy` |
| 27 | Summary table: loss functions for different output layers |
| 28 | Pointer to the Colab: translate this NN to Keras and train it |
| 29 | MIT OpenCourseWare end page |

---

## Slide 1 — Lecture 3B: Deep Learning for Computer Vision – The Basics

Title (red, bold): "Lecture 3B" and "Deep Learning for Computer Vision – The Basics" (wrapping onto three lines).

Bottom: MIT Management Sloan School logo (red/grey "MIT" above "MANAGEMENT" and "SLOAN SCHOOL"). Text beside it: "15.S04: Hands-on Deep Learning", "Spring 2024", and in red "Farias, Ramakrishnan".

*Figure: no*

## Slide 2 — Representing Images Digitally

Section divider: white text "Representing Images Digitally" on a dark red band with a grey block at the left.

*Figure: no*

## Slide 3 — How Grayscale Images are Represented

*OCW notice: Grayscale light intensity matrix © fast.ai. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use.*

Left: a small black-and-white image of a handwritten digit "5" (white strokes on black), a 28 by 28 pixel image. Right: the matrix of its pixel intensities, printed as R console output with column headers [,1] to [,28] and row headers [1,] to [28,]; 28 rows by 28 columns of integers between 0 and 255. Most entries are 0 (the black background). The nonzero entries trace the digit, for example row [6,] reads (from column 11) 3, 18, 18, 18, 126, 136, 175, 26, 166, 255, 247, 127; row [7,] includes 30, 36, 94, 154, 170, 253, 253, 253, 253, 253, 225, 172, 253, 242, 195, 64; row [8,] includes 49, 238, 253 repeated, 251, 93, 82, 82, 56, 39; the lower rows (for example [24,] and [25,]) include values such as 55, 172, 226, 253, 253, 253, 253, 244, 133, 11 and 136, 253, 253, 253, 212, 135, 132, 16. The matrix is too small to transcribe every cell reliably; only the values above were read.

Bottom left: "Image credit: fast.ai".

*Figure: yes — a 28x28 handwritten "5" beside its matrix of 0–255 pixel intensities*

## Slide 4 — How Grayscale Images are Represented

*OCW notice: Grayscale light intensity matrix © fast.ai. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use.*

The same "5" image and 28 by 28 intensity matrix as slide 3, now with three bullets below it:

- A grayscale image is a rectangular array of pixels
- The light intensity of each pixel is a number between 0 and 255. As the number increases from 0 to 255, the pixel goes from black through gray to white
- Each cell of the matrix shows the light intensity of the pixel at that location

*Build step: adds the three bullets and moves the notice below them, relative to slide 3 (the "Image credit: fast.ai" line is gone).*

*Figure: yes — a 28x28 handwritten "5" beside its matrix of 0–255 pixel intensities*

## Slide 5 — How Color Images are Represented

- Each pixel of a color image is represented by <u>three</u> intensities (not one), corresponding to the pixel's "redness", "blueness" and "greenness" (RGB)
- Each light intensity is still a number between 0 and 255
- Thus color images are represented as 3 matrices of numbers, corresponding to the Red, Green and Blue "channels" respectively.

("three" is underlined on the slide.)

*Figure: no*

## Slide 6 — How Color Images are Represented

*OCW notice: © unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use.*

Left: a photograph of a large domed neoclassical building with a columned front behind green lawn (the file panel below names it "killian.jpeg"; the lecturer calls it "a picture of Killian Court", transcript ≈1:00:26), with trees at both sides and blue sky; a small red square marks the top-left corner of the image. Beneath it is a file-info panel: "killian.jpeg", "JPEG image - 6 KB", Tags, Created "Today, 4:30 PM", Modified "Today, 4:30 PM", Content created "Friday, April 3, 2020 at 4:30 PM", Dimensions "200×200", Color space "RGB".

Right: three overlapping matrices, drawn as stacked cards with red borders and labelled "Red" (top right, back), "Green" (middle) and "Blue" (front), each showing a 10 by 10 block of intensities (columns [,1] to [,10], rows [1,] to [10,]), i.e. the pixels in the red-square corner. Only the front (Blue) matrix is fully visible:

| | [,1] | [,2] | [,3] | [,4] | [,5] | [,6] | [,7] | [,8] | [,9] | [,10] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [1,] | 251 | 232 | 233 | 237 | 230 | 243 | 255 | 255 | 250 | 246 |
| [2,] | 248 | 234 | 239 | 245 | 238 | 246 | 255 | 251 | 246 | 243 |
| [3,] | 255 | 241 | 238 | 236 | 229 | 241 | 253 | 249 | 238 | 234 |
| [4,] | 255 | 252 | 243 | 233 | 228 | 237 | 242 | 234 | 218 | 205 |
| [5,] | 255 | 255 | 249 | 231 | 228 | 231 | 224 | 215 | 204 | 166 |
| [6,] | 255 | 255 | 230 | 192 | 189 | 202 | 205 | 205 | 204 | 147 |
| [7,] | 231 | 231 | 188 | 140 | 138 | 152 | 156 | 159 | 177 | 136 |
| [8,] | 155 | 172 | 149 | 114 | 113 | 111 | 93 | 82 | 119 | 115 |
| [9,] | 107 | 130 | 108 | 93 | 113 | 100 | 67 | 66 | 81 | 95 |
| [10,] | 84 | 104 | 90 | 69 | 69 | 61 | 52 | 63 | 59 | 46 |

The Green matrix is visible only in its top rows: row [1,] reads 186, 171, 179, 185, 171, 172, 180, 171, 168, 180; row [2,] reads 177, 169, 180, 188, 176, 175, 178, 167, 169, 180; row [3,] is partly hidden (begins 175, 169, 174, 176, 169, 173, 178, 172, 171, 183, partly illegible). The Red matrix is visible in its top rows: row [1,] reads 147, 131, 138, 144, 131, 134, 144, 135, 133, 145; row [2,] reads 140, 131, 141, 149, 138, 138, 143, 132, 136, 146; its lower rows are hidden behind the Green and Blue cards.

*Figure: yes — a photograph of a domed building (killian.jpeg) with three stacked 10x10 RGB intensity matrices for its top-left corner*

## Slide 7 — Key Tasks in Computer Vision

Section divider: white text "Key Tasks in Computer Vision" on a dark red band with a grey block at the left.

*Figure: no*

## Slide 8 — Image Classification

*OCW notice: © unknown. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use.*

Four photographs stacked in a column, each with a thin arrow to a label on its right: a beagle puppy on grass with a red ball, labelled "Dog"; a close-up of a tabby cat's face, labelled "Cat"; a yellow Labrador puppy on a lead on grass, labelled "Dog"; a close-up of a ginger cat's face, labelled "Cat". The slide shows image classification as mapping a whole image to one label.

*Figure: yes — four dog and cat photographs each with an arrow to its class label*

## Slide 9 — Classification and Localization

*OCW notice: Sheep image © Nirmala Murali on Medium. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use.*

A photograph of three sheep (two lambs and an adult) standing on a gravel road beside grass. One orange rectangular box encloses all three sheep, with an orange tag at its top-left labelled "Sheep". Caption under the image: "Classification + Localization". Bottom left: underlined link text "Image Source".

*Figure: yes — sheep photograph with one bounding box labelled "Sheep" (classification plus localization)*

## Slide 10 — Object Detection

*OCW notice: Sheep image © Nirmala Murali on Medium. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use.*

The same sheep photograph with a separate coloured box around each animal: a red box labelled "Sheep 1" around the left lamb, a yellow box labelled "Sheep 2" around the middle lamb, and an orange box labelled "Sheep 3" around the adult. The boxes overlap slightly. Caption under the image: "Object Detection". Bottom left: underlined link text "Image Source".

*Figure: yes — sheep photograph with three separate labelled bounding boxes (object detection)*

## Slide 11 — Semantic Segmentation

*OCW notice: Sheep image © Nirmala Murali on Medium. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use.*

A flat colour-coded mask of the same scene: grey region labelled "Road" (left), green region labelled "Grass" (top right and right), and a single orange silhouette of all three sheep merged together labelled "Sheep". Caption under the image: "Semantic Segmentation". Below the notice, large text: "Every pixel needs to be classified into one of N categories". Bottom left: underlined link text "Image Source".

*Figure: yes — three-class colour mask (Road, Grass, Sheep) of the sheep scene*

## Slide 12 — Instance Segmentation

*OCW notice: Sheep image © Nirmala Murali on Medium. All rights reserved. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use.*

The same mask as slide 11, but the sheep silhouette is split into three differently coloured regions: pale yellow "Sheep 1" (left lamb), yellow "Sheep 2" (middle lamb) and orange "Sheep 3" (adult); grey "Road" and green "Grass" as before. Caption under the image: "Instance Segmentation". Below the notice, text: "Every pixel needs to be classified into one of N categories <u>and</u>" (the word "and" underlined) and "Different instances (e.g., Sheep 1, Sheep 2, Sheep 3) of the same category (e.g., Sheep) need to be identified". Bottom left: underlined link text "Image Source".

*Figure: yes — colour mask with Road, Grass and three separately coloured sheep instances*

## Slide 13 — Image Classification

Section divider: white text "Image Classification" on a dark red band with a grey block at the left.

*Figure: no*

## Slide 14 — Motivating application: Fashion MNIST

![Deck 3B, slide 14 — Motivating application: Fashion MNIST](../images/03-keras-and-vision-basics/3b-14.jpg)

Left text: "The fashion-mnist dataset consists of 70,000 images of clothing items across 10 categories." and "We will build a deep learning network <u>from scratch</u> to classify clothing into these 10 categories with over 90% accuracy!" ("from scratch" underlined).

Right: a large square mosaic of small greyscale clothing images on a black background, arranged in rows, each horizontal band mostly of one kind of item: the top bands look like tall items (trousers and dresses), followed by bands of tops and pullovers/coats, then shoes (sandals, sneakers), bags, and ankle boots at the bottom. The individual items are too small to identify one by one. Printed credit under it: "Sample of the Fashion MNIST images by Yuzamei. Source: Wikimedia Commons. License: CC BY-SA."

Bottom: "Source: https://www.kaggle.com/datasets/zalando-research/fashionmnist"

*Figure: yes — mosaic of Fashion MNIST clothing thumbnails*

## Slide 15 — A simple NN to classify grayscale clothing images

Hand-drawn style slide. At left, a red handwritten underlined label "Input" above a black square drawn with "28" written above it and "28" written to its left (a 28 by 28 input image). To the right of the square, a large red "?" (the network is not yet drawn).

*Figure: yes — hand-drawn 28x28 input square with a red question mark where the network will go*

## Slide 16 — A simple NN to classify grayscale clothing images

![Deck 3B, slide 16 — A simple NN to classify grayscale clothing images](../images/03-keras-and-vision-basics/3b-16.jpg)

(Printed page number: "17".)

Hand-drawn network, left to right, with a pink arrow between each stage. Column labels in handwriting, underlined: "Input", "\"Flattened\" Input", "Hidden Layer", "Output Layer".

- Input: the 28 by 28 square (sides marked "28").
- "Flattened" Input: a vertical column of circles labelled $x_1$, $x_2$, vertical dots, and $x_{784}$.
- Hidden Layer: a column of circles each containing a small drawn activation (a ReLU-like tick), numbered 1, 2, vertical dots, and 256 (256 hidden units).
- Output Layer: not drawn; in its place, red text: "How do we represent a 10-way output?"

*Build step: adds the flattened input, the 256-unit hidden layer and the red question about the output layer, relative to slide 15.*

*Figure: yes — hand-drawn network: 28x28 image, flattened 784 inputs, 256 hidden units, open question about the 10-way output*

## Slide 17 — Multi-Class Classification

![Deck 3B, slide 17 — Multi-Class Classification](../images/03-keras-and-vision-basics/3b-17.png)

(Printed page number: "18".)

Top text: "Suppose the output variable is categorical with 10 levels". Below, three regions separated by two vertical dark-red lines.

- Left region, text "We know how to output 10 numbers": a column of three circles (with vertical dots between the second and third, so ten in all), each containing a small straight diagonal line (a linear unit), each with an arrow pointing right.
- Middle region, text "We know how to output 10 probabilities": the same column, but each circle contains a small S-shaped curve (a sigmoid unit), each with an arrow pointing right.
- Right region: text "How do we output 10 probabilities that sum to 1.0?" with "that sum to 1.0" in red.

*Figure: yes — columns of linear units and of sigmoid units, with the open question of how to get ten probabilities that sum to 1*

## Slide 18 — The Softmax Layer

![Deck 3B, slide 18 — The Softmax Layer](../images/03-keras-and-vision-basics/3b-18.jpg)

(Printed page number: "19".)

Top text: "**softmax** takes in *n* arbitrary numbers and converts them to *n* probabilities" ("softmax" bold, "n" italic).

Diagram: at left a column of three linear-unit circles (diagonal line in each; vertical dots between the second and third), each with an arrow labelled $a_1$, $a_2$, ..., $a_{10}$ going right into a tall dark-red box labelled "softmax". From the box, three arrows (with vertical dots between the second and third) go right to three fractions, one per output:

$$\frac{e^{a_1}}{e^{a_1} + e^{a_2} + \ldots + e^{a_{10}}}, \qquad \frac{e^{a_2}}{e^{a_1} + e^{a_2} + \ldots + e^{a_{10}}}, \qquad \frac{e^{a_{10}}}{e^{a_1} + e^{a_2} + \ldots + e^{a_{10}}}$$

(the first, second and last outputs; the middle outputs are indicated by dots).

*Figure: yes — ten linear outputs a_1..a_10 feeding a softmax box that outputs the ratios e^{a_i} over the sum of e^{a_j}*

## Slide 19 — Summary: Output Layers for Regression and Classification

(Printed page number: "20".)

A two-column table with a dark-red header. Three of the cells in the "Output Layer" column show a drawn circle containing a symbol rather than words: a linear unit (small diagonal line) or a sigmoid unit (small S-curve).

| Output Variable | Output Layer |
| --- | --- |
| Single number (regression with a single output) | (circle containing a diagonal line: a linear unit) |
| Single probability (binary classification) | (circle containing an S-shaped curve: a sigmoid unit) |
| Vector of *n* numbers (regression with multiple outputs) | Stack of (circle containing a diagonal line: a linear unit) |
| Vector of n probabilities that add up to 1 (multi-class classification) | Softmax |

*Figure: no*

## Slide 20 — Refresher: How binary and categorical variables are encoded

(No printed page number.)

Title: "Refresher: How binary and categorical variables are encoded". A small table headed "BINARY CLASSIFICATION EXAMPLE" with two columns, "RAW DATA" and "ONE-HOT ENCODED VERSION":

| RAW DATA | ONE-HOT ENCODED VERSION |
| --- | --- |
| Yes | 1 |
| No | 0 |

The rest of the slide is empty.

*Figure: no*

## Slide 21 — Refresher: How binary and categorical variables are encoded

(No printed page number.)

The binary table of slide 20, plus a second table headed "MULTI-CLASS CLASSIFICATION EXAMPLE" with columns "RAW DATA" and "SPARSE ENCODED VERSION":

| RAW DATA | SPARSE ENCODED VERSION |
| --- | --- |
| T-shirt/top | 0 |
| Trouser | 1 |
| Pullover | 2 |
| Dress | 3 |
| Coat | 4 |
| Sandal | 5 |
| Shirt | 6 |
| Sneaker | 7 |
| Bag | 8 |
| Ankle boot | 9 |

*Build step: adds the multi-class table with sparse (integer) encoding relative to slide 20.*

*Figure: no*

## Slide 22 — Refresher: How binary and categorical variables are encoded

(No printed page number.)

The two tables of slide 21, plus a third column group headed "ONE-HOT ENCODED VERSION" to the right of the sparse column. Each row has ten 0/1 entries with a single 1 whose position equals the sparse code (the identity matrix):

| RAW DATA | SPARSE ENCODED VERSION | ONE-HOT ENCODED VERSION (10 entries) |
| --- | --- | --- |
| T-shirt/top | 0 | 1 0 0 0 0 0 0 0 0 0 |
| Trouser | 1 | 0 1 0 0 0 0 0 0 0 0 |
| Pullover | 2 | 0 0 1 0 0 0 0 0 0 0 |
| Dress | 3 | 0 0 0 1 0 0 0 0 0 0 |
| Coat | 4 | 0 0 0 0 1 0 0 0 0 0 |
| Sandal | 5 | 0 0 0 0 0 1 0 0 0 0 |
| Shirt | 6 | 0 0 0 0 0 0 1 0 0 0 |
| Sneaker | 7 | 0 0 0 0 0 0 0 1 0 0 |
| Bag | 8 | 0 0 0 0 0 0 0 0 1 0 |
| Ankle boot | 9 | 0 0 0 0 0 0 0 0 0 1 |

*Build step: adds the one-hot encoded version of the multi-class table relative to slide 21.*

*Figure: no*

## Slide 23 — Important: Pick the Keras crossentropy loss function that matches the encoding

(No printed page number.)

Title: "<u>Important</u>: Pick the Keras crossentropy loss function that matches the encoding" ("Important" underlined). Same tables as slide 22 (binary example, and multi-class with sparse and one-hot versions). To the right of the binary table, a dark-red arrow pointing right to red text: `binary_crossentropy`.

*Build step: adds the title change and the arrow to `binary_crossentropy` relative to slide 22.*

*Figure: no*

## Slide 24 — Important: Pick the Keras crossentropy loss function that matches the encoding

(No printed page number.)

As slide 23, with `binary_crossentropy` now in black, plus a dark-red arrow pointing down from the sparse-encoded column to red text below: `sparse_categorical_crossentropy`.

*Build step: adds the arrow and label `sparse_categorical_crossentropy` (red) relative to slide 23.*

*Figure: no*

## Slide 25 — Important: Pick the Keras crossentropy loss function that matches the encoding

(No printed page number.)

As slide 24, with an additional dark-red arrow pointing down from the one-hot-encoded column to the label `categorical_crossentropy` (red) at the lower right. `sparse_categorical_crossentropy` is now in black.

*Build step: adds the arrow and label `categorical_crossentropy` (red) relative to slide 24.*

*Figure: no*

## Slide 26 — Important: Pick the Keras crossentropy loss function that matches the encoding

(No printed page number.)

Identical content to slide 25, but the highlight moves: `sparse_categorical_crossentropy` is red again and `categorical_crossentropy` is black. The three pairings are therefore: binary encoding to `binary_crossentropy`, sparse (integer) encoding to `sparse_categorical_crossentropy`, one-hot encoding to `categorical_crossentropy`.

*Build step: only the red highlight changes (from `categorical_crossentropy` back to `sparse_categorical_crossentropy`) relative to slide 25.*

*Figure: no*

## Slide 27 — Summary: Loss functions for different output layers

(Printed page number: "28".)

The table of slide 19 with a third column "Loss Function" added; a red rectangle outlines the Loss Function column. The Output Layer cells again contain drawn units (linear: circle with a diagonal line; sigmoid: circle with an S-curve).

| Output Variable | Output Layer | Loss Function |
| --- | --- | --- |
| Single number (regression with a single output) | (linear unit) | Mean squared error |
| Single probability (binary classification) | (sigmoid unit) | Binary cross-entropy |
| Vector of *n* numbers (regression with multiple outputs) | Stack of (linear unit) | Mean squared error |
| Vector of n probabilities that add up to 1 (multi-class classification) | Softmax | Categorical cross-entropy |

*Build step: adds the Loss Function column (outlined in red) relative to slide 19.*

*Figure: no*

## Slide 28 — Let's translate this NN to Keras and train it!

(Printed page number: "29".)

Section divider: white text "Let's translate this NN to Keras and train it!" on a dark red band with a grey block at the left. Below, centred, an underlined orange hyperlink: "Colab" (its target URL is not printed on the slide).

*Figure: no*

## Slide 29 — MIT OpenCourseWare end page (not lecture content)

(No printed page number.) OCW's appended end page: "MIT OpenCourseWare", "https://ocw.mit.edu", "15.773 Hands-on Deep Learning", "Spring 2024", and "For information about citing these materials or our Terms of Use, visit: https://ocw.mit.edu/terms."

*Figure: no*
