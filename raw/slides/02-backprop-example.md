---
title: Lecture 2 — Backprop example (handwritten worked example)
lecture: 2
slides: 14
source_pdf: https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec02_backprop_example.pdf
note: Slide N = PDF page N (N = 1..14). The pages are handwritten, in black with red, green and orange ink. Every page prints a small number bottom-right that matches the PDF page. Page 14 is OCW's appended end page, not lecture content.
---

# Lecture 2 — Backprop example: page-by-page

Transcription of [`mit15_773_s24_lec02_backprop_example.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec02_backprop_example.pdf) (MIT 15.773 Hands-On Deep Learning, Spring 2024; lecturer Rama Ramakrishnan). The pages are handwritten, so this file transcribes the handwriting: every line, equation and diagram, with equations in LaTeX using the author's own symbols. The lecturer says in class that he will not walk through this example and points students to it. It computes the gradient of a squared-error loss for a tiny network, first "the old-fashioned way" and then by backpropagation on a computational graph. Where the written algebra contains an apparent slip it is transcribed as written and flagged `[sic: …]` (one: slide 7 writes $\partial a_2/\partial w_1$ for $\partial a_2/\partial w_2$).

Colour code used throughout: green ink marks the bias $b$, red ink marks the weights $w_1$, $w_2$, black is everything else; orange marks the phrase "computational graph" and the orange-filled operation boxes in the graph.

## Numbering

Headings use the PDF page number (1..14), one heading per page. Every page prints a plain number in the bottom-right corner that equals the PDF page number (the handwritten pages 1–13 and OCW's end page 14), so there is no mismatch to ignore. The PDF's text layer is OCR of the handwriting and should not be searched for content.

## Contents

| Slides | Section |
| ------ | ------- |
| 1-2 | The setup: two-input linear model with no hidden layers; its parameters; model and MSE loss |
| 3 | Gradient "the old-fashioned way" (direct differentiation) |
| 4-7 | Reorganising the calculation with intermediate variables $a_1$, $a_2$, $\hat{y}$; chain rule; local derivatives; putting it together |
| 8-13 | Backpropagation on the computational graph |
| 14 | MIT OpenCourseWare end page |

---

## Slide 1 — Consider a two-input problem with a continuous dependent variable and no hidden layers

Handwritten: "Consider a two-input problem with a continuous dependent variable and no hidden layers."

Diagram: two circled inputs, $x_1$ (upper) and $x_2$ (lower), each with an arrow pointing to a single circle on the right; that circle contains a short diagonal slash (the linear-activation icon). An arrow leaves it to the right. No labels on the edges yet.

Below: "What are the parameters to be learned?"

*Figure: yes — two inputs $x_1$, $x_2$ feeding one linear output node, no hidden layer*

## Slide 2 — The model and MSE loss

The network of slide 1 again, now with labels: red $w_1$ on the arrow from $x_1$, red $w_2$ on the arrow from $x_2$, and a green $b$ above the output node (the linear slash icon is still inside it; an arrow leaves to the right).

A boxed label at lower left: "The model and MSE loss". To its right:

$$\text{model}(x_1, x_2) = b + w_1 x_1 + w_2 x_2$$

$$\text{loss} = (\text{model}(x_1, x_2) - y)^2$$

$$\text{Loss} = (b + w_1 x_1 + w_2 x_2 - y)^2$$

(In the equations $b$ is green and $w_1$, $w_2$ are red.)

*Figure: yes — the two-input linear network with $w_1$, $w_2$ on the edges and bias $b$ on the node*

## Slide 3 — Let's calculate ∇Loss the "old-fashioned" way

Handwritten: "Let's calculate $\nabla \text{Loss} = \left[\dfrac{\partial \text{Loss}}{\partial b}\ \ \dfrac{\partial \text{Loss}}{\partial w_1}\ \ \dfrac{\partial \text{Loss}}{\partial w_2}\right]$ the "old-fashioned" way" followed by a smiley face.

At left: $\text{Loss} = (b + w_1 x_1 + w_2 x_2 - y)^2$, with three arrows fanning out to the three partial derivatives on the right:

$$\frac{\partial \text{Loss}}{\partial b} = 2\,(b + w_1 x_1 + w_2 x_2 - y)$$

$$\frac{\partial \text{Loss}}{\partial w_1} = 2\,(b + w_1 x_1 + w_2 x_2 - y)\,x_1$$

$$\frac{\partial \text{Loss}}{\partial w_2} = 2\,(b + w_1 x_1 + w_2 x_2 - y)\,x_2$$

(Colours as before: $b$ green, $w_1$, $w_2$ red.)

*Figure: no*

## Slide 4 — Now, let's organize the calculations a bit differently

A box at upper left: "Now, let's organize the calculations a bit differently". To its right:

"Let"

$$a_1 = w_1 x_1$$

$$a_2 = w_2 x_2$$

$$\hat{y} = b + a_1 + a_2$$

"Plugging" with an up-right arrow and then "into" with a down arrow (the arrows point from the three definitions to the loss expression below):

$$\text{Loss} = (b + w_1 x_1 + w_2 x_2 - y)^2$$

"we get:"

$$\text{Loss} = (\hat{y} - y)^2$$

At the bottom, three unanswered questions:

$$\frac{\partial \text{Loss}}{\partial b} = ?\qquad \frac{\partial \text{Loss}}{\partial w_1} = ?\qquad \frac{\partial \text{Loss}}{\partial w_2} = ?$$

*Figure: no*

## Slide 5 — We can apply the Chain Rule!

A box at top centre: "We can apply the Chain Rule!"

Left, the four equations:

$$\text{Loss} = (\hat{y} - y)^2,\quad \hat{y} = b + a_1 + a_2,\quad a_1 = w_1 x_1,\quad a_2 = w_2 x_2$$

A double arrow ("$\Rightarrow$") points to the chain-rule expansions on the right:

$$\frac{\partial \text{Loss}}{\partial b} = \frac{\partial \text{Loss}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial b}$$

$$\frac{\partial \text{Loss}}{\partial w_1} = \frac{\partial \text{Loss}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial a_1} \cdot \frac{\partial a_1}{\partial w_1}$$

$$\frac{\partial \text{Loss}}{\partial w_2} = \frac{\partial \text{Loss}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial a_2} \cdot \frac{\partial a_2}{\partial w_2}$$

Below: "Next, we need to calculate"

$$\frac{\partial \text{Loss}}{\partial \hat{y}},\quad \frac{\partial \hat{y}}{\partial b},\quad \frac{\partial a_1}{\partial w_1}\ \text{ and }\ \frac{\partial a_2}{\partial w_2}$$

(The factors $\partial\hat{y}/\partial a_1$ and $\partial\hat{y}/\partial a_2$ are not in this list, but slide 6 supplies them.)

*Figure: no*

## Slide 6 — But that's easy!

Handwritten: "But that's easy!" Four rows, each an equation on the left, a long arrow, and its derivative on the right:

$$\text{Loss} = (\hat{y} - y)^2 \;\longrightarrow\; \frac{\partial \text{Loss}}{\partial \hat{y}} = 2(\hat{y} - y)$$

$$\hat{y} = b + a_1 + a_2 \;\longrightarrow\; \frac{\partial \hat{y}}{\partial b} = \frac{\partial \hat{y}}{\partial a_1} = \frac{\partial \hat{y}}{\partial a_2} = 1$$

$$a_1 = w_1 x_1 \;\longrightarrow\; \frac{\partial a_1}{\partial w_1} = x_1$$

$$a_2 = w_2 x_2 \;\longrightarrow\; \frac{\partial a_2}{\partial w_2} = x_2$$

(The final "1" is underlined.)

*Figure: no*

## Slide 7 — Putting everything together

Handwritten: "Putting everything together:"

$$\frac{\partial \text{Loss}}{\partial b} = \frac{\partial \text{Loss}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial b} = 2(\hat{y} - y) \cdot 1$$

$$\frac{\partial \text{Loss}}{\partial w_1} = \frac{\partial \text{Loss}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial a_1} \cdot \frac{\partial a_1}{\partial w_1} = 2(\hat{y} - y) \cdot 1 \cdot x_1$$

$$\frac{\partial \text{Loss}}{\partial w_2} = \frac{\partial \text{Loss}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial a_2} \cdot \frac{\partial a_2}{\partial w_1} = 2(\hat{y} - y) \cdot 1 \cdot x_2$$

[sic: in the third row the last factor is written $\partial a_2 / \partial w_1$ (red $w_1$); by the chain rule on slide 5 and the result $x_2$ it should be $\partial a_2 / \partial w_2$.]

Then: "You can check this matches the "old fashioned" calculation!!" (the old-fashioned results of slide 3 equal these once $\hat{y} = b + w_1 x_1 + w_2 x_2$ is substituted).

Then: "OK, we are finally ready for backpropagation!!" followed by a smiley face.

*Figure: no*

## Slide 8 — We will rewrite these equations as a computational graph

A box at top holds the four equations:

$$a_1 = w_1 x_1,\quad a_2 = w_2 x_2,\quad \hat{y} = b + a_1 + a_2,\quad \text{Loss} = (\hat{y} - y)^2$$

A down arrow leads to the text "We will rewrite these equations as a computational graph" (the words "computational graph" in orange), then another down arrow to the graph, drawn left to right with inputs on the left and Loss on the right. Variables are circled (black outline); operations are orange-filled boxes.

- Top left: circled $x_1$ and circled red $w_1$, both with arrows into an orange box marked "*". An arrow goes from that box to circled $a_1$.
- Lower left: circled $x_2$ and circled red $w_2$, both with arrows into a second orange "*" box, with an arrow to circled $a_2$.
- Circled green $b$ sits above the centre with an arrow pointing down into an orange "+" box.
- Arrows from circled $a_1$ and from circled $a_2$ also go into the "+" box.
- An arrow from the "+" box goes to circled $\hat{y}$.
- An arrow from $\hat{y}$ goes to an orange box marked $(\square - \square)^2$ (two empty slots, squared). Circled $y$ is below it with an arrow pointing up into that box.
- An arrow from the box goes to circled "Loss".

*Figure: yes — computational graph: $x_1$,$w_1$ → * → $a_1$; $x_2$,$w_2$ → * → $a_2$; $b$,$a_1$,$a_2$ → + → $\hat{y}$; $\hat{y}$,$y$ → $(\square-\square)^2$ → Loss*

## Slide 9 — We can "attach" each of those little derivatives we calculated earlier to the graph

Handwritten: "We can "attach" each of those little derivatives we calculated earlier to the graph."

Below are three copies of the slide 8 computational graph, stacked vertically, each with one or two new magenta (pink) annotations. Magenta is the new ink on this page and marks a local derivative attached to an edge, with a magenta arrow running backwards (right to left) along that edge. The graph in each copy: circled $x_1$ and circled red $w_1$ feed an orange "*" box that feeds circled $a_1$; circled $x_2$ and circled red $w_2$ feed an orange "*" box that feeds circled $a_2$; circled green $b$ feeds down into an orange "+" box, which also receives $a_1$ and $a_2$; "+" feeds circled $\hat{y}$; $\hat{y}$ and circled $y$ (arrow up from below) feed the orange $(\square - \square)^2$ box; that box feeds circled "Loss".

First copy: a long magenta arrow curves from circled "Loss" back over the top to circled $\hat{y}$, labelled above it in magenta:

$$\frac{\partial \text{Loss}}{\partial \hat{y}} = 2(\hat{y} - y)$$

Second copy: a magenta arrow curves from $\hat{y}$ back up to circled $b$, labelled:

$$\frac{\partial \hat{y}}{\partial b} = 1$$

(the "1" is underlined).

Third copy: two magenta arrows leave the $\hat{y}$ end of the "+" box. One points up-left to circled $a_1$, labelled $\dfrac{\partial \hat{y}}{\partial a_1} = 1$; the other points down-left to circled $a_2$, labelled $\dfrac{\partial \hat{y}}{\partial a_2} = 1$ (the "1" is underlined in the second).

Page number "9" bottom-right.

*Figure: yes — three copies of the computational graph with magenta backward arrows labelled $\partial\text{Loss}/\partial\hat{y} = 2(\hat{y}-y)$, $\partial\hat{y}/\partial b = 1$, $\partial\hat{y}/\partial a_1 = 1$, $\partial\hat{y}/\partial a_2 = 1$*

## Slide 10 — Putting everything together

Top: another copy of the computational graph (same nodes and edges as slide 8) with two magenta backward arrows. One curves from circled $a_1$ back along the "*" box to circled $w_1$, labelled:

$$\frac{\partial a_1}{\partial w_1} = x_1$$

The other curves from circled $a_2$ back to circled $w_2$, labelled:

$$\frac{\partial a_2}{\partial w_2} = x_2$$

Middle, handwritten in black: "Putting everything together:"

Bottom: the full computational graph carrying all of the magenta local derivatives at once:

- From Loss back to $\hat{y}$: $\dfrac{\partial \text{Loss}}{\partial \hat{y}} = 2(\hat{y} - y)$.
- From $\hat{y}$ back to $b$: $\dfrac{\partial \hat{y}}{\partial b} = 1$.
- From $\hat{y}$ back to $a_1$: $\dfrac{\partial \hat{y}}{\partial a_1} = 1$.
- From $\hat{y}$ back to $a_2$: $\dfrac{\partial \hat{y}}{\partial a_2} = 1$.
- From $a_1$ back to $w_1$: $\dfrac{\partial a_1}{\partial w_1} = x_1$.
- From $a_2$ back to $w_2$: $\dfrac{\partial a_2}{\partial w_2} = x_2$.

Page number "10" bottom-right.

*Figure: yes — computational graph with all six local derivatives attached as magenta backward arrows*

## Slide 11 — This is called BACKPROPAGATION

Handwritten: "To calculate $\dfrac{\partial \text{Loss}}{\partial\,\text{any parameter}}$, start from the loss and travel backwards to the parameter, multiplying the partial derivatives as you go." (The words "any parameter" are written in red in the denominator; "backwards" is underlined.)

Then: "This is called" followed by the word "BACKPROPAGATION" in capitals, highlighted in a pink marker band.

Page number "11" bottom-right.

*Figure: no*

## Slide 12 — To calculate ∂Loss/∂w₁, for example

Handwritten: "To calculate $\dfrac{\partial \text{Loss}}{\partial w_1}$, for example:" (the $w_1$ in red).

Below it is the computational graph of slide 10 with all six magenta local derivatives attached: $\dfrac{\partial \text{Loss}}{\partial \hat{y}} = 2(\hat{y}-y)$ (Loss back to $\hat{y}$), $\dfrac{\partial \hat{y}}{\partial b} = 1$, $\dfrac{\partial \hat{y}}{\partial a_1} = 1$, $\dfrac{\partial \hat{y}}{\partial a_2} = 1$, $\dfrac{\partial a_1}{\partial w_1} = x_1$ and $\dfrac{\partial a_2}{\partial w_2} = x_2$. In addition, a yellow highlighter marks one path: circled "Loss", the magenta arrow from Loss back to circled $\hat{y}$, circled $\hat{y}$, the magenta arrow from $\hat{y}$ back to circled $a_1$ (labelled $\partial\hat{y}/\partial a_1 = 1$), circled $a_1$, the magenta arrow from $a_1$ back to circled red $w_1$ (labelled $\partial a_1/\partial w_1 = x_1$), and circled $w_1$. The yellow path is the backward route from Loss to $w_1$; it does not pass through $b$, $a_2$, $w_2$, $x_1$, $x_2$ or $y$, and those nodes and the arrows on them are not highlighted.

Then: "Multiplying all the partial derivative on the yellow path, we get the answer:" (the words "yellow path" highlighted in yellow). [sic: "derivative" is written in the singular.]

In magenta, with $w_1$ in the left-hand denominator in red:

$$\frac{\partial \text{Loss}}{\partial w_1} = \frac{\partial \text{Loss}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial a_1} \cdot \frac{\partial a_1}{\partial w_1} = 2(\hat{y}-y) \cdot 1 \cdot x_1$$

Page number "12" bottom-right.

*Figure: yes — computational graph with magenta local derivatives and the yellow-highlighted backward path Loss → $\hat{y}$ → $a_1$ → $w_1$*

## Slide 13 — Does this match what we calculated earlier? YES!! Backprop is very efficient

Handwritten: "Does this match what we calculated earlier?" and below it, in a boxed frame: "YES!!"

Lower on the page, a heading underlined: "Backprop is very efficient". Three dashed points:

- "Calculate once and use many times (e.g. $\dfrac{\partial \text{Loss}}{\partial \hat{y}}$)".
- "(When more than one neuron in the layer) traversing backward is a series of matrix multiplications" (the words "matrix multiplications" highlighted in pink).
- "GPUs are perfect for matrix multiplications!!" (the word "GPUs" highlighted in pink).

At lower right, a small hatched square (a doodle, with two diagonal strokes), used as an end mark.

Page number "13" bottom-right.

*Figure: no*

## Slide 14 — MIT OpenCourseWare end page

Printed (typed, not handwritten): "MIT OpenCourseWare, https://ocw.mit.edu/"; "15.773 Hands-on Deep Learning, Spring 2024"; "For information about citing these materials or our Terms of Use, visit: https://ocw.mit.edu/terms." This is OCW's appended end page, not lecture content. It prints "14" bottom-right.

*Figure: no*
