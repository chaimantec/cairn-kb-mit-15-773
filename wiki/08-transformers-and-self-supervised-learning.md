# Lecture 8 — Transformers (2/2) and Self-Supervised Learning

**Lecturer:** Rama Ramakrishnan ·
**Video:** [youtube.com/watch?v=v-lHsawHyaI](https://www.youtube.com/watch?v=v-lHsawHyaI) (76 min) ·
**Slides:**
[`mit15_773_s24_lec08.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)
(99 pages; transcribed in [`raw/slides/08-transformers-and-self-supervised-learning.md`](../raw/slides/08-transformers-and-self-supervised-learning.md)) ·
**Transcript:** [`raw/transcripts/08-transformers-and-self-supervised-learning.md`](../raw/transcripts/08-transformers-and-self-supervised-learning.md)

## What this lecture establishes

[Lecture 7](07-transformers.md) built a transformer encoder whose self-attention had **nothing to
learn**: every attention weight was a softmax of dot products of the incoming embeddings, so two
attention heads given the same input would give the same output. This lecture finishes the encoder
with the three elements lecture 7 left out (slide 12). The lecturer warns that its first half may be
"the most demanding 30 minutes of the entire course" (≈0:00).

First he redraws self-attention on an iPad as **matrix multiplication**: put the embeddings of a
sentence in a matrix $X$, and every pairwise dot product comes from the single product $XX^T$, so the
whole operation is a softmax of $XX^T$ times $X$. That compactness is what lets transformers run fast
on GPUs (≈7:46–13:17). Then he makes self-attention **tunable**. Each of the three copies of $X$ is
multiplied by its own matrix of learnable weights, the key, query and value matrices $A^K$, $A^Q$ and
$A^V$, and the result is the formula of "Attention Is All You Need",
$\text{softmax}(QK^T/\sqrt{d_k})\thinspace V$ (≈13:17–24:08; slides 16–20). Each attention head gets
its own three matrices, which is why different heads learn different patterns (slide 21). Two smaller
tweaks complete the encoder. **Residual connections** add each layer's input to its output, and
**layer normalization** standardizes each embedding (slides 23–27; ≈28:52–35:52).

The second half turns to using encoders. For **sequence classification** an artificial `<CLS>` token
is added at the start of every sentence, and its contextual embedding stands for the whole sentence
(slides 30–38). Then comes **self-supervised learning**, the way to pretrain on text that has no
labels. Hide part of the input and train a network to fill in the blanks (slides 39–78). Done with a
transformer encoder on massive amounts of English text, this gives **BERT** (slides 79–83).
Pretrained models like it sit on the Hugging Face Hub, and in the Colab the Hugging Face `pipeline`
function classifies sentiment, extracts named entities and answers a question with no training at
all (≈59:53–1:06:53). The lecture ends with transformers beyond text. The **Vision Transformer**
chops an image into patches, the **TabTransformer** embeds a table's categorical columns, and a
multi-modal classifier mixes text, image and table embeddings in one stack (slides 88–93;
≈1:06:53–1:16:13).

The catalog title, "Deep Learning for Natural Language – Transformers, Self-Supervised Learning", is
accurate. The deck's roadmap calls the lecture "Transformers (2/2)" (slide 1). The recording presents
slides 2–93, with two detours: lecture 7's Colab at ≈3:53–5:25, and the iPad derivation at
≈7:46–24:08 in place of slides 15–19, which the lecturer tells students to read afterwards. The deck's
last section, "Contrastive Learning (time permitting)" (slides 94–98), is not presented; Part 10
summarizes it from the slides alone. Neither Colab notebook is published on OCW. What is said below
about code follows the lecturer's narration, and only the functions and arguments he names are given.
The iPad drawings are not in the deck either, so they are described from the narration.

## Part 1 — Review, and lecture 7's model trained

### Where lecture 7 left off

The review restates why transformers are needed (slides 2–3; ≈0:00–0:47). The output must be as long
as the input, so that each word can be classified into the right slot, and the model must take each
word's **context** and the **order** of the words into account. Each word starts as a **stand-alone
embedding**, "pre-trained or random, doesn't really matter". Lecture 7's Colab started from random
weights. A **positional embedding** is added to it element by element, which gives the **positional
input embedding** $w_i$ that enters the encoder (slides 4–8; ≈0:47–1:33). Out of the encoder come
**contextual embeddings** $\hat w_i$, one per word (slide 9). For lecture 7's slot-filling problem
each contextual embedding goes through a ReLU layer and a softmax of its own, which classifies the
word into "one of 125 possibilities", as he says here (slide 10; ≈1:33–2:21). Lecture 7's slides say
123 slots, and its Colab's output vocabulary has 125 entries.

![Slide 8: six words, "the train slowly left the station", each with an arrow into a pink box "Input stand-alone embeddings" (annotated "initially random or pretrained (e.g., GloVe) weight vectors of embedding dimension D"); an olive box "Positional embeddings" (annotated "initially random of embedding dimension D"); for each word a pink and an olive arrow meet in a grey circle with a plus sign, and an arrow leads from each circle to a row of four grey cells, labelled w1 to w6 and together "Positional input embeddings", with a bracket marking the embedding dimension D.](../raw/images/08-transformers-and-self-supervised-learning/slide-8.jpg)

*Slide 8 — each word's stand-alone embedding plus its position's embedding gives the positional input embedding the encoder receives. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

A student asks how to decide between adding more attention heads and more transformer blocks; GPT-3,
the lecturer says, has 96 blocks (≈2:21–3:53). Both are worth doing. More heads in a block pick up
more patterns "at that level of abstraction". More blocks go up the levels of abstraction, as later
convolutional filters build on earlier ones, from lines to edges to noses, eyes and faces. A block
typically has "maybe a dozen heads or five, six, a dozen heads". The bigger the model, the more
capable it is, provided there is enough data to train it without overfitting.

### Lecture 7's Colab, trained

Lecture 7 set up its slot-filling model on the ATIS travel queries but stopped before training it
(see [lecture 7](07-transformers.md#part-8--the-colab-atis-slot-filling)). The lecturer does not
re-run it here, because it takes time, but reports what it does (≈3:53–5:25). Trained for 10 epochs
with a batch size of 64, "I just arbitrarily chose 64", the model reaches **99% accuracy** on the test
set with a single transformer block. There is a catch. Most words carry the slot O, "other", so a
naive model that labels every word O already does well, as with any classification problem where one
class predominates. On the slots that are not O, "which is really what you care about", accuracy is
**93%**, "which is actually pretty good". The notebook ends with queries to try, including ones meant
to break it, such as the cheapest flight from MIT to Mars.

## Part 2 — Self-attention as matrix multiplication

### Three copies of the input

Lecture 7 explained self-attention one word at a time: dot products of "station" with every word, a
softmax to turn them into weights that add up to 1, and a weighted average of the embeddings, which is
the contextual embedding of "station" (slide 13; ≈6:12–7:46). The same computation has to be repeated
for every word, and it can be organized "very efficiently". Without that, the lecturer says, "there
wouldn't be any transformer revolution", because packaging it as a few matrix operations is what puts
the whole thing on GPUs (≈7:00–7:46). He draws it on the iPad ("concocted last night because I was
very unhappy with the slides that follow"). The drawing is not in the deck.

Take the three-word sentence "I love HODL", with embeddings $w_1$, $w_2$, $w_3$ that already include
their positional encoding (≈8:38). Call the matrix of these three vectors $X$; in the drawing each
vector lies horizontally, one per row. Self-attention makes **three copies** of $X$ (≈8:38–9:23).

### All the dot products at once

Self-attention needs the dot product of every pair of words, "how attracted or related they are". Take
the first copy and transpose it, so the vectors stand vertically, and multiply the second copy by it.
Each cell of the resulting grid is the dot product of one pair of words, so the whole grid is the
single matrix product $XX^T$: "Boom, done" (≈9:23–10:57). Three words make 9 multiplications. A
million words would make "on the order of" a trillion, "on the order of" because $w_1 \cdot w_3$ equals
$w_3 \cdot w_1$, so some of them repeat (≈10:57).

The dot products can be negative, so a **softmax** is applied to each row: exponentiate every number
and divide by the row's total (≈10:57–11:42). Each row then holds weights that add up to 1. In the
drawing the first row reads 0.1, 0.7 and 0.2, "for the first word, take 0.1 of the first one, 0.7 of
the second, 0.2 of the third, and add them up". Multiplying this table by the third copy of $X$ does
exactly those weighted sums. The first contextual embedding is $0.1\thinspace w_1 + 0.7\thinspace w_2 + 0.2\thinspace w_3$,
and every other row works the same way (≈11:42–12:30). So the whole self-attention operation is one
compact formula, which the lecturer calls the magic that makes it "lightning fast on GPUs" (≈12:30–13:17):

$$\text{contextual embeddings} = \text{softmax}(XX^T)\thinspace X$$

where the softmax is taken row by row. Each row of the result is one word's contextual embedding,
$\hat w_i$ in lecture 7's notation (see [transformers](transformers.md#self-attention)).

## Part 3 — Making self-attention tunable

### The problem: nothing to learn

As built so far, the self-attention layer takes in the $w$'s and returns the $\hat w$'s, but "there
are no weights. There are no biases. There are no coefficients" (≈13:17). So what is learned? The
deck puts the consequence in the title of slide 14: "As it stands, the self-attention heads don't have
any internal weights so all the heads will produce the same output embeddings. We need to make each of
them 'tunable'." The fix is to inject weights, so that training can adapt the layer "to the
particularities of whatever problem you're working on" (≈13:17–14:03).

### A matrix for each copy

The iPad drawing repeats the three copies, but this time each copy of $X$ is first **multiplied by a
matrix** of weights learned by backpropagation (≈14:50–18:40). The first is the **key matrix**, which
the lecturer writes AK and the slides $A^K$, and it produces transformed embeddings called $K$. The
second, the query matrix $A^Q$, produces $Q$, and the third, the value matrix $A^V$, produces $V$:

$$K = XA^K, \qquad Q = XA^Q, \qquad V = XA^V$$

Then everything runs as before. The grid of pairwise dot products is now $QK^T$, the softmax is
applied to each row, and the table of weights multiplies $V$ (≈17:08–18:40):

$$\text{contextual embeddings} = \text{softmax}(QK^T)\thinspace V$$

At one point the captions have him multiply the softmax table "by 1"; the formula he lands on
multiplies it by $V$, the third copy (≈18:40). So the self-attention layer now holds **three
learnable matrices** (≈18:40–19:28).

Why should this help? Giving the layer weights it can learn gives it "more expressive power, more
modeling capacity", and whether it uses that capacity depends on the data and the training
(≈15:36). If transforming $X$ does not help, training can make the matrix the **identity matrix**, which
multiplies $X$ and gives $X$ back: "I have nothing to learn here". The names key, query and value come
from information retrieval. The lecturer finds that interpretation unhelpful, because transformers are
used far outside information retrieval, so he presents the matrices simply as what makes each copy
tunable (≈15:36–16:22).

### The same thing on the slides

The slides, which the lecturer skips in favour of the drawing and asks students to read afterwards
"because two different ways of understanding something always helps", make the same change one word
at a time (≈24:08–24:54; slides 15–20). Slide 15 refers to a handout, "HODL-SP24-Section-A-The
Self-Attention Layer.pdf", that is not on OCW. Slide 16 multiplies the positional input embeddings by a
matrix $\mathbf{A}$ before taking their dot products, so the weight that "station" ($w_6$) gives the
first word becomes

$$s_{1,6} = \frac{\exp\langle \mathbf{A} w_1, \mathbf{A} w_6 \rangle}{\sum_{i=1}^{6} \exp\langle \mathbf{A} w_i, \mathbf{A} w_6 \rangle}$$

where $\langle \cdot , \cdot \rangle$ is the dot product. "Multiplying by a matrix $\mathbf{A}$ = a dense
layer with a linear activation", and its numbers are "learnable" weights optimized with backprop, which
is what "tunable" means (slides 16–17). Slide 18 uses two different matrices, the key matrix $A^K$ for
the word being attended to and the query matrix $A^Q$ for the word doing the attending, "twice as
tunable as before":

$$s_{1,6} = \frac{\exp\langle A^K w_1, A^Q w_6 \rangle}{\sum_{i=1}^{6} \exp\langle A^K w_i, A^Q w_6 \rangle}$$

Finally a third matrix, $A^V$, is applied before the weighted sum (slide 19):

$$\hat w_6 = s_{1,6} A^V w_1 + s_{2,6} A^V w_2 + s_{3,6} A^V w_3 + s_{4,6} A^V w_4 + s_{5,6} A^V w_5 + s_{6,6} A^V w_6$$

instead of lecture 7's $\hat w_6 = s_{1,6} w_1 + \dots + s_{6,6} w_6$. The values in $A^K$, $A^Q$ and
$A^V$ are learned by stochastic gradient descent, which makes them tunable and lets the attention heads
learn different patterns (slide 20).

### The formula in the paper

The lecturer then opens "Attention Is All You Need" to show the transformer formula, which slide 20's
footer also prints (≈22:31–24:08):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The only part his drawing ignored is the division by $\sqrt{d_k}$, which he calls dividing "by a
particular number" and does not define further. When some of the numbers going into a softmax are very
big, all the others "get squashed to 0", and dividing keeps any one number from getting too big so that
the gradient flows properly: "important, but a bit of a technical detail". Otherwise it is "exactly the
formula we derived". "And congratulations, now you understand it" (≈24:08).

### Questions about the matrices

- **Are $A^K$, $A^Q$ and $A^V$ related?** No, they are independent matrices (≈19:28). The letter A is
  "just notation", since Q, K and V were already taken for the results (≈25:42).
- **How big are they?** If the embeddings are 50 long and the output should be 50 long too, a matrix
  is $50 \times 50$, 2,500 weights (≈19:28–20:13). Their size is something like a hyperparameter. Most
  implementations keep the key and query size equal to the embedding size, 10 in and 10 out. The value
  matrix often tends to be smaller. With 100-long embeddings and five heads, say, the keys and queries
  might be 100 to 100 but the values 100 to 20, because the five heads' values are concatenated back
  into 100 again; "I can tell you more offline" (≈25:42–26:30). Lecture 7's Colab, with five heads on
  512-long embeddings, reports about 5.3 million encoder parameters. The count worked out on the
  [lecture 7 page](07-transformers.md#part-8--the-colab-atis-slot-filling) matches that total when all
  three matrices are 512 to 512 per head. That is that page's arithmetic, not a statement of the
  lecture's.
- **What does each matrix learn?** "We don't know." The layer can pay attention to every pair of words,
  and the matrices give it ways to transform what comes in into potentially useful things. "The
  punchline is that yeah, it helps massively" (≈20:13). A general pattern in deep learning is that to
  add capacity you take a vector in the middle of a network, run it through a matrix, and better still
  through a ReLU as well (≈20:13–20:59).
- **Why $V$ rather than $X$ in the last step?** There are three copies, so there are three
  opportunities to make them more expressive, "we'll use all of them" (≈20:59–21:46).
- **Is this fine-tuning?** "We're not fine-tuning it, just to be clear on the vocabulary." The matrices
  are more weights, updated by backpropagation with everything else when the model is trained. Adding
  parameters always risks overfitting, so watch the validation set (≈21:46–22:31).
- **Where do the values start?** At random, like any weights (≈26:30).

### Each head gets its own matrices

This is what makes multi-head attention worthwhile (≈24:54). Without parameters, two attention heads
given the same input give identical outputs. With them, each head has its own $A^K$, $A^Q$ and $A^V$,
which start from different values and end up "very different" after training, so the heads' outputs
differ. Ten attention heads means 30 matrices to learn (≈27:17; slide 21).

![Slide 21: six grey positional input embeddings, w1 to w6, enter a red-outlined box and six blue contextual embeddings, w1-hat to w6-hat, leave it. Inside, two gold blocks labelled "Attention Head" with "(A^K, A^Q, A^V)" in purple feed a block "Concatenate and Project", which feeds a tall block "Feed-Forward Layer". Footer: "Important: Each attention head will have its own A^K, A^Q, A^V".](../raw/images/08-transformers-and-self-supervised-learning/slide-21.jpg)

*Slide 21 — multi-head attention: each head carries its own key, query and value matrices. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

Are there kinds of heads to aim for (≈27:17–28:04)? In a sentence one pattern may be about the meanings
of words, "bank", "station", "train"; another about grammar and tense; another about tone. A priori
nobody knows how many such patterns there are, just as nobody knows in advance how many filters a
convolutional network needs, so "you just give it a lot of capacity so that it can learn whatever it
wants" (≈28:04).

## Part 4 — Residual connections and layer normalization

### Residual connections

Slide 22 draws the encoder as two layers, a multi-head attention layer and a feed-forward layer. A
**residual connection** adds each layer's **input** to its **output**, element by element (slide 23;
≈28:52–31:12). The attention layer turns $w_1, \dots, w_6$ into some intermediate vectors of the same
length, so the original $w$'s can simply be added to them, and the feed-forward layer gets the same
treatment.

Why go to the trouble of transforming something and then add the original back? Once a network has
transformed its input, everything after that point sees only the transformed version: "But what if
your transformations are not great?" (≈29:37). As "an insurance policy", send the original along with
the transformed stuff. Passing both forward is the idea behind what is called a wide and deep network,
but concatenating the original at every layer would make the vectors longer and longer. Adding keeps
the size the same: a 100-long input plus a 100-long output is another 100-long vector (≈30:23).
Residual connections "improve the gradient flow during backpropagation dramatically", which is why they
are used so heavily, and not only in transformers. ResNet, from the computer-vision lectures, stands
for Residual Net, "because it was the first network to actually figure this out" (≈31:12; see
[transfer learning](transfer-learning.md)).

![Slide 23: "Another tweak: Residual Connection". Grey vectors w1 to w6 enter a gold "Multi-head Attention Layer" block, whose outputs enter a gold "Feed-Forward Layer" block, whose outputs are blue vectors w1-hat to w6-hat. A thick purple arrow loops under each block, from its input side to a purple circle with a plus sign at its output side. Text: "We sum the input embedding to the output embedding of the Attention / Feed-Forward Layers. This helps gradients flow better during backpropagation."](../raw/images/08-transformers-and-self-supervised-learning/slide-23.jpg)

*Slide 23 — a residual connection routes each layer's input around it and adds it to the layer's output. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

### Layer normalization

Since the start of the course, the inputs to a network have been kept in a narrow, well-defined range
(≈31:12–32:45). Pixel values were divided by 255 to lie between 0 and 1, and the heart-disease features
were standardized by subtracting the mean and dividing by the standard deviation, which puts the
numbers roughly between −1 and +1 (see [preparing data](data-preprocessing.md)). Backpropagation works
well only when no numbers get too big. **Layer normalization** applies the same discipline inside the
encoder: after the attention layer and after the feed-forward layer, after each residual addition,
"we standardize (i.e., subtract mean and divide by std) each embedding" (slide 24).

Slide 25 shows it in two steps. First, for each embedding, compute the mean and standard deviation of
its elements and standardize it; six embeddings give six means and six standard deviations
(≈33:31). Second, "translate and rescale each embedding dimension". The lecturer describes this as two
new parameters that rescale the numbers and move them around a little, "just because adding more
weights always helps make these things better". The way their dimensions work "gets slightly
complicated", so he does not dwell on it, and the slide points to Keras's documentation of its layer
normalization layer. What comes out is "a very well-behaved set of numbers" in a narrow range
(≈33:31–34:19).

![Slide 25: six rows of four blue cells (six embeddings) enter a dashed box labelled "Layer Normalization" with two panels. In panel (1), "Calc mean and std dev for each embedding and standardize", a thin arrow runs along each row. In panel (2), "Translate and rescale each embedding dimension", thin arrows run down each of the four columns. Six blue rows come out on the right.](../raw/images/08-transformers-and-self-supervised-learning/slide-25.jpg)

*Slide 25 — layer normalization standardizes each embedding along its row, then rescales and shifts each dimension down its column. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

Is this how vanishing gradients are avoided? There are two problems, exploding and vanishing gradients,
and both are bad. Layer normalization, batch normalization and their relatives are all ways of keeping
the numbers in a small range "so it doesn't cause gradient issues later" (≈32:45).

A student asks whether the encoder could then account for biased or hateful content. It is not so much
accounting for anything as capturing whatever patterns are in the data, "the bad stuff, too": a model
trained on biased, toxic or dangerous text "doesn't have a sense of values" and will pick it up. How to
mitigate that is "a whole course unto itself" (≈34:19–35:52).

## Part 5 — The transformer encoder, complete

Put together, the encoder is a multi-head attention layer, each head tunable with its own matrices,
then a residual connection and layer normalization, then the feed-forward layer, again with a residual
connection and layer normalization (slide 26; ≈34:19–35:52). This is the actual transformer of the 2017
paper (≈5:25–6:12). Its input and output have the same dimensions, so encoders are **stackable**, "vertically
as much as you want", and GPT-3 has 96 of them (slide 27; ≈35:52). It maps exactly onto the paper's
encoder figure, which slide 28 reproduces and which OCW excludes from its licence (described in the
slide file). In the figure, input embeddings plus positional encodings enter multi-head attention, and
each "Add & Norm" box is a residual connection ("add", hence the arrow carrying the input around the
block) followed by layer normalization (≈35:52–36:38).

![Slide 27: "Transformer Encoders are stackable!" Grey vectors w1 to w6 enter a red box holding a gold "Multi-head Attention Layer" and a gold "Feed-Forward Layer", each with a purple arrow looping from its input to a plus sign at its output and a small purple upward arrow after the plus sign; blue vectors w1-hat to w6-hat leave the box. A dashed label "N ×" sits at the top right of the red box, with an arrow pointing to it from the title.](../raw/images/08-transformers-and-self-supervised-learning/slide-27.jpg)

*Slide 27 — the complete encoder block: attention and feed-forward layers, each with a residual connection and layer normalization, stacked N times. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

### What backpropagation optimizes

Slide 29 lists everything that training updates in the slot-filling model (≈36:38–37:25):

1. the positional embeddings;
2. the stand-alone embeddings (unless pretrained and `Trainable=False`);
3. the matrices $A^K$, $A^Q$ and $A^V$ for each attention head (inside the encoder);
4. the layer normalization scale and bias parameters (inside the encoder);
5. the weights in the feed-forward layers (inside the encoder);
6. the weights in the dense layers outside the encoder;
7. the weights in the final softmax layer.

Step back and "it is just a mathematical model with a lot of parameters", fitted with gradient descent
or stochastic gradient descent (≈37:25; see [training a neural network](training-neural-networks.md)).
The diagram on slide 29 is lecture 7's slot-filling pipeline, rendered on the
[lecture 7 page](07-transformers.md#part-7--slot-filling-with-a-transformer-encoder).

### Questions about the encoder

- **Do the weights depend on how long the sentence is?** No (≈37:25–38:58). Each embedding is
  multiplied by $A^K$, so $A^K$ only needs to know how long an embedding is, not how many words there
  are. That makes the number of weights independent of sentence length. It depends on the vocabulary,
  which sets how many embeddings are needed, and the length matters only for the positional embeddings:
  a 1,000-long sentence needs a 1,000-long positional embedding matrix. That is why Google's Gemini 1.5
  Pro can take a context window of a million tokens without changing its number of parameters, though
  it is still very compute-heavy.
- **Which weights are optimized first?** All at once (≈38:58–39:44). The gradient of the loss with
  respect to, say, a billion parameters is a billion-long vector, and the update
  $W_{\text{new}} = W_{\text{old}} - \alpha \times \text{gradient}$, with learning rate $\alpha$, updates
  every weight together. Backpropagation computes it from the end backwards, but when it finishes,
  everything has been updated (see [gradient descent](gradient-descent.md)).
- **Why would two heads learn different things, given the same input?** "Because the initialization is
  different." The six matrices of two heads start from different random values. If they started the
  same, they would change the same way (≈39:44–40:30).
- **Is the transformer's input the sentence or the embeddings?** The embeddings. A tokenizer turns the
  sentence into tokens, which are integers, and the embedding layer maps them to embeddings. When
  backpropagation runs, it reaches all the way back to the embedding layer and updates its weights too
  (≈40:30–41:17).
- **Can attention heads be stacked rather than run in parallel?** They are typically parallel, because
  the whole block can be stacked to get more power (≈41:17).

## Part 6 — Classifying a whole sentence: the `<CLS>` token

### Three uses of a transformer

Slide 30 names the common use cases (≈41:17–42:48). In **sequence classification** a whole sentence
gets one label, the canonical example being movie-review sentiment: "I loved the movie" → Positive. In
**sequence labeling** every word gets a label, as in lecture 7's slot filling. In **sequence
generation** the model continues the sequence, "large language models and all that good stuff", using
a "Transformer Causal Encoder" covered in [lecture 9](09-large-language-models.md). Labeling is done (slide 31); classification is
next (slide 32).

![Slide 30: three use cases side by side. "Sequence classification": "I loved the movie" into a dark-red "Transformer Encoder" box, out "Positive". "Sequence labeling": "fly from Boston to Denver" into a "Transformer Encoder", out a Token/Label table: fly O, from O, Boston B-fromloc.city_name, to O, Denver B-toloc.city_name. "Sequence generation": "I loved the movie" into a "Transformer Causal Encoder*", out "I loved the movie, especially the cinematography and the background score", the continuation in red. Footer: "*covered in Lecture 9".](../raw/images/08-transformers-and-self-supervised-learning/slide-30.png)

*Slide 30 — sequence classification, sequence labeling and sequence generation. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

### Summarizing the contextual embeddings

Whether the encoder has one block or "106 blocks, I don't care", a sentence goes in and a set of
contextual embeddings comes out, one per token (slide 33; ≈42:02–42:48). To classify the sentence,
summarize them into **a single vector**, run it through a ReLU layer and a sigmoid, "and boom, we can do
a binary classification problem" (slides 34–35; ≈42:48).

![Slide 35: "… we can feed the sentence embedding into a dense ReLU layer, followed by a sigmoid (or softmax)". The words "i", "loved", "the", "movie", each with a row of four grey cells (positional input embeddings), go into a "Transformer Encoder" box and out as four rows of blue cells (contextual embeddings); an arrow leads to a single row of four green cells, then down to "Dense with ReLU", then "Dense with sigmoid", then the output "Positive!".](../raw/images/08-transformers-and-self-supervised-learning/slide-35.jpg)

*Slide 35 — classifying a sentence: summarize the contextual embeddings into one vector, then a ReLU layer and a sigmoid. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

The simple way is to **average** the embeddings element by element (slide 37; ≈42:48–43:35). Any
shortcomings? A student suggests the word order is lost. Not necessarily, the lecturer replies, since
the positional encoding carries position into every embedding. Another suggests the biggest numbers
will dominate. True, though layer normalization has hopefully kept the numbers in a small range. The
real point is that averaging "is going to lose some richness", "mushing it down" (≈43:35–44:20).

### A token for the whole sentence

The better way is to add an **artificial token**, the class token, written `<CLS>`, at the beginning
of every sentence (slide 38; ≈44:20–45:53). It is trained along with everything else, so it ends up with
an embedding of its own. Its contextual embedding is "very much aware of all the other words in the
sentence", so it captures what is going on in the sentence as a whole. After training, take that one
embedding alone and send it through a ReLU layer and a sigmoid. In the slide's words, "The embedding
of `<CLS>` will come to represent the sentence as a whole over the course of training".

![Slide 38: the same pipeline as slide 35, but a new first input row, the CLS token in green with its grey embedding, is outlined in red above "i", "loved", "the", "movie". The encoder's first output row is green, the other four blue. A thick arrow leads from the green row alone to "Dense with ReLU", then "Dense with sigmoid", then "Positive!". Red text: "The embedding of CLS will come to represent the sentence as a whole over the course of training".](../raw/images/08-transformers-and-self-supervised-learning/slide-38.png)

*Slide 38 — the `<CLS>` token: its contextual embedding becomes the sentence's embedding, with no averaging step. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

It illustrates "a meta principle in deep learning": whenever you find yourself making an ad hoc decision,
like averaging a bunch of things, ask whether the right way can be learned from the data directly with
backpropagation (≈45:08–45:53). Why put the token at the start rather than the end? Either works.
Sentences have different lengths, short ones are padded, and the transformer ignores the padding
tokens, so a token at the end would need much more "administrative bookkeeping". At the start it is
simply easier (≈45:53–46:38). Uses include sentiment, and routing: an email arrives at a call center,
and the model decides which department to send it to (≈46:38).

## Part 7 — Self-supervised learning

### Learning from all the text out there

If the input to a task is natural language, why restrict yourself to the training data you have?
"Wouldn't it be great to learn from all the text that's out there?" (slide 39; ≈46:38–47:26). To
route call-center emails written in English, "you shouldn't have to learn English just for your call
center application. You should learn English generally and use it for other things." The way to do it
is **self-supervised learning** (slide 40).

### Why transfer learning worked

Start from [lecture 4](04-cnns-and-transfer-learning.md)'s handbags and shoes. ResNet, made headless by
chopping off its final layer, fed a small network with one hidden layer, and that network classified
handbags and shoes very accurately "with only 100 examples" (slides 41–43; ≈47:26–48:11). The diagram
is lecture 4's slide 75, shown on the [transfer learning](transfer-learning.md) page. Why was it so
effective?

Because **neural networks are representation learners** (slide 44; ≈48:11). The output of every layer
is a transformed version of the raw input, and these transformed versions are called
**representations**, "a technical term" (slide 45). A deep network trained with supervised learning
therefore learns many representations plus a final regression model, linear or logistic, since the
last layer is a sigmoid, a softmax or linear (slide 46; ≈48:59). Networks beat logistic regression
because their representations have learned useful things about the input. "They have sort of
automatically feature-engineered for you" (≈48:59–49:47). See
[representation learning](representation-learning.md).

![Slide 45: inputs x1, x2, …, xn enter the first of five tall rose-coloured bars, the layers of a network; between each pair of bars is a thick dark-red arrow circled by a red dashed ellipse, and the last bar leads to "Output". Title: "The output of every layer in a DNN can be thought of as a transformed version of the 'raw' input. These transformed versions of the input are called representations".](../raw/images/08-transformers-and-self-supervised-learning/slide-45.png)

*Slide 45 — every layer's output, each circled arrow, is a representation of the input. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

If a representation is an **encoding** of the input, the part of the network that produces it is an
**encoder**: the first layer is one, the first two layers another, the first three another, so "a DNN
'contains' many encoders" (slide 47; ≈49:47).

![Slide 47: the same input layers drawn as four rose bars joined by arrows, followed by a plus sign and a separate bar labelled "Regression" in its own dashed box. Three nested dashed outlines start at the left edge: a small red one around the first bar, a light-blue one around the first two, and a green one around the first three, with the label "Encoder" below. Title: "If we think of a representation as an encoding of the raw input, the part of the NN that produces that encoding can be viewed as an encoder. A DNN 'contains' many encoders."](../raw/images/08-transformers-and-self-supervised-learning/slide-47.png)

*Slide 47 — a deep network as nested encoders plus a final regression model. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

### General knowledge, not specific knowledge

Do these representations capture **specific** knowledge, what is needed to connect the input to the
particular output the network was trained on, or **general** knowledge about the input that could
predict other outputs (slide 48)? It turns out they capture a lot of general knowledge, which is why
they can be reused for unrelated problems (slide 49; ≈49:47–50:34). In a network trained to classify
everyday objects into 1,000 categories, the first three layers' representations correspond to lines,
then edges, then more complex shapes (slide 50, from Zeiler and Fergus). In one trained to detect faces
they go from lines to edges to circles and finally faces (slide 51, from Lee et al., 2009). Both
figures are excluded from OCW's licence and described in the slide file. So a face-detection network
could plausibly be reused to build an emotion-detection network (slide 53; ≈50:34).

If you can somehow get an encoder that produces good representations of your input, a regression model
on top of them will do, and since you no longer "spend" precious data on learning the
representations, you need far less labeled data (slides 54–55; ≈50:34–51:19). That is what happened
with headless ResNet: "your training data is everything that the pre-trained model was trained on plus
your little 200 examples" (slides 56–57; ≈51:19).

### The general recipe

Find a deep network built on **similar inputs but different outputs** (slide 58; ≈51:19–52:06). What
comes out of its last layer before the output layer, or the one before that, is likely to be an
excellent representation of the input (slide 59). **Chop off** the output layer and use the resulting
headless model as an encoder (slide 60). Attach a **new output layer** and train with the labels you
actually care about (slide 61), either keeping the encoder fixed and learning only the new layer's
weights (slide 62) or **fine-tuning** all the layers (slide 63). "This is like the playbook we followed
for ResNet", and it works for other kinds of data too (≈52:06). See
[transfer learning](transfer-learning.md).

![Slide 60: "So we 'chop off' the output layer and use the resulting 'headless model' as an encoder". A red dashed box encloses "Input" and the first four dark-red layer bars, with three dots between the third and fourth; a green arrow leaves the box toward a fifth bar, the old output layer, which is struck through by a large dark "X".](../raw/images/08-transformers-and-self-supervised-learning/slide-60.png)

*Slide 60 — chop off the output layer, and what remains is an encoder. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

![Slide 63: "… or fine-tune all the layers". A red dashed box encloses the whole network: "Input", five light-green layer bars (the encoder's four and the new one) joined by arrows, and the label "New Output".](../raw/images/08-transformers-and-self-supervised-learning/slide-63.png)

*Slide 63 — with a new output layer attached, every layer, now drawn green, is trained: fine-tuning. Slide 62 keeps the encoder fixed and trains only the new layer. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

### Where do the labels come from?

Building such a generally useful model takes **labeled** data (slide 64; ≈52:06). ResNet was lucky:
ImageNet has, in the lecturer's words, about a million images, each labeled with one of 1,000
categories. For text, collecting the data is "no problem": scrape the internet, or download Wikipedia
(slide 65; ≈52:06–52:51). The problem is the second requirement, an output label for every piece of
text (slide 66). For an input sentence, what should the output label be (slide 67)?

### Masking: fill in the blanks

**Self-supervised learning** builds pretrained models without labeled data. Its key idea: "Predict a
subset of the input data using the rest of the input" (slide 68; ≈52:51). The most common technique,
one of many, is **masking** (≈52:51–53:37). Take an input, remove a small part of it, and the incomplete
input becomes the new input while the removed part becomes a "fake" label (slide 69). "If you're baking
donuts, you make a donut, and then you punch a hole in the middle of the donut. The donut with the hole
is your new input. The munchkin is the label" (≈53:37). Then train a deep network to predict the fake
labels from the masked inputs, "to fill in the blanks" (slide 70).

![Slide 69: "Masking: A Self-supervised Learning Technique". "1. We modify the original input data to create 'fake' (input, label) pairs by masking a part of the input and making it the label." Three rows: in each, a long blue bar ("Original input") points to a "Masked Input", the same bar with a white gap cut out, and a "'Fake' Label", an empty red-outlined bar holding only the blue piece that was removed. The gap is in the middle, near the left and near the right in the three rows.](../raw/images/08-transformers-and-self-supervised-learning/slide-69.png)

*Slide 69 — masking makes an (input, label) pair out of unlabeled data: the input with a hole, and the piece that was removed. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

For text, slide 71 masks words of the MIT Sloan School's mission statement, "The mission of the MIT
Sloan School of Management is to develop principled, innovative leaders who improve the world and to
generate ideas that advance management practice". The modified input replaces "mission",
"Management", "leaders", "generate" and "management" with `<MASK>`, and those five words are the fake
labels (≈54:25).

![Slide 71: "Masking Example". Top: the original input, MIT Sloan's mission statement. Left: the modified input, with five words replaced by MASK. An arrow leads into a box labelled "DNN" holding a row of dark-red layer bars, and from it to the "'Fake' Labels": the full sentence in grey with mission, Management, leaders, generate and management in blue and underlined. Dashed curved arrows run from the original input to the modified input and to the labels.](../raw/images/08-transformers-and-self-supervised-learning/slide-71.png)

*Slide 71 — masking text: the masked words become the labels the network learns to fill in. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

"Now for the amazing part." In learning to fill in the blanks, the network learns a good
**representation** of the input data (slide 72; ≈54:25). It makes sense: to fill in the blanks
successfully, the model has to learn how the variables are related to each other (slide 73). "If I say
the capital of France is dash, and you're like, Paris, OK, how did you know that?" Filling in the blanks
forces it to learn the connections between words (≈55:11). Once the model is built, **extract an
encoder** from it (slide 74), and fine-tune it as in transfer learning (slide 75). This is how to build
a generic pretrained model from unlabeled data (≈55:11).

![Slide 74: "Once a self-supervised model is built, we can extract an encoder from it …". A box labelled "Encoder" holds a row of dark-red layer bars, with one more bar just outside it to the right. "Input": the mission statement with five blanks. "Output": the full sentence with the five missing words in blue.](../raw/images/08-transformers-and-self-supervised-learning/slide-74.png)

*Slide 74 — the model trained to fill in blanks contains an encoder that can be reused. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

### With a transformer, masking is sequence labeling

A transformer encoder fits naturally, because it returns one output for every token that goes in
(slide 76; ≈55:11–55:57). "Masked self-supervised learning is just a sequence labeling problem"
(slide 77). Send in the sentence with its `<MASK>` tokens, through positional input embeddings, the
encoder, a dense ReLU layer and a softmax per token, exactly as in slot filling. The outputs at the
unmasked positions do not matter. Where a `<MASK>` went in, the right answer is the hidden word,
"mission" for instance. Those right answers make the loss function, backpropagation does the rest, and
"boom, you're done" (≈55:57).

![Slide 77: "Masked Self-Supervised Learning is just a sequence labeling problem". Tokens "the", MASK, "of", "…", MASK, "practice", each with a row of grey cells (positional input embeddings), go into a "TE" box, out as blue contextual embeddings, into "Dense (ReLU)", then to six rows of yellow cells, each with its own "SM" (softmax) box. Only the two masked rows have outputs: "mission" and "managem…" (cut off at the edge). Caption: "The DNN learns to predict the masked words from the rest of the sentence".](../raw/images/08-transformers-and-self-supervised-learning/slide-77.jpg)

*Slide 77 — masked self-supervised learning as sequence labeling: only the masked positions have labels. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

## Part 8 — BERT

Pretrain a transformer this way on massive amounts of English text and you get **BERT** (slides 78–79;
≈56:44). Google first used it to upgrade search in 2019, and lecture 7's Brazil visa query uses it
under the hood (see [lecture 7](07-transformers.md)). The BERT paper "will actually make sense to you
now". Slide 80 shows its model-architecture section: "BERT's model architecture is a multi-layer
bidirectional Transformer encoder". The paper writes $L$ for the number of layers (transformer
blocks), $H$ for the hidden size and $A$ for the number of self-attention heads, and reports two sizes:

| Model | $L$ | $H$ | $A$ | Total parameters |
| --- | --- | --- | --- | --- |
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

The embeddings are 768 long, and the feed-forward layer is four times as wide (≈56:44–57:36). The
paper's footnote on slide 80 sets that size to $4H$, "3072 for the $H = 768$ and 4096 for the
$H = 1024$". The lecturer says "4,096" for the 768-long model, which is BERT-Large's figure.
**Bidirectional** means each word can pay attention to every other word in the sentence. On Monday, in
[lecture 9](09-large-language-models.md#part-3--causal-self-attention), comes the causal transformer, in which a word attends only to the words before it (≈57:36).
The same slide quotes the paper on GPT, whose transformer "uses constrained self-attention where every
token can only attend to context to its left".

Conveniently, BERT was trained with the `<CLS>` token, so it can be used for sequence classification
"out of the box", without surgery (slide 82; ≈58:22). Slide 82's footnote points to the "HW2 colab". It
is a good starting point for sequence labeling too (slide 83). Many variations and improvements have
appeared since, and the Sentence Transformers library (https://www.sbert.net/index.html) collects a lot
of BERT-related code and resources (slide 84; ≈58:22–59:07).

So for any sequence classification or labeling problem with natural-language input, use a model like
BERT as a pretrained encoder: "Label 'a few hundred' examples, attach the right final layers to BERT and
fine-tune" (slide 85; ≈59:07). If the problem is a "standard" NLP problem, even that may be
unnecessary, because hubs hold pretrained models for the standard tasks that can be used without any
fine-tuning at all. The biggest is the **Hugging Face Hub** (slide 86). The slide says "Over 500,000
pretrained models available!! (as of Feb 27, 2024)", and its screenshot shows 525,416. The lecturer
counted 520,000 the night before and recalls perhaps 50,000 when he taught the course a year earlier
(≈59:53).

## Part 9 — The Hugging Face Colab

The notebook is not on OCW; this follows the narration (≈59:53–1:06:53).

**The Hub.** Hugging Face hosts pretrained models, data sets, demos people have built in its "Spaces",
and documentation. Its models are organized by task, computer vision and natural language among them.
Pick text classification and "it tells you all the models that are available", about 50,000 for that
task alone, and you can sort them by downloads or likes and take one as a starting point
(≈59:53–1:01:27).

**The rule.** When the input is natural-language text, the first question is whether the task is a
standard one. "If it's a standard task, do not reinvent the wheel" (≈1:01:27). The demonstration uses
Hugging Face's **Transformers library**, which "as of this year" comes pre-installed in Colab, and in
particular its **`pipeline` function**, which gives the ability to "out-of-the-box start using it
without any pre-training". The example text is a complaint addressed to Amazon: Bumblebee ordered an
Optimus Prime action figure from the store in Germany, was sent Megatron instead, and as "a lifelong
enemy of the Decepticons" demands an exchange (≈1:01:27–1:02:13).

**Sentiment.** `pipeline` with the task text classification creates a classifier. The first call
downloads the model. Run the text through it "as if it's just a little function", and the sentiment
comes back negative with a 90% probability (≈1:02:13–1:02:59). On the lecturer's test sentences:

- "I hated the movie" is negative.
- "If I said I loved the movie, I would be lying" is negative.
- "The movie left me speechless" is negative, though "a good classifier would probably have given you
  a probability around the 50% mark because it's right on the fence".
- "Incredible" is positive.
- "Almost but not quite entirely unlike anything good I've seen", inspired by Douglas Adams's
  "almost but not quite entirely unlike tea", fools it into positive (≈1:02:59–1:03:46).

**Named entities.** A pipeline for `ner`, named entity extraction, pulls company names, people's names
and the like out of text, a job people once hand-built "painstakingly" complex systems for; "now it's
just a pipeline away" (≈1:04:31). On the letter it tags Amazon as an organization (ORG), Germany as a
location (LOC) and Bumblebee as a person. The Transformer names fool it, and Optimus Prime and "Decept"
come out as miscellaneous. Entities extracted from, say, Reuters news stories can be put into a
structured table and fed to traditional machine learning (≈1:05:20).

**Question answering.** Asked "What does the customer want?" about the letter, the pipeline answers "an
exchange of Megatron" and gives where the answer starts and ends in the text (≈1:05:20–1:06:06). That
matters beyond the demo. Ask a large language model to say where in its input it found an answer, and
since "these things are going to hallucinate", you can check the passage it claims to use, "a way to
essentially do QA on LLM output" (≈1:06:06–1:06:53).

## Part 10 — Transformers beyond text

### Same block, different tokenizer

Transformers work across speech recognition, computer vision and many other domains (slide 88). The
architecture of the transformer block can be used as-is: "no surgery needed, no clever thinking
required". What varies from application to application is how the inputs are tokenized and encoded
into embeddings that can be fed to it. "That's the game" (slide 89; ≈1:06:53–1:07:40).

### The Vision Transformer

The **Vision Transformer**, perhaps the first transformer applied to vision, tokenizes a picture by
**chopping it into small squares** (slide 90, the paper's figure, excluded from OCW's licence;
≈1:07:40–1:10:45). In the example, one picture becomes nine patches. Each patch is just three tables
of numbers, its RGB values. Flatten them into one long vector and run it through a dense layer to get a
smaller vector, the patch's embedding. "Linear projection", in the figure, "is a synonym for, run it
through a dense layer", whose weights are learned. Order matters in a picture: a patch at the top left
is not one in the middle, "otherwise, every jumbled version is going to be the same thing". So there are
nine **position embeddings**, for positions 0 to 8, added to the patch embeddings "just like we did with
words". For classification, add a `<CLS>` token, run everything through the transformer, and put a
softmax on the `<CLS>` output: bird, ball, car. "This simple approach actually works, amazingly enough."

Asked what plays the role of a vocabulary for images, the lecturer answers that there is none
(≈1:11:31). Any image gives nine patches, each patch goes through the dense layer, and out come nine
embeddings for "the transformer meat grinder".

### The TabTransformer

For tabular, structured data, gradient boosting such as XGBoost works "really, really well", and the
lecturer sees no great edge for transformers there, so try both (≈1:12:20). But transformers can handle
tables too, and the **TabTransformer** was one of the first (slide 91). Categorical features, gender for
example, are "just text", so each gets an embedding, and those embeddings go through a transformer block
"exactly the block we have seen before" (≈1:12:20–1:13:54). The continuous features, cholesterol or
blood pressure in the heart-disease example, are collected into a vector. The contextual embeddings
coming out of the transformer are concatenated with the continuous features and run through one or more
dense layers to the output. In the paper's figure the continuous features pass through a layer
normalization first. The point to remember is that categorical variables are easily represented as
embeddings (≈1:13:54).

![Slide 91: Figure 1 of the TabTransformer paper, drawn bottom to top. "Categorical Features (x1, x2, …, xm)" go into "Column Embedding", then into a block marked "Transformer × N" containing Multi-Head Attention, Add & Norm, Feed Forward and Add & Norm, with skip arrows. "Continuous Features" go into "Layer Normalization" and straight up past the transformer. Both branches meet in "Concatenation", then "Multi-Layer Perceptron", then "Loss". "Figure license: CC0 1.0."](../raw/images/08-transformers-and-self-supervised-learning/slide-91.jpg)

*Slide 91 — the TabTransformer: categorical columns become embeddings for a transformer, continuous columns join afterwards. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

### Embeddings as a common language

Once the input is in the "common language" of embeddings, the encoder block needs no change: "You give
me embeddings, I give you great contextual embeddings out, and nobody gets hurt" (slide 92;
≈1:13:54–1:14:40). That makes multi-modal data easy (≈1:14:40–1:16:13). Slide 93's example classifies a
product from a review, a photograph of a desk chair and a small table of attributes (age 37 months,
supplier Acme, 13 parts). The text gets language embeddings, the image image embeddings as in the
Vision Transformer, the table tabular embeddings as in the TabTransformer. A `<CLS>` token goes on top,
everything runs through a stack of transformer blocks, and the contextual `<CLS>` embedding goes through
a ReLU layer and a sigmoid or softmax to the predicted label. The slide's photo is excluded from OCW's
licence and is described in the slide file. This is why Google's Gemini 1.5 Pro and GPT-4 Vision can
take an image and a question and answer it. Every modality is cast into embeddings, after which the
transformer "will decide, for example, that this word in your question actually is highly related to
that patch in the picture" (≈1:15:26).

### Not presented: contrastive learning

The deck ends with a section marked "Contrastive Learning (time permitting)" (slides 94–98), which the
recording does not reach; what follows is from the slides alone. Masking creates artificial labels for
unlabeled text. "How can we pretrain models on unlabeled image data?" (slide 95). For images,
**contrastive learning** "has been found to be very effective" (slide 96, citing "A Simple Framework
for Contrastive Learning of Visual Representations" by Chen et al., 2020). For every original image,
construct a pair of "augmented" images, and train the network to "maximize agreement": make the
representations of each augmented pair close to each other but far from the representations of the
other pairs. Slide 97 shows the paper's augmentations of one dog photograph: crop and resize, flip,
colour distortion, rotation, cutout, Gaussian noise, Gaussian blur and Sobel filtering (compare
[data augmentation](data-augmentation.md)). Once the model is built, extract its encoder and fine-tune
it (slide 98). The photographs on slides 96–98 are excluded from OCW's licence.

## Printed slips

- Slide 70 reads "We then use train a Deep Neural Network".
- Slide 72 prints a doubled closing quotation mark after "fill in the blanks".
- Slide 92 reads "This turns out to very useful for multi-modal data".
- Slide 91's title reads "The Tab Transformer: : A Transformer for Tabular Data".
- Slide 51's citation reads "Lee at al (2009)".
- Slide 81 carries an OCW notice for a "BERT figure", but the page shows the course's own `<CLS>`
  diagram from slide 38 under a new title.
- Slide 29 cuts the label "B-toloc.city_name" off at the right edge, and slides 77–78 cut the word
  "management" to "managem…".
- Spoken: "4,096" for the feed-forward size of BERT with 768-long embeddings, where slide 80's
  footnote gives 3072 (see Part 8); "one of 125 possibilities" per word, where lecture 7's slides
  say 123 slots and its Colab's output vocabulary has 125 entries (Part 1); and the handbags-and-shoes
  data as "100 examples" (≈48:11) and "your little 200 examples" (≈51:19), where lecture 4 used about
  100 images per class.

## Related pages

- [Transformers](transformers.md): self-attention, tunable attention and the paper's formula, multi-head
  attention, residual connections, layer normalization, the encoder, the `<CLS>` token, BERT and
  transformers beyond text, gathered as a topic.
- [Self-supervised learning](self-supervised-learning.md): masking, BERT, and contrastive learning as
  the image counterpart.
- [Transfer learning](transfer-learning.md): headless ResNet, the general recipe, and pretrained
  language models from the Hugging Face Hub.
- [Representation learning](representation-learning.md): networks as representation learners, and
  encoders that capture general knowledge.
- [Natural language processing](natural-language-processing.md): sequence classification, labeling and
  generation, and the course's NLP roadmap.
- [Preparing data for a network](data-preprocessing.md): standardization, which layer normalization
  applies inside the encoder.
- [Images and computer-vision tasks](images-and-vision-tasks.md): the Vision Transformer's patches.
- [Lecture 9](09-large-language-models.md): the causal transformer and next word prediction, which follow.
- [Lecture 7](07-transformers.md): the first half of the transformer, and the slot-filling Colab this
  lecture trains.
- [Lecture 4](04-cnns-and-transfer-learning.md): ResNet and the handbags-and-shoes classifier.
