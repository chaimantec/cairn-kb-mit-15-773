# Transformers

A **transformer** takes a sequence of vectors, one per word (or token), and returns the same number of
vectors of the same length, each **transformed** so that it takes the rest of the sentence into
account. "If 10 things come in, 10 things go out", which is where the name comes from (lecture 7,
≈13:10–13:55). The course builds its encoder half in [lecture 7](07-transformers.md) from three
requirements: an output as long as the input, the **context** of each word, and the **order** of the
words (lecture 7, slides 14–19). The architecture comes from the 2017 paper "Attention Is All You
Need" (https://arxiv.org/abs/1706.03762), and transformers now run search, translation, speech,
vision, generative AI and special-purpose systems such as AlphaFold (lecture 7, slide 2; ≈0:00–1:33).

The course builds the encoder over two lectures. [Lecture 7](07-transformers.md) builds
self-attention, multi-head attention, the feed-forward layers and positional encoding.
[Lecture 8](08-transformers-and-self-supervised-learning.md) completes it with the three elements
lecture 7 left out: the matrices that make self-attention tunable, residual connections and layer
normalization (lecture 8, slide 12). Then it turns to using encoders: the `<CLS>` token for classifying
a sentence, BERT, and transformers for images, tables and multi-modal data.
[Lecture 9](09-large-language-models.md) makes the
attention causal, which turns the encoder into the decoder behind GPT.

## Self-attention

Start from **stand-alone embeddings** $w_1, \dots, w_6$ of the words of "The train slowly left the
station", such as GloVe vectors or random ones (see [word embeddings](word-embeddings.md)). To make
the embedding of "station" contextual, take a **weighted average** of the embeddings of all the words
in the sentence, itself included, with weights $s_{i,6}$ that say how much attention "station" pays to
word $i$ (lecture 7, slides 26–36; ≈16:15–24:45):

$$\hat w_6 = s_{1,6} w_1 + s_{2,6} w_2 + s_{3,6} w_3 + s_{4,6} w_4 + s_{5,6} w_5 + s_{6,6} w_6$$

The weight of a word should reflect how related it is to "station", and relatedness is measured by the
**dot product** of the two stand-alone embeddings, which for vectors of length about 1 is the cosine of
the angle between them: near 1 for related words, 0 for unrelated ones, −1 for opposite ones (slides
37–40; ≈27:09–31:00). Dot products can be negative, so a **softmax** turns them into proper weights,
non-negative and adding up to 1 (slides 41–43; ≈31:00–33:17):

$$s_{1,6} = \frac{\exp\langle w_1, w_6 \rangle}{\sum_{i=1}^{6} \exp\langle w_i, w_6 \rangle}$$

where $\langle w_i, w_6 \rangle$ is the dot product of the $i$-th word's embedding with that of
"station". The effect is to pull each word's embedding towards the words of the sentence it is most
related to: "train" pulls "station" towards trains, platforms and departures, while "radio", related
but absent, gets zero weight (slides 44–47). Every word is treated the same way, so $w_1, \dots, w_6$
become contextual embeddings $\hat w_1, \dots, \hat w_6$, which replace them (≈27:09). This is
**self-attention** (slide 48).

![Slide 43 of lecture 7: the words of "The train slowly left the station" with w1 to w6 beneath them; six lines labelled s1,6 to s6,6 converge on w6-hat. w6 is circled and labelled "Stand-alone embedding", w6-hat circled and labelled "Contextual embedding". Below are the softmax formula for s1,6 and the boxed weighted sum for w6-hat.](../raw/images/07-transformers/slide-43.png)

*Lecture 7, slide 43 — self-attention for one word: softmax weights from dot products, then a weighted sum. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec07.pdf)*

As lecture 7 presents it, the self-attention box has **no parameters**: nothing in it is learned
(≈40:58–41:44). Lecture 8 answers how attention learns anything, below.

## Self-attention as matrix multiplication

Lecture 8 redraws the same computation for a whole sentence at once, on an iPad
([lecture 8](08-transformers-and-self-supervised-learning.md#part-2--self-attention-as-matrix-multiplication),
≈7:46–13:17). Put the sentence's embeddings in a matrix $X$, one embedding per row, and make three
copies of it. The dot product of every pair of words is a cell of the single product $XX^T$. A softmax
on each row turns the dot products into weights that add up to 1, and multiplying that table by the
third copy of $X$ takes the weighted averages:

$$\text{contextual embeddings} = \text{softmax}(XX^T)\thinspace X$$

Without this packaging into matrix operations, which run "lightning fast on GPUs", "there wouldn't be
any transformer revolution" (≈7:46, ≈12:30–13:17).

## Making attention tunable: keys, queries and values

To give self-attention something to learn, lecture 8 multiplies each copy of $X$ by its own matrix of
weights, learned by backpropagation: the **key matrix** $A^K$, the **query matrix** $A^Q$ and the
**value matrix** $A^V$, giving $K = XA^K$, $Q = XA^Q$ and $V = XA^V$ (≈14:50–18:40; slides 16–20). The
dot products become $QK^T$, and the operation becomes the formula of "Attention Is All You Need",
which slide 20 prints:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The lecture sets aside the division by $\sqrt{d_k}$, "a particular number" that keeps any value going
into the softmax from getting too big and squashing the rest to 0 (≈23:19). Word by word, as the slides
write it, the weight that "station" ($w_6$) gives word $i$ is a softmax of
$\langle A^K w_i, A^Q w_6 \rangle$, and $\hat w_6$ is the weighted sum of the $A^V w_i$ (slides 18–20).
If a transformation does not help, training can make its matrix the identity, so in the worst case
the layer simply has "nothing to learn here" (≈15:36). The names come from information retrieval, an
interpretation the lecturer does not find helpful outside that field (≈16:22). Most implementations
keep the key and query matrices square, the size of the embeddings; the value matrix is often smaller
when several heads' values are concatenated back to the original size (≈25:42–26:30).

## Multi-head attention

A sentence carries several patterns at once, such as tense, tone and the relationships between
entities, so the encoder runs several self-attention blocks, **attention heads**, side by side, "exactly
like having lots of filters in a convolutional network" (lecture 7, slides 49–50; ≈36:23–37:55; see
[convolutional neural networks](convolutional-neural-networks.md)). No head is told what to learn.
In the original paper's example, one head makes "perfect" draw on "law", another makes it draw on
itself (slide 51). Each head gives every word its own contextual vector; the vectors are
**concatenated**, which multiplies their length by the number of heads, and **projected** back to the
original length by a single dense layer (≈41:44–44:47). Heads only differ because each has **its own**
$A^K$, $A^Q$ and $A^V$, started from different random values: without those matrices two heads given
the same input would give the same output (lecture 8, slides 14 and 21; ≈24:54–28:04, ≈39:44–40:30).
Ten heads mean 30 matrices to learn.

## Feed-forward layers

Self-attention and the projection are linear, so the encoder ends with **feed-forward layers**: a ReLU
layer, as a rule of thumb about four times as wide as the vectors, and a dense layer projecting back to
the original length (lecture 7, slide 52; ≈44:47–46:20). The design "felt very ad hoc" when it
appeared, but it has proved "pretty difficult to beat" (≈46:20).

## Positional encoding

Dot products work on sets, not sequences: scramble the words and the contextual embeddings come out the
same (lecture 7, slides 54–56; ≈47:51–48:38). For problems where order does not matter, such as tabular
data, that is fine. For language, **positional encoding** fixes it: treat each position in the sentence,
0 up to the maximum length minus 1, as a categorical variable with its own learned **position
embedding**, and add it to the word's stand-alone embedding (slide 57; ≈49:25–50:58). In lecture 7's
example, "cat" in position 0 of "cat sat mat" goes in as $(0.5, 7.1) + (1.3, 3.9) = (1.8, 11.0)$, and the
same word in position 2 would get position 2's embedding instead (slides 58–61; ≈50:58–55:35). The
word table and the position table are independent and both learned by backpropagation. The lecture
calls this the simplest of several schemes.

## The transformer encoder

Positional input embeddings, multi-head attention, then feed-forward layers: this is a **transformer
encoder** (lecture 7, slides 62–64). The paper's version adds an "Add & Norm" step after the attention
and after the feed-forward layers, which lecture 8 supplies.

**Residual connections** add each layer's input to its output, element by element, which is possible
because the two are the same length (lecture 8, slide 23; ≈28:52–31:12). They are "an insurance
policy" in case a layer's transformation is not great, and they "improve the gradient flow during
backpropagation dramatically". ResNet, the Residual Net, was the first network to use them.
**Layer normalization** then standardizes each embedding, subtracting the mean of its elements and
dividing by their standard deviation, and rescales and shifts each dimension with learned parameters, so
that the numbers stay in the narrow range backpropagation needs (slides 24–25; ≈31:12–34:19). With
these, the block is the encoder of the 2017 paper.

![Slide 27 of lecture 8: grey vectors w1 to w6 enter a red box holding a "Multi-head Attention Layer" and a "Feed-Forward Layer", each with a purple arrow looping from its input to a plus sign at its output; blue vectors w1-hat to w6-hat leave the box, which is labelled "N ×".](../raw/images/08-transformers-and-self-supervised-learning/slide-27.jpg)

*Lecture 8, slide 27 — the complete encoder block with residual connections and layer normalization, stacked N times. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec08.pdf)*

![Slide 53 of lecture 7: two "Self-Attention" boxes, each taking w1 to w6 and giving w1-hat to w6-hat, feed a small "Concatenate and Project" box, then a larger "Feed-Forward Layers" box that gives the final w1-hat to w6-hat. A tall arrow on the left runs from "Start with random embeddings" to "End with contextual embeddings".](../raw/images/07-transformers/slide-53.jpg)

*Lecture 7, slide 53 — the encoder before positional encoding: attention heads, concatenate and project, feed-forward layers. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec07.pdf)*

Because an encoder's output has the same number of vectors, of the same length, as its input, encoders
can be **stacked** "like pancakes": "it's the perfect API" (slide 69; ≈1:00:15). GPT-3, the lecturer
says, has 96 of them. Going wide, with more heads, learns more patterns from the same input; going
deep, with more encoders, works at higher levels of abstraction, much as a convolutional network's later
layers combine edges into faces (≈1:01:01–1:02:34). Keeping one output per input also "preserves your
optionality": nothing is lost on the way up, and the output can be aggregated at the end if the task
needs one answer (≈35:38).

## Using an encoder: slot filling

For lecture 7's task, labelling every word of an ATIS travel query with one of 123 slots, the encoder's
contextual embeddings each go through a dense ReLU layer and a softmax of their own, so the network
makes one classification per word (lecture 7, slides 72–76; ≈1:08:01–1:11:06). Every layer, the
embeddings included, is trained by backpropagation. In the lecture's Colab the encoder has
512-dimensional embeddings, five heads and a 64-wide feed-forward layer, about 5.3 million parameters
(see [lecture 7](07-transformers.md), Part 8, and [Keras](keras.md)). Trained for 10 epochs, the model
reaches 99% accuracy on the test set and 93% on the words whose slot is not O, "other", which is what
matters, since most words are O (lecture 8, ≈3:53–5:25).

## Classifying a sentence: the `<CLS>` token

Transformers serve three common uses: **sequence classification**, one label per sentence;
**sequence labeling**, one label per word, as in slot filling; and **sequence generation**, continuing a
sequence, which needs the causal encoder of [lecture 9](09-large-language-models.md) (lecture 8, slide 30; see
[below](#causal-self-attention-and-the-decoder)). To classify a sentence,
the encoder's contextual embeddings must become one vector. Averaging them works but "is going to lose
some richness" (slide 37; ≈43:35–44:20). The better way adds an artificial **class token**, `<CLS>`, at
the start of every sentence. Trained with everything else, its contextual embedding comes to represent
the sentence as a whole, and it alone goes through a ReLU layer and a sigmoid (slide 38; ≈44:20–45:53).
It sits at the start rather than the end because padded sentences would make the end harder to find
(≈45:53–46:38).

## BERT

Pretrained by [self-supervised learning](self-supervised-learning.md) on massive amounts of English
text, filling in masked words, a transformer encoder becomes **BERT**, "a multi-layer bidirectional
Transformer encoder" in its paper's words (lecture 8, slides 77–80; ≈55:57–57:36). BERT-Base has 12
blocks, 768-long embeddings and 12 heads; BERT-Large 24 blocks, 1024 and 16. It was trained with a
`<CLS>` token, so it classifies sentences "out of the box", and it suits sequence labeling too
(slides 82–83).

## Causal self-attention and the decoder

For **next word prediction**, the self-supervised task behind GPT, the encoder cannot be used as it is
([lecture 9](09-large-language-models.md), slides 12–16; ≈7:03–9:24). Self-attention computes each word's contextual
embedding from all the words in the sentence, so a model asked to predict "sat" from "the cat" can see
"sat" in its input and copy it. **Causal self-attention**, also called **masked self-attention**, gives
zero weight to "future" words. In the table of attention weights every cell above the diagonal is zeroed,
and each row is renormalized to add up to 1 (slides 17–20; ≈11:47–14:05). Making every attention head
causal gives the **Transformer Causal Encoder**. It is also called the **Transformer Decoder**, after the
decoder half of the original paper's encoder-decoder, built for translation, which uses masked attention
(slides 26–29; ≈14:52–17:12). "Bidirectional", as in BERT, means each word can see every other word; the
causal encoder looks only backward (≈16:26). The bidirectional encoder serves masked prediction and
BERT, the causal one next word prediction and GPT (≈17:12–20:19). See
[large language models](large-language-models.md).

![Slide 26 of lecture 9: a red box holding two khaki "Causal Attention Head" boxes whose arrows meet in "Concatenate and Project", which feeds a tall "Feed-Forward Layer"; grey positional input embeddings w1 to w6 enter on the left and blue contextual embeddings ŵ1 to ŵ6 leave on the right. Title: "By replacing Self-Attention with Causal Self-Attention in the Transformer block, we get a Transformer Causal Encoder".](../raw/images/09-large-language-models/slide-26.png)

*Lecture 9, slide 26 — a transformer block with causal attention heads is a Transformer Causal Encoder. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec09.pdf)*

## Beyond text

The transformer block can be used as-is for other kinds of data. "What tends to vary from application
to application is how the inputs are encoded/tokenized" (lecture 8, slide 89; ≈1:06:53–1:07:40). The
**Vision Transformer** chops an image into patches, flattens each and projects it with a dense layer
into an embedding, adds a position embedding per patch and a `<CLS>` token, and classifies from the
`<CLS>` output (slide 90; ≈1:07:40–1:11:31). The **TabTransformer** embeds a table's categorical
features, runs them through transformer blocks, and concatenates the result with the continuous
features (slide 91; ≈1:12:20–1:13:54). Once every input is in the "common language" of embeddings, one
stack handles text, images and tables together, which is how multi-modal models take an image and a
question (slides 92–93; ≈1:13:54–1:16:13). See
[lecture 8, Part 10](08-transformers-and-self-supervised-learning.md#part-10--transformers-beyond-text).

## Where this comes up

- [Lecture 7](07-transformers.md): the whole construction, the ATIS slot-filling task and the Colab
  set-up.
- [Lecture 8](08-transformers-and-self-supervised-learning.md): self-attention as matrices, keys,
  queries and values, residual connections, layer normalization, the `<CLS>` token, BERT, and the Vision
  Transformer and TabTransformer.
- [Lecture 6](06-word-embeddings.md): stand-alone embeddings, and why contextual ones need transformers.
- [Lecture 9](09-large-language-models.md): causal self-attention, the Transformer Causal Encoder (the decoder), and GPT.
- See also [large language models](large-language-models.md),
  [self-supervised learning](self-supervised-learning.md),
  [natural language processing](natural-language-processing.md),
  [word embeddings](word-embeddings.md) and
  [multi-class classification and softmax](multiclass-classification.md).
