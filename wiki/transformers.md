# Transformers

A **transformer** takes a sequence of vectors, one per word (or token), and returns the same number of
vectors of the same length, each **transformed** so that it takes the rest of the sentence into
account. "If 10 things come in, 10 things go out", which is where the name comes from (lecture 7,
≈13:10–13:55). The course builds its encoder half in [lecture 7](07-transformers.md) from three
requirements: an output as long as the input, the **context** of each word, and the **order** of the
words (lecture 7, slides 14–19). The architecture comes from the 2017 paper "Attention Is All You
Need" (https://arxiv.org/abs/1706.03762), and transformers now run search, translation, speech,
vision, generative AI and special-purpose systems such as AlphaFold (lecture 7, slide 2; ≈0:00–1:33).

This knowledge base covers lecture 7 only. The lecture leaves three parts of the encoder for lecture 8,
which is not in the knowledge base yet: the linear projections that give self-attention its learned
weights, residual connections, and layer normalization (lecture 7, slide 70).

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
(≈40:58–41:44). How attention learns anything is the question lecture 8 answers, with projections of
the incoming embeddings "into three different spaces" (slide 70).

## Multi-head attention

A sentence carries several patterns at once, such as tense, tone and the relationships between
entities, so the encoder runs several self-attention blocks, **attention heads**, side by side, "exactly
like having lots of filters in a convolutional network" (lecture 7, slides 49–50; ≈36:23–37:55; see
[convolutional neural networks](convolutional-neural-networks.md)). No head is told what to learn.
In the original paper's example, one head makes "perfect" draw on "law", another makes it draw on
itself (slide 51). Each head gives every word its own contextual vector; the vectors are
**concatenated**, which multiplies their length by the number of heads, and **projected** back to the
original length by a single dense layer (≈41:44–44:47).

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
and after the feed-forward layers, residual connections and layer normalization, which lecture 7 leaves
for lecture 8 (slide 64's footnote).

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
512-dimensional embeddings, five heads and a 64-wide feed-forward layer, about 5.3 million parameters;
the model is not trained in that recording (see [lecture 7](07-transformers.md), Part 8, and
[Keras](keras.md)).

## Where this comes up

- [Lecture 7](07-transformers.md): the whole construction, the ATIS slot-filling task and the Colab
  set-up.
- [Lecture 6](06-word-embeddings.md): stand-alone embeddings, and why contextual ones need transformers.
- See also [natural language processing](natural-language-processing.md),
  [word embeddings](word-embeddings.md) and
  [multi-class classification and softmax](multiclass-classification.md).
