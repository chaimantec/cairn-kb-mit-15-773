# Lecture 7 — Transformers

**Lecturer:** Rama Ramakrishnan ·
**Video:** [youtube.com/watch?v=IeF7aATDaw4](https://www.youtube.com/watch?v=IeF7aATDaw4) (76 min) ·
**Slides:**
[`mit15_773_s24_lec07.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec07.pdf)
(78 pages; transcribed in [`raw/slides/07-transformers.md`](../raw/slides/07-transformers.md)) ·
**Transcript:** [`raw/transcripts/07-transformers.md`](../raw/transcripts/07-transformers.md)

## What this lecture establishes

Lecture 6 ended with a promise: stand-alone embeddings such as GloVe give a word one vector whatever
the sentence, and making them **contextual** is what transformers are for. This lecture builds the
**transformer encoder** that does it, one requirement at a time, on a concrete task: reading a travel
query such as "I want to fly from boston at 7 am and arrive in denver at 11 in the morning" and
labelling **every word** with one of 123 "slots" (departure city, arrival time, and so on) from the
ATIS data set (slides 7–13; ≈3:52–10:05).

That task needs three things: an output exactly as long as the input, attention to the **context** of
each word, and attention to word **order** (slides 14–19). **Self-attention** supplies the context. To
contextualize "station" in "The train slowly left the station", take a weighted average of the
stand-alone embeddings of all the words in the sentence, with weights that come from the dot product
of each word's embedding with the embedding of "station", passed through a softmax so they are
non-negative and add to 1 (slides 24–48; ≈16:15–36:23). Several such **attention heads** run side by
side and their outputs are concatenated and projected back to the original size, and a small
**feed-forward** ReLU network adds non-linearity (slides 49–53; ≈36:23–47:51). Self-attention works
on sets, not sequences, so **positional encoding** adds a learned embedding for each position in the
sentence to each word's embedding (slides 54–61; ≈47:51–57:55). The result is the transformer encoder:
as many vectors come out as went in, each the same length, so encoders can be stacked "like pancakes"
(slides 62–69; ≈57:55–1:01:01).

The lecture ends by applying an encoder to slot filling: positional input embeddings, a transformer
encoder, a dense ReLU layer and a softmax for every word (slides 71–76; ≈1:08:01–1:11:06). In Colab
the model is set up on ATIS, with 512-dimensional embeddings, five attention heads and about 5.3
million parameters in the encoder, but it is **not trained in this recording**: the lecturer stops at
9:45 because training takes a few minutes, and continues "on Wednesday" (≈1:15:49).

Three pieces are deliberately left for the next lecture: the linear projections that give
self-attention something to learn, residual connections and layer normalization (slide 70;
≈1:01:01). Until then the self-attention box, as described here, has **no parameters at all**, a
point a student raises and the lecturer defers (≈40:58–41:44).

The catalog titles the video "Deep Learning for Natural Language – Transformers"; the deck's roadmap
calls the lecture "Transformers (1/2)" (slide 1). The recording presents slides 2–71, opens the Colab
at ≈1:02:34, returns to slides 72–76 at ≈1:08:01 ("I forgot about doing this"), and goes back to the
Colab at ≈1:11:06. The Colab notebook is not published on OCW and is not in this knowledge base.
Everything said below about the code follows the lecturer's narration, and only the functions and
arguments he names are given.

## Part 1 — The problem: a label for every word

### Transformers everywhere

Transformers were invented for machine translation, but they have become "an incredibly effective
deep neural network architecture" for a vast array of domains, to the point where "you almost
reflexively will try a transformer first" (≈0:00). Slide 2 lists information retrieval and search,
machine translation, speech recognition, text-to-speech, computer vision, reinforcement learning,
generative AI (large language models, text-to-image models, image captioning) and numerous
special-purpose systems such as AlphaFold, the protein-folding AI, which "runs on a transformer
stack" (≈0:46). Even for the computer-vision problems the course solved with convolutional networks,
there are now transformers that are "actually quite good" (≈0:46).

### From a natural-language query to a structured one

The motivating use case is **search**, or more broadly information retrieval: people type or say
natural-language queries and the system has to make sense of what they want (slide 3; ≈1:33). Unlike
"write me a limerick about deep learning", these queries have right answers, and "accuracy is a high
bar" (≈1:33–2:18). The running example is "Find me all flights from BOS to LGA tomorrow morning"
(slide 4).

A common approach has been to convert the natural-language query into a **structured query**, such as
SQL, run it against a database and format the results for the user (slide 5; ≈2:18–3:05). SQL is just
one familiar choice of structured query. To do that, the system must automatically **extract the
travel-related entities** from the query: BOS, LGA, tomorrow morning, flights (slide 6; ≈3:05–3:52).

### The ATIS data set and slot filling

The data set is ATIS, the Airline Travel Information Systems data set (slide 7, which cites
https://aclanthology.org/H90-1021/). It holds several thousand queries, and the researchers who
compiled it tagged every word of every query by hand with the kind of travel entity it is, or none
(≈3:52). They call these labels **slots**, and the task is to classify each word of a query into the
right slot (slides 8–9; ≈4:38).

Slide 10 tags the query "I want to fly from boston at 7 am and arrive in denver at 11 in the
morning". "I want to fly from" gets five O's: O is the "other" slot, which "we don't really care
about" (≈4:38). "boston" is B-fromloc.city_name, clearly a departure city. "7 am" is two tokens,
B-depart_time.time then I-depart_time.time: an entity can take more than one word, and **B** marks the
token where it **begins** while **I** marks a token in the middle of it (≈5:23–6:09). "denver" is
B-toloc.city_name, "11" is B-arrive_time.time, and "morning" is B-arrive_time.period_of_day. In all,
every word is classified into one of **123 possible slots** (slide 11; ≈6:55), with labels such as
aircraft code, airline name, arrival date relative to today, and round trip. The researchers "thought
of every possibility".

Two questions about the set-up (≈7:40–8:28). Is "Boston" tagged as a departure city everywhere? No:
the tags were assigned by people reading each query, so Boston gets the departure tag only in a
sentence where it is the departure city, and a different tag where people are flying into Boston.
Does the data set handle compound queries, two trips in one request? No, it takes single queries,
because most people ask for a flight from here to there or the cheapest way to get somewhere.

### A classification for every word

So this is a **word-to-slot multi-class classification problem** (slides 12–13; ≈8:28–10:05). The
wrinkle is that the whole query is not classified into one of 123 possibilities; each of the 18 words
of the example is. If the query could be run through some deep neural network whose output layer had
18 nodes, "18 in, 18 out", a **123-way softmax** could be attached to each of the 18 outputs (slide 13;
≈9:17–10:05). "Isn't it cool that we can just casually talk about sticking a 123-way softmax onto each
one of 18 nodes?" (≈10:05). For softmax itself, see
[multi-class classification](multiclass-classification.md).

## Part 2 — Three requirements, and the architecture that meets them

### Same length, context and order

What must the network take into account (slides 14–15; ≈10:05–11:36)?

1. **The output must have the same length as the input**, so that each output element can be
   classified to the right slot type. And inputs come in different lengths, short sentences and long
   ones, so the network must accommodate a variable size of input while keeping the output's
   cardinality equal to the input's.
2. **The surrounding context of each word.** Seeing "Boston" alone, you cannot tell a departure city
   from an arrival city; you have to look for a "from" or a "to" around it.
3. **The order of the words.** Going from Boston to LaGuardia is very different from going from
   LaGuardia to Boston.

On context, slides 16–17 recall lecture 6: a word's meaning can change dramatically with its context,
and a single embedding like GloVe for all the contexts a word appears in "isn't good enough". If a word
has many meanings, its one vector is "some mushy average of all those meanings" (≈11:36). Slide 17's
examples: "see" in "I will see you soon", "I will see this project to its end" and "I see what you
mean"; "bank", the lecturer's favourite, in a loan, a job offer and "the left bank"; "it" in "The animal
didn't cross the street because it was too tired" and "... because it was too wide", where a network
has to work out what "it" refers to; and "station", a train station, a radio station, being stationed
somewhere, or the International Space Station (≈12:22). The "station" example returns for the rest of
the lecture. For order, slide 18 leaves a placeholder, "add your own examples" (≈13:10).

### The transformer

The **transformer** is "a very elegant architecture which checks these three boxes beautifully"
(slide 19; ≈13:10). The name comes from the third box: if ten things come in, ten things go out, and
each of the ten that go out is a transformed version of one that came in (≈13:10–13:55). It was
developed in 2017 and has had a dramatic and ongoing impact on deep learning (slide 20).

One illustration is Google search (slides 21–22; ≈13:55–15:26). Search is already so heavily optimized
that improvements are usually marginal, but when Google introduced BERT, a transformer model the
course covers in the next lecture, there was a significant jump in quality. Google's blog post, which
slides 21–22 cite (https://blog.google/products/search/search-language-understanding-bert/), shows the
query "2019 brazil traveler to usa need a visa". Before, the first result was a Washington Post
article on how U.S. citizens can travel to Brazil without a visa, the wrong direction, because word
order was not being taken into account. After, it was the U.S. Embassy in Brazil's page on tourist
visas.

The architecture comes from the paper "Attention Is All You Need" (slide 23, https://arxiv.org/abs/1706.03762),
whose model-architecture figure is "an iconic picture at this point" and which the class will
understand by the end of the next lecture (≈15:26). OCW excludes that figure from its licence, and the
slide file describes it in prose. The researchers, focused on machine translation, did not realize
what they had stumbled on; it was the rest of the research community that applied it to everything
else.

A student asks whether the many O's could be discarded first, with a two-stage model: an O-versus-not
classifier, then a second model only for the rest. That is "a totally fine way to do it", but even a
simple one-stage transformer gets "fantastic accuracy" (≈15:26–16:15).

The lecture then takes the three requirements one at a time, starting with context (slide 24).

## Part 3 — Self-attention: from stand-alone to contextual embeddings

### Start with stand-alone embeddings

Take the sentence "The train slowly left the station" (slides 26–27; ≈16:15–17:49). Each word gets a
stand-alone embedding, $w_1$ for "The" through $w_6$ for "station", drawn as a column of four grey
cells. Focus on "station", which could mean very different things: how can its embedding be modified
so that it incorporates the other words? Clearly this is a train station, so the presence of "train"
should alter the embedding of "station". "That's what taking context into account actually means"
(≈17:49).

### Suppose we knew how much attention to pay

Imagine we were told how much weight to give each of the other words: in the language of the
literature, how much **attention** to pay to each (slide 28; ≈17:49–18:35). Intuitively, which words
deserve the most? Students answer "train", then "slowly"; "the" deserves hardly any (slides 29–30;
≈18:35–19:20). That intuition can be written as weights that add up to 1 (slide 31; ≈19:20–20:05):

| The | train | slowly | left | the | station |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 0.3 | 0.12 | 0.08 | 0.05 | 0.4 |

"Station" itself gets a large weight, 40%: the aim is to move its own embedding slightly, so it has to
be the starting point, and "you can't ignore itself" (≈20:05). The numbers are made up for now.

Three students' questions pin this down (≈20:05–21:38). These six weights are pertinent **only to
"station"**; every other word will get its own set. Word order plays no part yet; it comes later. And
how would the model know that "left" matters less here? "It has to figure it out": the architecture
only gives it capabilities, and how they manifest "is all going to emerge from training".

### The contextual embedding is a weighted average

Given the weights, the simplest way to contextualize $w_6$ is a **weighted average** of the stand-alone
embeddings of all the words in the sentence, as a student suggests (slides 32–33; ≈21:38–22:23):
$0.05$ times the vector for "The" plus $0.3$ times the vector for "train", and so on. The result is
another vector, and "that vector, folks, is the contextual embedding vector of 'station'" (≈22:23).

![Slide 33: the words "The train slowly left the station", labelled w1 to w6, each above a column of four grey cells. The weights 0.05, 0.3, 0.12, 0.08, 0.05 and 0.4 are written beside the columns joined by plus signs, so that each weight multiplies the vector to its right, and they also sit in a grey strip below. A large dark-red U-turn arrow carries the strip up to the row of vectors. Red text at the right reads "Idea: For each word, we can calculate a weighted average of the stand-alone embeddings of all the words in the sentence".](../raw/images/07-transformers/slide-33.png)

*Slide 33 — the weights sit beside the vectors as one weighted sum, 0.05 times the first vector plus 0.3 times the second, and so on. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec07.pdf)*

The deck draws the stand-alone embedding in grey and the contextual one in blue, a colour scheme it
keeps for the rest of the lecture (slides 34–35; ≈23:13).

![Slide 35: the same six word vectors with their weights. A thin arrow runs down from the grey column of w6 to a blue column of four cells at the lower right, and a red dashed outline encloses both, labelled "Stand-alone embedding" beside the grey column and "Contextual version of w6" beside the blue one.](../raw/images/07-transformers/slide-35.png)

*Slide 35 — the grey stand-alone embedding of "station" becomes the blue contextual one. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec07.pdf)*

Written formally (slide 36; ≈23:59–24:45), with $s_{i,6}$ the weight of the $i$-th word on the sixth
word, "station", and $\hat w_6$ ("w6 with a little hat on it") the contextual embedding:

$$\hat w_6 = s_{1,6} w_1 + s_{2,6} w_2 + s_{3,6} w_3 + s_{4,6} w_4 + s_{5,6} w_5 + s_{6,6} w_6$$

The same logic is applied to every word of the sentence, each with its own weights, so $w_1$ through
$w_6$ become $\hat w_1$ through $\hat w_6$ (≈27:09). The contextual vector **replaces** the stand-alone
one; the original is not kept alongside it. As the vectors flow up through the transformer they get
"more and more and more contextualized" (≈26:19–27:09).

Two side questions (≈24:45–26:19). The context is always the query itself: a search engine gets a
full sentence and uses only that sentence as the context. A brand-new word with no stand-alone
embedding at all is a question the lecturer "punts on" until he has explained byte-pair encoding.
Search autocomplete uses a different mechanism, and he is not privy to how Google does it now.

### Where the weights come from: relatedness as a dot product

The weight of a word should be proportional to **how related it is** to "station": "train" very,
"the" probably not much (slides 37–38; ≈27:09–27:56). One way to quantify how related two words are
is the **dot product** of their stand-alone embeddings (slide 39; ≈27:56).

Slide 40 holds only the word "iPad"; the lecturer explains the dot product in a drawing on his iPad,
which is not in the deck (≈28:42–31:00). The dot product of the vectors for "train" and "station",
written with angle brackets, is the length of one times the length of the other times the cosine of
the angle $\theta$ between them:

$$\langle \text{train}, \text{station} \rangle = \lVert \text{train} \rVert \thinspace \lVert \text{station} \rVert \cos\theta$$

Assume for simplicity that the lengths are roughly 1. Then "all the action is" in the cosine. Two
vectors very close to each other have an angle near 0, and the cosine of 0 is 1. Two vectors 90 degrees
apart, orthogonal, have a cosine of 0. Two pointing in opposite directions, 180 degrees apart, have a
cosine of −1. So the dot product is close to 1 for related words, 0 for unrelated ones and −1 for
"anti-related" ones; "that's how dot products capture this notion of closeness or relatedness"
(≈30:14–31:00).

### Making proper weights with the softmax

Dot products cannot be used as weights as they are (slide 41; ≈31:00–31:47). A weighted average needs
weights that are **non-negative and add up to 1**, and cosines can be negative. Where has the class
done this before? The **softmax** (slide 42; ≈31:47). Exponentiate each dot product, which makes it
positive, then divide each by the sum of all of them:

$$s_{1,6} = \frac{\exp\langle w_1, w_6 \rangle}{\sum_{i=1}^{6} \exp\langle w_i, w_6 \rangle}$$

and likewise for $s_{2,6}$ through $s_{6,6}$. Here $\langle w_i, w_6 \rangle$ is the dot product of
the $i$-th word's stand-alone embedding with that of "station" (≈32:32). So, in summary: take all the
stand-alone embeddings, compute the weights with this formula, take the weighted average, "and boom,
done" (slide 43; ≈32:32–33:17).

![Slide 43: the words of "The train slowly left the station" with w1 to w6 beneath them; six dark-red lines labelled s1,6 to s6,6 run from the w's and converge on w6-hat at the lower right. w6 is circled in blue and labelled "Stand-alone embedding", w6-hat circled in red and labelled "Contextual embedding", with a large arrow between the labels. Below are the softmax formula for s1,6 and the boxed weighted sum for w6-hat.](../raw/images/07-transformers/slide-43.png)

*Slide 43 — the whole of self-attention for one word: softmax weights from dot products, then a weighted sum. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec07.pdf)*

### Why it works

Choosing weights this way drags the embedding of a word closer to the embeddings of the other words in
the sentence, **in proportion to how related they are** (slides 44–47; ≈33:17–34:50). "Station" has
many contexts; suppose for the sake of argument only two, trains and radio. In this sentence "train" is
closely related to "station" and so exerts a strong pull on it. "Radio" is also related to "station",
but it does not appear in the sentence, so it automatically gets zero weight. By moving "station"
closer to "train", equivalently by paying more attention to "train", the embedding is contextualized
to "trains, platforms, departures, tickets, and so on. It's like this portal into the whole train
world" (≈34:04). A sentence that mentions both a radio station and a train station is one transformers
"can deal with", the lecturer adds, without going into it (≈33:17).

![Slide 47: three dark-red dots in a row labelled "train", "station" and "radio", with a dashed arrow running from "station" towards "train". The bullets say that 'station' has many contexts, that 'train' exerts a strong "pull" on it in the current context, that 'radio' does not appear in the sentence and so has zero weight, and that moving station closer to train contextualizes it to trains, platforms and departures.](../raw/images/07-transformers/slide-47.png)

*Slide 47 — "train" pulls "station" towards it; "radio", absent from the sentence, pulls with zero weight. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec07.pdf)*

### This is self-attention

"This, folks, is called **self-attention**", and it is the key building block of transformers:
stand-alone embeddings come in, contextual embeddings go out (slide 48; ≈34:04–34:50). The slide
adds that the operation can be done very efficiently with matrix operations; the lecture does not
show them.

Students' questions fill in the picture (≈34:50–36:23):

- **How much weight does a word give itself?** Typically a high number, since a vector's cosine with
  itself is 1, but there is no guarantee it is the highest, because the embeddings need not have
  length 1; they are kept "smallish". One way to think of it: average everything else first, then
  average that with the old embedding.
- **Why keep one output per input, even for the O words?** Because keeping the number of vectors
  intact as they move through the transformer stack means there is no loss of information. At the end
  you can choose to aggregate or summarize; "it preserves your optionality as long as possible".
- **How long can a sentence be?** There is a **context window**, the maximum length of sentence the
  model handles, which is a parameter you set; the Colab uses 30.

### A question deferred: there is nothing to learn

A student asks how different attention heads (Part 4) can differ, when the weights are only a function
of dot products of the same embeddings (≈40:13–41:44). "Great question. And I literally have a note in
my slide saying if a student asks this good question, tell them to wait till Wednesday." As described
so far, the self-attention box **has no parameters**, no coefficients, nothing to learn; and if
neither of two heads is learning anything, "what's the point?" The answer comes in the next lecture: slide 70 lists "linear projections
of the incoming embeddings into three different spaces before the self-attention operation is carried
out" among the elements still to be covered. "And if you can't stand the suspense, read the book"
(≈41:44).

## Part 4 — Multi-head attention and the feed-forward layers

### Several heads, like several filters

Self-attention is easy enough that the lecture adds a tweak: **multi-head attention** (slide 49;
≈36:23–37:55). Each self-attention block is an **attention head**, and the encoder runs several of them
side by side. Why? Because a complicated sentence carries many patterns at once: slide 50 lists tense, tone
and the relationships between entities, and the lecturer adds grammar and facts versus opinions. There is "not one way to pay
attention", so each head can learn something different. It is "exactly like having lots of filters in
a convolutional network", where one filter might learn a line and another a curve (≈37:55), as
slide 49's footnote says: "Similar to having multiple filters in a convolutional layer". Nobody tells
a head what to learn; it learns in training. See
[convolutional neural networks](convolutional-neural-networks.md).

![Slide 49: two gold boxes labelled "Self-Attention", side by side, each taking w1 to w6 in from below and giving w1-hat to w6-hat out above. Arrows from both sets of outputs lead to a smaller gold box at the top labelled "Concatenate and Project", which gives six outputs. A footnote reads "Similar to having multiple filters in a convolutional layer".](../raw/images/07-transformers/slide-49.png)

*Slide 49 — two attention heads over the same inputs, combined by concatenating and projecting. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec07.pdf)*

Slide 51 shows two heads from the original paper on the sentence "The Law will never be perfect, but
its application should be just - this is what we are missing, in my opinion" (≈37:55–39:26). In one
head, the contextual embedding of "perfect" draws heavily on "law"; in the other, it draws heavily on
"perfect" itself and little else, and the patterns for other words differ subtly too. The figure is
excluded from OCW's licence and is described in the slide file. Why heads attend the way they do is
usually hard to interpret, but with many heads "the performance on the task that you care about gets
really much better" (≈39:26).

A student recalls that lecture 4's filters were specified as 1's and 0's. Those were only example
numbers, to show that a filter could detect a vertical or a horizontal line; a real convolutional
network starts from random weights and lets backpropagation find them, and so do attention heads
(≈39:26–40:58).

### Concatenate and project

Each head produces its own contextual vector for every word. The lecturer works through this on the
iPad, which is not in the deck (≈41:44–44:47). Call head 1's output for the first word $\hat w_1$ and
head 2's $z_1$; both are contextual embeddings of word 1. **Concatenate** them: put one after the other
to make a single vector twice as long. That breaks the rule of keeping every vector the same length,
so run the long vector through **a single dense layer** that projects it back to the original size.
With 20 heads the concatenation is 20 times as long, and one dense layer still brings it back. If the
vectors were 100 long going in, 100 come out, and preserving this size "is very important for reasons
that will become apparent a bit later".

### Feed-forward layers for non-linearity

So far nothing is non-linear: the attention boxes have no parameters, and the projection is a dense
layer. The final tweak for the day injects non-linearity with **feed-forward layers** at the end
(slide 52; ≈44:47–46:20). On the iPad, which again is not in the deck: the vector goes through a ReLU
layer, typically one, and as a rule of thumb that layer is about **four times as wide** as the vector,
400 for a 100-long vector; then another dense layer projects it back to 100.

Much of this "felt very ad hoc" when it came out. It came from strong intuitions rather than deep
theory, but since then people have tried to optimize every aspect, and "it's actually pretty difficult
to beat the starting architecture. ... It's a very robust architecture" (≈46:20).

### The story so far

Start with random stand-alone embeddings, or GloVe embeddings, it does not matter; send them through
several self-attention heads; concatenate and project; send the result through a ReLU layer and a
linear layer (slide 53; ≈47:05–47:51). If six vectors of 100 dimensions came in, six vectors of 100
dimensions come out, so the whole thing can be treated as a black box with the same number of things
of the same length on both sides. "The numbers will be different, because they've now been heavily
contextualized. The numbers are much smarter, in other words."

![Slide 53: on the right, two "Self-Attention" boxes each take w1 to w6 and give w1-hat to w6-hat; arrows lead to a small "Concatenate and Project" box, then a larger gold box, "Feed-Forward Layers", which gives the final w1-hat to w6-hat. On the left, a tall dark-red arrow runs up from "Start with random embeddings" to "End with contextual embeddings".](../raw/images/07-transformers/slide-53.jpg)

*Slide 53 — the encoder before positional encoding: attention heads, concatenate and project, feed-forward layers. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec07.pdf)*

## Part 5 — Word order: positional encoding

### Self-attention works on sets

Two of the three requirements are met: context, through the dot products in the self-attention layer,
and an output as long as the input (slide 54; ≈47:51). Word order is ignored completely. Whether the
sentence is "The train slowly left the station" or "The station slowly left the train", the
architecture cannot tell, because dot products "function on sets, not on sequences": every pair is
compared regardless of order. Scramble the words and you get exactly the same contextual embeddings
(slides 55–56; ≈47:51–48:38).

So if order does not matter for a problem, a transformer can be used as it is. Traditional structured
or tabular data, such as blood pressure and cholesterol predicting heart disease, has no order, and
transformers work for both sets and sequences (≈48:38–49:25).

### The fix: add a position embedding

Many schemes have been invented to tell a transformer about order. The lecture uses the simplest,
which "actually works pretty well in practice" (slide 57; ≈49:25–50:11). Treat each **position** in
the input as a categorical variable: if a sentence can be at most 30 words long, a word's position is
a number from 0 to 29. Give each possible position its own embedding, a **position embedding**, and
make each word's input embedding the **sum** of its stand-alone embedding and the embedding of its
position.

A student asks about a typo, "The train slowly a station": strictly, arithmetically, the output is
different, but transformers trained on lots of data are quite robust to such things (≈50:11–50:58).

### A worked example

Slide 58 sets up two tables (≈50:58–52:31). The vocabulary is "unknown", "cat", "mat", "I", "sit",
"love", "the", "you" and "on", and their stand-alone embeddings are only two long, against about 100
for the GloVe embeddings of lecture 6. Sentences can have at most 10 words, so positions run from 0 to 9,
and each position has an embedding of the same length, 2. Where do the position embeddings come from?
The answer to "the general question of where will these weights come from": they are learned by
backpropagation, starting from random numbers (≈51:43).

For the sentence "cat sat mat", look up each word's embedding and its position's embedding and add them
(slides 59–61; ≈52:31–53:18):

| Word | Position | Stand-alone embedding | Position embedding | Sum (input to the transformer) |
| --- | --- | --- | --- | --- |
| cat | 0 | (0.5, 7.1) | (1.3, 3.9) | (1.8, 11.0) |
| sat | 1 | (1.2, 5.3) | (6.3, 3.7) | (7.5, 9.0) |
| mat | 2 | (−2.0, −3.1) | (0.6, 8.1) | (−1.4, 5.0) |

So the positional-encoded embedding of "cat" here is (1.8, 11.0), not (0.5, 7.1). In "mat sat cat",
"cat" is in position 2, so its stand-alone embedding is unchanged but position 2's embedding, (0.6, 8.1),
is added instead (≈53:18). Slide 58 lists "sit" in the vocabulary while slides 59–61 have "sat" in the
same row with the same numbers (see Printed slips).

Questions on the example (≈53:18–55:35):

- **Is there a position table per sentence?** No, one table for every input: it says which embedding
  to use for a word in, say, the seventh position of any sentence.
- **What if a word appears twice?** In "cat cat cat", every "cat" gets the same stand-alone embedding,
  (0.5, 7.1), but the first adds (1.3, 3.9), the second (6.3, 3.7) and the third (0.6, 8.1), so the
  three sums differ even though the word is the same.
- **Does the position table depend on the vocabulary?** It is independent, and depends only on the
  assumed maximum sentence length. The two tables are learned independently as part of training.

The sum is what goes into the transformer; the slides' three-row tables are "just me being overly
verbose" (≈57:09). Why simply add, when a word's meaning might matter more than its position? We do
not know in advance for any sentence; trained on a lot of text, backpropagation finds values that make
accuracy as high as possible on average. There is "always a tension between our human intuition as to
how it should work, and whether you should just throw it into the meat grinder of backprop", and here
backprop "will actually do a pretty good job" (≈56:22–57:09).

Two questions about tokens (≈55:35–58:41). Filler words in transcribed speech, "um" and "uh", are
handled by **byte-pair encoding**, still to come, which makes tokens of characters, word fragments and
words, and such fillers become tokens like any other. And the starting point is always tokens, however
the tokenizer defines them: this lecture assumes a simple tokenizer of whole words, while GPT-4's
tokens may be parts of words, single characters or punctuation marks. The sentence about whether the
GPT family keeps punctuation contradicts itself in the captions (≈57:55); see the transcript's note.

## Part 6 — The transformer encoder

With positional input embeddings feeding the stack, all three requirements are met: context, order and
an output as long as the input (slide 62; ≈57:55–58:41). This is a **transformer encoder** (slide 63).

Slide 64 lines the lecture's diagram up with the encoder half of the paper's iconic figure, excluded
from OCW's licence and described in the slide file (≈58:41–59:30). From the bottom: the inputs, such as
"The cat sat on the mat", become input embeddings; the positional encoding is added, which is the plus
sign in the figure; the result goes through multi-head attention, then an "Add & Norm" step, then a
feed-forward network and another "Add & Norm", and out of the top. The add-and-norm steps, the
"layernorm and residual connections" of slide 64's footnote, are left for the next lecture.

The summary (slides 65–69; ≈59:30–1:00:15):

- The input embeddings can be random weights or pretrained embeddings.
- A position-dependent embedding is added to represent the position of each word in the sentence.
- Multi-headed attention gives a context-dependent representation.
- Finally a simple feed-forward network, typically one hidden layer of ReLUs and a linear layer.
- **Since encoders have the same-sized inputs and outputs, they can be daisy-chained**, or stacked, to
  get more modelling capacity.

That last point is "perhaps the most important" (≈1:00:15). Because the number of tokens and the length
of each vector are the same going in and coming out, "it's the perfect API", and encoders can be
stacked "like pancakes", one on top of another. GPT-3 has 96 of them, the lecturer says, and as always
in deep learning, more layers can do more complicated things, as long as there is enough data to keep
the model from overfitting (≈1:00:15–1:01:01).

Is going **wide**, with many attention heads, different from going **deep**, with many encoders
(≈1:01:01–1:02:34)? Both are done in practice. Within one block, more heads learn different patterns
from the same input. Stacked blocks learn to contextualize what the block below produced, operating at
higher levels of abstraction, as a later layer of a convolutional network combines the edges and
circles of earlier layers into a face.

Left for the next lecture (slide 70; ≈1:01:01): the linear projections of the incoming embeddings into
three different spaces before self-attention, which answer the "nothing to learn" question; residual
connections; and layer normalization. The footnote: "please see the textbook if you can't stand the
suspense".

## Part 7 — Slot filling with a transformer encoder

Applied to the word-to-slot problem (slides 71–76; ≈1:08:01–1:11:06), the input "fly from Boston to
Denver" has the correct answer O, O, B-fromloc.city_name, O, B-toloc.city_name (slide 72). Turn the words
into **positional input embeddings**, run them through a transformer encoder to get **contextual
embeddings**, five in and five out, now drawn in blue (slide 73). Run each through a **dense ReLU**
layer (slide 74), and attach a **softmax** to each of the five outputs, 123-way in the lecture's
telling, to classify every word (slide 75; ≈1:08:47). The weights in all these layers, the embeddings
included, are optimized by backpropagation (slide 76).

![Slide 75: five words, "fly", "from", "Boston", "to" and "Denver", each with a row of four grey cells headed "Positional input embeddings"; an arrow into a dark-red box "TE*" (Transformer Encoder); five rows of four blue cells, "Contextual embeddings"; an arrow into a dark-red box "Dense (ReLU)"; five rows of three yellow cells; and for each row its own small box "SM" under "Softmax", each pointing to that word's label: O, O, B-fromloc.city_name, O, B-toloc.city_name. A legend says the blue and grey rows indicate 4-element embedding vectors.](../raw/images/07-transformers/slide-75.jpg)

*Slide 75 — slot filling: positional input embeddings, a transformer encoder, a dense ReLU layer and one softmax per word. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec07.pdf)*

A student asks why the yellow vectors are three long when the blue ones are four. It is the modeller's
choice: once the transformer has produced its contextual embeddings, "my job is done", and what to do
with them depends on the problem. A good practice is to shrink long embeddings a little with a ReLU
layer, since "when in doubt, throw in a bit of non-linearity", and then classify with a softmax
(≈1:10:19–1:11:06). Another asks whether the positional embedding layer imported in the Colab arrives
already trained. It does not: like the text vectorization layer, it is "an empty shell" whose weights
are learned when the model is trained (≈1:09:34).

## Part 8 — The Colab: ATIS slot filling

The notebook is not on OCW; this follows the narration, which runs from ≈1:02:34 to ≈1:08:01 and from
≈1:11:06 to the end. **The model is defined but not trained in this lecture.**

**The data.** The ATIS data has been put somewhere easy to download (the captions' "raw box" is
unclear, presumably Dropbox; ≈1:03:20). Each row has a query, such as "What kind of ground transportation is available in
Denver?", "What's the airport at Orlando?" or "How much does the limo service cost within
Pittsburgh?", its slot-filling labels, and an intent for the whole query, which this exercise ignores. The query column is the input
and the slot-filling column the output, for both the training and the test data (≈1:03:20–1:04:55).

**The course's code.** The transformer-related Keras code is packaged into a small "HODL library",
"calling it a library is overstating it": a file of code from which the notebook imports a transformer
encoder and a positional embedding layer (≈1:04:08–1:04:55).

**Vectorizing the queries.** The input queries are vectorized with integer output, not multi-hot,
since the integers will be turned into embeddings, giving a vocabulary of **888 tokens** (≈1:05:41).
As in lecture 6, the empty string is reserved as the pad token, then UNK for unknown tokens, then the
most frequent words, "to", "from", "flights"; "turns out Boston is actually the most frequent", the lecturer adds. See
[text vectorization](text-vectorization.md).

**Vectorizing the slots.** The output labels are sentences too, so they need the same treatment. But
the default standardization would strip the periods and underscores and lowercase the B's, destroying
the labels' nomenclature, so standardization is set to none for the output side (≈1:06:29–1:07:14).
The output vocabulary has **125** entries, not the 123 slots of the lecture. Why? A student's answer is
inaudible and the lecturer's reply is garbled in the captions (≈1:07:14). The input side's vocabulary
reserves two extra entries, the pad token and UNK, which would account for the difference; that is
this page's reading, not the lecture's statement.

**The model** (≈1:11:06–1:14:14). Everything is learned from scratch, with no GloVe:

- The input is a query of at most **30 tokens**, the "max query length". A 35-word query is truncated
  to its first 30, and a 22-word one is padded with eight pad tokens.
- A **positional embedding layer** holds both tables, the word embeddings and the position
  embeddings, "totally distinct tables" that just happen to be packaged together, with embeddings
  **512** long.
- The **transformer encoder** needs only three things specified: the embedding dimension (512), the
  dimension of the dense ReLU layer inside it, `dense_dim` (64), and the number of attention heads,
  **five**, "because why not".
- On the encoder's blue outputs, "good old traditional DNN stuff": a dense ReLU layer of 128 `units`,
  dropout, and a dense softmax output over the 125-entry output vocabulary, for every position.

**About 5.3 million parameters in the encoder.** "For extra credit and for your personal edification",
after the next lecture, students are invited to calculate by hand the number of parameters inside the
transformer and check it against the summary's 5.3 million ("53 million — sorry, 5.3 million"); the
notebook lists the exact math (≈1:14:14–1:15:02). Any layer's weights can be inspected through its
weights attribute: the positional embedding layer holds one table of 888 tokens by 512 and one of 30
positions by 512 (≈1:15:02–1:15:49). The lecturer stops before training, "because the model, it's going
to take a few minutes to run", promising that what is not yet clear will become much clearer in the next
lecture (≈1:15:49).

**The page's own parameter count.** The lecture does not give the breakdown, and it depends on details
the lecture leaves for later, so what follows is this page's arithmetic under assumptions it names. It
assumes the attention layer has the projections slide 70 announces: for each of the five heads, a query,
a key and a value projection from 512 to 512 with biases, and an output projection from the five
concatenated heads, $5 \times 512 = 2{,}560$ long, back to 512, with biases. It also assumes a
feed-forward block of 512 → 64 → 512 and two layer normalizations, each with a scale and an offset per
dimension (slide 64's footnote). Then:

- query, key and value: $3 \times (512 \times 2{,}560 + 2{,}560) = 3{,}939{,}840$;
- output projection: $2{,}560 \times 512 + 512 = 1{,}311{,}232$;
- feed-forward: $(512 \times 64 + 64) + (64 \times 512 + 512) = 66{,}112$;
- layer normalizations: $2 \times (512 + 512) = 2{,}048$;

which totals **5,319,232**, about 5.3 million, as the summary says. A different per-head size would
give a different total, so the match supports these assumptions without the lecture confirming them.
Outside the encoder the counts follow from numbers the lecture states: the two embedding tables hold
$888 \times 512 + 30 \times 512 = 470{,}016$ weights, the 128-unit layer
$512 \times 128 + 128 = 65{,}664$, and the 125-way output $128 \times 125 + 125 = 16{,}125$; dropout has none. On these
assumptions the whole model has 5,871,037 parameters, a total the lecture does not report.

The encoder's 64-wide ReLU layer is narrower than the 512-long vectors it works on, unlike the
four-times-wider rule of thumb from the iPad drawing (≈45:34). The lecture does not comment on the
choice.

## Printed slips

- Slide 58 lists "sit" in the vocabulary, with embedding (1.2, 5.3); slides 59–61 put "sat" in the same
  row with the same numbers, and the worked sentence is "cat sat mat". The lecturer reads "sit" at
  ≈50:58 and "sat" afterwards.
- Slide 17's first row prints no word in its Word column; its examples are for "see", which the PDF's
  text layer holds there, hidden.
- Slides 72–76 cut the label "B-toloc.city_name" off at the right edge of the page.
- The slot-filling slides and their narration attach a 123-way softmax to every word; the Colab's
  output layer is 125-way (see Part 8).

## Related pages

- [Transformers](transformers.md): self-attention, multi-head attention, the feed-forward layers,
  positional encoding and the encoder, gathered as a topic.
- [Word embeddings](word-embeddings.md): stand-alone embeddings such as GloVe, and the contextual
  embeddings this lecture computes.
- [Natural language processing](natural-language-processing.md): the course's NLP roadmap, from one-hot
  vectors through embeddings to transformers.
- [Multi-class classification and softmax](multiclass-classification.md): the softmax that turns dot
  products into attention weights, and the per-word softmax of slot filling.
- [Text vectorization](text-vectorization.md): integer output, the pad and UNK tokens, truncation and
  padding, and turning standardization off for labels.
- [Keras](keras.md): the HODL transformer encoder and positional embedding layer, and inspecting a
  layer's weights.
- [Convolutional neural networks](convolutional-neural-networks.md): attention heads as the analogue of
  filters, and depth as levels of abstraction.
- [Representation learning](representation-learning.md): contextual embeddings as a learned
  representation that improves layer by layer.
- [Lecture 6](06-word-embeddings.md): stand-alone embeddings, GloVe, and why context needs transformers.
