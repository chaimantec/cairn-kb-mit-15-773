# Lecture 9 — Large Language Models (1/2): Next Word Prediction, Decoding and Tokenization

**Lecturer:** Rama Ramakrishnan ·
**Video:** [youtube.com/watch?v=KGDe1QvfKJ8](https://www.youtube.com/watch?v=KGDe1QvfKJ8) (75 min) ·
**Slides:**
[`mit15_773_s24_lec09.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec09.pdf)
(68 pages; transcribed in [`raw/slides/09-large-language-models.md`](../raw/slides/09-large-language-models.md)) ·
**Transcript:** [`raw/transcripts/09-large-language-models.md`](../raw/transcripts/09-large-language-models.md)

## What this lecture establishes

[Lecture 8](08-transformers-and-self-supervised-learning.md) pretrained a transformer encoder by
**masking**: hide some words and train the network to fill in the blanks, which gave BERT. This
lecture takes a second self-supervised task, **next word prediction**, and builds the model behind
GPT from it (slides 2–4; ≈0:00–3:10). The plain encoder cannot be trained this way. Self-attention lets
every word see every other word, so the model would simply copy the next word from its input (slides
12–16). The fix is **causal self-attention**: when computing a word's contextual embedding, give zero
weight to the words after it (slides 17–20). An encoder built from it is the **Transformer Causal
Encoder**, also called the **Transformer Decoder** (slides 26–29).

A model trained this way generates text one word at a time. It predicts a probability for every word in
the vocabulary, one word is chosen, the word is appended to the input, and the loop runs again (slides
31–36). Such models are **autoregressive** or **causal language models**, and GPT-3, with 96
transformer layers of 96 heads each, is one (slides 37–39). How the next word is chosen from the
softmax's table is called **decoding**. **Greedy decoding** always takes the most probable word.
**Random sampling** draws in proportion to the probabilities, and is tamed in three ways: **top-K**,
**top-P** (nucleus) sampling and **temperature** (slides 42–57). The last part is **tokenization**.
Unlike the standardize-tokenize-index-encode pipeline of [lecture 5](05-text-vectorization-and-bag-of-words.md),
GPT's **byte pair encoding** keeps case and punctuation and can spell any new word, by building a
vocabulary of characters, frequent words and frequent word fragments (slides 58–66).

The catalog title, "Generative AI – Large Language Models (LLMs) and Retrieval Augmented Generation
(RAG)", overstates the video: retrieval-augmented generation is not reached (it is in [lecture 10](10-adapting-llms.md)). The
deck's roadmap calls this lecture "Large Language Models (1/2)" (slide 1). The recording presents
slides 2–66 in order and ends on the token-visualizer demo. Three parts of it are not in the deck, and
are described below from the narration: temperature, which the lecturer derives on an iPad (slide 56
says only "iPad"); the OpenAI Playground demo (slide 57); and the token visualizer (slide 66).

## Part 1 — Next word prediction

### Masking, reviewed

The lecture opens by recalling masking (slide 2; ≈0:00–1:32). For images, pretrained models such as
ResNet needed somebody to label every image. Text has no labels, so replace some of the words in every
sentence with a mask token and train a network to fill in the blanks. Done over all of Wikipedia, this
gave BERT, which students were using in their homework. Slide 2 is lecture 8's masking example (lecture
8, slide 71) under the title "[Review] Masking\*", with the footnote "\*Sometimes referred to as 'Masked
Language Modeling'".

### A different self-supervised task

Next word prediction is "a different kind of self-supervised learning … which turns out to be weirdly
more interesting and powerful" (≈1:32). In a sense it is a special case of masking: instead of blanking
a random word, always blank the **last** one, and have the model predict it. Fragments of sentences work
as well as whole sentences. From "The mission of the MIT Sloan School of Management is …", give the
model "The" and ask for "mission", give it "The mission" and ask for "of", and so on (slide 3;
≈1:32–2:20). Slide 3's footnote gives the other name: "Causal Language Modeling".

![Slide 3: the sentence "The mission of the MIT Sloan School of Management is to develop principled, innovative leaders…" at the top as the Original Input, with dashed arrows to two tables. The left table, "Modified Input", lists growing prefixes: "The", "The mission", "The mission of", up to "The mission of the MIT Sloan … management". The right table, "Fake" Label, lists the next word of each: mission, of, the, MIT, Sloan, School, …, practice. A DNN box of dark red bars sits between them.](../raw/images/09-large-language-models/slide-3.png)

*Slide 3 — next word prediction: every prefix of a sentence is an input, and the word that follows it is its label. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec09.pdf)*

The plan is to take the transformer encoder used for BERT in lecture 8 and train it on this task
(slide 4; ≈2:20–3:10).

### One sentence, many examples

Writing every prefix out as a separate example would turn one sentence into dozens. Instead the
sentence is stacked against a copy of itself shifted by one word (slide 5; ≈3:10–4:44). For "the cat
sat on the mat", cut off the last word to get the input "the cat sat on the", and cut off the first word
to get the output "cat sat on the mat". Read row by row, "the" should predict "cat", "the cat" should
predict "sat", and so on, all inside one data point.

### The pipeline

The input then runs through the same stages as BERT's (slides 6–10; ≈4:44–6:16). The words are
tokenized (not shown) and turned into **positional input embeddings**. They pass through a **transformer
stack**, which produces **contextual embeddings**. These go through one or more dense ReLU layers,
"because it's always a good idea to stick some ReLUs at the very end". Then each position gets **its own
softmax**, whose number of categories is the vocabulary size. For now the vocabulary is taken to be
words, roughly 50,000 of them, so each softmax is "like a 50,000 way softmax", which is "exactly what we
did for BERT" (≈5:30). Tokens come later in the lecture (Part 9).

The next lecture corrects this layer. Its slide 3 is titled "Correction: The contextual embeddings
flow into a Dense layer with linear activations (not ReLU) followed by a Softmax", and it notes that the
dimension of those vectors is the vocabulary size ([lecture 10](10-adapting-llms.md), slides 2–3;
≈0:47–1:32; [slide file](../raw/slides/10-adapting-llms.md)). Every slide of lecture 9 still labels the
box "Dense (ReLU)".

![Slide 10: the next-word-prediction pipeline from left to right. The input words "the cat sat on the" each have a row of four grey squares (positional input embeddings), which feed a dark red box "TS" (transformer stack); out come five rows of four blue squares (contextual embeddings), which feed a box "Dense (ReLU)"; out come five rows of three yellow squares, each with its own small "SM" (softmax) box, and each SM box points to an output word: cat, sat, on, the, mat.](../raw/images/09-large-language-models/slide-10.jpg)

*Slide 10 — one softmax per position over the whole vocabulary; each position's target is the next word. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec09.pdf)*

Only the **last** prediction really matters for next word prediction, because it is the one based on
everything before it. "All the things before, we don't care so much" (≈5:30–6:16). A student asks how
the model would predict the end of a sentence. "The notion of a sentence has disappeared at this
point", the lecturer answers: the tokenizer for these models keeps punctuation, so periods and
exclamation marks are tokens like any other (≈6:16).

### The loss

The loss is the same as for any classification problem, the lecturer says in answer to a question
(≈9:24–11:47). With Fashion MNIST each image was classified into one of 10 categories with a softmax
and cross-entropy. Here every position is classified into one of 50,000, "so essentially, you have as
many classification problems as you have number of words in a sentence". A larger vocabulary means
more compute, but "conceptually, no difference". The loss is the sum of the per-position
cross-entropies, "or to be more precise, the average". At ≈10:58 he unhides a slide he had hidden in
case time ran short (slide 11). Each position's cross-entropy is minus the log of the probability given
to the right answer, and the average over the sentence is printed as

$$-(1/7)\thinspace[\log(\mathrm{prob}(\text{“the”})) + \log(\mathrm{prob}(\text{“cat”})) + \dots + \log(\mathrm{prob}(\text{“}\lt\text{E}\gt\text{”}))]$$

"1 over 7. Boom, that's it" (≈11:01). The formula's seven terms run from "the" to an end marker `<E>`,
while slide 11's diagram shows five output words; neither the slide nor the narration says how the seven
are counted. See [loss functions](loss-functions.md) for categorical cross-entropy.

## Part 2 — The problem: the encoder can see the answer

"Do you notice any problems with the way this thing has been set up?" (slide 12; ≈7:03). One student
points out that after a word like "the" many next words are possible, which is true but not the
problem. Another gets it: since the embeddings are contextual, "the output of the first word is going
to have access to the second word. And so it's kind of like cheating" ("Bingo", ≈7:50).

Slide 13 recalls why. In self-attention, "the contextual embedding for a word is based on ALL the words
in the sentence": each word's contextual embedding is a weighted average over every word, so "the last
word can see the first word, the first word can see the last word" (≈7:50–8:39). For next word
prediction that is peeking into the future. Given "the cat sat on the" as the input, with "mat" in the
output, "the whole thing becomes challenging only if I say, the cat sat on the dash. Now, predict the
dash" (≈8:39). To predict "sat", the network should use only "the cat". But self-attention can see "sat"
in the input, so it trivially learns to copy it (slides 14–16; ≈8:39–9:24). "This is the key problem
with just using the transformer as is."

![Slide 16: the same pipeline, titled "The problem: To predict sat, the network should only use 'the cat', but because Self-Attention can see sat, it will trivially predict the next word to be sat". The input words "the" and "cat" and the output word "sat" are red; the input word "sat" is highlighted in yellow, with a dotted arrow to it from the yellow "sat" in the title.](../raw/images/09-large-language-models/slide-16.jpg)

*Slide 16 — the input "sat" is visible to self-attention, so the prediction of "sat" can simply copy it. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec09.pdf)*

## Part 3 — Causal self-attention

The solution: "When calculating the contextual embedding of a word, give zero weight to 'future'
words" (slide 17; ≈11:47). "If you give zero weight to future words, it's almost as if they don't
exist."

Self-attention starts from a table of pairwise attention weights, one row per word, each row adding up
to 1. Multiplying a row's weights into the embeddings of "the cat sat on the" gives that word's
contextual embedding (≈12:32; see [transformers](transformers.md)). To hide the future, take the table
and zero out every cell above the diagonal, the cells in red on slide 18. Then **renormalize** "so that
the remaining cells … will still add up to one in each row" (slide 18; ≈12:32–13:20). For "sat", only
the weights for "the", "cat" and "sat" are left, so its contextual embedding is a weighted average of
those three words' embeddings alone (slide 19).

![Slide 19: two 5-by-5 grids of attention weights, rows and columns both labelled the, cat, sat, on, the. The left grid, "When calculating attention weights …", is all grey. An arrow leads to the right grid, where every cell above the diagonal is red, captioned "… we zero out the weights for the 'future' words (and renormalize so the weights in each row still add up to 1.0)". In the right grid the row for "sat" has its first three cells (the, cat, sat) in yellow.](../raw/images/09-large-language-models/slide-19.png)

*Slide 19 — causal self-attention: the red cells (future words) get zero weight, so "sat" attends only to "the", "cat" and itself. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec09.pdf)*

This tweak is **causal self-attention**, also called **masked self-attention**, "since future words
are being 'masked' from the calculation" (slide 20; ≈13:20). The prediction for each word now depends
only on the words before it: "the" alone predicts "cat", "the cat" predicts "sat", and all five input
words predict "mat" (slides 21–25; ≈14:05). To build it, every attention head in the transformer becomes
a causal attention head. The way it is done under the hood "is actually very elegant for computational
efficiency purposes, but I won't get into it because it gets a bit involved" (≈14:05). "Causal" here has
"no connection to causality. So it's just a term" (≈14:52).

![Slide 20: the same pair of grids as slide 18 under the title "This modification to Self-Attention is called Causal Self-Attention*": a grey grid on the left and, on the right, the grid with all cells above the diagonal in red. Footnote: "*Also called 'Masked Self-Attention' since future words are being 'masked' from the calculation".](../raw/images/09-large-language-models/slide-20.png)

*Slide 20 — the name: causal, or masked, self-attention. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec09.pdf)*

## Part 4 — The Transformer Causal Encoder, also called the decoder

Replace self-attention with causal self-attention in the transformer block and the result is a
**Transformer Causal Encoder** (slide 26; ≈14:52). "Suddenly, it starts working for next word
prediction. It can't cheat anymore."

![Slide 26: a red box holding two khaki "Causal Attention Head" boxes whose arrows meet in a "Concatenate and Project" box, which feeds a tall "Feed-Forward Layer" box. Grey positional input embeddings w1 to w6 enter from the left, and blue contextual embeddings ŵ1 to ŵ6 leave on the right. Title: "By replacing Self-Attention with Causal Self-Attention in the Transformer block, we get a Transformer Causal Encoder".](../raw/images/09-large-language-models/slide-26.png)

*Slide 26 — the transformer block with causal attention heads: a Transformer Causal Encoder. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec09.pdf)*

**Why "decoder".** The original transformer paper, "Attention is All You Need", was written for machine
translation, English to German and the like, and has an **encoder-decoder** architecture (slide 27;
≈14:52–15:39). The encoder is the one built in lectures 7 and 8. The course does not cover the
encoder-decoder, because it does not cover translation ("We won't cover sequence-to-sequence due to time
constraints"). But the paper's decoder uses "Masked (aka Causal) Multi-Head Attention" (slide 27), and
because of that the causal encoder is also called the **Transformer Decoder** (slide 28). So "decoder"
has two meanings: a synonym for the causal encoder, and the second half of a sequence-to-sequence
translation model. "It will become clear from context" (≈15:39), and in this course, which does no
translation, decoder and causal encoder mean the same thing. The summary (slide 29) sets the paper's
encoder figure beside a copy whose attention box reads "Causal Multi-Head Attention": the left is for
masked prediction, the right for next word prediction (≈16:26–17:12). The figures on slides 27 and 29
are excluded from OCW's licence and described in the slide file only.

A student asks about transformers that are "bidirectional". "All bidirectional means is that I can see
everything", the lecturer says: the encoder of lecture 8 is bidirectional, and the causal encoder looks
only backward (≈16:26).

**Training.** With a stack of causal encoders ("TCE" on slide 30) in place of the transformer stack,
next word prediction is trained exactly as set up in Part 1: inputs and outputs, the stack, a dense ReLU
layer, a softmax per position. "All important changes go into the attention layer and make it masked or
causal" (slide 30; ≈17:12–18:02).

**Questions: training versus use.** Is the masking only for training? After training, the lecturer
answers, the final prediction is the only thing you care about, and by definition it uses everything
that came before it. So the model works for inference unchanged: "you don't have to unmask it for
inference because you don't need to" (≈18:02–19:34). Masked prediction and next word prediction give
two different models with very different weights: "BERT is used for certain things. And this kind of
model, which is the basis of GPT, is going to be used for other things" (≈19:34–20:19). One can see
everything (BERT), the other cannot see the future (GPT).

## Part 5 — Generating text one word at a time

Once trained, give the model any input, say "It was a dark and" (slide 31; ≈20:19). The predictions at
the earlier positions are thrown away. Only the **softmax for the last token** matters, and it is a
probability distribution over the whole vocabulary, "a 50,000 way table of probabilities" that add up to
1 (slide 32; ≈21:06). On the slide's table, "stormy" is the most likely at 0.6 ("just for kicks"):

| Next word | Probability |
| --- | --- |
| aardvark | 0.0003 |
| … | |
| rainy | 0.3 |
| … | |
| stormy | 0.6 |
| … | |
| zebra | 0.00009 |

Choose a token from this table (slide 33). The simplest choice is the most likely word, "stormy".
**Append it to the input** and run the model again (slide 34; ≈21:52–22:40). The input has changed, so
the softmax is different and gives a new table, in which "night" is now the most likely at 0.95
(slide 35). Append "night" and repeat as needed (slide 36). The loop can be told to stop after, say, 100 tokens, or the model
may stop when it produces punctuation such as a period: "we have control over this when it stops and how
it stops" (≈22:40). "The basic building block is next word prediction, feed it back to the input, next word
prediction, keep on doing it. You keep on doing it, then, suddenly, it's writing entire novels for you"
(≈23:25).

![Slide 36: the generation pipeline for the input "It was a dark and stormy" ("stormy" in red): six rows of grey squares feed a tall "TCE" box, then blue contextual embeddings, "Dense (ReLU)" and yellow rows; only the last row continues through an "SM" box, to the output word "night" in red. Title: "… and repeat as needed."](../raw/images/09-large-language-models/slide-36.jpg)

*Slide 36 — the chosen word "stormy" has been appended to the input; the last position's softmax now gives "night". [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec09.pdf)*

Questions after this (≈23:25–25:45). Is a longer input better? It depends on the task, but "the more
helpful the input, the better". Does Google's autocomplete work this way? The lecturer doesn't know;
these things are kept under wraps. He suspects Google's generative AI panel does, but autocomplete may
not, because these models are compute-heavy. Is the vocabulary static? Yes, for a given model, and it is
"a token vocabulary", not a word vocabulary. How it copes with new words has "a very elegant solution
that's coming up": byte pair encoding (Part 9).

## Part 6 — Autoregressive language models and GPT-3

With the causal encoder, the course can now do all three uses of a transformer from lecture 8:
sequence classification and sequence labeling with BERT-like models, and now **sequence generation**,
completing "I loved the movie" into "I loved the movie, especially the cinematography and the background
score" (slide 37; ≈25:45). Slide 37 repeats lecture 8's slide 30 with generation ticked off.

"Such sequence generation models trained on text sequences are called *Autoregressive Language Models*
or *Causal Language Models*" (slide 38). The GPT family is the best-known example. "Autoregressive"
comes from econometrics: predict something, then feed the past predictions back in as inputs for the
next prediction (≈25:45–26:32). This is what a chatbot is doing as its reply appears word by word. Through
the API, a streaming option sends each token as it comes out of each pass (≈26:32). In the web
interface, the lecturer has heard from people in the field, the typing may be deliberately slowed to
feel human, "a UX trick", though "I don't know to what extent this is 100% true and how pervasive it
is" (≈27:18–28:05).

**GPT-3 is an autoregressive LLM** (slide 39; ≈28:05–29:38). The L, for "large", is there because it was
trained "on lots of data with lots of parameters. … Nothing more momentous than that." Slide 39 gives:

- Architecture: 96 transformer layers, and 96 heads in each multi-head attention layer.
- Training data for next word prediction: about 30 billion sentences from the Internet and books.

It also reproduces the GPT-3 paper's table of training data (http://arxiv.org/abs/2005.14165):

| Dataset | Quantity (tokens) |
| --- | --- |
| Common Crawl (filtered) | 410 billion |
| WebText2 | 19 billion |
| Books1 | 12 billion |
| Books2 | 55 billion |
| Wikipedia | 3 billion |

The training was "exactly, next word prediction. That's it." GPT-3's architecture is known because its
paper gives the details. GPT-4's was never published: "after GPT-3, everything became closed" (≈28:51).

Generating text covers a wide range of uses, since "code is just text" too: text generation, code
generation, code documentation, summarization, question answering and chatbots (slide 40; ≈29:38). "All
the excitement we see around GenAI from the time ChatGPT came out is precisely because this simple idea
of text in, text out is just so flexible." For anyone curious how these models are built, the lecturer
recommends Andrej Karpathy's video building GPT from scratch, and a blog post that builds GPT in NumPy
without any framework (slide 41, "GPT in 60 Lines of NumPy"; ≈29:38–30:26).

## Part 7 — Decoding: choosing the next token

"The process of choosing a token from the probability distribution generated by the softmax is called
**decoding**" (slide 43; ≈31:11). To get really good performance out of systems like ChatGPT, "you need
to be quite thoughtful about how to decode" (≈31:11).

![Slide 43: the generation pipeline for "It was a dark and" through a "TCE*" box, contextual embeddings, "Dense (ReLU)" and one "SM" box at the last position, with an arrow to a table: Next Word / Probability — aardvark 0.0003, …, rainy 0.3, …, stormy 0.6, …, zebra 0.00009. Title: "The process of choosing a token from the probability distribution generated by the softmax is called decoding".](../raw/images/09-large-language-models/slide-43.jpg)

*Slide 43 — decoding is the choice of one token from the last position's probability table. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec09.pdf)*

### Greedy decoding and random sampling

There are two simple, extreme options (slide 44; ≈31:56–32:42):

- **Greedy decoding**: choose the highest-probability token, "stormy" at 0.6. Slide 44's footnote adds
  that it is "Not necessarily optimal. Beam Search can help", with a link to a Borealis AI tutorial on
  decoding algorithms. Beam search is not discussed further.
- **Random sampling**: draw a token in proportion to its probability. Sampled 100 times from the table,
  "60 times, you probably get stormy", and a small fraction of the time something strange like
  "aardvark" or "zebra".

### Which one to use

It depends on the task (slide 45; ≈32:42–35:02). Use **greedy decoding** when factual accuracy matters
or the output must be **deterministic**. Two customers asking a support agent the same question should
not get different answers. For a given input the softmax table never changes, and neither does its
highest entry, so greedy decoding guarantees determinism. "For reasoning questions … math questions,
reasoning questions, logic questions, you should really sort of keep it as greedy as possible in my
experience" (≈33:28). Use **random sampling** when diversity and "creativity" matter: a poem, a haiku, a
screenplay. The outputs become stochastic, which is often fine, and varied responses keep a chatbot
from sounding "stilted" and "robotic" (≈34:15–35:02).

Two questions follow (≈35:02–36:33). Does sampling affect the loss or the probabilities? No: "the table
doesn't change, regardless of how you sample it. … You literally get the table, and then you literally
can forget how you got the table. And now decoding starts." Is random sampling why ChatGPT gives a
different answer to the same prompt? "Correct. That's exactly why." Bing's "creative, balanced,
precise" settings, he says, manipulate these same parameters under the hood. Through the API you can set
them directly (≈36:33).

### Why random sampling goes wrong: the head and the tail

Sort the softmax table from high to low probability. The hope is that a good model concentrates its
probability on a **short head** of "good" tokens and leaves a **long tail** of low-probability "bad",
irrelevant ones, so random sampling picks from the head (slide 46; ≈36:33–39:44). But "while the
probability of choosing any individual token in the tail is small, the probability of sampling *some*
token from the tail can still be high" (slide 46). In the table above, "stormy" (0.6) and the runner-up
(0.3; "rainy" on slide 44, "night" as he says it) together have 0.9, so "there is a 10% probability it's going to be one of these words … It might be some
random nonsense word" (≈39:44). And the model cannot recover: "if the LLM happens to sample a token from
the tail … it'll just go off the rails, which is why every word that gets generated, it's really
important to get it right" (≈39:44–40:30). Asked whether there is a technical definition of the head,
he says no: it depends on the question and the model, perhaps 20 words for one question, 40 for another,
10 for another model (≈40:30).

The example, from GPT-3.5 (slides 47–50; ≈40:30–43:40). Given the prompt "Students at the MIT Sloan
School of Management are", the most likely next words are "invited" (3.18%), "given" (2.69%),
"expected" (2.59%), "required" (2.50%) and "able" (2.09%), all small, though the rest of the vocabulary
is smaller still (slide 47). Choosing "invited" gives a reasonable continuation: "… are invited to
submit their original white papers to the annual MIT" (slide 48). Forcing two tail words, "masters"
(0.05%) and "spending" (0.11%) (slide 49), gives "… are masters of chaos. They routinely blow past
deadlines, fracture" ("And then I couldn't take it anymore. I stopped it") and "… are spending the
semester learning life skills — through knitting socks" (slide 50). "I'm not making this stuff up. This
is GPT 3.5. So, yes, it will go off the rails" (≈42:52).

Is that text in the training data? Probably not as such; the model "is basically doing some very rough
and approximate pattern matching from all the training data it was trained on". What it gives is "a
response that is not implausible. There is no guarantee of correctness. … It's giving you a
probabilistically plausible response. That's it" (≈43:40–44:27). Asked to list the awards Professor
Ramakrishnan has won, it produced an impressive list, none of it true (≈44:27). A student wonders
whether later words are weighted more heavily, since "students at MIT" should have steered it. No, the
lecturer answers: training gave it sentence fragments and asked for the next word, and the more of the
input it sees, the more clues it has, as "the capital of France is" narrows
the cone of uncertainty far more than "the capital of". He passes on an expression for what LLMs do,
"subtractive sculpting": every word chips away at a block of marble until the statue is clear
(≈45:12–46:44).

## Part 8 — Tuning random sampling

All three ways of taming random sampling (slides 51–57) **first modify the distribution, then sample**.
Each focuses on the head and kills the tail, which raises the question of how to decide what the head is
(≈45:58–46:44).

### Top-K sampling

"Consider only the *K* most probable words. Renormalize their probabilities so they add up to 1. Do
random sampling from this set" (slide 51; ≈46:44). K might be 10, 20, 30, 40 or 50, "very problem
dependent". On slide 52, with the prompt "It was a dark and ____" and K = 2, only "stormy" (0.6) and
"rainy" (0.2) are kept, and renormalized they become 0.75 and 0.25 (≈47:32), that is, 0.6 and 0.2 each
divided by their total, 0.8. Top-K is "very commonly used". Its shortcoming is the assumption that the
head always has K words, whatever the input: "obviously, it's not a well-supported assumption"
(≈47:32).

![Slide 52: "Top-K Sampling with K=2", prompt "It was a dark and ____". A bar chart of probability over eight words: stormy 0.6, rainy 0.2, foggy 0.1, then damp, cold, humid, dreary and wet, each smaller with no value printed. A red rectangle encloses the stormy and rainy bars. An arrow labelled "Renormalized" leads to a chart of two bars, stormy 0.75 and rainy 0.25, and then an arrow labelled "Sampling".](../raw/images/09-large-language-models/slide-52.png)

*Slide 52 — top-K with K = 2: keep the two most probable words, renormalize, then sample. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec09.pdf)*

### Top-P (nucleus) sampling

What matters is that the chosen words hold most of the probability, however many there are (≈48:17).
**Top-P**, also called **nucleus sampling**: "Consider the minimum set of words whose total probability
exceeds *P*. Renormalize their probabilities so they add up to 1. Do random sampling from this set"
(slide 53). Sometimes that is two words, sometimes 20. With P = 0.9, adding up 0.6, 0.2 and 0.1 reaches
0.9 at "foggy", so those three are kept, and renormalized they become 0.67, 0.22 and 0.11 (slide 54;
≈48:17–49:03), each divided by 0.9. "This, actually, is even more effective in my opinion because it
sort of fluctuates. It doesn't hardcode the number of words." If "foggy" had been 0.12, it would still
be included: the method keeps adding words until the total crosses P (≈49:03).

![Slide 54: "Top-P Sampling with P=0.9", the same eight-bar chart as slide 52 (stormy 0.6, rainy 0.2, foggy 0.1, then damp, cold, humid, dreary, wet), with the red rectangle now enclosing stormy, rainy and foggy. "Renormalized" leads to three bars, stormy 0.67, rainy 0.22 and foggy 0.11, then "Sampling".](../raw/images/09-large-language-models/slide-54.png)

*Slide 54 — top-P with P = 0.9: keep the fewest words whose probabilities add up to 0.9, renormalize, then sample. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec09.pdf)*

Could one just drop every word below a probability threshold? You can, but you won't know how many words
crept over it, and a word just under the line is lost: "there's no right answer, unfortunately"
(≈49:50–50:35). No rule fixes the head's size in advance, so these are heuristics, and in practice people
try them all. Top-K and top-P can be combined: take words until the total crosses P or the count reaches
K, whichever comes first (≈50:35).

### Temperature

The third method is **temperature**: "By adjusting this parameter, we can make the 'head' of the
distribution less or more important" (slide 55). Top-K and top-P draw a hard line. Temperature "is like a
softer way to do the same thing" (≈51:21). Slide 56 says only "iPad", and the lecturer derives it in a
drawing that is not in the deck (≈51:21–56:01), described here from the narration.

The probabilities come from a softmax. Call the numbers in the output layer $a_1, a_2, \dots, a_n$, one
per word (about 50,000 of them). The softmax turns them into probabilities by exponentiating each and
dividing by the sum (see [multi-class classification and softmax](multiclass-classification.md)):

$$\text{probability of word } i = \frac{e^{a_i}}{e^{a_1} + e^{a_2} + \dots + e^{a_n}}$$

Temperature adds one new parameter, $T$, that divides every $a$:

$$\text{probability of word } i = \frac{e^{a_i/T}}{e^{a_1/T} + e^{a_2/T} + \dots + e^{a_n/T}}$$

- **$T$ close to 0.** Dividing by a tiny $T$ makes every $a$ huge in size, and the biggest $a$, "already
  big", becomes "massive", so its exponential dominates everything else. The word with the biggest $a$
  gets a probability of 1 or close to 1 and every other word close to 0, so "stormy" might get 0.999.
  Sampling from that always picks the biggest, so **temperature 0 is greedy decoding** (≈53:43–56:01).
- **$T$ large.** As $T$ grows, the distribution becomes flat: every word becomes about equally likely,
  and sampling becomes "truly random" (≈56:01).

### The OpenAI Playground demo

At platform.openai.com the lecturer opens the **OpenAI Playground** (slide 57; ≈56:50–1:00:45), where
you choose a model, type text, and can ask it to show the softmax probabilities of each word it
generates. Temperature defaults to 1; 0 makes it greedy, and it can go above 1. "They don't have support
for top-k, OpenAI, but they do have support for top-p." He keeps the output short ("Let's just call it
30") and enters "Students at the MIT Sloan School of Management are". After a false start ("Oh, I know
what it is. GPT 3.5"), clicking a word shows its probabilities, with "invited" at about the slides'
3.18–3.19% ("Close enough", ≈57:37–58:24).
With the temperature at 0 it gives "invited" every time (≈59:10). People use the Playground heavily for
debugging, he says, to see what probability distributions a model gives for a use case and tune it from
there. Then he sets the temperature to its maximum, 2: "Ladies and gentlemen, I present to you a modern
large language model" — garbage, from changing one parameter. "When you're building applications on top
of these large language models, you've got to be very, very careful with these parameters"
(≈59:59–1:00:45).

## Part 9 — Tokenization: byte pair encoding

### What the earlier pipeline loses

Every model so far, homeworks included, vectorized text with the STIE pipeline of [lecture 5](05-text-vectorization-and-bag-of-words.md):
standardize, tokenize, index, encode (slide 59; ≈1:01:31; see [text vectorization](text-vectorization.md)).
Standardizing strips punctuation, lowercases everything, sometimes removes stop words and stems. Yet GPT
keeps punctuation and case, and "you can actually make up a word as part of your question, and it'll use
that word consistently in the output". The lecturer defined "reldoh", "hodler backwards", as "a student
who understands deep learning backwards", and asked for a sentence. He got one in which "Jane was a true
reldoh, effortlessly explaining even the most complex deep learning concepts in reverse order"
(≈1:02:21). So the disadvantages to fix are keeping punctuation, keeping case and handling new words
(≈1:03:10).

### Characters, words and sub-words

Modern models use other tokenization schemes: the GPT family uses **Byte Pair Encoding** (BPE), BERT
uses **WordPiece** (slide 60; ≈1:03:10). The common idea starts from individual **characters**, which
can compose any word: "reldoh" is six character tokens. But characters alone give the model no
information about words, so it would have to learn every word from scratch (≈1:03:58). Frequent
**whole words** should be in the vocabulary too. Infrequent ones are not worth an embedding vector each,
and can be built on the fly from smaller pieces. In between come **sub-words**, word fragments frequent
enough to be worth adding, like "ize" in "standardize" and "normalize" (≈1:04:44). So BPE's vocabulary
ends up as characters, full words that occur frequently enough to be worth adding, and sub-words that
occur frequently enough to be worth adding (slide 60). "Characters" includes upper and lower case
letters, digits and punctuation, all atomic units (≈1:05:30).

### BPE worked through

"Start with each character as a token, and 'merge' tokens that most frequently occur next to each other.
Don't merge across whitespace. Stop when the size of your vocabulary reaches a user-defined limit (we will
assume 12 in the example below)" (slide 61; ≈1:05:30). The training corpus is the single sentence "The cat
sat on the mat". GPT would not lowercase it, but for simplicity the example standardizes it to "the cat
sat on the mat", written as characters with "_" for each space (slides 61–62; ≈1:05:30–1:06:16). The
starting vocabulary is the nine distinct letters [t] [h] [e] [c] [a] [s] [o] [n] [m]. Then, step by step
(slides 62–64; ≈1:06:16–1:09:23):

| Step | Adjacent pairs and counts | Merge | Vocabulary after it |
| --- | --- | --- | --- |
| 1 | [t,h] 2, [h,e] 2, [c,a] 1, [a,t] 3, [s,a] 1, [o,n] 1, [m,a] 1 | a,t → at | the nine letters + [at] |
| 2 | [t,h] 2, [h,e] 2, [c,at] 1, [s,at] 1, [o,n] 1, [m,at] 1 | t,h → th | + [at] [th] |
| 3 | [th,e] 2, [c,at] 1, [s,at] 1, [o,n] 1, [m,at] 1 | th,e → the | + [at] [th] [the] |

"a, t shows up three times, in mat, sat, and in cat" (≈1:07:03), so it is merged first. When several
pairs tie, "just pick one randomly", here t,h (≈1:08:35). After each merge the new token is added to the
vocabulary and the corpus is rewritten with it. After the third merge the corpus reads "the, _, c, at,
_, s, at, _, o, n, _, the, _, m, at". The vocabulary now holds nine letters and three merged tokens,
which is the limit of 12. The
algorithm "came from someone trying to figure out a way to compress data": if "ABC" shows up all the
time, send it as one token instead of three (≈1:07:50–1:08:35).

In the real models the process continues until the vocabulary reaches its preset size: about 50,000 for
GPT-2 and GPT-3 and about 100,000 for GPT-4 (≈1:09:23–1:10:08). Slide 61's footnote says "GPT-2/3 and
GPT-3.5/4 appear to have a vocab size of ~50,000 and ~100,000 respectively".

**Tokenizing new text.** "When a new piece of text arrives, the BPE tokenization will apply the merges in
the same order" (slide 65; ≈1:10:08–1:10:54). For "the rat", written [t,h,e,_,r,a,t], first a,t → at
gives [t,h,e,_,r,at], then t,h → th gives [th,e,_,r,at], then th,e → the gives [the,_,r,at]. The tokens
that go into the model are "the", the space, "r" and "at".

### The token visualizer

The lecturer then tokenizes text in a GPT tokenizer visualizer at observablehq.com (slide 66;
≈1:10:54–1:14:23). In "Hands-on deep learning", uppercase "H" is its own token, number 39, the dash and
"on" are tokens, and "deep" and "learning" each come as one token with the space in front of it.
"Deep" with a leading space is a different token from "deep" without one. Most words arrive after a space,
so attaching the space to the start of the word saves compute (≈1:11:39). Case matters too: "Deep" and
"deep" are different tokens. Uppercase "The" with a space is token 383, lowercase "the" is 262, and
"the" without any space is different again (≈1:12:25–1:13:12). "Jane" is one token. "Rama" is not, though lowercase "rama" is
("I have no idea what they were scraping"). Typed again later in the text, "Jane" splits into "J"
with its space and "ane" (≈1:13:12–1:13:59). He suggests trying your own name. The tokens' spelling is as he reads them off the
screen; the visualizer is not in the deck.

Slide 67, a sidebar on coding BPE from scratch, is not mentioned in the video. It links Andrej
Karpathy's video "Let's build the GPT Tokenizer" and his GitHub repository minbpe, behind two "Link"
hyperlinks whose targets are not printed.

## Printed slips

- Slide 11's loss formula has a factor of $1/7$ and seven terms from "the" to an end marker `<E>`, while
  its diagram shows five output words (cat, sat, on, the, mat). Neither slide nor narration reconciles
  them.
- The "Dense (ReLU)" box on slides 9–12, 14–16, 30–31, 34, 36 and 43: the next lecture corrects it
  to "a Dense layer with linear activations (not ReLU) followed by a Softmax" ([lecture 10](10-adapting-llms.md),
  slide 3). See Part 1.
- Slide 43 prints "TCE\*" with no footnote for the asterisk.
- Slide 61 prints the starting corpus as "[t,h,e,\_,c,a,t,\_,s,a,t,\_o,n,\_t,h,e\_m,a,t]", with commas
  missing; slides 63–64 repeat it in full. Their "corpus after 1st merge" also reads "t,h,e\_,m".
- Slide 41's OCW notice credits Andrej Karpathy, but the upper of its two screenshots is the header of
  Jay Mody's blog post.
- Spoken: "night had a probability of 0.3" (≈38:07), where slide 44 gives 0.3 to "rainy"; "the transformer
  encoder is also referred to sometimes as the transformer decoder" (≈15:39), where slide 28 says the
  Transformer *Causal* Encoder; the probabilities "add up to 0" (≈54:29), meant as 1; Fashion MNIST's "10
  digits" (≈9:24), whose ten classes are clothing items.

## Related pages

- [Lecture 10](10-adapting-llms.md): the second LLM lecture: the dense-layer correction, instruction
  tuning and ChatGPT, prompting, retrieval-augmented generation, fine-tuning and LoRA.
- [Large language models](large-language-models.md): next word prediction, causal self-attention, the
  Transformer Causal Encoder, generating text and GPT-3, gathered as a topic.
- [Decoding and sampling](decoding-and-sampling.md): greedy decoding, random sampling, top-K, top-P and
  temperature.
- [Text vectorization](text-vectorization.md): the STIE pipeline, and byte pair encoding, which replaces
  it for LLMs.
- [Transformers](transformers.md): self-attention, the encoder, and its causal version.
- [Self-supervised learning](self-supervised-learning.md): masking, and next word prediction as the
  second task.
- [Multi-class classification and softmax](multiclass-classification.md): the softmax that temperature
  modifies.
- [Loss functions](loss-functions.md): categorical cross-entropy, averaged over positions here.
- [Natural language processing](natural-language-processing.md): sequence classification, labeling and
  generation.
- [Lecture 8](08-transformers-and-self-supervised-learning.md): the encoder, masking and BERT, which this
  lecture builds on.
