# Natural language processing

**Natural language processing (NLP)** is the effort to analyse and make predictions with language.
The course's NLP sequence starts in [lecture 5](05-text-vectorization-and-bag-of-words.md) and, by
the roadmap on its first slide, runs through embeddings (lecture 6), transformers in theory
(lecture 7) and in applications with self-supervised learning (lecture 8), to LLMs (lectures 9–10)
(lecture 5, slide 1; ≈0:00–0:46). Lecture 7's own roadmap calls lectures 7 and 8 "Transformers (1/2)"
and "(2/2)" (lecture 7, slide 1). This knowledge base so far covers lectures 5 to 9.

## Why text

Human knowledge, the internet, human communication and cultural production are all "(mostly) natural
language text", and text is also "the way in which we think and communicate". A system that could
read and "understand" all of it, all of the medical literature for instance, would be extraordinary
(slide 2; ≈0:46–2:18). NLP is already everywhere: by Google's account, autocomplete saves 200 years of
typing a day and "made mobile possible" (slide 3; ≈2:18–3:50).

## Text in, text out

Even the simplest formalism, text in and text out, covers "an enormous, enormous range of
applicability" (slides 5–14; ≈3:50–7:46): classifying text (sentiment, routing, intent, filtering),
extracting data from free-form text (company financials from news, codes from doctors' notes),
summarizing, generating (marketing copy, sales emails, "troublingly for educators, college
application essays"), generating code, "because code is just text", and answering questions about a
set of documents, which is how chatbots are built. Call centers are where "a lot of money is being
spent right now".

## Representing text

Like most of deep learning, an NLP task is "fancy regression": an input $x$ goes through a deep
network $f(x, w)$ with weights $w$ to an output $y$. Here $x$ is text and $y$ can be text, labels or
numbers (slides 22–23; ≈12:23–13:10). Two questions follow (slides 24–25; ≈13:10–13:57):

1. **How do you represent $x$?** Images could be fed in as pixel values; a sentence cannot.
   Lecture 5 answers with [text vectorization](text-vectorization.md) and
   [the Bag of Words model](bag-of-words.md), and [lecture 6](06-word-embeddings.md) with
   [word embeddings](word-embeddings.md).
2. **Which network architecture is best for processing text?** Lectures 7 and 8, with transformers.

## From one-hot vectors to contextual embeddings

Lecture 5's one-hot vectors are as long as the vocabulary and say nothing about meaning: any two of
them are the same distance apart. [Lecture 6](06-word-embeddings.md) replaces them with
[word embeddings](word-embeddings.md), short dense vectors learned from data in which related words
sit close together and directions carry relationships. GloVe learns them from how often words share
a sentence in Wikipedia. These are **stand-alone** embeddings, one vector per word whatever the
sentence, so a word with several meanings, like "bank", gets an average of them. **Contextual**
embeddings take the surrounding words into account, and "the key to calculating contextual word
embeddings" is the transformer, the subject of lectures 7 and 8 (lecture 6, slides 22–25;
≈12:23–14:48).

[Lecture 7](07-transformers.md) builds the [transformer](transformers.md) encoder: self-attention
for context, several attention heads, feed-forward layers, and position embeddings for word order.
Its task is one where every word needs its own answer: **slot filling** on the ATIS travel queries,
tagging each word of "I want to fly from boston at 7 am …" with one of 123 slots such as
B-fromloc.city_name, so that a natural-language query can be turned into a structured one such as
SQL (lecture 7, slides 3–13; ≈1:33–10:05).

[Lecture 8](08-transformers-and-self-supervised-learning.md) completes the encoder and names three common uses of it (slide 30): **sequence
classification**, one label for a whole text, such as a review's sentiment or the department a
call-center email should go to; **sequence labeling**, one label per word, such as slot filling; and
**sequence generation**, continuing the text, which [lecture 9](09-large-language-models.md) takes up. For classification an
artificial `<CLS>` token at the start of the sentence collects the sentence's meaning (slide 38). Then
[self-supervised learning](self-supervised-learning.md) pretrains an encoder on unlabeled text by
masking words and predicting them, which gives **BERT** (slides 68–80). For any classification or
labeling problem on text, a model like BERT can be fine-tuned on "a few hundred" labelled examples, and
for a "standard" NLP problem a pretrained model from a hub may need no fine-tuning at all (slide 85). In
the Colab, the Hugging Face `pipeline` function classifies sentiment, extracts named entities and
answers a question about a letter, out of the box (≈1:01:27–1:06:53).

[Lecture 9](09-large-language-models.md) completes the trio with **sequence generation**. A transformer with causal
self-attention, trained on vast amounts of text to predict the next word, becomes an autoregressive
language model such as GPT-3. It generates text by predicting a word, appending it to the input and
predicting again (slides 26–40). The lecture's other two parts are how the next word is chosen from the
softmax, and how byte pair encoding splits text into tokens (slides 42–66). See
[large language models](large-language-models.md).

## The arc of NLP progress

Lecture 5 tells the history in four eras (slides 19–21; ≈10:06–12:23). **Hand-crafted rules** written
by linguists (to the early 1990s) gave way to **statistical machine learning** (1990s to early 2010s),
which simply counted things, such as how often words co-occur, and beat the linguists' rules, hence
Frederick Jelinek's "Every time I fire a linguist, the performance of the speech recognizer goes up".
Deep learning arrived around 2012 and brought **recurrent neural networks**, and in 2017 the
**transformer** "replaced everything else across the board". The course leapfrogs straight to
transformers and spends no time on RNNs, though work to revive RNNs is under way and "still very early
days".

![Slide 21: four dark red boxes joined by arrows — "Hand-crated rules based on linguistics (up to early 1990s)", "Statistical/ML (1990s - early 2010s)", "Recurrent Neural Networks (2014-2017)", "Transformers (2017 - Current)" — with a brace under the first two above the Jelinek quote, and the Transformers box circled in red dots above "We will leapfrog to this in HODL!"](../raw/images/05-text-vectorization-and-bag-of-words/slide-21.png)

*Lecture 5, slide 21 — four eras of NLP, and the course's leap to transformers. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec05.pdf)*

## Beyond English

Asked whether all this is only English, the lecturer says there are models specialized to other
languages and truly multilingual models, and many modern LLMs are trained from the start on several
high-resource languages. He adds that Google's Gemini 1.5 Pro, given a single grammar manual for a
language spoken by about 200 people, translated into it at human-level proficiency without having
been trained on it (lecture 5, ≈7:46–9:19).

## Where this comes up

- [Lecture 5](05-text-vectorization-and-bag-of-words.md): why NLP, its applications, the arc of
  progress, text vectorization and the Bag of Words model.
- [Lecture 6](06-word-embeddings.md): word embeddings, GloVe, and why contextual embeddings need
  transformers.
- [Lecture 7](07-transformers.md): transformers for search queries, the ATIS slot-filling task, and
  the transformer encoder.
- [Lecture 8](08-transformers-and-self-supervised-learning.md): sequence classification, labeling and generation, the `<CLS>` token,
  self-supervised pretraining, BERT, and Hugging Face pipelines for sentiment, named entities and
  question answering.
- [Lecture 9](09-large-language-models.md): sequence generation with causal language models, decoding, and byte pair
  encoding.
- See also [large language models](large-language-models.md), [word embeddings](word-embeddings.md), [transformers](transformers.md), [representation learning](representation-learning.md) and
  [multi-class classification and softmax](multiclass-classification.md), whose softmax lecture 3
  connects to how GPT-4 emits words.
