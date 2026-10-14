# Natural language processing

**Natural language processing (NLP)** is the effort to analyse and make predictions with language.
The course's NLP sequence starts in [lecture 5](05-text-vectorization-and-bag-of-words.md) and, by
the roadmap on its first slide, runs through embeddings (lecture 6), transformers in theory
(lecture 7) and in applications with self-supervised learning (lecture 8), to LLMs (lectures 9–10)
(lecture 5, slide 1; ≈0:00–0:46). This knowledge base so far covers lecture 5 only.

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
   [the Bag of Words model](bag-of-words.md), and lecture 6 with embeddings.
2. **Which network architecture is best for processing text?** Lectures 7 and 8, with transformers.

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
- See also [representation learning](representation-learning.md) and
  [multi-class classification and softmax](multiclass-classification.md), whose softmax lecture 3
  connects to how GPT-4 emits words.
