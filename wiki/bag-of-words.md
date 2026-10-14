# The Bag of Words model

[Text vectorization](text-vectorization.md) turns a sentence into a table, one one-hot row per word.
A network wants a single vector of fixed length, so the rows have to be combined. Combining them by
adding or OR-ing gives the **Bag of Words** model: the bag records which words occur, and how often,
and nothing about their order ([lecture 5](05-text-vectorization-and-bag-of-words.md), slides 45–57;
≈28:45–35:03).

## Why not flatten the table?

Stacking the table's rows or columns into one long vector, as was done with images, has two problems
(slides 51–53; ≈31:12–32:45). It is far too long: with a 50,000-token vocabulary, a six-word sentence
becomes a 300,000-long vector, and connecting that to a 100-unit hidden layer takes
$300{,}000 \times 100$, 30 million, weights. "Too much. Can't learn anything." And its length varies
with the sentence, while a network's inputs all have to be the same length.

## Count encoding and multi-hot encoding

Aggregating the one-hot rows solves both problems, because the result is always one vector as long
as the vocabulary (slides 54–57; ≈32:45–34:18):

- **Count encoding** sums the rows. Each position holds the number of times its word occurs. The
  lecturer also calls it sum encoding.
- **Multi-hot encoding** ORs the rows. Each position holds 1 if its word occurs at all, 0 if not.

"The cat sat on the mat" contains "the" twice. Count encoding puts a 2 in the column for "the" and
multi-hot encoding a 1; every other word in the sentence gets a 1 in both (slide 56).

![Slide 56: the hand-drawn 6 × 100 one-hot table for "the cat sat on the mat", and beneath it two single-row boxes: "Count Encoding" with 1s under cat, mat, on and sat and an orange 2 under "the", and "Multi-hot Encoding", identical except for an orange 1 under "the".](../raw/images/05-text-vectorization-and-bag-of-words/slide-56.jpg)

*Lecture 5, slide 56 — count encoding sums the one-hot rows; multi-hot encoding ORs them. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec05.pdf)*

## What the bag throws away

"The cat sat on the mat" and "the mat sat on the cat" have exactly the same count and multi-hot
encodings. "That's a huge limitation. But shockingly, for many applications, it doesn't matter. It's
good enough" (≈35:03). The model's shortcomings (slides 58–61; ≈35:03–38:55):

- It loses **sequentiality**, the meaning carried by word order.
- The vectors are **sparse**, mostly zeros.
- **Every input is as long as the vocabulary**, the collected works of Shakespeare and "I love you"
  alike, so short inputs waste compute.
- Keeping only the most frequent words shortens the vector, but words outside the vocabulary all
  become [`<UNK>`](text-vectorization.md#3-index-the-vocabulary-and-unk) and can no longer be told
  apart. A long vocabulary, on the other hand, means more weights to learn, more compute and more
  risk of overfitting (slide 61).

It is still "a really good default for many NLP tasks", and the course's rule is to do the simple
thing first (≈38:55).

## N-grams: putting some order back

Order matters. "Kate Winslet's performance as a detective trying to solve a terrible crime in a small
Pennsylvania town is anything but disappointing" is a positive review made of negative-looking words:
"terrible" belongs to the crime, and "anything but" flips "disappointing" (≈1:04:29–1:05:15).

**Bigrams** add every pair of adjacent words to the tokens, so "the cat sat on the mat" yields "the",
"the cat", "cat", "cat sat", "sat", and so on; three words are **trigrams**, and in general
**n-grams** (≈1:05:15–1:06:03). Pairs multiply the possibilities, 25 million from a 5,000-word
vocabulary, though most never occur. The course's bigram model raises the vocabulary from 5,000 to
20,000 tokens. N-grams are also slower to build: "Bigrams, trigrams, all of them will get much more
compute-intensive" (≈1:06:03–1:07:41). In Keras this is the text vectorization layer's `ngrams`
setting.

## Results on song lyrics

Lecture 5's Colab classifies about 90,000 songs as hip-hop, rock or pop from their lyrics, with a
network of one hidden layer of eight ReLUs and a three-way softmax (slide 65). Rock is 55% of the
songs, so always guessing rock is the baseline (≈53:38).

| Model | Vocabulary | Epochs | Test accuracy |
| --- | --- | --- | --- |
| Baseline: always rock | — | — | 55% (rock's share of the songs) |
| Single words, multi-hot | 5,000 tokens | 10 | 72% (87% train, 73% validation) |
| Single words and bigrams, multi-hot, with dropout | 20,000 tokens | 3 | 75% |

The figures are from ≈53:38, ≈1:03:44 and ≈1:10:00. The lecturer calls the jump from 72% to 75% "a
meaningful jump, just by using bigrams", and suggests ten epochs might do better.

## Where this comes up

- [Lecture 5](05-text-vectorization-and-bag-of-words.md): count and multi-hot encoding, the model's
  shortcomings, bigrams, and the song-genre classifier.
- See also [text vectorization](text-vectorization.md) and
  [natural language processing](natural-language-processing.md).
