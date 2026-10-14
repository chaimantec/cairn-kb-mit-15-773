# See also — related knowledge bases

Other Cairn knowledge bases whose material genuinely bears on 15.773. Each entry says what that KB
is good for and what it does not cover, so you can decide *before* spending a fetch on it.

Pass the **repo URL** as the `kb` argument to `kb_read` / `kb_list`. The
`raw.githubusercontent.com` links are direct `web_fetch` targets that return plain markdown.

---

- **MIT 6.7960 — Deep Learning** (MIT, Phillip Isola, Sara Beery and Jeremy Bernstein, Fall 2024).
  Partial: lectures 1–19 of its 24 catalog recordings.
  KB: `https://github.com/chaimantec/cairn-kb-mit-6-7960` — pass this as `kb`.

  The mathematical counterpart to 15.773, which deliberately "skips the math". Where 15.773
  lecture 1 asks students to take on faith that activations must be nonlinear, and leaves training
  to the next deck, 6.7960 gives the graduate-level treatment. See its
  [multilayer perceptron](https://raw.githubusercontent.com/chaimantec/cairn-kb-mit-6-7960/main/wiki/multilayer-perceptron.md),
  [activation functions](https://raw.githubusercontent.com/chaimantec/cairn-kb-mit-6-7960/main/wiki/activation-functions.md),
  [gradient descent](https://raw.githubusercontent.com/chaimantec/cairn-kb-mit-6-7960/main/wiki/gradient-descent.md),
  [backpropagation](https://raw.githubusercontent.com/chaimantec/cairn-kb-mit-6-7960/main/wiki/backpropagation.md)
  and [representation learning](https://raw.githubusercontent.com/chaimantec/cairn-kb-mit-6-7960/main/wiki/representation-learning.md)
  pages. For 15.773 lecture 4, its
  [convolution](https://raw.githubusercontent.com/chaimantec/cairn-kb-mit-6-7960/main/wiki/convolution.md),
  [transfer learning](https://raw.githubusercontent.com/chaimantec/cairn-kb-mit-6-7960/main/wiki/transfer-learning.md)
  and [data augmentation](https://raw.githubusercontent.com/chaimantec/cairn-kb-mit-6-7960/main/wiki/data-augmentation.md)
  pages give the more formal treatment. Its notation and depth differ: use 15.773's own pages first
  for this course's framing and examples.
  Start at [INDEX](https://raw.githubusercontent.com/chaimantec/cairn-kb-mit-6-7960/main/INDEX.md).

- **CS224N — Natural Language Processing with Deep Learning** (Stanford, Christopher Manning,
  Spring 2024). Complete: all 23 lectures.
  KB: `https://github.com/chaimantec/cairn-kb-cs224n` — pass this as `kb`.

  A second, more technical treatment of the language half of 15.773 (embeddings, transformers,
  pretraining, fine-tuning), which this KB covers only as far as lecture 5's bag of words. For
  lecture 5, its
  [subword modeling](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs224n/main/wiki/subword-modeling.md)
  page explains the byte-pair encoding that lecture 5 defers, and its
  [lecture 1](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs224n/main/wiki/01-intro-and-word-vectors.md)
  page explains why one-hot word vectors fall short. For lecture 1's material it has an
  [activation functions](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs224n/main/wiki/activation-functions.md)
  page and derives neural-network gradients by hand. Its examples come from NLP.
  Start at [INDEX](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs224n/main/INDEX.md).
