# CLIP: one embedding space for text and images

**CLIP** (Contrastive Language-Image Pretraining) is a model OpenAI built to compute embeddings for text
and for images that live in the same "concept space". The embedding of a caption lands close to the
embeddings of the images it describes, and far from those of unrelated images. The course teaches it in
[lecture 11](11-diffusion-models.md) (slides 46–73; ≈39:18–54:02) as the piece that lets a
[diffusion model](diffusion-models.md) follow a text prompt. The paper is "Learning Transferable Visual
Models From Natural Language Supervision" by Alec Radford, Jong Wook Kim and others
(<https://arxiv.org/pdf/2103.00020.pdf>; slide 57). Citations are to lecture 11.

## The problem it solves

To steer image generation with a prompt, we want "an embedding for any text prompt that is 'close' to the
embeddings of the images that correspond to that prompt" (slide 46). Computing *some* embedding of each
is easy with "standard architectures": a text encoder such as BERT, and an image encoder such as ResNet,
whose penultimate layer represents the image well (slide 48; ≈40:04; see
[transfer learning](transfer-learning.md)). But encoders trained separately give embeddings with no
reason to be related: "They were not trained together, so there's no basis for them to be related"
(≈40:49).

So the embeddings must meet **two requirements** (slides 49–51):

1. For an image and a caption that describes it, the embeddings should be **close**.
2. For an image and an unrelated caption, the embeddings should be **far** apart.

Together these make the two embeddings "refer to the same underlying concept", that is, "live" in the
same "concept space" (slides 52–53). **Closeness** is measured by **cosine similarity** (slides 54–55), as
for [word embeddings](word-embeddings.md) and for the text chunks of
[retrieval-augmented generation](retrieval-augmented-generation.md).

## How CLIP is trained

The steps are on slides 58–61 (≈43:11–44:43):

1. Set a 12-layer, 8-head **Transformer Causal Encoder** stack as the text encoder (see
   [large language models](large-language-models.md)), and **ResNet-50**, with its top chopped off, as
   the image encoder ("with some small tweaks").
2. Initialize both with random weights.
3. Grab a batch of `<image, caption>` pairs.
4. Run the images through the image encoder and the captions through the text encoder to get embeddings.
5. Calculate the "closeness score", the cosine similarity, for **every** image-caption pair in the batch.

Lay the scores out as a grid, with images across the top and their captions down the side in the same
order (slides 62–63). The **diagonal** cells are the matching pairs, coloured green, and their scores
should be as high as possible (slide 64). The **off-diagonal** cells are mismatched pairs, coloured red,
and theirs should be as small as possible (slide 68).

Why both? A loss that only rewards the green cells, "minimize [– sum of green cells]" (slide 65), fails.
"The optimizer can simply ignore the input, make all the embeddings the same and achieve a perfect cosine
similarity of 1.0 for all pairs of embeddings!" (slide 67). The lecturer calls this "model collapse"
(≈46:20). The fix is slide 69's loss: "minimize [sum of red cells – sum of green cells]". In the notation
of the CLIP paper's figure (slide 70), a batch of $N$ pairs has text embeddings $T_1, \dots, T_N$ and image
embeddings $I_1, \dots, I_N$, with $T_i$ the caption of image $I_i$. The slide's loss is then

$$\text{loss} = \sum_{i \ne j} \cos(T_i, I_j) - \sum_{i} \cos(T_i, I_i)$$

where $\cos$ is cosine similarity. The symbols are this page's rendering of slide 69's words. Each batch's
loss is backpropagated through both encoders, batch after batch (≈47:07; see
[loss functions](loss-functions.md) and [backpropagation](backpropagation.md)). Lecture 8's unpresented slides on contrastive learning for images describe the same pattern: the
representations of matching pairs should be close, and far from those of the other pairs (see
[self-supervised learning](self-supervised-learning.md#images-contrastive-learning)).

CLIP was trained this way on **400 million** image-caption pairs scraped from the Internet (slide 71). The
captions came from the **alt text** people attach to images on the web, often for accessibility. It is
more verbose than a typical caption, which is better: "there's more stuff for the model to learn from"
(≈47:52–48:39).

## Using it

**Its text encoder alone.** After training, "we can send in any text and get an embedding that is close
to the embedding of any image described by that text" (slide 72). A diffusion model takes that embedding
of the prompt as an extra input, which is how text-to-image models are steered (slides 74–81; see
[diffusion models](diffusion-models.md#steering-with-a-text-prompt)).

![Slide 72: "We can use CLIP's text encoder by itself", above a Text encoder box producing a blue embedding, circled in red dashes, and an Image encoder box producing a yellow one.](../raw/images/11-diffusion-models/slide-72.png)

*Lecture 11, slide 72 — the half of CLIP that a text-to-image model uses. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

**Zero-shot image classification** (slide 73; ≈49:25–54:02). An ordinary classifier has a fixed set of
labels, and a new class means retraining. With CLIP, embed each candidate label with the text encoder,
$T_1$ to $T_n$ for $n$ labels, and the image with the image encoder, $I$. Compute the cosine similarities,
and pick the label with the highest. The labels can change every time: "an insanely flexible image
classification system". The CLIP paper's figure on slide 73 phrases each class name as "A photo of a
{object}." and picks "A photo of a dog."

**Multimodal models.** The lecturer calls CLIP "a breakthrough": one of the first ways to give very
different data types embeddings "that all happen to live in the same space". "If you look at the modern
multimodal large language models, they are all based on the same exact idea" (≈55:37–56:22). Later models
such as BLIP and BLIP 2 build on it (≈42:24). Compare lecture 8's multi-modal transformer, which puts text,
image and table embeddings into one stack (see [transformers](transformers.md#beyond-text)).

## Where this comes up

- [Lecture 11](11-diffusion-models.md): the requirements, CLIP's training and loss, zero-shot
  classification, and conditioning a diffusion model on the CLIP embedding of a prompt (Parts 9–12).
- [Lecture 8](08-transformers-and-self-supervised-learning.md): contrastive learning for images, on
  unpresented slides, and multi-modal transformers.
- See also [diffusion models](diffusion-models.md), [word embeddings](word-embeddings.md),
  [representation learning](representation-learning.md) and
  [AI, machine learning, deep learning and generative AI](ai-ml-dl-genai.md).
