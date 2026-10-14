# Transfer learning

**Transfer learning** is customizing a network that someone else has already trained to your own
problem, rather than designing and training a network from scratch
([lecture 4](04-cnns-and-transfer-learning.md), slide 64; ≈1:01:37). The lecturer calls it "the
norm by which people do these things" (≈49:08). It is what lets the course classify handbags and
shoes from fewer than 100 training images.

## Why it works: two research trends

Transfer learning rests on two trends (slides 62–63; ≈1:00:05–1:01:37). First, researchers have
designed architectures matched to types of data. Slide 62's table gives residual connections for
all types, convolutional layers for images, and transformers for sequences such as natural
language, audio, video and gene sequences. Second, they have used those architectures to train
high-performance networks on vast real-world data sets, and have made many of them publicly
available.

A network that does well on a large collection of everyday images has built "a smart, hierarchical
representation of these objects across its layers": colour gradations and lines in its first
layer, edges in the second, honeycomb shapes in the third, and eventually things like a torso or a
Labrador retriever (slides 67–69; ≈1:06:15–1:07:02). Those layers hold knowledge that is useful
well beyond the task the network was trained for. See
[convolutional neural networks](convolutional-neural-networks.md) and
[representation learning](representation-learning.md).

## Choosing a pre-trained network

Look for a network trained on data like yours. Handbags and shoes are everyday objects, as opposed
to MRIs or X-rays, so look for networks trained on everyday images (slide 65; ≈1:01:37). The first
thing to check, of course, is whether someone has already built the exact classifier you need and
put it on GitHub. **ImageNet** is a database of millions of images of everyday objects in 1,000
categories. Slide 66 gives 1.2 million training images and 100,000 test images. **ResNet** is a
family of networks trained on ImageNet that did very well in its competition (slide 70;
≈1:07:49). The slides draw ResNet 34.

## Headless ResNet as a feature extractor

ResNet classifies into 1,000 categories, and the handbags problem has two, so it cannot be used as
is (slide 71). Instead, take ResNet and **stop just before its final layer**, the 1,000-way fully
connected layer. What comes out of this **"headless" ResNet** is a smart representation of the
image, as if it had been tagged with lots of general tags (slides 72–73; ≈1:08:37–1:09:22). Feed
that representation to a small network of your own, here one hidden layer and a sigmoid output.
Because its inputs are at a much higher level of abstraction than raw pixels, it can learn from very
few examples (slides 74–75).

![Slide 75 — a small network on top of headless ResNet](../raw/images/04-cnns-and-transfer-learning/slide-75.png)

*Lecture 4, slide 75 — the output of headless ResNet becomes the input layer of a small one-hidden-layer network.*

In the course's Colab (≈1:10:54–1:14:01; the notebook is not on OCW), Keras downloads ResNet with
`include_top=False`, "the very final layer of the thing, don't give it to me". The model has about
23 million parameters. Every image is run through it **once**, and its output, a $7 \times 7 \times 2{,}048$
tensor per image, is stored. The small model then flattens that tensor and applies a dense layer of
256 ReLU units, a dropout layer and a sigmoid. It trains in moments, because ResNet's outputs are
not recomputed every epoch, and scores **100%** on the test set, against 87% for a CNN trained from
scratch on the same 98 images. It then correctly classifies a shoe and a handbag held up to the
webcam in class.

## Fine-tuning

You can get fancier: connect headless ResNet directly to your small network and train **the whole
network end to end**. This is **fine-tuning**. You must start the training from the weights and
biases that came with ResNet, "the crown jewel that's been learned", not from scratch (slide 77;
≈1:09:22–1:10:09). The course leaves it to homework 1.

## Where to find pre-trained models

Pre-trained models are on the TensorFlow Hub, the PyTorch Hub and the Hugging Face Hub (slides
78–80). The Hugging Face Hub listed 505,594 models on February 13, 2024. The lecturer recalls about
50,000 a year earlier (≈1:10:09).

One caution came up in questions: weights that everyone can download carry "some adversarial risk"
(≈1:10:09–1:10:54).

## Pretrained word embeddings

The same logic carries over to text. [Lecture 6](06-word-embeddings.md) downloads GloVe's word
embeddings, trained on Wikipedia, and builds a song-genre classifier on top of them, first frozen and
then trained further (≈1:02:07, ≈1:09:04). The lecturer draws the parallel with handbags and shoes
himself: "when your data is really small, try to use a pre-trained model" (≈1:07:32). With about
50,000 songs, though, embeddings learned from scratch did best. A general-purpose tool like GloVe is
"a good starting point" when you do not have a lot of data, "but when you have a lot of data, you
should always try to do your own thing and see if it's any better" (≈1:16:47). See
[word embeddings](word-embeddings.md#pretrained-or-your-own).

## Where this comes up

- [Lecture 4](04-cnns-and-transfer-learning.md): the idea, ResNet, the Colab and the live demo.
- [Lecture 6](06-word-embeddings.md): pretrained GloVe embeddings, frozen and fine-tuned.
- See also [data augmentation](data-augmentation.md), the other remedy for a small data set.
