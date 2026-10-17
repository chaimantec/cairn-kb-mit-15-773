# Lecture 11 — Diffusion Models for Image Generation (Text-to-Image)

**Lecturer:** Rama Ramakrishnan (the captions label him only "PROFESSOR"; see the
[transcript header](../raw/transcripts/11-diffusion-models.md)) ·
**Video:** [youtube.com/watch?v=NQBhhRG-Pe4](https://www.youtube.com/watch?v=NQBhhRG-Pe4) (75 min) ·
**Slides:**
[`mit15_773_s24_lec11.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)
(90 pages; transcribed in [`raw/slides/11-diffusion-models.md`](../raw/slides/11-diffusion-models.md)) ·
**Transcript:** [`raw/transcripts/11-diffusion-models.md`](../raw/transcripts/11-diffusion-models.md)

## What this lecture establishes

This is the course's last lecture ("Wednesday is going to be project presentations", 0:00). It explains
how a text-to-image model such as Midjourney, or the text-to-video model Sora, makes a picture. OpenAI
describes Sora as "text-conditional diffusion models" built on "a transformer architecture" (slide 6). The
course had already covered transformers, so "today's class is really about text conditional diffusion
models, the other building block" (≈2:26). The lecture has two parts (slide 7): how to get a model to
generate a good image at all, and how to "control" what it generates with a text prompt.

**Part one** is the **diffusion model**. Nobody knows how to train a model to turn random noise into a
photograph of Killian Court. But the reverse is easy: add random numbers to every pixel. So take each
training image and make a sequence of noisier and noisier copies of it. Each copy, paired with the
slightly *less* noisy copy before it, is one training example $(x, y)$. A **de-noising** network trained
on those pairs by stochastic gradient descent can remove a little noise from anything. To generate,
start from pure noise and denoise it again and again until an image "has emerged from the fog"
(slides 8–35; ≈3:13–18:40). In practice the network predicts the noise and subtracts it (slide 37), and
the network is usually a **U-Net**, an image-to-image architecture (slides 38–41).

**Part two** is steering. OpenAI's **CLIP** trains a text encoder and an image encoder together, on 400
million image-caption pairs, so that a caption's embedding lands close to its image's embedding and far
from unrelated ones (slides 42–71). Concatenate the CLIP text embedding of each training image's caption
with the denoiser's input, and train as before. Then pure noise plus a prompt, denoised repeatedly,
becomes an image of what the prompt describes (slides 74–81; ≈54:10–56:22). The latent diffusion model
behind Stable Diffusion adds two refinements: it weaves the prompt in through attention, and it denoises
small embeddings instead of pixels, which makes it much faster (slides 82–85). The lecture ends with
Hugging Face's Diffusers library and computer-vision pipelines in Colab (slides 88–89).

The catalog titles the video "Generative AI – Text-to-Image Models". The deck is "Lecture 11: Diffusion
Models for Image Generation (Gen AI)" (slide 1). The recording presents slides 2–89 in order, with long
question periods at ≈19:25–27:51 and ≈56:22–1:00:59. Slide numbers here are PDF pages. From page 49 on, the
deck prints a number one higher than the page.

## Part 1 — Text-to-image and text-to-video

The lecturer opens with an image he generated "just for kicks last night". He asked ChatGPT for "a
photorealistic image of graduate students in a class on deep learning". It is not bad, apart from "a
noticeable absence of an instructor" and students "facing in various directions" (slide 2; 0:00–≈0:48).

![Slide 2: a ChatGPT conversation. The prompt "Create a photorealistic image of graduate students in a class on deep learning", and ChatGPT's generated image of a sunlit lecture hall full of students at laptops, facing a glowing network diagram and equations on a digital board, with no instructor in sight.](../raw/images/11-diffusion-models/slide-2.jpg)

*Slide 2 — the opening example: ChatGPT (DALL-E 3) given the lecturer's prompt; note the missing instructor. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

Slide 3 shows a Midjourney text-to-image model, credited to a prompt by Alex Serban on MSPowerUser: "A
quaint Italian seaside village with colorful buildings, boats, and the reflection of the setting sun on
the water, in the impressionist style of Claude Monet, with visible brush strokes and dappled light." The
lecturer calls the result "pretty unbelievable" (≈0:48).

![Slide 3: the Monet-style seaside-village prompt at left, an arrow into a dark red "Text-to-Image" box, and at right the generated painting of pastel houses on a hillside above a harbour with two boats and coloured reflections in the water.](../raw/images/11-diffusion-models/slide-3.jpg)

*Slide 3 — a prompt in, a painting out: the text-to-image task this lecture explains. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

On February 15, 2024, OpenAI released **Sora**, a text-to-video model that "can produce a one minute
video from a text prompt" (slide 4; ≈1:33). Given "In an ornate, historical hall, a massive tidal wave
peaks and begins to crash. Two surfers, seizing the moment, skillfully navigate the face of the wave", it
produced the video the class watched (slide 5). Such a thing "has never happened in history, and
therefore, it was not in the training data" (≈1:33).

![Slide 5: a full-slide frame from the Sora video, an enormous curling wave inside a grand hall with a painted vaulted ceiling and lit balconies, two tiny surfers on its face.](../raw/images/11-diffusion-models/slide-5.jpg)

*Slide 5 — a frame of Sora's video for the tidal-wave prompt, a scene that cannot be in any training data. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

Sora's technical report opens with the sentence on slide 6: "we train text-conditional diffusion models
jointly on videos and images of variable durations, resolutions and aspect ratios. We leverage a
transformer architecture that operates on spacetime patches of video and image latent codes". The
[transformer](transformers.md) is familiar by now, which leaves text-conditional diffusion models as the
subject of the lecture (≈2:26).

The plan has two parts (slide 7; ≈2:58–3:13). First, how a model can generate an image from a class of
images at all. Second, how to "control" the generation with a text prompt. The literature also says
condition, steer or control. "That's basically what they mean".

## Part 2 — Generating an image from noise

Say we want a model that generates images of **stately college buildings**, of which "our very own
Killian Court is the finest example" (slide 8; ≈3:13). As always in machine learning, start by collecting
data: slide 9 is a Google Image Search for "stately college buildings", standing in for "millions of
training images" (≈4:00). Two requirements follow. Every image the model produces should be different,
"otherwise it's kind of boring": Killian Court one time, the Rotunda of the University of Virginia
another (slide 10). And every one must still be a stately college building, not "just some random stuff"
(≈4:00–4:45).

The answer, which the lecturer still finds "really astonishing", is to give the model **noise** as its
input: an image "in which all the pixel values are randomly picked" (slide 11; ≈4:45). Noise is different
every time it is drawn, so the output differs every time. And a model trained on stately college
buildings produces stately college buildings, "not … a picture of a Labrador Retriever" (≈5:32).

![Slide 11: two rows, each a grey speckled "Noise" image feeding an arrow into a dark red "Model" box, which outputs a photograph: the MIT Dome in the top row, the Rotunda at the University of Virginia in the bottom row.](../raw/images/11-diffusion-models/slide-11.jpg)

*Slide 11 — the set-up: random noise in, a different college building out each time. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

But how do you train a model to make an image out of pure noise (slides 12–13)? It sounds ridiculous:
"You basically give it a bunch of random numbers and say, give me Killian Court", and one could stop
there as a dead end (≈5:32). The way out is a heuristic from George Pólya's *How to Solve It*, a small
book listing the heuristics mathematicians use. Perhaps the most common is to reverse the question:
"Most of the time, nothing will come out of it. But maybe some other times, something amazing comes out"
(≈6:18). Reversed, the question is: given an image, can we create a noisy version of it (slide 14)? Yes,
"of course, we can do that" (≈7:05).

![Slide 14: the diagram reversed: the MIT Dome photograph at left, an arrow into a dark red "Model" box, and an arrow out to a grey noise image; below, in red, "Given an image, can we create a 'noisy' version of it?"](../raw/images/11-diffusion-models/slide-14.jpg)

*Slide 14 — Pólya's heuristic applied: the image-to-noise direction is the easy one. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

## Part 3 — Adding noise to an image (Colab)

"Just add random numbers to every pixel!" (slide 15). The lecturer shows how in a Colab, which is not
published on OCW: slides 15 and 16 print "(Colab link)" without its target. This section follows the
narration (≈7:05–10:58).

He imports NumPy and the Python Imaging Library, **PIL**, "very handy for image manipulations". He reads
in `killian.png`, a photo of Killian Court uploaded before class, and converts it to a NumPy array. A
colour image is three tables of numbers, one each for red, green and blue, with values from 0 to 255, so
he divides by 255 to put every value between 0 and 1. The array's shape is 411 by 583 by 3 (≈7:52–8:38;
see [images and vision tasks](images-and-vision-tasks.md)).

To add noise, draw a random number for **every pixel**, "not like we sample once", from a normal
distribution with mean 0 and a small standard deviation, and add it to that pixel. The call is
`np.random.normal` with a standard deviation of 0.3 and "as many of these things as the shape of the
image" (≈8:38–9:26). The numbers shift: "The 0.23 has become 0.18, 0.15 has become -0.17". Some values
are now negative and some above 1, so they are **clipped**: values under 0 set to 0, values over 1 set to
1, the rest unchanged (≈9:26–10:13). "That's all it takes to actually add noise to an image. One line of
NumPy."

![Slide 15: three versions of the MIT Dome photo side by side, headed "Original Image", "Slightly Noisy Image" (coloured speckle over a still-recognisable building) and "Very Noisy Image" (uniform grey speckle, the building gone); below, "Just add random numbers to every pixel!".](../raw/images/11-diffusion-models/slide-15.jpg)

*Slide 15 — the same photo with no noise, a little and a lot. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

Put the same line in a loop and raise the standard deviation, 0.3, 0.4, 0.5 and so on, and you get a
sequence from the clean image to a very noisy one (slide 16; ≈10:13).

![Slide 16: a row of eleven small thumbnails of the MIT Dome photo, clean at the left and progressively noisier to the right, the last one washed out to grey.](../raw/images/11-diffusion-models/slide-16.jpg)

*Slide 16 — raising the noise's standard deviation step by step: the raw material for training. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

## Part 4 — Training pairs and the de-noising model

"That suggests an idea!" (slide 17; ≈10:58). Students propose comparing the generated image with the
original through a loss function, and training a network to reconstruct images from their noisy
counterparts ("definitely on the right track", ≈11:44–12:30). Concretely, take each image in the training
set and create many noisy versions of it (slide 18). Then cut that sequence into **input-output pairs**
(slides 19–24; ≈12:30–14:03). The slightly noisy second image is the input $x_1$, and the clean first
image is the output $y_1$. The third image is $x_2$ with the second as $y_2$, and so on, to the noisiest
image $x_{10}$ paired with $y_{10}$, the one before it.

![Slide 20: the row of eleven noisy-Dome thumbnails with two red up-arrows, labelled y1 under the clean first thumbnail and x1 under the slightly noisier second one.](../raw/images/11-diffusion-models/slide-20.jpg)

*Slide 20 — the first training pair: the input is one step noisier than the output. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

What is the relationship between $x$ and $y$ (slide 23)? Slide 24 answers: **$x$ is an image, and $y$ is a
"less noisy" version of the image**. The word *slightly* is "really, really important. … You're not going
from the image to full noise. That's an impossible leap. You're going from the image to a slightly noisy
version of that image. It is that slightly that allows all the magic to happen" (≈14:03).

![Slide 24: the noisy-Dome row with the last two thumbnails labelled y10 (tenth) and x10 (eleventh, noisiest), and below, "x = image" and "y = 'less noisy' version of the image".](../raw/images/11-diffusion-models/slide-24.jpg)

*Slide 24 — every pair has the same relationship: y is x with a little less noise. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

The lecturer steps back to a "larger comment about machine learning and deep learning": it is "like this
black box where if you can find interesting input output pairs, you can learn a function to go from the
input to the output", and some applications of that are "incredibly non-obvious" (≈14:03–14:49). His
example is a Google Sheets feature that suggests a formula once you select a range of numbers. It was
learned by treating users' selected ranges as inputs and the formulas they then typed as outputs, "a
million examples of this pair", (≈14:49–15:36; the captions garble "a formula" here).

So with these pairs we can build a **de-noising model** (slide 26): a deep network that takes an image
$x$ and outputs a less noisy version $y$. It is trained "by running stochastic gradient descent on the
training data (as usual)" (slide 27), "or Adam or something", and with enough data "you have a network
which can denoise anything you give it" (≈15:36–16:23; see [gradient descent](gradient-descent.md)).

![Slide 28: "An 'image de-noising' model!": a slightly noisy MIT Dome photo labelled x (image) feeds a dark red DNN box, which outputs the clean Dome photo labelled y ("less noisy" version of the image).](../raw/images/11-diffusion-models/slide-28.jpg)

*Slide 28 — the de-noising model: a network trained on the pairs to remove one step of noise. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

**Why only slightly less noisy?** A student asks (≈16:23). "In general, you have to do what you can to
help the model", and the lecturer cites an old adage, "you can't cross a ditch in two jumps": instead you
build a bridge. If the network can denoise something slightly, and do it really well, that capability is
enough to denoise anything (≈16:23–17:08). Another student suggests pairing every noisy copy with the one
clean image. The answer is that you want "the amount of learning it has to do" in each step to be "as
bounded and small as possible". With that pairing, "the gap is still really high" for the noisiest
inputs (≈17:08–17:54).

## Part 5 — Generating: start from pure noise and repeatedly denoise

Once the de-noising model is trained, the original problem (slide 30: noise in, a building out) is
solved: "Start with 'pure' noise and repeatedly denoise it!" (slides 31–34; ≈17:54–18:40). Feed pure
noise to the denoising network. Feed its output back in, and keep going, "and then, before you know it,
Killian Court has emerged from the fog. It's pretty insane that it actually works, this idea."

![Slide 34: four stacked rows, each a small image feeding a dark red "Denoising DNN" box that outputs a slightly cleaner image, with dashed arrows carrying each output into the next row's input and a vertical ellipsis between rows two and three; the top rows are grey noise and the bottom row's output is a clear MIT Dome.](../raw/images/11-diffusion-models/slide-34.jpg)

*Slide 34 — generation is a loop: the same denoising network applied over and over, each output fed back in. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

The model produces a sequence of less noisy images, and "the final one is the 'answer'" (slide 35).

![Slide 35: a strip of eleven images labelled Input (pure grey noise, left) to Output (a clear MIT Dome, right), and below it a large noise image going through a "Denoising DNN" box to a full photograph of the Dome.](../raw/images/11-diffusion-models/slide-35.jpg)

*Slide 35 — the whole sequence from pure noise to the answer, and the same thing drawn as one box. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

**How many steps?** The lecturer is "glossing over" a lot of detail here. Initially the loop had to run
"like 1,000 times", each denoising step "a baby step". Research is very active, and "now, we can, I
think, do it with 50 steps or 100 steps" (≈18:40). That is why diffusion models tend to be slower than a
[large language model](large-language-models.md), and why Midjourney takes a while to return an image:
"it's going through this incremental denoising loop" (≈19:25).

### Questions on generation

These answers come from the question period (≈19:25–27:51).

- **Do we need the noise that came from a training image?** No. "We don't want to necessarily
  regenerate images that were in the training data. That's kind of pointless." The noised versions of
  Killian Court are used for training only. To make a new image, grab fresh noise, "send it in. It gives
  you a stately college building. End of story" (≈20:11).
- **What the model is doing.** Each sample is a point picked from the noise distribution. The model maps
  it to a point in "the distribution of stately college buildings, which might be in a strange, crazy
  distribution". So "it's a way to transform one distribution to another distribution" (≈20:56–21:43).
- **Why a hand with four fingers?** The model "does not understand the notion of fingers". No domain
  knowledge was injected, only "pixel values, a bunch of pictures", so it produces "a statistically
  plausible sample from that distribution". Macro-level details come out right because there are many
  right answers: many roofs still look like a roof. "When it comes to five fingers, there aren't many
  possible right answers, which is why you notice the error very quickly." Better data and better ways to
  steer the process have improved this (≈22:29–24:46).
- **Is there randomness beyond the starting noise?** Yes, "in the process as well". The point the noise is mapped to is used "as the mean value" and the result is
  sampled around it, so the same noise twice need not give the same image (≈24:46–25:32).
- **Does it matter that the noise is pseudorandom?** "Theoretically, it's probably possible" that the model
  latches onto that signal, "but in practice, it really doesn't matter" (≈26:20).
- **Where are the tokens or tags for columns or fingernails?** "There's nothing. It's learning everything
  from the pixel values." Steering by text comes later (≈27:06–27:51).

## Part 6 — The diffusion model, and predicting the noise

"This is called a *diffusion* model" (slide 36). Slide 36 shows the first page of "Deep Unsupervised
Learning using Nonequilibrium Thermodynamics" by Jascha Sohl-Dickstein, Eric A. Weiss, Niru
Maheswaranathan and Surya Ganguli (<https://arxiv.org/pdf/1503.03585.pdf>), "the original paper that
figured this out". There, taking an image and creating noisy versions of it to make training data is
called the **forward process**, and going back is the **reverse process**. The lecturer recommends the
paper as "really well written" (≈27:51–28:37).

Shortly afterwards other researchers made a change that turned out to be "a big improvement in practice"
in the quality of the images (slide 37; ≈28:37–29:50). Instead of training the network to predict the
less noisy image, train it to **predict the noise** $\epsilon$ in its input, and subtract that:

$$y = x - \epsilon$$

Here $x$ is the input image, $\epsilon$ ("noise") is the network's output, and $y$ is the "less noisy"
version of the image. Slide 37 cites "Denoising Diffusion Probabilistic Models" by Jonathan Ho, Ajay Jain
and Pieter Abbeel (<https://arxiv.org/abs/2006.11239>). "This feels arithmetically equivalent, but in
practice, it ends up generating much higher quality images", and there is "some very interesting theory"
about why. Most diffusion models today work this way: "iterative subtracting of predicted noise".

![Slide 37: x (image) feeds a dark red DNN box that outputs epsilon ("noise"); below, the formula y = x − epsilon beside the words "'less noisy' version of the image"; at bottom right, the title block of the "Denoising Diffusion Probabilistic Models" paper by Ho, Jain and Abbeel.](../raw/images/11-diffusion-models/slide-37.png)

*Slide 37 — the key improvement: predict the noise, then subtract it. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

## Part 7 — The U-Net

So far the course has taken an image in and put out a label, and taken text in. It has not built a
network whose output is also an image (≈29:23–30:09). "The dominant architecture to take an image as an
input and produce an image as an output is called the **U-Net**" (slide 38). The lecturer gives "a quick
idea of how it works" in the interest of time (≈30:09).

The slides use the U-Net paper's figure from the University of Freiburg, which OCW excludes from its
licence, so it is described here rather than shown. A U-shaped column of feature maps runs down the left
and back up the right. The left side starts from a one-channel "input image tile" of 572 by 572 and halves
the size at each max-pooling step, while the channels grow from 64 to 128, 256, 512 and 1024 at the
bottom (28 by 28 there). The right side climbs back up through 512, 256, 128 and 64 channels to a
two-channel "output" of 388 by 388. Grey "copy and crop" arrows join each level of the left side to the
same level on the right. The legend names five operations: "conv 3x3, ReLU", "copy and crop", "max pool
2x2", "up-conv 2x2" and "conv 1x1" (slide 38).

- **The left half** "uses standard convolutional and pooling layers" (slide 39). It is "a good, old
  convolutional neural network": convolutional blocks and max pooling, so the image gets "much smaller but
  much deeper", a 3D volume, as in [lecture 4](04-cnns-and-transfer-learning.md) (≈30:09–31:39; see
  [convolutional neural networks](convolutional-neural-networks.md)).
- **The right half** reverses the process, "a new kind of convolutional layer" (slide 40). To "enlarge"
  back to the original size it uses "an 'inverse' of the convolution layer called an Up-Convolution (or
  De-Convolution) layer", implemented as `Conv2DTranspose` in Keras ("see 9.2 in the textbook"). It grows
  the size gradually "till the output you have matches the size of the input that came in" (≈31:39–32:25).
- **Cross-connections.** At every level on the way up, the network also attaches "whatever was in the
  mirror image" on the left side (slide 41, "like residual connections"). The lecturer recalls the
  [residual connection](transformers.md) from lecture 8: a layer that sees only the previous layer's
  output is restricted, and "the more information it has, the more able it's probably to do whatever it
  can". Giving it everything would mean far more parameters, though, so "there is a bit of a balance you
  have to strike, and this was the balance struck by these researchers" (≈32:25–33:56). The figure has a
  grey arrow at each of its four upper levels, and slide 41 boxes the top three.

The U-Net "was originally invented for some medical segmentation use cases, but it is just heavily used
for everything now" (≈33:56).

**When to use one.** "Anytime you have an image-to-image" problem: colorizing a black-and-white image,
making an image higher resolution, or classifying every pixel of an image into one of 10 things. In
general, whenever "the shape of the output" should be "basically the same shape as the input but with
other data" (≈34:42; per-pixel classification is the segmentation of
[images and vision tasks](images-and-vision-tasks.md)).

**Residual connections are not only for images.** ResNet pioneered them, and a transformer block uses
them in two places, adding the input back after self-attention before layer normalization (≈35:28). The
lecturer also names "Deep and Wide Network, if I remember, or DenseNet", and draws a parallel in
structured data: after feature engineering, "you should take your old features and your new features and
send both in" (≈36:13).

## Part 8 — Steering with a text prompt: the intuition

"Now, we have seen how to generate a good image. Now, let's figure out how to steer it or condition it
with a text prompt", "the Holy Grail" (slide 42; ≈36:13–37:00). The intuition builds over slides 43–45:

- We want to take the text prompt into account when generating the image.
- If we had a "rough" image that corresponds to the prompt, we could feed it in and denoise it. If the
  prompt is "cute Labrador Retriever" and a very noisy picture of a Labrador "just happens to be handy",
  the system will denoise it into a better one. "But we don't!" (≈37:00).
- But what if we had an **embedding for the prompt** that is "close" to the embeddings of all the images
  that correspond to that prompt? Imagine "all the images in the universe that correspond to that
  prompt", each with an embedding, and the prompt's embedding "smack where all the image embeddings are"
  (≈37:47).
- Then feed that embedding to the de-noising model to "steer" the generation, since it is already close
  to the embedding of the image we want (≈38:33).

"This is a bunch of me talking and hand waving. It will all become very clear" (≈38:33). The task, then
(slide 46), is "to calculate an embedding for any text prompt that is 'close' to the embeddings of the
images that correspond to that prompt". It feels "almost impossible", "but there's a very clever idea
that OpenAI came up with" (≈39:18).

## Part 9 — Text and image embeddings

Take an image and its caption, "A yellow Labrador retriever on a cushion" (slide 47). How do we compute
an embedding from each? With "standard architectures" (slide 48; ≈40:04): a **text encoder** such as BERT,
since "you know the answer. Run it through a transformer", and an **image encoder** such as ResNet, whose
"penultimate layer, one of the final layers" is a very good representation of the image (see
[transfer learning](transfer-learning.md) and [representation learning](representation-learning.md)).
The slides' Labrador photo is excluded by OCW and not shown here.

But a BERT and a ResNet trained separately give embeddings with no reason to be related: "They were not
trained together, so there's no basis for them to be related" (≈40:49). The embeddings must satisfy
"two important requirements" (slide 49):

1. **Given an image and a caption that describes it, the embeddings should be "close" to each other**
   (slide 50).
2. **Given an image and an unrelated caption, their embeddings should be "far" from each other**
   (slide 51). The slide's unrelated caption is "A train rounding a bend with beautiful Fall foliage all
   around".

Together these ensure that the text embedding and the image embedding "are referring to the same
underlying concept", that they "live" in the same "concept space". Then the embedding of any text prompt
is close to the embeddings of the images that correspond to it (slides 52–53; ≈41:38–42:24). And
closeness is measured, as a student answers at once, by **cosine similarity** (slides 54–55; ≈42:24),
the same measure lecture 10 used to retrieve text chunks (see
[word embeddings](word-embeddings.md) and [retrieval-augmented generation](retrieval-augmented-generation.md)).

## Part 10 — CLIP

How do we compute embeddings that satisfy both requirements (slide 56)? "OpenAI built a model called
**CLIP** (Contrastive Language-Image Pretraining) to solve this problem" (slide 57). The paper is "Learning
Transferable Visual Models From Natural Language Supervision" by Alec Radford, Jong Wook Kim and others
(<https://arxiv.org/pdf/2103.00020.pdf>). CLIP "forms the basis for a whole bunch of models that have
sprung up after this called BLIP and BLIP 2" (≈42:24). It is described at length on the
[CLIP](clip.md) topic page. In outline (slides 58–61; ≈43:11–44:43):

1. Set a 12-layer, 8-head [Transformer Causal Encoder](large-language-models.md) stack as the text encoder
   and ResNet-50 as the image encoder ("with some small tweaks"). The text embedding is the one the
   encoder produces for next word prediction. ResNet-50 has its top chopped off, and "whatever was left is
   the image encoder" (≈43:11).
2. Initialize both with random weights.
3. Grab a batch of `<image, caption>` pairs. The slides use three: the Labrador, "A blue jay
   with an acorn in its mouth", and the train.
4. Run the images through the image encoder and the captions through the text encoder to get their
   embeddings, "a forward pass".
5. Calculate the "closeness score", the cosine similarity, for **every** image-caption pair.

**The closeness grid.** Slides 62–64 lay this out as a 3-by-3 grid, with the three images across the top
and the three captions down the side, both in the same order. Each cell holds the cosine similarity of
one caption's embedding with one image's embedding. ("It took me like five minutes or 10 minutes to do
this PowerPoint. You're welcome", ≈44:43.) The three **diagonal** cells, outlined in green, are the
matching pairs, and "we want these scores to be as high as possible" (slide 64; ≈44:43–45:33). The grid
slides use OCW-excluded photos, so there is no image of them here.

**The loss, first attempt.** Maximize the sum of the green cells. Since a loss is minimized, minimize
the negative of the sum of the green cells (slide 65; ≈45:33). Will that do the trick (slide 66)? A
student sees the flaw: it would "squish everything to the same spot in space". Slide 67: "No. The
optimizer can simply ignore the input, make all the embeddings the same and achieve a perfect cosine
similarity of 1.0 for all pairs of embeddings!" The lecturer's example is making every embedding 0, and
he calls this failure "model collapse" (≈45:33–46:20). (Slide 67's "the same" is the precise version: a
zero vector has no direction, so its cosine similarity is undefined. This note is the page's own.)

**The loss, fixed.** Also push the **off-diagonal** cells, the mismatched pairs, outlined in red on slide
68, to be as small as possible. Slide 69: "maximize the sum of the green cells and minimize the sum of
the red cells", so the loss is "minimize [sum of red cells – sum of green cells]" (≈46:20–47:07). Written
out in the notation of the CLIP paper's figure (slide 70), which calls the text embeddings
$T_1, \dots, T_N$ and the image embeddings $I_1, \dots, I_N$ for a batch of $N$ pairs, with pair $i$
matching text $T_i$ to image $I_i$, the loss on slide 69 is

$$\text{loss} = \sum_{i \ne j} \cos(T_i, I_j) - \sum_{i} \cos(T_i, I_i)$$

where $\cos$ is cosine similarity, the first sum runs over the mismatched (red) cells and the second over
the matching (green) diagonal. The symbols are this page's rendering of the slide's words. Each batch is
run through both encoders, the loss is computed and backpropagated through both networks, "batch, batch,
batch" (≈47:07; see [loss functions](loss-functions.md)).

Slide 70 is "the official picture from the CLIP paper", excluded by OCW. It shows the same thing: a text
encoder producing $T_1 \dots T_N$, an image encoder producing $I_1 \dots I_N$, and an $N \times N$ grid of
their products with the diagonal shaded. "The diagonal is maximized and the off diagonals are minimized"
(≈47:52). CLIP was trained this way on **400 million** `<image, caption>` pairs scraped from
the Internet (slide 71).

**Where the captions came from.** Nobody labelled 400 million images. A student guesses Google Search;
another gets it: the **alt text** that people attach to images on the web, often for accessibility. Alt
text tends to be a more verbose description than a caption, and "more verbose, longer the better,
because there's more stuff for the model to learn from" (≈47:52–48:39).

## Part 11 — CLIP's text encoder alone, and zero-shot classification

With CLIP trained, "we can use CLIP's text encoder by itself": send in any text and get an embedding
close to the embedding of any image that text describes (slide 72; ≈48:39–49:25). That is the piece the
diffusion model needs.

![Slide 72: "We can use CLIP's text encoder by itself. We can send in any text and get an embedding that is close to the embedding of any image described by that text", above two dark red boxes, a Text encoder producing a blue three-block embedding (circled in red dashes) and an Image encoder producing a yellow one.](../raw/images/11-diffusion-models/slide-72.png)

*Slide 72 — after training, only the text encoder is needed to turn a prompt into an image-space embedding. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

"BTW, CLIP can be used for **zero-shot image classification**" (slide 73; ≈49:25–54:02). An ordinary
classifier is trained on a fixed set of labels. If you had five classes and tomorrow there is a sixth,
"you have to go back to the drawing board and retrain the whole thing". Instead, imagine handing the
classifier an image and a list of possible labels, a different list each time, and having it pick one:
"an insanely flexible image classification system". After several student attempts, one gets it: pick
the label whose text embedding is closest to the image embedding ("You're not missing anything. That's
the right answer", followed by applause, ≈53:16). A label such as "dog" or "cat" is just text, so embed
each of the $n$ labels with the text encoder ($T_1$ to $T_n$), embed the image once ($I$), compute the
cosine similarities, and take the highest (≈53:16–54:02). The CLIP paper's figure on slide 73, excluded by
OCW, does this with class names "plane", "car", "dog" … "bird" placed into the template "A photo of a
{object}." and picks "A photo of a dog." The lecturer calls it "a side use of CLIP unrelated to diffusion
models, but I just thought it was really clever" (≈54:02).

## Part 12 — Conditioning the denoiser on the prompt

Now the original goal: steer the diffusion model with a prompt (slide 74; ≈54:02). Recall the training
pairs: $x$ is an image and $y$ its less noisy version (slide 75). **Expand $x$ to include the caption**
(slide 76): $x$ = [image, CLIP text embedding of caption], with $y$ still the less noisy version of the
image, "or as we saw earlier, it could be just the noise component of that image". Take each training
image's caption, run it through CLIP's text encoder, and concatenate that embedding with the image as the
new input (≈54:51). Then train the model "like before" (slide 77).

![Slide 77: two inputs feed a dark red "Denoising DNN" box: the "CLIP text embedding of caption" (three blue blocks) above, and x, the image, below; the box outputs y, the "less noisy" version of the image.](../raw/images/11-diffusion-models/slide-77.png)

*Slide 77 — the only change from part one: the denoiser also receives the CLIP embedding of the image's caption. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

After this model is trained (slide 78), start with pure noise **and a prompt** and repeatedly denoise
(slides 79–81). Give it "Killian court at MIT during springtime" with a bunch of noise. Because the
prompt's CLIP embedding "lives in the same image space as all Killian Court embeddings, Killian Court
images", the repeated denoising arrives at Killian Court. "That's how they steer the image. It's a
two-step process" (≈55:37).

![Slide 81: four stacked denoising steps, each fed the prompt "Killian court at MIT during springtime" and the previous image; the images go from grey noise in the top rows, through a vertical ellipsis, to a clear MIT Dome at the bottom.](../raw/images/11-diffusion-models/slide-81.jpg)

*Slide 81 — text-to-image generation: the same loop as slide 34, with the prompt fed to every step. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

The lecturer thinks CLIP "was a breakthrough", as one of the earliest examples of creating embeddings for
very different data types, images and text, "that all happen to live in the same space, the same concept
space". "If you look at the modern multimodal large language models, they are all based on the same exact
idea" (≈55:37–56:22; see [AI, ML, DL and generative AI](ai-ml-dl-genai.md) on multimodality).

### Questions: Sora and physics, speed, and the noise distribution

These answers come from the question period (≈56:22–1:00:59). OpenAI had not yet published Sora's full
technical report, so "we don't know for sure", and the lecturer flags that "this may be wrong".

- **Does Sora use a physics engine?** The consensus, he says, is no. Computer-vision experts say it was
  trained on a lot of video-game data along with actual videos, and on a corpus so large that it "learned
  to mimic certain physics aspects … just as a side effect". He compares that with LLMs: solving a new
  high-school maths problem after training on such problems is not surprising, but solving French
  maths problems after reading English ones and French literature "is the new news" (≈57:08–57:54). The
  evidence against a physics engine is the videos people post on Twitter: sand coming through the base of
  a beach chair, an object still visible behind an opaque one. It has learned "to fake the laws of
  physics in some context. Much like you can fake five fingers" (≈1:00:12).
- **Video as frames.** That part is not in doubt: a video is a series of frames, each an image, and
  "there is a sequentiality to it, which is where the transformer stack will come in" (≈59:26).
- **Why is Stable Diffusion faster?** "There's a very key trick. It's coming": the latent space of Part 13
  (≈58:40).
- **What if the noise were not normal?** Change it to a Poisson or some other distribution and "it will
  definitely change the results", because the math behind why this works "heavily depends on the Gaussian
  assumption" (≈58:40).

## Part 13 — Inside the denoising DNN: latent diffusion

"Let's take a quick look inside the" Denoising DNN (slide 82). The lecturer uses the figure of "High-Resolution
Image Synthesis with Latent Diffusion Models" by Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser and Björn Ommer (<https://arxiv.org/abs/2112.10752>). He calls it "the original paper,
which took that idea of the diffusion model" and asked whether it could be made much faster, since
diffusion models are slow (≈1:00:59). The figure is excluded by OCW, so it is described here. A pink
"Pixel Space" holds an encoder $\mathcal{E}$ from the image $x$, and a decoder $\mathcal{D}$ to the output
$\tilde{x}$. A green "Latent Space" holds the diffusion process and a "Denoising U-Net $\epsilon_\theta$"
with four Q, K, V cross-attention boxes. A grey "Conditioning" box (semantic map, text, representations,
images) feeds an encoder $\tau_\theta$ into those boxes. He picks out three things (≈1:00:59–1:02:34):

1. **It uses the U-Net** (slide 83) to go from image to image.
2. **The CLIP embedding of the text prompt is "woven" into the U-Net through an attention mechanism**
   (slide 84): "you can see the QKV business here, which should be familiar at this point". The prompt is
   incorporated through the same key, query and value machinery as in
   [lecture 8](08-transformers-and-self-supervised-learning.md) (see [transformers](transformers.md)).
3. **The diffusion process operates in "latent space" instead of "pixel space", "and this speeds things
   up dramatically"** (slide 85). Instead of denoising the whole image, run the image through an image
   encoder to get an embedding, and denoise the embedding again and again. "These embeddings are much
   smaller than images, therefore, they are much faster to process." "Once you've done it 1,000 times", the almost noiseless embedding goes through an image decoder to give the image. "Hence, it's called the
   **latent diffusion model**."

The pressure for still more speed is commercial: "who wants to wait for ten seconds, right?" (≈1:02:34).

## Part 14 — Applications

"These models are transforming the art world": the site lexica.art collects images together with the
prompts that created them, a source of ideas for anyone working in the space (slide 86; ≈1:02:34).

"But their applicability is far-reaching" (slide 87). Slide 87 shows a New York Times article, "A.I. Turns
Its Artistry to Creating New Human Proteins" (<https://www.nytimes.com/2023/01/09/science/artificial-intelligence-proteins.html>),
with protein structures generated by a diffusion model. Its quotation: "Much as DALL-E leverages the
relationship between captions and photographs, similar systems can leverage the relationship between a
description of what the protein can do and the shape it adopts." The lecturer recalls AlphaFold, which
creates a protein's 3D structure from its amino-acid sequence: "I don't think they use a diffusion model,
but you can imagine using" one (≈1:02:34–1:03:21). The general point is that the objects created "don't have
to be images". All you need is enough training data about such things and a meaningful notion of noising
them: 3D things, protein structures, "a whole bunch of very interesting applications in biomedical
sciences". There are even ways to use diffusion models for large language modeling (≈1:03:21–1:04:07).

## Part 15 — The Diffusers Colab

"We will use the Diffusers library" (slide 88), Hugging Face's library of diffusion models. Its
documentation page is excluded by OCW. The notebook is not on OCW, and slide 88's "Colab" link does not
print its target, so this follows the narration (≈1:04:07–1:09:49), naming only what the lecturer says.

- **The same philosophy as for text.** Hugging Face's `pipeline` gives quick access to a pretrained model,
  "very similar to what we did in lecture 8 for NLP" (≈1:04:52–1:05:38; see
  [lecture 8](08-transformers-and-self-supervised-learning.md)). He installs `diffusers` and creates "the
  StableDiffusionPipeline from pretrained" with Stable Diffusion version 1.4 (≈1:05:38).
- **PyTorch.** The course has used TensorFlow, "but a lot of these models, unfortunately, happen to be in
  PyTorch. So knowing a little bit of PyTorch is actually very helpful" (≈1:05:38; see [Keras](keras.md)).
  The weights are loaded in the FP16 storage format, smaller than 32 bits, so they download faster.
- **A first prompt.** "A photograph of an astronaut riding a horse", "one of the original famous meme
  prompts". He sets a seed for reproducibility and runs the pipeline, which goes through **50 denoising
  steps** and produces an astronaut on a horse. The seed sets the random starting point, so a different
  seed gives "an astronaut riding another horse" (≈1:06:23). Such prompts are popular because the
  result is "guaranteed not to be in the training data", so the model is not regurgitating (≈1:07:10).
  A student's prompt, "MIT professor riding a horse", produces two of them, to laughter (≈1:07:10–1:07:56).
- **Negative prompting.** Give two prompts: "create an image which embodies the first prompt but not the
  second", in effect subtracting the second. "Labrador in the style of Vermeer" over 50 steps gives a
  Labrador in a blue scarf. Adding the negative prompt "blue" removes it: "The blue is gone" (≈1:07:56–1:08:43).
- **More steps?** Asked whether 1,000 steps instead of 50 helps: "typically, if you do more of these
  things, it gets better", because each step denoises very slightly, "so errors won't accumulate"
  (≈1:09:30).

## Part 16 — A lightning tour of Hugging Face computer-vision pipelines

The lecturer adds a tour he did not have time for earlier (slide 89; ≈1:09:49 to the end; the Colab link's
target is not printed). Just as for text, "if you have a computer vision problem that just maps to a
standard computer vision task, you can just use the HuggingFace Hub". The tasks are those of
[images and vision tasks](images-and-vision-tasks.md).

- **Image classification.** Import `pipeline`, name the task, and run "a very famous image" through it.
  The default model says Egyptian cat, at 94% probability, a reasonable answer for a busy picture with
  more than one object in it (≈1:10:17–1:11:03). You can pick any model from the Hub, where image
  classification lists 10,487 models. With Microsoft's ResNet the answer is tiger cat, 94% (≈1:11:03–1:11:48).
- **Object detection**, "which we didn't talk about in class". An object-detection pipeline finds and boxes
  the cat (at "100% probability, I guess"), the remotes and the couch, with **bounding boxes**. "Off the
  shelf, ready to go. No heavy lifting required" (≈1:11:48–1:12:34).
- **Image segmentation**, for the exact contour of each object instead of a box. Each object found gets a
  **mask** saying which pixels belong to it. The first object it finds is "perfectly delineated". Overlaid on the image,
  the masks then pick out the remotes and, at a student's suggestion, the couch, which is "pretty good,
  except that the middle part has gotten confused" (≈1:13:21–1:14:08).

"So Hugging Face is all these things, and so you should definitely check it out" (≈1:14:08). With a
minute left and no questions, the class ends: "See you on Wednesday" (≈1:14:58).

## Printed slips

- Slide 1 prints the course number "15.S04", not 15.773.
- Slide 2's credit reads "from the prompt from the prompt quoted above"; slide 9's credit block has
  "Toeldo Courthouse"; slide 76's title is "What if expand x to include its caption".
- Slide 87's caption credits "Namrata Anand" and its OCW notice "Namruta Anand".
- Slides 43 and 44 carry later bullets in the text layer that the page does not show. Slide 44's hidden
  copy reads "close" to the image, where slide 45 prints "close" to the embedding of the image.
- Spoken: "y is the noisy image" (≈28:37), where slides 24 and 37 define $y$ as the less noisy version;
  "it can just make all the embeddings 0" (≈46:20) for slide 67's "the same"; "model collapse" for that
  failure, which the slides do not name. The captions garble "a formula" (≈14:49) and part of a question
  about U-Nets (≈34:42).

## Related pages

- [Diffusion models](diffusion-models.md): the forward and reverse processes, training pairs, generating
  by repeated denoising, predicting the noise, the U-Net, conditioning on text, latent diffusion and the
  Diffusers library, gathered as a topic.
- [CLIP](clip.md): text and image encoders trained into one concept space, the closeness grid and its
  loss, and zero-shot classification.
- [Images and computer-vision tasks](images-and-vision-tasks.md): pixels and channels, and the
  classification, detection and segmentation tasks of the Hugging Face tour.
- [Convolutional neural networks](convolutional-neural-networks.md): the convolution and pooling layers
  of the U-Net's left half.
- [Transformers](transformers.md): residual connections, and the key, query and value attention that
  weaves the prompt into the U-Net.
- [Word embeddings](word-embeddings.md) and [representation learning](representation-learning.md):
  embeddings, cosine similarity and the idea of a shared space.
- [Self-supervised learning](self-supervised-learning.md): lecture 8's contrastive learning for images,
  the family CLIP's training belongs to.
- [AI, machine learning, deep learning and generative AI](ai-ml-dl-genai.md): text-to-image and
  multimodality in the course's map of the field.
- [Lecture 8](08-transformers-and-self-supervised-learning.md): attention's $A^K$, $A^Q$, $A^V$
  matrices, residual connections and the first Hugging Face pipelines.
- [Lecture 1](01-introduction.md): the course's first text-to-image example (slide 45).
