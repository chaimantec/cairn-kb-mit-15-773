# Diffusion models

A **diffusion model** generates an image by starting from random noise and removing a little of it at a
time, until a picture is left. It learns to do that from the opposite process, which is easy: take
training images and add noise to them, step by step. The course teaches it in
[lecture 11](11-diffusion-models.md), the last lecture, as the building block of text-to-image models
such as Midjourney and Stable Diffusion, and of the text-to-video model Sora, which OpenAI describes as
"text-conditional diffusion models" on "a transformer architecture" (lecture 11, slide 6). The lecture deliberately skips the mathematics: "there's a whole bunch of detail here, which I'm
glossing over" (≈18:40). This page follows its intuitive account. Citations are to lecture 11 unless they say otherwise.

## The problem: an image from noise

Suppose we want a model that generates images of one kind, "stately college buildings" such as MIT's
Killian Court, and we have many training images of them (slides 8–9). Each image the model produces
should be different. The way to get that is to give the model **noise** as its input, an image "in which
all the pixel values are randomly picked" (slide 11; ≈4:45). Noise differs every time it is drawn, so the
output does too. A model trained on college buildings produces college buildings.

But nobody knows directly how to train a network to turn pure noise into a photograph (slides 12–13).
Following George Pólya's heuristic in *How to Solve It*, reverse the question (≈6:18): given an image,
can we make a noisy version? That is easy (slide 14).

## Adding noise

"Just add random numbers to every pixel!" (slide 15). In the lecture's Colab, an image is read in with
PIL, converted to a NumPy array and divided by 255 so every value lies between 0 and 1. For every pixel,
a number drawn from a normal distribution with mean 0 and a small standard deviation (0.3, via
`np.random.normal`) is added. The result is then clipped back to between 0 and 1 (≈7:52–10:13). Raising the
standard deviation, 0.3, 0.4, 0.5 and so on, gives a sequence from the clean image to one that is almost
pure noise (slide 16).

![Slide 16: a row of eleven thumbnails of the MIT Dome photo, clean at the left and progressively noisier to the right.](../raw/images/11-diffusion-models/slide-16.jpg)

*Lecture 11, slide 16 — one training image and its noisier and noisier copies. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

In the language of the first diffusion paper, cited on slide 36 (Sohl-Dickstein, Weiss, Maheswaranathan
and Ganguli, "Deep Unsupervised Learning using Nonequilibrium Thermodynamics",
<https://arxiv.org/pdf/1503.03585.pdf>), this noising is the **forward process**. Generating, which runs
the other way, is the **reverse process** (≈27:51).

## Training: pairs one step apart

Cut each image's sequence of noisy copies into **input-output pairs** (slides 19–24). The input $x$ is an
image, and the output $y$ is the copy one step *less* noisy than it. So $x_1$ is the second copy and
$y_1$ the clean image, $x_2$ the third and $y_2$ the second, and so on. Every pair has the same
relationship: **$y$ is a "less noisy" version of $x$** (slide 24).

The word *slightly* is the point. "You're not going from the image to full noise. That's an impossible
leap. You're going from the image to a slightly noisy version of that image. It is that slightly that
allows all the magic to happen" (≈14:03). The lecturer's adage is "you can't cross a ditch in two jumps"
(≈16:23). Keep the amount the network must learn in each step "as bounded and small as possible", and a
network that removes a little noise very well can remove any amount, one step at a time (≈17:08–17:54).

A deep network trained on these pairs by stochastic gradient descent, "as usual", is an **image
de-noising model** (slides 26–28; see [gradient descent](gradient-descent.md) and
[training a neural network](training-neural-networks.md)).

## Generating: denoise again and again

Once trained, "start with 'pure' noise and repeatedly denoise it!" (slides 31–34). Feed noise to the
denoising network, feed its output back in, and keep going until "Killian Court has emerged from the
fog". The model produces a sequence of less noisy images, and "the final one is the 'answer'" (slide 35;
≈17:54–18:40).

![Slide 34: four stacked rows, each an image feeding a "Denoising DNN" box that outputs a slightly cleaner image, with dashed arrows feeding each output into the next row; grey noise at the top resolves into the MIT Dome at the bottom.](../raw/images/11-diffusion-models/slide-34.jpg)

*Lecture 11, slide 34 — generation is the denoising network applied in a loop. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

The loop is why these models are slow. Originally it took "like 1,000" baby steps; now "I think, … 50
steps or 100 steps". That is why a diffusion model tends to take longer than a
[large language model](large-language-models.md) to answer (≈18:40–19:25).

Two ways of seeing what has been learned come from the question period. The model is "a way to transform
one distribution to another distribution": a point sampled from the noise distribution is mapped to a
point in the distribution of college buildings (≈20:56–21:43). And it knows only pixels. It "does not
understand the notion of fingers", so a generated hand can have four of them. Many right answers make a
roof easy to get right, while "when it comes to five fingers, there aren't many possible right answers,
which is why you notice the error very quickly" (≈22:29–24:00). The randomness is not only in the
starting noise but "in the process as well" (≈24:46). The math behind why diffusion works "heavily
depends on the Gaussian assumption", so changing the noise to another distribution would change the
results (≈58:40).

## Predicting the noise

The improvement almost every diffusion model now uses is on slide 37, citing "Denoising Diffusion
Probabilistic Models" by Jonathan Ho, Ajay Jain and Pieter Abbeel (<https://arxiv.org/abs/2006.11239>).
Instead of training the network to output the less noisy image, train it to output the **noise**
$\epsilon$ in its input image $x$, and subtract it to get the less noisy version $y$:

$$y = x - \epsilon$$

"This feels arithmetically equivalent, but in practice, it ends up generating much higher quality images",
and so most diffusion models do "iterative subtracting of predicted noise" (≈28:37–29:50).

![Slide 37: x (image) into a DNN box that outputs epsilon ("noise"); below, y = x − epsilon, "'less noisy' version of the image"; at bottom right, the title block of the "Denoising Diffusion Probabilistic Models" paper.](../raw/images/11-diffusion-models/slide-37.png)

*Lecture 11, slide 37 — predict the noise, then subtract it. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

## The network: a U-Net

A denoiser takes an image in and puts an image of the same size out. "The dominant architecture to take
an image as an input and produce an image as an output is called the **U-Net**" (slide 38; ≈30:09). Its
**left half** is a familiar [convolutional neural network](convolutional-neural-networks.md): convolutions
and max pooling make the image smaller and deeper (slide 39). Its **right half** reverses that with
**up-convolution** (de-convolution) layers, `Conv2DTranspose` in Keras, until the output is the input's
size again (slide 40). **Cross-connections** carry each level of the left half across to the same level
of the right half, "like residual connections" (slide 41), because a layer given more information can do
more with it (≈32:25–33:56). Invented for medical image segmentation, it is "heavily used for everything
now", from colorizing to raising resolution to classifying every pixel (≈33:56–34:42). The U-Net figure
the slides use is excluded by OCW; [lecture 11, Part 7](11-diffusion-models.md#part-7--the-u-net)
describes it.

## Steering with a text prompt

To make the image match a prompt, the denoiser needs some representation of the prompt that is "close" to
the images the prompt describes. [CLIP](clip.md) supplies one: a text encoder trained so that a caption's
embedding sits near the embeddings of matching images (slides 42–73). Then **expand the input**: $x$ =
[image, CLIP text embedding of caption], with $y$ still the less noisy image (or the noise). Concatenate
each training image's caption embedding with the image, and train as before (slides 75–77; ≈54:02–55:37).

![Slide 77: the "CLIP text embedding of caption" and the image x both feed a "Denoising DNN" box, which outputs y, the "less noisy" version of the image.](../raw/images/11-diffusion-models/slide-77.png)

*Lecture 11, slide 77 — conditioning: the caption's CLIP embedding is a second input to the denoiser. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec11.pdf)*

At generation time, give it pure noise and a prompt, "Killian court at MIT during springtime", and
denoise repeatedly with the prompt fed to every step (slides 79–81). Since the prompt's embedding lives
in the same space as Killian Court images, the loop arrives at Killian Court (≈55:37). Conditioning,
steering and controlling are synonyms for this in the literature (≈3:13).

## Latent diffusion and Stable Diffusion

The latent diffusion model (Rombach, Blattmann, Lorenz, Esser and Ommer, "High-Resolution Image
Synthesis with Latent Diffusion Models", <https://arxiv.org/abs/2112.10752>; slides 82–85) is how such
models were made fast. The lecturer picks out three things in its figure (≈1:00:59–1:02:34):

1. The denoiser is a **U-Net** (slide 83).
2. The CLIP embedding of the prompt is "woven" into the U-Net **through an attention mechanism**, with
   the same keys, queries and values as a [transformer](transformers.md) (slide 84).
3. The diffusion runs in **latent space**, not pixel space. An image encoder turns the image into an
   embedding, the denoising loop works on that much smaller embedding, and an image decoder turns the
   final embedding back into an image. "These embeddings are much smaller than images, therefore, they
   are much faster to process", hence the name (slide 85).

## In code: Hugging Face Diffusers

Hugging Face's **Diffusers** library packages pretrained diffusion models (slide 88). In the lecture's
Colab, which is not on OCW, a `StableDiffusionPipeline` is loaded "from pretrained" with Stable
Diffusion 1.4, in PyTorch, with FP16 weights. A seed sets the random starting point, and the pipeline
runs 50 denoising steps per image ("a photograph of an astronaut riding a horse"). A different seed gives
a different image. **Negative prompting** subtracts a second prompt: "Labrador in the style of Vermeer",
with the negative prompt "blue", loses the blue scarf. More steps typically give a better image, because
each step denoises only slightly (≈1:04:07–1:09:49; see
[lecture 11, Part 15](11-diffusion-models.md#part-15--the-diffusers-colab)).

## Beyond images

The objects generated "don't have to be images". Anything with enough training data and a meaningful
notion of noising will do: 3D objects, and protein structures, as in a New York Times article on protein
design with diffusion models (slide 87). There are even diffusion approaches to language modeling
(≈1:02:34–1:04:07). Sora treats a video as a sequence of frames, with a transformer handling the
sequence (≈59:26).

## Where this comes up

- [Lecture 1](01-introduction.md): text-to-image as one of generative AI's tasks, with a Stable Diffusion
  prompt (slide 45).
- [Lecture 11](11-diffusion-models.md): everything on this page, from noise to Stable Diffusion, with the
  Colabs.
- See also [CLIP](clip.md), [convolutional neural networks](convolutional-neural-networks.md) (the U-Net's
  layers), [images and computer-vision tasks](images-and-vision-tasks.md) and
  [AI, machine learning, deep learning and generative AI](ai-ml-dl-genai.md).
