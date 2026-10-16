---
title: Lecture 10 — Finetuning (companion deck 10.5, not presented in a video)
lecture: 10
deck: 10.5 (mit15_773_s24_lec105.pdf)
slides: 33
source_pdf: https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec105.pdf
note: Slide N = PDF page N (N = 1..33). Every page that prints a number bottom-right prints its own page number. Page 33 is OCW's appended end page, not lecture content.
---

# Lecture 10 — Finetuning (deck 10.5): slide-by-slide

Text and figures of all 33 slides of
[`mit15_773_s24_lec105.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec105.pdf)
(MIT 15.773 Hands-On Deep Learning, Spring 2024; lecturer Rama Ramakrishnan), titled "Lecture 10.5:
Finetuning" on its roadmap (slide 1). **No video presents this deck.** Lecture 10's video covers
fine-tuning and LoRA from the main deck,
[`10-adapting-llms.md`](10-adapting-llms.md) (slides 82–112), and lecture 11's video opens on
diffusion models. Deck 10.5 is a variant of that fine-tuning section, and most of its pages repeat the
main deck's. Cite it as "deck 10.5, slide N", and prefer the main deck where the two agree. Diagrams,
tables and screenshots are described in prose since the KB is read as text.

**How it differs from the main deck.** Page by page, deck 10.5's slides correspond to the main deck's
slides as follows: 2–5 → 82–85, 7–9 → 87–89, 10 → 91, 11–27 → 95–111, 29 → 93, 30 → 92, 31 → 113, 32 →
114. The differences that matter:

- Slide 3's diagram still has the "Dense (ReLU)" box that the main deck corrects (its slides 3 and 83).
- Slide 5's list of the weights that fine-tuning updates (repeated on slide 20) has seven items, ending
  "Weights in Dense layers outside TCE" and "Weights in final Softmax layer". The main deck's slides 85
  and 104 merge these into six, ending "Weights in final Dense/Softmax layers".
- Slide 6 replaces the main deck's slide 94 and adds "Bit first… why would we want to do this?" (sic).
- Slides 7–9 put the product-review example as a question, "'Write a positive review of the following
  product' What do you think?", and slide 8 adds "This is Alpaca, running off of Llama 7B. Reads like
  Marketing copy." The main deck does not say which model wrote the marketing-copy outputs.
- Slide 10 is titled "Lets get better at Fake Reviews", where the main deck's slide 91 reads "A dataset
  for instruction tuning (example)".
- Slide 14 adds "A bit long" and a smiley to the 5,261-day estimate.
- Slide 21 describes $A^K$ as "a 8096x8096 matrix (one for each self attention layer)", and slides 24–25
  draw 8096 × 8096 and 8096 × 2 matrices. Slide 25 counts "2 x 8096 = 16,192 parameters i.e., just 0.02%
  of the original 64M". These are the numbers the lecturer says in class (≈1:15:49). The main deck's
  slides 105, 108 and 109 use 8192 (64 heads of 8192 × 128) and count "4 x 8192 = 32,768 parameters",
  0.05%.
- Slide 28 is "Colab (if time permits)" with no link target printed, where the main deck's slide 112 says
  "Stay tuned for video and colab."
- The main deck's slides 86 and 90 (two text slides that introduce the example and the idea of
  fine-tuning on reviews) have no counterpart here.

No page of this deck prints an OCW notice. But slides 7, 8 and 30 show the rag & bone product page,
slides 9 and 29 the Urban Outfitters product page, slides 11 and 12 Meta's Llama 2 web page, and slide
31 Andrej Karpathy's effort-benefit curve. The main deck marks each of them as excluded from OCW's
Creative Commons licence (its slides 87–89, 92, 93, 95 and 113). Since an exclusion follows the image,
they are described in prose here and not rendered. Slide 32 is the "How LLMs Are Built?" flow chart from
"Large Language Models: A Survey" by Minaee et al., without the CC BY credit the main deck's slide 114
prints beside it. Slide 10's screenshot of an instruction-tuning dataset is unmarked third-party content,
transcribed in outline.

**Images.** None. Every figure here is either excluded material reused without a notice, or a repeat of
a main-deck figure that is rendered there: slide 2 → main deck 58 (the ladder), 3 → 85 (the pipeline,
whose ReLU box the main deck corrects), 4 → 84, 5 and 20 → 85, 24 and 25 → 109 (the low-rank diagram,
here with 8096), and 32 → 114.

## Numbering

Headings use the PDF page number (1..33). Every page that prints a number bottom-right prints its own
page number; pages 1, 2, 4, 7–10, 29–31 and 33 print none.

## Contents

| Slides | Section |
| ------ | ------- |
| 1 | Roadmap: "Lecture 10.5: Finetuning" |
| 2–5 | Fine-tuning a causal LLM: the ladder, the pipeline, input-output examples, what gets trained |
| 6–10 | Why fine-tune: marketing-copy product reviews, and a dataset of review examples |
| 11–19 | Llama 2; how hard it is to train and fine-tune Llama-2-70b; what consumes memory |
| 20–27 | LoRA: low-rank adaptation of the attention matrices; memory afterwards |
| 28 | Colab (if time permits) |
| 29–30 | The results of supervised fine-tuning |
| 31–32 | The effort-benefit curve; the topics covered |
| 33 | MIT OpenCourseWare end page |

## Slide 1 — Lecture 10.5: Finetuning

Title slide. Underlined dark red heading: "The “Deep Learning for NLP” Lecture Roadmap". Large bold dark red title: "Lecture 10.5: Finetuning".

Below, a dark red list with a strikethrough through every line (earlier lectures, shown as covered): "Lecture 5: Text Vectorization and the Bag-of-Words Model" (wraps over two lines), "Lecture 6: Embeddings", "Lecture 7: Transformers – (1/2)", "Lecture 8: Transformers – (2/2)", "Lecture 9: LLMs (1/2)", "Lecture 10: LLMs (2/2)".

A dark red horizontal rule, then the MIT Sloan School of Management logo at lower left and the text "15.S04: Hands-on Deep Learning", "Spring 2024", and in red "Farias, Ramakrishnan". A small red hand-drawn scribble sits at the upper right (a stray pen mark, no text).

*Figure: no*

## Slide 2 — Let’s look at Fine Tuning next

Title: "Let’s look at Fine Tuning next".

Diagram: a large pale pinkish-grey curved arrow rising from lower left to upper right, with four dark red dots on it that grow larger along the arrow. Labels, left to right, each with an italic caption beneath:

- Smallest dot: "Zero-Shot Prompting", caption "Prompt = A clear instruction".
- Next dot: "Few-Shot Prompting", caption "Prompt = A few examples".
- Next dot: "Retrieval-Augmented Generation", caption "Prompt = Question + relevant chunks of text".
- Largest dot: "Fine-Tuning", caption "The LLM is further trained with examples" ("trained" underlined). The Fine-Tuning dot and its label sit inside a dash-dot rounded rectangle that highlights it as the topic of this lecture.

*Figure: yes — rising arrow of four approaches (zero-shot, few-shot, retrieval-augmented generation, fine-tuning) with growing dots, fine-tuning boxed*

## Slide 3 — In Fine-Tuning, we take a causal LLM (like GPT) and …

Title: "In Fine-Tuning, we take a causal LLM (like GPT) and …".

Diagram, left to right, with italic column labels: "Input" (far left), "Positional input embeddings", "Contextual embeddings", "Softmax", "Output" (far right). Four short dark red arrows enter from the left, one per input row. Four rows of four grey rounded squares (positional input embeddings) feed, via an arrow from the third row, a dark red box "TCE\*". An arrow leads from it to four rows of four blue rounded squares (contextual embeddings), then via an arrow to a dark red box "Dense (ReLU)" (two lines), then via an arrow to four rows of three yellow rounded squares. From each yellow row a dark red arrow leads to its own small dark red box "SM" (four SM boxes). One further short dark red arrow leaves the last SM box to the right (towards the Output column, which has no words).

Footnote: "\*Transformer Causal Encoder stack" ("Causal" in red). Printed page number 3.

*Figure: yes — causal LLM pipeline: input arrows, grey embeddings, TCE stack, blue contextual embeddings, Dense (ReLU), yellow rows, four SM boxes*

## Slide 4 — … train it further on domain-specific input-output examples …

Title (two lines): "… train it further on domain-specific input-output examples …".

Left: a two-row grey table. Row 1: "Input" | "Explain the moon landing to a 6 year old in a few sentences." (italic). Row 2: "Desired output" | "People went to the moon in a big rocket, walked around and came back to Earth. They took pictures of what they saw and sent them back so we could all see them." A thick dark red arrow points right from the table to the diagram.

Right: a red-outlined box "Input" above a vertical column of italic words, one per line: Explain, the, moon, landing, to, a, 6, year, old, in, a, few, sentences., then (highlighted yellow) People, went, to, the, moon, in, a, big, rocket, … . A tall black rectangle labelled "Causal LLM" stands to the right of the column, with "…" dots to its left and right. Black arrows run from the input words "sentences.", "People", "went" and from the last input row (the "…" after rocket) into the box, and arrows leave the box on the right to a column under a red-outlined box "“Labels”": People, went, to, the, moon, in, a, big, rocket, … (all highlighted yellow). So the label column is the desired output, aligned with the input positions starting at "sentences.".

Footnote: "\* This is essentially the Supervised Fine-tuning step we discussed earlier to transform GPT-3 to GPT-3.5" (note: no asterisk appears elsewhere on the slide).

*Figure: yes — Input/desired-output example next to a Causal LLM box with input prompt plus output words and the output words as labels*

## Slide 5 — … and update the weights

Title: "… and update the weights".

Same layout as slide 3, with these changes: the box reads "TCE" (no asterisk) and the footnote is gone (page number 5 printed). Red dashed rounded outlines now enclose three regions: the grey positional embeddings together with the TCE box; the "Dense (ReLU)" box; and the SM boxes (the column of four SM boxes, with the final output arrow). Beneath the diagram a red italic numbered list:

1. Positional embeddings
2. Stand-alone embeddings (unless pretrained and Trainable=False)
3. Matrices $A^K, A^Q, A^V$ for each attention head (inside TCE)
4. Layer norm scale and bias parameters (inside TCE)
5. Weights in Feed-Forward layers (inside TCE)
6. Weights in Dense layers outside TCE
7. Weights in final Softmax layer

*Figure: yes — causal LLM pipeline with the trainable regions outlined in red dashes and a numbered list of weights that get updated*

## Slide 6 — [Text only: why fine-tune?]

No title. Body text:

"For small causal LLMs (like GPT-2), this is possible but for larger LLMs, we will face computational challenges."

"We will explain the nature of these challenges and describe an alternative approach that addresses these challenges."

"Bit first… why would we want to do this?" (printed "Bit" as shown). Page number 6 printed.

*Figure: no*

## Slide 7 — ‘Write a positive review of the following product’ What do you think?

Title (two lines): "‘Write a positive review of the following product’ What do you think?"

Left: a screenshot of a product page (a clothing retailer's listing for a slim-fit jean; brand wordmark "rag & bone NEW YORK" at top). It has a short description paragraph, two feature icons ("Authentic Stretch", "Machine Washable"), collapsed "Size & Fit" and expanded "Features & Details" headings with fabric and care lines ("98% Cotton 2% Polyurethane" among them), and a product ID line. (Third-party content; described, not copied.)

Right, black text (the same description as typed on the right):

"Timeless. Authentic. Iconic. One of the building blocks of a perfect wardrobe. The Fit 2 - our most loved slim-fitting jean. For good reason. Mid-rise, tailored through the hip and thigh with a narrow leg opening. Here in Authentic Stretch: selvedge-inspired denim with an optimal, shape-holding touch of comfort stretch. Featured in raw Japanese unwashed denim. Authentic Stretch. Machine Washable. Mid rise slim fit, tailored through hip and thigh"


*Figure: yes — screenshot of a retailer product listing (rag & bone slim-fit jean) with its description text repeated at right*

## Slide 8 — ‘Write a positive review of the following product’ What do you think?

Same as slide 7, with one addition: red text at lower right (large): "This is Alpaca, running off of Llama 7B. Reads like Marketing copy." Everything else (screenshot, description text) is unchanged.

*Figure: yes — same product listing as slide 7 with a red comment that the output is Alpaca on Llama 7B and reads like marketing copy*

## Slide 9 — ‘Write a positive review of the following product’ What do you think?

Title (two lines): "‘Write a positive review of the following product’ What do you think?"

Left: a screenshot of another retailer's product page (wordmark "URBAN OUTFITTERS" at top; a "Details" section with a product SKU and colour code line, a short description of a crew neck jersey tank top by Champion, "Features" bullets, "Content + Care" bullets such as "100% Cotton" and "Machine wash", "Size + Fit" lines for a model, and a short "Champion" paragraph beginning "Since 1919"). (Third-party content; described, not copied.)

Right, black text:

"Champion's Crew Neck Jersey Tank Top is a great addition to any wardrobe. The boxy and cropped silhouette is flattering on all body types, and the cotton fabrication is soft and comfortable. The crew neckline is classic and timeless, and the sleeveless design makes it perfect for warmer weather. The Heritage style tee is an exclusive from Urban Outfitters, and the 100% cotton fabrication and machine washability make it easy to care for. The boxy and cropped silhouette is flattering on"

Red text below: "Same story – reads like marketing copy."


*Figure: yes — screenshot of a retailer product listing (Urban Outfitters, Champion tank top) with the generated description at right*

## Slide 10 — Lets get better at Fake Reviews

Title: "Lets get better at Fake Reviews" (as printed).

Three cream-coloured code-style JSON records in monospace small print, each with the keys "instruction", "input" and "output" (the "input" holds a product description copied from a retailer listing, the "output" holds a review written by a person). Transcribed as legible at this resolution:

Record 1: "instruction": "Write a positive review for the following product", "input": "The Workwear Pocket Long Sleeve T-Shirt.Wear it as a layer, wear it by itself, wear it with pride. It's as rugged and dependable as your favorite hammer offering 6.75-ounce, 100% cotton jersey knit, a generous left-chest pocket, side-seamed construction to minimize twisting and crew neck comfort. Made of 6.75-ounce, 100% cotton jersey our long sleeve pocket t-shirt makes for perfect workwear layered or worn alone. It features a crew neck with a generous left chest pocket and side-seamed construction to minimize twisting. Ash fiber content is 99%/1% polyester. Heather gray fiber content is 90% cotton/10% polyester. 100% Cotton Imported Machine Wash Long-sleeve t-shirt in midweight jersey featuring spade patch pocket at chest with logo tag Ribbed cuffs Original fit Package Dimensions:\n  \n12.4 x 7.6 x 1.8 inches Shipping Weight:\n  \n1 pounds (View shipping rates and policies)", "output": "The Best! Carharrt t-shirts are my husband's favorite.  They fit well, the fabric is very substantial, they wear well, and he especially likes the pocket. He wears these alone, with a light t-shirt under, or under a long sleeve button down Carharrt work shirt."

Record 2: "instruction": "Write a neutral review for the following product", "input": "These overalls are made of 12-ounce, firm-hand 100 percent ring-spun cotton duck and have adjustable front-elastic suspenders, a multi-compartment bib pocket, a zipper fly, two large lower-front pockets a left-leg hammer loop, double-front knee construction with cleanout bottoms that accommodate knee pads, two reinforced back pockets, triple-stitched main seams and metal rivets at vital stress points. Care-Do not bleach Our duck bib overall is made of heavy 12-ounce, cotton duck. It features a variety of tool and utility pockets as well as a hammer loop. Double knees can accommodate knee pads and feature cleanout bottoms. Back pockets are reinforced for durability. 100% Cotton Imported Machine wash warm with like colors, Do not bleach, Tumble dry medium remove promptly Bib overall in 12-ounce heavyweight cotton duck featuring hammer loop and reinforced paneling at knees Side utility pockets and bib pocket with logo patch pocket double knees Kindly refer last Image for sizing details Multiple tool pockets Package Dimensions:\n  \n15.6 x 6.3 x 3.5 inches Shipping Weight:\n  \n2.95 pounds (View shipping rates and policies)", "output": "... a little leary about a new pair being as good because. I have a 15 year old pair of these that got a hole burned through them when using a cut off saw and was a little leary about a new pair being as good because... well, I am in my 50's and have seen the vast majority of consumer/professional goods turn to crap. So far so good."

Record 3: "instruction": "Write a negative review for the following product", "input": "The iconic, timeless chuck taylor all star sneaker. Perfect in its simplicity since 1917. Lightweight, breathable canvas construction. Vulcanized rubber sole delivers durable traction. An ortholite insole cushions each and every step. Versatile, fashionable, and they keep looking better every time you wear them. Ortholite is a trademark of O2 partners, llc. 100% Textile Imported Rubber sole Shaft measures approximately low-top from arch Lace-up, low-top sneaker OrthoLite insole for cushioning Medial eyelets for airflow Canvas upper Product Dimensions:\n  \n5 x 5 x 0.7 inches Shipping Weight:\n  \n12 ounces (View shipping rates and policies)", "output": "I love the classic look of Chuck Taylors I love the classic look of Chuck Taylors, but wow, these are legitimately the most uncomfortable shoes I've ever worn. I wore them recently to my work where I'm on my feet for 10 hours and am moving around a lot and lifting stuff, climbing up and down ladders, and by the end of the night I was limping and my feet were throbbing. I have another pair of Chuck Taylors, mid tops, and those have some padding in the sole and I can wear them all night and feel fine. I guess I'm going to have to get some inserts for these if I want to wear them longer than an hour."

(Small print; a few characters in the long "input" strings may be misread, the instruction and output texts and the dimensions are as printed. The three records are wrapped in braces with trailing commas.)

*Figure: yes — three JSON training records with instruction, input (product description) and output (review) fields*

## Slide 11 — The Llama 2 family of open LLMs are widely used for fine-tuning

Title (two lines): "The Llama 2 family of open LLMs are widely used for fine-tuning".

A screenshot of Meta's Llama 2 web page. Left: small label "Large language model", heading "Llama 2: open source, free for research and commercial use", paragraph "We're unlocking the power of these large language models. Our latest version of Llama – Llama 2 – is now accessible to individuals, creators, researchers, and businesses so they can experiment, innovate, and scale their ideas responsibly.", and a blue "Download the model" button. Right: a pale-blue illustration of a translucent panel with a label "Llama 2" and a card listing "Llama-2-7B", "Llama-2-13B", "Llama-2-70B". Below, in black: "https://llama.meta.com/llama2". Printed page number 11.


*Figure: yes — Meta Llama 2 web page screenshot listing Llama-2-7B, 13B and 70B*

## Slide 12 — Let’s first understand how “hard” it is to build the biggest model in the family: Llama-2-70b

Title (two lines): "Let’s first understand how “hard” it is to build the biggest model in the family: Llama-2-70b".

Same as slide 11 (same screenshot and URL; printed page number 12), with one addition: a thin dark red line with an arrowhead runs from the end of the title ("Llama-2-70b") down to the "Llama-2-70B" entry in the illustration.


*Figure: yes — same Llama 2 screenshot with a dark red arrow from the title to the Llama-2-70B label*

## Slide 13 — How hard is it to train Llama-2-70b?

Title: "How hard is it to train Llama-2-70b?"

- The model is gigantic
  - 70 billion parameters x 2 bytes per parameter x ~3-6 = 420-840GB ("420-840GB" in red)
    - 2 bytes/parameter assuming we use “fp16” (i.e., 16-bit numbers)
    - 3-6x multiplier for each parameter since we need to store gradient, optimizer states etc. in addition to the weights themselves, and some require higher precision (all red except the closing "(more on this later)" in black)
- An A100 (or H100) GPU has 80GB of RAM and so we need between 6 and 11 GPUs to accommodate 420-840GB ("we need between 6 and 11 GPUs to accommodate 420-840GB" in red)

Printed page number 13.

*Figure: no*

## Slide 14 — How hard is it to train Llama-2-70b?

Title: "How hard is it to train Llama-2-70b?"

- Llama-2-70b was trained on 2 trillion ($2 \times 10^{12}$) tokens
- An A100 GPU can (optimistically) process just about 400 tokens/GPU per second
- So 11 GPUs will take $2 \times 10^{12}/(11 \times 400)$ seconds which is about 5,261 days. A bit long (smiley-face emoji)
- Let’s say we want to do it in about a month.
  - With 2048 GPUs, we would need about 28 days
  - A simple cost estimate (at \$2.5/ GPU-hr) for this training run is \$4M
  - We would expect the actual cost to be a lot higher since it takes multiple runs to get things right

Printed page number 14. (Figures copied as printed.)

*Figure: no*

## Slide 15 — To fine-tune Llama-2-70b with fewer resources, we need to do two things

Title (two lines): "To fine-tune Llama-2-70b with fewer resources, we need to do two things" ("fewer resources" in red).

- Reduce the size of the dataset
- Reduce the memory required so that we can process the data using fewer GPUs (ideally, just one GPU)

Printed page number 15.

*Figure: no*

## Slide 16 — To fine-tune Llama-2-70b with fewer resources, we need to do two things

Title (two lines): "To fine-tune Llama-2-70b with fewer resources, we need to do two things" (all black).

A green check-mark emoji sits at the left of the first bullet.

- Reduce the size of the dataset: We are in luck here! Finetuning datasets can be much smaller than the original corpus used to train the LLM in the first place
  - The famous Alpaca fine-tuning dataset has 50k instruction-answer pairs at 4096 tokens each, so ~ 200M tokens in total, which is <<<< 2T tokens used to train Llama-2-70b
  - Assuming we can process 400 tokens/GPU per second, 7 GPUs will take 2 x 108/(7x400) seconds which is only about 20 hours (as opposed to 28 days for training Llama-2-70b)!
- Reduce the memory required so that we can process the data using fewer GPUs (ideally, just one GPU)

Printed page number 16. (The expression is printed "2 x 108/(7x400)", apparently a flattened 2 x 10^8, with no superscript visible in the render; copied as printed.)

*Figure: no*

## Slide 17 — Next, let’s discuss how to reduce the memory requirements

Title (two lines): "Next, let’s discuss how to reduce the memory requirements".

Same text as slide 16's two bullets (including the "2 x 108/(7x400)" as printed), with these changes: the title is different, the green check-mark is gone, and the whole first bullet with its two sub-bullets is faded to light grey. Only the second bullet stays in black: "Reduce the memory required so that we can process the data using fewer GPUs (ideally, just one GPU)". Printed page number 17.

*Figure: no*

## Slide 18 — What consumes memory?

Title: "What consumes memory?"

Table (dark red header row; "Optimized" column empty):

| | Naive Memory Usage | Optimized |
| --- | --- | --- |
| Model parameters | #Params x 2Bytes = 140GB | |
| Gradient computations | Same as above = 140 GB | |
| Optimizer state | 1-4x the memory needed for parameters = 140-560GB | |
| Total | 420-840GB | |

A thick dark red bracket-style arrow at the left runs up from beside the two bullets below to the "Optimizer state" row, pointing into it.

- It turns out that state-of-the-art optimizers (like Adam) need to store information related to past gradients
- In fact, this requires approximately 1-4x the amount of memory in addition to the parameters themselves

Printed page number 18.

*Figure: yes — memory table with an empty Optimized column and an arrow from the bullets to the Optimizer state row*

## Slide 19 — What consumes memory?

Title: "What consumes memory?"

Same table as slide 18 with the "Optimized" column now filled, and the two bullets and left arrow removed:

| | Naive Memory Usage | Optimized |
| --- | --- | --- |
| Model parameters | #Params x 2Bytes = 140GB | 140GB\* |
| Gradient computations | Same as above = 140 GB | ~ zero by computing ’just in time’\*\* |
| Optimizer state | 1-4x the memory needed for parameters = 140-560GB | We will show how to reduce this to ~ zero! (in red) |
| Total | 420-840GB | 140GB |

Footnotes: "\*This can be reduced with ‘quantization’ but this often can result in performance degradation" and "\*\* This requires an old trick called ‘gradient checkpointing’ that is beyond the scope of our discussion here." Printed page number 19.

*Figure: yes — memory table with naive versus optimized columns*

## Slide 20 — We will fine-tune only the matrices inside the causal self-attention blocks, and will keep everything else frozen

Title (two lines): "We will fine-tune only the matrices inside the causal self-attention blocks, and will keep everything else frozen".

Same diagram as slide 5 (input arrows, grey positional embeddings, "TCE", blue contextual embeddings, "Dense (ReLU)", yellow rows, four "SM" boxes, final arrow) and the same seven-item italic list, with these changes: only one dashed red rounded outline remains, around the TCE box alone (the outlines around the embeddings, Dense box and SM boxes are gone); and in the list only item 3, "Matrices $A^K, A^Q, A^V$ for each attention head (inside TCE)", is in red, while the other six items (1, 2, 4, 5, 6, 7) are black. Item wording is the same as on slide 5. Printed page number 20.

*Figure: yes — the pipeline with only the TCE block outlined and list item 3 (attention matrices) highlighted in red*

## Slide 21 — Lets consider the weight matrix A^K

Title: "Lets consider the weight matrix $A^K$" (K as a superscript).

- In Llama-2-70b, this is a 8096x8096 matrix (one for each self attention layer)
  - ~64 M parameters to store per $A^K$

Printed page number 21. (Figures as printed: "8096", not 8192.)

*Figure: no*

## Slide 22 — An update to A^K can be thought of as the sum of the original A^K and the change ΔA^K

Title (two lines): "An update to $A^K$ can be thought of as the sum of the original $A^K$ and the change $\Delta A^K$".

Two 5x5 tables of numbers, side by side, with large labels beneath: $A^K$ under the left one, "+" between, $\Delta A^K$ under the right one.

Left table ($A^K$):

| 1.1 | 0.2 | -0.7 | 2.3 | -0.3 |
| --- | --- | --- | --- | --- |
| -0.4 | -0.9 | 2.0 | 3.7 | 2.1 |
| 0.4 | -1.2 | 0.3 | 2.8 | 1.3 |
| 2.3 | -0.3 | -0.9 | 2.0 | 0.1 |
| 0.2 | -0.7 | -0.5 | -1.2 | 0.3 |

Right table ($\Delta A^K$):

| 0.01 | 0.02 | 0.04 | -.01 | -.03 |
| --- | --- | --- | --- | --- |
| -.04 | -.09 | -.04 | -.09 | 0.02 |
| 0.01 | -.02 | 0.04 | -.08 | .07 |
| -.05 | -.01 | -.09 | .02 | 0.01 |
| 0.07 | -.01 | -.05 | -.12 | .03 |

- In general, the change matrix $\Delta A^K$ will be as big as $A^K$
- Can we make it smaller? (red)

Printed page number 22.

*Figure: yes — two 5x5 number tables, A^K plus the much smaller-valued change matrix delta A^K*

## Slide 23 — [Idea: force the change matrix to be simple]

No title text; a light-bulb emoji sits at the upper left. A thin dark red rectangle holds large italic text:

"A finetune will likely make very small changes to the original model weights. This suggests that we can “force” the change matrix $\Delta A^K$ to be “simple” and it will still get the job done."

Printed page number 23.

*Figure: no*

## Slide 24 — We can force ΔA^K to be “simple” by forcing to be “low rank”

Title (two lines): "We can force $\Delta A^K$ to be “simple” by forcing to be “low rank”".

Diagram of matrix shapes, all drawn as solid dark red blocks. At left a large square labelled "8096" along the top and "8096" down its left side. Then the symbol "=~". Then a tall thin vertical bar with "8096" beside it and a red "2" above it, then an "x", then a flat wide horizontal bar with "8096" above it and a red "2" at its right end. So the 8096x8096 matrix is approximated by (8096x2) times (2x8096).

Printed page number 24.

*Figure: yes — large 8096 by 8096 block approximately equal to a tall 8096 by 2 bar times a flat 2 by 8096 bar*

## Slide 25 — The number of parameters in “simple” ΔA^K

Title (two lines): "The number of parameters in “simple” $\Delta A^K$".

Same matrix diagram as slide 24 (8096 by 8096 block "=~" 8096 by 2 bar "x" 2 by 8096 bar, with the same "8096" and red "2" labels), with these additions, as three bullets at lower right:

- Recall that $A^K$ has 64M parameters (=8096x8096)
- But simple $\Delta A^K$ has just 2 x 8096 = 16,192 parameters i.e., just 0.02% of the original 64M! (all red; "just 0.02% of the original 64M!" in bold; the text runs past the right edge of the slide after "i.e.,")
- This idea is called Low Rank Adaptation (LORA)\*

Footer link text at lower left (underlined, gold): "LoRA: Low-Rank Adaptation of Large Language Models" (the asterisk in the bullet refers to this reference). Printed page number 25.

*Figure: yes — the 8096 by 8096 block approximated by a 8096 by 2 bar times a 2 by 8096 bar, with parameter counts*

## Slide 26 — LoRA Optimization

Title: "LoRA Optimization".

- Freeze all base model parameters
- Initialize $\Delta A^K$ $\Delta A^Q$ $\Delta A^v$ to zero in each self attention layer (the third matrix's superscript is printed as lower-case "v")
- Update $\Delta A^K$ $\Delta A^Q$ $\Delta A^v$ (via SGD as usual) by updating the two “skinny” matrices for each

The last bullet is cut off after "for each" on this page (a build step: the sentence is unfinished). Below it, two dark red bars are drawn, one thin vertical bar and one thin horizontal bar beside it (the tall skinny and flat skinny matrix shapes, without labels). Printed page number 26.

*Figure: no*

## Slide 27 — What does memory look like now?

Title: "What does memory look like now?"

Table (dark red header row):

| | Naive Memory Usage | Optimized |
| --- | --- | --- |
| Model Parameters | #Params x 2Bytes = 140GB | The same\* |
| Gradient Computations | ~#Neurons ~ #Params ~140GB | Roughly zero by computing these ’just in time’\*\* |
| Optimizer State | #Params x 1-4Bytes x 2 = 140-560GB | Using LoRA, we only need to optimize the two “skinny” matrices so # of parameters ~0 (red) |
| Total | 420-840GB | 140GB |

Text below: "Now, Llama-2-70b can be comfortably be finetuned on 2 GPUs and Llama-2-7b and Llama-2-13b can comfortably be finetuned on a single GPU."

(The asterisks in the table are as printed; this page prints no footnote text for them.) Printed page number 27.

*Figure: yes — memory table naive versus optimized with LoRA*

## Slide 28 — Colab (if time permits)

Title: "Colab (if time permits)". The body is empty. (The link target is not printed on the page.) Printed page number 28.

*Figure: no*

## Slide 29 — The results of Supervised Finetuning

Title: "The results of Supervised Finetuning".

Left: the same Urban Outfitters product-page screenshot as slide 9 (Champion crew neck jersey tank top; third-party content, described not copied). Right, normal-weight black text: the same description as slide 9 ("Champion's Crew Neck Jersey Tank Top is a great addition to any wardrobe. The boxy and cropped silhouette is flattering on all body types, and the cotton fabrication is soft and comfortable. The crew neckline is classic and timeless, and the sleeveless design makes it perfect for warmer weather. The Heritage style tee is an exclusive from Urban Outfitters, and the 100% cotton fabrication and machine washability make it easy to care for. The boxy and cropped silhouette is flattering on"). A thick dark red downward arrow at far right points from that text to a bold black block below it:

"I love this tank top! It's so soft and comfy, and the fit is perfect. I'm 5'7" and 120 lbs, and I got a small. It's not too tight, but it's not too loose either. It's a great tank top to wear to the gym or just around the house. I would definitely recommend this tank top to anyone who is looking for a comfy tank top to wear around the house or to the gym."


*Figure: yes — Urban Outfitters listing, the old description, and an arrow to the fine-tuned model's bold review text*

## Slide 30 — The results of Supervised Finetuning

Title: "The results of Supervised Finetuning".

Left: the same rag & bone product-page screenshot as slide 7 (third-party content, described not copied). Right, two black text blocks:

"Write a review:" then "These are the best jeans I've ever owned. I'm 5'10" and 160 lbs. I bought a size 32 and they fit perfectly. I've been wearing them for a few months now and they still look brand new. I've washed them a few times and they've held up great. I've never owned a pair of jeans that fit this well and look this good after a few months of wearing them. I'll definitely be buying more of these in the future."

"Write a horrible review:" then "These are the worst jeans I've ever owned. I'm 5'10" and 160 lbs. I ordered a 32 and they fit like a 30. They're too tight in the thighs and too loose in the waist. I'm going to return them and try a 30, but I'm not optimistic. I've been wearing Levi's 501s for 30 years and these are not even close to the quality of those jeans. I'm going to stick with Levi"


*Figure: yes — rag & bone listing with a positive and a horrible generated review*

## Slide 31 — The effort-benefit curve for adaptation strategies depends on the size of the base LLM

Title (two lines): "The effort-benefit curve for adaptation strategies depends on the size of the base LLM".

A line chart (a screenshot of a social-media post's figure). Y axis: "task accuracy", from "0%" at the bottom to "100%" at the top (a horizontal line marks 100%). X axis: an arrow labelled "effort/ complexity" at the right, with four coloured vertical bands left to right: light blue "zero-shot prompts" (with the in-band label "prompt engineering"), pink "few-shot prompts", green "retrieval-augmented few-shot prompting", and light yellow "finetuning" (with the in-band label "model training (finetuning, RLHF) data engine"). Two series with arrowheads at each segment end:

- Blue, labelled "big base model": starts at 0% at the left, rises steeply to about 50% (mid-blue band), then to about 55% at the band's end, jumps to about 65% at the start of the pink band, rises to about 72% at its end, jumps to about 78% at the start of the green band, reaches about 83% at its end, rises to about 90% early in the yellow band, then creeps to just under 100% at the right end.
- Red, labelled "small base model": starts at 0% and stays near 0% through the blue band, rises slowly to about 10% at the end of the pink band, about 15% at the end of the green band, then rises more steeply through the yellow band to about 33% at the right end.

Source line at bottom: "https://twitter.com/karpathy/status/1655994367033884672?s=20". Values are read off the plot approximately.


*Figure: yes — effort-benefit curve: task accuracy versus effort for a big base model and a small base model across prompting, retrieval, and finetuning stages*

## Slide 32 — We have covered all these topics!

Title (left side, large, three lines): "We have covered all these topics!" The slide's dark red title bar is partly hidden behind the figure, which fills the right two-thirds of the page.

Diagram (a figure from a survey paper, a flowchart): a top box "How LLMs Are Built?" with a downward arrow through a vertical chain of nine peach boxes: "Data Cleaning", "Tokenizations", "Positional Encoding", "LLM Architectures", "Model Pre-training", "Fine-tuning and Instruction Tuning", "Alignment", "Decoding Strategies", "Cost-Effective Training/Inference, Adaptation & Compression". Each is joined to a pale green dashed box of examples, alternating left and right:

- Data Cleaning (left): Data Filtering (Removing Noise, Handling Outliers, Addressing Imbalances, Text Preprocessing), Deduplication.
- Tokenizations (right): BytePairEncoding (circled in red), WordPieceEncoding, SentencePieceEncoding.
- Positional Encoding (left): Absolute Positional Embeddings (circled in red), Relative Positional Embeddings, Rotary Position Embeddings, Relative Positional Bias.
- LLM Architectures (right): Encoder-Only and Decoder-Only (circled together in red), Encoder-Decoder, "...".
- Model Pre-training (left): Masked Language Modeling and Causal Language Modeling (circled together in red), Next Sentence Prediction, Mixture of Experts.
- Fine-tuning and Instruction Tuning (right): Supervised Fine-tuning (circled in red), General Fine-tuning, Multi-turn Instructions, Instruction Following.
- Alignment (left): Supervised learning, Reinforcement Learning from Human Feedback (circled in red), Direct Preference Optimization, Kahneman-Tversky Optimization.
- Decoding Strategies (right): Greedy Search, Beam Search, Top-k Sampling, Top-p Sampling (all four circled together in red).
- Cost-Effective Training/Inference, Adaptation & Compression (left): Optimized Training (Zero Redundancy Optimizer, Receptance Weighted Key Value), Low-Rank Adaption (circled in red), Knowledge Distillation, Quantization.

Bottom left: underlined gold link text "Large Language Models: A Survey" and, in red, "(February 20, 2024!)".


*Figure: yes — survey flowchart of how LLMs are built, with the topics covered in the course circled in red*

## Slide 33 — [OCW end page]

OCW end page. Text: "MIT OpenCourseWare", "https://ocw.mit.edu", "15.773 Hands-on Deep Learning", "Spring 2024", "For information about citing these materials or our Terms of Use, visit: https://ocw.mit.edu/terms."

*Figure: no*
