# Adapting LLMs: prompting, fine-tuning and LoRA

"'Adaptation' is the process of taking a base (aka foundation …) LLM and tailoring it for a specific
task" ([lecture 10](10-adapting-llms.md), slide 57). A base [large language model](large-language-models.md)
knows a lot from pretraining, but a narrow business use case, such as "a medical use case, a legal use
case", needs it adapted "with business domain-specific data" (slide 56; ≈29:24). Turning GPT-3 into
ChatGPT by [instruction tuning](instruction-tuning.md) was itself an adaptation, and the lecture
generalizes from it.

## The ladder of adaptation

There are four rungs, each heavier than the one below (lecture 10, slide 58; ≈29:24–30:13):

| Rung | What changes |
| --- | --- |
| Zero-shot prompting | Prompt = a clear instruction |
| Few-shot prompting | Prompt = a few examples |
| Retrieval-augmented generation | Prompt = question + relevant chunks of text |
| Fine-tuning | The LLM is further *trained* with examples |

![Lecture 10, slide 58: "The Ladder of LLM Adaptation", a curved arrow rising to the right with four dots of increasing size: Zero-Shot Prompting (Prompt = A clear instruction), Few-Shot Prompting (Prompt = A few examples), Retrieval-Augmented Generation (Prompt = Question + relevant chunks of text) and Fine-Tuning (The LLM is further trained with examples).](../raw/images/10-adapting-llms/slide-58.png)

*Lecture 10, slide 58 — the first three rungs change only the prompt; fine-tuning changes the weights. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

The first three leave the model untouched: "You're just using it as a black box" (≈1:02:40). The "shot"
in zero-shot and few-shot "is a synonym for 'example'" (≈30:13). The third rung has its own page,
[retrieval-augmented generation](retrieval-augmented-generation.md).

## Zero-shot prompting and prompt engineering

**Zero-shot prompting** asks for what you want, clearly, with no examples (lecture 10, slides 59–61;
≈30:13–31:47). Rather than building a special-purpose classifier to spot reviews that describe a product
defect, give the model "Tell me if a product defect in being described in this product review:" and a
one-star review ("The curve of the back of the chair does not leave enough room to sit comfortably"), and
it answers "Yes, this is a product defect." This "actually works remarkably well", and better with bigger
models.

Designing the prompt so that the model responds as you want is **prompt engineering** (slide 62). ChatGPT
often gets "what is the fifth word of this sentence?" wrong. Asked first to list all the words and then
give the fifth, it gets it right (slides 63–64). The rule: "be very, very prescriptive as to what you want
it to do and break down all the steps. Don't make it guess things" (≈32:33). Researchers keep
"discovering" strategies that work widely, such as "Let's think step by step" (slide 65). On the GSM8K
test with PaLM 2-L, "Take a deep breath and work on this problem step-by-step" scored 80.2 against 71.8
for "Let's think step by step" (slide 66). ChatGPT already applies such instructions through a hidden
**system prompt** that your prompt is appended to (≈32:33). Asking for a well-defined output format,
such as markdown or JSON, can also raise the chance of a right answer (≈1:01:54).

## Few-shot prompting and in-context learning

**Few-shot prompting** puts a few examples of the task in the prompt (lecture 10, slides 68–73;
≈34:06–34:52). The grammar corrector from the GPT-3 paper shows three "Poor English input" / "Good
English output" pairs ("I eated the purple berries." → "I ate the purple berries.") and ends with a poor
sentence alone. GPT-3 corrects "in another project" to "on another project".

"The ability of LLMs to learn how to do a task with just instructions/examples in the prompt is called
**In-Context Learning**" (slide 74). It covers both zero-shot and few-shot prompting. GPT and GPT-2 could
not do it. It "was new in GPT-3 and what they call an emergent capability. It was completely
unanticipated by the people who built it" (≈34:52).

## Fine-tuning

**Fine-tuning** trains the LLM further, so "the weights are going to change" (lecture 10, slides 82–85;
≈1:02:40–1:03:26). Take a causal LLM and train it on domain-specific input-output examples by next word
prediction, the input followed by the shifted output. This "is essentially the Supervised Fine-tuning step
we discussed earlier to transform GPT-3 to GPT-3.5" (slide 84's footnote), and "exactly like we saw with
BERT … And even with ResNet" (see [transfer learning](transfer-learning.md)). In full fine-tuning every
weight is updated: positional embeddings, stand-alone embeddings unless frozen, the attention matrices
$A^K$, $A^Q$, $A^V$, layer norm parameters, feed-forward weights, and the final dense and softmax layers
(slide 85).

![Lecture 10, slide 85: the causal LLM pipeline (input, positional input embeddings, TCE, contextual embeddings, Dense, softmax boxes) with red dashed outlines around the embeddings and TCE, the Dense box and the softmax boxes, above a numbered list of the six kinds of weights that are updated.](../raw/images/10-adapting-llms/slide-85.jpg)

*Lecture 10, slide 85 — what full fine-tuning updates. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

The example is generating synthetic product reviews (slides 86–93; ≈1:03:26–1:06:34). Asked to write a
positive review from a product page, an LLM produces marketing copy ("Timeless. Authentic. Iconic. …").
Fine-tuned on a few hundred [instruction, product description, product review] examples, it writes
"These are the best jeans I've ever owned. I'm 5'10" and 160 lbs. …", or, asked for a horrible review,
"These are the worst jeans I've ever owned. …". The scale is the point. Pretraining took billions of
sentences and instruction tuning tens of thousands of examples, "to make it work for your narrow business
use case, you only had to spend a couple of hundred examples" (≈1:06:34). To make a model generally
knowledgeable about a company, plain next word prediction on its text will do. To make it follow
instructions on a specific problem, "you need supervision" (≈1:07:23–1:08:08).

### Why a large LLM is hard to fine-tune

Small models such as GPT-2 are easy to fine-tune; large ones raise "computational challenges" (slide 94).
The lecture works through **Llama-2-70b**, the largest of the open Llama 2 family, "widely used for
fine-tuning" (slides 95–103):

- 70 billion parameters, times 2 bytes each, times a multiplier of ~3–6 (for gradients and optimizer states) is
  420–840GB, which needs 6 to 11 GPUs of 80GB (slide 97).
- Training it on 2 trillion tokens at about 400 tokens per GPU per second would take 11 GPUs about 5,261
  days, or 2,048 GPUs about 28 days, for an estimated \$4M (slide 98).

Fine-tuning needs much less data: the Alpaca dataset's 50k instruction-answer pairs are about 200M tokens,
about 20 hours on 7 GPUs (slide 100). Memory is the remaining problem. The model parameters take 140GB,
the gradients another 140GB (reducible to about zero by **gradient checkpointing**), and the optimizer
state 1–4 times the parameters, 140–560GB, because optimizers like Adam store information about past
gradients (slides 102–103). The lecturer read older numbers off the screen in class (a multiplier of 3 to
4, 420–560 GB, 6 to 7 GPUs). The published slides give the ones above.

## LoRA: low-rank adaptation

LoRA removes the optimizer state's share by training very few parameters (lecture 10, slides 104–111;
≈1:13:29–1:16:35). Fine-tune "only the matrices inside the causal self-attention blocks, and … keep
everything else frozen" (slide 104). Llama-2-70b's key matrix $A^K$ in each self-attention layer combines
64 heads of $8192 \times 128$ into one $8192 \times 8192$ matrix, about 64M parameters (slide 105).
Fine-tuning turns $A^K$ into $A^K + \Delta A^K$, and in general the change $\Delta A^K$ is as big as $A^K$
(slide 106).

But "a finetune will likely make very small changes to the original model weights", so force
$\Delta A^K$ to be "simple" (slide 107): **low rank**. Approximate it by an $8192 \times 2$ matrix times a
$2 \times 8192$ matrix, "two thin, skinny matrices" (slide 108; ≈1:15:02). The change then has "just 4 x
8192 = 32,768 parameters i.e., just 0.05% of the original 64M!" (slide 109). In class the lecturer gave the
earlier version's 16,192 and 0.02%, which deck 10.5's slide 25 prints.

![Lecture 10, slide 109: a large 8192 by 8192 square approximately equal to a tall thin bar (2 wide) times a short wide bar (2 high), with bullets: A^K has 64M parameters (=8192 x 8192); simple ΔA^K has just 4 x 8192 = 32,768 parameters, just 0.05% of the original 64M; this idea is called Low Rank Adaptation (LORA).](../raw/images/10-adapting-llms/slide-109.png)

*Lecture 10, slide 109 — the change to each attention matrix is learned as two skinny matrices. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

The procedure (slide 110):

- Freeze all base model parameters.
- Initialize $\Delta A^K$, $\Delta A^Q$ and $\Delta A^V$ to zero in each self-attention layer.
- Update them, by SGD as usual, by updating only the two skinny matrices of each.

The optimizer state then shrinks to about zero, and the memory total falls from 420–840GB to 140GB.
"Now, Llama-2-70b can be comfortably be finetuned on 2 GPUs and Llama-2-7b and Llama-2-13b can
comfortably be finetuned on a single GPU" (slide 111). LoRA is "incredibly widely used in the industry"
(≈1:15:49). The lecture's LoRA Colab was left for a video walkthrough that is not part of this KB (slide
112). The term "parameter-efficient fine-tuning" (PEFT), in the video's catalog title, is not used in the
lecture.

## Which rung to use

Slide 113, not reached in class, makes the point with Andrej Karpathy's effort-benefit curve: how much
each rung buys depends on the size of the base model. A big base model gets most of the way with
prompting. A small one gains little from prompting and retrieval and needs fine-tuning (the chart is
excluded from OCW's licence; the [slide file](../raw/slides/10-adapting-llms.md) describes it).

## Where this comes up

- [Lecture 10](10-adapting-llms.md): the ladder, zero-shot and few-shot prompting, prompt engineering,
  in-context learning, fine-tuning for product reviews, the cost of Llama-2-70b, and LoRA. Its companion
  deck 10.5 ([slide file](../raw/slides/10-finetuning.md)) repeats the fine-tuning and LoRA slides in an
  earlier version.
- [Lecture 4](04-cnns-and-transfer-learning.md) and [lecture 8](08-transformers-and-self-supervised-learning.md):
  fine-tuning ResNet and BERT, the same operation on smaller networks (see
  [transfer learning](transfer-learning.md)).
- See also [instruction tuning](instruction-tuning.md),
  [retrieval-augmented generation](retrieval-augmented-generation.md) and
  [large language models](large-language-models.md).
