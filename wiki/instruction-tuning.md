# Instruction tuning

A base [large language model](large-language-models.md) such as GPT-3 is trained only to predict the next
word. That makes it a superb autocomplete and a poor assistant. Asked "help me write a short note to
introduce myself to my neighbor", GPT-3 answers "what is a good introduction to a resume.", because much of
its training text is lists and it takes a request for the start of one ([lecture 10](10-adapting-llms.md),
slides 7–13; ≈3:55–5:28). It "wasn't *explicitly* trained to generate good answers to user
instructions". **Instruction tuning** trains it explicitly. It is how OpenAI turned GPT-3 into
**GPT-3.5/InstructGPT**, and the same recipe applied to conversations produced **ChatGPT** (slides 15–51).

A note on the name. OpenAI's approach has two parts, (1) supervised fine-tuning and (2) reinforcement
learning from human feedback. "'Instruction Tuning' usually refers to just (1) but we will use it as a
shorthand to refer to the full approach" (lecture 10, slide 16's footnote, citing
<http://arxiv.org/abs/2203.02155>, the paper "Training language models to follow instructions with human
feedback"). "This is not a consistent piece of terminology" (≈6:14). This page follows the lecture.

## The recipe in five steps

Lecture 10's slide 52 puts it in a nutshell:

1. Start with a base LLM (like GPT-3) that has been trained to predict the next word.
2. Get humans to write responses to thousands of instructions. Fine-tune the base LLM with this data.
3. Generate multiple responses to each instruction using the fine-tuned LLM. Ask humans to rank the
   responses. Do this for several thousand instructions.
4. Use this data to build a Reward Model.
5. Using the Reward Model, train the fine-tuned LLM further using Reinforcement Learning with several
   thousand questions.

![Lecture 10, slide 52: five dark red chevrons labelled Step 1 to Step 5, each beside a box with one step: start with a base LLM trained to predict the next word; have humans write responses to instructions and fine-tune; generate several responses and have humans rank them; build a Reward Model; train further with reinforcement learning using the Reward Model.](../raw/images/10-adapting-llms/slide-52.png)

*Lecture 10, slide 52 — instruction tuning in a nutshell. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

## Supervised fine-tuning (SFT)

People wrote high-quality answers to instructions, about 12,500 instruction-answer pairs (lecture 10,
slides 17–18; ≈7:03). GPT-3 was then trained further on them "using next word prediction as before, no
difference" (≈7:48). The input is the instruction followed by the answer, and the labels are the answer
shifted by one word. After "… in a few sentences." the model must predict "People", after "People" it
must predict "went", and so on (slide 20). This step is **supervised fine-tuning**, and "GPT-3 did much
better on instructions" (slide 21). It is the same operation as fine-tuning any LLM on domain examples
(slide 84's footnote; see [adapting LLMs](adapting-llms.md#fine-tuning)). Its limit is cost: "writing
high-quality answers to thousands of instructions is difficult and expensive" (slide 22).

## Ranking, and the Reward Model

"What's easier than writing a good answer? *Ranking* answers written by somebody else!" (slides 23–24).
GPT-3 itself writes the answers to be ranked. Because it samples its next word, it can give several
different answers to one question. "The fact that we had these stochastic outputs because of sampling is
now a feature, not a bug", with "a nice temperature of 1 or 1.1 or something so that it's nice and random"
(slides 25–27; ≈10:08; see [decoding and sampling](decoding-and-sampling.md)). OpenAI collected 33,000
instructions, generated several answers to each, and had humans rank them from most to least helpful
(slide 28). For two answers, each data point is [Instruction, Preferred answer, Other answer] (slide 29).

The **Reward Model** is a language model with a single numerical output: how good an answer is for that
instruction (slide 30; ≈12:27–13:13). Two exact copies score the instruction with the preferred answer,
$r_P$, and with the other answer, $r_O$, and are trained with SGD on the loss

$$\text{Loss} = -\log(\sigma(r_p - r_O))$$

where $\sigma$ is the sigmoid (slides 31–32; ≈13:13–15:34). The loss is small when the preferred answer's
rating is well above the other's and large when it is below (worked through on the
[lecture 10 page](10-adapting-llms.md#the-reward-model-and-its-loss), as the page's own check). Once
trained, the Reward Model rates any instruction-answer pair: GPT-3's off-topic reply to the moon-landing
instruction gets 1.5, the human-written answer 3.2 (slide 33). "In essence, the Reward Model has learned how
humans rank responses" (slide 34), and it can rate tens of thousands of answers that no human ranked.

Both answers are needed, "much like if you want to build a dog-cat classifier, you have to show pictures
of both" (≈14:46). Labelers follow guideline documents "100, 200 pages long" (≈15:34). Giving the same
question to several labelers and keeping only the rankings they agree on reduces disagreement between
labelers, but not a bias they all share. "The reward model has probably very faithfully learned all the
biases of the human labelers" (≈19:27–20:13).

## Reinforcement learning from human feedback (RLHF)

GPT-3 answers an instruction, the Reward Model rates the answer, and the rating is used "to nudge the
internal weights of GPT-3 in the right direction" by **reinforcement learning** (slides 35–39;
≈17:10–17:56). Slide 38's footnote names the algorithm, Proximal Policy Optimization, which the course does
not cover. OpenAI used about 31,000 questions for this step (slide 39). Steps 2 and 3 together are
**Reinforcement Learning from Human Feedback**, "because we use reinforcement learning. And since humans
rank the answers which led to the building of the reward model, we get human feedback" (slide 41;
≈17:56).

![Lecture 10, slide 39: the instruction "Explain the moon landing to a six year old" goes into GPT-3, whose answer goes with the instruction into the Reward Model, which outputs the rating 1.5; a dash-dot arrow curves from the rating back to GPT-3. Red text: "OpenAI used ~31,000 questions for this step".](../raw/images/10-adapting-llms/slide-39.png)

*Lecture 10, slide 39 — the RLHF loop. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

## From GPT-3 to ChatGPT, and beyond

The three steps took GPT-3 to GPT-3.5/InstructGPT, whose answer to the neighbor prompt is a proper note
(slides 40, 44–45). ChatGPT followed "the exact same playbook", but used "conversations as the training
data" instead of single instruction-answer pairs. "That is the only difference" (slides 46–47; ≈24:48).
Trained on conversations, it can take follow-on requests such as "can you make it more formal?" (slides
49–50). Slide 51 continues the lineage to GPT-4, ChatGPT Plugins and GPT-4 V.

The thumbs-up and thumbs-down buttons in ChatGPT collect data for the ranking step: "all the models have
the thumbs up, thumbs down stuff going on somewhere" (slide 42; ≈11:40). None of it changes the model
during your conversation. The signals go into training logs, and the model is periodically retrained
(≈27:51–28:39).

## Helper LLMs

Steps 2 and 3 are the expensive, human ones. Researchers now use LLMs for them as well: "an
instruction-following LLM" writes the responses and "a helper LLM" ranks them (lecture 10, slides 53–54;
≈28:39). OpenAI did not do this for the original ChatGPT, but there are now enough good LLMs to.

## Instruction tuning as adaptation

The lecture presents instruction tuning as the first example of **adaptation**. A base LLM had to be
adapted "with high-quality data" before it was useful at following instructions, and the same holds for a
business use case (slide 56). That leads to the ladder of prompting, retrieval and fine-tuning on the
[adapting LLMs](adapting-llms.md) page. Slide 114 places "Supervised Fine-tuning" and "Reinforcement
Learning from Human Feedback" among the steps of building an LLM that the course covered.

## Where this comes up

- [Lecture 10](10-adapting-llms.md): GPT-3's failure to follow instructions, supervised fine-tuning, the
  Reward Model and its loss, RLHF, InstructGPT and ChatGPT, helper LLMs.
- See also [large language models](large-language-models.md), [adapting LLMs](adapting-llms.md),
  [decoding and sampling](decoding-and-sampling.md) (sampling several answers to rank) and
  [loss functions](loss-functions.md).
