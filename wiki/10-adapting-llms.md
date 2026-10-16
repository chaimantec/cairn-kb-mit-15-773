# Lecture 10 — Large Language Models (2/2): Instruction Tuning, Prompting, RAG and Fine-Tuning

**Lecturer:** Rama Ramakrishnan ·
**Video:** [youtube.com/watch?v=d-tngNnaG4U](https://www.youtube.com/watch?v=d-tngNnaG4U) (77 min) ·
**Slides:**
[`mit15_773_s24_lec10.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)
(115 pages; transcribed in [`raw/slides/10-adapting-llms.md`](../raw/slides/10-adapting-llms.md)), and
the companion deck
[`mit15_773_s24_lec105.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec105.pdf)
("Lecture 10.5: Finetuning", 33 pages, not presented; transcribed in
[`raw/slides/10-finetuning.md`](../raw/slides/10-finetuning.md)) ·
**Transcript:** [`raw/transcripts/10-adapting-llms.md`](../raw/transcripts/10-adapting-llms.md)

## What this lecture establishes

[Lecture 9](09-large-language-models.md) built GPT-3 by next word prediction. This lecture asks why
GPT-3, for all its fluency, was bad at doing what it was asked, and how OpenAI fixed that. It was "only
trained to predict the next word", so it treated a request as the start of a list to continue (slides
7–13; ≈3:55–5:28). The fix is **instruction tuning**, which takes three steps. In **supervised
fine-tuning**, people write good answers to instructions and GPT-3 is trained further on them by next word
prediction. Then people **rank** several answers GPT-3 wrote, and those rankings train a **Reward Model**
that scores any answer. Finally the Reward Model's scores "nudge" GPT-3 by **reinforcement learning from
human feedback** (RLHF). This produced GPT-3.5/InstructGPT, and the same recipe applied to conversations
produced ChatGPT (slides 15–54; ≈5:28–29:24).

The second half generalizes the idea. Instruction tuning **adapted** a base LLM to follow instructions,
and adapting one to a business use case works the same way. Adaptation is a ladder of four rungs (slide
58): **zero-shot prompting** (a clear instruction), **few-shot prompting** (a few examples in the
prompt), **retrieval-augmented generation** (the question plus relevant chunks of your own documents,
found by embedding similarity) and **fine-tuning** (training the LLM further). The first two need no
training at all, an ability called **in-context learning**. RAG is shown in a Colab that answers questions
about the 2022 Winter Olympics. Fine-tuning turns marketing copy into believable product reviews with
"a couple of hundred examples", but a 70-billion-parameter model needs hundreds of gigabytes of memory to
fine-tune. **LoRA** (low-rank adaptation) solves that: freeze the model and learn each attention matrix's
change as the product of two "skinny" matrices. Then Llama-2-70b fits on two GPUs (slides 94–111;
≈1:08:08–1:16:35).

The catalog titles the video "Generative AI – Adapting LLMs with PEFT". The deck's roadmap calls it
"LLMs (2/2)" (slide 1), and the term PEFT is neither said nor printed; LoRA is the parameter-efficient
method taught. The recording presents slides 2–112 and ends on slide 112, a LoRA Colab the lecturer
leaves for a video walkthrough ("time does not permit", ≈1:16:35). Slides 113–114 are not reached. The
Colabs are not published on OCW. The RAG notebook is described below from the narration only, naming only
the calls and settings the lecturer says.

**The published deck is a revised version of the one shown in class.** Several numbers the lecturer reads
off the screen differ from the PDF: the memory and GPU counts for Llama-2-70b (slides 97, 98, 102) and the
LoRA parameter count (slide 109). This page gives the printed numbers and notes what was said. The
companion deck 10.5 is an intermediate version of the fine-tuning section that matches the class on LoRA
(see [the companion deck](#the-companion-deck-105) below).

## Part 1 — The correction, and GPT-3's scale

### The dense layer has linear activations

The lecture opens with a correction to lecture 9 (slides 2–3; ≈0:00–1:32). In a causal language model,
"when the contextual embeddings come out, you don't actually have to use ReLU activations here. You can
literally just run it through just a single dense layer with linear activations and then pass it into a
softmax." Slide 3 strikes through the "(ReLU)" of lecture 9's "Dense (ReLU)" box. The vectors coming out
of that dense layer are "as long as your vocabulary", so that the softmax gives a probability for every
token, "one word or token out of that entire 50,000-long vocabulary" (≈0:47–1:32). The lecturer points
it out because the way masked language models like BERT work differs in this detail from causal
language models like GPT-3.

![Slide 3: lecture 9's next-word-prediction pipeline for "the cat sat on the". Each input word has a row of grey squares (positional input embeddings) feeding a dark red box "TCE*" (Transformer Causal Encoder stack); out come rows of blue squares (contextual embeddings), which feed a box "Dense" with "(ReLU)" struck through; out come rows of yellow squares, each into its own "SM" softmax box, which outputs the next word: cat, sat, on, the, mat. Two annotations: "Linear activations, not ReLU" pointing at the struck-through ReLU, and "The dimension of these vectors is the vocab size" pointing at the yellow rows.](../raw/images/10-adapting-llms/slide-3.jpg)

*Slide 3 — the struck-through "(ReLU)": the dense layer before each softmax is linear, and its output is as long as the vocabulary. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

### GPT, GPT-2 and GPT-3: the difference is scale

GPT stands for **Generative Pretrained Transformer**. GPT and GPT-2 were trained like GPT-3, predicting
the next word with the same kind of transformer stack (slide 4; ≈1:32–2:22). The only difference "was
literally one of scale, scale of network and scale of data". Slide 4's table gives the sizes:

| Model | Number of parameters |
| --- | --- |
| GPT | 117 million |
| GPT-2 | 1.5 billion |
| GPT-3 | 175 billion |

A bigger network needs more data to train well. And here "more became different", almost "some of phase
change … between 2 and 3" (≈2:22). Given the start of an essay styled as "The importance of being on
twitter" by the humorist Jerome K. Jerome, GPT-3 writes a continuation that sounds like him (slides 5–6;
≈3:09). The surprise was not the continuation but its staying power. GPT-2 and GPT "after the first one,
two, or three sentences … will become incoherent and meander", while GPT-3 "can keep faking it for a
longer time. … Researchers did not expect this" (≈3:09–3:55).

## Part 2 — Why GPT-3 does not follow instructions

GPT-3 "was not good at following user instructions" (slide 7). Asked "help me write a short note to
introduce myself to my neighbor", it answers with lines such as "what is a good introduction to a
resume." (slides 8–10; ≈3:55). The reason is its training data, which "are basically lots of lists of
things". Ask it to continue "the capital of France" and it gives a list of capitals: "It thinks that you
need to complete some sort of list. … It's an autocomplete" (≈4:42). It can also produce offensive or
factually incorrect output (slide 11).

Why? GPT-3 "wasn't *explicitly* trained to generate good answers to user instructions. It is simply
trying to guess the next word repeatedly" (slides 12–13; ≈5:28). The internet is not mostly
instructions, so nothing in its training told it there was an instruction to follow. Hence the idea:
"let's explicitly train it with instruction data" (slide 14).

## Part 3 — Instruction tuning, step 1: supervised fine-tuning

OpenAI's approach is in the paper "Training language models to follow instructions with human feedback"
(slide 15), which "is what actually put ChatGPT on the map. And it's very readable" (≈6:14). It took
GPT-3 to **GPT-3.5**, also called **InstructGPT**, and "a small tweak after that gets you ChatGPT"
(slide 16). A note on the name: the approach has two main steps, (1) supervised fine-tuning and (2)
reinforcement learning from human feedback. "'Instruction Tuning' usually refers to just (1) but we will
use it as a shorthand to refer to the full approach" (slide 16's footnote, citing
<http://arxiv.org/abs/2203.02155>). The lecturer flags this: "This is not a consistent piece of
terminology" (≈6:14).

Step 1: "Get humans to write (high-quality) answers to instructions. About 12,500 such instruction-answer
pairs were created" (slides 17–18; ≈7:03). For "Explain the moon landing to a 6 year old in a few
sentences", GPT-3's answer was another question, "Explain the theory of gravity to a 6 year old", because
it thought it was continuing a list. The human-written answer was "People went to the moon in a big
rocket, walked around and came back to Earth. They took pictures of what they saw and sent them back so
we could all see them" (slide 18; the footnote calls it a lightly edited version of OpenAI's example).

GPT-3 is then trained some more on these pairs "using next word prediction as before, no difference"
(slide 19; ≈7:48). The input is the instruction followed by the answer. The labels are the answer
"shifted up", so when the model reaches "sentences" it must predict "People", given "People" it must
predict "went", and so on. That is exactly the shift of "the cat sat on the" against "cat sat on the
mat" in lecture 9 (slide 20). Nothing is predicted for the instruction words themselves.

![Slide 20: a tall black box "GPT-3". Down the left, under "Input", the words of the instruction "Explain the moon landing to a 6 year old in a few sentences." one per line, followed by the answer "People went to the moon in a big rocket …". Down the right, under "Labels", only the answer's words. Arrows run from "sentences.", "People" and "went" through GPT-3 to the labels "People", "went" and "to", then dots, then one arrow from "rocket" to the final "…".](../raw/images/10-adapting-llms/slide-20.png)

*Slide 20 — supervised fine-tuning is next word prediction on instruction + answer, with only the answer's words as labels. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

This step is called **supervised fine-tuning (SFT)**, and it helped: "GPT-3 did much better on
instructions" (slide 21; ≈8:35). But "writing high-quality answers to thousands of instructions is
difficult and expensive" (slide 22).

## Part 4 — Step 2: ranking answers, and the Reward Model

"What is easier than writing a good answer to a question?" the lecturer asks (slide 23; ≈8:35). One
student suggests writing a question for a given answer, a technique "actually used very heavily in
LLMs", though not the one discussed here. Another says "Thumbs up, thumbs down", and that is the answer:
"it's much easier to be a critic than to be a creator" (slide 24; ≈9:21). So who writes the answers to be
ranked? "We will just ask GPT-3 to write the answers. It might be crap, but we don't care because we can
rank them" (≈9:21–10:08).

GPT-3 can give several answers to one question because it **samples** the next word (slides 25–26;
[decoding and sampling](decoding-and-sampling.md)). "The fact that we had these stochastic outputs because
of sampling is now a feature, not a bug": run it three times "with a nice temperature of 1 or 1.1 or
something so that it's nice and random", and have humans rank the answers from most useful to least
useful (slide 27; ≈10:08). OpenAI "collected 33,000 instructions, fed them to GPT-3 and generated several
answers to each instruction" and had humans rank them (slide 28; ≈10:53). Keeping to two answers for
simplicity, each data point is [Instruction, Preferred answer = A, Other answer = B] (slide 29). The
slide's footnote says the deck describes the slightly simpler version of
<http://arxiv.org/abs/2009.01325>.

A student asks whether ChatGPT's own thumbs-up and thumbs-down buttons are for this. "Exactly right. …
all the models have the thumbs up, thumbs down stuff going on somewhere. They are all collecting data for
this step" (slide 42; ≈11:40). "If you're not sure who the product is, you're the product." ChatGPT
trains on your conversations unless you opt out in its data controls, and when the lecturer last checked,
opting out also lost your chat history. The API is opted out by default, and he thinks MIT's Microsoft
Copilot licence is too (≈11:40–12:27).

### The Reward Model and its loss

From this data OpenAI builds a **Reward Model**: a language model whose single numerical output says
"how good an answer is this to that particular instruction" (slide 30; ≈12:27–13:13). Two exact copies of
it are run, one on the instruction and the preferred answer, giving a rating $r_P$, and one on the
instruction and the other answer, giving $r_O$. The loss "encourages the model to give higher numbers to
the better answer" (slide 31; ≈13:13–14:00):

$$\text{Loss} = -\log(\sigma(r_p - r_O))$$

where $\sigma$ is the sigmoid: "You basically take the difference of these two things. Do a sigmoid, take
the logarithm" (≈14:00). The lecturer invites students to "convince yourself" that rating the better answer
higher lowers the loss. *Working it through (this page's own check, not the lecture's):* if $r_p$ is well
above $r_O$, the difference is large and positive, its sigmoid is close to 1, and $-\log$ of a number
close to 1 is close to 0. If $r_p$ is below $r_O$, the sigmoid is close to 0 and $-\log$ of it is large.
So minimizing the loss pushes the preferred answer's rating above the other's. In the terms of
lecture 2 (again this page's observation, not the lecture's), the formula is minus the log of a
probability, like [binary cross-entropy](loss-functions.md) for a "yes" label, with $\sigma(r_p - r_O)$
as the probability that the preferred answer is the better one.

![Slide 31: two copies of a black "Reward Model" box. The upper one takes "Instruction" and "Preferred Answer" and outputs r_P; the lower one takes "Instruction" and "Other Answer" and outputs r_O. Both arrows converge on a box with the formula Loss = −log(σ(r_p − r_O)).](../raw/images/10-adapting-llms/slide-31.png)

*Slide 31 — the same Reward Model scores both answers, and the loss falls as the preferred answer's score rises above the other's. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

Why show it both answers rather than only good ones? "Much like if you want to build a dog-cat
classifier, you have to show pictures of both" (≈14:46). Asked whether the popular response might not be
the accurate one, the lecturer points to the labelers' guidelines. Companies such as OpenAI have "a huge
document, 100, 200 pages long" teaching the rankers how to handle corner cases, and it is on the web
(≈14:46–15:34).

The Reward Model is trained with SGD like any network: "Better answers have lower loss. Back propagation
through the network, keep updating the weights, and, boom, you're done" (slide 32; ≈15:34). Afterwards
it rates any instruction-answer pair. In slide 33's example, GPT-3's gravity question gets 1.5 and the
human-written moon-landing answer 3.2. "We have learned how humans rank responses". People can rank only
so many answers, but the model can rank tens of thousands quickly (slide 34; ≈16:23).

![Slide 33: two rows split by a dashed line, under a heading "Rating". Top row: the instruction "Explain the moon landing to a six year old" and the answer "Explain the theory of gravity to a 6 year old." feed a black box (unlabelled on the slide) that outputs 1.5. Bottom row: the same instruction and the human-written answer "People went to the moon in a big rocket, walked around and came back to Earth. They took pictures …" feed a "Reward Model" box that outputs 3.2.](../raw/images/10-adapting-llms/slide-33.png)

*Slide 33 — after training, the Reward Model rates any answer: GPT-3's off-topic reply 1.5, the human answer 3.2. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

## Part 5 — Step 3: reinforcement learning from human feedback

The Reward Model is then used to improve GPT-3 itself (slides 35–39; ≈17:10–17:56). Send an instruction
into GPT-3 and get its answer. Feed the instruction *and* the answer to the Reward Model, which outputs a
rating. Then use that rating "to nudge the internal weights of GPT-3 in the right direction". The nudge
uses **reinforcement learning**, "which, just in the interest of time, we can't get into in this
lecture". Slide 38's footnote names the algorithm, Proximal Policy Optimization. "OpenAI used ~31,000
questions for this step" (slide 39). The result is GPT-3.5/InstructGPT (slide 40).

![Slide 39: left to right, the instruction "Explain the moon landing to a six year old" goes into a black "GPT-3" box, whose answer "Explain the theory of gravity to a 6 year old." goes, with the instruction, into a "Reward Model" box that outputs the rating 1.5. A dash-dot arrow curves from the rating back to GPT-3. Red text under the instruction: "OpenAI used ~31,000 questions for this step".](../raw/images/10-adapting-llms/slide-39.png)

*Slide 39 — the RLHF loop: GPT-3 answers, the Reward Model rates the answer, and the rating is fed back to adjust GPT-3. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

Steps 2 and 3 together are called **Reinforcement Learning from Human Feedback**: "because we use
reinforcement learning. And since humans rank the answers which led to the building of the reward model,
we get human feedback" (slide 41; ≈17:56). Slide 40 lists the three steps on the arrow from GPT-3 to
GPT-3.5/InstructGPT.

![Slide 40: the lineage GPT → GPT-2 → GPT-3 → GPT-3.5/InstructGPT → ChatGPT, with the arrow from GPT-3 to GPT-3.5/InstructGPT in red. A bracket under it points to three bullets: fine-tune GPT-3 using human-written question-answer pairs; have GPT-3 provide multiple answers for each question and have humans rank them, and with this data build the Reward Model; using the rating from the Reward Model, fine-tune GPT-3 using reinforcement learning.](../raw/images/10-adapting-llms/slide-40.png)

*Slide 40 — the three steps that turned GPT-3 into GPT-3.5/InstructGPT. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

### Questions on the data and its biases

The questions came from OpenAI's API: GPT-3 was already commercially available, so OpenAI collected the
questions people were sending, filtered them for quality and added some human-written ones, which are
expensive (≈18:42–19:27). On labeler bias, the lecturer agrees that "the reward model has probably very
faithfully learned all the biases of the human labelers". Giving the same question to many labelers and
keeping only rankings they agree on reduces disagreement between labelers, but "if everybody is biased in
the same direction, it won't protect you against that" (≈19:27–20:13). On Gemini's image-generation
trouble, it is unclear where in the pipeline the cause lies. Some systems intercept and reroute questions,
and the ranking step can teach a model to apply a rule "where it should not be applied" (≈20:58–21:45).

Why does a long answer not wander off topic, if each word is sampled? "Each word it utters now becomes
part of the input to the next word it utters", so "it's not truly a random walk". The rankers' rubric,
which covers helpfulness, accuracy, toxicity and perhaps coherence, may also play a role (≈21:45–23:17).
Asked whether attention could cover only the last few words, the lecturer says yes: that is
**sliding-window attention**, used mostly because it is compute-efficient; "default is full attention"
(≈23:17–24:02).

## Part 6 — From InstructGPT to ChatGPT

On the neighbor-note prompt, GPT-3.5/InstructGPT writes a proper short note ("Hello Neighbor, My name is
[Name] and I recently moved into the neighborhood…"), "like night and day difference" (slides 43–45;
≈24:02). To get from GPT-3.5 to **ChatGPT**, OpenAI "followed the exact same playbook". The difference is
that "instead of just instruction-answer pairs, use conversations as the training data", with the same
supervised fine-tuning, Reward Model and reinforcement learning (slides 46–47; ≈24:48). "That is the only
difference to go from 3.5 to ChatGPT." Because it was trained on conversations, ChatGPT answers the
neighbor prompt well and also handles a follow-up, "can you make it more formal?" (slides 48–50). Slide
51 continues the lineage to GPT-4, ChatGPT Plugins and GPT-4 V.

A student asks how a whole conversation is ranked: a person reading two transcripts that start with the
same question "will be able to assess which one is a better transcript" (≈26:19–27:05). Another asks
whether ChatGPT's robotic tone comes from reinforcement learning. The lecturer does not know, since the
ranking rubric favours human-sounding answers (≈27:05–27:51). None of this tuning happens in real time
during a conversation. Thumbs-up and thumbs-down signals go into the training logs, and the model is
retrained periodically (≈27:51–28:39). A model adapted for a new use case, such as medicine, uses the
techniques of the second half of the lecture (≈25:34–26:19).

## Part 7 — Instruction tuning in a nutshell, and helper LLMs

Slide 52 summarizes the recipe in five steps (≈28:39):

1. Start with a base LLM (like GPT-3) that has been trained to predict the next word.
2. Get humans to write responses to thousands of instructions. Fine-tune the base LLM with this data.
3. Generate multiple responses to each instruction using the fine-tuned LLM. Ask humans to rank the
   responses. Do this for several thousand instructions.
4. Use this data to build a Reward Model.
5. Using the Reward Model, train the fine-tuned LLM further using Reinforcement Learning with several
   thousand questions.

![Slide 52: five dark red chevrons labelled Step 1 to Step 5, each beside a box with one step of instruction tuning: start with a base LLM trained to predict the next word; have humans write responses and fine-tune; generate several responses, have humans rank them; build a Reward Model; train further with reinforcement learning using the Reward Model.](../raw/images/10-adapting-llms/slide-52.png)

*Slide 52 — instruction tuning in a nutshell. Steps 2 and 3 are the ones that need people. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

Steps 2 and 3 are "still human labor intensive" (slide 53). Researchers have since automated them with
**helper LLMs**: "an instruction-following LLM" writes the responses and "a helper LLM" ranks them (slide
54). "This is not what OpenAI did in the beginning with ChatGPT", but there are now enough good LLMs to do
it this way (≈28:39). Slide 55 points to the lecturer's own write-up, "How to use LLMs as 'helpers' to
build and customize other LLMs".

## Part 8 — Adapting a base LLM: the ladder

Instruction tuning was an example of something more general (slide 56; ≈28:39–29:24). To make GPT-3
useful at following instructions "we had to adapt it with high-quality data". To make a base LLM useful
for "a medical use case, a legal use case, or some other narrow business use case, you have to adapt it
with business domain-specific data". "**Adaptation** is the process of taking a base (aka foundation …)
LLM and tailoring it for a specific task" (slide 57). There is a ladder of ways to do it, each heavier
than the last (slide 58; ≈29:24–30:13):

| Rung | What changes |
| --- | --- |
| Zero-shot prompting | Prompt = a clear instruction |
| Few-shot prompting | Prompt = a few examples |
| Retrieval-augmented generation | Prompt = question + relevant chunks of text |
| Fine-tuning | The LLM is further *trained* with examples |

![Slide 58: "The Ladder of LLM Adaptation", a broad curved arrow rising to the right with four dark red dots of increasing size: Zero-Shot Prompting (Prompt = A clear instruction), Few-Shot Prompting (Prompt = A few examples), Retrieval-Augmented Generation (Prompt = Question + relevant chunks of text) and Fine-Tuning (The LLM is further trained with examples).](../raw/images/10-adapting-llms/slide-58.png)

*Slide 58 — the ladder of adaptation: the first three rungs change only the prompt; fine-tuning changes the weights. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

The "shot" in zero-shot and few-shot "is a synonym for 'example'" (≈30:13). The topic page
[adapting LLMs](adapting-llms.md) gathers all four rungs.

## Part 9 — Zero-shot prompting and prompt engineering

**Zero-shot prompting** asks for what you want without a single example (≈30:13). The example is a
product-review defect detector. The task is not sentiment ("that's kind of boring") but whether a review
describes a product defect. A one-star Wayfair review says "The curve of the back of the chair does not
leave enough room to sit comfortably" (slide 59). Once you would have collected reviews and built a
special-purpose NLP classifier. Instead, the prompt "Tell me if a product defect in being described in
this product review:" followed by the review gets the answer "Yes, this is a product defect." (slides
60–61; ≈31:00). Zero-shot "actually works remarkably well", and bigger models do it better.

Designing the prompt so that the LLM responds as wanted is **prompt engineering** (slide 62; ≈31:47).
Asked "what is the fifth word of this sentence?", ChatGPT often gets it wrong ("is", or "the"). Ask it
first to list all the words in the sentence and then give the fifth, and it answers "word" correctly
(slides 63–64): "be very, very prescriptive as to what you want it to do and break down all the steps.
Don't make it guess things" (≈32:33).

Researchers keep "discovering" simple, widely applicable strategies (slide 65; ≈32:33). Adding "Let's
think step by step" gets the paper's juggler question (16 balls, half golf balls, half of those blue)
right, 4, where the plain prompt answers 8. Such instructions are now built in. Your prompt is appended to
a hidden **system prompt** that tells ChatGPT to think step by step and take its time, and the system
prompts "have been jailbroken" and can be found on the web (≈32:33–33:19). A newer finding: "Take a
deep breath and work on this problem step-by-step" scored 80.2 on the GSM8K test with PaLM 2-L, against
71.8 for "Let's think step by step" (slide 66's table). The lecturer adds, to laughter, that telling a
model it is an officer on the Starship Enterprise reportedly helps on reasoning questions. Students add
promising a tip, and claiming another model solved it. Slide 67 points to a recent prompt-engineering
survey (<https://arxiv.org/abs/2401.14423>).

## Part 10 — Few-shot prompting and in-context learning

**Few-shot prompting** puts a few examples of the task into the prompt (slides 68–73; ≈34:06–34:52). To
build a grammar corrector, the prompt from the GPT-3 paper gives three pairs of "Poor English input" and
"Good English output", the first being "I eated the purple berries." → "I ate the purple berries.", and
ends with a poor-English sentence alone. GPT-3 completes it, changing "work with you in another project"
to "on another project": "it just learns on the fly, what you have in mind" (slide 73).

The ability of LLMs "to learn how to do a task with just instructions/examples in the prompt is called
**In-Context Learning**" (slide 74). It covers both zero-shot and few-shot prompting. GPT and GPT-2 could
not do it: "That was new in GPT-3 and what they call an emergent capability. It was completely
unanticipated by the people who built it" (≈34:52).

## Part 11 — Retrieval-augmented generation

**Retrieval-augmented generation (RAG)**, "also called indexing sometimes" (slide 75), lets a chatbot use
proprietary data (≈35:39). Take a customer-support operation with "this massive FAQ database": "Can't we
just include the entire dataset in the prompt?" (slide 76). Nothing stops you in theory. What stops you in
practice is the **context window**: "for any LLM, the prompt plus the output, the length cannot exceed a
predefined limit" (slide 77; ≈36:26). It is the counterpart of the maximum sequence length of the course's
earlier models, and it counts input and output together.

![Slide 77: a "Prompt" box feeds a black "LLM" box, which outputs a taller stack: the prompt (in faded grey) with the "Output" below it. A brace spans the whole stack, labelled "This is called the 'context window'". Title: for any LLM the prompt plus output cannot exceed a predefined limit, the Context Window, which varies from LLM to LLM and limits how much information you can pack into the prompt.](../raw/images/10-adapting-llms/slide-77.png)

*Slide 77 — the context window covers the prompt and the output together. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

Worse, in a conversation "the entire conversation is fed in every single time. That's how it actually
remembers what's going on earlier in the conversation. It doesn't have any memory per se" (slide 78;
≈37:13). Each new question carries all the earlier turns, so with a long FAQ and a few follow-ups, "before you
know it, boom, you've blown out the context window" (≈38:01). For text longer than the window there is "a whole research cottage industry".
The simplest scheme is a moving window over the last tokens, and a cleverer one has another LLM summarize
what no longer fits (≈38:01).

So instead "we first retrieve the relevant content from the database or the FAQ and then send it to the
LLM along with the question" (slide 79; ≈38:46). Pictorially (slide 80, a figure by Sebastian Raschka):

1. Split the documents into chunks, for example each question and answer of the FAQ.
2. Compute a **contextual embedding** of each chunk: "Run it through something like BERT, you're done"
   ([lecture 8](08-transformers-and-self-supervised-learning.md)).
3. When a question comes in, compute its embedding too.
4. Find the chunks most similar to the question, and pack those into the prompt with the question.
   "Maybe you have 10,000 questions, but you can only accommodate 5 of them in your prompt", so pick the
   5 most relevant (≈39:32–40:17).

Students' questions fill in the details. RAG is for data the model would not know because "it is not in
the public internet", not for the Steve Jobs style it already learned in pretraining (≈40:17–41:04).
Chunking loses some information, and how to chunk (by section, by paragraph) is "a whole empirical
cottage industry" (≈41:04). Picking 5 chunks of 10,000 is itself a weighting, "a weight of 0 and these a
weight of 1" (≈41:50). The documents need only be "relatively clean" (≈41:50). In the lecturer's view
RAG "is … the most prevalent business application of LLMs that I've seen up to date" (≈42:36). See
[retrieval-augmented generation](retrieval-augmented-generation.md) for the topic.

## Part 12 — The RAG Colab: the 2022 Winter Olympics

The notebook (slide 81, "HODL-SP24-Section-A-Lec-10-Retrieval-Augmented-Generation", whose link target
is not printed) is not on OCW. This section follows the narration (≈41:50–1:02:40).

**Set-up.** It installs the OpenAI library and **tiktoken** and imports pandas. It needs an OpenAI API
key: "You have to get your own key to make it all work" (≈42:36–43:23). The chat model is
**GPT-3.5-turbo**. Embeddings come from an OpenAI embedding model, captioned as "embedding-ada-002", so
"you don't have to use your own BERT" (≈43:23–44:59).

**Without retrieval.** Asked about the 2020 Summer Olympics, the model correctly says "Barshim of Qatar
and Tamberi of Italy both won the gold" (≈44:59). Asked "Which athletes won the gold in curling in the
'22 Olympics?", it says the Swedish men's team and the South Korean women's team. Sweden is right, but
Great Britain won the women's gold (≈45:44). The model's training cutoff was September 2021, "so as far as
it's concerned, the '22 Olympics haven't happened yet. Yet, it confidently gave you the wrong answer".
That is **hallucination**: "a very eloquent, confident, wrong answer" (≈46:29). Adding "Answer the
question as truthfully as possible. And if you're unsure of the answer, say sorry, I don't know" makes it
answer "Sorry, I don't know." (≈47:14).

**Context by hand.** The extra information in a prompt "is called **context**" (≈48:00). Pasting the
Wikipedia article on curling at the 2022 Winter Olympics into the prompt, messy formatting and all, gets
the right answer, Great Britain. It also handles a harder question: "Yes, Oskar Eriksson won multiple
medals in curling. He won a gold in the men's event and a bronze in the mixed doubles" (≈48:46–49:32).

**Retrieval, automatically.** The lecturer gives the context window as 16,385 tokens "for GPT-4 and
3.5-turbo", GPT-4's as "up to 128,000 tokens", and Google Gemini 1.5 Pro's as 1 million (≈49:32–50:20).
Relatedness between a question's embedding and a chunk's is measured by **cosine similarity**, explained
in [lecture 7](07-transformers.md) (≈51:05). OpenAI provides the Wikipedia articles about the Olympics
already chunked and embedded, which saves a long computation (≈51:51–53:25). An embedding is a vector of
length 1536. "HODL is amazing!" and "HODL is incredible" have a cosine similarity of 0.9934 against a
maximum of 1, "comforting because amazing and incredible are obviously synonyms" (≈52:38). A helper,
`strings_ranked_by_relatedness`, returns the five chunks most related to a question. For the curling
question their similarities run from 0.88 down to 0.867, all about curling and medals (≈54:14–54:59).

The prompt is then assembled: a header ("Use the below articles to answer the subsequent question. …
if you're unsure of the answer, say I'm sorry, I don't know"), then Wikipedia chunks in order of
relatedness until a token budget is reached, then the question. tiktoken counts the tokens. The budget is
3,700, partly to show how, "and also because it's charging my credit card for every token" (≈55:47–56:32).

**Results.** This time the answer left out the Great Britain team ("Last night it didn't. Welcome to
stochasticity", ≈58:04). Asked about the 2016 Winter Olympics, which did not happen, it says "Sorry, I
don't know." With the "be truthful" instruction removed, it answers about the 2022 Olympics instead,
"an irrelevant question accurately" (≈58:04–58:51). The morals: RAG is heavily used in industry, token
budgets must be watched, and "small wording changes in the prompt can actually dramatically alter
behavior, which makes it very difficult in enterprise settings to do QA on this stuff". The lecturer
cites Air Canada's chatbot, which gave a customer wrong advice; the court ruled for the passenger, and the
chatbot was pulled. "Without a human in the loop checking these answers, it's kind of dangerous"
(≈58:51–59:36).

Two closing questions. Uploading a PDF to GPT-4 does not chunk it: GPT-4-Turbo's window of 128,000 tokens
holds it whole. The chunking in RAG "you have to do. The LLM doesn't even know you're doing it". When
GPT-4 says it is "doing a Bing search", a program is retrieving, chunking and embedding results and
packing them into the prompt (≈1:00:23–1:01:08). LLMs cope with messy formatting because well-formatted
answers to badly formatted inputs were ranked higher during fine-tuning. Asking for output in a format
such as markdown or JSON can improve the answer itself (≈1:01:08–1:01:54).

## Part 13 — Fine-tuning

Everything so far left the LLM's weights alone: "You're just using it as a black box. With fine-tuning,
you actually will train it further" (slide 82; ≈1:02:40). Take a causal LLM such as GPT and train it
further on **domain-specific input-output examples**, the input followed by the shifted output, which
"is essentially the Supervised Fine-tuning step we discussed earlier" (slides 83–84; ≈1:02:40–1:03:26).
"This is basically fine-tuning exactly like we saw with BERT … And even with ResNet" (see
[transfer learning](transfer-learning.md)). The lecturer notes that his diagram "still" showed a ReLU:
"There is no ReLU here". The published slide 83 prints a plain Dense box.

![Slide 84: a two-row table (Input: "Explain the moon landing to a 6 year old in a few sentences."; Desired output: "People went to the moon in a big rocket…") with a thick arrow to a diagram: a tall black "Causal LLM" box, the instruction and answer words listed one per line under "Input" on the left, and the answer words, highlighted yellow, under "Labels" on the right, each aligned with the input word before it.](../raw/images/10-adapting-llms/slide-84.png)

*Slide 84 — fine-tuning on a domain example is the same next-word training as supervised fine-tuning. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

Every weight is updated (slide 85): (1) positional embeddings; (2) stand-alone embeddings, unless
pretrained and frozen ("Trainable=False"); (3) the matrices $A^K$, $A^Q$, $A^V$ of each attention head,
inside the Transformer Causal Encoder (TCE); (4) layer norm scale and bias parameters; (5) the
feed-forward layers' weights; and (6) the weights in the final dense and softmax layers. Items 3–5 are the
pieces of the encoder built in [lecture 8](08-transformers-and-self-supervised-learning.md).

![Slide 85: the causal LLM pipeline of four rows (input, positional input embeddings, TCE, contextual embeddings, Dense, yellow rows, four SM boxes), with red dashed outlines around the positional embeddings together with the TCE, around the Dense box and around the SM boxes. Below, a red numbered list of six kinds of weights: positional embeddings; stand-alone embeddings (unless pretrained and Trainable=False); matrices A^K, A^Q, A^V for each attention head (inside TCE); layer norm scale and bias parameters (inside TCE); weights in feed-forward layers (inside TCE); weights in final Dense/Softmax layers.](../raw/images/10-adapting-llms/slide-85.jpg)

*Slide 85 — what full fine-tuning updates: everything outlined in red, listed below the diagram. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

### Example: synthetic product reviews

The motivating example is generating **synthetic product reviews** from product descriptions, for a
product that simulates customer behaviour in e-commerce, since writing reviews by hand is slow (slides
86–87; ≈1:03:26). Asked to "Write a positive product review using the following information" with a rag &
bone jeans page, an LLM answers "Timeless. Authentic. Iconic. …". "This reads like marketing copy because
there's a whole bunch of marketing copy on the internet" (slides 87–88; ≈1:04:13). For an Urban
Outfitters tank top it writes "The boxy and cropped silhouette is flattering on all body types". "Come
on!" (slide 89). The slides do not name the model; deck 10.5's version of this slide says "This is
Alpaca, running off of Llama 7B" (deck 10.5, slide 8).

So fine-tune the LLM with [instruction, product description, product review] examples (slide 90). Slide
91 shows three records of such a dataset, each with an "instruction" ("Write a positive review for the
following product", or neutral, or negative), an "input" (a product description) and an "output" (a real
review, such as "The Best! Carharrt t-shirts are my husband's favorite. They fit well…"). "You just have
to get a few hundred of these product review examples, just a few hundred" (≈1:05:01). After fine-tuning,
the same jeans get "These are the best jeans I've ever owned. I'm 5'10" and 160 lbs. …", and "Write a
horrible review:" gets "These are the worst jeans I've ever owned. … I'm going to return them and try a
30, but I'm not optimistic. … I'm going to stick with Levi" (slides 92–93; ≈1:05:47). "These read like real
reviews!"

For the lecturer "the biggest benefit" is the numbers. Pretraining took billions of sentences, and SFT and
RLHF tens of thousands of examples, but "to make it work for your narrow business use case, you only had
to spend a couple of hundred examples. … Nobody's going to [collect 30,000]. … But a couple of hundred,
anybody can do" (≈1:06:34). Is fine-tuning always supervised? To make the model "generally smart about
the company's business details", plain next word prediction on company text will do. To make it follow
instructions on your problem, "you need supervision" (≈1:07:23–1:08:08).

## Part 14 — Why fine-tuning a large LLM is hard

"For small causal LLMs (like GPT-2), fine-tuning isn't difficult but for larger LLMs, we will face
computational challenges" (slide 94; ≈1:08:08). A small model such as Google's Gemma, "like 2 billion
parameters or something", fits on one GPU (you still need a GPU). The **Llama 2** family of open LLMs,
Llama-2-7B, 13B and 70B, is widely used for fine-tuning because you can download the model and do what
you want with it. "It's not strictly open because there are some footnote considerations … But for most
purposes, it's open enough" (slide 95; ≈1:08:54).

How hard is the biggest, **Llama-2-70b**? Slides 97–98 work it out:

- **Memory.** 70 billion parameters, times 2 bytes per parameter (16-bit "fp16" numbers), times a multiplier of
  ~3–6, "since we need to store gradient, optimizer states etc. in addition to the weights themselves",
  makes 420–840GB. An A100 or H100 GPU has 80GB of RAM, so this needs between 6 and 11 GPUs (slide 97).
  In class the lecturer read a multiplier of 3 to 4, said it "should really be 1 to 6", and gave 420 to
  560 gigabytes and 6 to 7 GPUs (≈1:09:39–1:10:26). The published slide carries the corrected numbers.
- **Compute.** Llama-2-70b was trained on 2 trillion ($2 \times 10^{12}$) tokens. An A100 processes,
  optimistically, about 400 tokens per GPU per second (a forward pass). So 11 GPUs would take
  $2 \times 10^{12}/(11 \times 400)$ seconds, about 5,261 days. To finish in about a month takes 2,048
  GPUs (about 28 days), at an estimated \$4M at \$2.5 per GPU-hour (slide 98). In class: 7 GPUs and
  8,000 days (≈1:10:26). The real cost would be "a lot higher", since it takes multiple runs to get right
  (≈1:11:12).

To fine-tune with fewer resources, do two things (slides 99–101; ≈1:11:12). First, **reduce the size of
the dataset**, and "we are in luck here". The famous **Alpaca** fine-tuning dataset has 50k
instruction-answer pairs at 4,096 tokens each, about 200M tokens, far fewer than 2T. At 400 tokens per
GPU per second, 7 GPUs take $2 \times 10^8/(7 \times 400)$ seconds, "only about 20 hours" (slide 100,
which prints the power as "108"). Second, **reduce the memory required**, ideally to one GPU, "literally
on Colab".

What consumes memory? Slides 102–103 tabulate it for Llama-2-70b:

| | Naive memory usage | Optimized |
| --- | --- | --- |
| Model parameters | number of parameters times 2 bytes = 140GB | 140GB (can be reduced by quantization, often at a cost in performance) |
| Gradient computations | Same as above = 140GB | ~ zero, by computing them "just in time" (gradient checkpointing) |
| Optimizer state | 1–4 times the memory needed for parameters = 140–560GB | to be reduced to ~ zero |
| Total | 420–840GB | 140GB |

State-of-the-art optimizers such as [Adam](gradient-descent.md) "need to store information related to
past gradients", which takes about 1–4 times the memory of the parameters (slide 102). **Gradient
checkpointing** trades time for memory: "I don't mind it running longer, but I don't want to use as much
memory" (≈1:13:29). A student asks about Microsoft's 1-bit model. Such approximations are part of a field
called **LLM quantization**, and each "will cost you something in terms of accuracy", so "there's always a
trade-off" (≈1:11:58–1:12:43). In class the lecturer read the optimizer state as "2x" and the total as 560
gigabytes (≈1:12:43–1:13:29).

## Part 15 — LoRA: low-rank adaptation

The remaining term, the optimizer state, is removed by training far fewer parameters, "a breakthrough
from maybe a year ago" (≈1:13:29). "We will fine-tune only the matrices inside the causal self-attention
blocks, and will keep everything else frozen": item 3 of slide 85's list (slide 104).

Take the key matrix $A^K$ (slide 105). Llama-2-70b has 64 self-attention heads in every self-attention
layer. Each head's $A^K$ is $8192 \times 128$, and combining all 64 gives one $8192 \times 8192$ matrix,
so each layer stores about 64M parameters for $A^K$ alone. Fine-tuning changes it. The result can be
thought of as the original plus the change, $A^K + \Delta A^K$, and in general $\Delta A^K$ "will be as
big as $A^K$" (slide 106; ≈1:14:15–1:15:02).

"Can we make it smaller?" A fine-tune "will likely make very small changes to the original model weights.
This suggests that we can 'force' the change matrix $\Delta A^K$ to be 'simple' and it will still get the
job done" (slide 107). Simple means **low rank**: approximate the $8192 \times 8192$ matrix by an
$8192 \times 2$ matrix times a $2 \times 8192$ matrix, "two thin, skinny matrices, which, if you multiply
them, gets you the original matrix" (slide 108; ≈1:15:02–1:15:49). "I'm not going to get into the
mathematical details here. This is called a low-rank approximation."

![Slide 109: a large dark red square labelled 8192 by 8192, "=~", a tall thin bar (labelled 8096 at its side and 2 above it) "x" a short wide bar (8192 above, 2 at its end). Bullets: Recall that A^K has 64M parameters (=8192 x 8192); but simple ΔA^K has just 4 x 8192 = 32,768 parameters, i.e., just 0.05% of the original 64M; this idea is called Low Rank Adaptation (LORA). Footer link: "LoRA: Low-Rank Adaptation of Large Language Models".](../raw/images/10-adapting-llms/slide-109.png)

*Slide 109 — LoRA learns the change to an 8192-by-8192 matrix as two skinny matrices of rank 2: 32,768 numbers instead of 64 million. [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

Slide 109 counts the parameters: "simple $\Delta A^K$ has just 4 x 8192 = 32,768 parameters i.e., just
0.05% of the original 64M!" In class the lecturer said "each one is just 8,000 times 2, 16,000", for
"16,192 parameters, which is 0.02% of the original 64 million" (≈1:15:49). Those are the numbers of deck
10.5's slide 25, "2 x 8096 = 16,192", and slide 109 itself still labels the tall matrix "8096".
*This page's own check:* two matrices of $8192 \times 2$ numbers each make
$2 \times 2 \times 8192 = 32{,}768$, and $32{,}768 / (8192 \times 8192) \approx 0.05$ percent, as slide
109 prints. The spoken 16,192 is $2 \times 8096$, the size of one skinny matrix (with 8096 for 8192). This is **Low-Rank Adaptation (LoRA)**, "incredibly widely used in the industry".

LoRA fine-tuning (slide 110; ≈1:15:49–1:16:35):

- Freeze all the base model's parameters.
- Initialize $\Delta A^K$, $\Delta A^Q$ and $\Delta A^V$ to zero in each self-attention layer.
- Update them, by SGD as usual, by updating only the two skinny matrices of each.

With LoRA only the skinny matrices are optimized, so the optimizer state is about zero, and the total
falls from 420–840GB to 140GB (slide 111). "Now, Llama-2-70b can be comfortably be finetuned on 2 GPUs
and Llama-2-7b and Llama-2-13b can comfortably be finetuned on a single GPU", or in the lecturer's words
"on a single Colab GPU" (≈1:16:35).

## Part 16 — Not reached in class

Slide 112, "LoRA Finetuning Colab", says only "Stay tuned for video and colab". "9:54, time does not
permit. … I will do a video walkthrough tomorrow or day after" (≈1:16:35). The walkthrough is not part of
this KB. Two more slides were not shown:

- Slide 113, "The effort-benefit curve for adaptation strategies depends on the size of the base LLM", is
  Andrej Karpathy's chart, excluded from OCW's licence and so not shown here. It plots task accuracy
  against effort across zero-shot prompts, few-shot prompts, retrieval-augmented few-shot prompting and
  fine-tuning. A big base model rises steeply with prompting alone (about half way up with zero-shot
  prompts) and approaches 100% with fine-tuning; a small one is still only about 10–15% after
  retrieval and reaches about a third with fine-tuning. The slide file describes it in full.
- Slide 114, "We have covered all these topics!", maps the course onto the "How LLMs Are Built?" flow chart
  of the survey "Large Language Models: A Survey" (February 20, 2024), with red outlines on what the course
  covered: byte pair encoding, absolute positional embeddings, encoder-only and decoder-only
  architectures, masked and causal language modeling, supervised fine-tuning, RLHF, the four decoding
  strategies, and low-rank adaptation.

![Slide 114: "We have covered all these topics!" beside the "How LLMs Are Built?" flow chart from "Large Language Models: A Survey" (Minaee et al., CC BY): a column of nine steps (data cleaning, tokenizations, positional encoding, LLM architectures, model pre-training, fine-tuning and instruction tuning, alignment, decoding strategies, cost-effective training/inference, adaptation and compression), each with its sub-topics. Red outlines mark byte pair encoding, absolute positional embeddings, encoder-only and decoder-only, masked and causal language modeling, supervised fine-tuning, RLHF, greedy and beam search with top-k and top-p sampling, and low-rank adaption.](../raw/images/10-adapting-llms/slide-114.jpg)

*Slide 114 — the survey's map of how LLMs are built, with the parts this course covered outlined in red (figure by Minaee et al., CC BY). [Deck](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec10.pdf)*

## The companion deck 10.5

OCW also publishes "Lecture 10.5: Finetuning"
([`mit15_773_s24_lec105.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec105.pdf),
transcribed in [`raw/slides/10-finetuning.md`](../raw/slides/10-finetuning.md)). No video presents it.
It is a variant of this lecture's fine-tuning section (slides 82–114), and most of its 33 pages repeat
those slides. Its differences are useful. Its LoRA slides use the numbers the lecturer says in class
(8096, "2 x 8096 = 16,192", 0.02%; its slides 21, 24–25). Its pipeline still has the "Dense (ReLU)" box
(its slide 3). It names the model behind the marketing-copy reviews, "Alpaca, running off of Llama 7B"
(its slide 8). Its list of updated weights has seven items, splitting the final dense and softmax layers
(its slide 5). Cite it as "deck 10.5, slide N".

## Printed slips

- Slides 100–101 print "2 x 108/(7x400)" for $2 \times 10^8/(7 \times 400)$; slide 98's $10^{12}$ is a
  proper superscript.
- Slide 109 labels the tall skinny matrix "8096" where the rest of the slide says 8192, a remnant of the
  older version in deck 10.5. It also prints "LORA" where its link and slide 110 say "LoRA".
- Slide 110 prints the third matrix as $\Delta A^v$ (lowercase v), and its last bullet stops at "for
  each", above two bars drawn as the skinny matrices.
- Slide 111 prints "can be comfortably be finetuned", and its asterisks have no footnotes on that page.
- Slide 57 prints "a base (aka foundation or base) LLM"; slide 60's prompt reads "a product defect in
  being described"; slides 28–29 spell "Open AI".
- Spoken against the published deck: the memory and GPU figures for Llama-2-70b (Part 14) and the LoRA
  count (Part 15), read from an earlier version of the slides; "what about ChatGPT and GPT-2?" (≈1:32),
  where slide 4 asks about GPT and GPT-2; "GPT-3-turbo" (≈45:44) for GPT-3.5-turbo; "0.934" (≈52:38) for
  the 0.9934 just read out; "1,600 something tokens" (≈56:32) where 16,385 was given at ≈49:32.

## Related pages

- [Instruction tuning](instruction-tuning.md): supervised fine-tuning, the Reward Model and RLHF, from
  GPT-3 to InstructGPT and ChatGPT, gathered as a topic.
- [Adapting LLMs](adapting-llms.md): the ladder of adaptation: zero-shot and few-shot prompting, prompt
  engineering, in-context learning, fine-tuning and LoRA.
- [Retrieval-augmented generation](retrieval-augmented-generation.md): the context window, chunks,
  embeddings and cosine similarity, and the Olympics Colab.
- [Large language models](large-language-models.md): next word prediction, GPT-3 and its scale, and what
  instruction tuning added.
- [Decoding and sampling](decoding-and-sampling.md): the sampling that lets GPT-3 write several answers
  to rank.
- [Transfer learning](transfer-learning.md): fine-tuning a pretrained network, from ResNet and BERT to
  LLMs.
- [Loss functions](loss-functions.md): binary cross-entropy, of which the Reward Model's loss is a form.
- [Lecture 9](09-large-language-models.md): next word prediction, the Transformer Causal Encoder and
  GPT-3, which this lecture builds on.
- [Lecture 8](08-transformers-and-self-supervised-learning.md): the encoder's $A^K$, $A^Q$, $A^V$
  matrices, which LoRA adapts.
