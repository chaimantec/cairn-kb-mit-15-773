---
title: Lecture 1 — Introduction to Neural Networks and Deep Learning (slide deck)
lecture: 1
slides: 110
source_pdf: https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec01.pdf
note: Slide N = PDF page N (N = 1..110). Page 110 is OCW's appended end page, not lecture content. Some pages print stale static numbers that are ignored; they are listed in the "Slide numbers vs PDF pages" section.
---

# Lecture 1 — Introduction to Neural Networks and Deep Learning: slide-by-slide

Text and figures of all 110 slides of
[`mit15_773_s24_lec01.pdf`](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/mit15_773_s24_lec01.pdf)
(MIT 15.773 Hands-On Deep Learning, Spring 2024; lecturer Rama Ramakrishnan), transcribed from the deck. Cite these as "slide N" — N is the PDF page number; slide 110 is OCW's appended end page. Diagrams, plots and photographs are described in prose since the KB is read as text.

Many slides are **build steps** — the same slide re-shown with one more element revealed. They are transcribed individually so that a citation to any one of them resolves, with a note of what each adds relative to the previous one.

## Numbering: slide numbers vs PDF pages

Headings use the PDF page number. Pages 13–18 print 20–25, page 21 prints 29, page 31 prints 39, page 79 prints 87, pages 81–85 print 89–93 and page 88 prints 96 — stale static numbers left over from an earlier version of the deck, inconsistent with their neighbours, so ignored. They are visibly printed in the bottom-right corner (checked against crops of those pages), so a reader who opens PDF page 13 sees "20" there: cite and look up by PDF page. Pages 1, 11, 12, 19, 20, 22–24, 30, 33, 34, 36–38 and 110 print no number.

## Contents

| Slides | Section |
| ------ | ------- |
| 1 | Title |
| 2–4 | Prerequisites, grading, instructor introduction (Rama Ramakrishnan) |
| 5–9 | Why HODL: why the course was created; its "philosophy" (concepts, skip the math, coding, build a V1.0 DL model yourself) |
| 10–13 | Overview: AI, ML, DL and Generative AI; AI founded in 1956 (Dartmouth workshop photo); the AI → ML → DL → Generative AI progression |
| 14–15 | Traditional (rule-based) approach to AI and why it is difficult (Polanyi's Paradox) |
| 16–19 | Machine learning: learning from input-output examples; ML models list |
| 20–26 | Structured vs unstructured data; manual representation / feature extraction as the "human bottleneck" |
| 27–32 | Deep learning: automatic representation learning from unstructured data; three forces (algorithms, data, GPUs); neural networks |
| 33–38 | Immediate application of DL: giving sensors the ability to recognize things; examples; smart binoculars |
| 39–40 | Outputs: structured outputs (numbers, probabilities) |
| 41–53 | Unstructured outputs and Generative AI: image/text/music generation, text-to-text, multi-modal (ChatGPT parking-sign example), the landscape of Gen AI |
| 54–55 | Section dividers: "If you understand Deep Learning, everything becomes possible!"; "What's a Neural Network?" |
| 56–64 | Logistic regression recalled; the job-interview classification example (GPA, Experience); logistic regression drawn as a network; worked prediction; weights and bias terminology |
| 65–78 | Advantage of the network lens: learning smart representations; stacking linear functions and activation functions; two-stage network; "This is a Neural Network!" |
| 79–85 | Terminology: neurons, layers, activation functions, input/hidden/output layers, dense (fully connected); logistic regression as an NN with no hidden layers |
| 86–87 | Deep learning is just NNs with lots of hidden layers (ResNet / VGG architecture figure) |
| 88–93 | Activation functions: definition; sigmoid, linear, ReLU; visual shorthand |
| 94–96 | Recap: designing a DNN (user chooses layers, units, activations); DNN for the interview classifier |
| 97–100 | Practice: setting up a simple 2-3-1 NN; counting parameters (13); trained parameter values |
| 101–105 | Predicting with the NN: hidden-layer and output-layer computations; worked example (x1 = 2.3, x2 = 10.2 gives y = 0.226); network as a single function |
| 106–107 | Contrast with logistic regression; complexity of even a simple network |
| 108–109 | Summary: a deep (feedforward) neural network; designing a DNN |
| 110 | MIT OpenCourseWare end page |

---

## Slide 1 — Lecture 1: Introduction to Neural Networks and Deep Learning

Title (red, bold; "Lecture 1:" underlined): "Lecture 1: Introduction to Neural Networks and Deep Learning".

Bottom: MIT Management Sloan School logo (red/grey "MIT" above "MANAGEMENT" and "SLOAN SCHOOL"). Text beside it: "15.773: Hands-on Deep Learning", "Spring 2024", and in red "Farias, Ramakrishnan".

*Figure: no*

## Slide 2 — Prerequisites 👀

(The title is followed by an "eyes" emoji.)

- Familiarity with Python at this level (a red arrow points right from this bullet to the screenshot)
- Familiarity with fundamental machine learning concepts (such as training/validation/testing, overfitting/underfitting, and regularization).
- If you have taken 15.071/15.072(or will be taking it concurrently) OR if you have other relevant coursework or work experience, you should be fine.

Right: a browser screenshot of kaggle.com/learn/python (URL bar shows "kaggle.com/learn/python"). It shows the Kaggle "Python" course page: heading "Python", subtitle "Learn the most important language for data science.", a "Begin Course" button with a progress bar and "5 hours to go", tabs "Course" and "Discussion", and a "Lessons" list with columns "Tutorial" and "Exercise": 1 "Hello, Python" (A quick introduction to Python syntax, variable assignment, and numbers); 2 "Functions and Getting Help" (Calling functions and defining our own, and using Python's builtin documentation); 3 "Booleans and Conditionals" (cut off). It indicates the expected level of Python.

*Figure: yes — screenshot of Kaggle "Python" course page*

## Slide 3 — Grading

An embedded text block (from the syllabus):

**Grading**
Your course grade will be based on 2 homework assignments, a final project, and class participation:

| Component | Weight |
| --- | --- |
| Class participation | 10% |
| Homework assignments (25% x 2) | 50% |
| Final project | 40% |

*Figure: no*

## Slide 4 — Rama Ramakrishnan

Subtitle: "Professor of the Practice, AI/ML"

- **Education:** Ph.D. and M.S. (Operations Research; MIT), B.Tech. (Engineering; Indian Institute of Technology)
- **Industry Experience**
  - McKinsey followed by 4 data science/machine learning startups (in asset management, transportation, retail and ecommerce). Exits to Oracle, Demandware and Salesforce
  - Post acquisition, Chief Analytics Officer at Oracle Retail; SVP of Data Science at Salesforce
  - My most recent startup – CQuotient – is now Salesforce Einstein AI for Commerce and is one of the top personalization engines in the world (live on ~10,000 e-commerce sites worldwide)
- **Interests:** Applying AI/ML to business problems (especially shortest-path-to-human-impact applications e.g., healthcare, drug development)
- **Outside Activities:** Active angel investor, AI advisor to venture firms, several startups and a few large companies

*Figure: no*

## Slide 5 — Why HODL?

[Section divider.] Text: "Why HODL?" (HODL = Hands-On Deep Learning, the course.)

*Figure: no*

## Slide 6 — Why did we create HODL?

- DL is one of the most exciting and profound technology developments of our lifetimes
- It is important for Sloanies to understand how to use DL to transform businesses and create exciting new products/services
- While MIT has other (excellent) DL courses, we wanted one that was a better fit for Sloan

*Figure: no*

## Slide 7 — HODL's "philosophy"

- Focus on the key <u>concepts</u> that underlie DL
- Skip the math* (but we are happy to geek out in office hours and/or suggest readings for those who are interested)

Footnote: "*If you are looking for a 'mathy' DL course, this course won't be a good fit and you may want to consider dropping it."

*Figure: no*

## Slide 8 — HODL's "philosophy"

Previous two bullets shown greyed out, plus a new bullet:

- (greyed) Focus on the key concepts that underlie DL
- (greyed) Skip the math (but we are happy to geek out in office hours and/or suggest readings for those who are interested)
- Focus on <u>coding</u> DL models …
  - It is the only way to develop a visceral (not just intellectual) understanding of how this stuff works
  - Successful new products and services are often inspired by hands-on tinkering

*Build step: adds the "Focus on coding DL models …" bullet and two sub-bullets relative to slide 7 (the footnote is gone, earlier bullets greyed).*

*Figure: no*

## Slide 9 — HODL's "philosophy"

First three bullets (and sub-bullets) greyed out; new bullet in black:

- (greyed) Focus on the key concepts that underlie DL
- (greyed) Skip the math (but we are happy to geek out in office hours and/or suggest readings for those who are interested)
- (greyed) Focus on coding DL models …
  - (greyed) It is the only way to develop a visceral (not just intellectual) understanding of how this stuff works
  - (greyed) Successful new products and services are often inspired by hands-on tinkering
- … but only up to a point
  - We aren't trying to teach you how to be ML engineers
  - (red text) But we want you to be able to build a V1.0 DL model <u>by yourself</u> without looking for a Data Scientist or ML Engineer to help you out

*Build step: adds the "… but only up to a point" bullet and its two sub-bullets relative to slide 8.*

*Figure: no*

## Slide 10 — [Introduction to AI, ML, DL and Generative AI]

(No title.) "We will start with a <u>very quick</u> introduction to the relationship between"

- Artificial Intelligence (AI)
- Machine Learning (ML)
- Deep Learning (DL)
- Generative AI

*Figure: no*

## Slide 11 — The field of Artificial Intelligence originated in 1956

A black-and-white group photograph of seven men sitting on grass in front of a building with steps and a dark door. Caption printed under the photo (small monospaced text): "In the back row from left to right are Oliver Selfridge, Nathaniel Rochester, Marvin Minsky, and John McCarthy. In front on the left is Ray Solomonoff; on the right, Claude Shannon. The identity of the person between Solomonoff and Shannon remained a mystery for some time. THE MINSKY FAMILY". Below the photo, a URL: "https://spectrum.ieee.org/dartmouth-ai-workshop". (This is a photo of the 1956 Dartmouth AI workshop participants.)

*OCW notice: © the Minsky Family (photo by Nathaniel Rochester, 1956). All rights reserved — excluded from the CC license.*

*Figure: yes — photograph of the 1956 Dartmouth workshop participants*

## Slide 12 — MIT was well-represented 👍

Same photo, caption and URL as slide 11 ("https://spectrum.ieee.org/dartmouth-ai-workshop"), with three black arrows from people in the photo to labels on the right: "Marvin Minsky" (arrow to the man in the back row, centre-left), "John McCarthy" (arrow to the man in the back row, right of centre), "Claude Shannon" (arrow to the man at front right). The title is followed by a thumbs-up emoji.

*Build step: same photo as slide 11, now with arrows and name labels for Minsky, McCarthy and Shannon.*

*OCW notice: © the Minsky Family (photo by Nathaniel Rochester, 1956). All rights reserved — excluded from the CC license.*

*Figure: yes — photograph with name-label arrows*

## Slide 13 — In the decades since its founding, it has gone through several "breakthroughs"

A flow diagram of four rounded boxes in a row joined by dashed block arrows pointing right: "Traditional AI" (black outline) → "Machine Learning (ML)" (red outline) → "Deep Learning (DL)" (red outline) → "Generative AI" (red outline). Shows the progression of AI approaches.

*Figure: yes — flow diagram of AI → ML → DL → Generative AI*

## Slide 14 — The traditional approach to AI

Top right: a small version of the four-box flow diagram (Traditional AI → Machine Learning (ML) → Deep Learning (DL) → Generative AI (e.g., ChatGPT)), with "Traditional AI" highlighted by a red outline.

Left: a large dark-red filled circle labelled "Artificial Intelligence".

Right text:
- <u>The Goal</u>: Give computers the ability to do tasks that traditionally only humans have been able to do (italic)
- <u>Traditional approach</u>: (red) Ask human experts how they do it, write it down as IF-THEN rules, explicitly program these rules into the computer
- A dashed down-arrow, then red text: "Success in only a few areas"

*Figure: yes — flow diagram (top right) and AI circle*

## Slide 15 — Why is this so difficult?

- "We know more than we can tell" (Polanyi's Paradox)
  - We can do lots of things easily but find it very hard to describe how exactly we do them
- We can't write down if-then rules to cover all situations, edge cases etc. (i.e., we can't generalize to new situations)

*Figure: no*

## Slide 16 — To address this problem, a different approach was developed

A box containing: "Instead of explicitly telling the computer what to do …"

*Figure: no*

## Slide 17 — To address this problem, a different approach was developed

The box now has the first sentence greyed out ("Instead of explicitly telling the computer what to do …") and new red italic text: "Provide the computer with lots of <u>examples</u> of inputs-and-outputs and use <u>statistical techniques</u> to learn the relationship between inputs and outputs".

*Build step: adds the red "Provide the computer with lots of examples…" text relative to slide 16.*

*Figure: no*

## Slide 18 — This is Machine Learning

A Venn-style diagram: a large dark-red ellipse labelled "Artificial Intelligence" containing a smaller orange ellipse labelled "Machine Learning" (ML is a subset of AI). To the right, text: "learn from input-output examples using statistical techniques*" (the asterisk is printed with no matching footnote on this slide).

*Figure: yes — nested-ellipse (Venn) diagram, ML inside AI*

## Slide 19 — There are numerous ways* to create Machine Learning models

A diagram: the word "Input" with a dark-red arrow into a red-outlined box labelled "Machine Learning Model" (red text), with an arrow out to "Output".

Two bullet columns below:
- Left: Linear Regression; Logistic Regression; Classification and Regression Trees; Support Vector Machines
- Right: Random Forests; Gradient Boosted Machines; Neural Networks; …..

Footnote: "*Covered in detail in courses like 15.071 The Analytics Edge"

*Figure: yes — input → ML model → output diagram*

## Slide 20 — [Machine Learning impact on structured data]

(No title.) "Machine Learning has had tremendous impact and is used worldwide across numerous applications (e.g., credit scoring, loan granting, disease prediction, demand forecasting, ....) <red>where the input data is <u>structured</u></red>"

*Figure: no*

## Slide 21 — Structured input data = data that can be "numericalized" into a spreadsheet*

(The word "Structured" is in red.) A spreadsheet-style table with a merged header row "INPUT" over the first six columns and "OUTPUT" over the last column (the output column is shaded green):

| Age | Smoker | Exercise | Cholesterol | Family History | Blood Pressure | Cardiac Arrest (OUTPUT) |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | No | 120 | 190 | Yes | 120/80 | No |
| 45 | Yes | 30 | 220 | No | 130/90 | Yes |
| 50 | No | 60 | 210 | Yes | 125/85 | No |
| 35 | Yes | 45 | 230 | No | 135/88 | Yes |
| 40 | No | 150 | 180 | Yes | 118/78 | No |
| 55 | Yes | 10 | 240 | Yes | 140/92 | Yes |
| 28 | No | 180 | 170 | No | 115/75 | No |
| 60 | Yes | 20 | 250 | Yes | 145/95 | Yes |
| 48 | No | 90 | 200 | No | 128/82 | No |
| 53 | Yes | 35 | 235 | Yes | 133/89 | Yes |

Footnote: "*informal definition"

*Figure: yes — spreadsheet table of structured data (screenshot-style)*

## Slide 22 — But the situation is different for unstructured input data (images, videos, text, audio, ...)

("unstructured" is red and underlined.) Three labelled rows:
- **Images** — a photograph of a cream/yellow Labrador puppy sitting on grass beside a gravel path, with a red leash.
- **Text** — italic text: "Four score and seven years ago our fathers brought forth, upon this continent, …"
- **Audio** — "…"

*OCW notice: © unknown (puppy image). All rights reserved — excluded from the CC license.*

*Figure: yes — photograph of a puppy plus text/audio examples*

## Slide 23 — The reason: The "raw form" of unstructured data has no intrinsic meaning

Left: the puppy photo (same as slide 22). Right: three overlapping printed numeric matrices (R-style output with column headers [,1] … [,10] and row headers [1,] … [10,]), stacked like colour channels and labelled "Red" (back, top right), "Green" (middle) and "Blue" (front). Each shows pixel intensities 0–255 for a 10×10 patch. The Blue matrix is fully visible:

| | [,1] | [,2] | [,3] | [,4] | [,5] | [,6] | [,7] | [,8] | [,9] | [,10] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [1,] | 251 | 232 | 233 | 237 | 230 | 243 | 255 | 255 | 250 | 246 |
| [2,] | 248 | 234 | 239 | 245 | 238 | 246 | 255 | 251 | 246 | 243 |
| [3,] | 255 | 241 | 238 | 236 | 229 | 241 | 253 | 249 | 238 | 234 |
| [4,] | 255 | 252 | 243 | 233 | 228 | 237 | 242 | 234 | 218 | 205 |
| [5,] | 255 | 255 | 249 | 231 | 228 | 231 | 224 | 215 | 204 | 166 |
| [6,] | 255 | 255 | 230 | 192 | 189 | 202 | 205 | 205 | 204 | 147 |
| [7,] | 231 | 231 | 188 | 140 | 138 | 152 | 156 | 159 | 177 | 136 |
| [8,] | 155 | 172 | 149 | 114 | 113 | 111 | 93 | 82 | 119 | 115 |
| [9,] | 107 | 130 | 108 | 93 | 113 | 100 | 67 | 66 | 81 | 95 |
| [10,] | 84 | 104 | 90 | 69 | 69 | 61 | 52 | 63 | 59 | 46 |

The Green matrix is partly hidden behind Blue; visible rows: [1,] 186 171 179 185 171 172 180 171 168 180; [2,] 177 169 180 188 176 175 178 167 169 180; row [3,] only partly visible (starts "175 169 174 176 169 173 178 172 171 183", faint) and rows 4–10 hidden. The Red matrix visible rows: [1,] 147 131 138 144 131 134 144 135 133 145; [2,] 140 131 141 149 138 138 143 132 136 146; remaining rows hidden. Point: the raw form of an image is just arrays of numbers per colour channel with no intrinsic meaning.

*OCW notice: © unknown (puppy image). All rights reserved — excluded from the CC license.*

*Figure: yes — puppy photo with three stacked RGB pixel-value matrices*

## Slide 24 — To use ML on unstructured data, we had to manually create a better representation of the data first*

("manually" and "representation" are red.) Diagram: on the left, caption "“Unstructured” data" above six photographs of birds (black crows/ravens and yellow goldfinches) arranged in a staggered layout, each with a tiny URL caption. A thick black arrow labelled "Manual feature extraction" points right to a table headed "Datasets in “structured” form like tables":

| | Beak length | Wing span | Primary color | … |
| --- | --- | --- | --- | --- |
| Bird 1 | 3 cm | 43 cm | black | … |
| Bird 2 | 0.5 cm | 19 cm | yellow | … |
| Bird 3 | 5 cm | 48 cm | black | … |
| Bird 4 | 4 cm | 45 cm | black | … |
| … | … | … | … | … |

Below: "Figure source: https://magazine.sebastianraschka.com/p/ai-and-open-source-in-2023". Footnote: "*sometimes referred to as feature extraction".

*OCW notice: © Sebastian Raschka (unstructured data figure); bird images © Edwin Butter/Shutterstock, © Svetlana Foote/Shutterstock, © Mick Thompson, © Ian Routley, © Marin Audubon Society, © Jay McGowan (bird photographs). All rights reserved — excluded from the CC license.*

*Figure: yes — bird photos → manual feature extraction → structured table*

## Slide 25 — Learning effective representations is vital

("representations" is red.)

- The raw data has to somehow be transformed into a different representation
- Historically, researchers <red>manually</red> developed these representations and then fed them to traditional machine learning algorithms (often just linear/logistic regression!)
- But this required massive human effort and thus sharply limited the reach of Machine Learning

*Figure: no*

## Slide 26 — [Human bottleneck]

(No title.) "But developing good representations (before ML could be used) required massive human effort and this “human bottleneck” sharply limited the reach of Machine Learning"

*Figure: no*

## Slide 27 — To address this problem, a different approach was developed – Deep Learning

("Deep Learning" in the title is red.) Nested ellipses: a large dark-red ellipse "Artificial Intelligence" containing an orange ellipse "Machine Learning" containing a white ellipse with red text "Deep Learning" (DL ⊂ ML ⊂ AI).

*Build step: adds the innermost "Deep Learning" ellipse to the AI/ML diagram of slide 18.*

*Figure: yes — nested-ellipse (Venn) diagram AI ⊃ ML ⊃ DL*

## Slide 28 — Deep Learning can handle unstructured input data without upfront manual preprocessing!!

("without upfront manual preprocessing" is underlined.) Left: the nested AI ⊃ ML ⊃ DL ellipses (Artificial Intelligence / Machine Learning / Deep Learning). Right: text "Structured *and* Unstructured data" ("Unstructured" red) with an arrow into a dark-red rounded box "Deep Learning", then an arrow to "Prediction". Beneath the text are three example inputs stacked vertically: a small blurry spreadsheet of numbers (structured data), the puppy photo, and italic text "Four score and seven years ago our fathers brought forth, upon this continent, …".

*OCW notice: © unknown (puppy image). All rights reserved — excluded from the CC license.*

*Figure: yes — nested-ellipse diagram plus structured/unstructured data → DL → prediction*

## Slide 29 — What can Deep Learning do that traditional Machine Learning can't?

- (red) It can <u>automatically</u> extract smart representations from raw, unstructured data.
- We can simply feed these smart representations to traditional models like linear regression or logistic regression and achieve amazing performance

*Figure: no*

## Slide 30 — [Human bottleneck demolished]

(No title.) "This demolishes the “human bottleneck” for using Machine Learning with unstructured data"

*Figure: no*

## Slide 31 — Deep Learning

(Red) "The breakthrough came from the confluence of three forces …"
- New algorithmic ideas
- Unprecedented amounts of data (due to the digitization of everything)
- Compute power (from the use of powerful Graphics Processing Units (GPUs))

*Figure: no*

## Slide 32 — Deep Learning

Left: the three-forces text of slide 31 greyed out. A dark-red arrow points to red text on the right: "… applied to an old ML idea: Neural Networks". Below it a small network drawing: three red input circles labelled "Input" (left column), four blue hidden circles labelled "Hidden" (middle column), two green output circles labelled "Output" (right column); every input node is connected by an arrow to every hidden node, and every hidden node to every output node (fully connected, 3-4-2). Under it, small link: "https://en.wikipedia.org/wiki/Artificial_neural_network". Credit line: "Artificial neural network figure by Glossier.ca. License: CC BY-SA. Source: Wikimedia Commons." (a CC attribution, not an "all rights reserved" exclusion notice).

*Build step: greys out slide 31's list and adds the "applied to an old ML idea: Neural Networks" text and network figure.*

*Figure: yes — fully connected 3-4-2 neural network drawing*

## Slide 33 — What is the immediate application of Deep Learning?

Section-divider slide with a large dark-red block layout: a grey square and dark-red banner at the top and a dark-red text box reading "What is the *immediate* application of Deep Learning?" (white text, "immediate" bold italic).

*Figure: no*

## Slide 34 — [Every "sensor" can be given the ability…]

(No title.) Italic text: "Every “sensor” can be given the ability to detect, recognize and classify what it is sensing"

*Figure: no*

## Slide 35 — Examples

A collage of four example applications:
- Top left: an Apple "Use Face ID on your iPhone or iPad Pro" web graphic ("Face ID lets you securely unlock your iPhone or iPad, authenticate purchases, sign in to apps, and more — all with just a glance.") showing three phones: face scan with green dotted ring, lock screen at 9:41, and payment confirmation.
- Bottom left: medical images labelled "Breast cancer" (mammogram with a yellow box), "COVID-19" (chest X-ray), and "Skin cancer" (four skin-lesion photographs), with "6.S191 Introduction to Deep Learning / introtodeeplearning.com @MITDeepLearning" footers.
- Top right: a New York City street photo with red bounding boxes around cars (a yellow taxi), pedestrians and a traffic light (object detection), with the "6.S191 Introduction to Deep Learning" footer.
- Bottom right: "Smartphone glass defect" (grey image of a curved glass edge) with a red arrow to "Detected by system" (same image with red box, circles highlighting defects and score "0.99"). URL: https://dwfritz.com/smart-cosmetic-defect-detection-increases-productivity/

*OCW notice: © Apple, Inc. (iPhone images); © Al-Masry (breast cancer); © unknown (COVID, cancer); © Abhijit Ramesh (city street); © Springer Nature (glass defect). All rights reserved — excluded from the CC license.*

*Figure: yes — collage of DL application screenshots/photographs*

## Slide 36 — [You can create dramatically better products by "attaching" DL to sensors]

(No title.) Top: the greyed-out italic text of slide 34 ("Every “sensor” can be given the ability to detect, recognize and classify what it is sensing"). A dark-red down-arrow, then black italic text: "You can create dramatically better products and services by “attaching” DL to sensors"

*Build step: adds the arrow and the "You can create dramatically better products…" text to slide 34's statement (greyed).*

*Figure: no*

## Slide 37 — (Spotted last week!) Binoculars + DL = Smart Binoculars

An image from the Swarovski Optik AX VISIO website: green-and-black binoculars on a light grey background, large white text "AX VISIO" and "THE FIRST SMART BINOCULARS". URL: https://www.swarovskioptik.com/us/en/hunting/products/binoculars/ax-visio

*OCW notice: © Swarovski Optik (images from AX VISIO website). All rights reserved — excluded from the CC license.*

*Figure: yes — product image of smart binoculars*

## Slide 38 — [Binoculars + DL = bird identification]

(No title.) Left: a photo of a bird (cedar waxwing) on a branch with red berries, with two partial red arc brackets around the bird. Centre text: "+ DL =". Right: the same scene (portrait crop) with four red corner arcs framing the bird and the label "Cedar Waxwing" in red at the bottom, i.e. the system identifies the species. URL: https://www.swarovskioptik.com/us/en/hunting/products/binoculars/ax-visio

*OCW notice: © Swarovski Optik (images from AX VISIO website). All rights reserved — excluded from the CC license.*

*Figure: yes — before/after bird photos showing DL species label*

## Slide 39 — Now, let's turn our attention to the output

Diagram: "Structured *and* Unstructured data" (arrow) → dark-red rounded box "Deep Learning" → (arrow) → red "?" (the output is unknown). A dashed arrow from the title area points down to the "?".

*Figure: yes — input → Deep Learning → "?" diagram*

## Slide 40 — With Deep Learning, we could predict/generate structured outputs easily

("structured" and "outputs" are red and underlined.) Left: "Structured *and* Unstructured data" → "Deep Learning" box → arrow. Right, "<u>Examples</u>:"
- (red) A single number
  - The probability that a borrower will repay a loan
  - The demand for a product next week
- (red) A few numbers
  - The 4 probabilities that an image contains a chair, stool, table or sofa
  - The two GPS coordinates of a taxi

*Build step: replaces the "?" of slide 39 with a list of structured-output examples.*

*Figure: yes — input → Deep Learning → output diagram*

## Slide 41 — But we couldn't generate unstructured output very well!

("unstructured output" is red and underlined.) Diagram: "Structured *and* Unstructured data" → dark-red box "Deep Learning" → arrow to three output types stacked and labelled: "Images" (the puppy photo), "Text" (italic: "Four score and seven years ago our fathers brought forth, upon this continent, …"), "Audio" ("…"). A dashed arrow from above points down at the outputs.

*OCW notice: © unknown (puppy image). All rights reserved — excluded from the CC license.*

*Figure: yes — input → Deep Learning → unstructured output types (puppy photo)*

## Slide 42 — And then Generative AI happened

Nested ellipses: Artificial Intelligence (dark red) ⊃ Machine Learning (orange) ⊃ Deep Learning (white, red text) ⊃ "Gen AI" (small pale green-grey ellipse at the bottom).

*Build step: adds the innermost "Gen AI" ellipse to the diagram of slide 27.*

*Figure: yes — nested-ellipse diagram AI ⊃ ML ⊃ DL ⊃ Gen AI*

## Slide 43 — Gen AI can produce unstructured outputs

("unstructured outputs" is red.) Left: the nested AI ⊃ ML ⊃ DL ⊃ Gen AI ellipses. Right: "Unstructured input" → dark-red box "Gen AI" → red "Unstructured output". Below the input: the quoted prompt "generate a picture of a cute labrador retriever puppy"; below the output: the puppy photo.

*OCW notice: © unknown (puppy image). All rights reserved — excluded from the CC license.*

*Figure: yes — nested-ellipse diagram and input → Gen AI → output with puppy image*

## Slide 44 — Image-to-text

Diagram: a photograph of two tabby cats lying on a bright pink blanket with two TV remotes → dark-red box "Gen AI" → italic text output "two cats laying on a pink blanket with remotes" [sic — "laying" for "lying" as printed]. URL: https://huggingface.co/spaces/nielsr/comparing-captioning-models

*OCW notice: © nielsr on Hugging Face (image of cats). All rights reserved — excluded from the CC license.*

*Figure: yes — image → Gen AI → caption diagram with cat photo*

## Slide 45 — Text-to-image

Diagram: a block of grey bold prompt text → dark-red box "Gen AI" → an image of ancient stone temple ruins with columns and steps in a misty forest. The prompt text reads: "temple in ruins, forest, stairs, columns, cinematic, detailed, atmospheric, epic, concept art, Matte painting, background, mist, photo-realistic, concept art, volumetric light, cinematic epic + rule of thirds octane render, 8k, corona render, movie concept art, octane render, cinematic, trending on artstation, movie concept art, cinematic composition , ultra-detailed, realistic , hyper-realistic , volumetric lighting, 8k –ar 2:3 –test –uplight". Caption: "Image generated using Stable Diffusion AI from prompt quoted above." URL: https://mpost.io/best-100-stable-diffusion-prompts-the-most-beautiful-ai-text-to-image-prompts/

*Figure: yes — text prompt → Gen AI → generated temple image*

## Slide 46 — Text-to-music

Diagram: prompt text "The main soundtrack of an arcade game. It is fast-paced and upbeat, with a catchy electric guitar riff. The music is repetitive and easy to remember, but with unexpected sounds, like cymbal crashes or drum rolls." → dark-red box "Gen AI" → a grey loudspeaker icon (audio output). URL: https://google-research.github.io/seanet/musiclm/examples/. Caption: "Audio generated using MusicLM from prompt the main soundtrack of an arcade game. It is fast-paced and upbeat, with a catchy electric guitar riff. The music is repetitive and easy to remember, but with unexpected sounds, like cymbal crashes or drum rolls."

*Figure: yes — text → Gen AI → audio (speaker icon) diagram*

## Slide 47 — Text-to-text

Diagram: "text" → dark-red box "GenAI" → "text".

*Figure: yes — text → GenAI → text diagram*

## Slide 48 — Multi-modal

Diagram: "[text, image]" ("image" in red) → dark-red box "GenAI" → "text".

*Figure: yes — [text, image] → GenAI → text diagram*

## Slide 49 — A fun multi-modal example

Diagram: "[text, image]" (image in red) → dark-red box "ChatGPT" → arrow (output not yet shown). Below the image slot: a photograph of a tall pole stacked with many parking signs (red "NO PARKING 11AM–1PM STREET CLEANING", green "15 … SCHOOL BUSINESS" and "1 HOUR PARKING MON–FRI" signs, others cut off) in front of a tree. Under the "text" slot (dotted arrow) is a dark bar with the prompt: "It's Wednesday at 4 pm. Can I park at this spot right now? Tell me in 1 line." URL: https://twitter.com/petergyang/status/1707169696049668472

*OCW notice: © petergyang on Twitter/unknown (parking signs photo). All rights reserved — excluded from the CC license.*

*Figure: yes — multi-modal ChatGPT prompt diagram with parking-signs photo*

## Slide 50 — A fun multi-modal example

Same as slide 49, now with the output: "text" label after the ChatGPT box and a dark chat bubble with the ChatGPT logo reading "Yes, you can park for up to 1 hour starting at 4 pm."

*Build step: adds the output "text" label and ChatGPT's answer relative to slide 49.*

*OCW notice: © petergyang on Twitter/unknown (parking signs photo). All rights reserved — excluded from the CC license.*

*Figure: yes — multi-modal ChatGPT example with answer*

## Slide 51 — The landscape

Nested-ellipse diagram: Artificial Intelligence (dark red) ⊃ Machine Learning (orange) ⊃ Deep Learning (white, red text) ⊃ Gen AI (pale ellipse, label partly hidden "Gen AI"). Inside Gen AI are three small circles: a dark-blue circle (arrow label "Text-to-image", left), a purple circle (arrow label "Text-to-music", right), and a red circle labelled "LLM" containing two black dots with arrows to labels "ChatGPT" and "Bard". Right-hand text: "… and the circles inside GenAI are merging!"

*Figure: yes — nested-ellipse landscape diagram of Gen AI sub-areas*

## Slide 52 — Summary: X-to-Y

Diagram: "X" → dark-red box "Deep Learning" → "Y". Text: "X and Y can be *anything <u>and</u> it can be multi-modal!*"

*Figure: yes — X → Deep Learning → Y diagram*

## Slide 53 — Note that ALL the AI excitement is due to the success of Deep Learning

Same nested-ellipse landscape diagram as slide 51 (AI ⊃ ML ⊃ Deep Learning ⊃ Gen AI with Text-to-image, Text-to-music, LLM, ChatGPT, Bard), with the "Deep Learning" label emphasised in larger red text; the right-hand "… and the circles inside GenAI are merging!" text is removed.

*Build step: same diagram as slide 51 with "Deep Learning" label enlarged and the right-hand text removed.*

*Figure: yes — nested-ellipse landscape diagram*

## Slide 54 — If you understand Deep Learning, everything becomes possible! 🙂

Section-divider: white text on a dark-red banner (grey square at left), followed by a smiling emoji.

*Figure: no*

## Slide 55 — OK, let's start at the beginning. What's a Neural Network?

Section-divider: white text on a dark-red banner (grey square at left).

*Figure: no*

## Slide 56 — Recall Logistic Regression

Diagram: "[x$_1$, x$_2$, x$_3$, …, x$_n$]" → dark-red box "Logistic Regression" → "probability". Below, the formula:

$$Pr(Y=1) = \frac{1}{1+e^{-(\beta_0+\beta_1 x_1+\beta_2 x_2+\cdots+\beta_n x_n)}}$$

*Figure: yes — input vector → Logistic Regression → probability diagram*

## Slide 57 — Recall Logistic Regression

$$Pr(Y=1) = \frac{1}{1+e^{-(\beta_0+\beta_1 x_1+\beta_2 x_2+\cdots+\beta_n x_n)}}$$

"Let us look at it through the lens of a “network” of mathematical operations."

*Figure: no*

## Slide 58 — We will work with this simple classification example

- Given independent variables …
  - GPA
  - Experience
- … predict who will be called for a job interview

Right: a monospaced data listing with columns "Interview", "GPA", "Experience" (row labels 1–16, then "...", then 25–29):

| Row | Interview | GPA | Experience |
| --- | --- | --- | --- |
| 1 | 0 | 3.27 | 1.93 |
| 2 | 0 | 3.37 | 0.07 |
| 3 | 0 | 3.57 | 1.91 |
| 4 | 0 | 3.91 | 4.35 |
| 5 | 0 | 3.20 | 1.70 |
| 6 | 1 | 3.90 | 2.41 |
| 7 | 1 | 3.94 | 3.00 |
| 8 | 0 | 3.66 | 2.47 |
| 9 | 0 | 3.63 | 0.93 |
| 10 | 0 | 3.06 | 4.14 |
| 11 | 0 | 3.21 | 3.34 |
| 12 | 0 | 3.18 | 3.97 |
| 13 | 0 | 3.69 | 0.54 |
| 14 | 0 | 3.38 | 3.62 |
| 15 | 1 | 3.77 | 2.06 |
| 16 | 0 | 3.50 | 4.10 |
| ... | ... | ... | ... |
| 25 | 1 | 3.31 | 3.46 |
| 26 | 0 | 3.78 | 0.29 |
| 27 | 0 | 3.87 | 1.21 |
| 28 | 0 | 4.00 | 0.49 |
| 29 | 1 | 3.87 | 2.11 |

*Figure: yes — data listing (screenshot of R-style output)*

## Slide 59 — We will work with this simple classification example

Left: the same data listing as slide 58 (rows 1–16, "...", 25–29, same values). Right: "We can estimate this logistic regression model and find values for the coefficients:"

$$P(Y=1) = \frac{1}{1+e^{-(0.4+0.2\cdot \text{GPA}+0.5\cdot \text{Experience})}}$$

Below it, a line chart of the standard sigmoid curve (the x-axis is unlabelled): x-axis from -5 to 5 (integer ticks), y-axis gridlines labelled 0, 0.5, 1. One dark-red series, an S-shaped (logistic) curve: near 0 at x = -5, about 0.12 at x = -2, crossing 0.5 at x = 0, about 0.88 at x = 2, and flattening toward 1 at x = 5. (Values read approximately from the plot; no data labels printed. Corrected after a check against the rendered page: the first reading put the 0.5 crossing near x = 0.5.)

*Build step: adds the fitted model equation and the sigmoid plot relative to slide 58 (data listing moves to the left).*

*Figure: yes — data listing and sigmoid (logistic) curve plot*

## Slide 60 — We can re-write this formula as a network with input data flowing through two functions that have been connected

("network" is red; "two" is underlined.) "Model equation:"

$$P(Y=1) = \frac{1}{1+e^{-(0.4+0.2\cdot \text{GPA}+0.5\cdot \text{Experience})}}$$

The rest of the slide is empty (the network is built up on the following slides).

*Figure: no*

## Slide 61 — We can re-write this formula as a network with input data flowing through two functions that have been connected

("network" red; "two" underlined.) "Model equation:" and

$$P(Y=1) = \frac{1}{1+e^{-(0.4+0.2\cdot \text{GPA}+0.5\cdot \text{Experience})}}$$

Network diagram: two empty circles on the left labelled "GPA" (top) and "Experience" (bottom). A dark-red arrow from the GPA circle to a circle containing "+" is labelled ".2 * GPA" (the ".2" in red); an arrow from the Experience circle to the same "+" circle is labelled ".5 * Experience" (".5" in red). A green "0.4" with an arrow points into the "+" circle (the bias). An arrow from "+" goes to a circle containing $\frac{1}{1+e^{-x}}$ (the sigmoid), and an arrow from it to the full formula $\frac{1}{1+e^{-(0.4+0.2\cdot \text{GPA}+0.5\cdot \text{Experience})}}$ at the right.

*Build step: adds the network drawing (GPA and Experience nodes, weights .2 and .5, bias 0.4, "+" node, sigmoid node) relative to slide 60.*

*Figure: yes — logistic regression drawn as a network (GPA, Experience → + → sigmoid)*

## Slide 62 — Let's make a prediction with this "network"

"Consider a job applicant with a GPA of 3.8 and 1.2 years of experience."

Same network with numbers filled in (blue = values): GPA node contains 3.8; Experience node contains 1.2. Working beside the GPA edge: ".2 GPA = .2(3.8) = .76" (".2" red, "3.8" blue). Working beside the Experience edge: ".5 Experience = .5(1.2) = 0.6". Green bias "0.4" into the "+" node. Output of the "+" node: "= 1.76" (blue). Then the sigmoid node $\frac{1}{1+e^{-x}}$, whose output is "0.85" (blue). A double-headed red vertical arrow labelled "Same as" connects the 0.85 to the formula

$$\frac{1}{1+e^{-(0.4+0.2\cdot 3.8+0.5\cdot 1.2)}}$$

*Build step: adds the worked numerical example (3.8, 1.2 → 0.76, 0.6, 1.76 → 0.85) to the network of slide 61.*

*Figure: yes — network with worked numerical prediction*

## Slide 63 — The general logistic regression model viewed through a network lens

"Model equation:"

$$P(Y=1) = \frac{1}{1+e^{-(11+3x_1-9x_2+\cdots+17x_k)}}$$

Network: input circles $x_1$, $x_2$, a vertical column of three dots, and $x_k$. Red edge labels: "$3x_1$" (from $x_1$), "$-9x_2$" (from $x_2$), "$17x_k$" (from $x_k$), all arrows into a "+" circle; a green "11" arrow also points into "+" (bias). "+" → circle with $\frac{1}{1+e^{-x}}$ → the formula $\frac{1}{1+e^{-(11+3x_1-9x_2+\cdots+17x_k)}}$. Text: "Notice how the data flows through the network from left to right".

*Figure: yes — general logistic regression drawn as a network*

## Slide 64 — Terminology

Same network as slide 63 (without the model-equation line): $x_1$, $x_2$, …, $x_k$ with red edge labels $3x_1$, $-9x_2$, $17x_k$, green bias 11, "+" node, sigmoid node, and final formula $\frac{1}{1+e^{-(11+3x_1-9x_2+\cdots+17x_k)}}$.

- Multipliers on values from each node = coefficients = <red>weights</red>
- Intercept = <green>bias</green>

*Figure: yes — logistic regression network with terminology (weights red, bias green)*

## Slide 65 — What's the advantage of viewing through a network "lens"?

Section-divider: white text on a dark-red banner (grey square at left).

*Figure: no*

## Slide 66 — Recall the notion of learning smart representations of the input data

("learning smart representations" is red.) Network: input circles $x_1$, $x_2$, three dots, $x_k$, each with an arrow to a "+" circle; "+" → circle $\frac{1}{1+e^{-x}}$ → output arrow. No weights or biases drawn. Footnote: "*I am not showing the weights and biases to avoid clutter"

*Figure: yes — logistic regression network without weights*

## Slide 67 — To learn smart representations, we would like to transform the inputs one or more times before we do the prediction

("transform the inputs one or more times" red; "before" underlined.) Left: input circles $x_1$, $x_2$, three dots, $x_k$. Centre: a large red "?". Right: a "+" circle → sigmoid circle $\frac{1}{1+e^{-x}}$ → output arrow. No edges from the inputs to the "+" node.

*Figure: yes — inputs, "?" gap, and "+"/sigmoid output stage*

## Slide 68 — To learn smart representations, we would like to transform the inputs one or more times before we do the prediction

Same as slide 67, with the "?" replaced by red text: "What is the simplest thing we can do here?"

*Build step: replaces the "?" of slide 67 with the question "What is the simplest thing we can do here?"*

*Figure: yes — partial network diagram (inputs, gap, "+"/sigmoid)*

## Slide 69 — To learn smart representations, we would like to transform the inputs one or more times before we do the prediction

Same as slide 68; the question "What is the simplest thing we can do here?" is now black, and new red text below: "We can insert linear functions".

*Build step: adds "We can insert linear functions" relative to slide 68.*

*Figure: yes — partial network diagram*

## Slide 70 — To learn smart representations, we would like to transform the inputs one or more times before we do the prediction

Left: inputs $x_1$, $x_2$, dots, $x_k$ each with a dark-red arrow into a single "+" circle at the top, with red label "A linear function" above; an arrow leaves the "+" to the right. Right (unchanged): "+" → sigmoid $\frac{1}{1+e^{-x}}$ → output arrow.

*Build step: adds one "+" node (a linear function) connected from all inputs.*

*Figure: yes — partial network diagram, one linear function*

## Slide 71 — To learn smart representations, we would like to transform the inputs one or more times before we do the prediction

Three "+" circles stacked vertically, each receiving an arrow from every input $x_1$, $x_2$, $x_k$ (dark-red arrows into the top "+", light-blue arrows into the middle "+", black arrows into the bottom "+"); each "+" has an output arrow to the right. Red text: "We can ”stack” as many linear functions as we want". Right (unchanged): "+" → sigmoid → output.

*Build step: adds two more "+" nodes (three in total) fully connected from the inputs.*

*Figure: yes — network with three stacked linear functions*

## Slide 72 — To learn smart representations, we would like to transform the inputs one or more times before we do the prediction

Same diagram as slide 71, with red text: "Notice that we have transformed a k-dimensional input to a 3-dimensional vector".

*Build step: replaces the "stack" text of slide 71 with the k-dimensional → 3-dimensional note.*

*Figure: yes — network with three linear nodes*

## Slide 73 — To learn smart representations, we would like to transform the inputs one or more times before we do the prediction

Red text at top: "We can “flow” this 3-dimensional vector through another function". Same three "+" nodes fully connected from the inputs (red/blue/black arrows), and now each "+" has an arrow into a circle labelled $f(x)$, each with an output arrow. Right (unchanged): "+" → sigmoid → output.

*Build step: adds an $f(x)$ node after each "+" node relative to slide 72.*

*Figure: yes — network with three "+" nodes each followed by f(x)*

## Slide 74 — To learn smart representations, we would like to transform the inputs one or more times before we do the prediction

Same as slide 73, with red text at top right: "Notice that this vector is still 3-dimensional".

*Build step: swaps the top text for the "still 3-dimensional" note.*

*Figure: yes — network with three "+"/f(x) pairs*

## Slide 75 — To learn smart representations, we would like to transform the inputs one or more times before we do the prediction

Red text at top: "We can do this repeatedly". The first stage (three "+" → $f(x)$ pairs) is enclosed in a dotted rectangle; a second dotted rectangle contains a second stage of three "+" → $f(x)$ pairs. Each $f(x)$ of stage one has arrows to all three "+" nodes of stage two. Stage-two $f(x)$ nodes have output arrows to the right. Far right (unchanged): "+" → sigmoid → output, not yet connected.

*Build step: adds the second dotted stage of three "+"/f(x) pairs, fully connected from stage one.*

*Figure: yes — two-stage network*

## Slide 76 — To learn smart representations, we would like to transform the inputs one or more times before we do the prediction

Same two-stage network as slide 75, now with the three stage-two $f(x)$ outputs connected by arrows into the final "+" node, which feeds the sigmoid $\frac{1}{1+e^{-x}}$ node and an output arrow. Red text at top right: "And when we are done, we can “feed” this to logistic regression as usual!!"

*Build step: connects the stage-two outputs to the final "+"/sigmoid, completing the network.*

*Figure: yes — complete two-hidden-stage network*

## Slide 77 — Key Takeaway: Instead of feeding the "raw" input to logistic regression, we feed a repeatedly transformed input

Two panels separated by a vertical dark-red line: "Before" (left): the simple logistic regression network of slide 66 ($x_1$ … $x_k$ → "+" → sigmoid → output). "After" (right): the complete two-stage network of slide 76 ($x_1$ … $x_k$ → three "+"/f(x) → three "+"/f(x) → "+" → sigmoid → output).

*Figure: yes — before/after network diagrams*

## Slide 78 — This is a Neural Network!

The complete network of slide 76 drawn large with no annotations: inputs $x_1$, $x_2$, dots, $x_k$; first stage: three "+" nodes each followed by $f(x)$; second stage: three "+" nodes each followed by $f(x)$; then a final "+" node and a sigmoid node $\frac{1}{1+e^{-x}}$ with an output arrow. Edges as before (inputs to first stage in red/blue/black; stage one to stage two all-to-all; stage two to final "+").

*Build step: slide 76 without the dotted rectangles and annotation, titled "This is a Neural Network!".*

*Figure: yes — complete neural network diagram*

## Slide 79 — Terminology

Same network as slide 78, with dashed red rectangles around each "+"/$f(x)$ pair (three in stage one, three in stage two) and around the final "+"/sigmoid pair. Red text at top right: "These are called *Neurons*".

*Build step: adds dashed red boxes marking each "+"+function pair as a neuron, relative to slide 78.*

*Figure: yes — network with neurons boxed*

## Slide 80 — Aside: The "neural" connection

A colour illustration of three biological neurons in a row: A (yellow), B (blue), C (pink), with grey neurons in the background. Dark-red arrows show an "Impulse" entering A, passing to B and onward to C and out. Labels: "A", "B", "C", "Dendrite", "Axon", "Axon Terminal", "Synapse"; a circular inset magnifies the synapse (blue terminal with vesicles above a pink receiving membrane, labelled "Synapse"). Text: "In the brain, the dendrites of one neuron connect to the axons of others at synapses, forming a <u>network</u>"

*Figure: yes — illustration of biological neurons and a synapse*

## Slide 81 — Terminology

Same network as slides 78–79 (inputs $x_1$, $x_2$, …, $x_k$; two stages of three "+"/$f(x)$ pairs; final "+"/sigmoid). Large dashed red rectangles enclose the whole first stage (three "+"/$f(x)$ pairs, as one vertical stack), the whole second stage, and the final "+"/sigmoid pair. Red text at top right: "A vertical stack of neurons is called a *Layer*".

*Build step: replaces the per-neuron boxes of slide 79 with per-layer boxes.*

*Figure: yes — network with layers boxed*

## Slide 82 — Terminology

Same network, with dashed red boxes around each $f(x)$ node (three in stage one, three in stage two) and around the final sigmoid node $\frac{1}{1+e^{-x}}$. Red text at top right: "These are called *Activation Functions*".

*Build step: boxes the activation-function nodes instead of layers.*

*Figure: yes — network with activation functions boxed*

## Slide 83 — Terminology

Same network with dashed red boxes around: the input nodes (labelled "Input “Layer”" at bottom left), the first stage (labelled "Hidden Layer"), the second stage (labelled "Hidden Layer"), and the final "+"/sigmoid pair (labelled "Output Layer").

*Build step: boxes and names the input, hidden and output layers.*

*Figure: yes — network with input/hidden/output layers labelled*

## Slide 84 — Terminology

Same network with dashed red boxes around the first and second stages (the two hidden stacks). Red text at the bottom: "When every neuron in a layer is connected to every neuron in the next layer, it is called Dense or Fully Connected".

*Figure: yes — network with two stages boxed*

## Slide 85 — The general logistic regression model is an NN but a simple one since it has no hidden layers

("no" is underlined.) Model equation shown at top right:

$$P(Y=1) = \frac{1}{1+e^{-(11+3x_1-9x_2+\cdots+17x_k)}}$$

Network: input nodes $x_1$, $x_2$, dots, $x_k$ in a dashed red box labelled "Input “Layer”"; red edge labels "$3x_1$", "$-9x_2$", "$17x_k$"; green bias "11" into a "+" node; "+" → sigmoid $\frac{1}{1+e^{-x}}$ → output arrow; the "+"/sigmoid pair is enclosed in a dashed red box labelled "Output Layer".

*Figure: yes — logistic regression network with input and output layers boxed*

## Slide 86 — Deep Learning is just neural networks with lots and lots of …

("Deep Learning" underlined; "just" italic.) Title only; the body is empty.

*Figure: no*

## Slide 87 — Deep Learning is just neural networks with lots and lots of hidden layers

("hidden layers" is red.) A figure from the ResNet paper showing three architectures stacked vertically, each a left-to-right chain of narrow vertical blocks labelled with layer types:
- "34-layer residual" (top): "image" → "7x7 conv, 64, /2" → "pool, /2" → many "3x3 conv, 64" blocks (6), then "3x3 conv, 128, /2" and "3x3 conv, 128" blocks, then "3x3 conv, 256, /2" and "3x3 conv, 256" blocks, then "3x3 conv, 512, /2" and "3x3 conv, 512" blocks → "avg pool" → "fc 1000". Curved arcs above the blocks (solid, with a few dotted ones where dimensions change) show skip connections bridging pairs of layers.
- "34-layer plain" (middle): the same chain of blocks without skip-connection arcs.
- "VGG-19" (bottom): "image" → "3x3 conv, 64" ×2 → "pool, /2" → "3x3 conv, 128" ×2 → "pool, /2" → "3x3 conv, 256" ×4 → "pool, /2" → "3x3 conv, 512" ×4 → "pool, /2" → "3x3 conv, 512" ×4 → "pool, /2" → "fc 4096" → "fc 4096" → "fc 1000", with "output size" annotations 224, 112, 56, 28, 14, 7, 1 beneath.
The point is that deep networks have dozens of layers. Block text is very small; the labels above are read at best resolution and listed from the printed figure (exact block counts per group approximate).

URL: https://arxiv.org/pdf/1512.03385.pdf

*OCW notice: © Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun/arXiv. This content is excluded from our Creative Commons license. For more information, see https://ocw.mit.edu/help/faq-fair-use/. (Printed without "All rights reserved"; covers the architecture figure.)*

*Figure: yes — ResNet/VGG architecture diagram*

## Slide 88 — Let's now turn to Activation Functions

The same network as slide 82 (inputs $x_1$…$x_k$, two stages of "+"/$f(x)$, final "+"/sigmoid) with the seven activation-function nodes (three $f(x)$ in each stage, plus the final sigmoid) boxed in dashed red.

*Figure: yes — network with activation functions boxed*

## Slide 89 — Activation functions

Section-style slide with a dark-red banner title "Activation functions". Text: "The activation function of a node is just a function that receives a <u>single</u> number and outputs a <u>single</u> number (i.e., scalar in -> scalar out)". Diagram: two arrows entering a "+" circle, an arrow to a red-outlined circle "$f(x)$", and an arrow out (the $f(x)$ node and its arrows in red).

*Figure: yes — "+" node followed by f(x) node*

## Slide 90 — Common Activation Functions

"Sigmoid activation function:"

$$\sigma(a) = \frac{1}{1+e^{-a}}$$

Chart: x-axis "$a$" from -5 to 5 (integer ticks); y-axis gridlines labelled 0, 0.5, 1. One dark-red series, an S-shaped curve: ≈0.01 at a = -5, ≈0.12 at a = -2, 0.5 at a = 0, ≈0.88 at a = 2, ≈0.99 at a = 5 (approximate; no data labels).

*Figure: yes — plot of sigmoid function (1 series)*

## Slide 91 — Common Activation Functions

"Linear activation function:"

$$f(a) = a$$

Chart: x-axis "$a$" from -5 to 5, y-axis from -5 to 5 (integer gridlines). One dark-red series: a straight diagonal line through the origin from (-5, -5) to (5, 5).

*Figure: yes — plot of linear function (1 series)*

## Slide 92 — Common Activation Functions

"ReLU:"

$$g(a) = \max(0, a)$$

("**Re**ctified **L**inear **U**nit" — the letters "Re", "L" and "U" are bold.) Chart: x-axis "$a$" from -5 to 5, y-axis from 0 to 5. One dark-red series: flat at 0 from a = -5 to 0, then a straight line rising with slope 1 to (5, 5).

*Figure: yes — plot of ReLU function (1 series)*

## Slide 93 — We will use this "visual shorthand" from now on

Left: three node pictures with three incoming arrows each: (1) a "+" circle with output arrow only (no activation node); (2) a "+" circle followed by an oval "$max(0, x)$"; (3) a "+" circle followed by an oval $\frac{1}{1+e^{-x}}$. A large dark-red right-pointing block arrow in the middle. Right: three small circles containing mini-icons, labelled: a diagonal straight line = "Linear Activation"; a flat-then-rising hinge = "ReLU Activation"; an S-curve = "Sigmoid Activation".

*Figure: yes — mapping of node diagrams to visual shorthand icons*

## Slide 94 — Recap: Designing a DNN

Network diagram with column headings "Input Layer", "Hidden Layer 1", "Hidden Layer 2", "Output layer". Inputs $x_1$ … (vertical dots) … $x_k$ each with an arrow into the two circles of the input layer (labelled "…" between); the input-layer circles are fully connected to three circles of Hidden Layer 1; Hidden Layer 1 is fully connected to two circles of Hidden Layer 2; Hidden Layer 2 has outgoing arrows, followed by three dark-red dots ("●●●", meaning more layers), then a single circle (Output layer) with an output arrow.

Boxed text at bottom (italic): "<u>User</u> chooses the **# of hidden layers** (blue), **# units in each layer** (olive green), the **activation function(s)** (red) for the hidden layers and for the output layer"

*Figure: yes — generic deep neural network diagram*

## Slide 95 — Let's use a DNN for our interview classifier

- Recall the problem:
  - Two input variables (i.e., GPA and experience)
  - An output variable that should be between 0 and 1

Boxed text at bottom (as slide 94): "*User* chooses the **# of hidden layers**, **# units in each layer**, the **activation function(s)** for the hidden layers and for the output layer"

*Figure: no*

## Slide 96 — Let's use a DNN for our interview classifier

- Recall the problem:
  - Two input variables (i.e., GPA and experience)
  - An output variable that should be between 0 and 1
- Design choices
  - We will use **one hidden layer** (blue) with **3 neurons** (olive) **(ReLU)** (red-brown)
  - Since the output is constrained to be in (0,1), we will use the **sigmoid for the output layer** (red)

Boxed text at bottom (as slide 94).

*Build step: adds the "Design choices" bullets relative to slide 95.*

*Figure: no*

## Slide 97 — Let's practice setting up a simple NN

Only three column headings are shown: "Input Layer", "Hidden layer", "Output layer". The rest is empty.

*Figure: no*

## Slide 98 — Let's practice setting up a simple NN

Network: Input Layer with two circles $x_1$ and $x_2$; Hidden layer with three circles each containing the ReLU hinge icon; Output layer with one circle containing the sigmoid S-curve icon and output arrow labelled "y". Every input is connected by an arrow to every hidden node (6 edges), and every hidden node to the output node (3 edges). Red question at bottom: "How many parameters (i.e., weights and biases) does this network have?"

*Build step: adds the network drawing and the parameter-count question relative to slide 97.*

*Figure: yes — 2-3-1 network with ReLU hidden units and sigmoid output*

## Slide 99 — Let's practice setting up a simple NN

Same 2-3-1 network as slide 98, now with a red "?" on each of the 9 edges (weights) and a green "?" on each of the 4 nodes with biases (three hidden nodes and the output node). Bottom: "How many parameters (i.e., weights and biases) does this network have? 13" (the answer "13" in red). (6 + 3 weights + 3 + 1 biases = 13.)

*Build step: adds the weight/bias "?" markers and the answer 13.*

*Figure: yes — 2-3-1 network with parameter placeholders*

## Slide 100 — Let's assume that we have trained* this network on data and have found these values for the parameters

Same 2-3-1 network with parameter values filled in. Red = weights (edge labels), green = biases (node labels). Reading the labels by position:
- Biases: top hidden node -0.3; middle hidden node 0.2; bottom hidden node 0.5; output node 0.05.
- Input-to-hidden weights (six edges): 0.5 on the $x_1$ → top-hidden edge; -0.1 on the $x_1$ → middle-hidden edge; 0.2 on the $x_1$ → bottom-hidden edge; 0.1 on the $x_2$ → top-hidden edge; 0.3 on the $x_2$ → middle-hidden edge; -0.1 on the $x_2$ → bottom-hidden edge. (Edge assignment inferred from label placement near each edge; the labels 0.2 and 0.1 sit near the crossing edges.)
- Hidden-to-output weights: -0.2 (top hidden), -0.3 (middle hidden), -0.15 (bottom hidden).
- Hidden nodes use the ReLU icon; the output node uses the sigmoid icon; output labelled "y".

Footnote: "*details in the next class"

*Build step: fills the "?" placeholders of slide 99 with numbers.*

*Figure: yes — 2-3-1 network with parameter values*

## Slide 101 — Predicting with the NN

The trained 2-3-1 network of slide 100 with hidden-node outputs labelled $a_1$ (top), $a_2$ (middle), $a_3$ (bottom). Same parameter values: hidden biases -0.3, 0.2, 0.5 (green); output bias 0.05 (green); input-to-hidden weights 0.5, -0.1, 0.2 (from $x_1$ to top, middle, bottom) and 0.1, 0.3, -0.1 (from $x_2$ to top, middle, bottom) (red); hidden-to-output weights -0.2, -0.3, -0.15 (red); output "y".

"Output of **hidden layer** :"
- Top node: $\max(0,\ -0.3 + 0.5x_1 + 0.1x_2) = a_1$
- Middle node: $\max(0,\ 0.2 - 0.1x_1 + 0.3x_2) = a_2$
- Bottom node: $\max(0,\ 0.5 + 0.2x_1 - 0.1x_2) = a_3$

(Biases in green, weights in red.)

*Build step: adds node labels $a_1,a_2,a_3$ and the hidden-layer formulas to the network of slide 100.*

*Figure: yes — 2-3-1 network with parameter values and hidden-layer formulas*

## Slide 102 — Predicting with the NN

Same network and parameters as slide 101 (node labels $a_1$, $a_2$, $a_3$).

- Recall $a_1$, $a_2$, and $a_3$ are the output of the hidden layer nodes
- Output of **output layer node**:

$$\frac{1}{1+e^{-(0.05-0.2a_1-0.3a_2-0.15a_3)}}$$

(0.05 in green, the three weights in red.)

*Build step: replaces the hidden-layer formulas of slide 101 with the output-node formula.*

*Figure: yes — 2-3-1 network with parameter values and output-layer formula*

## Slide 103 — Predicting with the NN

Same network with example inputs filled in (blue): $x_1 = 2.3$, $x_2 = 10.2$. Hidden outputs shown beside the nodes: "1.87 = $a_1$", "3.03 = $a_2$", "0 = $a_3$". Weights, biases and structure as on slide 101.

"Output of **hidden layer nodes**:"
- Top node: $\max(0, -0.3 + 0.5 \cdot 2.3 + 0.1 \cdot 10.2) = 1.87 = a_1$
- Middle node: $\max(0, 0.2 - 0.1 \cdot 2.3 + 0.3 \cdot 10.2) = 3.03 = a_2$
- Bottom node: $\max(0, 0.5 + 0.2 \cdot 2.3 - 0.1 \cdot 10.2) = 0 = a_3$

(The bottom node's raw value is -0.06, clipped to 0 by ReLU.)

*Build step: plugs in x1 = 2.3 and x2 = 10.2 and the computed hidden outputs.*

*Figure: yes — 2-3-1 network with worked numerical forward pass (hidden layer)*

## Slide 104 — Predicting with the NN

Same network as slide 103 with output label "y = 0.226" (blue) on the output arrow. "Output of **output layer node**:"

$$\frac{1}{1+e^{-(0.05-0.2\cdot 1.87-0.3\cdot 3.03-0.15\cdot 0)}} = 0.226$$

*Build step: adds the output-layer calculation and y = 0.226 to slide 103.*

*Figure: yes — 2-3-1 network with worked numerical forward pass (output)*

## Slide 105 — The Network can be written as this function

Top: a small copy of the labelled 2-3-1 network (parameters as in slide 101). A large double-headed dark-red arrow with the red word "Equivalent". Bottom formula:

$$y = \frac{1}{1+e^{-(.05-0.2(\max(0,-0.3+0.5x_1+0.1x_2))-0.3(\max(0,0.2-0.1x_1+0.3x_2))-0.15(\max(0.5+0.2x_1-0.1x_2))}}$$

[sic — as printed, the third max term reads "max(0.5+0.2x_1−0.1x_2)" without the "0," argument, and the final closing parenthesis of the exponent is missing / parentheses are unbalanced; the intended expression is $\max(0, 0.5+0.2x_1-0.1x_2)$ as on slide 101.]

*Figure: yes — network diagram with equivalent closed-form function*

## Slide 106 — Contrast with the Logistic Regression model from before

Top: the logistic-regression network of slide 61: nodes "GPA" and "Experience", edge labels ".2 * GPA" and ".5 * Experience" (".2", ".5" red), green bias "0.4" into "+", sigmoid node $\frac{1}{1+e^{-x}}$, and the formula $\frac{1}{1+e^{-(0.4+0.2\cdot \text{GPA}+0.5\cdot \text{Experience})}}$. A double-headed dark-red arrow with red word "Equivalent". Bottom:

$$P(Y=1) = \frac{1}{1+e^{-(0.4+0.2\cdot \text{GPA}+0.5\cdot \text{Experience})}}$$

*Figure: yes — logistic regression network with equivalent formula*

## Slide 107 — Note the complexity of even this simple network compared to the logistic regression model

First formula (the neural network):

$$y = \frac{1}{1+e^{-(.05-0.2(\max(0,-0.3+0.5x_1+0.1x_2))-0.3(\max(0,0.2-0.1x_1+0.3x_2))-0.15(\max(0.5+0.2x_1-0.1x_2))}}$$

[sic — same missing "0," in the third max and unbalanced parentheses as slide 105.]

Second formula (logistic regression):

$$y = \frac{1}{1+e^{-(0.4+0.2x_1+0.5x_2)}}$$

Text: "*y* is a much more complex function of its inputs *x$_1$* and *x$_2$* compared to (say) [small inline image of $\frac{1}{1+e^{-(0.4+0.2\cdot \text{GPA}+0.5\cdot \text{Experience})}}$] and it can capture more complex relationships between *x* and *y*."

*Figure: no*

## Slide 108 — Summary: A Deep Neural Network

Diagram: Input Layer with three circles (label "Input Layer (Independent Variables)", bold), two hidden layers of four circles each with ReLU hinge icons (label "Hidden Layers", bold), and a single output circle with a sigmoid S-curve icon and output arrow (label "Output Layer", bold). Each layer is fully connected to the next (dense dark-red edges: 3→4, 4→4, 4→1).

- This is a **feedforward** (or **vanilla**) neural network
- In general, the arrangement of neurons into layers, the activation functions, and the connections between layers are referred to as the network's **architecture**

*Figure: yes — 3-4-4-1 feedforward network with ReLU hidden units and sigmoid output*

## Slide 109 — Summary: Designing a DNN

Same diagram and boxed text as slide 94 (Input Layer with $x_1$ … $x_k$ → two input-layer circles; Hidden Layer 1 with three circles; Hidden Layer 2 with two circles; "●●●"; Output layer with one circle). Boxed text: "*User* chooses the **# of hidden layers**, **# units in each layer**, the **activation function(s)** for the hidden layers and for the output layer" (colours as slide 94: blue, olive, red).

*Figure: yes — generic deep neural network diagram*

## Slide 110 — MIT OpenCourseWare end page

Not lecture content: MIT OpenCourseWare's appended end page. It reads "MIT OpenCourseWare https://ocw.mit.edu", "15.773 Hands-on Deep Learning", "Spring 2024", "For information about citing these materials or our Terms of Use, visit: https://ocw.mit.edu/terms."

*Figure: no*
