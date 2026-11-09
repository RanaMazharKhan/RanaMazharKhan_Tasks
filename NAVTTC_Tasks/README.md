# NAVTTC Tasks — AI (Machine Learning & Deep Learning), 3 Months

Prepared from: `AI_MACHINE_LEARNING_DEAP_LEARNING_3_Months__NAVTTC.pdf` — **Annexure-I (List of Tasks)**
Scope of this delivery: **Weeks 1–7, Tasks 1–64** (all tasks assigned up to Week 7, exactly as listed in the PDF). Weeks 1–5 were delivered first; Weeks 6–7 were added later in the same format.

Every task file is written the way a student would write it in class:

- One or two short **comment lines** at the top stating the task/problem — nothing else
- Plain, simple **code** below that solves it — no banner comments, no headings

The reference links from the PDF for each task are listed in the tables below (the practice was done from those links; the code itself is written fresh).

> All code in this folder is original, written on the topics of the cited tutorials.

## How to run
```bash
python3 Week_01/Task_15_For_Loops.py
bash Week_01/Task_02_Linux_Commands.sh
```
Most files need only Python. NumPy/Pandas/Statistics/ML files need: `pip install numpy pandas scipy` (ML files run with NumPy only — scikit-learn versions are included as commented code).

---

## Week 01 — Linux, Python Fundamentals
Topics in PDF: Debian/Linux basics, software installation, values/expressions, data structures, conditional execution, loops.

| Task | Title | PDF link |
|---|---|---|
| 1 | Installation (Anaconda, PyTorch, TensorFlow, VSCode, PyCharm) | — (.md file) |
| 2 | Linux Commands (pwd, cd, ls, cat, mkdir, cp, mv, chmod, ps…) | — (.sh file) |
| 3 | Values, Expressions and Statements | Week-1 schedule topic* |
| 4 | Numbers, Booleans and Strings | Week-1 schedule topic* |
| 5 | Operators, Variables and Keywords | Week-1 schedule topic* |
| 6 | String Operations | Week-1 schedule topic* |
| 7 | Input, Type Casting and Comments | Week-1 schedule topic* |
| 8 | Lists — Introduction | Week-1 schedule topic* |
| 9 | Tuples — Introduction | Week-1 schedule topic* |
| 10 | Dictionaries — Introduction | Week-1 schedule topic* |
| 11 | Sets — Introduction | Week-1 schedule topic* |
| 12 | Conditional Execution (if/elif/else) | Week-1 schedule topic* |
| 13 | Break, Continue and Pass | Week-1 schedule topic* |
| 14 | Nested Conditionals, Ternary, Loops Intro, List Comprehension, Iterators | Week-1 schedule topic* |
| 15 | For Loops | GeeksforGeeks |
| 16 | While Loops | GeeksforGeeks |
| 17 | Break Statement | GeeksforGeeks |
| 18 | Continue Statement | GeeksforGeeks |
| 19 | Looping Techniques (enumerate, zip, items, sorted, reversed) | GeeksforGeeks |
| 21 | List Data Type | Programiz |
| 22 | Tuple Data Type | Programiz |
| 23 | String Data Type | Programiz |
| 24 | Set Data Type | Programiz |
| 25 | Dictionary Data Type | Programiz |

\* Tasks 3–14 have **no description/link in the PDF's Annexure-I table** (the Description column is blank for them in the PDF). They were therefore built directly from the Week-1 Learning Units / Day-wise topics in the PDF schedule (Day 3: values, expressions, numbers, strings, operators, input, type casting • Day 4: lists, tuples, dictionaries, sets • Day 5: conditionals, break/continue/pass, nested conditionals, ternary, loops, list comprehension, iterators).

## Week 02 — Functions, Exceptions, Statistics
Topics in PDF: functions & scope, lambda/map/filter, file handling, exception handling, OOP, measures of central tendency & dispersion.

| Task | Title | PDF link |
|---|---|---|
| 20 | User Defined Functions (+ lambda, map/filter, inner functions) | GeeksforGeeks |
| 26 | Exception Handling | Programiz |
| 27 | User Defined Exceptions | Programiz |
| 49 | Mean, Median and Mode | W3Schools |
| 50 | Standard Deviation and Variance | W3Schools |
| Bonus | OOP Basics (class, inheritance, polymorphism) | Week-2 schedule topic |
| Bonus | File Handling | Week-2 schedule topic |

> Why are Week-2 task numbers not in order? Because that's the PDF's own numbering: Annexure-I assigns Task 20 and 26–27 to Week 2, and the Statistics tasks 49–50 to Week 2 as well (see the Week column in the PDF table).

## Week 03 — NumPy
PDF lists NumPy tasks for "Weeks 3, 4" together. They are placed in Week 03 here and Pandas in Week 04, matching the Week 3/4 schedule focus in the PDF (Week 3 = NumPy, Week 4 = Pandas/Seaborn).

| Task | Title | PDF link |
|---|---|---|
| 28 | Creating Arrays | W3Schools |
| 29 | Array Indexing | W3Schools |
| 30 | Array Slicing | W3Schools |
| 31 | Data Types (PDF table title says "Slicing" again, but its link is the Data Types page — file follows the link) | W3Schools |
| 32 | Copy vs View | W3Schools |
| 33 | Array Shape | W3Schools |
| 34 | Array Reshaping | W3Schools |
| 35 | Array Iterating | W3Schools |
| 36 | Array Joining | W3Schools |
| 37 | Array Splitting | W3Schools |
| 38 | Array Searching | W3Schools |
| 39 | Array Sorting | W3Schools |
| 40 | Random Numbers | W3Schools |

## Week 04 — Pandas

| Task | Title | PDF link |
|---|---|---|
| 41 | Pandas Basics | W3Schools |
| 42 | Pandas Getting Started / Installation | W3Schools |
| 43 | Pandas Series | W3Schools |
| 44 | Pandas DataFrames | W3Schools |
| 45 | Pandas Read CSV | W3Schools |
| 46 | Pandas Analyzing Data | W3Schools |
| 47 | Pandas Cleaning Data | W3Schools |
| 48 | Pandas Correlations | W3Schools |

Task 45 is self-contained: it reads CSV text built inside the file through the same `pd.read_csv()` function used for real `.csv` files (path version is shown in a comment).

## Week 05 — Machine Learning I
Topics in PDF: Multivariate Linear Regression, Polynomial Regression, Logistic Regression (binary + multiclass), code practice.

| Task | Title | PDF link |
|---|---|---|
| 51 | Linear Regression | StackAbuse |
| 52 | Logistic Regression | Towards Data Science + DataCamp (both links are in the PDF; both are cited in the file) |

Tasks 51–52 run with **NumPy only** (gradient descent / closed-form, as taught in the Week 4–5 syllabus in the PDF), so they work even without scikit-learn. The scikit-learn implementations in the style of the linked tutorials are included inside the files as commented code with install instructions.

---

## Week 06 — Machine Learning II

| Task | Title | PDF link |
|---|---|---|
| 53 | Decision Tree | DataCamp |
| 54 | SVM | StackAbuse |
| 55 | Time Series Analysis | DataQuest |

Tasks 53–54 run with **NumPy only** (tree / SVM built from scratch: gini splits, hinge-loss gradient descent), so they work even without scikit-learn. The scikit-learn implementations in the style of the linked tutorials are included inside the files as commented code. Task 55 uses pandas (resample + rolling average) on a date-indexed sales series built inside the file.

## Week 07 — Neural Networks / Deep Learning / CNN

| Task | Title | PDF link |
|---|---|---|
| 56 | Demonstration of Neural Networks | Analytics Vidhya |
| 57 | MLP | Machine Learning Mastery |
| 58 | Feed Forward Neural Networks | Built In |
| 59 | Neural Network (Loss Functions) | Analytics Vidhya |
| 60 | Linguistics using Machine Learning | Medium (Towards AI) |
| 61 | Text Processing | PythonSpot (NLTK) |
| 62 | Text Analysis | Medium (Towards AI) |
| 63 | Demonstrate Convolution Neural Network | Towards Data Science |
| 64 | CNN on CIFAR-10 Dataset | Analytics Vidhya |

Tasks 56–58 run with **NumPy only** (networks built from scratch: XOR backpropagation, an MLP classifier, one forward pass). Task 59 implements the 7 loss functions of the linked guide in NumPy. Tasks 60–62 are pure-Python text work (tokenizing, stop words, simple stemming, word frequency, sentiment by word lists); the NLTK version of Task 61 is included as commented code. Task 63 demonstrates a CNN's convolution + relu + pooling in NumPy; Task 64 shows the CIFAR-10 data-preparation steps in NumPy with the full Keras training code (in the style of the linked tutorial) included as commented code.

---

## Remaining course (not in this delivery)
Task 65 (Weeks 10–12: Microsoft Azure video lectures) is listed in Annexure-I and can be built in the same format next.
