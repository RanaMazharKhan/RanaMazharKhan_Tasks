# Class Tasks — Practice Tasks from Course Notebooks

Solved practice tasks from the course's Jupyter notebooks (Google Drive: "AIML-NIAI Course-Students").
Every practice task found at the end of each `.ipynb` file is solved here as its own file.

## Style
Each file is written the way a student would write it in class:
- one or two short **comment lines** at the top stating the task — nothing else
- plain, simple **code** below that solves it

## Folders (topics)

| Folder | Notebooks covered | Files |
|---|---|---|
| `Python_Basics/` | Day 1 code notebook; Day 2 Python part 1, hands-on notebook, homework notebook | 20 |
| `OOP/` | Day 3 OOP concepts + OOP practice exercises | 13 |
| `NumPy/` | Day 4 NumPy practice exercises | 13 |
| `Statistics/` | Day 6 descriptive + inferential statistics; Day 8 probability distributions | 20 |
| `Data_Cleaning/` | Day 15 data cleansing practice part 1 + part 2 (Zameen.com data) | 92 |
| `ML_Regression/` | Day 9 ML practical lecture; Day 11 KNN; Day 12 bias-variance + grid/random search | 27 |
| `ML_Classification/` | Day 13 encoding pipelines + classification metrics; Day 14 logistic regression; Day 17 Naive Bayes; October 2026 decision trees & random forests | 53 |
| `data/` | `zameenData.csv`, `house_prices.csv`, `Telco-Customer-Churn.csv` | — |

**Total: 238 solved task files.** File names are `Task1_<Topic>.py`, `Task2_<Topic>.py`, … numbered in order inside each folder.

## Notebooks with no practice tasks (nothing to solve)
- Day 4 `NumPy_Concepts_and_Code.ipynb` — ends with a summary only
- Day 3 `OOP_Concepts_and_Code.ipynb` — ends with a summary only
- Day 3 `Core_Python_for_ML_Part_2.ipynb` — ends with a concept summary only
- Day 5 (pandas) — the Drive folder is empty
- Day 7 hypothesis-testing notebook — ends with a summary flowchart + reference card
- Day 10 — only `.docx` notes and Google Docs, no notebooks
- Day 11 `sgd_gradient_descent.ipynb` — ends with a cheat sheet
- Day 14 `01_Logistic_Regression_Practical.ipynb` — ends with a summary table
- Day 16 — only a project-proposals doc
- Day 17 `dataCleansing1/2.ipynb` — byte-identical copies of the Day 15 files
- Day 18 SVM guide and the `ensemble_limitations_practical.ipynb` — end with summaries, no explicit tasks

## How to run
```bash
python3 Python_Basics/Task1_AI_Tool_Homework.py
```
Needs: `pip install numpy pandas matplotlib seaborn scikit-learn scipy`
(Data_Cleaning Task92 also uses `category_encoders`: `pip install category_encoders`.)
Data files live in `data/` and are loaded with a path relative to each file, so everything runs from inside this folder.
