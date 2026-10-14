# Class Tasks — Practice Tasks from Course Notebooks

Solved practice tasks from the course's Jupyter notebooks, one file per topic.
Each file runs top to bottom on its own and is written the way a student would
write it in class: a short comment stating each task, then plain simple code.

## Files

| File | Topic | Source notebooks |
|---|---|---|
| `Python_Basics.py` | Variables, types, strings, lists, f-strings | Day 1–2 Python notebooks |
| `OOP.py` | Classes, inheritance, polymorphism, encapsulation | Day 3 OOP notebooks |
| `NumPy.py` | Arrays, slicing, filtering, reshape, vectorization | Day 4 NumPy notebooks |
| `Statistics.py` | Descriptive stats, z-scores, t-tests, chi-square, distributions, CLT | Day 6–8 statistics notebooks |
| `Data_Cleaning.py` | Full cleaning + EDA pipeline on the Zameen.com dataset | Day 15 data cleansing notebooks |
| `ML_Regression.py` | ML workflow, KNN, bias-variance, grid/random search | Day 9, 11, 12 ML notebooks |
| `ML_Classification.py` | Encoding pipelines, metrics, logistic regression, Naive Bayes, trees, forests | Day 13, 14, 17 + Oct 2026 notebooks |

`data/` holds the CSVs used by the data/ML files:
`zameenData.csv`, `house_prices.csv`, `Telco-Customer-Churn.csv`.

## How to run

```bash
python3 Python_Basics.py
```

Needs: `pip install numpy pandas matplotlib seaborn scikit-learn scipy`
