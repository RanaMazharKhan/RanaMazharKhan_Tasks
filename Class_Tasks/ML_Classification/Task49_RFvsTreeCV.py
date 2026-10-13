# Homework 1 - Compare Decision Tree vs Random Forest on wine using 5-fold CV.
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X, y = load_wine(return_X_y=True)

tree = DecisionTreeClassifier(random_state=42)
forest = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

for name, model in [("Decision Tree", tree), ("Random Forest", forest)]:
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: mean = {scores.mean():.3f}, std = {scores.std():.3f}")

# The forest scores higher and its scores vary less across folds - averaging
# many trees smooths out the mistakes of single trees.
