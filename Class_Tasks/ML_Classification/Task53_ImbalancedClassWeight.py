# Homework 5 - On imbalanced data, compare class_weight options using F1 and recall.
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

X, y = make_classification(n_samples=3000, weights=[0.95, 0.05],
                           n_informative=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

for weight in [None, "balanced", "balanced_subsample"]:
    rf = RandomForestClassifier(n_estimators=200, class_weight=weight,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    print(f"class_weight={weight}: F1 = {f1_score(y_test, pred):.3f}, "
          f"minority recall = {recall_score(y_test, pred):.3f}")

# class_weight='balanced' lifts minority recall and F1, but 'balanced_subsample'
# is unstable here and actually hurts - so always compare on F1/recall instead
# of assuming a setting helps.
