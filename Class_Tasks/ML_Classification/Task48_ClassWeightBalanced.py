# Homework 5 - Try class_weight='balanced' on imbalanced data, compare minority recall.
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=2000, weights=[0.95, 0.05],
                           n_informative=5, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)

for weight in [None, "balanced"]:
    tree = DecisionTreeClassifier(max_depth=5, class_weight=weight, random_state=0)
    tree.fit(X_train, y_train)
    report = classification_report(y_test, tree.predict(X_test), output_dict=True)
    print(f"class_weight={weight}: minority recall = {report['1']['recall']:.3f}, "
          f"accuracy = {report['accuracy']:.3f}")

# 'balanced' gives the rare class more importance, so its recall goes up -
# the model stops ignoring the 5% minority just to keep accuracy high.
