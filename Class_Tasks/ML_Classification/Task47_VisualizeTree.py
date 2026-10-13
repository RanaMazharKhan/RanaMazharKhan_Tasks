# Homework 4 - Visualize the final tree and write 3 rules the tree learned (in the comments below).
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
print("Test accuracy:", round(tree.score(X_test, y_test), 3))

plt.figure(figsize=(14, 8))
plot_tree(tree, feature_names=wine.feature_names,
          class_names=wine.target_names, filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree on Wine (max_depth=3)")
plt.show()

print(export_text(tree, feature_names=list(wine.feature_names)))

# 3 rules my tree learned:
# 1. If color_intensity <= 3.82 and ash <= 3.00 and od280/od315 <= 3.73 -> class 1
# 2. If color_intensity > 3.82 and flavanoids <= 1.58 and alcalinity_of_ash <= 17.65 -> class 1
# 3. If color_intensity > 3.82 and flavanoids > 1.58 and proline > 724.50 -> class 0
