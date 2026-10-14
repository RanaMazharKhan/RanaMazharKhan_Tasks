# Machine Learning - Classification (practice tasks)

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.datasets import load_wine, load_breast_cancer, make_classification
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, roc_curve, roc_auc_score,
                             precision_score, f1_score, recall_score)
from sklearn.inspection import permutation_importance
from category_encoders import BinaryEncoder

DATA = os.path.join(os.path.dirname(__file__), "data")

# ---
# Load the telco churn data, clean TotalCharges, label encode Churn, split 80/20.
# We split BEFORE encoding so the encoders only learn from train data (no leakage).
df = pd.read_csv(os.path.join(DATA, "Telco-Customer-Churn.csv"))
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Missing TotalCharges:", df["TotalCharges"].isna().sum())
df["Churn_encoded"] = LabelEncoder().fit_transform(df["Churn"])
y = df["Churn_encoded"]
X = df.drop(columns=["Churn", "Churn_encoded", "customerID"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape, " Test:", X_test.shape)

# ---
# The main encodings: one-hot, ordinal, frequency, target and binary.
enc = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
print("One-hot shape:", enc.fit_transform(X_train[["gender", "InternetService"]]).shape)

ord_enc = OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]])
print(pd.DataFrame({"Contract": X_train["Contract"],
                    "ordinal": ord_enc.fit_transform(X_train[["Contract"]]).ravel()}
                   ).drop_duplicates().sort_values("ordinal").to_string(index=False))

freq = X_train["PaymentMethod"].value_counts(normalize=True)
print("\nPaymentMethod frequencies:\n", freq.round(3))

train_y = X_train.copy()
train_y["Churn"] = y_train.values
target_means = train_y.groupby("PaymentMethod")["Churn"].mean()
print("\nPaymentMethod churn rates:\n", target_means.sort_values(ascending=False).round(3))

oh_cols = OneHotEncoder(drop="first", sparse_output=False).fit_transform(
    X_train[["PaymentMethod"]]).shape[1]
bin_cols = BinaryEncoder(cols=["PaymentMethod"]).fit_transform(
    X_train[["PaymentMethod"]]).shape[1]
print("\nOne-hot columns:", oh_cols, "| Binary encoded columns:", bin_cols)
# Binary encoding needs only 2 columns for 4 categories, saving width.

# ---
# One ColumnTransformer that cleans + encodes numeric and categorical columns.
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical_features = [c for c in X_train.columns if c not in numeric_features]

numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)])

print("Before:", X_train.shape, "| After:", preprocessor.fit_transform(X_train).shape)
# Columns grew because one-hot turns each category value into its own column.

# Leaving PaymentMethod out: remainder='drop' vs 'passthrough'.
used_cat = [c for c in categorical_features if c != "PaymentMethod"]
for rem in ["drop", "passthrough"]:
    ct = ColumnTransformer([("num", numeric_pipe, numeric_features),
                            ("cat", categorical_pipe, used_cat)], remainder=rem)
    print(f"remainder={rem}:", ct.fit_transform(X_train).shape)
# 'drop' throws the unused column away; 'passthrough' keeps it as raw text.

# ---
# Full pipeline (preprocess + logistic regression): fit, predict, evaluate.
log_reg_pipe = Pipeline([("preprocessor", preprocessor),
                         ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
log_reg_pipe.fit(X_train, y_train)
pred = log_reg_pipe.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, target_names=["No Churn", "Churn"]))

# What breaks without the imputer? The scaler chokes on NaN TotalCharges.
broken = ColumnTransformer([
    ("num", Pipeline([("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]),
     categorical_features)])
bad_pipe = Pipeline([("preprocessor", broken),
                     ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
try:
    bad_pipe.fit(X_train, y_train)
except ValueError as e:
    print("Without imputer ->", e)
# This is exactly the bug the SimpleImputer step in a pipeline prevents.

# ---
# Same pipeline, different classifiers: random forest vs gradient boosting.
for name, clf in [("RandomForest",
                   RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)),
                  ("GradientBoosting", GradientBoostingClassifier(random_state=42))]:
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    print(name, "test accuracy:", round(accuracy_score(y_test, pipe.predict(X_test)), 4))

# ---
# Grid search the regularization strength C of logistic regression (5-fold CV).
grid = GridSearchCV(log_reg_pipe,
                    {"classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
                    cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV accuracy:", round(grid.best_score_, 4))

# ---
# make_pipeline builds the same pipeline with auto-generated step names.
pipe_mp = make_pipeline(preprocessor, LogisticRegression(max_iter=1000, random_state=42))
pipe_mp.fit(X_train, y_train)
print("Pipeline steps:     ", list(log_reg_pipe.named_steps.keys()))
print("make_pipeline steps:", list(pipe_mp.named_steps.keys()))
print("Same predictions:", np.array_equal(log_reg_pipe.predict(X_test),
                                          pipe_mp.predict(X_test)))
# Pipeline keeps the names we gave; make_pipeline makes them from class names.

c_param = [k for k in pipe_mp.get_params() if k.endswith("__C")][0]
print("C lives under:", c_param)
grid2 = GridSearchCV(pipe_mp, {c_param: [0.01, 0.1, 1.0, 10.0]}, cv=5, n_jobs=-1)
grid2.fit(X_train, y_train)
print("Best params:", grid2.best_params_, "| Best CV accuracy:", round(grid2.best_score_, 4))

# ---
# Mini project: tune a random forest churn pipeline and report the final result.
rf_pipe = Pipeline([("preprocessor", preprocessor),
                    ("classifier", RandomForestClassifier(random_state=42))])
grid3 = GridSearchCV(rf_pipe,
                     {"classifier__n_estimators": [200, 400],
                      "classifier__max_depth": [8, 12, None]},
                     cv=5, n_jobs=-1)
grid3.fit(X_train, y_train)
best = grid3.best_estimator_
pred = best.predict(X_test)
print("Best params:", grid3.best_params_)
print("Test accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, target_names=["No Churn", "Churn"]))
joblib.dump(best, "/tmp/my_churn_pipeline.pkl")

# One-hot for the nominal columns (no fake ordering), median imputation +
# scaling for numerics. The pipeline remembers every step, applies the same
# preprocessing to the test set, and GridSearchCV tunes without leakage.

# ---
# Threshold tuning: catching 90% of churners instead of 80% costs FPR and precision.
df2 = pd.read_csv(os.path.join(DATA, "Telco-Customer-Churn.csv"))
df2["Churn"] = df2["Churn"].map({"Yes": 1, "No": 0})
df2["TotalCharges"] = pd.to_numeric(df2["TotalCharges"], errors="coerce")
df2["TotalCharges"] = df2["TotalCharges"].fillna(df2["TotalCharges"].median())
model_df = pd.get_dummies(df2.drop(columns=["customerID"]),
                          columns=df2.drop(columns=["customerID"]).select_dtypes(
                              include="object").columns,
                          drop_first=True)
Xb = model_df.drop(columns=["Churn"])
yb = model_df["Churn"]
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    Xb, yb, test_size=0.25, random_state=42, stratify=yb)
scaler_b = StandardScaler().fit(Xb_train)
Xb_train_s, Xb_test_s = scaler_b.transform(Xb_train), scaler_b.transform(Xb_test)

lr_b = LogisticRegression(max_iter=1000, random_state=42).fit(Xb_train_s, yb_train)
y_proba = lr_b.predict_proba(Xb_test_s)[:, 1]
fpr, tpr, thresholds = roc_curve(yb_test, y_proba)
for target in [0.80, 0.90]:
    idx = np.argmax(tpr >= target)
    prec = precision_score(yb_test, (y_proba >= thresholds[idx]).astype(int))
    print(f"TPR>={target}: threshold={thresholds[idx]:.3f}, "
          f"FPR={fpr[idx]:.3f}, precision={prec:.3f}")
# A lower threshold catches more churners but wrongly flags more loyal customers.

# ---
# ROC comparison of logistic regression, random forest and gradient boosting.
rf_b = RandomForestClassifier(n_estimators=300, max_depth=6,
                              random_state=42).fit(Xb_train, yb_train)
gb_b = GradientBoostingClassifier(random_state=42).fit(Xb_train, yb_train)

plt.figure(figsize=(6.5, 6.5))
for name, proba, color in [
        ("Logistic Regression", lr_b.predict_proba(Xb_test_s)[:, 1], "#1565c0"),
        ("Random Forest", rf_b.predict_proba(Xb_test)[:, 1], "#2e7d32"),
        ("Gradient Boosting", gb_b.predict_proba(Xb_test)[:, 1], "#ef6c00")]:
    fpr_c, tpr_c, _ = roc_curve(yb_test, proba)
    auc = roc_auc_score(yb_test, proba)
    print(f"{name}: AUC = {auc:.3f}")
    plt.plot(fpr_c, tpr_c, linewidth=2.5, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Model Comparison via ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# ---
# class_weight='balanced' makes logistic regression catch more churners.
for weight in [None, "balanced"]:
    pipe = Pipeline([("preprocessor", preprocessor),
                     ("classifier", LogisticRegression(max_iter=1000, random_state=42,
                                                       class_weight=weight))])
    pipe.fit(X_train, y_train)
    rep = classification_report(y_test, pipe.predict(X_test), output_dict=True)
    print(f"class_weight={weight}: churn recall = {rep['1']['recall']:.3f}, "
          f"churn precision = {rep['1']['precision']:.3f}")
# Balanced weights raise recall, usually at the cost of a little precision.

# ---
# Target encoding must learn its means from the TRAIN set only - never all data.
train_only = X_train.copy()
train_only["Churn"] = y_train.values
train_means = train_only.groupby("PaymentMethod")["Churn"].mean()

full = pd.concat([X_train, X_test]).copy()
full["Churn"] = pd.concat([y_train, y_test]).values
full_means = full.groupby("PaymentMethod")["Churn"].mean()

print("Train-only means:\n", train_means.round(4))
print("\nDifference if test labels sneak in:\n", (full_means - train_means).round(4))
# The means come from the target itself, so full-data means leak test answers
# into the training features and inflate test scores.

# ---
# Make a small fake churn dataset (2000 customers) and check the target balance.
def make_churn_data():
    rng = np.random.default_rng(42)
    n = 2000
    gender = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
    tenure = rng.integers(0, 73, n)
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.44, 0.21])
    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment = rng.choice(["Electronic check", "Mailed check",
                          "Bank transfer (automatic)", "Credit card (automatic)"], n)
    probs = np.zeros((n, 3))
    for i in range(n):
        t = tenure[i]
        p_month = np.clip(0.75 - t / 100, 0.08, 0.85)
        p_two = np.clip(t / 110, 0.05, 0.55)
        p_one = max(0.02, 1 - p_month - p_two)
        tot = p_month + p_one + p_two
        probs[i] = [p_month / tot, p_one / tot, p_two / tot]
    contract = np.array([rng.choice(["Month-to-month", "One year", "Two year"], p=cp)
                         for cp in probs])
    base = np.where(internet == "Fiber optic", 70, np.where(internet == "DSL", 45, 20))
    monthly = (base + rng.normal(10, 6, n).clip(0, 30)
               + rng.normal(0, 5, n)).clip(18, 120).round(2)
    total = (monthly * tenure + rng.normal(0, 20, n)).clip(0, None).round(2)
    z = (-1.6 - 0.045 * tenure + 0.020 * monthly
         + np.where(contract == "Month-to-month", 1.1,
                    np.where(contract == "One year", -0.1, -1.0))
         + np.where(senior == 1, 0.35, 0)
         + np.where(internet == "Fiber optic", 0.35, 0)
         + rng.normal(0, 0.5, n))
    churn = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "InternetService": internet, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
        "Churn": np.where(churn == 1, "Yes", "No")})


syn = make_churn_data()
print(syn["Churn"].value_counts())
print(f"Churn rate: {(syn['Churn'] == 'Yes').mean() * 100:.1f}%")

# ---
# One-hot encode the fake data, scale it, train logistic regression, evaluate.
Xs = pd.get_dummies(syn.drop(columns=["Churn"]), drop_first=True)
ys = (syn["Churn"] == "Yes").astype(int)
Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    Xs, ys, test_size=0.25, random_state=1, stratify=ys)

scaler_s = StandardScaler()
Xs_train_s = scaler_s.fit_transform(Xs_train)
Xs_test_s = scaler_s.transform(Xs_test)

lr_s = LogisticRegression(max_iter=2000).fit(Xs_train_s, ys_train)
pred_s = lr_s.predict(Xs_test_s)
print("Test accuracy:", round(accuracy_score(ys_test, pred_s), 4))
print(classification_report(ys_test, pred_s, target_names=["No Churn", "Churn"]))
ConfusionMatrixDisplay.from_predictions(ys_test, pred_s,
                                         display_labels=["No Churn", "Churn"])
plt.title("Confusion Matrix - Churn Model")
plt.show()

# ---
# Which features push customers toward churning, and which keep them?
coef = pd.Series(lr_s.coef_[0], index=Xs.columns).sort_values()
print(coef.round(3))
print("\nRaises churn most:", coef.index[-1], "=", round(coef.iloc[-1], 3))
print("Lowers churn most:", coef.index[0], "=", round(coef.iloc[0], 3))
# Positive coefficient -> higher feature value pushes toward churning.

# ---
# Multiclass logistic regression: predict Contract type (3 classes) not churn.
Xc = pd.get_dummies(syn.drop(columns=["Contract"]), drop_first=True)
yc = syn["Contract"]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.25, random_state=1, stratify=yc)
scaler_c = StandardScaler().fit(Xc_train)
Xc_train_s, Xc_test_s = scaler_c.transform(Xc_train), scaler_c.transform(Xc_test)

multi = LogisticRegression(max_iter=2000).fit(Xc_train_s, yc_train)
pred_c = multi.predict(Xc_test_s)
print("Test accuracy:", round(accuracy_score(yc_test, pred_c), 4))
print(classification_report(yc_test, pred_c))
ConfusionMatrixDisplay.from_predictions(yc_test, pred_c)
plt.title("Confusion Matrix - Contract Model")
plt.show()

# One customer's predicted probability for each contract type.
probs_c = multi.predict_proba(Xc_test_s[0:1])[0]
for cls, p in zip(multi.classes_, probs_c):
    print(f"P({cls}) = {p:.3f}")
print("Sum:", round(probs_c.sum(), 3))

# ---
# Feature engineering: add AvgChargePerMonth and see if accuracy improves.
def train_churn(frame):
    Xf = pd.get_dummies(frame.drop(columns=["Churn"]), drop_first=True)
    yf = (frame["Churn"] == "Yes").astype(int)
    Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(
        Xf, yf, test_size=0.25, random_state=1, stratify=yf)
    scaler_f = StandardScaler()
    model = LogisticRegression(max_iter=2000)
    model.fit(scaler_f.fit_transform(Xf_tr), yf_tr)
    return accuracy_score(yf_te, model.predict(scaler_f.transform(Xf_te)))


base_acc = train_churn(syn)
syn["AvgChargePerMonth"] = syn["TotalCharges"] / (syn["tenure"] + 1)
new_acc = train_churn(syn)
print("Accuracy before:", round(base_acc, 4))
print("Accuracy after:", round(new_acc, 4))
# Barely changes: the new feature is just a mix of two the model already sees.

# ---
# Bayes theorem: when "free" appears in normal mail too, it is weak spam evidence.
p_spam, p_free_spam = 0.30, 0.60
for p_free_ham in [0.05, 0.30]:
    p_free = p_free_spam * p_spam + p_free_ham * (1 - p_spam)
    print(f"P(free|ham) = {p_free_ham}: P(spam|free) = "
          f"{round(p_free_spam * p_spam / p_free, 3)}")

# ---
# Spam filter with MultinomialNB; smoothing tames words never seen in training.
messages = [
    ("win a free prize now", "spam"), ("free money click now", "spam"),
    ("claim your free gift card", "spam"), ("urgent you won cash prize", "spam"),
    ("limited offer buy now cheap", "spam"), ("free entry win cash now", "spam"),
    ("congratulations winner claim prize", "spam"), ("cheap loans click here now", "spam"),
    ("you won a free iphone click here", "spam"),          # my own spam
    ("double your money fast guaranteed", "spam"),         # my own spam
    ("act now limited lottery winner", "spam"),            # my own spam
    ("are we meeting for lunch today", "ham"), ("please send the report by friday", "ham"),
    ("can you call me after the class", "ham"), ("the meeting is moved to monday", "ham"),
    ("happy birthday see you at dinner", "ham"), ("i will send the notes tonight", "ham"),
    ("lunch at noon in the cafeteria", "ham"), ("thanks for the help with homework", "ham"),
    ("mom called about dinner on sunday", "ham"),          # my own ham
    ("the football match starts at nine", "ham"),          # my own ham
    ("remember to buy milk and bread", "ham")]             # my own ham
texts = [m[0] for m in messages]
labels = [m[1] for m in messages]

cv = CountVectorizer()
X_counts = cv.fit_transform(texts)

new_msgs = ["free cash prize now", "see you at lunch", "send me the report", "win a free gift"]
mnb = MultinomialNB(alpha=1.0).fit(X_counts, labels)
proba = mnb.predict_proba(cv.transform(new_msgs))
result = pd.DataFrame({"message": new_msgs, "prediction": mnb.predict(cv.transform(new_msgs)),
                       "P(ham)": proba[:, 0].round(3), "P(spam)": proba[:, 1].round(3)})
print(result.to_string(index=False))

# "cryptocurrency" was never seen in training: watch alpha control the confidence.
weird = ["buy cryptocurrency now"]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for alpha in [0.1, 1.0, 10.0]:
        p = MultinomialNB(alpha=alpha).fit(X_counts, labels).predict_proba(
            cv.transform(weird))[0]
        print(f"alpha = {alpha}: P(ham) = {p[0]:.3f}, P(spam) = {p[1]:.3f}")
# Tiny alpha stays spiky (near 0/1, over-confident); big alpha flattens toward
# uniform. Smoothing exists to fix the zero-probability problem.

# ---
# Naive Bayes counts the same evidence twice if features are duplicated.
cancer = load_breast_cancer(as_frame=True)
Xnb, ynb = cancer.data, cancer.target
Xnb_tr, Xnb_te, ynb_tr, ynb_te = train_test_split(
    Xnb, ynb, test_size=0.25, random_state=0, stratify=ynb)

rows = []
for k in [1, 3, 10, 30]:
    m = GaussianNB().fit(np.hstack([Xnb_tr.values] * k), ynb_tr)
    Xk_te = np.hstack([Xnb_te.values] * k)
    p = m.predict_proba(Xk_te)
    rows.append({"copies": k,
                 "accuracy": round(accuracy_score(ynb_te, m.predict(Xk_te)), 3),
                 "avg_confidence": round(p.max(axis=1).mean(), 3)})
print(pd.DataFrame(rows).to_string(index=False))
# Accuracy barely moves, but confidence climbs: copies add no new information.

# ---
# Pick the right Naive Bayes variant for the data: Gaussian, Multinomial, Bernoulli.
rng_nb = np.random.default_rng(0)

scores = np.vstack([rng_nb.normal(70, 10, (60, 3)), rng_nb.normal(55, 12, (60, 3))])
y_scores = np.array([1] * 60 + [0] * 60)
print("(a) exam scores -> GaussianNB, accuracy:",
      round(GaussianNB().fit(scores, y_scores).score(scores, y_scores), 3))

counts = rng_nb.poisson([[3, 1, 0, 2]] * 40 + [[0, 2, 4, 1]] * 40)
y_words = np.array([0] * 40 + [1] * 40)
print("(b) word counts  -> MultinomialNB, accuracy:",
      round(MultinomialNB().fit(counts, y_words).score(counts, y_words), 3))

clicks = (rng_nb.random((80, 50)) < 0.3).astype(int)
y_clicks = (clicks[:, :5].sum(axis=1) > 2).astype(int)
print("(c) button clicks -> BernoulliNB, accuracy:",
      round(BernoulliNB().fit(clicks, y_clicks).score(clicks, y_clicks), 3))

# ---
# Decision tree on wine: train it, see how leaf size controls overfitting, tune it.
Xw, yw = load_wine(return_X_y=True)
Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    Xw, yw, test_size=0.25, random_state=42, stratify=yw)

tree = DecisionTreeClassifier(random_state=42).fit(Xw_train, yw_train)
print("Train accuracy:", round(tree.score(Xw_train, yw_train), 3))
print("Test accuracy:", round(tree.score(Xw_test, yw_test), 3))

leaf_vals = list(range(1, 31))
tr_s, te_s = [], []
for leaf in leaf_vals:
    t = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=42)
    t.fit(Xw_train, yw_train)
    tr_s.append(t.score(Xw_train, yw_train))
    te_s.append(t.score(Xw_test, yw_test))

plt.figure(figsize=(8, 4.5))
plt.plot(leaf_vals, tr_s, "o-", label="Train")
plt.plot(leaf_vals, te_s, "s-", label="Test")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# Small leaves overfit (train ~1.0); bigger leaves close the train/test gap.

grid_t = GridSearchCV(DecisionTreeClassifier(random_state=42),
                      {"criterion": ["gini", "entropy"],
                       "max_depth": [2, 3, 4, 5, 7, 10, None],
                       "min_samples_split": [2, 5, 10],
                       "min_samples_leaf": [1, 2, 5]},
                      cv=5, n_jobs=-1)
grid_t.fit(Xw_train, yw_train)
print("Best params:", grid_t.best_params_)
print("Best CV score:", round(grid_t.best_score_, 4))
print("Test score:", round(grid_t.best_estimator_.score(Xw_test, yw_test), 4))

# ---
# Look inside a small tree: plot it and read off 3 rules it learned.
wine = load_wine()
tree3 = DecisionTreeClassifier(max_depth=3, random_state=42)
tree3.fit(Xw_train, yw_train)
print("Test accuracy:", round(tree3.score(Xw_test, yw_test), 3))

plt.figure(figsize=(14, 8))
plot_tree(tree3, feature_names=wine.feature_names,
          class_names=wine.target_names, filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree on Wine (max_depth=3)")
plt.show()
print(export_text(tree3, feature_names=list(wine.feature_names)))

# 3 rules my tree learned:
# 1. If color_intensity <= 3.82 and ash <= 3.00 and od280/od315 <= 3.73 -> class 1
# 2. If color_intensity > 3.82 and flavanoids <= 1.58 and alcalinity_of_ash <= 17.65 -> class 1
# 3. If color_intensity > 3.82 and flavanoids > 1.58 and proline > 724.50 -> class 0

# ---
# Imbalanced data: class_weight='balanced' helps the minority class, but check it.
Xi, yi = make_classification(n_samples=3000, weights=[0.95, 0.05],
                             n_informative=8, random_state=42)
Xi_train, Xi_test, yi_train, yi_test = train_test_split(
    Xi, yi, test_size=0.3, random_state=42, stratify=yi)

for weight in [None, "balanced"]:
    t = DecisionTreeClassifier(max_depth=5, class_weight=weight, random_state=42)
    t.fit(Xi_train, yi_train)
    rep = classification_report(yi_test, t.predict(Xi_test), output_dict=True)
    print(f"Tree class_weight={weight}: minority recall = {rep['1']['recall']:.3f}, "
          f"accuracy = {rep['accuracy']:.3f}")

for weight in [None, "balanced", "balanced_subsample"]:
    rf = RandomForestClassifier(n_estimators=200, class_weight=weight,
                                random_state=42, n_jobs=-1)
    rf.fit(Xi_train, yi_train)
    p = rf.predict(Xi_test)
    print(f"Forest class_weight={weight}: F1 = {f1_score(yi_test, p):.3f}, "
          f"minority recall = {recall_score(yi_test, p):.3f}")
# 'balanced' lifts minority recall; 'balanced_subsample' can hurt - compare on
# F1/recall instead of assuming a setting helps.

# ---
# Random forest vs single tree (5-fold CV), and how OOB score grows with trees.
for name, model in [("Decision Tree", DecisionTreeClassifier(random_state=42)),
                    ("Random Forest", RandomForestClassifier(n_estimators=300,
                                                             random_state=42, n_jobs=-1))]:
    scores = cross_val_score(model, Xw, yw, cv=5)
    print(f"{name}: mean = {scores.mean():.3f}, std = {scores.std():.3f}")
# The forest scores higher and varies less - averaging smooths single-tree mistakes.

n_vals = list(range(10, 301, 20))
oob = []
for n in n_vals:
    rf = RandomForestClassifier(n_estimators=n, oob_score=True,
                                random_state=42, n_jobs=-1)
    rf.fit(Xw, yw)
    oob.append(rf.oob_score_)

plt.figure(figsize=(8, 4.5))
plt.plot(n_vals, oob, "o-")
plt.xlabel("n_estimators")
plt.ylabel("OOB score")
plt.title("OOB Score vs Number of Trees")
plt.grid(alpha=0.3)
plt.show()
# OOB rises fast then flattens: more trees stop helping but do not hurt.

# ---
# Tune the forest: 50 random tries, then a fine grid around the winner.
base_rf = RandomForestClassifier(random_state=42, n_jobs=-1).fit(Xw_train, yw_train)
base_acc = base_rf.score(Xw_test, yw_test)
print("Baseline test accuracy:", round(base_acc, 4))

rand_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {"n_estimators": randint(100, 400),
     "max_depth": [3, 5, 8, 12, None],
     "min_samples_leaf": randint(1, 10),
     "max_features": ["sqrt", "log2", 0.5, 1.0]},
    n_iter=50, cv=3, random_state=42, n_jobs=-1)
rand_search.fit(Xw_train, yw_train)
print("Random search best:", rand_search.best_params_)

best_p = rand_search.best_params_
if best_p["max_depth"] is None:
    depths = [None]
else:
    depths = [d for d in [best_p["max_depth"] - 2, best_p["max_depth"],
                          best_p["max_depth"] + 2] if d > 0]
grid_f = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {"n_estimators": [best_p["n_estimators"] - 50, best_p["n_estimators"],
                      best_p["n_estimators"] + 50],
     "max_depth": depths,
     "min_samples_leaf": [max(1, best_p["min_samples_leaf"] - 1),
                          best_p["min_samples_leaf"],
                          best_p["min_samples_leaf"] + 1]},
    cv=3, n_jobs=-1)
grid_f.fit(Xw_train, yw_train)
final_acc = grid_f.best_estimator_.score(Xw_test, yw_test)
print("Fine grid best:", grid_f.best_params_)
print("Final test accuracy:", round(final_acc, 4))
print("Improvement over baseline:", round(final_acc - base_acc, 4))

# ---
# Permutation importance: which wine features actually matter?
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    wine_df, yw, test_size=0.25, random_state=42, stratify=yw)

rf_imp = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf_imp.fit(Xf_train, yf_train)
print("Full accuracy:", round(rf_imp.score(Xf_test, yf_test), 4))

result = permutation_importance(rf_imp, Xf_test, yf_test, n_repeats=10, random_state=42)
importance = pd.Series(result.importances_mean,
                       index=wine_df.columns).sort_values(ascending=False)
print("\nTop 5 features:")
print(importance.head(5).round(3))

drop_cols = importance.tail(10).index.tolist()
rf_small = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf_small.fit(Xf_train.drop(columns=drop_cols), yf_train)
print("\nAccuracy after dropping 10 least important:",
      round(rf_small.score(Xf_test.drop(columns=drop_cols), yf_test), 4))
# Dropping the weakest features barely matters - a few strong ones carry the signal.
