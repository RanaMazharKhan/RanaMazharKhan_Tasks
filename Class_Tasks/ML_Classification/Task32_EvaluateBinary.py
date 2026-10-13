# Exercise 4 - Print the classification report and plot the confusion matrix.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

rng = np.random.default_rng(42)
n = 2000
gender = rng.choice(["Male", "Female"], n)
senior = rng.choice([0, 1], n, p=[0.84, 0.16])
partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
dependents = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
tenure = rng.integers(0, 73, n)
phone_service = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.44, 0.21])
paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
payment = rng.choice(["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"], n)
contract_probs = np.zeros((n, 3))
for i in range(n):
    t = tenure[i]
    p_month = np.clip(0.75 - t / 100, 0.08, 0.85)
    p_two = np.clip(t / 110, 0.05, 0.55)
    p_one = max(0.02, 1 - p_month - p_two)
    tot = p_month + p_one + p_two
    contract_probs[i] = [p_month / tot, p_one / tot, p_two / tot]
contract = np.array([rng.choice(["Month-to-month", "One year", "Two year"], p=cp)
                     for cp in contract_probs])
base_charge = np.where(internet_service == "Fiber optic", 70,
                       np.where(internet_service == "DSL", 45, 20))
addon_charge = rng.normal(10, 6, n).clip(0, 30)
monthly_charges = (base_charge + addon_charge + rng.normal(0, 5, n)).clip(18, 120).round(2)
total_charges = (monthly_charges * tenure + rng.normal(0, 20, n)).clip(0, None).round(2)
z = (-1.6 - 0.045 * tenure + 0.020 * monthly_charges
     + np.where(contract == "Month-to-month", 1.1, np.where(contract == "One year", -0.1, -1.0))
     + np.where(senior == 1, 0.35, 0)
     + np.where(internet_service == "Fiber optic", 0.35, 0)
     + rng.normal(0, 0.5, n))
churn = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
df = pd.DataFrame({
    "customerID": [f"C{10000+i}" for i in range(n)],
    "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone_service, "InternetService": internet_service,
    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    "Churn": np.where(churn == 1, "Yes", "No"),
})

X = pd.get_dummies(df.drop(columns=["customerID", "Churn"]), drop_first=True)
y = (df["Churn"] == "Yes").astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)

print(classification_report(y_test, pred, target_names=["No Churn", "Churn"]))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["No Churn", "Churn"])
plt.title("Confusion Matrix - Churn Model")
plt.show()
