# Task: mini-project - make a 2000-customer dataset and analyze the
# distributions of churned, monthly_charges and complaints.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# make the data
rng = np.random.default_rng(123)
n = 2000
df = pd.DataFrame({
    "customer_id": np.arange(1, n + 1),
    "churned": rng.binomial(1, 0.20, n),
    "monthly_charges": (rng.gamma(4.0, 18.0, n) + 15).round(2),
    "complaints": rng.poisson(1.5, n)
})
print(df.info())

# variable types: churned = binary, monthly_charges = continuous, complaints = count

# plot the three distributions
df["churned"].value_counts().plot(kind="bar", title="Churned")
plt.show()
plt.hist(df["monthly_charges"], bins=40)
plt.title("Monthly charges")
plt.show()
df["complaints"].value_counts().sort_index().plot(kind="bar", title="Complaints")
plt.show()

# descriptive stats
print(df[["churned", "monthly_charges", "complaints"]].agg(["mean", "var", "std"]).T)

# proposed distributions: churned -> Bernoulli, monthly_charges -> Gamma, complaints -> Poisson

# probabilities from the fitted distributions
p = df["churned"].mean()
print("P(churn) =", round(p, 4))
lam = df["complaints"].mean()
print("P(3 complaints) =", round(stats.poisson(lam).pmf(3), 4))

# compare theoretical vs empirical for complaints
emp = df["complaints"].value_counts(normalize=True).sort_index()
k = np.arange(emp.index.min(), emp.index.max() + 1)
emp = emp.reindex(k, fill_value=0)
plt.bar(k, emp, alpha=0.6, label="Empirical")
plt.plot(k, stats.poisson(lam).pmf(k), "o-", label="Theoretical Poisson")
plt.legend()
plt.title("Complaints: empirical vs theoretical")
plt.show()

# conclusion: Bernoulli fits churn, Poisson fits complaints, and Gamma fits
# the right-skewed monthly charges.
