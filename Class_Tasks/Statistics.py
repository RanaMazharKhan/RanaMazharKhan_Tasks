# Statistics - descriptive stats, sampling, inference and distributions.
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# Mean, median, mode, range, variance and std dev of one small dataset.
values = [12, 15, 12, 18, 20, 12, 25]
print("Mean:", np.mean(values))
print("Median:", np.median(values))
print("Mode:", max(set(values), key=values.count))
print("Range:", max(values) - min(values))
print("Sample variance:", np.var(values, ddof=1))
print("Sample std dev:", np.std(values, ddof=1))
# ---
# Find the 25th, 50th and 75th percentiles of the same values.
for p in [25, 50, 75]:
    print(p, "th percentile:", np.percentile(values, p))
# ---
# Draw 1000 samples from a right-skewed (exponential) distribution,
# check mean vs median and skewness, and plot them.
sample = np.random.default_rng(2).exponential(scale=2.0, size=1000)
print("Mean:", sample.mean())
print("Median:", np.median(sample))
print("Skewness:", stats.skew(sample))

plt.hist(sample, bins=40)
plt.axvline(sample.mean(), color="red", label="Mean")
plt.axvline(np.median(sample), color="orange", label="Median")
plt.legend()
plt.title("Right-skewed sample")
plt.show()
# ---
# Find the correlation between X and Y and draw a scatter plot.
x = np.array([10, 20, 30, 40, 50])
y = np.array([12, 18, 33, 38, 55])
corr = np.corrcoef(x, y)[0, 1]
print("Correlation:", corr)

plt.scatter(x, y)
plt.title("Correlation = " + str(round(corr, 2)))
plt.show()
# ---
# Turn heights into z-scores and check the mean is ~0 and std is ~1.
heights = np.array([150, 160, 165, 170, 180, 190])
z = (heights - heights.mean()) / heights.std(ddof=1)
print("Z-scores:", z)
print("Mean of z:", z.mean())      # about 0
print("Std of z:", z.std(ddof=1))  # about 1
# ---
# Standardize two features with very different scales and check both
# end up with the same std dev (1).
a = np.random.default_rng(3).normal(50, 5, 200)
b = np.random.default_rng(3).normal(500, 300, 200)

za = (a - a.mean()) / a.std(ddof=1)
zb = (b - b.mean()) / b.std(ddof=1)
print("Before:", a.std(ddof=1), b.std(ddof=1))
print("After:", za.std(ddof=1), zb.std(ddof=1))
# ---
# Flag price outliers in a small products table with the IQR rule.
products = pd.DataFrame({
    "name": ["Chair", "Desk", "Lamp", "Monitor", "Laptop", "Pen"],
    "price": [45, 120, 25, 220, 1200, 2]
})
q1 = products["price"].quantile(0.25)
q3 = products["price"].quantile(0.75)
iqr = q3 - q1
products["outlier"] = (products["price"] < q1 - 1.5 * iqr) | (products["price"] > q3 + 1.5 * iqr)
print(products)
# ---
# Take a sample of 30 from a population of 50000 and compare the
# sample mean with the true population mean.
rng = np.random.default_rng(1)
population = rng.normal(loc=60, scale=12, size=50000)
sample = rng.choice(population, size=30)
print("Population mean:", population.mean())
print("Sample mean:", sample.mean())
# ---
# The empirical rule: share of a Normal distribution within +-1 std dev.
inside = stats.norm.cdf(1) - stats.norm.cdf(-1)
print("Within 1 std dev:", inside)  # empirical rule says about 0.68
# ---
# Central Limit Theorem: 3000 sample means from an exponential distribution,
# with sample size 2 and with 100 - watch the shape become normal.
rng = np.random.default_rng(5)
means_2 = [rng.exponential(scale=3.0, size=2).mean() for _ in range(3000)]
means_100 = [rng.exponential(scale=3.0, size=100).mean() for _ in range(3000)]

plt.subplot(1, 2, 1)
plt.hist(means_2, bins=40)
plt.title("Sample size 2")

plt.subplot(1, 2, 2)
plt.hist(means_100, bins=40)
plt.title("Sample size 100")

plt.show()
# Bigger samples give a bell-shaped histogram: this is the CLT.
# ---
# Standard error by hand, then a 90% confidence interval for a sample
# of 35 from normal(80, 15).
se = 8 / math.sqrt(25)
print("Standard error (n=25, s=8):", se)

rng = np.random.default_rng(7)
sample = rng.normal(loc=80, scale=15, size=35)
low, high = stats.t.interval(0.90, len(sample) - 1,
                             loc=sample.mean(),
                             scale=stats.sem(sample))
print("90% CI:", low, "to", high)
print("Width:", high - low)
# ---
# Two-sample t-test: is group 2's mean different from group 1's?
# The CTR example applies the same test to two algorithms.
rng = np.random.default_rng(9)
group1 = rng.normal(60, 10, 40)
group2 = rng.normal(65, 10, 40)
t_stat, p_value = stats.ttest_ind(group1, group2)
print("p-value:", p_value)
if p_value < 0.05:
    print("Reject H0: the means are significantly different.")
else:
    print("Fail to reject H0: no significant difference.")

rng = np.random.default_rng(11)
algo_a = rng.normal(0.045, 0.01, 200)
algo_b = rng.normal(0.052, 0.01, 200)
t_stat, p_value = stats.ttest_ind(algo_a, algo_b)
print("CTR p-value:", p_value)
if p_value < 0.05:
    print("Significant: algorithm B has a different CTR.")
else:
    print("Not significant.")
# Significance alone is not enough to switch in production: the actual gain
# may be too small to be worth the cost and risk of changing.
# ---
# Chi-square test: is device type associated with purchase completion?
table = pd.DataFrame({"Yes": [80, 55, 15], "No": [120, 45, 35]},
                     index=["Mobile", "Desktop", "Tablet"])
print(table)
chi2, p_value, dof, expected = stats.chi2_contingency(table)
print("p-value:", p_value)
if p_value < 0.05:
    print("Significant: device type is associated with purchase completion.")
else:
    print("Not significant: no association found.")
# ---
# Three common distributions: expected tosses until the first Head
# (geometric), Beta(2,2) vs Beta(10,10), and Poisson convergence.
print("Expected tosses:", stats.geom(0.5).mean())

x = np.linspace(0, 1, 300)
plt.plot(x, stats.beta(2, 2).pdf(x), label="Beta(2,2)")
plt.plot(x, stats.beta(10, 10).pdf(x), label="Beta(10,10)")
plt.legend()
plt.title("Beta(2,2) vs Beta(10,10)")
plt.show()
# Beta(10,10) is more concentrated around 0.5.

dist = stats.poisson(1)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, n in zip(axes, [50, 1000, 50000]):
    samples = dist.rvs(size=n, random_state=42)
    k = np.arange(samples.min(), samples.max() + 1)
    ax.hist(samples, bins=np.arange(k.min() - 0.5, k.max() + 1.5),
            density=True, alpha=0.6, label="Empirical")
    ax.plot(k, dist.pmf(k), "o-", markersize=3, label="Theoretical")
    ax.set_title("n = " + str(n))
    ax.legend(fontsize=8)
plt.suptitle("Poisson(lambda=1)")
plt.show()
print("With 50000 samples the empirical bars match the theoretical PMF.")
# ---
# Mini-project: make a 2000-customer dataset and analyze the distributions
# of churned (Bernoulli), monthly_charges (Gamma) and complaints (Poisson).
rng = np.random.default_rng(123)
n = 2000
df = pd.DataFrame({
    "customer_id": np.arange(1, n + 1),
    "churned": rng.binomial(1, 0.20, n),
    "monthly_charges": (rng.gamma(4.0, 18.0, n) + 15).round(2),
    "complaints": rng.poisson(1.5, n)
})

df["churned"].value_counts().plot(kind="bar", title="Churned")
plt.show()
plt.hist(df["monthly_charges"], bins=40)
plt.title("Monthly charges")
plt.show()
df["complaints"].value_counts().sort_index().plot(kind="bar", title="Complaints")
plt.show()

print(df[["churned", "monthly_charges", "complaints"]].agg(["mean", "var", "std"]).T)

p = df["churned"].mean()
print("P(churn) =", round(p, 4))
lam = df["complaints"].mean()
print("P(3 complaints) =", round(stats.poisson(lam).pmf(3), 4))

emp = df["complaints"].value_counts(normalize=True).sort_index()
k = np.arange(emp.index.min(), emp.index.max() + 1)
emp = emp.reindex(k, fill_value=0)
plt.bar(k, emp, alpha=0.6, label="Empirical")
plt.plot(k, stats.poisson(lam).pmf(k), "o-", label="Theoretical Poisson")
plt.legend()
plt.title("Complaints: empirical vs theoretical")
plt.show()
# Bernoulli fits churn, Poisson fits complaints, and Gamma fits the
# right-skewed monthly charges.
