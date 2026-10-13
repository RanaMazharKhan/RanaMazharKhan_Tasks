# Task: repeat the Poisson convergence check with lambda=1 and see if the
# empirical distribution still converges to the theoretical PMF.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
print("Yes, with 50000 samples the empirical bars match the theoretical PMF.")
