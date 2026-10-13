# Task: make a population of 50000 from normal(60, 12), take one sample of 30
# and compare the sample mean with the true population mean.
import numpy as np

rng = np.random.default_rng(1)
population = rng.normal(loc=60, scale=12, size=50000)
sample = rng.choice(population, size=30)

print("Population mean:", population.mean())
print("Sample mean:", sample.mean())
