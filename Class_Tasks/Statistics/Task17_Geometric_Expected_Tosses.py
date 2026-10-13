# Task: with a fair coin (p = 0.5), what is the expected number of tosses
# until the first Head? (Geometric distribution)
from scipy import stats

print("Expected tosses:", stats.geom(0.5).mean())
