# Task: use norm.cdf to find the share of a Normal distribution within +-1
# std dev and compare with the 68% empirical rule.
from scipy.stats import norm

inside = norm.cdf(1) - norm.cdf(-1)
print("Within 1 std dev:", inside)
print("Empirical rule says about 0.68")
