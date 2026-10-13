# Task: build a device vs purchase contingency table, run a chi-square test
# and interpret the result.
import pandas as pd
from scipy import stats

table = pd.DataFrame({"Yes": [80, 55, 15], "No": [120, 45, 35]},
                     index=["Mobile", "Desktop", "Tablet"])
print(table)

chi2, p_value, dof, expected = stats.chi2_contingency(table)
print("p-value:", p_value)
if p_value < 0.05:
    print("Significant: device type is associated with purchase completion.")
else:
    print("Not significant: no association found.")
