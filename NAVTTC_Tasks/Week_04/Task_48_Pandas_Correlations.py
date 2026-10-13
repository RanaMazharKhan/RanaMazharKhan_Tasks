# Task 48 - Pandas - Data Correlations
# Problem: For data Hours=[2,4,6] and Marks=[50,70,90], print the
# correlation matrix and say in a comment whether the relation is
# strong positive.

import pandas as pd
df = pd.DataFrame({"Hours": [2, 4, 6], "Marks": [50, 70, 90]})
print(df.corr(numeric_only=True))
# The value is close to +1, so the relation is strong positive:
# more hours studied -> higher marks.
