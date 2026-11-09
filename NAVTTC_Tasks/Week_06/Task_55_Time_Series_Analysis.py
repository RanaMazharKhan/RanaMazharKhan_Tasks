# Task 55 - Machine Learning - Time Series Analysis
# Problem: Make a time series of daily shop sales for 30 days using
# pandas. Print the first rows, weekly totals (resample), a 7-day
# rolling average, and the average daily sales.

import pandas as pd
import numpy as np

dates = pd.date_range("2026-01-01", periods=30, freq="D")
np.random.seed(5)
sales = 100 + np.random.randint(-20, 30, size=30) + np.arange(30)

data = pd.DataFrame({"sales": sales}, index=dates)
print("First 5 rows:")
print(data.head())

print("\nSummary:")
print(data.describe())

weekly = data.resample("W").sum()
print("\nWeekly total sales:")
print(weekly)

data["rolling_7"] = data["sales"].rolling(window=7).mean()
print("\nLast 5 days with 7-day rolling average:")
print(data.tail().round(1))

print("\nAverage daily sales:", round(data["sales"].mean(), 1))
print("Best day:", data["sales"].idxmax().date(), "sales =", data["sales"].max())
print("Simple forecast for next day (last rolling avg):", round(data["rolling_7"].iloc[-1], 1))
