# Task: make a small products DataFrame, find the IQR of price and flag
# products outside Q1 - 1.5*IQR and Q3 + 1.5*IQR as outliers.
import pandas as pd

products = pd.DataFrame({
    "name": ["Chair", "Desk", "Lamp", "Monitor", "Laptop", "Pen"],
    "price": [45, 120, 25, 220, 1200, 2]
})

q1 = products["price"].quantile(0.25)
q3 = products["price"].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5 * iqr
high = q3 + 1.5 * iqr

products["outlier"] = (products["price"] < low) | (products["price"] > high)
print(products)
