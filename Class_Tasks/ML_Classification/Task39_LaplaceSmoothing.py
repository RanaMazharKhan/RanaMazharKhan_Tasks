# Practice Q2 - Set alpha to 0.1 and 10. How do the smoothed Outlook probabilities change?
import pandas as pd

data = pd.DataFrame({
    "Outlook":  ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast",
                 "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
    "Temp":     ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool",
                 "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal",
                 "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
    "Wind":     ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong",
                 "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
    "Play":     ["No", "No", "Yes", "Yes", "Yes", "No", "Yes",
                 "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"],
})


def smoothed_likelihood(feature, alpha=1.0):
    counts = pd.crosstab(data[feature], data["Play"])
    K = counts.shape[0]
    return (counts + alpha) / (counts.sum() + alpha * K)


for alpha in [0.1, 1.0, 10.0]:
    print(f"--- alpha = {alpha} ---")
    print(smoothed_likelihood("Outlook", alpha=alpha).round(3))
    print()

# With alpha = 0.1 the probabilities stay close to the raw counts (spiky),
# with alpha = 10 they get pulled toward uniform (flat) - strong smoothing
# washes out the differences the data actually shows.
