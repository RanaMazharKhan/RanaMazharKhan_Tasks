# Python Basics - practice tasks from the course notebooks

# Create variables and print their values and types.
city = "Lahore"
age = 21
height = 5.9
print(city, type(city))
print(age, type(age))
print(height, type(height))

# ---
# Print a sentence about your city using an f-string.
city = "Lahore"
print(f"City: {city}, Population type: {type(city).__name__}")

# ---
# Convert strings to numbers: int("45") + 5, and float("5") + float("3.2").
result = int("45") + 5
print(result)
total = float("5") + float("3.2")
print(total)

# ---
# Cast a float to int and see what is lost.
charges = 85.5
print("original:", charges)
print("as int:", int(charges))  # int() drops the .5, it does not round

# ---
# Given temperatures = [30, 32, 29, 35, 31], print length and sum.
temperatures = [30, 32, 29, 35, 31]
print(len(temperatures))
print(sum(temperatures))

# ---
# Given scores = [55, 89, 76, 92, 40, 67], print min, max and average.
scores = [55, 89, 76, 92, 40, 67]
print(min(scores))
print(max(scores))
print(sum(scores) / len(scores))

# ---
# Given house sizes, print count, smallest, largest and average (2 decimals).
house_sizes = [850, 900, 1200, 1500, 2000, 750, 1100]
print(len(house_sizes))
print(min(house_sizes))
print(max(house_sizes))
print(f"{sum(house_sizes) / len(house_sizes):.2f}")

# ---
# Sort values without changing the original list.
values = [3, 1, 4, 1, 5, 9, 2, 6]
print(sorted(values))
print(values)

# ---
# String slicing: first 4 and last 4 characters of "Data Science".
text = "Data Science"
print(text[:4])
print(text[-4:])

# ---
# Clean up a messy string: strip spaces and title case it.
name = "  machine learning  "
print(name.strip().title())

# ---
# From raw = "  Income: 55000  ", extract just the number as an int.
raw = "  Income: 55000  "
number = int(raw.split(":")[1].strip())
print(number)

# ---
# Nested list of 3 products [name, price]; print the price of the 2nd product.
products = [["Laptop", 1200], ["Mouse", 25], ["Keyboard", 45]]
print(products[1][1])

# ---
# Unpack a student record and print a formatted sentence with average marks.
student_record = ["Ayesha", 21, [90, 87, 93]]
name, age, marks = student_record
average = sum(marks) / len(marks)
print(f"{name} is {age} years old and scored an average of {average:.1f}")

# ---
# Customer Eligibility Checker: long_enough (tenure >= 6), affordable
# (charges < 80), eligible only if both are true. Print all three.
tenure_months = 8
monthly_charges = 65.50
contract_type = "Month-to-month"

long_enough = tenure_months >= 6
affordable = monthly_charges < 80
eligible = long_enough and affordable

print("long_enough:", long_enough)
print("affordable:", affordable)
print("eligible:", eligible)

# optional: also require a month-to-month contract
eligible2 = long_enough and affordable and contract_type == "Month-to-month"
print("eligible with contract check:", eligible2)

# ---
# Three customers with tenure_months (int) and monthly_charges (float).
# Flag each as "high value, long-term": tenure > 12 AND charges > 70.
customers = [
    ("Customer 1", 4, 55.0),
    ("Customer 2", 24, 85.5),
    ("Customer 3", 14, 60.25),
]

for name, tenure, charges in customers:
    high_value = tenure > 12 and charges > 70
    print(name, "is high value & long-term:", high_value)
