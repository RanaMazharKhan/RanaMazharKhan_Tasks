# From raw = "  Income: 55000  ", extract just the number as an integer.

raw = "  Income: 55000  "
number = int(raw.split(":")[1].strip())
print(number)
