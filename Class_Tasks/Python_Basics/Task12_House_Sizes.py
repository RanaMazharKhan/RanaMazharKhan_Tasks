# Given house sizes, print the count, smallest, largest
# and average (rounded to 2 decimals).

house_sizes = [850, 900, 1200, 1500, 2000, 750, 1100]
print(len(house_sizes))
print(min(house_sizes))
print(max(house_sizes))
average = sum(house_sizes) / len(house_sizes)
print(f"{average:.2f}")
