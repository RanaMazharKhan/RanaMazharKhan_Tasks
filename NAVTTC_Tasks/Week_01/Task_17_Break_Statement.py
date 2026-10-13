# Task 17 - Python - Break Statement
# Problem: Loop through the numbers [3, 7, 9, 12, 15] and stop at the
# first number that is divisible by 5. Print that number.

numbers = [3, 7, 9, 12, 15]
for n in numbers:
    if n % 5 == 0:
        print("First number divisible by 5:", n)
        break
