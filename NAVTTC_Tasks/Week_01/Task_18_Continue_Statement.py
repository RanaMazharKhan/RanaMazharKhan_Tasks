# Task 18 - Python - Continue Statement
# Problem: Print numbers from 1 to 8, skipping the number 5 only.

for n in range(1, 9):
    if n == 5:
        continue
    print(n, end=" ")
print()
