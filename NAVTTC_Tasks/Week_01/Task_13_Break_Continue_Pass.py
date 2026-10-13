# Task 13 - Python - Break, Continue and Pass Statements
# Problem: Print numbers from 1 to 10 but stop the loop as soon as the
# number 6 comes.

for n in range(1, 11):
    if n == 6:
        break
    print(n, end=" ")
print()
