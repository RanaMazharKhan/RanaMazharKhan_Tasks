# Task 26 - Python - Exception Handling
# Problem: Write code that converts the string 'abc' to int inside
# try/except and prints 'Invalid number' if it fails.

text = "abc"
try:
    number = int(text)
    print(number)
except ValueError:
    print("Invalid number")
