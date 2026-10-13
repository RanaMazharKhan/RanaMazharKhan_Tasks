# Task 27 - Python - User Defined Exceptions
# Problem: Create a custom exception NegativeNumberError and raise it
# if a given number (-4) is negative. Print the error message.

class NegativeNumberError(Exception):
    pass

number = -4
try:
    if number < 0:
        raise NegativeNumberError("Number must not be negative")
    print("Number is fine")
except NegativeNumberError as e:
    print("Custom error:", e)
