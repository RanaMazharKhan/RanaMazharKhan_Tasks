# Bonus - OOP in Python - Classes, Inheritance, Polymorphism
# Problem: Create a class Student with name and marks, a method
# result() that returns 'Pass' if marks >= 50 else 'Fail', and print
# the result for a student with 72 marks.

class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks
    def result(self):
        return "Pass" if self.marks >= 50 else "Fail"

s1 = Student("Rana", 72)
print(s1.name, "->", s1.result())
