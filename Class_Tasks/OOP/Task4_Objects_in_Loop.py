# Task 4: class Employee with name and salary. Make a list of 3 employees
# and print each one's name and salary with an f-string.
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

staff = [Employee("Ali", 50000), Employee("Sara", 60000), Employee("Bilal", 55000)]

for emp in staff:
    print(f"{emp.name} earns {emp.salary}")
