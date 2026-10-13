# Task 5: add a __str__ method to Employee so printing an object shows
# "Employee: Ali, Salary: 50000". Test it on one object.
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def __str__(self):
        return f"Employee: {self.name}, Salary: {self.salary}"

emp = Employee("Ali", 50000)
print(emp)
