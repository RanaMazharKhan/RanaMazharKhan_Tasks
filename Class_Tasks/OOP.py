# Object Oriented Programming practice tasks

# First class: class Book with title and author, and class Movie with a
# summary() method that returns "title (year)". Make one object of each.
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

book = Book("The Alchemist", "Paulo Coelho")
print(book.title, "-", book.author)

class Movie:
    def __init__(self, title, year):
        self.title = title
        self.year = year

    def summary(self):
        return self.title + " (" + str(self.year) + ")"

movie = Movie("Inception", 2010)
print(movie.summary())

# ---
# Class attribute vs instance attribute: Car has wheels = 4 shared by all
# cars, brand belongs to each car. Make two cars and print both for each.
class Car:
    wheels = 4

    def __init__(self, brand):
        self.brand = brand

car1 = Car("Toyota")
car2 = Car("Honda")
print(car1.wheels, car1.brand)
print(car2.wheels, car2.brand)

# ---
# Objects in a loop: class Employee with name and salary, with __str__ so
# printing an object shows "Employee: Ali, Salary: 50000". Make 3 employees
# and print each one in a loop.
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def __str__(self):
        return f"Employee: {self.name}, Salary: {self.salary}"

staff = [Employee("Ali", 50000), Employee("Sara", 60000), Employee("Bilal", 55000)]
for emp in staff:
    print(emp)

# ---
# Class Rectangle with width and height. Methods area() and perimeter().
# Make an object and print both.
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

r = Rectangle(5, 3)
print("Area:", r.area())
print("Perimeter:", r.perimeter())

# ---
# Encapsulation: class Wallet with protected _balance. withdraw() only takes
# the amount if there is enough balance, else prints "Insufficient funds."
# Test one good withdrawal and one bad withdrawal.
class Wallet:
    def __init__(self, balance):
        self._balance = balance

    def withdraw(self, amount):
        if amount <= self._balance:
            self._balance = self._balance - amount
        else:
            print("Insufficient funds.")

    def get_balance(self):
        return self._balance

w = Wallet(1000)
w.withdraw(300)
print(w.get_balance())
w.withdraw(2000)
print(w.get_balance())

# ---
# Inheritance: parent Vehicle with brand, speed and describe(). Child
# ElectricVehicle adds battery_range and overrides describe() with super().
class Vehicle:
    def __init__(self, brand, speed):
        self.brand = brand
        self.speed = speed

    def describe(self):
        return self.brand + " runs at " + str(self.speed) + " km/h."

class ElectricVehicle(Vehicle):
    def __init__(self, brand, speed, battery_range):
        super().__init__(brand, speed)
        self.battery_range = battery_range

    def describe(self):
        return super().describe() + " Battery range is " + str(self.battery_range) + " km."

ev = ElectricVehicle("Tesla", 250, 500)
print(ev.describe())

# ---
# Polymorphism: classes Circle and Square each with area(). Put one of each
# in a list and print each area in a loop - same method name, different code.
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.1416 * self.radius ** 2

class Square:
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2

shapes = [Circle(3), Square(4)]
for shape in shapes:
    print(shape.area())

# ---
# class Product with name and price. Make 4 products, then use a list
# comprehension to get names of products priced above 100.
class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price

items = [Product("Pen", 50), Product("Bag", 300), Product("Book", 120), Product("Bottle", 80)]

costly = [p.name for p in items if p.price > 100]
print(costly)

# ---
# class Dataset storing a list of numbers, with mean(), minimum(), maximum()
# and normalize() (scales values to 0-1). Test on [10,20,30,40,50].
class Dataset:
    def __init__(self, values):
        self.values = values

    def mean(self):
        return sum(self.values) / len(self.values)

    def minimum(self):
        return min(self.values)

    def maximum(self):
        return max(self.values)

    def normalize(self):
        lo = self.minimum()
        hi = self.maximum()
        return [(v - lo) / (hi - lo) for v in self.values]

data = Dataset([10, 20, 30, 40, 50])
print("Mean:", data.mean())
print("Min:", data.minimum())
print("Max:", data.maximum())
print("Normalized:", data.normalize())

# ---
# Base class BaseModel with is_trained=False and a predict() guard.
# ConstantModel stores the max value, AverageModel stores the mean.
# Train both on [5,10,15,20,25], predict 3 values each, then check the
# untrained model raises ValueError.
class BaseModel:
    def __init__(self):
        self.is_trained = False

    def predict(self, n):
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call fit() first.")

class ConstantModel(BaseModel):
    def fit(self, training_values):
        self.constant = max(training_values)
        self.is_trained = True

    def predict(self, n):
        super().predict(n)
        return [self.constant] * n

class AverageModel(BaseModel):
    def fit(self, training_values):
        self.average = sum(training_values) / len(training_values)
        self.is_trained = True

    def predict(self, n):
        super().predict(n)
        return [self.average] * n

train_values = [5, 10, 15, 20, 25]
m1 = ConstantModel()
m1.fit(train_values)
print("Constant model:", m1.predict(3))

m2 = AverageModel()
m2.fit(train_values)
print("Average model:", m2.predict(3))

m3 = AverageModel()  # never trained
try:
    m3.predict(3)
except ValueError as e:
    print("Error caught:", e)

# ---
# class StudentRecord with name and marks. Methods average() and grade()
# (A >=90, B >=75, C >=60, else F). Make 4 records, then list names of
# students with grade A or B using a list comprehension.
class StudentRecord:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def average(self):
        return sum(self.marks) / len(self.marks)

    def grade(self):
        avg = self.average()
        if avg >= 90:
            return "A"
        elif avg >= 75:
            return "B"
        elif avg >= 60:
            return "C"
        else:
            return "F"

students = [
    StudentRecord("Ali", [78, 85, 90]),
    StudentRecord("Sara", [92, 88, 95]),
    StudentRecord("Bilal", [55, 60, 58]),
    StudentRecord("Hina", [76, 80, 74]),
]

top = [s.name for s in students if s.grade() in ("A", "B")]
print(top)
