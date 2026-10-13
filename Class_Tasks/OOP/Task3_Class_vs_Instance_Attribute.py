# Task 3: class Car with class attribute wheels=4 and instance attribute brand.
# Make two cars and print wheels and brand for each.
class Car:
    wheels = 4

    def __init__(self, brand):
        self.brand = brand

car1 = Car("Toyota")
car2 = Car("Honda")
print(car1.wheels, car1.brand)
print(car2.wheels, car2.brand)
