# Task 9: classes Circle and Square each with area(). Put one of each in a list
# and print each area in a loop (polymorphism - same method name, different code).
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return self.radius ** 2 * 3.1416

class Square:
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2

shapes = [Circle(3), Square(4)]
for shape in shapes:
    print(shape.area())
