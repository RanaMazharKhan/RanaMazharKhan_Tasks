# Task 6: class Rectangle with width and height. Methods area() and perimeter().
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
