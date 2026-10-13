# Task 10: class Product with name and price. Make 4 products, then use a list
# comprehension to get names of products priced above 100.
class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price

items = [Product("Pen", 50), Product("Bag", 300), Product("Book", 120), Product("Bottle", 80)]

costly = [p.name for p in items if p.price > 100]
print(costly)
