# Task 19 - Python - Looping Techniques
# Problem: Two lists: products = ['Pen','Book','Bag'] and prices =
# [50, 300, 900]. Use zip() to print each product with its price.

products = ["Pen", "Book", "Bag"]
prices = [50, 300, 900]
for product, price in zip(products, prices):
    print(product, ":", price)
