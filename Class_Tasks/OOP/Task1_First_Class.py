# Task 1: make a class Book with title and author, create one book object and print both.
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

book = Book("The Alchemist", "Paulo Coelho")
print(book.title)
print(book.author)
