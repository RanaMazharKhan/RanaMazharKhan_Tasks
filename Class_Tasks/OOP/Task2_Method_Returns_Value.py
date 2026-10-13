# Task 2: add a summary() method to Movie that returns "title (year)", then print it.
class Movie:
    def __init__(self, title, year):
        self.title = title
        self.year = year

    def summary(self):
        return self.title + " (" + str(self.year) + ")"

movie = Movie("Inception", 2010)
print(movie.summary())
