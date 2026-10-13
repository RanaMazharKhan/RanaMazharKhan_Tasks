# Task 11: class Dataset storing a list of numbers, with mean(), minimum(),
# maximum() and normalize() (scales values to 0-1). Test on [10,20,30,40,50].
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
