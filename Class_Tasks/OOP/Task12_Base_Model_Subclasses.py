# Task 12: base class BaseModel with is_trained=False and a predict() guard.
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

data = [5, 10, 15, 20, 25]
m1 = ConstantModel()
m1.fit(data)
print("Constant model:", m1.predict(3))

m2 = AverageModel()
m2.fit(data)
print("Average model:", m2.predict(3))

m3 = AverageModel()   # never trained
try:
    m3.predict(3)
except ValueError as e:
    print("Error caught:", e)
