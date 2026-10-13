# Task 7: class Wallet with protected _balance. withdraw() only allows the amount
# if enough balance is there, else prints "Insufficient funds.". Test both cases.
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
w.withdraw(300)          # works
print(w.get_balance())
w.withdraw(2000)         # fails
print(w.get_balance())
