# Task 8: parent class Vehicle with brand, speed and describe(). Child class
# ElectricVehicle adds battery_range and overrides describe() using super().
class Vehicle:
    def __init__(self, brand, speed):
        self.brand = brand
        self.speed = speed

    def describe(self):
        return self.brand + " runs at " + str(self.speed) + " km/h."

class ElectricVehicle(Vehicle):
    def __init__(self, brand, speed, battery_range):
        super().__init__(brand, speed)
        self.battery_range = battery_range

    def describe(self):
        return super().describe() + " Battery range is " + str(self.battery_range) + " km."

ev = ElectricVehicle("Tesla", 250, 500)
print(ev.describe())
