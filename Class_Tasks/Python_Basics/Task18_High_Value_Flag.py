# For each customer, check if they are "high value, long-term":
# tenure_months > 12 AND monthly_charges > 70. Print labeled results.

customer1_tenure, customer1_charges = 4, 55.0
customer2_tenure, customer2_charges = 24, 85.5
customer3_tenure, customer3_charges = 14, 60.25

print("Customer 1 is high value & long-term:", customer1_tenure > 12 and customer1_charges > 70)
print("Customer 2 is high value & long-term:", customer2_tenure > 12 and customer2_charges > 70)
print("Customer 3 is high value & long-term:", customer3_tenure > 12 and customer3_charges > 70)
