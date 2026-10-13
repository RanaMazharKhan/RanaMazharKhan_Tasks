# Reflection:
# 1. Customer 2 ended up "high value, long-term" (tenure 24, charges 85.5).
#    Yes, that matched - I gave them big numbers on purpose.
# 2. Flags like this are used in ML to label data, e.g. marking customers
#    for churn prediction or special offers.

customer2_tenure, customer2_charges = 24, 85.5
flag = customer2_tenure > 12 and customer2_charges > 70
print("Customer 2 is high value & long-term:", flag)
