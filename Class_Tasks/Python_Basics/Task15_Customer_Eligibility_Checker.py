# Customer Eligibility Checker: long_enough (tenure >= 6), affordable
# (charges < 80), eligible only if both are true. Print all three with labels.
# Cast monthly_charges to int and show what was lost.

tenure_months = 8
monthly_charges = 65.50
contract_type = "Month-to-month"

long_enough = tenure_months >= 6
affordable = monthly_charges < 80
eligible = long_enough and affordable

print("long_enough:", long_enough)
print("affordable:", affordable)
print("eligible:", eligible)

print(int(monthly_charges))  # 65, the .50 part is lost (int truncates)

# optional challenge: also require a month-to-month contract
eligible2 = long_enough and affordable and contract_type == "Month-to-month"
print("eligible with contract check:", eligible2)
