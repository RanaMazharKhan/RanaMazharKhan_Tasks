# Pick one customer's monthly_charges, cast it to int, print both values.
# Explain in a comment what was lost.

charges = 85.5
print("original:", charges)
print("as int:", int(charges))
# int() dropped the .5 part. It truncates, it does not round.
