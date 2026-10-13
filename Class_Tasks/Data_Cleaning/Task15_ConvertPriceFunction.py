# Task: write convert_into_crore(text) that parses strings like 'PKR1.2 Crore'
# and returns the value in Crore (1 Crore=100 Lakh, 1 Arab=100 Crore).

import re

def convert_into_crore(text):
    if text is None or not isinstance(text, str):
        return None
    m = re.search(r"(\d+\.?\d*)\s*([a-zA-Z]+)", text)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "crore":
        return num
    if unit == "lakh":
        return num / 100
    if unit == "arab":
        return num * 100
    if unit == "thousand":
        return num / 10000
    return None

print(convert_into_crore("PKR1.2 Crore"))   # 1.2
print(convert_into_crore("PKR50 Lakh"))     # 0.5
print(convert_into_crore("PKR2 Arab"))      # 200
print(convert_into_crore("PKR5"))           # None - no unit
