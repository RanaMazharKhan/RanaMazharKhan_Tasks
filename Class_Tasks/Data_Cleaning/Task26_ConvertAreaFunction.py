# Task: write convert_into_marla(text) - 1 Kanal = 20 Marla,
# 1 Sq. Yd. = value*9/225 Marla, Marla stays the same.

import re

def convert_into_marla(text):
    m = re.search(r"(\d+\.?\d*)", text)
    if not m:
        return None
    val = float(m.group(1))
    t = text.lower()
    if "kanal" in t:
        return val * 20
    if "sq" in t or "yd" in t:
        return val * 9 / 225
    return val

print(convert_into_marla("2 Kanal"))      # 40
print(convert_into_marla("120 Sq. Yd.")) # 4.8
print(convert_into_marla("5 Marla"))     # 5
