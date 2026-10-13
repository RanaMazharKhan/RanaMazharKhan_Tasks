# Bonus - File Handling in Python
# Problem: Write the numbers 1 to 3, each on its own line, into a temp
# file, read the file back, and print how many lines it has.

import os, tempfile
path = os.path.join(tempfile.gettempdir(), "navttc_numbers.txt")
with open(path, "w") as f:
    for n in (1, 2, 3):
        f.write(str(n) + "\n")
with open(path, "r") as f:
    lines = f.readlines()
print("Number of lines:", len(lines))
os.remove(path)
