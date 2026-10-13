# Task: write get_loc(loc) that strips the trailing ', Lahore, Punjab'
# from a location string.

def get_loc(loc):
    return loc.replace(", Lahore, Punjab", "")

print(get_loc("Garden Town, Lahore, Punjab"))
print(get_loc("Gulberg, Lahore, Punjab"))
