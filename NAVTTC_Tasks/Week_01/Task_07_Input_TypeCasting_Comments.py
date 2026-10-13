# Task 7 - Python - Input, Type Casting and Comments
# Problem: A user types marks as the string '78'. Cast it to int, add
# 2 grace marks, and print the final marks.

marks_text = "78"          # as if typed with input()
marks = int(marks_text)    # type casting
final_marks = marks + 2
print("Final marks:", final_marks)
