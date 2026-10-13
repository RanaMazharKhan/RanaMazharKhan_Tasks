# Given student_record = ["Ayesha", 21, [90, 87, 93]], print a formatted
# sentence with name, age and average marks using an f-string.

student_record = ["Ayesha", 21, [90, 87, 93]]
name, age, marks = student_record
average = sum(marks) / len(marks)
print(f"{name} is {age} years old and scored an average of {average:.1f}")
