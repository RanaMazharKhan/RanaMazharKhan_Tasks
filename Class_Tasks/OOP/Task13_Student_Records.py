# Task 13: class StudentRecord with name and marks. Methods average() and grade()
# (A >=90, B >=75, C >=60, else F). Make 4 records, then list names of
# students with grade A or B using a list comprehension.
class StudentRecord:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def average(self):
        return sum(self.marks) / len(self.marks)

    def grade(self):
        avg = self.average()
        if avg >= 90:
            return "A"
        elif avg >= 75:
            return "B"
        elif avg >= 60:
            return "C"
        else:
            return "F"

students = [
    StudentRecord("Ali", [78, 85, 90]),
    StudentRecord("Sara", [92, 88, 95]),
    StudentRecord("Bilal", [55, 60, 58]),
    StudentRecord("Hina", [76, 80, 74]),
]

top = [s.name for s in students if s.grade() in ("A", "B")]
print(top)
