class Person():
    def __init__(self,name,age,roll_no):
     self.name=name
     self.age=age
     self.roll_no=roll_no

class Student(Person):
   def __init__(self, name, age, roll_no):
      super().__init__(name, age, roll_no)

p1 = Student("Pankaj",18,18)

print(p1.name,p1.age,p1.roll_no)