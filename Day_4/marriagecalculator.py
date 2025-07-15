gen = int(input("Enter gender (1 for male, 2 for female): "))
if gen not in [1, 2]:
    print("INVALID INPUT!")
    exit()
age = int(input("Enter age: "))

if age >= 18:
    print("You are an adult")
    if gen == 1 and age >= 21:
        print("You can marry")
    elif gen == 1 and age < 21:
        print("You cannot marry")
    elif gen == 2 and age >= 18:
        print("You can marry")
    elif gen == 2 and age < 18:
        print("You cannot marry")
else:
    print("You are not an adult")
    print("You cannot marry")