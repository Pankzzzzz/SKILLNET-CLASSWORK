import random

bill = float(input("Enter total bill amount: "))
friends = int(input("Enter number of friends to split the bill: "))
tip_percent = random.randint(10, 50)

if friends <= 0:
    print("Invalid number of friends!")
    exit()

each_share = bill / friends
tip_amount = bill * (tip_percent / 100)
total_with_tip = bill + tip_amount
each_share_with_tip = total_with_tip / friends

print(f"Bill per person (without tip): ₹{each_share:.2f}")
print(f"Random tip percentage: {tip_percent}%")
print(f"Tip amount: ₹{tip_amount:.2f}")
print(f"Total bill with tip: ₹{total_with_tip:.2f}")
print(f"Each person pays (with tip): ₹{each_share_with_tip:.2f}")