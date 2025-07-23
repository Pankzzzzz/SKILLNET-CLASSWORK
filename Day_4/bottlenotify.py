total_bottles = int(input("Enter the total numbers of bottles:"))
drunk= int(input("Number of bottles drunk per day:"))
if drunk>total_bottles:
    print("You don't have enough bottles")
    print(f"You only have {total_bottles} bottles")
else :
 bottles_left=total_bottles
 Day=1

 while bottles_left>0:
    bottle_drunk=min(drunk,bottles_left)
    bottles_left -= bottle_drunk
    print(f"Day {Day}: Drunk {drunk} bottle {bottles_left} left.")

 print("No more bottles left!")