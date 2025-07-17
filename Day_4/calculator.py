print("Calculator")
num1=float(input("Enter first number:"))
op=input("Enter the operation +,-,*,/:")
num2=float(input("Enter second number:"))

if op== '+':
 print(f"Result: {num1}+{num2}={num1+num2}")
elif op=='-':
 print(f"Result: {num1}-{num2}={num1-num2}")
elif op=='*':
 print(f"Result: {num1}*{num2}={num1*num2}")
elif op=='/':
    if num2==0:
     print("Undefined")
    else:
     print(f"Result: {num1}/{num2}={num1/num2}")
else :
 print("INVALID INPUT")