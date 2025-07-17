def add(a,b):
    print(a+b)
def sub(a,b):
    print(a-b)
def multi(a,b):
    print(a*b)
def div(a,b):
    if b==0:
     print("Undefined")
    else:
     print(a/b)

a=float(input("Enter first number:"))
op=input("Enter the operation +,-,*,/:")
b=float(input("Enter second number:"))

if op=="+":
   add(a,b)
elif op=="-":
   sub(a,b)
elif op=="*":
   multi(a,b)
elif op=="/":
   div(a,b)
else:
 print("Invaild operation")