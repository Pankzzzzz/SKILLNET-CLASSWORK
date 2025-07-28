class bank:
    def __init__(self):
        self._balance = 15000
    
    def deposite(self,amount):
        self._balance += amount
        return  amount

    # def get_balance(self):
    #     return self._balance
    
acc = bank()
acc.deposite(1000)
print("Balance:",acc.deposite())