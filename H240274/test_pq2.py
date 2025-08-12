def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def prime_nums_generator():
    n = 2
    while True:
        if is_prime(n):
            yield n
        n += 1

primes = prime_nums_generator()

n = int(input("Enter number of prime numbers you want to generate? "))

print("First",n,"Prime numbers:")
for _ in range(n):
    print(next(primes))
