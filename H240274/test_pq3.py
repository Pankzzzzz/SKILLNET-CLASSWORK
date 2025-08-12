import time
from functools import wraps

def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{current_time}] Calling function: {func._name_}")
        print(f"Arguments: args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"[{current_time}] Function {func._name_} returned: {result}")
        return result
    return wrapper

@log_function_call
def add(a, b):
    return a + b

@log_function_call
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

add(10, 5)
greet("Alice", greeting="Hi")