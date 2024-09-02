def add(a, b):
    return a + b

try:
    a = int(input("Enter a number: "))
    b = int(input("Enter another number: "))
    result = add(a, b)
    print(f"The sum of {a} and {b} is {result}")
except ValueError:
    print("Invalid input. Please enter a valid number.")

