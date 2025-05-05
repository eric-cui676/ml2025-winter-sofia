# Ask for a positive integer N
N = int(input("Enter a positive integer N: "))

# Read N numbers one by one and store them in a list
numbers = []
for i in range(N):
    num = int(input(f"Enter number {i+1}: "))
    numbers.append(num)

# Ask for integer X
X = int(input("Enter an integer X to search for: "))

# Check if X is in the list and print result
if X in numbers:
    index = numbers.index(X) + 1  # 1-based index
    print(index)
else:
    print(-1)
