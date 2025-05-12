# module5_oop.py

class NumberHandler:
    def __init__(self):
        self.numbers = []

    def insert_numbers(self, nums):
        self.numbers = nums

    def find_number(self, target):
        try:
            return self.numbers.index(target) + 1
        except ValueError:
            return -1

def main():
    nh = NumberHandler()

    N = int(input("Enter a positive integer (N): "))
    nums = []
    for i in range(N):
        nums.append(int(input(f"Enter number {i+1}: ")))
    nh.insert_numbers(nums)

    X = int(input("Enter the number to search for (X): "))
    result = nh.find_number(X)
    print(result)

if __name__ == "__main__":
    main()
