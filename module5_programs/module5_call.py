# module5_call.py

from module5_mod import NumberHandler

def main():
    nh = NumberHandler()

    # Read the count
    while True:
        try:
            N = int(input("Enter a positive integer (N): "))
            if N <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    nums = []
    for i in range(N):
        while True:
            try:
                num = int(input(f"Enter number {i+1}: "))
                nums.append(num)
                break
            except ValueError:
                print("Invalid input. Please enter an integer.")

    nh.insert_numbers(nums)

    while True:
        try:
            X = int(input("Enter the number to search for (X): "))
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    result = nh.find_number(X)
    print(result)

if __name__ == "__main__":
    main()
