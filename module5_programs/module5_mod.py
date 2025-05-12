# module5_mod.py

class NumberHandler:
    def __init__(self):
        self.numbers = []

    def insert_numbers(self, nums):
        self.numbers = nums

    def find_number(self, target):
        try:
            # Return 1-based index
            return self.numbers.index(target) + 1
        except ValueError:
            return -1
