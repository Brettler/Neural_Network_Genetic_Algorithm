# def check_ones(filename):
#     with open(filename, 'r') as file:
#         counter = 0
#         for line in file:
#             string, label = line.strip().split() # Assuming data is separated by a space
#             if int(label) == 1:
#                 if string.count('1') < 8:
#                     print(f"String {string} labeled 1 has 8 or more '1's.")
#                 else:
#                     counter+=1
#                     raise ValueError(f"String {string} labeled 1 has fewer than 8 '1's.")
#         print(f"counter is {counter}")
#
# check_ones('nn1.txt')


import random

# The length of each binary string
binary_str_len = 16

# The number of strings to generate
num_strings = 20000

# Open the file for writing
with open('binary_file.txt', 'w') as f:
    for i in range(num_strings):
        # Randomly generate a binary string
        binary_str = ''.join(str(random.randint(0, 1)) for _ in range(binary_str_len))
        # Count the number of 0's and 1's
        num_zeros = binary_str.count('0')
        num_ones = binary_str.count('1')

        # Decide the label based on the counts of 0's and 1's
        if 8 <= num_ones:
            label = 1
        else:
            label = 0

        # Write the binary string and its label to the file
        f.write(binary_str + '   ' + str(label) + '\n')
