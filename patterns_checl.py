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

#################################################################################

# import random
#
# # The length of each binary string
# binary_str_len = 16
#
# # The number of strings to generate
# num_strings = 20000
#
# # Open the file for writing
# with open('binary_file.txt', 'w') as f:
#     for i in range(num_strings):
#         # Randomly generate a binary string
#         binary_str = ''.join(str(random.randint(0, 1)) for _ in range(binary_str_len))
#         # Count the number of 0's and 1's
#         num_zeros = binary_str.count('0')
#         num_ones = binary_str.count('1')
#
#         # Decide the label based on the counts of 0's and 1's
#         if 8 <= num_ones:
#             label = 1
#         else:
#             label = 0
#
#         # Write the binary string and its label to the file
#         f.write(binary_str + '   ' + str(label) + '\n')

#################################################################################

import pandas as pd

# Load the dataset
df = pd.read_csv('./nn0.txt', sep='\s+', header=None, names=['pattern', 'class'])

# Convert integer patterns to string
df['pattern'] = df['pattern'].astype(str)

# Initialize a list to store the results
results = []

# Check each pattern option
for pattern in df['pattern'].unique():
    # Filter the dataset for the current pattern
    pattern_df = df[df['pattern'] == pattern]

    # Get the unique labels for the current pattern
    unique_labels = pattern_df['class'].unique()

    # Iterate over the range options
    for lower in range(7, 13):
        for upper in range(lower + 1, 17):
            # Initialize dictionaries to store the outlier counts for each label
            count_outside_1s = {}
            count_outside_0s = {}

            # Check the count of 1s and 0s for each label
            skip_iteration = False  # Flag to skip the iteration
            for label in unique_labels:
                label_df = pattern_df[pattern_df['class'] == label]
                count_1s = label_df['pattern'].apply(lambda x: str(x).count('1'))
                count_0s = label_df['pattern'].apply(lambda x: str(x).count('0'))

                count_outside_1s[label] = label_df[(count_1s < lower) | (count_1s > upper)].shape[0]
                count_outside_0s[label] = label_df[(count_0s < lower) | (count_0s > upper)].shape[0]

                # Check if the count exceeds the threshold
                if count_outside_1s[label] + count_outside_0s[label] > 200:
                    skip_iteration = True
                    break

            # Skip the iteration if count exceeds the threshold
            if skip_iteration:
                continue

            # Store the results
            results.append(((pattern, lower, upper), '1s', count_outside_1s))
            results.append(((pattern, lower, upper), '0s', count_outside_0s))

# Before sorting, print out some values to debug
for result in results:
    print(result)


# Sort the results by the count of patterns outside the range
results.sort(key=lambda x: sum(sum(d.values()) for d in x[2].values()))

# Print the results
for (pattern, lower, upper), bit, counts in results:
    print(f'Pattern: {pattern}, Range for {bit}: {lower}-{upper}')
    for label, count in counts.items():
        print(f'Label {label}, Count Outside: {count}')
    print()

