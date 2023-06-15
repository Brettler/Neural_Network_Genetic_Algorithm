input_file = "nn1.txt"
output_file = "testnet1.txt"

with open(input_file, "r") as file:
    lines = file.readlines()

cleaned_lines = [line.split()[0] + "\n" for line in lines]

with open(output_file, "w") as file:
    file.writelines(cleaned_lines)

print("Labels removed and saved in", output_file)

########################################################################
# Read the data from the original file
with open("nn1.txt", "r") as f:
    data = f.readlines()

# Extract only the second column (last character) of each line
labels = [line.strip()[-1] for line in data]

# Write the extracted labels to a new file
with open("true_label1.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")


# Read the data from the original file
with open("binary_file.txt", "r") as f:
    data = f.readlines()

# Extract only the second column (last character) of each line
labels = [line.strip()[-1] for line in data]

# Write the extracted labels to a new file
with open("true_label_moreZeros.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")
