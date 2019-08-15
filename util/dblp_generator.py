import math
import random
import os

part_csv_file = "./data/dblp_titles.csv"
output_file_prefix = "./data/small/dblp_"
inject_string = "performance optimization of gpu string processing"

selectivities = [0.01,0.32]
# selectivities = [0.0025,0.005,0.01,0.02,0.04,0.08,0.16,0.32,0.64]


def inject_matching_string(selectivity):
    with open(part_csv_file, 'r') as inF:
        content = inF.readlines()

    print("file read")

    number_of_lines = len(content)
    occurrences = int(math.ceil(number_of_lines * selectivity))
    match_positions = random.sample(range(1,number_of_lines + 1), occurrences)
    match_positions.sort()

    print("match positions found")

    match_index = 0

    with open(output_file_prefix + str(selectivity) + "_data.csv", 'w') as outF:
        for i in range(0, number_of_lines):
            if i < match_positions[match_index]:
                outF.write(content[i])
            else:
                outF.write(inject_string)
                outF.write("\n")
                if (match_index < len(match_positions) - 1):
                    match_index += 1

    with open(output_file_prefix + str(selectivity) + "_data.csv", 'rb+') as outF:
        outF.seek(-1, os.SEEK_END)
        outF.truncate()
    
    print(selectivity)
    print("file written")

for sel in selectivities:
    inject_matching_string(sel)
