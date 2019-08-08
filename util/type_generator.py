import math
import random
import os

part_tbl_file = "../TPC-H/dbgen/part.tbl"
column_id = 4
part_csv_file = "../data-backup/type/part.csv"
output_file_prefix = "./type_"
inject_string = "SMALL POLISHED SILVER"
selectivity = 0.02


def extract_from_tbl():
    with open(part_tbl_file, 'r') as inF:
        content = inF.readlines()
    with open(part_csv_file, 'w') as outF:
        for line in content:
            outF.write(line.split('|')[column_id] + "\n")

def inject_matching_string():
    with open(part_csv_file, 'r') as inF:
        content = inF.readlines()

    number_of_lines = len(content)
    occurrences = int(math.ceil(number_of_lines * selectivity))
    match_positions = random.sample(range(1,number_of_lines + 1), occurrences)

    with open(output_file_prefix + str(selectivity) + "_data.csv", 'w') as outF:
        for i in range(0, number_of_lines):
            if i in match_positions:
                outF.write(inject_string)
                outF.write("\n")
            else:
                outF.write(content[i])
    
    with open(output_file_prefix + str(selectivity) + "_data.csv", 'rb+') as outF:
        outF.seek(-1, os.SEEK_END)
        outF.truncate()

inject_matching_string()