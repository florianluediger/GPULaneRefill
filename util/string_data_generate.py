import strgen
import math
import random

min_char_per_line = 1000
max_char_per_line = 1000
number_of_lines = 35000
selectivity = 0.1
regex_letters = "[a-z]"

occurrences = int(math.ceil(number_of_lines * selectivity))
match_positions = random.sample(range(1,number_of_lines + 1), occurrences)
number_of_randoms = number_of_lines - occurrences

search_string = strgen.StringGenerator("{0}{{{1}:{2}}}".format(regex_letters, min_char_per_line, max_char_per_line)).render_list(1, unique=False)[0]

data = strgen.StringGenerator("{0}{{{1}:{2}}}".format(regex_letters, min_char_per_line, max_char_per_line)).render_list(number_of_randoms, unique=False)

match_index = 0
generate_index = 0

with open("search_string.csv", "w") as output:
    output.write(search_string)

with open("string_data.csv", "w") as output:
    for i in range(1,number_of_lines + 1):
        if i in match_positions:
            output.write("{0}".format(search_string))
            match_index += 1
        else:
            output.write("{0}".format(data[generate_index]))
            generate_index += 1
        if (i < number_of_lines):
            output.write("\n")