data_file = "../data/type/type_0.02_data.csv"
match_string_file = "../data/type/search.csv"

with open(match_string_file, "r") as inF:
    match_string = inF.read()

number_of_matches = 0

with open(data_file, "r") as inF:
    search_string = inF.read()

for line in search_string.split("\n"):
    if (line == match_string):
        number_of_matches += 1
    
print(number_of_matches)
