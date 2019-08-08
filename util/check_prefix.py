data_file = "../data/dblp/small/dblp_0.08_data.csv"
match_string_file = "../data/dblp/search.csv"

with open(match_string_file, "r") as inF:
    match_string = inF.read()

number_of_matches = 0

with open(data_file, "r") as inF:
    search_string = inF.read()

for line in search_string.split("\n"):
    if (line.startswith(match_string)):
        number_of_matches += 1
    
print(number_of_matches)
