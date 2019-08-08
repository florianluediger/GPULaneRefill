import re

filename = "../data/dblp/dblp_0.64_data.csv"

with open(filename, 'r') as inF:
    content = inF.read()

trimmed = list(map(lambda x: re.sub(r"[^a-zA-Z0-9]+", ' ', x), content.splitlines()))

with open(filename, 'w') as outF:
    for line in trimmed:
        outF.write(line)
        outF.write("\n")
