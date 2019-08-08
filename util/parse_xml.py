with open("dblp.xml", 'r') as inF:
    content = inF.read()

lines = []

for line in content.splitlines():
    if line.startswith("<title>"):
        lines.append(line[7:-8].lower())

srtd = list(set(lines))

with open("dblp_titles.csv", 'w') as outF:
    for line in srtd:
        outF.write(line)
        outF.write("\n")
