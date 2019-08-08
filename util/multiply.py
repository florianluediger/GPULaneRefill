multiplications = 5
infile = "../data/dblp/small/dblp_0.02_data.csv"
outfile = "../data/dblp/dblp_0.02_data.csv"

with open(infile, 'r') as inF:
    content = inF.read()

with open(outfile, 'w') as outF:
    for i in range(1, multiplications + 1):
        outF.write(content)
        if (i < multiplications):
            outF.write("\n")
