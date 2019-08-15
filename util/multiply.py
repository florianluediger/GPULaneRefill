import sys

multiplications = 5

for path in sys.argv[1:]:
    infile = path
    outfile = path.replace("small/","")

    with open(infile, 'r') as inF:
        content = inF.read()

    with open(outfile, 'w') as outF:
        for i in range(1, multiplications + 1):
            outF.write(content)
            if (i < multiplications):
                outF.write("\n")
