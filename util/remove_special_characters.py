import re
import sys

for path in sys.argv[1:]:
    filename = path

    with open(filename, 'r') as inF:
        content = inF.read()

    trimmed = list(map(lambda x: re.sub(r"[^a-zA-Z0-9]+", ' ', x), content.splitlines()))

    with open(filename, 'w') as outF:
        for line in trimmed:
            outF.write(line)
            outF.write("\n")
