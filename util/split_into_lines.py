n = 50

with open("book.csv", "r") as input:
    content = input.read()


split_up = [content[i:i+n] for i in range (0, len(content), n)]

if (len(split_up[-1]) < n):
    split_up = split_up[:-1]

with open("book_split.csv", "w") as output:
    for line in split_up:
        output.write("%s\n" % line)