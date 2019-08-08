prefix_length = 12

with open("dblp_titles_sorted.csv", 'r') as inF:
    content = inF.read()

lines = content.splitlines()

max_string = ""
max_number = 0
previous_string = lines[0][:prefix_length]
previous_number = 1

for i in range(1,len(lines)):
    if lines[i][:prefix_length] == previous_string:
        previous_number += 1
    else:
        if previous_number > max_number:
            max_string = previous_string
            max_number = previous_number
        previous_number = 1
        previous_string = lines[i][:prefix_length]

print(max_string)
print(max_number)
