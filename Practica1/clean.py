import csv

with open("genero.txt", newline='\n') as f_input, open("genero_clean.txt", 'w', newline='\n') as f_output:
    csv_input = csv.reader(f_input)
    csv_output = csv.writer(f_output)
    csv_output.writerow(next(csv_input))

    for row in csv_input:
        row[0] = 1 if row[0] == "Male" else 0
        csv_output.writerow(row)


with open("default.txt", newline='\n') as f_input, open("default_clean.txt", 'w', newline='\n') as f_output:
    csv_input = csv.reader(f_input, delimiter="\t")
    csv_output = csv.writer(f_output)
    csv_output.writerow(next(csv_input))

    for row in csv_input:
        row[1] = 1 if row[1] == "Yes" else 0
        row[2] = 1 if row[2] == "Yes" else 0
        csv_output.writerow(row)
