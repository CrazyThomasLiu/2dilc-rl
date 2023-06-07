import csv
import pdb
f = open('numbers2.csv', 'r')

with f:

    reader = csv.reader(f, delimiter=",")

    for row in reader:

        for e in row:
            a=float(e)
            pdb.set_trace()
            print(e)