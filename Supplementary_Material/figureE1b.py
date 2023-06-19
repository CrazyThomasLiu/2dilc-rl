import os
import numpy as np
import csv
import pdb
import pandas as pd
time_length=200
# set the current path
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
"""1. Load data"""
figure_data_path=os.path.join(current_dir, "FigureE1b_original.csv")
'get the length of the data'
f_ref=open(figure_data_path,'r')
length=len(f_ref.readlines())
batch=int(length/time_length)
rl_input=np.zeros((batch,time_length))
f_ref=open(figure_data_path,'r')
number=0
num=0
with f_ref:
    reader=csv.reader(f_ref)
    for row in reader:
        rl_input[number, num] = float(row[0])
        num+=1
        if num==time_length:
            #pdb.set_trace()
            num=0
            number+=1

"""2. save data to csv"""



array = np.array(rl_input)
print(type(array))
print(array)
df = pd.DataFrame(array)
print(type(df))
print(df)
df.to_csv('FigureE1b.csv')
