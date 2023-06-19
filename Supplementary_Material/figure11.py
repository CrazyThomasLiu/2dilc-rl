import os
import numpy as np
import csv
import pdb

time_length=300
# set the current path
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
"""1. Load data"""
figure_data_path=os.path.join(current_dir, "Figure11_original.csv")
'get the length of the data'
f_ref=open(figure_data_path,'r')
length=len(f_ref.readlines())-1
num=0
rmse=np.zeros(length)
batch=np.zeros(length,dtype=int)
f_ref=open(figure_data_path,'r')
with f_ref:
    reader=csv.DictReader(f_ref)
    for row in reader:
        batch[num] = int(row['Step'])/time_length
        rmse[num]=row['Value']
        num+=1
with open('Figure11.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Batch','RMSE'])
    writer.writerows(map(lambda x,y: [x,y], batch,rmse))