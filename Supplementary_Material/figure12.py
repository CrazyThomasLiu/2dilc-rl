import os
import numpy as np
import csv
import pdb
import pandas as pd
time_length=300
# set the current path
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
"""1. Load data"""
figure_data_path=os.path.join(current_dir, "Figure12_original.csv")
'get the length of the data'
f_ref=open(figure_data_path,'r')
length=len(f_ref.readlines())
batch=int(length/time_length)
output=np.zeros((batch,time_length))
f_ref=open(figure_data_path,'r')
number=0
num=0
with f_ref:
    reader=csv.reader(f_ref)
    for row in reader:
        output[number, num] = float(row[0])
        num+=1
        if num==time_length:
            #pdb.set_trace()
            num=0
            number+=1

#pdb.set_trace()
"""2. save data to csv"""



array = np.array(output)
print(type(array))
print(array)
df = pd.DataFrame(array)
print(type(df))
print(df)
df.to_csv('Figure12.csv')
'''
with open('Figure7.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(['None','Batch'])
    batch_list=[str(i+1) for i in range(batch)]
    batch_list=batch_list.insert(0,'Time')
    writer.writerow(batch_list)
    tem=[output[item,0] for item in range(batch)]
    pdb.set_trace()
    #writer.writerow(['1', output[item,0] for item in range(batch)])
    #writer.writerows(map(lambda x,y: [x,y], batch,rmse))
'''