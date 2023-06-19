from control.matlab import *  # MATLAB-like functions
import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import Subplot
import os
import csv
import pandas as pd
current_path=os.path.abspath(__file__)
current_dir=os.path.dirname(current_path)
distance_path=os.path.join(current_dir,"reward.csv")#距离文件的路径
'length of the data'
lenth=333
f_ref=open(distance_path,'r')
y_ref=[lenth]
t=np.zeros(lenth)
num=0
y_ref=np.zeros(lenth)
with f_ref:
    reader=csv.DictReader(f_ref)
    for row in reader:
        #pdb.set_trace()
        y_ref[num]=row['Value']
        t[num]=row['Step']
        num+=1
        #print(row['Wall time'],row['Step'],row['Value'])
        if num>332:
            break


reward_max=y_ref.max()
print(reward_max)
pdb.set_trace()
a=2