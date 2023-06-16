import os
import pdb
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
batch_rmse=2000
save_figure=True
# set the current path
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
"""1. Load the 2dilc_rl rmse"""
rl_dir=os.path.join(current_dir, "rmse.csv")
length=batch_rmse+1
f_rl=open(rl_dir,'r')
#y_rl=[lenth]
t=np.zeros(length)
num=0
y_rl=np.zeros(length)
with f_rl:
    reader=csv.DictReader(f_rl)
    for row in reader:
        #pdb.set_trace()
        y_rl[num]=row['Value']
        t[num]=row['Step']
        num+=1
        #print(row['Wall time'],row['Step'],row['Value'])
        #if num>333:
        #    break

reward_min=y_rl.min()
print(reward_min)
pdb.set_trace()
a=2