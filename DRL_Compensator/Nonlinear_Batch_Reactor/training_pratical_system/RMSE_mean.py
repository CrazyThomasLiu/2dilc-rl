import os
import pdb
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
batch_rmse=50
save_figure=True
# set the current path
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
"""1. Load the 2dilc_rl rmse training finished"""
rl_dir=os.path.join(current_dir, "2DILC-RL_training_finished.csv")
length=batch_rmse
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
        num+=1
        #print(row['Wall time'],row['Step'],row['Value'])
        #if num>333:
        #    break
ilc_rl_mean=y_rl.mean()
print(ilc_rl_mean)
#pdb.set_trace()
"""2. Load the 2dilc rmse training finished"""
rl_dir=os.path.join(current_dir, "2DILC_training_finished.csv")
length=batch_rmse
f_ilc=open(rl_dir,'r')
#y_rl=[lenth]
t=np.zeros(length)
num=0
y_ilc=np.zeros(length)
with f_ilc:
    reader=csv.DictReader(f_ilc)
    for row in reader:
        #pdb.set_trace()
        y_ilc[num]=row['Value']
        num+=1
        #print(row['Wall time'],row['Step'],row['Value'])
        #if num>333:
        #    break
ilc_mean=y_ilc.mean()
print(ilc_mean)



pdb.set_trace()
a=2