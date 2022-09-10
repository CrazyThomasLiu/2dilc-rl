import numpy as np
import matplotlib.pyplot as plt
import csv
import pdb
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
import os
"""give the path"""
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)

"""load the training from the csv date"""
data_dir=os.path.join(current_dir, "episodemeanreward.csv")
f_ILC=open(data_dir,'r')
num=0
reward=np.zeros(2000)
step=np.zeros(2000)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        reward[num]=row['Value']
        step[num] = row['Step']
        num+=1
reward_show=reward[:600-1]
step_show=step[:600-1]
#pdb.set_trace()
plt.figure()
#batch_time=range(batch)
#x_major_locator=MultipleLocator(1)
#ax=plt.gca()
#ax为两条坐标轴的实例
#ax.xaxis.set_major_locator(x_major_locator)
plt.plot(step_show,reward_show)

plt.grid()
xlable = 'Step '
ylable = 'Episode Mean Reward'

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14
         }
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
#plt.legend(['SAC-based 2D feedback Controller','P-ILC Controller'])
#plt.savefig('SAEforRMES.png',dpi=600)
plt.savefig('trainingCurve.png',dpi=700)
plt.show()
pdb.set_trace()
a=2