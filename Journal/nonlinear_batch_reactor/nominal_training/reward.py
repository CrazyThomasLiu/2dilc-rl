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


#pdb.set_trace()
fig=plt.figure(figsize=(9.0,5.5))
#plt.rcParams['figure.figsize']=(20.0,4.0)
#plt.rcParams['savefig.dpi']=200
plt.rcParams['figure.dpi']=200
font2 = {'family': 'Arial',
         'weight': 'bold',
         'size': 18,
         }

#plt.plot((t/200),y_ref,linewidth=2,color='tab:cyan',linestyle='solid',marker='D')
plt.plot((t/300),y_ref,linewidth=2,color='tab:orange',linestyle='solid')
#plt.title(title)
plt.grid()
xlable = 'Training Batch Number'
ylable = 'Average Reward per Batch'
plt.xlabel(xlable,font2)
plt.ylabel(ylable,font2 )
#plt.legend(['y_Ref','y_out','u'])
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
ax=plt.gca()
labels=ax.get_xticklabels()+ax.get_yticklabels()
#pdb.set_trace()
[label.set_fontname('Arial') for label in labels]
plt.tick_params(axis='both',width=1.5,length=5)
# 设置图框线粗细
bwith=1.5
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
#plt.savefig('Nonlinear_batch_reactor_reward.png',dpi=900)
plt.savefig('Nonlinear_batch_reactor_reward.pdf')
plt.show()
#pdb.set_trace()
a=2