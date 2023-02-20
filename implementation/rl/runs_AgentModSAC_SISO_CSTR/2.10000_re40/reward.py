from control.matlab import *  # MATLAB-like functions
import pdb
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
current_path=os.path.abspath(__file__)
current_dir=os.path.dirname(current_path)
distance_path=os.path.join(current_dir,"reward.csv")#距离文件的路径
'length of the data'
lenth=999
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
        if num>998:
            break


#pdb.set_trace()
plt.figure()
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 15,
         }

title = 'Reward'
plt.plot((t/300),y_ref,linewidth=1.5)
#plt.title(title)
plt.grid()
xlable = 'Traning Iteartions Number'
ylable = 'Reward Mean'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
#plt.legend(['y_Ref','y_out','u'])
plt.savefig('reward.png',bbox_inches='tight',dpi=700)
plt.show()
#pdb.set_trace()
a=2