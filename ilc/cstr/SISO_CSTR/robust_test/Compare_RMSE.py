import pdb
import os
import control
from control.matlab import *  # MATLAB-like functions
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
import math
import csv
"""give the path"""
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
#pdb.set_trace()
batch=50
"""1.load the sqrt from the RMSE60%"""
ILCorigin_dir=os.path.join(current_dir, "RMSE_60%.csv")
f_ILC=open(ILCorigin_dir,'r')
num=0
y_ILC_60=np.zeros(200)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_ILC_60[num]=row['Value']
        num+=1
y_ILC_60_show=y_ILC_60[:batch]
#pdb.set_trace()
"""2.load the sqrt from the RMSE80%"""
ILCorigin_dir=os.path.join(current_dir, "RMSE_80%.csv")
f_ILC=open(ILCorigin_dir,'r')
num=0
y_ILC_80=np.zeros(200)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_ILC_80[num]=row['Value']
        num+=1
y_ILC_80_show=y_ILC_80[:batch]
#pdb.set_trace()
"""10000.load the sqrt from the 90%"""
ILCorigin_dir=os.path.join(current_dir, "RMSE_90%.csv")
f_ILC=open(ILCorigin_dir,'r')
num=0
y_ILC_90=np.zeros(200)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_ILC_90[num]=row['Value']
        num+=1
y_ILC_90_show=y_ILC_90[:batch]
#pdb.set_trace()
#pdb.set_trace()

plt.figure()
batch_time=range(batch)
x_major_locator=MultipleLocator(1)
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(batch_time,y_ILC_60_show,linewidth=1.5,color='red',linestyle='--')
plt.plot(batch_time,y_ILC_80_show,linewidth=1.5,linestyle='--')
plt.plot(batch_time,y_ILC_90_show,linewidth=1.5,linestyle='--')
#plt.plot(batch_time,SAE,linewidth=2)
#pdb.set_trace()
#plt.plot(batch_time,y_ILC_show,linewidth=1,color='red')
#plt.plot(batch_time,y_ILC_show,linewidth=1.5,color='black',linestyle=':')
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14
         }
plt.grid()
xlable = 'Batch:k '
ylable = 'Root Mean Squared Error (RMSE)'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
#plt.legend(['None','Fixed Parameters-varying','Time-sinus','Batch-sinus'])
plt.legend(['60%','80%','90%'])
plt.savefig('Compare_RMSE.png',dpi=700)
plt.show()



pdb.set_trace()
a=2

