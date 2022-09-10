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
batch=20
"""1.load the sqrt form the origin"""
ILCorigin_dir=os.path.join(current_dir, "origin.csv")
f_ILC=open(ILCorigin_dir,'r')
num=0
y_ILC_origin=np.zeros(200)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_ILC_origin[num]=row['Value']
        num+=1
y_ILC_origin_show=y_ILC_origin[:20]
#pdb.set_trace()
"""2.load the sqrt form the fixedparameters"""
ILCfixed_dir=os.path.join(current_dir, "fixedparameters.csv")
f_ILC=open(ILCfixed_dir,'r')
num=0
y_ILC_fixed=np.zeros(200)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_ILC_fixed[num]=row['Value']
        num+=1
y_ILC_fixed_show=y_ILC_fixed[:20]
#pdb.set_trace()
#pdb.set_trace()
"""3.load the sqrt form the timesinus"""
ILCtimesinus_dir=os.path.join(current_dir, "timesinus.csv")
f_ILC=open(ILCtimesinus_dir,'r')
num=0
y_ILC_timesinus=np.zeros(200)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_ILC_timesinus[num]=row['Value']
        num+=1
y_ILC_timesinus_show=y_ILC_timesinus[:20]
#pdb.set_trace()

"""4.load the sqrt form the batchsinus"""
ILCbatchsinus_dir=os.path.join(current_dir, "bathsinus.csv")
f_ILC=open(ILCbatchsinus_dir,'r')
num=0
y_ILC_batchsinus=np.zeros(200)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_ILC_batchsinus[num]=row['Value']
        num+=1
y_ILC_batchsinus_show=y_ILC_batchsinus[:20]
#pdb.set_trace()

plt.figure()
batch_time=range(batch)
x_major_locator=MultipleLocator(1)
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(batch_time,y_ILC_origin_show,linewidth=1.5,color='red',linestyle='--')
plt.plot(batch_time,y_ILC_fixed_show,linewidth=1.5,linestyle='--')
plt.plot(batch_time,y_ILC_timesinus_show,linewidth=1.5,linestyle='--')
plt.plot(batch_time,y_ILC_batchsinus_show,linewidth=1.5,linestyle='--')
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
plt.legend(['None','Fixed Parameters-varying','Time-sinus','Batch-sinus'])
plt.savefig('Compare_RMSE.png',dpi=700)
plt.show()



pdb.set_trace()
a=2

