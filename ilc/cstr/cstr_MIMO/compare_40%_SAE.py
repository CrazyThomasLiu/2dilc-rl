import sys
import os
config_path=os.path.split(os.path.abspath(__file__))[0]
config_path=config_path.rsplit('/',2)[0]
sys.path.append(config_path)
import pdb
import pdb
import os
import control
from control.matlab import *  # MATLAB-like functions
import torch
import pprint
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
import math
import csv
"""Plot the 3d visibale figure """
T_length=200
batch_SAE=100
save_figure=True
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 8
         }
#define the reference trajectory
y_ref=np.ones((T_length,2))
y_ref[:,0]=0.57*y_ref[:,0]
y_ref[:,1]=395*y_ref[:,1]
#pdb.set_trace()
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
"""1. Load the sqrt form the 40%"""
ILCsac_dir=os.path.join(current_dir, "2dilc_MIMO_cstr_40%_y1.csv")
f_ILC=open(ILCsac_dir,'r')
num=0
y_ILC_y1=np.zeros(100)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_ILC_y1[num]=row['Value']
        num+=1
ILCsac_dir=os.path.join(current_dir, "2dilc_MIMO_cstr_40%_y2.csv")
f_ILC=open(ILCsac_dir,'r')
num=0
y_ILC_y2=np.zeros(100)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_ILC_y2[num]=row['Value']
        num+=1
#pdb.set_trace()
"""2. Load the sqrt form the 40% with time-varying"""
ILCsac_dir=os.path.join(current_dir, "2dilc_MIMO_cstr_40%_time_y1.csv")
f_ILC=open(ILCsac_dir,'r')
num=0
y_ILC_time_y1=np.zeros(100)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_ILC_time_y1[num]=row['Value']
        num+=1
ILCsac_dir=os.path.join(current_dir, "2dilc_MIMO_cstr_40%_time_y2.csv")
f_ILC=open(ILCsac_dir,'r')
num=0
y_ILC_time_y2=np.zeros(100)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_ILC_time_y2[num]=row['Value']
        num+=1
"""10000. Load the sqrt form the 40% with time-varying"""
ILCsac_dir=os.path.join(current_dir, "2dilc_MIMO_cstr_40%_time_batch_y1.csv")
f_ILC=open(ILCsac_dir,'r')
num=0
y_ILC_time_batch_y1=np.zeros(100)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_ILC_time_batch_y1[num]=row['Value']
        num+=1
ILCsac_dir=os.path.join(current_dir, "2dilc_MIMO_cstr_40%_time_batch_y2.csv")
f_ILC=open(ILCsac_dir,'r')
num=0
y_ILC_time_batch_y2=np.zeros(100)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_ILC_time_batch_y2[num]=row['Value']
        num+=1
#pdb.set_trace()
"10000.1 Plot of the y1"
plt.subplot(2,1,1)
batch_time=range(1,batch_SAE+1)
x_major_locator=MultipleLocator(int(batch_SAE/20))
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(batch_time,y_ILC_y1,linewidth=3,color='red',linestyle=':')
plt.plot(batch_time,y_ILC_time_y1,linewidth=2,color='blue',linestyle=':')
plt.plot(batch_time,y_ILC_time_batch_y1,linewidth=2,color='green',linestyle=':')
#plt.plot(batch_time,SAE_tem,linewidth=10000,color='red',linestyle=':')
plt.grid()
xlable = 'Batch:k'
#ylable = 'Root Mean Squared Error (RMSE)'
ylable = 'RMSE'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.legend(['Normal-CSTR','Time-varying','Time and Batch-varying'])
plt.title('Production Concentration')
"10000.2 Plot of the y2"

plt.subplot(2,1,2)
x_major_locator=MultipleLocator(int(batch_SAE/20))
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(batch_time,y_ILC_y2,linewidth=3,color='red',linestyle=':')
plt.plot(batch_time,y_ILC_time_y2,linewidth=2,color='blue',linestyle=':')
plt.plot(batch_time,y_ILC_time_batch_y2,linewidth=2,color='green',linestyle=':')
plt.grid()
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.legend(['Normal-CSTR','Time-varying','Time and Batch-varying'])
plt.title('Temperature')
if save_figure==True:
    plt.savefig('Compare_40%_SAE.png',dpi=700)
plt.show()
pdb.set_trace()
a=2

