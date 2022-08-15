from control.matlab import *  # MATLAB-like functions
import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os
import csv
import math
import pandas as pd
current_path=os.path.abspath(__file__)
current_dir=os.path.dirname(current_path)
distance_path=os.path.join(current_dir,"discreteformatlab.csv")#距离文件的路径

f_matlab=open(distance_path,'r')
num=0
y_matlab=np.zeros(180*20)
with f_matlab:
    reader=csv.DictReader(f_matlab)
    for row in reader:
        #pdb.set_trace()
        y_matlab[num]=row['Value']
        num+=1
        #print(row['Wall time'],row['Step'],row['Value'])
#pdb.set_trace()
############################################333
distance_path=os.path.join(current_dir,"pythondiscrtesystem.csv")#距离文件的路径

f_python=open(distance_path,'r')
num=0
y_python=np.zeros(180*20)
with f_python:
    reader=csv.DictReader(f_python)
    for row in reader:
        #pdb.set_trace()
        y_python[num]=row['Value']
        num+=1

T = np.linspace(0, 3.5,36)
#pdb.set_trace()
plt.figure()
y_matlab_show=y_matlab[:36]
y_python_show=y_python[:36]
#pdb.set_trace()
plt.plot(T,y_matlab_show,linewidth=1,color='black',linestyle=':')
#pdb.set_trace()
plt.plot(T,y_python_show,linewidth=1,color='red')

#plt.plot(t,y_out_s,'--',color='red')
#plt.plot(t,action_s)
#plt.title(title)
x_major_locator=MultipleLocator(1)
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)

#刻度间隔
#########################
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 15,
         }
plt.grid()
xlable = 'time:t '
ylable = 'Response'
title = 'Simulation Comparison between Matlab and Python'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.legend(['Matlab','Python'])
plt.title(title)
plt.savefig('compare_discretesystem.png',dpi=600)
plt.show()
#pdb.set_trace()
a=2