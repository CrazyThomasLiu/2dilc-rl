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
"""1. Load the 2dilc rmse"""
ILC_dir=os.path.join(current_dir, "2DILC_training_finished.csv")
f_ILC=open(ILC_dir,'r')
num=0
y_2dilc=np.zeros(batch_rmse)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_2dilc[num]=row['Value']
        num+=1

"""2. Load the 2dilc_rl rmse"""
rl_dir=os.path.join(current_dir, "2DILC-RL_training_finished.csv")
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
        #t[num]=row['Step']
        num+=1
        #print(row['Wall time'],row['Step'],row['Value'])
        #if num>333:
        #    break


# reduce the length of the rmse
y_2dilc_show=y_2dilc[0:batch_rmse]
y_2dilc_rl_show=y_rl[0:batch_rmse]
"2. Plot of the sum RMSE"
batch_time=range(1,batch_rmse+1)
fig=plt.figure(figsize=(7,5.5))
x_major_locator=MultipleLocator(int(batch_rmse/10))
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#pdb.set_trace()
plt.plot(batch_time,y_2dilc_show,linewidth=1.5,color='tab:blue',linestyle = 'dashdot')
plt.plot(batch_time,y_2dilc_rl_show,linewidth=1.5,color='tab:orange',linestyle='solid')
#plt.plot(batch_time,y_2dilc_rl_show,linewidth=2,color='black',linestyle='dotted')
plt.grid()

xlable = 'Batch:$\mathit{k} $'
ylable = 'RMSE:$\mathit{I_{k}}$'
font2 = {'family': 'Arial',
         'weight': 'bold',
         'size': 18,
         }
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.legend(['2D Iterative Learning Control Scheme','2D ILC-RL Control Scheme'])
if save_figure==True:
    #plt.savefig('Nonlinear_batch_reactor_compare_rmse.png',dpi=900)
    plt.savefig('Nonlinear_batch_reactor_compare_rmse.pdf')

plt.show()

pdb.set_trace()
a=2