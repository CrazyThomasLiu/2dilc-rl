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
"""1. Load the 2dilc rmse"""
ILC_dir=os.path.join(current_dir, "2dilc_batch2000.csv")
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
"""
ILC_dir=os.path.join(current_dir, "2dilc_rl.csv")
f_ILC=open(ILC_dir,'r')
num=0
y_2dilc_rl=np.zeros(batch_rmse)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        #pdb.set_trace()
        y_2dilc_rl[num]=row['Value']
        num+=1
"""
rl_dir=os.path.join(current_dir, "2dilc_rl.csv")
lenth=batch_rmse+1
f_rl=open(rl_dir,'r')
#y_rl=[lenth]
t=np.zeros(lenth)
num=0
y_rl=np.zeros(lenth)
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


# reduce the length of the rmse
y_2dilc_show=y_2dilc[0:batch_rmse]
#y_2dilc_rl_show=y_2dilc_rl[0:batch_rmse]
"10000. Plot of the sum RMSE"
batch_time=range(1,batch_rmse+1)
plt.figure()
x_major_locator=MultipleLocator(int(batch_rmse/20))
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(batch_time,y_2dilc_show,linewidth=2,linestyle = '-.')
plt.plot((t/200),y_rl,linewidth=1.5)
#plt.plot(batch_time,y_2dilc_rl_show,linewidth=2,color='black',linestyle='dotted')
plt.grid()

xlable = 'Batch:k'
ylable = 'Root Mean Squared Error (RMSE)'
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14
         }
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.legend(['2D Iterative Learning Controller','2D ILC-RL Control Scheme'])
if save_figure==True:
    plt.savefig('Control_Performance_rmse.png',dpi=700)

plt.show()

pdb.set_trace()
a=2