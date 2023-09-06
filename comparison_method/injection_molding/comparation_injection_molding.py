import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import csv
save_figure=True
"""give the path"""
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)

batch=50
"Load the data from .csv data"
"""1. Load the 2dilc rmse"""
ILC_dir=os.path.join(current_dir, "2DILC_RMSE_injection_molding_process.csv")
f_ILC=open(ILC_dir,'r')
num=0
rmse_ilc=np.zeros(batch)
with f_ILC:
    reader=csv.DictReader(f_ILC)
    for row in reader:
        rmse_ilc[num]=row['Value']
        num+=1

"""2. Load the ilc-rl rmse"""
RL_dir=os.path.join(current_dir, "2DILC-RL_RMSE.csv")
f_RL=open(RL_dir,'r')
num=0
rmse_rl=np.zeros(batch)
with f_RL:
    reader=csv.DictReader(f_RL)
    for row in reader:
        rmse_rl[num]=row['Value']
        num+=1


"""3. Load the  PI indirect ILC rmse"""
PI_dir=os.path.join(current_dir, "PI_ILC_RMSE.csv")
f_PI=open(PI_dir,'r')
num=0
rmse_pi=np.zeros(batch)
with f_PI:
    reader=csv.DictReader(f_PI)
    for row in reader:
        rmse_pi[num]=row['Value']
        num+=1

"Plot of the comparation figure"
batch_time=range(1,batch+1)
fig=plt.figure(figsize=(9,6.5))
x_major_locator=MultipleLocator(int(batch/10))
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(batch_time,rmse_pi,linewidth=1.5,color='tab:red',linestyle='dashed')
plt.plot(batch_time,rmse_ilc,linewidth=2,color='tab:blue',linestyle = 'dashdot')
plt.plot(batch_time,rmse_rl,linewidth=2,color='tab:orange',linestyle='solid')
plt.grid()
xlable = 'Batch:$\mathit{k} $'
ylable = 'RMSE:$\mathit{I_{k}}$'
font2_rmse = {'family': 'Arial',
         'weight': 'bold',
         'size': 22,
         }
font2_legend = {'family': 'Arial',
         'weight': 'bold',
         'size': 16,
         }
plt.xlabel(xlable,font2_rmse )
plt.ylabel(ylable,font2_rmse )
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.legend(['PI-based Indirect ILC [JPC,2019]','2D Iterative Learning Controller','2D ILC-RL Control Scheme'],prop=font2_legend)
# setting the frame lines
bwith=1.5
TK=plt.gca()
TK.spines['bottom'].set_linewidth(bwith)
TK.spines['top'].set_linewidth(bwith)
TK.spines['left'].set_linewidth(bwith)
TK.spines['right'].set_linewidth(bwith)
if save_figure==True:
    plt.savefig('Injection_molding_compare_rmse.pdf')

plt.show()
