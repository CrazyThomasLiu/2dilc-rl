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
"""1. Load the rmse data"""
rl_dir=os.path.join(current_dir, "rmse.csv")
length=batch_rmse
f_rl=open(rl_dir,'r')
t=np.zeros(length)
num=0
y_rl=np.zeros(length)
with f_rl:
    reader=csv.DictReader(f_rl)
    for row in reader:
        y_rl[num]=row['Value']
        t[num]=row['Step']
        num+=1

"2. Plot of the sum RMSE"
batch_time=range(1,batch_rmse+1)
fig=plt.figure(figsize=(9.0,5.5))
font2 = {'family': 'Arial',
         'weight': 'bold',
         'size': 18,
         }
x_major_locator=MultipleLocator(int(batch_rmse/10))

plt.plot((t/300),y_rl,linewidth=1,color='tab:blue',linestyle='solid')
plt.grid()
xlable = 'Batch:$\mathit{k} $'
ylable = 'RMSE:$\mathit{I_{k}}$'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim((-0.2,9))
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
labels=ax.get_xticklabels()+ax.get_yticklabels()
[label.set_fontname('Arial') for label in labels]
plt.tick_params(axis='both',width=1.5,length=5)
bwith=1.5
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
if save_figure==True:
    plt.savefig('Nonlinear_batch_reactor_rmse.pdf')

plt.show()
