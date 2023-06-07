import os
import pdb
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
batch=50
time_length=200
length=10000
save_figure=True
time_section=149
# set the current path
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
"""1. Load the rl input"""
rl_dir=os.path.join(current_dir, "Injection_molding_rl_input.csv")
f = open(rl_dir, 'r')
t=np.zeros(length)
num=0
rl_input=np.zeros(length)
with f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        for item in row:
            #pdb.set_trace()
            rl_input[num]=float(item)
            num+=1

"""2. set the time transaction"""
rl_input_show=np.zeros(batch)
for item in range(batch):
    rl_input_show[item]=rl_input[200*item+time_section]


#pdb.set_trace()
"3. Plot of time transaction"
batch_time=range(1,batch+1)
fig=plt.figure(figsize=(9.0,5.5))
font2 = {'family': 'Arial',
         'weight': 'bold',
         'size': 18,
         }
x_major_locator=MultipleLocator(int(batch/10))

plt.plot(batch_time,rl_input_show,linewidth=1.5,color='black',linestyle='solid')
plt.grid()
# plot the sin function

#x=np.arange(1,batch+1)
#y=0.5*np.sin(np.pi*x/5)

#plt.plot(x,y)


xlable = 'Batch:$\mathit{k} $'
ylable = 'DRL Compensation Signal:$\mathit{u_{k,150}}$'
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
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

if save_figure==True:
    #plt.savefig('Linear_injection_molding_rmse.png',dpi=900)
    plt.savefig('Linear_injection_molding_rl_input_time150.pdf')
plt.show()

pdb.set_trace()
a=2