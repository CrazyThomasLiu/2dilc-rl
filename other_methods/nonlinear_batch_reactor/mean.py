import sys
import os
import pdb
import pprint
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

batch=50
"Load the data from .csv data"
PI_dir=os.path.join(current_dir, "PI_ILC_RMSE_nonlinear_batch_reactor.csv")
f_PI=open(PI_dir,'r')
num=0
rmse_pi=np.zeros(batch)
with f_PI:
    reader=csv.DictReader(f_PI)
    for row in reader:
        rmse_pi[num]=row['Value']
        num+=1

mean=np.mean(rmse_pi)
print(mean)
pdb.set_trace()
a=2