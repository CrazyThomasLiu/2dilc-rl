import os
import numpy as np
import csv
import pdb

time_length=200
# set the current path
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
"""1. Load data"""
figure_data_path=os.path.join(current_dir, "Figure8_original1.csv")
'get the 2D ILC-RL Control Scheme '
'get the length of the data'
f_ref=open(figure_data_path,'r')
length=len(f_ref.readlines())-1
num=0
ilc_rl=np.zeros(length)
f_ref=open(figure_data_path,'r')
with f_ref:
    reader=csv.DictReader(f_ref)
    for row in reader:
        ilc_rl[num]=row['Value']
        num+=1

'get the 2D Iterative Learning Control Scheme '
figure_data_path=os.path.join(current_dir, "Figure8_original2.csv")
'get the length of the data'
f_ref=open(figure_data_path,'r')
length=len(f_ref.readlines())-1
num=0
ilc=np.zeros(length)
f_ref=open(figure_data_path,'r')
with f_ref:
    reader=csv.DictReader(f_ref)
    for row in reader:
        ilc[num]=row['Value']
        num+=1

'get the PI-based indirect-type ILC '
figure_data_path=os.path.join(current_dir, "Figure8_original3.csv")
'get the length of the data'
f_ref=open(figure_data_path,'r')
length=len(f_ref.readlines())-1
num=0
pi_ilc=np.zeros(length)
f_ref=open(figure_data_path,'r')
with f_ref:
    reader=csv.DictReader(f_ref)
    for row in reader:
        pi_ilc[num]=row['Value']
        num+=1



with open('Figure8.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['2D ILC-RL Control Scheme','2D Iterative Learning Control Scheme','PI-based Indirect ILC'],)
    writer.writerows(map(lambda x,y,z: [x,y,z], ilc_rl,ilc,pi_ilc))