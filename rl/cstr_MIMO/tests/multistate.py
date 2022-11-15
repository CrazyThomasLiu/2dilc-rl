import numpy as np
import pdb


a=np.array([[1,2,3],[4,5,6]])
b=np.array([[7,8,9],[10,11,12]])

a[:,0]=b[:,2]
print("a:",a)
b[:,2]=np.array((1,1))
c=np.array((1,1))
print("a:",a)
print("b:",b)
pdb.set_trace()

a=2