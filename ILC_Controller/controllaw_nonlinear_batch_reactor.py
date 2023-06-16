import numpy as np
import cvxpy as cp
import pdb
import math
m=2  # dimension of the state
n=1  # input
#r=2# No useful
l=1   # output

# the known system parameters
A=np.array([[ 0.989771222, -1.10609469*math.pow(10,-5)],
            [ 0.0557223632,  0.992262023]],)
#B=np.random.randn(m, n)
B=np.array([[-2.32098291*math.pow(10,-9)],
                [ 4.16789297*math.pow(10,-4)]],)
#C=np.random.randn(l,m)
C=np.array([[0.,1.]])
#D=np.random.randn(m,r)
D=np.zeros((m,n))

E1=np.array([[1,-1.1*math.pow(10,-5)],[0.056,1]])
E1=E1*0.8
F1=np.eye(m)

E2=np.array([[-2.32*math.pow(10,-9)],[4.17*math.pow(10,-4)]])
E2=E2*0.8
F2=np.eye(n)

A_bar=np.block([
    [A,np.zeros((m,l))],
    [-C@A,np.eye(l)]
    ])
B_bar=np.block([[B],[-C@B]])
D_bar=np.block([[np.eye(m)],[-C]])
E1_bar=np.block([[E1],[-C@E1]])
F1_bar=np.block([[F1,np.zeros((m,l))]])
E2_bar=np.block([[E2],[-C@E2]])
F2_bar=F2

# set the hyperparameters
alpha=0.8
beta=0.8
C_bar=np.ones((1,(m+l)))
# set the variables
Q_h= cp.Variable((m,m),symmetric=True)
Q_v= cp.Variable((l,l),symmetric=True)
#Q=np.block([[Q_h,np.zeros((m,l))],[np.zeros((l,m)),Q_v]])
Q=cp.bmat([[Q_h,np.zeros((m,l))],[np.zeros((l,m)),Q_v]])
Q_albe=cp.bmat([[alpha*Q_h,np.zeros((m,l))],[beta*np.zeros((l,m)),Q_v]])
M= cp.Variable((n,m+l))
sigma1=cp.Variable(1,pos=True)
sigma2=cp.Variable(1,pos=True)
gamma=cp.Variable(1,pos=True)
a=np.zeros((1,m))
b=np.zeros((m,1))

#
if n==1:
    inmatrix_sigma2=cp.reshape(-sigma2,shape=[1,1])

else:
    inmatrix_sigma2=-sigma2*np.eye(n)

inequalitymatrix=cp.bmat([
    [-Q_albe,   Q@A_bar.T+M.T@B_bar.T,  Q@F1_bar.T, M.T@F2_bar.T,   Q@C_bar.T,  np.zeros((m+l,m))],
    [A_bar@Q.T+B_bar@M, -Q+sigma1*E1_bar@E1_bar.T+sigma2*E2_bar@E2_bar.T, np.zeros((m+l,m)),    np.zeros((m+l,n)), np.zeros((m+l,1)), D_bar],
    [F1_bar@Q.T,    np.zeros((m,m+l)),  -sigma1*np.eye(m),  np.zeros((m,n)),    np.zeros((m,1)),    np.zeros((m,m))],
    [F2_bar@M,  np.zeros((n,m+l)),  np.zeros((n,m)),   inmatrix_sigma2,    np.zeros((n,1)),   np.zeros((n,m))  ],
    [C_bar@Q.T,  np.zeros((1,m+l)), np.zeros((1,m)),    np.zeros((1,n)),    cp.reshape(-gamma,shape=[1,1]),   np.zeros((1,m))],
    [np.zeros((m,m+l)), D_bar.T,    np.zeros((m,m)),    np.zeros((m,n)),    np.zeros((m,1)),    -gamma*np.eye(m) ]

])
#pdb.set_trace()
# define the constrains
constraints = [Q >> 0]
constraints += [Q_albe >> 0]
constraints += [inequalitymatrix << 0]
#define the problem
prob=cp.Problem(cp.Minimize(gamma),constraints)

prob.solve(verbose=True,max_iters=1000000)

K=M.value@(np.linalg.inv(Q.value))
print(K)
