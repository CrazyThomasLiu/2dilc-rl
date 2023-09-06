import numpy as np
import cvxpy as cp
import pdb
"""Theorem 1 : Robust PI controller
from paper 'PI based indirect-type iterative learning control for batch processes with time-varying uncertainties: A 2D FM model based approach' Journal of process control,2019"""
'1. the dimensions of the state space'
m=3
n=1
r=1# No useful
l=1
'2. The original batch system'
A=np.array([[1.607,-0.608,-0.928],[1.,0.,0.],[0.,0.,0.]])
B=np.array([[1.239],[0.0],[1.]])
C=np.array([[1.,0.,0.]])
E=np.array([[1,0,0],[0,1,0],[0,0,1]])
F_A=np.array([[0.0804,-0.0304,-0.0464],[0.,0.,0.],[0.,0.,0.]])
F_B=np.array([[0.062],[0.],[0.]])

'3. the robust PI controller tuning'
# define the unknown varibales
D_hat=np.block([[np.eye(m)],
                [np.zeros((l,m))]])
E_hat=np.block([[E],
                [np.zeros((l,m))]
                ])
C_hat=np.block([[-C,np.zeros((l,l))]])

"4. Solving the K_P and K_I"
#setting the alpha and r from paper
alpha=0.5
r=0.45
beta_1=1-abs(alpha)
beta_2=1/((1/beta_1)-1)
P_1= cp.Variable((m,m),symmetric=True)
#P_1= cp.Variable((m,m))
P_2= cp.Variable((m,l))
P_3= cp.Variable((l,l),symmetric=True)
R_1= cp.Variable((n,m))
R_2= cp.Variable((n,l))
varepsilon=cp.Variable(1,pos=True)
gamma=cp.Variable(1,pos=True)
P=cp.bmat([[P_1, P_2],
               [P_2.T, P_3]])
A_hat_P=cp.bmat([[A@P_1-B@R_1, A@P_2-B@R_2],
               [-C@P_1+P_2.T, -C@P_2+P_3]])
P_F_hat=cp.bmat([[P_1@(F_A.T)-(R_1.T)@(F_B.T)],
                 [(P_2.T)@(F_A.T)-(R_2.T)@(F_B.T)]])
Lambda_1=-alpha*A_hat_P.T-alpha*A_hat_P+(alpha*alpha-r*r)*P+varepsilon*alpha*alpha*E_hat@E_hat.T
Lambda_2=A_hat_P.T-varepsilon*alpha*E_hat@E_hat.T
Lambda_3=-P+varepsilon*E_hat@E_hat.T

"5. Define the inequality"
inequalitymatrix=cp.bmat([[Lambda_1,np.zeros((m+l,m)),Lambda_2,P@C_hat.T,np.zeros((m+l,m+l)),P_F_hat],
                          [np.zeros((m,m+l)),-(beta_1/1)*gamma*np.eye(m),D_hat.T,np.zeros((m,l)),D_hat.T,np.zeros((m,m))],
                          [Lambda_2.T,D_hat,Lambda_3,np.zeros((m+l,l)),np.zeros((m+l,m+l)),np.zeros((m+l,m))],
                          [C_hat@P.T,np.zeros((l,m)),np.zeros((l,m+l)),-(beta_1/1)*np.eye(l),np.zeros((l,m+l)),np.zeros((l,m))],
                          [np.zeros((m+l,m+l)),D_hat,np.zeros((m+l,m+l)),np.zeros((m+l,l)),-beta_2*P,np.zeros((m+l,m))],
                          [P_F_hat.T,np.zeros((m,m)),np.zeros((m,m+l)),np.zeros((m,l)),np.zeros((m,m+l)),-varepsilon*np.eye(m)],
                            ])

'6. Define the contrains'
constraints = [inequalitymatrix << 0]
constraints += [P_1 >> 0]
constraints += [P_3 >> 0]
prob=cp.Problem(cp.Minimize(gamma),constraints)
prob.solve(verbose=True,max_iters=1000000)
result=np.block([[R_1.value,R_2.value]])@np.linalg.inv(P.value)
K_P_hat=result[0,0:3]
K_I_hat=result[0,3]
K_I=-K_I_hat
K_P=K_P_hat@C.T@(np.linalg.inv(C@C.T))+K_I_hat
print('Kp',K_P)
print('Ki',K_I)

