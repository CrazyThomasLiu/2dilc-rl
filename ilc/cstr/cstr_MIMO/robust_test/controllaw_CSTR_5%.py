import numpy as np
import cvxpy as cp
import pdb
import math
m=2  # dimension of the state
n=2  # input
#r=2# No useful
l=2   # output

# the known system parameters
#A = np.random.randn(m, m)
A=np.array([[0.982506552,-0.000276745177],[1.50865638,1.04539887]])

#B=np.random.randn(m, n)
B=np.array([[0.00991262171,-5.7632678*math.pow(10,-8)],[0.00750890853,0.000427851636]])

#C=np.random.randn(l,m)
C=np.array([[1.,0.],[0.,1.]])

#D=np.random.randn(m,r)
D=np.zeros((m,n))

#E1=np.random.randn(m,m)
#E1=np.array([[0.0,0.0],[0,0]])
E1=np.array([[0.05,-0.000015],[0.075,0.05]])
#F1=np.random.randn(m,m)
F1=np.eye(m)

#E2=np.random.randn(m,n)
#E2=np.array([[0.0],[0.0]])
E2=np.array([[0.0005,0.0],[0.000375,0.0000215]])
F2=np.eye(n)

#pdb.set_trace()
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
#sovle the inequality matrix
#prob.solve(solver=cp.SCS,verbose=True)
#prob.solve(verbose=True,max_iters=20000)
prob.solve(verbose=True)
#prob.solve(solver=cp.OSQP)
#prob.solve(solver=cp.ECOS)
"""
prob.solve()
print("optimal value with defaut:", prob.value)
prob.solve(solver=cp.CVXOPT)
print("optimal value with CVXOPT:", prob.value)
prob.solve(solver=cp.SCS)
print("optimal value with SCS:", prob.value)
"""
#prob.solve(verbose=True,solver=cp.CVXOPT)
#prob.solve(verbose=True,solver=cp.Dqcp2Dcp)
#print("optimal value with CVXOPT:", prob.value)
#prob.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
#print("optimal value with SciPy/HiGHS:", prob.value)
# Solve with GLOP.
#prob.solve(solver=cp.GLOP)
#print("optimal value with GLOP:", prob.value)

# Solve with GLPK.
#prob.solve(solver=cp.GLPK)
#print("optimal value with GLPK:", prob.value)

# Solve with GLPK_MI.
#prob.solve(solver=cp.GLPK_MI)
#print("optimal value with GLPK_MI:", prob.value)

# Solve with GUROBI.
#prob.solve(solver=cp.GUROBI)
#print("optimal value with GUROBI:", prob.value)

# Solve with MOSEK.
#prob.solve(solver=cp.MOSEK)
#print("optimal value with MOSEK:", prob.value)

# Solve with CBC.
#prob.solve(solver=cp.CBC)
#print("optimal value with CBC:", prob.value)

# Solve with CPLEX.
#prob.solve(solver=cp.CPLEX)
#print("optimal value with CPLEX:", prob.value)

# Solve with NAG.
#prob.solve(solver=cp.NAG)
#print("optimal value with NAG:", prob.value)

# Solve with PDLP.
#prob.solve(solver=cp.PDLP)
#print("optimal value with PDLP:", prob.value)

# Solve with SCIP.
#prob.solve(solver=cp.SCIP)
#print("optimal value with SCIP:", prob.value)

# Solve with XPRESS.
#prob.solve(solver=cp.XPRESS)
#print("optimal value with XPRESS:", prob.value)

pdb.set_trace()
K=M.value@(np.linalg.inv(Q.value))
pdb.set_trace()
a=2
delta1_elements=np.random.randn(m,1)
delta1=np.diag(delta1_elements)
delta2_elements=np.random.randn(m,1)
delta2=np.diag(delta2_elements)

