import numpy as np
import matplotlib.pyplot as plt

# uT = D*uXX and variables; CHANGE AS NEEDED
delX = 0.1
delT = 0.05
numTimeStep = 15
minX, maxX = 0, 1
minT, maxT = 0, delT*numTimeStep
D = 1.44
alpha = (D * delT) / delX**2

# J for space, N for time
J = int((maxX - minX) / delX + 1)
N = int((maxT - minT) / delT + 1)

# initialize the NxJ matrix for u
u = np.zeros((N, J))

# boundary conditions (dirichlet); CHANGE AS NEEDED
minBC, maxBC = 2, 0.5
u[:, 0] = minBC
u[:, J-1] = maxBC

# initial conditions
x = np.arange(minX, maxX + delX, delX)
u0 = 2 - 1.5 * x + np.sin(np.pi * x) # CHANGE AS NEEDED
u[0] = u0.copy()

# tridiagonal matrix
pe = (1 - alpha) * np.ones(J-2)
qe = (alpha / 2) * np.ones(J-3)
pi = (1 + alpha) * np.ones(J-2)
qi = -1 * (alpha / 2) * np.ones(J-3)
mate = np.diag(pe) + np.diag(qe, k=1) + np.diag(qe, k=-1)
mati = np.diag(pi) + np.diag(qi, k=1) + np.diag(qi, k=-1)
invi = np.linalg.inv(mati)

# crank nicolson method
for i in range(N-1):
	de = np.zeros(J-2)
	di = np.zeros(J-2)
	de[0] = u[i, 0]
	di[0] = u[i+1, 0]
	de[J-3] = u[i, J-1]
	di[J-3] = u[i+1, J-1]
	ee = (alpha / 2) * de.copy()
	ei = (alpha / 2) * di.copy()
	mul = np.dot(mate, u[i, 1:J-1].T)
	add = mul + ee.T + ei.T
	u[i+1, 1:J-1] = np.dot(invi, add)

#plotting code
meshX, meshT = np.meshgrid(np.arange(minX, maxX + delX, delX), np.arange(minT, maxT + delT, delT))
title = 'Crank Nicolson Finite Difference Scheme'
plt.figure()
plt.contourf(meshX, meshT, u, 50, cmap='jet')
plt.colorbar()
plt.suptitle(title, fontsize=14, y=0.97)
plt.title('with Dirichlet Boundary Conditions', fontsize=13)
plt.xlabel('x; space mesh', fontsize=12)
plt.ylabel('t; time mesh || delT * numTimeStep', fontsize=12)
plt.savefig('crank-nicolson-dirichlet.png')
# plt.show()