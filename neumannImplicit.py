import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# uT = D*uXX and variables; CHANGE AS NEEDED
delX = 0.1
delT = 0.05
numTimeStep = 200
minX, maxX = -5, 5
minT, maxT = 0, delT*numTimeStep
D = 1
alpha = (D * delT) / delX**2

# J for space, N for time
J = int((maxX - minX) / delX + 1)
N = int((maxT - minT) / delT + 1)

# initialize the NxJ matrix for u
u = np.zeros((N, J))

# boundary conditions (neumann); CHANGE AS NEEDED
minBC, maxBC = 0, 0

# initial conditions
x = np.arange(minX, maxX + delX, delX)
u0 = np.exp(-x**2) # CHANGE AS NEEDED
u[0] = u0.copy()

# tridiagonal matrix
p = (1 + 2*alpha) * np.ones(J)
q = -1 * alpha * np.ones(J-1)
mat = np.diag(p) + np.diag(q, k=1) + np.diag(q, k=-1)
mat[0, 1] = -2 * alpha
mat[J-1, J-2] = -2 * alpha
inv = np.linalg.inv(mat)

# implicit method
for i in range(N-1):
	d = np.zeros(J)
	d[0] = -2 * delX * minBC
	d[J-1] = 2 * delX * maxBC
	e = alpha * d.copy()
	add = u[i].T + e.T
	u[i+1] = np.dot(inv, add)

#plotting code
meshX, meshT = np.meshgrid(np.arange(minX, maxX + delX, delX), np.arange(minT, maxT + delT, delT))
title = 'Implicit Finite Difference Scheme'
plt.figure()
plt.contourf(meshX, meshT, u, 50, cmap='jet')
plt.colorbar()
plt.suptitle(title, fontsize=14, y=0.97)
plt.title('with Neumann Boundary Conditions', fontsize=13)
plt.xlabel('x; space mesh', fontsize=12)
plt.ylabel('t; time mesh || delT * numTimeStep', fontsize=12)
plt.savefig('implicit-neumann.png')
# plt.show()