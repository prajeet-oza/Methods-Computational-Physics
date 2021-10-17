import numpy as np
import matplotlib.pyplot as plt

# uT = D*uXX and variables; CHANGE AS NEEDED
delX = 0.1
delT = 0.01
numTimeStep = 30
minX, maxX = 0, 1
minT, maxT = 0, delT*numTimeStep
D = 0.5/10**0
alpha = (D * delT) / delX**2

# J for space, N for time
J = int((maxX - minX) / delX + 1)
N = int((maxT - minT) / delT + 1)

# initialize the NxJ matrix for u
u = np.zeros((N, J))

# boundary conditions (dirichlet); CHANGE AS NEEDED
minBC, maxBC = 0, 0
u[:, 0] = minBC
u[:, J-1] = maxBC

# initial conditions
x = np.arange(minX, maxX + delX, delX)
u0 = 4 * x * (1 - x) # CHANGE AS NEEDED
u[0] = u0.copy()

# tridiagonal matrix
p = (1 - 2*alpha) * np.ones(J-2)
q = alpha * np.ones(J-3)
mat = np.diag(p) + np.diag(q, k=1) + np.diag(q, k=-1)

# implicit method
for i in range(N-1):
	d = np.zeros(J-2)
	d[0] = u[i, 0]
	d[J-3] = u[i, J-1]
	e = alpha * d.copy()
	mul = np.dot(mat, u[i, 1:J-1].T)
	u[i+1, 1:J-1] = mul + e.T
# print(u)
#plotting code
meshX, meshT = np.meshgrid(np.arange(minX, maxX + delX, delX), np.arange(minT, maxT + delT, delT))
title = 'Explicit Finite Difference Scheme with alpha = ' + str('%.2f' % alpha)
plt.figure()
plt.contourf(meshX, meshT, u, 50, cmap='jet')
plt.colorbar()
plt.suptitle(title, fontsize=14, y=0.97)
plt.title('with Dirichlet Boundary Conditions', fontsize=13)
plt.xlabel('x; space mesh', fontsize=12)
plt.ylabel('t; time mesh || delT * numTimeStep', fontsize=12)
# plt.savefig('explicit-dirichlet-alpha-1point0.png')
plt.show()