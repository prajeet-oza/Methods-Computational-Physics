import numpy as np
import matplotlib.pyplot as plt

# uT = D*uXX and variables; CHANGE AS NEEDED
delX = 0.01
delT = 50
numTimeStep = 1000
minX, maxX = 0, 10
minT, maxT = 0, delT*numTimeStep
D = 1/10**4
alpha = (D * delT) / delX**2

# J for space, N for time
J = int((maxX - minX) / delX + 1)
N = int((maxT - minT) / delT + 1)

# initialize the NxJ matrix for u
u = np.zeros((N, J))

# boundary conditions (dirichlet); CHANGE AS NEEDED
minBC, maxBC = 0, 100
u[:, 0] = minBC
u[:, J-1] = maxBC

# initial conditions
x = np.arange(minX, maxX + delX, delX)
# u0 = 4 * x * (1 - x) # CHANGE AS NEEDED
# u[0] = u0.copy()

# tridiagonal matrix
p = (1 + 2*alpha) * np.ones(J-2)
q = -1 * alpha * np.ones(J-3)
mat = np.diag(p) + np.diag(q, k=1) + np.diag(q, k=-1)
inv = np.linalg.inv(mat)

# implicit method
for i in range(N-1):
	d = np.zeros(J-2)
	d[0] = u[i+1, 0]
	d[J-3] = u[i+1, J-1]
	d = alpha * d.copy()
	add = u[i, 1:J-1].T + d.T
	u[i+1, 1:J-1] = np.dot(inv, add)

#plotting code
meshX, meshT = np.meshgrid(np.arange(minX, maxX + delX, delX), np.arange(minT, maxT + delT, delT))
title = 'Implicit Finite Difference Scheme with alpha = ' + str('%.2f' % alpha)
plt.figure()
plt.contourf(meshX, meshT, u, 50, cmap='jet')
plt.colorbar()
plt.suptitle(title, fontsize=14, y=0.97)
plt.title('with Dirichlet Boundary Conditions', fontsize=13)
plt.xlabel('x; space mesh', fontsize=12)
plt.ylabel('t; time mesh || delT * numTimeStep', fontsize=12)
# plt.savefig('implicit-dirichlet-alpha-0point5.png')
plt.show()