import numpy as np
import matplotlib.pyplot as plt

# taking inputs for the differential equation
print('\n ' + '#'*10 + ' ORDINARY DIFFERENTIAL EQUATION ' + '#'*10)
print("\n  For ODE of the form: A y'' + B y' + C y = D x")
A = float(input('  Enter the value of A: '))
B = float(input('  Enter the value of B: '))
C = float(input('  Enter the value of C: '))
D = float(input('  Enter the value of D: '))

# taking input for the boundary conditions for the differential equation
print('\n  With the boundary conditions, y(L) = XL, y(U) = XU')
minX = float(input('  Enter the value of L: '))
maxX = float(input('  Enter the value of U: '))
minY = float(input('  Enter the value of XL: '))
maxY = float(input('  Enter the value of XU: '))

# setting up the starting values
delX = 0.01 # delX for the increment in X
X = np.arange(minX, maxX + delX, delX) # values of X in the given interval
N = X.shape[0] # number of steps, or size of the X vector

Y = np.zeros(N) # initialize the Y vector
# applying boundary conditions to the Y vector
Y[0] = minY
Y[-1] = maxY

# from central difference formulas,
# we get the first derivative f'(x) = (f(x + dx) - f(x - dx)) / 2 * dx
# we get the second derivative f''(x) = (f(x + dx) + f(x - dx) - 2 * f(x)) / dx**2
# and applying these formula to the differential equation
# gives us coefficients for f(x), f(x + dx), f(x - dx)
# Vm1: f(x - dx), Ve0: f(x), Vp1: f(x + dx)
# also, the use of central difference gives us an error of O(dx**2)
# so, a value lower than 0.01 should give us a accuracy of 4 significant figures
Vm1 = A / delX**2 - B / (2 * delX)
Ve0 = C - 2 * A / delX**2
Vp1 = A / delX**2 + B / (2 * delX)

# creating the tridiagonal matrix with the above-mentioned coefficients
diagm1 = Vm1 * np.ones(N-3) # diagonal below the main diagonal
diage0 = Ve0 * np.ones(N-2) # main diagonal
diagp1 = Vp1 * np.ones(N-3) # diagonal above the main diagonal
# finally generating the tridiagonal matrix, for points (minX + dx) to (maxX - dx)
# so, N-2 terms, giving a (N-2) x (N-2) tridiagonal matrix
triDiagMat = np.diag(diagm1, k=-1) + np.diag(diage0) + np.diag(diagp1, k=1)
# and the first and last rows have one term each, which doesn't exactly fit in the tridiagonal matrix
# creating that as separate vector, to add to the tridiagonal dot Y vector values
# the vector is defined as (missing term, 0, 0, ... 0, 0, missing term)
# this will be of length (N-2)
addDiagMat = np.zeros((N-2, 1))
addDiagMat[0, 0] = Vm1 * Y[0]
addDiagMat[-1, 0] = Vp1 * Y[-1]

# calculating the x and y values from (minX + dx) to (maxX - dx) points
# (tridiagonal matrix dot Y vector) + missing values vector = D times X vector
# from the above equation, we can find the vector Y
# subtract take the missing values vector to the other side of the equation
# pre-multiply the inverse of tridiagonal matrix on both sides
xMat = X[1:-1].reshape(X[1:-1].shape[0], 1)
yMat = np.linalg.inv(triDiagMat) @ (D * xMat - addDiagMat)

# storing the values of Y, calculated with the above code
Y[1:-1] = yMat.ravel().copy()

# plotting the solution
plt.figure()
plt.plot(X, Y, label='Numerical Solution')
plt.title("Solving ODE: {}y'' + {}y' + {}y = {}x".format(A, B, C, D), fontsize=14)
plt.ylabel('Y variable', fontsize=14)
plt.xlabel('X variable', fontsize=14)
plt.legend()
plt.show()