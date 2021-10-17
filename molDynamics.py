import numpy as np
import matplotlib.pyplot as plt

########## FUNCTIONS USED THROUGH THE CODE ##########

# defining the fcc system, i.e., with 4 times n**3 atoms
def fccSystem(n, N, A, sigma):
	# n = times A for super cell
	# N = number of atoms
	x_mesh, y_mesh, z_mesh = np.meshgrid(np.arange(0, n * A, A), np.arange(0, n * A, A), np.arange(0, n * A, A))
	x_mesh = x_mesh.reshape(-1)
	y_mesh = y_mesh.reshape(-1)
	z_mesh = z_mesh.reshape(-1)

	# initializing the r vector with initial positions of atoms in fcc
	rvec = np.zeros((N, 3))
	for i in range(4):
		st = i * N // 4
		en = (i+1) * N // 4
		if i == 0:
			a, b, c = 0.0, 0.0, 0.0
		elif i == 1:
			a, b, c = 0.5, 0.5, 0.0
		elif i == 2:
			a, b, c = 0.5, 0.0, 0.5
		else:
			a, b, c = 0.0, 0.5, 0.5

		# fcc observe a unit cell with 4 atoms, with orthogonal lattice vectors
		# atom 1: 0.0, 0.0, 0.0
		# atom 2: 0.5, 0.5, 0.0
		# atom 3: 0.5, 0.0, 0.5
		# atom 4: 0.0, 0.5, 0.5
		# above coordinates in fractions
		rvec[st:en, 0] = x_mesh + a * A * np.ones(x_mesh.shape)
		rvec[st:en, 1] = y_mesh + b * A * np.ones(y_mesh.shape)
		rvec[st:en, 2] = z_mesh + c * A * np.ones(z_mesh.shape)
	# reduced r vector, i.e., dividing by sigma
	rvec_rv = rvec.copy() / sigma

	return rvec_rv

# initializing the velocity of atoms
# this based on the maxwell-boltzmann distribution
# the distribution is a gaussian distribution with zero mean, and kT/m variance
# the reduced form of this distribution will be a gaussian distribution
# and this reduced distribution with be zero mean, and variance equal to reduced temperature
def velSystem(N, t_rv):
	vvec_rv = np.sqrt(t_rv) * np.random.randn(N, 3)
	return vvec_rv

# calculate the position vectors and their magnitudes for a pair of i and j atoms
def rijPairs(rvec_rv, N):
	rijvec_rv = np.zeros((N, N, 3))
	rijmag_rv = np.zeros((N, N))

	for i in range(N):
		for j in range(i+1, N):
			rijvec_rv[i, j] = rvec_rv[i] - rvec_rv[j]
			rijmag_rv[i, j] = np.sqrt(rijvec_rv[i, j, 0]**2 + rijvec_rv[i, j, 1]**2 + rijvec_rv[i, j, 2]**2)

			rijvec_rv[j, i] = - rijvec_rv[i, j]
			rijmag_rv[j, i] = rijmag_rv[i, j]

	return rijvec_rv, rijmag_rv

# defining the reduced form of Lennard-Jones potential
def LJPotential(r_rv):
	phi_rv = 4 * (1 / r_rv**12) - 4 * (1 / r_rv**6)
	return phi_rv

# defining the reduced form of the first derivative of Lennard-Jones potential
def LJdPotential(r_rv):
	dphi_rv = - 48 * (1 / r_rv**13) + 24 * (1 / r_rv**7)
	return dphi_rv

# defining the reduced form of interactions between a pair of i and j atoms
def LJForce(r_rv, rvec_rv):
	unitvec_rv = rvec_rv / r_rv
	const_rv = - 48 * (1 / r_rv**13) + 24 * (1 / r_rv**7)
	fvec_rv = const_rv * unitvec_rv
	return fvec_rv

# calculate f_ij vector and phi_ij from the Lennard-Jones
def LJPairs(rijvec_rv, rijmag_rv, N):
	fijvec_rv = np.zeros((N, N, 3))
	phiij_rv = np.zeros((N, N))

	for i in range(N):
		for j in range(i+1, N):
			fijvec_rv[i, j] = - LJForce(rijmag_rv[i, j], rijvec_rv[i, j])
			phiij_rv[i, j] = LJPotential(rijmag_rv[i, j])

			fijvec_rv[j, i] = - fijvec_rv[i, j]
			phiij_rv[j, i] = phiij_rv[i, j]

	return fijvec_rv, phiij_rv

# calculate Force vectors
def Force(fijvec_rv, N):
	force_rv = np.zeros((N, 3))

	for i in range(N):
		for j in range(N):
			force_rv[i] += fijvec_rv[i, j]

	return force_rv

# calculate the reduced potential energy
def U(phiij_rv, N):
	u_rv = 0.5 * phiij_rv.sum()
	return u_rv

# calculate the reduced kinetic energy
def K(vvec_rv, N):
	k_rv = 0
	for i in range(N):
		k_rv += 0.5 * (vvec_rv[i, 0]**2 + vvec_rv[i, 1]**2 + vvec_rv[i, 2]**2)

	return k_rv

# calculate the reduced temperature
def T(k_rv, N):
	t_rv = 2 * k_rv / (3 * N)
	return t_rv

# calculate the reduced pressure
def P(rijmag_rv, N, vol_rv, t_rv):
	sum2 = 0

	for i in range(N):
		for j in range(i+1, N):
			sum2 += rijmag_rv[i, j] * LJdPotential(rijmag_rv[i, j])

	p_rv = N * t_rv / vol_rv - sum2 / (3 * vol_rv)

	return p_rv

# applying velocity verlet algorithm for position
def verletPosition(rT_rv, vT_rv, aT_rv, dT_rv):
	# naming convention for variables used:
	# T ---> current state
	# TmdT ---> current minus dT state
	# TpdT ---> current plus dT state
	# also, _rv after variables sugggests reduced variable
	rTpdT_rv = rT_rv + vT_rv * dT_rv + 0.5 * aT_rv * (dT_rv**2)
	return rTpdT_rv

# applying velocity verlet algorithm for velocity
def verletVelocity(vT_rv, aT_rv, aTpdT_rv, dT_rv):
	vTpdT_rv = vT_rv + 0.5 * (aT_rv + aTpdT_rv) * dT_rv
	return vTpdT_rv
# it is to be noted that velocity verlet algorithm was applied
# this is favoured in place of verlet algorithm
# this favouring is based the executionability of the algorithm with greater memory requirement
# with the addition calculation of r(t + 2dt) in verlet algorithm to find velocity for every step
# this added calculation step seemed unnecessary in comparison to velocity verlet

########## MAIN CODE ##########

# defining staring variables
n_lattice = 3 # number of unit cells in one direction to form a super cell
rho_rv = 0.55 # reduced density
dT_rv = 0.005 # reduced time step
ti_rv = 0.2 # reduced initial temperature
A = 5 # lattice parameter
n_equi = 4000 # number of equilibrium steps
n_calc = 20000 # number of calculation steps

sigma = A * np.cbrt(rho_rv / 4) # sigma, reduction variable, units of distance
N = 4 * n_lattice**3 # number of atoms
vol_rv = (n_lattice * A / sigma)**3 # reduced volume

# of the starting variables and some related calculations:
# 1) the use of A variable is not necessary as it eventually gets eliminated from all the equations,
# 		though it has been used just for the sake of completeness

rT0_rv = fccSystem(n_lattice, N, A, sigma) # intialise the fcc system
rijT0_rv, rijT0mag_rv = rijPairs(rT0_rv, N) # i-j pairs, vectors and their magnitude
fijT0_rv, phiijT0_rv = LJPairs(rijT0_rv, rijT0mag_rv, N) # f-ij and phi_ij pairs from LJ
ForceT0_rv = Force(fijT0_rv, N) # force vector, current state

vT0_rv = velSystem(N, ti_rv) # initialise the velocity vectors
aT0_rv = ForceT0_rv.copy() # reduced acceleration equals reduced force, as reduced mass = 1

# variables to store quantities of interest during the equilibrium steps
# mostly for the purpose of plotting them
k_equi = np.zeros(n_equi + 1)
u_equi = np.zeros(n_equi + 1)
e_equi = np.zeros(n_equi + 1)
t_equi = np.zeros(n_equi + 1)
p_equi = np.zeros(n_equi + 1)

# variables to store the quantities of interest during the calculation steps
k_calc = np.zeros(n_calc + 1)
u_calc = np.zeros(n_calc + 1)
e_calc = np.zeros(n_calc + 1)
t_calc = np.zeros(n_calc + 1)
p_calc = np.zeros(n_calc + 1)

# calculate the initial values for the quantities, of course, reduced values
k_equi[0] = K(vT0_rv, N)
u_equi[0] = U(phiijT0_rv, N)
e_equi[0] = k_equi[0] + u_equi[0]
t_equi[0] = T(k_equi[0], N)
p_equi[0] = P(rijT0mag_rv, N, vol_rv, t_equi[0])

# continuing the values from equilibrium to calculation, mostly to plot them
k_calc[0] = k_equi[-1]
u_calc[0] = u_equi[-1]
e_calc[0] = e_equi[-1]
t_calc[0] = t_equi[-1]
p_calc[0] = p_equi[-1]

# viewing the initial values
print(' Starting KE: ', k_equi[0])
print(' Starting PE: ', u_equi[0])
print(' Starting TE: ', e_equi[0])
print(' Starting Temp: ', t_equi[0])
print(' Starting Pressure: ', p_equi[0])
print('\n')

# starting the equilibrium steps
print('#'*10 + ' EQUILIBRIUM STEPS OF MD ' + '#'*10)
for i in range(n_equi):
	# find positions for next time step
	rT0pdT_rv = verletPosition(rT0_rv, vT0_rv, aT0_rv, dT_rv)
	
	# calculate i-j pair quantities
	rijT0pdT_rv, rijT0pdTmag_rv = rijPairs(rT0pdT_rv, N)
	fijT0pdT_rv, phiijT0pdT_rv = LJPairs(rijT0pdT_rv, rijT0pdTmag_rv, N)
	# forces and accelerations for the next step
	ForceT0pdT_rv = Force(fijT0pdT_rv, N)
	aT0pdT_rv = ForceT0pdT_rv.copy()
	# velocity for the next step
	vT0pdT_rv = verletVelocity(vT0_rv, aT0_rv, aT0pdT_rv, dT_rv)

	# calculate energies, temperature and pressure
	k_equi[i+1] = K(vT0pdT_rv, N)
	u_equi[i+1] = U(phiijT0pdT_rv, N)
	e_equi[i+1] = k_equi[i+1] + u_equi[i+1]
	t_equi[i+1] = T(k_equi[i+1], N)
	p_equi[i+1] = P(rijT0pdTmag_rv, N, vol_rv, t_equi[i+1])

	# updating the position, velocity and acceleration for the next loop / iteration
	rT0_rv = rT0pdT_rv.copy()
	vT0_rv = vT0pdT_rv.copy()
	aT0_rv = aT0pdT_rv.copy()

	if (i+1) % 100 == 0:
		print('  MD Equilibrium Step {}... DONE!\n'.format(i+1))

# viewing values after equilibrium steps
print(' After Equilibrium, KE: ', k_equi[-1])
print(' After Equilibrium, PE: ', u_equi[-1])
print(' After Equilibrium, TE: ', e_equi[-1])
print(' After Equilibrium, Temp: ', t_equi[-1])
print(' After Equilibrium, Pressure: ', p_equi[-1])

# continuing the values from equilibrium to calculation, mostly to plot them
k_calc[0] = k_equi[-1]
u_calc[0] = u_equi[-1]
e_calc[0] = e_equi[-1]
t_calc[0] = t_equi[-1]
p_calc[0] = p_equi[-1]

# starting the calculation steps
print('#'*10 + ' CALCULATION STEPS OF MD ' + '#'*10)
for i in range(n_calc):
	rT0pdT_rv = verletPosition(rT0_rv, vT0_rv, aT0_rv, dT_rv)
	
	rijT0pdT_rv, rijT0pdTmag_rv = rijPairs(rT0pdT_rv, N)
	fijT0pdT_rv, phiijT0pdT_rv = LJPairs(rijT0pdT_rv, rijT0pdTmag_rv, N)
	ForceT0pdT_rv = Force(fijT0pdT_rv, N)
	aT0pdT_rv = ForceT0pdT_rv.copy()

	vT0pdT_rv = verletVelocity(vT0_rv, aT0_rv, aT0pdT_rv, dT_rv)

	k_calc[i+1] = K(vT0pdT_rv, N)
	u_calc[i+1] = U(phiijT0pdT_rv, N)
	e_calc[i+1] = k_calc[i+1] + u_calc[i+1]
	t_calc[i+1] = T(k_calc[i+1], N)
	p_calc[i+1] = P(rijT0pdTmag_rv, N, vol_rv, t_calc[i+1])

	rT0_rv = rT0pdT_rv.copy()
	vT0_rv = vT0pdT_rv.copy()
	aT0_rv = aT0pdT_rv.copy()

	if (i+1) % 500 == 0:
		print('  MD Calculation Step {}... DONE!\n'.format(i+1))

# viewing the average values, after the entire MD calculations is over
print(' Average KE: ', k_calc.mean())
print(' Average PE: ', u_calc.mean())
print(' Average TE: ', e_calc.mean())
print(' Average Temp: ', t_calc.mean())
print(' Average Pressure: ', p_calc.mean())

# plotting the reduced energies, temperature and pressure at various stages
plt.figure()
plt.plot(k_equi / N, label='K* / N')
plt.plot(u_equi / N, label='U* / N')
plt.plot(e_equi / N, label='E* / N')
plt.plot(t_equi, label='Temp')
plt.plot(p_equi, label='Pressure')
plt.hlines(0, 0, n_equi, label='y = 0')
plt.title('Equilibrium MD Simulation', fontsize=14)
plt.ylabel('Reduced Variables', fontsize=14)
plt.xlabel('MD Steps', fontsize=14)
plt.legend(loc='upper left')
plt.show()

plt.figure()
plt.plot(k_calc / N, label='K* / N')
plt.plot(u_calc / N, label='U* / N')
plt.plot(e_calc / N, label='E* / N')
plt.plot(t_calc, label='Temp')
plt.plot(p_calc, label='Pressure')
plt.hlines(0, 0, n_calc, label='y = 0')
plt.title('Calculation MD Simulation', fontsize=14)
plt.ylabel('Reduced Variables', fontsize=14)
plt.xlabel('MD Steps', fontsize=14)
plt.legend(loc='upper left')
plt.show()

plt.figure()
plt.plot(k_calc / N, label='K* / N')
plt.hlines(k_calc.mean() / N, 0, n_calc, 'r', '--', label='mean')
plt.title('Calculation MD Simulation, KE', fontsize=14)
plt.ylabel('Reduced Variables', fontsize=14)
plt.xlabel('MD Steps', fontsize=14)
plt.legend()
plt.show()

plt.figure()
plt.plot(u_calc / N, label='U* / N')
plt.hlines(u_calc.mean() / N, 0, n_calc, 'r', '--', label='mean')
plt.title('Calculation MD Simulation, PE', fontsize=14)
plt.ylabel('Reduced Variables', fontsize=14)
plt.xlabel('MD Steps', fontsize=14)
plt.legend()
plt.show()

plt.figure()
plt.plot(e_calc / N, label='E* / N')
plt.hlines(e_calc.mean() / N, 0, n_calc, 'r', '--', label='mean')
plt.title('Calculation MD Simulation, TE', fontsize=14)
plt.ylabel('Reduced Variables', fontsize=14)
plt.xlabel('MD Steps', fontsize=14)
plt.legend()
plt.show()

plt.figure()
plt.plot(t_calc, label='Temp')
plt.hlines(t_calc.mean(), 0, n_calc, 'r', '--', label='mean')
plt.title('Calculation MD Simulation, Temp', fontsize=14)
plt.ylabel('Reduced Variables', fontsize=14)
plt.xlabel('MD Steps', fontsize=14)
plt.legend()
plt.show()

plt.figure()
plt.plot(p_calc, label='Pressure')
plt.hlines(p_calc.mean(), 0, n_calc, 'r', '--', label='mean')
plt.title('Calculation MD Simulation, Pressure', fontsize=14)
plt.ylabel('Reduced Variables', fontsize=14)
plt.xlabel('MD Steps', fontsize=14)
plt.legend()
plt.show()