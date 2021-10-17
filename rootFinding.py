import numpy as np

########## FUNCTIONS USED IN THE CODE ##########

# Lennard Jones potential
def LJPotential(r, sig, eps):
	V = 4 * eps * ((sig / r)**12 - (sig / r)**6)
	return V

# negative of the first derivative of Lennard Jones
def LJForce(r, sig, eps):
	F = (24 * eps / r) * (2 * (sig / r)**12 - (sig / r)**6)
	return F

########## FUNCTIONS USED IN THE CODE ##########

# taking input for sigma and epsilon
print('\n ' + '#'*10 + ' LENNARD-JONES POTENTIAL ' + '#'*10)
epsilon = float(input('  Enter the value for epsilon: '))
sigma = float(input('  Enter the value for sigma: '))

# defining constants and variable values
dx = 0.001 # shifting the values slightly to avoid zero in denominator
lowLim = 0 + dx # lower limit of search
upLim = 40 + dx # upper limit of search
delta = 1 # interval for each search
search = np.arange(lowLim, upLim + delta, delta) # setting up the interval
tolerance = 10**-2 # tolerance of the root finding algorithm

lowPot = [] # array to store lower limit for the interval, potential root finding
upPot = [] # array to store upper limit for the interval, potential root finding
rootPot = [] # array to store roots, potential root finding

# searching the intervals for potential root finding
# bisection method, brute force search
for s in search:
	if LJPotential(s, sigma, epsilon) * LJPotential(s + delta, sigma, epsilon) < 0:
		lowPot.append(s)
		upPot.append(s + delta)
	elif LJPotential(s, sigma, epsilon) == 0:
		rootPot.append(s)

lowForce = [] # array to store lower limit for the interval, force root finding
upForce = [] # array to store upper limit for the interval, force root finding
rootForce = [] # array to store roots, force root finding

# searching the intervals for potential root finding
# bisection method, brute force search
for s in search:
	if LJForce(s, sigma, epsilon) * LJForce(s + delta, sigma, epsilon) < 0:
		lowForce.append(s)
		upForce.append(s + delta)
	elif LJForce(s, sigma, epsilon) == 0:
		rootForce.append(s)

lowTear = [] # array to store lower limit for the interval, tearing the atoms root finding
upTear = []# array to store upper limit for the interval, teraing the atoms root finding
rootTear = [] # array to store roots, tearing the atoms root finding
tearOff = -0.0001 # tearing of atoms cutoff

# searching the intervals for tearing the atoms root finding
# bisection method, brute force search
for s in search:
	if (LJPotential(s, sigma, epsilon) - tearOff) * (LJPotential(s + delta, sigma, epsilon) - tearOff) < 0:
		lowTear.append(s)
		upTear.append(s + delta)
	elif (LJPotential(s, sigma, epsilon) - tearOff) == 0:
		rootTear.append(s)

# finding the roots for the selected intervals from searching
# bisection method, potential roots
for i in range(len(lowPot)):
	a0 = lowPot[i]
	b0 = upPot[i]
	while (abs(a0 - b0) / 2) > tolerance:
		c0 = a0 + (b0 - a0) / 2
		if LJPotential(a0, sigma, epsilon) * LJPotential(c0, sigma, epsilon) < 0:
			b0 = c0
		elif LJPotential(c0, sigma, epsilon) * LJPotential(b0, sigma, epsilon) < 0:
			a0 = c0
	root = a0 + (b0 - a0) / 2
	rootPot.append(root)

# finding the roots for the selected intervals from searching
# bisection method, for force roots
for i in range(len(lowForce)):
	a0 = lowForce[i]
	b0 = upForce[i]
	while (abs(a0 - b0) / 2) > tolerance:
		c0 = a0 + (b0 - a0) / 2
		if LJForce(a0, sigma, epsilon) * LJForce(c0, sigma, epsilon) < 0:
			b0 = c0
		elif LJForce(c0, sigma, epsilon) * LJForce(b0, sigma, epsilon) < 0:
			a0 = c0
	root = a0 + (b0 - a0) / 2
	rootForce.append(root)

# finding the roots for the selected intervals from searching
# bisection method, tearing the atoms roots
for i in range(len(lowTear)):
	a0 = lowTear[i]
	b0 = upTear[i]
	while (abs(a0 - b0) / 2) > tolerance:
		c0 = a0 + (b0 - a0) / 2
		if (LJPotential(a0, sigma, epsilon) - tearOff) * (LJPotential(c0, sigma, epsilon) - tearOff) < 0:
			b0 = c0
		elif (LJPotential(c0, sigma, epsilon) - tearOff) * (LJPotential(b0, sigma, epsilon) - tearOff) < 0:
			a0 = c0
	root = a0 + (b0 - a0) / 2
	rootTear.append(root)

# displaying the results / roots
truePotRoot = sigma
print('\n  Root to the LJ Potential: ', rootPot[0])
print('  True value for the root to LJ Potential: ', truePotRoot)

trueForceRoot = sigma * 2**(1/6)
print('\n  Root to the LJ Force: ', rootForce[0])
print('  True value for the root to LJ Force: ', trueForceRoot)

potValue = LJPotential(rootForce[0], sigma, epsilon)
truePotValue = LJPotential(sigma * 2**(1/6), sigma, epsilon)
print('\n  Value for LJ Potential at the force root: ', potValue)
print('  Value for LJ Potential at the true value of force root: ', truePotValue)

print('\n  Separation to tear the atom pair: ', max(rootTear))
