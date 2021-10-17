import numpy as np
import matplotlib.pyplot as plt

########## FUNCTIONS USED THROUGH THE CODE ##########

# initial spin configuration, by putting a threshold on uniformly distributed random numbers
def confg(Nx, Ny):
	spin = np.zeros((Nx, Ny))
	
	for i in range(Nx):
		for j in range(Ny):
			if np.random.random() > 0.5:
				spin[i, j] = 1
			else:
				spin[i, j] = -1

	return spin

# ratio for the metropolis step
def ratio(f, Sa):
	r = np.exp(-2 * Sa * (J * f + B))
	return r

# metropolis algorithm
def metrop(spin):
	accpt = 0
	Nx = spin.shape[0]
	Ny = spin.shape[1]

	for ix in range(Nx): # indexing the neighbours
		ixp1 = (ix + 1) % Nx # right x-axis index
		ixm1 = (ix - 1) % Nx # left x-axis index
		for iy in range(Ny):
			iyp1 = (iy + 1) % Ny # right y-axis index
			iym1 = (iy - 1) % Ny # left y-axis index

			f = spin[ixp1, iy] + spin[ixm1, iy] + spin[ix, iyp1] + spin[ix, iym1]
			if np.random.random() < ratio(f, spin[ix, iy]): # based on ratio, flip or retain the spin
				spin[ix, iy] = - spin[ix, iy]
				accpt += 1
	accpt = accpt / (Nx * Ny) # counting the acceptance ratio

	return spin, accpt

# sum function for the calculating the magnetization and energy after a number of sweeps
def sum_func(spin, mag, energy):
	pairs = 0
	Nx = spin.shape[0]
	Ny = spin.shape[1]
	# sweep, total = 0, 1 ---> levels of calculation
	# value, square = 0, 1 ---> types of calculation
	# prop[levels][type] ---> format to navigate with type and levels in the matrix

	for ix in range(Nx): # summing the spins and pairs, and calculating magnetization
		ixm1 = (ix - 1) % Nx
		for iy in range(Ny):
			iym1 = (iy -1) % Ny

			pairs = pairs + spin[ix, iy] * spin[ix, iym1] + spin[ix, iy] * spin[ixm1, iy]
			mag[0, 0] = mag[0, 0] + spin[ix, iy]

	# per sweep calculations
	mag[0, 1] = mag[0, 0]**2 # magnetization square, useful for susceptibility calculation
	energy[0, 0] = - J * pairs - B * mag[0, 0] # energy calculation
	energy[0, 1] = energy[0, 0]**2 # energy square, useful for Cb calculation

	# adding the sweep to the total, which will be later averaged over the number of sweeps
	mag[1, 0] = mag[1, 0] + mag[0, 0]
	mag[1, 1] = mag[1, 1] + mag[0, 1]
	energy[1, 0] = energy[1, 0] + energy[0, 0]
	energy[1, 1] = energy[1, 1] + energy[0, 1]

	return mag, energy


########## MAIN CODE ##########

# defining starting variables
B = 0 # magnetic field
J = 0.3 # interaction strength
Nx = 20 # number of lattice points in x
Ny = 20 # number of lattice points in y
Ntherm = 20 # number of thermalization steps
Nsweep = 500 # number of sweep steps
Nfreq = 5 # nfreq'th sweep counted for calculations, to avoid correlated sweeps
mag = np.zeros((2, 2)) # magnetization matrix
energy = np.zeros((2, 2)) # energy matrix
chi = np.zeros((2, 2)) # susceptibility matrix
cb = np.zeros((2, 2)) # Cb matrix

spin = confg(Nx, Ny) # initializing the lattice with spins
init_spin = spin.copy()

taccpt = 0
for i in range(Ntherm): # thermalization step
	if i == 0:
		print('\n Thermalizing... \n')
	if i == Ntherm-1:
		print(' Thermalizing... DONE! \n')
	spin, accpt = metrop(spin)
	taccpt += accpt
print(' Thermalizing, ACCEPTANCE RATIO: ', taccpt / Ntherm) # acceptance ratio through the thermalization step

taccpt = 0
for itr in range(Nsweep): # sweeping through the lattice, summing for properties and calculating
	if itr == 0:
		print('\n Sweeping and Summing... \n')

	spin, accpt = metrop(spin) # metropolis step
	taccpt += accpt
	if (itr+1) % Nfreq == 0: # considers every nfreq'th sweep for calculation
		print('\n SWEEP ', itr+1, '-'*10, '>')
		print('  Acceptance ratio: ', accpt)

		# before calculations, initialize the sweep section the matrices to zero
		# like a clean slate for every considered sweep
		mag[0, 0], mag[0, 1] = 0, 0
		energy[0, 0], energy[0, 1] = 0, 0
		chi[0, 0], chi[0, 1] = 0, 0
		cb[0, 0], cb[0, 1] = 0, 0

		swp = (itr+1)/Nfreq
		mag, energy = sum_func(spin, mag, energy) # summing and calculating magnetization and energy
		
		# susceptibility and Cb calculations,
		# which require sums from magnetization and energy
		# hence, they are done after the sum step
		chi[0, 0] = mag[1, 1] / swp - (mag[1, 0] / swp)**2
		chi[0, 1] = chi[0, 0]**2
		chi[1, 0] = chi[1, 0] + chi[0, 0]
		chi[1, 1] = chi[1, 1] + chi[0, 1]

		cb[0, 0] = energy[1, 1] / swp - (energy[1, 0] / swp)**2
		cb[0, 1] = cb[0, 0]**2
		cb[1, 0] = cb[1, 0] + cb[0, 0]
		cb[1, 1] = cb[1, 1] + cb[0, 1]

		# printing the calculated values after every considered sweep
		print('  Magnetization: ', mag[0, 0])
		print('  Energy: ', energy[0, 0])
		print('  Susceptibility: ', chi[0, 0])
		print('  Cb: ', cb[0, 0])

		if itr == Nsweep-1:
			print('\n Sweeping and Summing... DONE! \n')

# acceptance ratio through the sweep steps
print(' Sweeping, TOTAL ACCEPTANCE: ', taccpt / Nsweep)

# averaging the properties over multiple sweeps
print('\n Averaging... \n')

# count of sweeps that were considered for calculation
count = np.floor(Nsweep / Nfreq)

# magnetization, average and standard deviation
M = mag[1, 0] / count
sigM = (mag[1, 1] / count - M**2) / count
if sigM < 0: sigM = 0

# energy, average and standard deviation
E = energy[1, 0] / count
sigE = (energy[1, 1] / count - E**2) / count
if sigE < 0: sigE = 0

# susceptibility, average and standard deviation
Sus = chi[1, 0] / count
sigSus = (chi[1, 1] / count - Sus**2) / count
if sigSus < 0: sigSus = 0

# Cb, average and standard deviation
CB = cb[1, 0] / count
sigCB = (cb[1, 1] / count - CB**2) / count
if sigCB < 0: sigCB = 0

# presenting the property values
print('\n Magnetization: ', M)
print(' Std. Deviation: ', np.sqrt(sigM))

print('\n Energy: ', E)
print(' Std. Deviation: ', np.sqrt(sigE))

print('\n Susceptibility: ', Sus)
print(' Std. Deviation: ', np.sqrt(sigSus))

print('\n CB: ', CB)
print(' Std. Deviation: ', np.sqrt(sigCB))

plt.figure()
plt.title('Visualise the initial configuration', fontsize=13)
plt.xlabel('X lattice points, 0 to Nx', fontsize=12)
plt.ylabel('Y lattice points, 0 to Ny', fontsize=12)
plt.imshow(init_spin, cmap='hot')
plt.show()

plt.figure()
plt.title('Visualise the final configuration', fontsize=13)
plt.xlabel('X lattice points, 0 to Nx', fontsize=12)
plt.ylabel('Y lattice points, 0 to Ny', fontsize=12)
plt.imshow(spin, cmap='hot')
plt.show()