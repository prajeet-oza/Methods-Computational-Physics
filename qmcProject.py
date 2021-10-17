import numpy as np
import matplotlib.pyplot as plt

########## FUNCTIONS USED THROUGH THE CODE ##########

# part of trial wavefunction, chi1 * chi2 * f12
def Chi(R):
	chi = np.exp(- R / A)
	return chi

# first derivative of chi
def FDChi(R):
	fdchi = - Chi(R) / A
	return fdchi

# second derivative of chi
def SDChi(R):
	sdchi = Chi(R) / A**2
	return sdchi

# laplacian of chi
def LapChi(R):
	lapchi = SDChi(R) + 2 * FDChi(R) / R
	return lapchi

# part of trial wavefunction, chi1 * chi2 * f12
def Fee(R):
	fee = np.exp(R / (alpha * (1 + beta * R)))
	return fee

# first derivative of f
def FDFee(R):
	fdfee = Fee(R) / (alpha * (1 + beta * R)**2)
	return fdfee

# second derivative of f
def SDFee(R):
	sdfee = ((FDFee(R))**2 / Fee(R)) - (2 * beta * Fee(R) / (alpha * (1 + beta * R)**3))
	return sdfee

# laplacian of f
def LapFee(R):
	lapfee = SDFee(R) + 2 * FDFee(R) / R
	return lapfee

# distance calculating function
def Dist(X, Y, Z):
	dist = np.sqrt(X**2 + Y**2 + Z**2)
	return dist

# initializing the configuration
def intcfg():
	confg = np.zeros((6, 1))
	for i in range(6):
		confg[i, 0] = A * np.random.uniform(-0.5, 0.5) # randomly selecting points with a uniform distribution

	confg[2, 0] = confg[2, 0] + Sby2 # displacing by half of separation in z direction, electron 1
	confg[5, 0] = confg[5, 0] - Sby2 # displacing by half of separation in z direction, electron 2

	wght = Phi(confg)**2 # calculating the weight with the trial wavefunction
	return confg, wght

# calculating the distances b/w electrons and protons, r1L, r1R, r2L, r2R, r12
def radii(confg):
	X1 = confg[0, 0]
	Y1 = confg[1, 0]
	Z1 = confg[2, 0]
	X2 = confg[3, 0]
	Y2 = confg[4, 0]
	Z2 = confg[5, 0]

	r1L = Dist(X1, Y1, Z1 + Sby2)
	r1R = Dist(X1, Y1, Z1 - Sby2)
	r2L = Dist(X2, Y2, Z2 + Sby2)
	r2R = Dist(X2, Y2, Z2 - Sby2)
	r12 = Dist(X1 - X2, Y1 - Y2, Z1 - Z2)

	return r1L, r1R, r2L, r2R, r12

# calculating trial wavefunction
def Phi(confg):
	r1L, r1R, r2L, r2R, r12 = radii(confg)
	chi1 = Chi(r1L) + Chi(r1R)
	chi2 = Chi(r2L) + Chi(r2R)
	f = Fee(r12)
	phi = chi1 * chi2 * f

	return phi

# local energy calculating function
def ELocal(confg):
	X1 = confg[0, 0]
	Y1 = confg[1, 0]
	Z1 = confg[2, 0]
	X2 = confg[3, 0]
	Y2 = confg[4, 0]
	Z2 = confg[5, 0]

	r1L, r1R, r2L, r2R, r12 = radii(confg)

	r12fd = X1*(X1 - X2) + Y1*(Y1 - Y2) + Z1*(Z1 - Z2) # first derivative of r12
	const = Sby2 * (Z1 - Z2) # constant useful to calculate dot product with first derivative of r1L, r1R, r2L, r2R

	# first derivative dot products with r12
	r1Ldot = (r12fd + const) / r12 / r1L
	r1Rdot = (r12fd - const) / r12 / r1R
	r2Ldot = (r12fd + const - r12**2) / r12 / r2L
	r2Rdot = (r12fd - const - r12**2) / r12 / r2R

	# parts of trial wavefunction
	chi1 = Chi(r1L) + Chi(r1R)
	chi2 = Chi(r2L) + Chi(r2R)
	f = Fee(r12)

	# parts of KE operator: divergence and division by trial wavefunction
	# correlations factor of trial wavefunction, f
	Ecorr = 2 * LapFee(r12) / f
	# chi factor of trial wavefunctions, chi1 and chi2
	Ee1 = (LapChi(r1L) + LapChi(r1R)) / chi1
	Ee2 = (LapChi(r2L) + LapChi(r2R)) / chi2
	# cross terms
	Ecross1 = (r1Ldot * FDChi(r1L) + r1Rdot * FDChi(r1R)) / chi1
	Ecross2 = (r2Ldot * FDChi(r2L) + r2Rdot * FDChi(r2R)) / chi2
	Ecross = 2 * FDFee(r12) * (Ecross1 - Ecross2) / f

	# combining all the parts, and getting KE value
	KE = - HbyM * (Ecorr + Ee1 + Ee2 + Ecross) / 2
	# calculating PE
	PE = - e2 * (1/r1L + 1/r1R + 1/r2L + 1/r2R - 1/r12)
	# summing to get local energy
	elocal = KE + PE

	return elocal

# calculating the drift
def Drift(confg):
	X1 = confg[0, 0]
	Y1 = confg[1, 0]
	Z1 = confg[2, 0]
	X2 = confg[3, 0]
	Y2 = confg[4, 0]
	Z2 = confg[5, 0]

	r1L, r1R, r2L, r2R, r12 = radii(confg)

	# parts of the trial wavefunctions
	chi1 = Chi(r1L) + Chi(r1R)
	chi2 = Chi(r2L) + Chi(r2R)
	f = Fee(r12)

	# parts of the shift
	# first derivatives of log(wavefunction) = log(chi1) + log(chi2) + log(f)
	factChi1 = (FDChi(r1L) / r1L + FDChi(r1R) / r1R) / chi1
	factChi1Z = (FDChi(r1L) / r1L - FDChi(r1R) / r1R) / chi1 # useful for the Sby2 term in z direction
	
	factChi2 = (FDChi(r2L) / r2L + FDChi(r2R) / r2R) / chi2
	factChi2Z = (FDChi(r2L) / r2L - FDChi(r2R) / r2R) / chi2 # useful for the Sby2 term in z direction

	factFee = FDFee(r12) / r12 / f

	# setting the shift variable
	shift = np.zeros(confg.shape)

	# attempted to splice the arrays for computing shift, but it slows the code:
	# so comment and uncooment if you want to try splicing route to calculate shift
	# shift[0:3, 0] = HdTbyM * (confg[0:3, 0] * factChi1 + (confg[0:3, 0] - confg[3:6, 0]) * factFee)
	# shift[2, 0] = shift[2, 0] + Sby2 * factChi1Z * HdTbyM

	# shift[3:6, 0] = HdTbyM * (confg[3:6, 0] * factChi2 + (confg[3:6, 0] - confg[0:3, 0]) * factFee)
	# shift[5, 0] = shift[5, 0] + Sby2 * factChi2Z * HdTbyM

	# summing the parts to calculate shift
	shift[0, 0] = HdTbyM * (X1 * factChi1 + (X1 - X2) * factFee)
	shift[1, 0] = HdTbyM * (Y1 * factChi1 + (Y1 - Y2) * factFee)
	shift[2, 0] = HdTbyM * (Z1 * factChi1 + (Z1 - Z2) * factFee + Sby2*factChi1Z)

	shift[3, 0] = HdTbyM * (X2 * factChi2 - (X1 - X2) * factFee)
	shift[4, 0] = HdTbyM * (X2 * factChi2 - (X1 - X2) * factFee)
	shift[5, 0] = HdTbyM * (Z2 * factChi2 - (Z1 - Z2) * factFee + Sby2*factChi2Z)

	return shift

# metropolis algorithm implementation with phi^2 as weight
def Metrop(confg, wght, accpt):
	prev_confg = confg.copy()

	# getting a trial configuration
	# delta from defined variables times a randomly chosen direction with a uniform distribution
	for i in range(confg.shape[0]):
		confg[i, 0] = confg[i, 0] + delta * np.random.uniform(-0.5, 0.5)

	# trial weight
	wght_t = Phi(confg)**2
	ratio = wght_t / wght # ratio for accept or reject
	mu = np.random.uniform(0, 1)

	if ratio > mu: # accepting or rejecting
		wght = wght_t
		accpt += 1
	else:
		confg = prev_confg.copy()

	return confg, wght, accpt

# initializing the ensemble
def intEns():
	confg, wght = intcfg() # starting with a configuration
	accpt = 0
	for i in range(30): # thermalising before picking the ensemble
		confg, wght, accpt = Metrop(confg, wght, accpt) # metropolis step

	accpt = 0
	ensemble = np.zeros((confg.shape[0], Nensmbl))
	wghtEns = np.ones(Nensmbl) # initilizing the weight to 1, i.e., equi-weighted start
	for itr in range(10 * Nensmbl): # looping for ensemble
		confg, wght, accpt = Metrop(confg, wght, accpt) # metropolis step
		if (itr+1) % 10 == 0: # picking every 10th configuration to add to the ensemble
			ival = (itr+1)//10 - 1
			ensemble[:, ival] = confg.copy().reshape(ensemble[:, ival].shape)

	return ensemble, wghtEns

# time step for path integral
def TStep(ensemble, wghtEns):
	energySum = 0 # initialise the weight and energy sums to zero
	wghtSum = 0

	for itr in range(Nensmbl): # looping over the ensemble to solve the path integral, and then calculate epsilon
		confg = ensemble[:, itr].reshape((ensemble[:, itr].shape[0], 1))

		shift = Drift(confg) # calculate drift in the configuration
		# based on the path integral section of PIMC, it has two factors
		# one: a gaussian distribution with mean as (configuration + drift), and variance as hbar^2 * dt / m
		# two: an exponential term with epsilon minus En
		# now, the section following the path integral also states that the En part can be observed as weight
		# and energy can be observed as a weighted average of local energy
		# provided the weights are summed up to 1, for the ensemble

		# getting a configuration based on the gaussian distribution
		confg = confg + dT * shift + np.sqrt(HdTbyM) * np.random.randn()
		epsilon = ELocal(confg) # calculate the local energy for the configuration

		wghtEns[itr] = wghtEns[itr] * np.exp(- epsilon * dT) # calculate the weights for the configuration

		energySum = energySum + wghtEns[itr] * epsilon # take a weighted average to sum the energy
		wghtSum = wghtSum + wghtEns[itr] # sum the weights, which are later to be normalised to 1

		ensemble[:, itr] = confg.copy().reshape(ensemble[:, itr].shape) # replace the drifted configuration to the ensemble

	epsilon = energySum / wghtSum # local energy is the division of energy and weight sums
	wghtEns = (Nensmbl / wghtSum) * wghtEns # normalise the weights to 1

	return ensemble, wghtEns, epsilon


########## MAIN CODE ##########

# defining starting variables 
S = np.arange(0.3, 2.9, 0.1) # values of separation used for the calculations, in angstrom
dT = 0.1 # time step for path integral monte carlo
delta = 0.4	# delta d, for metropolis algorithm
HbyM = 7.6359 # hbar^2 / m normalised for angstrom and eV
e2 = 14.409 # electron charge and permittivity, normalised for angstrom and eV
A0 = HbyM / e2 # bohr radius for hydrogen
HdTbyM = HbyM * dT # useful constant, hbar^2 * dt / m
beta = 0.25 # variational parameters
alpha = 2 * A0 # variational parameters

Ntherm_mtp = 1000 # variational monte carlo (VMC) thermalisation steps
Nfreq_mtp = 50 # VMC nfreq'th sweep counted for calculation to avoid correlated sweeps
Nsweep_mtp = 500000 # VMC number of sweeps

Ntherm_pmc = 500 # path integral monte carlo (PIMC) thermalisation steps
Nfreq_pmc = 20 # PIMC nfreq'th sweep counted for calculation to avoid correlated sweeps
Nsweep_pmc = 10000 # PIMC number of sweeps
Nensmbl = 20 # ensemble size

# energy storing variables
energy_metrop = np.zeros(S.shape) # VMC
energy_pathmc = np.zeros(S.shape) # PIMC

ind = 0 # counting variable, used to store energies later in the code

# varying separation to observe the variation in energy
for iS in S:
	print('\n\n SEPARATION b/w protons (in Angstroms): ', iS)
	Sby2 = iS / 2 # useful constant
	
	# calculating constant A from the trial wavefunction
	A = A0
	Aold = 0
	if (abs(A - Aold)) > 10**(-6):
		Aold = A
		A = A0 / (1 + np.exp(- iS / Aold))

	# starting the VMC calculations
	print('\n VARIATIONAL MONTE CARLO ----->')
	
	confg_mtp, wght_mtp = intcfg() # initializing the configuration

	accpt_mtp = 0 # VMC acceptance ratio calculating variable
	for i in range(Ntherm_mtp): # thermalization step
		if i == 0:
			print('\n   Thermalizing... \n')
		if i == Ntherm_mtp - 1:
			print('   Thermalizing... DONE! \n')

		confg_mtp, wght_mtp, accpt_mtp = Metrop(confg_mtp, wght_mtp, accpt_mtp) # metropolis step
	print('   Thermalizing, ACCEPTANCE RATIO: ', accpt_mtp / Ntherm_mtp)

	taccpt_mtp = 0 # variable to count total acceptance for the sweeps
	epsln_mtp = np.zeros((2, 1)) # VMC epsilon storing variable

	for iswp in range(Nsweep_mtp): # sweeping for VMC
		if iswp == 0:
			print('\n   Sweeping and Summing... ')

		accpt_mtp = 0
		confg_mtp, wght_mtp, accpt_mtp = Metrop(confg_mtp, wght_mtp, accpt_mtp) # metropolis step
		taccpt_mtp += accpt_mtp

		if (iswp+1) % Nfreq_mtp == 0: # considering nfreq'th sweep for epsilon calculation
			epsln_mtp[0, 0] = ELocal(confg_mtp) # local energy, epsilon calculating function
			epsln_mtp[1, 0] = epsln_mtp[1, 0] + epsln_mtp[0, 0] # storing the sum of epsilon for all considered sweeps

		if iswp == Nsweep_mtp-1:
			print('\n   Sweeping and Summing... DONE! \n')

	print('   Sweeping, TOTAL ACCEPTANCE: ', taccpt_mtp / Nsweep_mtp)

	print('\n   Averaging... \n')

	count_mtp = np.floor(Nsweep_mtp / Nfreq_mtp) # number of sweeps considered

	eigen_mtp = epsln_mtp[1, 0] / count_mtp # least eigen value by averaging the epsilon total obtained from VMC

	energy_metrop[ind] = eigen_mtp + e2 / iS + e2 / A0 # storing the final/total energy from VMC
	print('   TOTAL ENERGY: ', energy_metrop[ind])

	# starting the PIMC calculations
	print('\n PATH INTEGRAL MONTE CARLO ----->')

	ensemble_pmc, wghtEns_pmc = intEns() # initializing the ensemble

	for i in range(Ntherm_pmc): # thermalization step, with time step
		if i == 0:
			print('\n   Thermalizing... \n')
		if i == Ntherm_pmc - 1:
			print('   Thermalizing... DONE! \n')
		ensemble_pmc, wghtEns_pmc, epsln_dump = TStep(ensemble_pmc, wghtEns_pmc) # time step

	epsln_pmc = np.zeros((2, 1)) # PIMC epsilon storing variable

	for iswp in range(Nsweep_pmc): # sweeping for PIMC
		if iswp == 0:
			print('   Sweeping and Summing... \n')

		ensemble_pmc, wghtEns_pmc, epsln_pmc[0, 0] = TStep(ensemble_pmc, wghtEns_pmc) # time step

		if (iswp+1) % Nfreq_pmc == 0: # considering nfreq'th for epsilon calculation
			epsln_pmc[1, 0] = epsln_pmc[1, 0] + epsln_pmc[0, 0] # storing the sum of epsilon for all considered sweeps
		
		if iswp == Nsweep_pmc-1:
			print('   Sweeping and Summing... DONE! \n')

	print('   Averaging... \n')
	
	count_pmc = np.floor(Nsweep_pmc / Nfreq_pmc) # number of sweeps considered

	eigen_pmc = epsln_pmc[1, 0] / count_pmc # least eigen value by averaging the epsilon total from PIMC calculations

	energy_pathmc[ind] = eigen_pmc + e2 / iS + e2 / A0 # storing the final/total energy from PIMC 
	print('   TOTAL ENERGY: ', energy_pathmc[ind])
	ind += 1

# plotting the energies for VMC and PIMC calculations for various separations
plt.figure()
plt.plot(S, energy_metrop - energy_metrop[-1], label='Variational MC')
plt.plot(S, energy_pathmc - energy_pathmc[-1], label='Path Integral MC')
plt.hlines(0, S[0], S[-1], label='y = 0')
plt.title('Quantum Monte Carlo, VMC and PIMC', fontsize=14)
plt.ylabel('Binding Energy for H2 molecule', fontsize=14)
plt.xlabel('Separation b/w protons', fontsize=14)
plt.legend()
plt.show()