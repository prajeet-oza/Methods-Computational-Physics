import numpy as np
import matplotlib.pyplot as plt

########## FUNCTIONS USED THROUGH THE CODE ##########

# weight function, w(x) = normal distribution with mean 0 and variance 1
def weight(x, mode):
	if mode == 'gaussian':
		w = 1 / (np.sqrt(2 * np.pi * 2)) * np.exp(- x**2 / 4)
	if mode == 'exponential':
		w = np.exp(-x)
		# w = 0.25 * np.exp(-0.25 * x)
	return w

# metropolis algorithm implementation
def metrop(x, delta, accpt, mode):
	if mode == 'gaussian': # for gaussian random walk
		dirtn = np.random.uniform(-1, 1) # direction variable
		x_t = x + delta * dirtn # trial step
	if mode == 'exponential': # for exponential random walk
		x_t = -1
		while x_t < 0: # considering the x > 0 bound on exponential distribution
			dirtn = np.random.uniform(-1, 1) # direction variable
			x_t = x + delta * dirtn # trial step

	ratio = weight(x_t, mode) / weight(x, mode) # ratio
	mu = np.random.uniform(0, 1)

	if ratio > mu: # comparing, accept or reject
		x = x_t
		accpt += 1

	return x, accpt

# integrating function F(x) = sqrt(2 * pi) x^2
def funcFofX(x, mode):
	if mode == 'gaussian':
		f = x**2 * np.sqrt(2 * np.pi * 2)
	if mode == 'exponential':
		f = x**2 * np.exp(- x**2 / 4) / np.exp(-x)
		# f = 2 * np.sqrt(x)
	return f

# correlation factor: average of (f_i * f_(i+k))
def corr(arrf, Nstep, k):
	corr_val = 0
	for i in range(Nstep - k):
		corr_val += arrf[i] * arrf[i+k]
	corr_val = corr_val  / (Nstep - k)

	return corr_val

########## MAIN CODE ##########

# defining starting variables
x0 = 5000 # starting point
Ntherm = 1000 # number of thermalization steps
Nstep = 100000 # number of sweeps on metropolis algorithm
Nfreq = 1 # nfreq'th sweep is considered for calculation

# gaussian distribution random walk
print('\n')
print('#'*10 + ' METROPOLIS ALGORITHM, GAUSSIAN ' + '#'*10)

delta = 2 # delta d
walk = [] # walk points list
func = [] # function value list
total = np.zeros((3, 1))
accpt = 0
x_prev = x0
mode = 'gaussian'
for i in range(Ntherm): # thermalization step
	if i == 0:
		print('\n Thermalizing... \n')
	if i == Ntherm - 1:
		print(' Thermalizing... DONE! \n')
	
	x_next, accpt = metrop(x_prev, delta, accpt, mode)
	x_prev = x_next
print(' Thermalizing, ACCEPTANCE RATIO: ', accpt / Ntherm)

accpt = 0
for i in range(Nstep): # metropolis step
	if i == 0:
		print('\n Random Walk... \n')
	if i == Nstep - 1:
		print(' Random Walk... DONE! \n')
	
	walk.append(x_prev)
	fval = funcFofX(x_prev, mode)
	func.append(fval)
	if (i+1) % Nfreq == 0:
		total[0, 0] += fval
		total[1, 0] += fval**2

	x_next, accpt = metrop(x_prev, delta, accpt, mode)
	x_prev = x_next
print(' Random Walk, ACCEPTANCE RATIO: ', accpt / Nstep)

count = Nstep // Nfreq # number of sweeps considered for calculations

# calculating the integral and standard deviation or sigma
total[0, 0] = total[0, 0] / count
total[1, 0] = total[1, 0] / count
total[2, 0] = (total[1, 0] - total[0, 0]**2 )

print('\n Value of the Integral: ', total[0, 0])

# calculating the auto-correlation function
print('\n Auto-Correlation... CALCULATING')
autocorr = np.zeros((20, 1))
autocorr[0, 0] = (corr(func, Nstep, 0) - total[0, 0]**2) / total[2, 0]
print('\n Auto-Correlation... k = 0: ', autocorr[0, 0])

for i in range(autocorr.shape[0]-1):
	if 10**i < Nstep:
		autocorr[i+1, 0] = (corr(func, Nstep, 10**i) - total[0, 0]**2) / total[2, 0]
		if autocorr[i+1, 0] < 0:
			autocorr[i+1, 0] = 0
		print('\n Auto-Correlation... k = 10^{}: {}'.format(i, autocorr[i+1, 0]))

walk_gauss = walk.copy()

# exponential distribution random walk 
print('\n')
print('#'*10 + ' METROPOLIS ALGORITHM, EXPONENTIAL ' + '#'*10)

delta = 0.5 # delta d
x0 = 0
walk = [] # walk points list
func = [] # function value list
total = np.zeros((3, 1))
accpt = 0
x_prev = x0
mode = 'exponential'
for i in range(Ntherm): # thermalization step
	if i == 0:
		print('\n Thermalizing... \n')
	if i == Ntherm - 1:
		print(' Thermalizing... DONE! \n')
	
	x_next, accpt = metrop(x_prev, delta, accpt, mode)
	x_prev = x_next
print(' Thermalizing, ACCEPTANCE RATIO: ', accpt / Ntherm)

accpt = 0
for i in range(Nstep): # metropolis step
	if i == 0:
		print('\n Random Walk... \n')
	if i == Nstep - 1:
		print(' Random Walk... DONE! \n')
	
	walk.append(x_prev)
	fval = funcFofX(x_prev, mode)
	func.append(fval)
	if (i+1) % Nfreq == 0:
		total[0, 0] += fval
		total[1, 0] += fval**2

	x_next, accpt = metrop(x_prev, delta, accpt, mode)
	x_prev = x_next
print(' Random Walk, ACCEPTANCE RATIO: ', accpt / Nstep)

count = Nstep // Nfreq # number of sweeps considered for calculations

# calculating the integral and standard deviation or sigma
total[0, 0] = total[0, 0] / count
total[1, 0] = total[1, 0] / count
total[2, 0] = (total[1, 0] - total[0, 0]**2 )

print('\n Value of the Integral: ', total[0, 0] * 2)

# calculating the auto-correlation function
print('\n Auto-Correlation... CALCULATING')
autocorr = np.zeros((20, 1))
autocorr[0, 0] = (corr(func, Nstep, 0) - total[0, 0]**2) / total[2, 0]
print('\n Auto-Correlation... k = 0: ', autocorr[0, 0])

for i in range(autocorr.shape[0]-1):
	if 10**i < Nstep:
		autocorr[i+1, 0] = (corr(func, Nstep, 10**i) - total[0, 0]**2) / total[2, 0]
		if autocorr[i+1, 0] < 0:
			autocorr[i+1, 0] = 0
		print('\n Auto-Correlation... k = 10^{}: {}'.format(i, autocorr[i+1, 0]))

# plotting the histogram for the points obtained from metropolis algorithm, GAUSSIAN
plt.figure()
plt.hist(walk_gauss)
plt.title('Histogram for points from the Metropolis Algorithm, GAUSSIAN', fontsize=11)
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Bins', fontsize=14)
plt.show()

# plotting the histogram for the points obtained from metropolis algorithm, EXPONENTIAL
plt.figure()
plt.hist(walk)
plt.title('Histogram for points from the Metropolis Algorithm, EXPONENTIAL', fontsize=11)
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Bins', fontsize=14)
plt.show()