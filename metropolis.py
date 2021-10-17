import numpy as np
import matplotlib.pyplot as plt

########## FUNCTIONS USED THROUGH THE CODE ##########

# weight function, w(x) = normal distribution with mean 0 and variance 1
def weight(x):
	w = 1 / (np.sqrt(2 * np.pi)) * np.exp(- x**2 / 2)
	return w

# metropolis algorithm implementation
def metrop(x, delta, accpt):
	dirtn = np.random.uniform(-1, 1) # direction variable
	x_t = x + delta * dirtn # trial step

	ratio = weight(x_t) / weight(x) # ratio
	mu = np.random.uniform(0, 1)

	if ratio > mu: # comparing, accept or reject
		x = x_t
		accpt += 1

	return x, accpt

# integrating function F(x) = sqrt(2 * pi) x^2
def funcFofX(x):
	f = x**2 * np.sqrt(2 * np.pi)
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
delta = 2 # delta d
x0 = 0 # starting point
Ntherm = 1000 # number of thermalization steps
Nstep = 10000 # number of sweeps on metropolis algorithm
walk = [] # walk points list
func = [] # function value list
total = np.zeros((3, 1))

print('\n')
print('#'*10 + ' METROPOLIS ALGORITHM ' + '#'*10)

accpt = 0
x_prev = x0
for i in range(Ntherm): # thermalization step
	if i == 0:
		print(' Thermalizing... \n')
	if i == Ntherm - 1:
		print(' Thermalizing... DONE! \n')
	
	x_next, accpt = metrop(x_prev, delta, accpt)
	x_prev = x_next
print(' Thermalizing, ACCEPTANCE RATIO: ', accpt / Ntherm)

accpt = 0
for i in range(Nstep): # metropolis step
	if i == 0:
		print('\n Random Walk... \n')
	if i == Ntherm - 1:
		print(' Random Walk... DONE! \n')
	
	walk.append(x_prev)
	fval = funcFofX(x_prev)
	func.append(fval)
	total[0, 0] += fval
	total[1, 0] += fval**2

	x_next, accpt = metrop(x_prev, delta, accpt)
	x_prev = x_next
print(' Random Walk, ACCEPTANCE RATIO: ', accpt / Nstep)

# calculating the integral and standard deviation  or sigma
total[0, 0] = total[0, 0] / Nstep
total[1, 0] = total[1, 0] / Nstep
total[2, 0] = total[1, 0] - total[0, 0]**2

print('\n Value of the Integral: ', total[0, 0])

# calculating the auto-correlation function
print('\n Auto-Correlation... CALCULATING')
autocorr = np.zeros((8, 1))
autocorr[0, 0] = (corr(func, Nstep, 0) - total[0, 0]**2) / total[2, 0]
print('\n Auto-Correlation... k = 0: ', autocorr[0, 0])

for i in range(autocorr.shape[0]-1):
	if 10**i < Nstep:
		autocorr[i+1, 0] = (corr(func, Nstep, 10**i) - total[0, 0]**2) / total[2, 0]
		if autocorr[i+1, 0] < 0:
			autocorr[i+1, 0] = 0
		print('\n Auto-Correlation... k = 10^{}: {}'.format(i, autocorr[i+1, 0]))

print('\n')
print('#'*10 + ' COMPARISON WITH OTHER METHODS ' + '#'*10)

total2 = np.zeros((3, 1))

# randomly picking a point based on normal distribution with mean 0 and variance 1
# then applying monte carlo simulation to calculate the integral
for i in range(Nstep):
	fval2 = funcFofX(np.random.randn())
	total2[0, 0] += fval2
	total2[1, 0] += fval2**2

# calculating the integral value and standard deviation or sigma
total2[0, 0] = total2[0, 0] / Nstep
total2[1, 0] = total2[1, 0] / Nstep
total2[2, 0] = total2[1, 0] - total2[0, 0]**2

sigI = np.sqrt(total2[2, 0] / Nstep)

print('\n Value of the Integral: ', total2[0, 0])
print('\n Sigma for the Integral: ', sigI)

# plotting the histogram for the points obtained from metropolis algorithm
plt.figure()
plt.hist(walk)
plt.title('Histogram for points from the Metropolis Algorithm', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Bins', fontsize=14)
plt.show()