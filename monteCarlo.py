import numpy as np

########## FUNCTIONS USED THROUGH THE CODE ##########

# function for change of variable from X to Y
def funcXofY(y):
	x = 2 - np.sqrt(4 - 3*y)
	return x

# original function F(x) = 1 / (1 + x^2)
def funcFofX(x):
	f = 1 / (1 + x**2)
	return f

# weight function w(x) = (4 - 2x) / 3
def funcWofX(x):
	w = (4 - 2*x) / 3
	return w


########## MAIN CODE ##########

power10 = 6 	# powers of 10
total = np.zeros((power10, 2))	# sum the function value for the iterations
totalsq = np.zeros((power10, 2))	# sum the square of function for the iterations
sig = np.zeros((power10, 2))	# sigma or standard deviation or sqrt(variance)
val = np.zeros((power10, 2))	# value of the integral

# y0 = np.random.uniform(0, 1, 10**power10)

for i in range(power10): # for each power of 10, until the last mentioned power
	n = i + 1
	step_limit = 10**n
	for j in range(step_limit):
		# y = y0[j]
		# y = np.random.uniform(0, 1)
		
		# generate y with uniform distribution,
		# change the variable to x,
		# calculate the function value and weight
		y = np.random.random()
		x = funcXofY(y)
		f = funcFofX(x)
		w = funcWofX(x)
		fbw = f / w

		# without weight function case, sum and square sum
		total[i, 0] += f
		totalsq[i, 0] += f**2

		# with weight function case, sum and square sum
		total[i, 1] += fbw
		totalsq[i, 1] += fbw**2

	# calculate the integral value and sigma value
	intg0 = total[i, 0] / step_limit
	intg1 = total[i, 1] / step_limit
	intgsq0 = totalsq[i, 0] / step_limit
	intgsq1 = totalsq[i, 1] / step_limit
	sigma0 = np.sqrt(intgsq0 - intg0**2) / np.sqrt(step_limit)
	sigma1 = np.sqrt(intgsq1 - intg1**2) / np.sqrt(step_limit)

	val[i, 0] = intg0
	val[i, 1] = intg1

	sig[i, 0] = sigma0
	sig[i, 1] = sigma1

# displaying the results
print('N \t| I w/o w(x) \t| sigI w/o w(x) | I with w(x) \t| sigI with w(x)')
print('-'*70)
for i in range(power10):
	print('10^{} \t| {:.5f} \t| {:.5f} \t| {:.5f} \t| {:.5f}'.format(i+1, val[i, 0], sig[i, 0], val[i, 1], sig[i, 1]))
