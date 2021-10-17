import numpy as np
import matplotlib.pyplot as plt

#defining the first order derivative of y wrt t
def func(y0, t0):
	#insert/change the function here
	# f = -y0*t0 #case 1; use y0i = 1 and t0i = 0
	# f = -np.exp(-y0-t0) #case 0; use y0i = 0 and t0i = 0
	f = 2 - np.exp(-4*t0) - 2*y0 #case 3; use y0i = 1 and t0i = 0
	return f

#defining the solution function of the first ODE
def func_real(t0):
	#insert/change the function here
	# f = np.exp((-1*t0**2)/2) #case 1; use y0i = 1 and t0i = 0
	# f = -t0 #case 2; use y0i = 0 and t0i = 0
	f = 1 + 0.5*np.exp(-4*t0) - 0.5*np.exp(-2*t0) #case 3; use y0i = 1 and t0i = 0
	return f

#initialise the y0, t0 and set the step size h
y0i = 1
t0i = 0
h0 = [0.5, 0.1] #insert/change the step size array

#define T or limit of t
T0 = 5

#euler method, normal/usual
y_euler = {}
for h in h0:
	#initialise the y0, t0 and set the step size h
	y0 = y0i
	t0 = t0i
	y0_euler = [y0]
	t0_euler = [t0]
	for i in range(int(T0/h)):
		y1 = y0 + h*func(y0, t0)
		t1 = t0 + h
		y0_euler.append(y1)
		t0_euler.append(t1)
		y0 = y1
		t0 = t1
	y_euler[h] = y0_euler.copy()

#midpoint method
y_midpt = {}
for h in h0:
	#initialise the y0, t0 and set the step size h
	y0 = y0i
	t0 = t0i
	y0_midpt = [y0]
	t0_midpt = [t0]
	for i in range(int(T0/h)):
		k1 = func(y0, t0)
		y1 = y0 + h*func(y0 + h*k1/2, t0 + h/2)
		t1 = t0 + h
		y0_midpt.append(y1)
		t0_midpt.append(t1)
		y0 = y1
		t0 = t1
	y_midpt[h] = y0_midpt.copy()

#average method, adams-bashforth method
y_avgmd = {}
for h in h0:
	#initialise the y0, t0 and set the step size h
	y0 = y0i
	t0 = t0i
	y0_avgmd = [y0]
	t0_avgmd = [t0]
	y1 = y0 + h*func(y0, t0)
	t1 = t0 + h
	y0_avgmd.append(y1)
	t0_avgmd.append(t1)
	for i in range(int(T0/h - 1)):
		y2 = y1 + 1.5*h*func(y1, t1) - 0.5*h*func(y0, t0)
		t2 = t1 + h
		y0_avgmd.append(y2)
		t0_avgmd.append(t2)
		y0 = y1
		t0 = t1
		y1 = y2
		t1 = t2
	y_avgmd[h] = y0_avgmd.copy()

#runge-kutta method
y_rngkt = {}
for h in h0:
	#initialise the y0, t0 and set the step size h
	y0 = y0i
	t0 = t0i
	y0_rngkt = [y0]
	t0_rngkt = [t0]
	for i in range(int(T0/h)):
		k1 = func(y0, t0)
		k2 = func(y0 + h*k1/2, t0 + h/2)
		k3 = func(y0 + h*k2/2, t0 + h/2)
		k4 = func(y0 + h*k3, t0 + h)
		y1 = y0 + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
		t1 = t0 + h
		y0_rngkt.append(y1)
		t0_rngkt.append(t1)
		y0 = y1
		t0 = t1
	y_rngkt[h] = y0_rngkt.copy()

#plot generation code begins
Tx = np.arange(t0i, T0+0.001, 0.001) #to generate the solution plot of the ODE
T = {}
for j in h0:
	T[j] = np.arange(t0i, T0+j, j)
methods = ['euler', 'midpoint', 'average', 'runge-kutta']
y0s = {0:y_euler, 1:y_midpt, 2:y_avgmd, 3:y_rngkt}

plt.figure('First Order Differential Equations', figsize=(20, 30))
for i in range(4):
	txt = methods[i] + ' method'
	plt.subplot(2, 2, i+1)
	plt.title('Solving first order Differential Equations (' + methods[i] + ' method)')
	plt.xlabel('t / units')
	plt.ylabel('y / units')
	plt.plot(Tx, func_real(Tx), label='solution to first ODE')
	for j in h0:
		plt.plot(T[j], y0s[i][j], label=txt+' h='+str(j))
	plt.legend(loc='upper right', fontsize='small', framealpha=0.7)
plt.show()