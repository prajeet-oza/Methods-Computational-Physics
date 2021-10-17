import numpy as np
import matplotlib.pyplot as plt

def dxdt(t0, v0):
	fdxdt = v0
	return fdxdt

def dvdt(t0, x0):
	fdvdt = -x0
	return fdvdt

#defining the initial values for x, v, t
x0i = 1
v0i = 0
t0i = 0
h0 = [0.5, 0.3, 0.01] #insert/change the step size array

#defining time for observation
T0 = 25

#midpoint method, 
x_midpt = {}
v_midpt = {}
for h in h0:
	#initializing the x0, v0, t0 for the step size h
	x0 = x0i
	v0 = v0i
	t0 = t0i
	#initiallize the arrays to store x and v at each iteration
	x0_midpt = [x0]
	v0_midpt = [v0]
	for i in range(int(T0/h)):
		k1x = dxdt(t0, v0)
		k1v = dvdt(t0, x0)

		x1 = x0 + h*dxdt(t0 + h/2, v0 + h*k1x/2)
		v1 = v0 + h*dvdt(t0 + h/2, x0 + h*k1v/2)
		t1 = t0 + h
		x0_midpt.append(x1)
		v0_midpt.append(v1)
		x0 = x1
		v0 = v1
		t0 = t1
	#for each h, storing the arrays of x and v in a dictionary
	x_midpt[h] = x0_midpt.copy()
	v_midpt[h] = v0_midpt.copy()

#runge-kutta method, 
x_rngkt = {}
v_rngkt = {}
T = {}
for h in h0:
	#initializing the x0, v0, t0 for the step size h
	x0 = x0i
	v0 = v0i
	t0 = t0i
	#initiallize the arrays to store x and v at each iteration
	x0_rngkt = [x0]
	v0_rngkt = [v0]
	t0_rngkt = [t0]
	for i in range(int(T0/h)):
		k1x = dxdt(t0, v0)
		k2x = dxdt(t0 + h/2, v0 + h*k1x/2)
		k3x = dxdt(t0 + h/2, v0 + h*k2x/2)
		k4x = dxdt(t0 + h, v0 + h*k3x)

		k1v = dvdt(t0, x0)
		k2v = dvdt(t0 + h/2, x0 + h*k1v/2)
		k3v = dvdt(t0 + h/2, x0 + h*k2v/2)
		k4v = dvdt(t0 + h, x0 + h*k3v)

		x1 = x0 + (h/6)*(k1x + 2*k2x + 2*k3x + k4x)
		v1 = v0 + (h/6)*(k1v + 2*k2v + 2*k3v + k4v)
		t1 = t0 + h
		x0_rngkt.append(x1)
		v0_rngkt.append(v1)
		t0_rngkt.append(t1)
		x0 = x1
		v0 = v1
		t0 = t1
	#for each h, storing the arrays of x and v in a dictionary
	x_rngkt[h] = x0_rngkt.copy()
	v_rngkt[h] = v0_rngkt.copy()
	#storing time in a dictionary, for each h
	T[h] = t0_rngkt.copy()

#plot generation code
methods = ['midpoint', 'runge-kutta']
x0s = {0:x_midpt, 1:x_rngkt}
v0s = {0:v_midpt, 1:v_rngkt}

plt.figure('Second Order Differential Equations', figsize=(20, 30))
for i in range(2):
	txt = methods[i] + ' method'
	plt.subplot(2, 2, i+1)
	plt.title('Solving Simple Harmonic Motion (SHM) (' + methods[i] + ' method)')
	plt.xlabel('t in secs')
	plt.ylabel('x in m')
	for j in h0:
		plt.plot(T[j], x0s[i][j], label='x(t) at h='+str(j))
	plt.legend(loc='upper right', fontsize='small', framealpha=0.7)

for i in range(2):
	txt = methods[i] + ' method'
	plt.subplot(2, 2, i+3)
	plt.title('Solving Simple Harmonic Motion (SHM) (' + methods[i] + ' method)')
	plt.xlabel('t in secs')
	plt.ylabel('v in m/s2')
	for j in h0:
		plt.plot(T[j], v0s[i][j], label='v(t) at h='+str(j))
	plt.legend(loc='upper right', fontsize='small', framealpha=0.7)
plt.show()