import numpy as np
import matplotlib.pyplot as plt

#alpha*tan(alpha*a) == beta
def evenfnc(a, V0, E):
	alpha = np.sqrt(0.264*E)
	beta = np.sqrt(0.264*(V0-E))
	val = alpha*np.sin(alpha*a) - beta*np.cos(alpha*a)
	return val

#alpha*cot(alpha*a) == -beta
def oddfnc(a, V0, E):
	alpha = np.sqrt(0.264*E)
	beta = np.sqrt(0.264*(V0-E))
	val = alpha*np.cos(alpha*a) + beta*np.sin(alpha*a)
	return val

#wavefunction at left of -a
def WV1(D, V0, E, x):
	beta = np.sqrt(0.264*(V0-E))
	val = D*np.exp(beta*x)
	return val

#wavefunction between -a and a
def WV2(A, B, E, x):
	alpha = np.sqrt(0.264*E)
	val = A*np.sin(alpha*x) + B*np.cos(alpha*x)
	return val

#wavefunction at right of a
def WV3(F, V0, E, x):
	beta = np.sqrt(0.264*(V0-E))
	val = F*np.exp(-beta*x)
	return val

print('enter the length of the well (2a) in angstrom: ')
temp = int(input())
a = temp / 2

print('enter the potential bound for the well (V0) in eV: ')
V0 = int(input())

#even parity solutions / eigen values
evenroots = []
evena0 = []
evenb0 = []
#identifying a0 and b0 between 0ev and 1eV
for i in range(0, 10):
	x = i*0.1
	if evenfnc(a, V0, x)*evenfnc(a, V0, x+0.1) < 0:
		evena0.append(x)
		evenb0.append(x+0.1)
	elif evenfnc(a, V0, x) == 0:
		evenroots.append(x)
#identifying a0 and b0 between 1eV and 'V0'eV
for i in range(1, V0):
	if evenfnc(a, V0, i)*evenfnc(a, V0, i+1) < 0:
		evena0.append(i)
		evenb0.append(i+1)
	elif evenfnc(a, V0, i) == 0:
		evenroots.append(i)
#using bisection method to find roots using the identified regions
for i in range(len(evena0)):
	a0e = evena0[i]
	b0e = evenb0[i]
	while abs(a0e-b0e)/2 > 10**-3:
		c0e = a0e + (b0e - a0e)/2
		if evenfnc(a, V0, a0e)*evenfnc(a, V0, c0e) < 0:
			b0e = c0e
		elif evenfnc(a, V0, b0e)*evenfnc(a, V0, c0e) < 0:
			a0e = c0e
	roote = a0e + (b0e - a0e)/2
	evenroots.append(roote)

print(' ')
print('Eigen Values, Even Parity: ', evenroots)

#odd parity solutions / eigen values
oddroots = []
odda0 = []
oddb0 = []
#identifying a0 and b0 between 0ev and 1eV
for i in range(0, 10):
	x = i*0.1
	if oddfnc(a, V0, x)*oddfnc(a, V0, x+0.1) < 0:
		odda0.append(x)
		oddb0.append(x+0.1)
	elif oddfnc(a, V0, x) == 0:
		oddroots.append(x)
#identifying a0 and b0 between 1eV and 'V0'eV
for i in range(1, V0):
	if oddfnc(a, V0, i)*oddfnc(a, V0, i+1) < 0:
		odda0.append(i)
		oddb0.append(i+1)
	elif oddfnc(a, V0, i) == 0:
		oddroots.append(i)
#using bisection method to find roots using the identified regions
for i in range(len(odda0)):
	a0o = odda0[i]
	b0o = oddb0[i]
	while abs(a0o-b0o)/2 > 10**-3:
		c0o = a0o + (b0o - a0o)/2
		if oddfnc(a, V0, a0o)*oddfnc(a, V0, c0o) < 0:
			b0o = c0o
		elif oddfnc(a, V0, b0o)*oddfnc(a, V0, c0o) < 0:
			a0o = c0o
	rooto = a0o + (b0o - a0o)/2
	oddroots.append(rooto)

print(' ')
print('Eigen Values, Odd Parity: ', oddroots)

#set A and B in the wavefunction as 1
A = 1
B = 1
#calculate F for the wavefunction
Feven = []
#even parity, D == F
for E in evenroots:
	alpha = np.sqrt(0.264*E)
	beta = np.sqrt(0.264*(V0-E))
	tempF = B * np.cos(alpha*a) * np.exp(beta*a)
	Feven.append(tempF)

Fodd = []
#odd parity, D == -F
for E in oddroots:
	alpha = np.sqrt(0.264*E)
	beta = np.sqrt(0.264*(V0-E))
	tempF = A * np.sin(alpha*a) * np.exp(beta*a)
	Fodd.append(tempF)

#plot generation codes
t1 = np.arange(-3*a, -a+0.1, 0.1)
t2 = np.arange(-a, a+0.1, 0.1)
t3 = np.arange(a, 3*a+0.1, 0.1)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728', '#8c564b', '#210982', 
			'#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#119c0f', '#25d1c4', '#ef4e9f']

f1 = plt.figure('Wavefunctions, Even Parity solution')
for i in range(len(evenroots)):
	txt = 'eigenval ' + str(evenroots[i])
	plt.title('Wavefunctions, Even Parity solution')
	plt.xlabel('x in angstrom')
	plt.ylabel('psi wavefunction')
	plt.plot(t1, WV1(Feven[i], V0, evenroots[i], t1), colors[i], label=txt)
	plt.plot(t2, WV2(0, B, evenroots[i], t2), colors[i])
	plt.plot(t3, WV3(Feven[i], V0, evenroots[i], t3), colors[i])
plt.legend(loc='upper right', fontsize='small', framealpha=0.7)
# plt.show()

f2 = plt.figure('Wavefunctions, Odd Parity solution')
for i in range(len(oddroots)):
	txt = 'eigenval ' + str(oddroots[i])
	plt.title('Wavefunctions, Odd Parity solution')
	plt.xlabel('x in angstrom')
	plt.ylabel('psi wavefunction')
	plt.plot(t1, WV1(-Fodd[i], V0, oddroots[i], t1), colors[i], label=txt)
	plt.plot(t2, WV2(A, 0, oddroots[i], t2), colors[i])
	plt.plot(t3, WV3(Fodd[i], V0, oddroots[i], t3), colors[i])
plt.legend(loc='upper right', fontsize='small', framealpha=0.7)
plt.show()