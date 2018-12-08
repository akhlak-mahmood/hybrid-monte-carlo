
import numpy as np 
from scipy.stats import maxwell


def init_dynamics(L, pos, vel, steps=1000, dt=0.01):
	# initialize an empty dictionary with all the values 
	# we will be calculating

	return {
		'length': L,
		'timestep': dt,
		'steps': steps,
		'position': [pos.copy()],
		'velocity': [vel.copy()],
		'force': [],
		'temperature': np.zeros(steps),
		'target_temp': np.zeros(steps),
		'kinetic': np.zeros(steps),
		'potential': np.zeros(steps),
		'pressure': np.zeros(steps),
		'virial': np.zeros(steps),
		'volume': np.zeros(steps),
		'density': np.zeros(steps)
	}


def fcc_positions(N, L):
	M = 1
	n = 0

	pos = np.zeros((N, 3))

	while 4*M*M*M < N:
	    M += 1
	    
	a = L/M

	xCell = np.array([0.25, 0.75, 0.75, 0.25])
	yCell = np.array([0.25, 0.75, 0.25, 0.25])
	zCell = np.array([0.25, 0.25, 0.75, 0.75])

	for x in range(M):
	    for y in range(M):
	        for z in range(M):
	            for k in range(4):
	                if n < N:
	                    pos[n][0] = (x + xCell[k]) * a
	                    pos[n][1] = (y + yCell[k]) * a
	                    pos[n][2] = (z + zCell[k]) * a
	                    n += 1
	return pos


def no_overlap(D, pos, i):
	""" Check if there's overlap between ith atoms
		With the other atoms. """

	for j in range(0, i):
		r2 = 0
		for d in range(D):
			dx = pos[i, d] - pos[j, d]
			r2 += dx * dx

		if r2 < (1.16 * SIG)**2:
			return False
	return True

def random_positions(N, D, L):
	pos = -1 * np.zeros((N, D))

	i = 0
	while True:
		for d in range(D):
			pos[i, d] = np.random.randint(1, L)

		if no_overlap(D, pos, i):
			i += 1

		if i == N:
			break

	return pos

def crystal_positions():
	pos = -1 * np.zeros((16, 2))
	for n in range(16):
		pos[n][0] = int(n/4) + 0.5
		pos[n][1] = (n % 4) + 0.5

	return pos

def mb_velocities(N, D, vmax):
	vel = np.zeros((N, D))

	for i in range(N):
		for d in range(D):
			while True:
				v = maxwell.rvs(loc=0, scale=1) - vmax
				if abs(v) <= vmax:
					vel[i, d] = v
					break
	return vel

