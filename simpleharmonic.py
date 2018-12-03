import numpy as np

# 1D SHO
class SHO(object):
	def __init__(self):
		pass 

	def calculate_force_potential(self, x, L):
		pass

	def mc_weight(self, pos, vel, t, u, k):
		H = 0.5 * np.dot(pos, pos) + 0.5 * np.dot(vel, vel)
		# exp of H
		return np.exp(H)

	def ke_temp(self, vel):
		N, D = vel.shape
		v2 = 0
		for i in range(N):
			v2 += np.dot(vel[i, :], vel[i, :])

		kavg = 0.5 * v2 / N

		# average kinetic energy, temperature
		return kavg, 2.0 * kavg / D
