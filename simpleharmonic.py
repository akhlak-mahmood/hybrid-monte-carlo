import numpy as np

# 1D SHO
class SHO(object):
	def __init__(self):
		pass 

	def calculate_force_potential(self, x, L):
		return -x, 0.5 * x *x

	def mc_weight(self, pos, vel, t, u, k):
		return np.exp(u+k)

	def ke_temp(self, v):
		k = 0.5 * v * v 
		return k, 2*k
