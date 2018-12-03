import numpy as np

class SHO(object):
	def __init__(self):
		pass 

	def calculate_force_potential(self, pos, L):
		pass

	def mc_weight(self, pos, vel, t, u, k):

		# exp of H
		return np.exp(u+k)
