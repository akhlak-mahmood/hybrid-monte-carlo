import numpy as np

class LJ(object):

	def __init__(self):
		pass

	def calculate_force_potential(self, pos, L):
		""" Calculate LJ force and potential
			Based on given position and box length. """
		mass = 1
		N, D = pos.shape

		F = np.zeros((N, D))
		U = np.zeros(N)

		# for each pair
		for i in range(N-1):
			# do not count double
			for j in range(i+1, N):

				# component distance
				sij = pos[i, :] - pos[j, :]

				for d in range(D):
					# component
					if np.abs(sij[d]) > 0.5 * L:
						# pbc distance
						sij[d] = sij[d] - np.copysign(L, sij[d])

				# dot product = r^2
				rij = np.dot(sij, sij)

				if rij < Rc*Rc:

					r2 = 1.0/rij    # 1/r^2
					r6 = r2**3.0    # 1/r^6 
					r12 = r6**2.0   # 1/r^12

					u = 4 * (r12 - r6) - Ucut
					f = 24 * (2.0*r12 - r6) * r2

					U[i] += u / 2
					U[j] += u / 2

					F[i, :] += sij * f
					F[j, :] -= sij * f

		# acceleration, average pot energy
		return F/mass, np.sum(U)/N


	def mc_weight(self, pos, vel, t, u, k):

		# exp of H
		return np.exp(u+k)
