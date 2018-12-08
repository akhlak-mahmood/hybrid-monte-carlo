import sys
import numpy as np


def lf_loop(model, L, pos, vel, steps, dt, T=None, incr=0):
	""" Hybrid monte carlo with leapfrog method.
		Return history of the whole MD run and final velocities. """

	N, D = pos.shape
	a = np.zeros((N, D))

	vol = L * L * L 
	density = N / vol

	traj = []
	velocities = []
	tempincr = []

	potential = np.zeros(steps)
	kinetic = np.zeros(steps)
	temp = np.zeros(steps)
	pressure = np.zeros(steps)

	traj.append(pos.copy())
	velocities.append(vel.copy())

	sys.stdout.write("Running Leapfrog [step {}] ... ".format(0))
	sys.stdout.flush()

	a, potential[0], virial = model.calculate_force_potential(pos, L)
	kinetic[0], temp[0] = model.ke_temp(vel)
	pressure[0] = density * temp[0] + virial / vol

	# make momentum half step at the very begining
	# p = p - eps * grad(U)/2
	vel = vel + 0.5 * dt * a

	for s in range(1, steps):
		sys.stdout.write("\rRunning Leapfrog [step {}] ... ".format(s))
		sys.stdout.flush()

		# rebound pbc positions
		for d in range(D):
			indices = np.where(pos[:, d] > L)
			pos[indices, d] -= L
			indices = np.where(pos[:, d] < 0)
			pos[indices, d] += L

		kinetic[s], temp[s] = model.ke_temp(vel)

		# make full q step
		# q = q + eps * p 
		pos = pos + dt * vel 
		traj.append(pos.copy())

		if T is None:
			# NVE Ensemble
			chi = 1
		else:
			# if temperature increment is specified
			# and not the 0th step
			# increment temperature every unit time
			if incr and s and (dt * s) % 1 == 0:
				T += incr
				print('t =', dt*s, 'T =', T)
			tempincr.append(T)

			# NVT Ensemble
			# velocity rescale factor
			chi = np.sqrt(T/temp[s])

		# make p full step, if not the last one
		if s < steps - 1:
			a, potential[s], virial = model.calculate_force_potential(pos, L)
			pressure[s] = density * temp[s] + virial / vol

			# p = p - eps * grad(U)
			vel = chi * vel + dt * a
			velocities.append(vel.copy())


		# reset COM velocity
		vcom = np.sum(vel, axis=0)/N
		vel -= vcom/N


	# make the final p half step
	a, potential[-1], virial = model.calculate_force_potential(pos, L)
	pressure[-1] = density * temp[-1] + virial / vol

	# p = p - eps * grad(U) / 2
	vel = chi * vel + 0.5 * dt * a

	velocities.append(vel.copy())
	kinetic[-1], temp[-1] = model.ke_temp(vel)

	print('done.')

	return traj, velocities, temp, potential, kinetic, pressure

