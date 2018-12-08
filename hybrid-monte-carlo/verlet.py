import sys
import numpy as np

def vv_loop(model, L, pos, vel, steps, dt, T=None, incr=0):
	N, D = pos.shape
	a = np.zeros((N, D))

	vol = L * L * L
	density = N/vol

	traj = []
	velocities = []
	tempincr = []

	potential = np.zeros(steps)
	kinetic = np.zeros(steps)
	temp = np.zeros(steps)
	pressure = np.zeros(steps)

	traj.append(pos.copy())
	velocities.append(vel.copy())

	for s in range(steps):
		sys.stdout.write("\rRunning Velocity-Verlet [step {}] ... ".format(s))
		sys.stdout.flush()

		# rebound pbc positions
		for d in range(D):
			indices = np.where(pos[:, d] > L)
			pos[indices, d] -= L
			indices = np.where(pos[:, d] < 0)
			pos[indices, d] += L

		kinetic[s], temp[s] = model.ke_temp(vel)

		# velocity verlet, before force
		pos += vel * dt + 0.5 * a * dt*dt

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

		# velocity verlet, before force
		vel = chi * vel + 0.5 * a * dt

		# find forces
		a, potential[s], virial = model.calculate_force_potential(pos, L)

		pressure[s] = density * temp[s] + virial / vol

		# velocity verlet, after force
		vel += 0.5 * a * dt

		# reset COM velocity
		vcom = np.sum(vel, axis=0)/N
		vel -= vcom/N

		traj.append(pos.copy())
		velocities.append(vel.copy())

	# if increment was not specified,
	# return the measuered temperatures
	if len(tempincr) != steps:
		tempincr = temp

	print('done.')
	
	# list of coords at each timesteps, velocities, temp, PE, KE
	return traj, velocities, tempincr, potential, kinetic, pressure

