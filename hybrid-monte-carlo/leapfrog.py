import sys
import numpy as np

np.seterr(all='raise')

def lf_loop(model, MD, target_temp=None, t_inc=0, target_pres=None, p_inc=0, increment_factor=20.0):
	""" Hybrid monte carlo with leapfrog method.
		Return history of the whole MD run and final velocities. """

	L = MD['length']
	steps = MD['steps']
	dt = MD['timestep']

	pos = MD['position'][0]
	vel = MD['velocity'][0]

	assert pos.shape == vel.shape

	N, D = pos.shape

	vol = L * L * L 
	density = N / vol
	a, pot, vir = model.calculate_force_potential(pos, L)
	ke, temp = model.ke_temp(vel)
	pres = density * temp + vir / vol

	MD['force'].append(a.copy())
	MD['potential'][0] = pot.copy()
	MD['virial'][0] = vir
	MD['kinetic'][0] = ke
	MD['temperature'][0] = temp
	MD['target_temp'][0] = target_temp
	MD['density'][0] = density
	MD['volume'][0] = vol 
	MD['pressure'][0] = pres.copy()
	MD['target_pres'][0] = target_pres


	# make momentum half step at the very begining
	# p = p - eps * grad(U)/2
	vel = vel + 0.5 * dt * a

	for s in range(1, steps):
		if s % 100 == 0:
			sys.stdout.write("\rRunning Leapfrog [step {}] ... ".format(s+1))
			sys.stdout.flush()

		# rebound pbc positions
		for d in range(D):
			indices = np.where(pos[:, d] > L)
			pos[indices, d] -= L
			indices = np.where(pos[:, d] < 0)
			pos[indices, d] += L

		ke, temp = model.ke_temp(vel)
		MD['kinetic'][s] = ke
		MD['temperature'][s] = temp

		# make full q step
		# q = q + eps * p 
		pos = pos + dt * vel 
		MD['position'].append(pos.copy())

		if target_temp is None:
			# NVE Ensemble
			chi = 1
		else:
			# if temperature increment is specified
			# and not the 0th step
			# increment temperature every unit time
			if t_inc and s > 1 and (dt * s) % 1 == 0:
				target_temp += t_inc
				print('t =', dt*s, 'T =', target_temp)

			# NVT Ensemble
			# velocity rescale factor

			# increase very slowly, otherwise it will be chaos
			temp_inc = (target_temp - temp) / increment_factor
			chi = np.sqrt(temp_inc/temp)


		# record target temp for the current step
		MD['target_temp'][s] = temp_inc

		density = N / vol
		pres = density * temp + vir / vol
		MD['pressure'][s] = pres.copy()

		if target_pres is None:
			# No pressure control
			pass
		else:
			# if pressure increment is specified
			# and not the 0th step
			# increment pressure every unit time
			if p_inc and s > 1 and (dt * s) % 1 == 0:
				target_pres += p_inc
				print('t =', dt*s, 'P =', target_pres)

			# Barostat
			# increase very slowly, otherwise it will be chaos
			pres_inc = (target_pres - pres) / increment_factor
			target_vol = vir / (pres + pres_inc - density * temp)
			target_length = np.power(np.abs(target_vol), 1.0/D)
			psi = target_length / L

			# rescale the coordinates
			pos = psi * pos

			# rescale length
			L = target_length

		# record target pressure for the current step
		MD['target_pres'][s] = pres_inc

		# recalculate 
		vol = L ** D
		density = N / vol

		MD['density'][s] = density
		MD['volume'][s] = vol

		# make p full step, if not the last one
		if s < steps - 1:
			a, pot, vir = model.calculate_force_potential(pos, L)
			vel = chi * vel + dt * a

			MD['force'].append(a.copy())
			MD['potential'][s] = pot.copy()
			MD['virial'][s] = vir
			MD['velocity'].append(vel.copy())


		# reset COM velocity
		vcom = np.sum(vel, axis=0)/N
		vel -= vcom/N


	# make the final p half step
	a, pot, vir = model.calculate_force_potential(pos, L)

	# p = p - eps * grad(U) / 2
	vel = chi * vel + 0.5 * dt * a

	MD['force'].append(a.copy())
	MD['potential'][s] = pot.copy()
	MD['virial'][s] = vir
	MD['velocity'].append(vel.copy())

	if s % 100 == 0:
		print('done.')

	return MD

