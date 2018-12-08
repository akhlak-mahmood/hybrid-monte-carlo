#!/usr/bin/env python3

import sys
import numpy as np

import utils
import plot
from lennardjones import LJ, SIG
from simpleharmonic import SHO

from pdb import set_trace

def HMC(model, L, pos, vel, mc_steps, md_steps, dt):
	""" Return N Metropolis configuration samples from initial
		Positions and momenta """

	traj = []
	velocities = []

	potential = np.zeros(mc_steps)
	kinetic = np.zeros(mc_steps)
	temp = np.zeros(mc_steps)

	# find initial values
	a, u = model.calculate_force_potential(pos, L)
	k, t = model.ke_temp(vel)

	traj.append(pos.copy())
	velocities.append(vel.copy())

	potential[0] = u 
	kinetic[0] = k 
	temp[0] = t
	
	w = model.mc_weight(pos, vel, t, u, k)

	for s in range(1, mc_steps):
		print("Running Metropolis [{}/{}] ... ".format(s, mc_steps))
 
		# make a small change using MD
		Xtraj, velocities, T, U, K = lf_loop(model, L, pos, vel, md_steps, dt)
		
		# negate the momentum/vel to make the proposal symmetric
		trial_pos, trial_vel, trial_temp, trial_U, trial_K = Xtraj[-1], velocities[-1], T[-1], U[-1], K[-1]

		# weight for the trial
		wt = model.mc_weight(trial_pos, -trial_vel, trial_temp, trial_U, trial_K)

		# ratio of the weights = probability of the trial config
		r = wt / w
		
		if r >= 1:
			w = wt
			pos = trial_pos.copy()
			vel = trial_vel.copy()
			t = trial_temp.copy()
			u = trial_U.copy()
			k = trial_K.copy()
		
		else:
			# we are moving to the trial position with probability r
			# i.e. only if the generated random no is less than r
			# eq. r = 0.2, then prob of generating a number less than 0.2 is rand() < 0.2
			if np.random.rand() < r:
				w = wt
				pos = trial_pos.copy()
				vel = trial_vel.copy()
				t = trial_temp.copy()
				u = trial_U.copy()
				k = trial_K.copy()

		traj.append(pos.copy())
		velocities.append(vel.copy())
		potential[s] = u.copy()
		kinetic[s] = k.copy()
		temp[s] = t.copy()

		print('Metropolis done.')

	# Metropolis trajectory, vel, T, U, K
	return traj, velocities, temp, potential, kinetic


def lf_loop(model, L, pos, vel, steps, dt, T=None, incr=0):
	""" Hybrid monte carlo with leapfrog method.
		Return history of the whole MD run and final velocities. """

	N, D = pos.shape
	a = np.zeros((N, D))

	traj = []
	velocities = []
	tempincr = []

	potential = np.zeros(steps)
	kinetic = np.zeros(steps)
	temp = np.zeros(steps)

	print("Running Leapfrog [{} steps] ... ".format(steps), end='')

	traj.append(pos.copy())
	velocities.append(vel.copy())

	a, potential[0] = model.calculate_force_potential(pos, L)
	kinetic[0], temp[0] = model.ke_temp(vel)

	# make momentum half step at the very begining
	# p = p - eps * grad(U)/2
	vel = vel + 0.5 * dt * a

	for s in range(1, steps):
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
			a, potential[s] = model.calculate_force_potential(pos, L)

			# p = p - eps * grad(U)
			vel = chi * vel + dt * a
			velocities.append(vel.copy())


		# reset COM velocity
		vcom = np.sum(vel, axis=0)/N
		vel -= vcom/N


	# make the final p half step
	a, potential[-1] = model.calculate_force_potential(pos, L)

	# p = p - eps * grad(U) / 2
	vel = chi * vel + 0.5 * dt * a

	velocities.append(vel.copy())
	kinetic[-1], temp[-1] = model.ke_temp(vel)

	print('done.')

	return traj, velocities, temp, potential, kinetic

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
	return traj, velocities, pressure, potential, kinetic

def Problem_01():
	D = 3
	# N = 32
	# Rho = 0.841856
	# mass = 1.0

	# V = N * mass / Rho
	# L = np.power(V, 1.0/3.0)


	N = 16
	L = 4

	steps = 3000
	dt = 0.01

	pos = utils.fcc_positions(N, L)
	# plot.pos(pos, L)

	vel = utils.mb_velocities(N, D, 1.5)

	model = LJ()

	X, V, T, U, K = vv_loop(model, L, pos, vel, steps, dt)
	# X, V, T, U, K = HMC(model, L, pos, vel, steps, 5, dt)

	plot.energy(dt, steps, T, U, K)

	# plot.pos(X[-1], L)
	plot.animate3D(X, L, T, steps, dt)

	plot.velocity_distribution(V[-1])


def Problem_02():
	N = 16
	D = 2
	L = 4

	steps = 3000
	dt = 0.005

	pos = crystal_positions()
	# plot.pos(pos, L)
	# plot.radial_distribution(pos, L, L)

	# Set tiny random velocities
	# For normal distribution N(mu, sigma^2)
	# sigma * np.random.randn(...) + mu
	vel = np.random.randn(N, D) / 10000

	model = LJ()

	X, V, T, U, K = lf_loop(model, L, pos, vel, steps, dt)

	plot.energy(dt, steps, T, U, K)

	# rdf of last step
	# plot.radial_distribution(X[-1], L, L)

	plot.animate(X, L, T, steps, dt)

	plot.velocity_distribution(V[-1])

	# save last step positions
	np.savetxt('problem-02.xyz', X[-1])

def Problem_03():
	N = 16
	D = 2
	L = 4

	steps = 5000
	dt = 0.01

	# load problem-2 positions
	pos = np.loadtxt('problem-02.xyz')

	# plot.pos(pos, L)
	# plot.radial_distribution(pos, L, L)

	# Set tiny random velocities
	# For normal distribution N(mu, sigma^2)
	# sigma * np.random.randn(...) + mu
	vel = np.random.randn(N, D) / 10000

	model = LJ()

	X, V, T, U, K = vv_loop(model, L, pos, vel, steps, dt, 0.5, 0.1)

	plot.energy(dt, steps, T, U, K)

	# rdf of last step
	# plot.radial_distribution(X[-1], L, L)

	plot.animate(X, L, T, steps, dt)

	# plot.velocity_distribution(V[-1])


if __name__=='__main__':
	import sys 
	if len(sys.argv) > 1:
		if sys.argv[1] == '1':
			Problem_01()
		elif sys.argv[1] == '2':
			Problem_02()
		elif sys.argv[1] == '3':
			Problem_03()
	else:
		print("Please specify problem number.")

