#!/usr/bin/env python3

import sys
import numpy as np

import utils
import plot
from lennardjones import LJ, SIG
from simpleharmonic import SHO
from verlet import vv_loop
from leapfrog import lf_loop

from pdb import set_trace

def HMC(model, MC, md_steps,
	target_temp=None, target_pres=None):
	""" Return N Metropolis configuration samples from initial
		Positions and momenta """

	L = MC['length']
	mc_steps = MC['steps']
	dt = MC['timestep']

	pos = MC['position'][0]
	vel_orig = MC['velocity'][0]

	assert pos.shape == vel_orig.shape

	N, D = pos.shape

	vol = MC['volume'][0]
	density = MC['density'][0]

	# find initial values
	a, pot, vir = model.calculate_force_potential(pos, L)
	ke, temp = model.ke_temp(vel_orig)
	pres = density * temp + vir / vol

	MC['force'].append(a.copy())
	MC['potential'][0] = pot.copy()
	MC['virial'][0] = vir
	MC['kinetic'][0] = ke
	MC['temperature'][0] = temp
	MC['target_temp'][0] = target_temp
	MC['pressure'][0] = pres.copy()
	MC['target_pres'][0] = target_pres

	MC['accepted'] = 0
	MC['rejected'] = 0

	# calculate initial weight
	w = model.mc_weight(pos, vel_orig, temp, pot, ke)

	for s in range(1, mc_steps):
		sys.stdout.write("\rRunning Hybrid [step {0}] acceptance {1:.2f} ... ".format(s+1,
			MC['accepted']/s))
		sys.stdout.flush()

		# markov chain, reinit velocities
		vel = utils.mb_velocities(N, D, 1.5)

		# trial = make a small change using MD

		DYN = utils.init_dynamics(pos, vel, md_steps, dt, L)
		NVT = lf_loop(model, DYN,
			target_temp=target_temp, increment_factor=5)

		NPT = utils.restart_dynamics(NVT, md_steps, dt)
		MD = lf_loop(model, NPT,
			target_temp=target_temp,
			target_pres=target_pres, increment_factor=5)


		# weight for the trial
		wt = model.mc_weight(MD['position'][-1], MD['velocity'][-1],
			MD['temperature'][-1], MD['potential'][-1], MD['kinetic'][-1])

		# ratio of the weights = probability of the trial config
		r = wt / w
		
		# we are moving to the trial position with probability r
		# i.e. if r >= 1 or the generated random number is less than r
		if r >= 1 or np.random.rand() < r:
			w = wt
			pos = MC['position'][-1].copy()
			MC['accepted'] += 1
			MC['position'].append(MD['position'][-1].copy())
			MC['velocity'].append(MD['velocity'][-1].copy())
			MC['force'].append(MD['force'][-1].copy())

			MC['potential'][s] = MD['potential'][-1].copy()
			MC['kinetic'][s] = MD['kinetic'][-1].copy()
			MC['temperature'][s] = MD['temperature'][-1].copy()
			MC['pressure'][s] = MD['pressure'][-1].copy()
			MC['virial'][s] = MD['virial'][-1].copy()

		# MC rejected, use the previous values
		else:
			MC['rejected'] += 1
			MC['position'].append(MC['position'][-1].copy())
			MC['velocity'].append(MC['velocity'][-1].copy())
			MC['force'].append(MC['force'][-1].copy())

			MC['potential'][s] = MC['potential'][s-1].copy()
			MC['kinetic'][s] = MC['kinetic'][s-1].copy()
			MC['temperature'][s] = MC['temperature'][s-1].copy()
			MC['pressure'][s] = MC['pressure'][s-1].copy()
			MC['virial'][s] = MC['virial'][s-1].copy()			

		MC['target_temp'][s] = target_temp
		MC['density'][s] = density
		MC['volume'][s] = vol

	sys.stdout.write("\rMetropolis done [{} steps].\n".format(s))
	sys.stdout.flush()

	return MC


def acceptance_run(dt, md_steps):
	D = 3
	N = 16
	L = 4

	steps = 1000
	print("Running dt = {0}, steps = {1}  ... ".format(dt, md_steps))

	pos = utils.fcc_positions(N, L)
	vel = utils.mb_velocities(N, D, 1.5)
	model = LJ()
	DYN = utils.init_dynamics(pos, vel, steps, dt, L)

	res = HMC(model, DYN, md_steps)['accepted']

	print((dt, md_steps, res))

	return res

def main():

	D = 3
	N = 32
	mass = 1.0

	steps = 100
	dt = 0.01

	density = 0.5
	vol = N * mass / density
	L = np.power(vol, 1.0/D)

	pos = utils.fcc_positions(N, L)
	vel = utils.mb_velocities(N, D, 1.5)

	DYN = utils.init_dynamics(pos, vel, steps, dt, L)
	print('length = {}, volume = {}, density = {}'.format(DYN['length'], DYN['volume'][0], DYN['density'][0]))
	# plot.pos(pos, L)
	input('Press ENTER to continue ... ')

	model = LJ()

	# NTP = lf_loop(model, DYN, target_temp=1, target_pres=2)
	# np.save('npt-lf.npy', NTP)

	NTP = HMC(model, DYN, 15, 1, 2)
	np.save('npt-hmc.npy', NTP)


	plot.energy(NTP)
	plot.velocity_distribution(NTP['velocity'][-1])

	# plot.pos(NTP['position'][-1], L)

	# plot.radial_distribution(NTP['position'][-1], L)

	# plot.animate3D(NTP['position'], L, NTP['temperature'], steps, dt)


def run():
	res = []
	for dt in np.arange(0.01, 0.5, 0.01):
		for steps in np.arange(5, 21, 1):
			acceptance = acceptance_run(dt, steps)
			res.append((dt, steps, acceptance))

	np.save('acceptance.npy', res)


main()
