#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from scipy.stats import maxwell, linregress

from pdb import set_trace

Rc = 3

SIG = 1
EPS = 1

# one time calculations
rc6 = 1.0 / (Rc**6)
Ucut = 4 * ((rc6 * rc6) - rc6)

# graphs setup
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('axes', labelsize=18)
plt.rc('figure', figsize=(12,12))

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

def temperature_kinetic(vel):
	N, D = vel.shape
	v2 = 0
	for i in range(N):
		v2 += np.dot(vel[i, :], vel[i, :])

	kavg = 0.5 * v2 / N

	# average kinetic energy, temperature
	return kavg, 2.0 * kavg / D

def plot_energy(dt, steps, U, K, T):
	time = [dt * i for i in range(steps)]

	plt.subplot(4,1,1)
	plt.plot(time, K)
	plt.ylabel('$E_k$')

	plt.subplot(4,1,2)
	plt.plot(time, U)
	plt.ylabel('$E_p$')

	plt.subplot(4,1,3)
	plt.plot(time, U+K)
	plt.ylabel('$E_{tot}$')

	plt.subplot(4,1,4)
	plt.plot(time, T)
	plt.ylabel('$Temperature$')

	plt.xlabel('$Time$')
	plt.show()

def plot_msd(dt, steps, R, D):
	time = [dt * i for i in range(steps)]

	res = linregress(time, R)
	print("Diffusion Constant =  {}".format(res.slope / 2 / D))

	plt.subplot(1,1,1)
	plt.plot(time, R)
	plt.ylabel('$r^2(t)$')
	plt.xlabel('$Time$')
	plt.show()

def plot_msd_temp(T, R):
	x = []
	y = []

	for i, t in enumerate(T):
		if t not in x:
			x.append(t)
			y.append(R[i])

	plt.subplot(1,1,1)
	plt.plot(x, y)
	plt.ylabel('$r^2(t)$')
	plt.xlabel('$Temperature$')
	plt.grid()
	plt.show()

def plot_velocity_distribution(vel):
	N, D = vel.shape
	# Plot final velocity ditribution
	# Should be a M-B distribution
	v = np.zeros(N)
	for n in range(N):
		v[n] = np.sqrt(np.dot(vel[n], vel[n]))


	# plot density normalized speed distribution 
	plt.title("$Final\; velocity\; distribution$")
	plt.hist(v, bins=30, density=True)

	# # fit the velocity data with M-B dist
	# params = maxwell.fit(v, floc=0)
	# print('Maxwell fit parameters = ', params)
	# # plot maxwell 
	# x = np.linspace(0, 3*max(v), 100)
	# plt.plot(x, maxwell.pdf(x, *params), lw=3)

	plt.show()

def radial_distribution(pos, L, resolution=100):
	N, D = pos.shape
	volume = L**D
	density = N/volume

	rdf = []

	for i, r in enumerate(np.linspace(0, L, resolution)):
		rn = r + L/resolution
		r2 = r * r
		rn2 = rn * rn
		count = 0
		for n in range(1, N):
			d = pos[0,:] - pos[n, :]
			d2 = np.dot(d, d)
			if d2 > r2 and d2 < rn2:
				count += 1
		if count > 0:
			rdf.append([r, count/density])

	plt.plot([r[0] for r in rdf], [c[1] for c in rdf], 'k-')
	plt.ylabel("$g(r)$")
	plt.xlabel("$r$")
	plt.ylim(0, max([c[1] for c in rdf])+1)
	plt.show()

def animate(traj, L, T, steps, dt, intv=10):
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(xlim=(0, L), ylim=(0, L))
    ax.add_patch(Rectangle((0,0),L,L,linewidth=2,edgecolor='b',facecolor='none'))
    fno = ax.text(0.8*L, L-0.5, 'frame', fontsize=16)
    temp = ax.text(0.8*L, 0.25, 'temp', fontsize=16)
    line, = ax.plot([], [], 'ro', markersize=22)

    def frame(i):
        pos = traj[i]
        line.set_data(pos[:,0], pos[:,1])
        fno.set_text("t: {0:.1f}".format(i*dt))
        temp.set_text("T: {0:.1f}".format(T[i]))
        return line, fno, temp,

    anim = animation.FuncAnimation(fig, frame,
                               frames=steps, interval=intv, blit=True)
    plt.show()

def calculate_force_potential(pos, L):
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

				r2 = 1.0/rij	# 1/r^2
				r6 = r2**3.0	# 1/r^6	
				r12 = r6**2.0	# 1/r^12

				u = 4 * (r12 - r6) - Ucut
				f = 24 * (2.0*r12 - r6) * r2

				U[i] += u / 2
				U[j] += u / 2

				F[i, :] += sij * f
				F[j, :] -= sij * f

	# acceleration, average pot energy
	return F/mass, np.sum(U)/N


def Metropolis(qi, pi, L, N, MD, dt):
    """ Return N Metropolis configuration samples from initial
    	Positions and momenta """
    q = qi
    cfg = [qi]

    # we need the very first values to start MD
    a, U = calculate_force_potential(q, L)

    for i in range(N):
 
        # make a small change of config
        # we will call hmc here
        qt = q + np.random.randint(0,2) - 1

        # run MD and get final values
        
        # calculate the weights
        wq = initial_distribution(q)
        wt = initial_distribution(qt)

        # ratio of the weights = probability of the trial config
        r = wt / wq
        
        if r >= 1:
            q = qt
        
        else:
            # we are moving to the trial position with probability r
            # i.e. only if the generated random no is less than r
            # eq. r = 0.2, then prob of generating a number less than 0.2 is rand() < 0.2
            if np.random.rand() < r:
                q = qt

        cfg.append(q)

    return cfg


def check_lf(L, pos, vel, steps, dt, T=None, incr=0):
	""" Check LF method without using Metropolis """
	N, D = pos.shape
	a = np.zeros((N, D))

	traj = []
	tempincr = []

	potential = np.zeros(steps+1)
	kinetic = np.zeros(steps+1)
	temp = np.zeros(steps+1)

	a, potential[0] = calculate_force_potential(pos, L)
	kinetic[0], temp[0] = temperature_kinetic(vel)
	traj.append(pos)

	for i in range(steps):
		pos, vel, a, potential[i+1] = lf_loop(L, 1, dt, pos, vel, a, potential[i])
		kinetic[i+1], temp[i+1] = temperature_kinetic(vel)
		traj.append(pos)

		# reset COM velocity
		vcom = np.sum(vel, axis=0)/N
		vel -= vcom/N

	plot_energy(dt, steps+1, potential, kinetic, temp)

	return traj, vel, temp


def lf_loop(L, pos, vel, steps, dt, T=None, incr=0):
	""" Hybrid monte carlo with leapfrog method.
		Return MD position and momenta after <steps> steps. """

	N, D = pos.shape
	a = np.zeros((N, D))

	traj = []
	tempincr = []

	potential = np.zeros(steps)
	kinetic = np.zeros(steps)
	temp = np.zeros(steps)

	print("Running dynamics [{} steps] ... ".format(steps), end='')

	a, potential[0] = calculate_force_potential(pos, L)
	kinetic[0], temp[0] = temperature_kinetic(vel)

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

		traj.append(pos.copy())

		kinetic[s], temp[s] = temperature_kinetic(vel)

		# make full q step
		# q = q + eps * p 
		pos = pos + dt * vel 

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
			a, potential[s] = calculate_force_potential(pos, L)

			# p = p - eps * grad(U)
			vel = chi * vel + dt * a


		# reset COM velocity
		vcom = np.sum(vel, axis=0)/N
		vel -= vcom/N


	# make the final p half step
	a, potential[-1] = calculate_force_potential(pos, L)

	# p = p - eps * grad(U) / 2
	vel = chi * vel + 0.5 * dt * a

	kinetic[-1], temp[-1] = temperature_kinetic(vel)

	print('done.')

	return traj, vel, temp, potential, kinetic

def vv_loop(L, pos, vel, steps, dt, T=None, incr=0):
	N, D = pos.shape
	a = np.zeros((N, D))

	traj = []
	tempincr = []

	potential = np.zeros(steps)
	kinetic = np.zeros(steps)
	temp = np.zeros(steps)

	print("Running dynamics [{} steps] ... ".format(steps))

	for s in range(steps):
		# rebound pbc positions
		for d in range(D):
			indices = np.where(pos[:, d] > L)
			pos[indices, d] -= L
			indices = np.where(pos[:, d] < 0)
			pos[indices, d] += L

		traj.append(pos.copy())

		kinetic[s], temp[s] = temperature_kinetic(vel)

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
		a, potential[s] = calculate_force_potential(pos, L)

		# velocity verlet, after force
		vel += 0.5 * a * dt

		# reset COM velocity
		vcom = np.sum(vel, axis=0)/N
		vel -= vcom/N

	# if increment was not specified,
	# return the measuered temperatures
	if len(tempincr) != steps:
		tempincr = temp
	
	# list of coords at each timesteps, final velocities, temp
	return traj, vel, tempincr, potential, kinetic

def plot_pos(pos, L):
	plt.plot(pos[:, 0], pos[:, 1], 'ko', markersize=22)
	plt.xlim(0, L)
	plt.ylim(0, L)
	plt.show()


def Problem_01():
	N = 16
	D = 2
	L = 10

	steps = 3000
	dt = 0.01

	pos = random_positions(N, D, L)
	plot_pos(pos, L)

	vel = mb_velocities(N, D, 1.5)

	X, vel, T, U, K = vv_loop(L, pos, vel, steps, dt)

	plot_energy(dt, steps, U, K, T)

	animate(X, L, T, steps, dt)

	plot_velocity_distribution(vel)


def Problem_02():
	N = 16
	D = 2
	L = 4

	steps = 3000
	dt = 0.005

	pos = crystal_positions()
	# plot_pos(pos, L)
	# radial_distribution(pos, L, L)

	# Set tiny random velocities
	# For normal distribution N(mu, sigma^2)
	# sigma * np.random.randn(...) + mu
	vel = np.random.randn(N, D) / 10000

	X, vel, T, U, K = lf_loop(L, pos, vel, steps, dt)

	plot_energy(dt, steps, U, K, T)

	# rdf of last step
	# radial_distribution(X[-1], L, L)

	animate(X, L, T, steps, dt)

	plot_velocity_distribution(vel)

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

	# plot_pos(pos, L)
	# radial_distribution(pos, L, L)

	# Set tiny random velocities
	# For normal distribution N(mu, sigma^2)
	# sigma * np.random.randn(...) + mu
	vel = np.random.randn(N, D) / 10000

	X, vel, T, U, K = lf_loop(L, pos, vel, steps, dt, 0.5, 0.1)

	plot_energy(dt, steps, U, K, T)

	# rdf of last step
	# radial_distribution(X[-1], L, L)

	animate(X, L, T, steps, dt)

	# plot_velocity_distribution(vel)


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

