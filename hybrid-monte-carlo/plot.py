import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.patches import Rectangle
from scipy.stats import linregress

# graphs setup
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('axes', labelsize=18)
plt.rc('figure', figsize=(12,12))


def energy(MD):
	dt = MD['timestep']
	steps = MD['steps']
	T = MD['temperature']
	U = MD['potential']
	K = MD['kinetic']
	P = MD['pressure']
	V = MD['volume']

	time = [dt * i for i in range(steps)]

	# plt.subplot(4,1,1)
	# plt.plot(time, K)
	# plt.ylabel('$E_k$')

	plt.subplot(4,1,1)
	plt.plot(time, K+U)
	plt.ylabel('$E_{tot}$')

	# plt.subplot(4,1,2)
	# plt.plot(time, U)
	# plt.ylabel('$E_p$')

	plt.subplot(4,1,2)
	plt.plot(time, V)
	plt.ylabel('$volume$')

	plt.subplot(4,1,3)
	plt.plot(time, P)
	plt.ylabel('$Pressure$')

	plt.subplot(4,1,4)
	plt.plot(time, T)
	plt.ylabel('$Temperature$')

	plt.xlabel('$Time$')
	plt.show()

def msd(dt, steps, R, D):
	time = [dt * i for i in range(steps)]

	res = linregress(time, R)
	print("Diffusion Constant =  {}".format(res.slope / 2 / D))

	plt.subplot(1,1,1)
	plt.plot(time, R)
	plt.ylabel('$r^2(t)$')
	plt.xlabel('$Time$')
	plt.show()

def msd_temp(T, R):
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

def velocity_distribution(vel):
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

def pos(pos, L):
	N, D = pos.shape
	if D == 3:
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter3D(pos[:,0], pos[:,1], pos[:,2], marker='o', c='k', s=16*L)
		plt.show()
	else:
		plt.plot(pos[:, 0], pos[:, 1], 'ko', markersize=22)
		plt.xlim(0, L)
		plt.ylim(0, L)
		plt.show()


def animate(traj, L, T, steps, dt, intv=20):
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



def animate3D(traj, L, T, steps, dt, intv=20):
	pos = traj[0]

	fig = plt.figure(figsize=(12,12))
	ax = fig.add_subplot(111, projection='3d')

	title = ax.set_title('3D')
	graph, = ax.plot(pos[:,0], pos[:,1], pos[:,2], 'ko', markersize=20)

	# Setting the axes properties
	ax.set_xlim3d([0.0, L])
	ax.set_xlabel('X')

	ax.set_ylim3d([0.0, L])
	ax.set_ylabel('Y')

	ax.set_zlim3d([0.0, L])
	ax.set_zlabel('Z')

	# make the panes transparent
	ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

	# make the grid lines transparent
	ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
	ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
	ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

	def update_graph(i):
	    pos = traj[i]
	    graph.set_data(pos[:,0], pos[:,1])
	    graph.set_3d_properties(pos[:,2])
	    title.set_text('Time={0:.1f}, Temp={1:.1f}'.format(i*dt, T[i]))
	    return title, graph, 

	ani = animation.FuncAnimation(fig, update_graph, steps, 
	                               interval=intv, blit=True)

	MOVIE = False

	if MOVIE:
		print('Saving {}.mp4 ... '.format(MOVIE), end = '')
		ani.save('{}.mp4'.format(MOVIE), fps=10, extra_args=['-vcodec', 'libx264'])
		print('done.')

	plt.show()
