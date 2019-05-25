import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import animation
from matplotlib.gridspec import GridSpec

[t, a, S, xl, rl] = np.load('sr_run.npy')
k = len(a)
m = int(np.sqrt(len(S)))


def split_x(x):
    k = int(len(x) / 3)
    return x[0:(k * 2)].reshape([2, k]).transpose(), x[(k * 2):]


fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(GridSpec(2, 3).new_subplotspec((0, 0), colspan=1), aspect='equal',
                      autoscale_on=False, xlim=(-0.5, m-0.5), ylim=(-0.5, m-0.5))

ax1.imshow(S.reshape([m, m]), cmap='Greys')
ax1.axis('off')

ax = fig.add_subplot(GridSpec(2, 3).new_subplotspec((0, 1), colspan=2, rowspan=2), aspect='equal',
                     autoscale_on=False, projection='3d', xlim=(0, 1), ylim=(0, 1), zlim=(0, 2))

ax2 = fig.add_subplot(GridSpec(2, 3).new_subplotspec((1, 0), colspan=1), autoscale_on=False,
                      xlim=(0, len(rl)-1), ylim=(min(rl), max(rl)), yscale='log')

line, = ax2.plot([], [])
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Residual')


tops = np.block([t, a.reshape([k, 1])])
bottoms = np.block([t, np.zeros([k, 1])])
lines = [[list(s0), list(s1)] for s0, s1 in zip(tops, bottoms)]
lines = Line3DCollection(lines)
ax.add_collection3d(lines)
ax.scatter(t[:, 0], t[:, 1], a, s=50, label='Sources')

th, ah = split_x(xl[0])
est = ax.scatter(th[:, 0], th[:, 1], ah, marker='x', s=60, c='r', label='Estimates')
ax.legend()


def animate(i):
    th, ah = split_x(xl[i])
    est._offsets3d = (th[:, 0], th[:, 1], ah)
    line.set_data(range(i+1), rl[0:i+1])
    return est


anim = animation.FuncAnimation(fig, animate, frames=len(xl), interval=400, blit=False)

anim.save('sr3.gif', writer='imagemagick', fps=2)

plt.show()

