import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import animation
from matplotlib.gridspec import GridSpec
import src

[t, a, ml, S, sig, xl, rl] = np.load('sr_run.npy')
k = len(a)
m = int(np.sqrt(len(S)))
psf = src.GaussianPSF(sig)


def make_image_s(mls, x):
    t_h, a_h = src.split_x(x)
    s = np.matmul(src.phi(t_h, mls, psf), a_h)
    return s.reshape([m, m]) * -1 + max(s)


fig = plt.figure(figsize=(11, 5))
ax1 = fig.add_subplot(GridSpec(2, 4).new_subplotspec((0, 0), colspan=1), aspect='equal',
                      autoscale_on=False, xlim=(-0.5, m-0.5), ylim=(-0.5, m-0.5))

ax1.imshow(S.reshape([m, m]) * -1 + max(S), cmap='Greys')
ax1.axis('off')
ax1.set_title('Input Image')

ax4 = fig.add_subplot(GridSpec(2, 4).new_subplotspec((0, 2), colspan=2, rowspan=2), aspect='equal',
                      autoscale_on=False, projection='3d', xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))

ax2 = fig.add_subplot(GridSpec(2, 4).new_subplotspec((0, 1), colspan=1, rowspan=1), aspect='equal',
                      autoscale_on=False, xlim=(-0.5, m-0.5), ylim=(-0.5, m-0.5))
ax2.imshow(S.reshape([m, m]) * -1 + max(S), cmap='Greys')
ax2.axis('off')
ax2.set_title('Estimate Signal')

ax3 = fig.add_subplot(GridSpec(2, 4).new_subplotspec((1, 0), colspan=2), autoscale_on=False,
                      xlim=(0, len(rl)-1), ylim=(min(rl), max(rl)), yscale='log')

line, = ax3.plot([], [])
ax3.set_xlabel('Iterations')
ax3.set_ylabel('Residual')


tops = np.block([t, a.reshape([k, 1])])
bottoms = np.block([t, np.zeros([k, 1])])
lines = [[list(s0), list(s1)] for s0, s1 in zip(tops, bottoms)]
lines = Line3DCollection(lines)
ax4.add_collection3d(lines)
ax4.scatter(t[:, 0], t[:, 1], a, s=50, label='True Sources')

th, ah = src.split_x(xl[0])
est = ax4.scatter(th[:, 0], th[:, 1], ah, marker='x', s=60, c='r', label='Estimates')
ax4.legend()


def animate(i):
    t_h, a_h = src.split_x(xl[i])
    est._offsets3d = (t_h[:, 0], t_h[:, 1], a_h)
    line.set_data(range(i+1), rl[0:i+1])
    ax2.imshow(make_image_s(ml, xl[i]), cmap='Greys')
    # return est


anim = animation.FuncAnimation(fig, animate, frames=len(xl), interval=200, blit=False)
anim.save('figs/sr4.gif', writer='imagemagick', fps=4)
plt.show()
