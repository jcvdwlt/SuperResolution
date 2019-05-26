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


m1, m2 = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
ml2 = np.stack([m1.flatten(), m2.flatten()]).transpose()


def make_image_s(mls, x):
    t_h, a_h = src.split_x(x)
    s = np.matmul(src.phi(t_h, mls, psf), a_h)
    return s.reshape([100, 100])


im = make_image_s(ml2, xl[0])

fig = plt.figure(figsize=(11, 4))
ax1 = fig.add_subplot(131, projection='3d')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_zlim([0, 1])
tops = np.block([t, a.reshape([k, 1])])
bottoms = np.block([t, np.zeros([k, 1])])
lines = [[list(s0), list(s1)] for s0, s1 in zip(tops, bottoms)]
lines = Line3DCollection(lines)
ax1.add_collection3d(lines)
ax1.scatter(t[:, 0], t[:, 1], a, s=50, label='True Sources')
ax1.view_init(elev=20, azim=340)


ax2 = fig.add_subplot(132, projection='3d')
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.set_zlim([0, 1])

ax2.plot_surface(m1,  m2, im)
ax2.view_init(elev=20, azim=250)

ax3 = fig.add_subplot(133)
ax3.imshow(S.reshape([m, m]).T * -1 + max(S), cmap='Greys')
ax3.axis('off')

plt.show()
