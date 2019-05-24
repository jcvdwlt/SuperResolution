import src
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# np.random.seed(10)

K = 3
d = 2
m = 5

sp = src.SrProblem(k=K, va=0.4, m=m, sig=0.3)


def split_x(x):
    return x[0:(K * d)].reshape([d, K]).transpose(), x[(K * d):]


def flat_x(a, t):
    return np.concatenate([t.flatten(), a])


def res(x):
    t, a = split_x(x)
    return sp.res(t, a)


def jac(x):
    t, a = split_x(x)
    return sp.jac(t, a)


xs = 0.4 * np.ones(K*3) + np.random.random(K*3) * 0.2
gn = src.LM(xs, jac, res, 10**-2, 50, 10**-5)
gn.iterate_ns()

th, ah = split_x(gn.x)

# Plotting
plt.figure(1)
plt.imshow(sp.S.reshape([m, m]).T, cmap='Greys')

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 2])

tops = np.block([sp.t, sp.a.reshape([sp.k, 1])])
bottoms = np.block([sp.t, np.zeros([sp.k, 1])])
lines = [[list(s0), list(s1)] for s0, s1 in zip(tops, bottoms)]
lines = Line3DCollection(lines)
ax.add_collection3d(lines)
ax.scatter(sp.t[:, 0], sp.t[:, 1], sp.a, s=50, label='Sources')
ax.scatter(th[:, 0], th[:, 1], ah, marker='x', s=60, c='r', label='Estimates')
ax.legend()

plt.figure(3)
plt.plot(gn.rl)
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Residual')
plt.show()

np.save('sr_run', [sp.t, sp.a, sp.S, gn.xl])
