import numpy as np


class SrProblem:
    def __init__(self, k, va, m, sig, d=2):
        self.k = k
        self.m = m
        self.sig = sig
        self.d = d

        self.t = np.random.random([k, d])
        self.a = np.ones(k) + va*np.random.random(k) - va/2

        m1, m2 = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))
        self.ml = np.stack([m1.flatten(), m2.flatten()]).transpose()
        self.S = np.matmul(self.phi(self.t), self.a)

    def psf(self, t, l):
        return np.exp(-np.sum((t - l) ** 2) / (self.sig ** 2))

    def d_psf(self, t, l, d):
        return 2 * (t[d] - l[d]) * self.psf(t, l) / self.sig ** 2

    def phi(self, th):
        p = np.zeros([self.m ** self.d, self.k])
        for i in range(self.m ** self.d):
            for j in range(self.k):
                p[i, j] = self.psf(self.ml[i, :], th[j, :])
        return p

    def res(self, th, ah):
        return np.matmul(self.phi(th), ah) - self.S

    def jac(self, th, ah):
        jc = np.zeros([self.m**self.d, self.k * (self.d + 1)])

        for i in range(self.m**self.d):
            for dd in range(self.d):
                for j in range(self.k):
                    jc[i, j + dd * self.k] = self.d_psf(self.ml[i, :], th[j, :], dd) * ah[j]

            for j in range(self.k):
                jc[i, j + self.d * self.k] = self.psf(self.ml[i, :], th[j, :])

        return jc


class LM:
    def __init__(self, start_x, jac, res, v, max_iter, tol):
        self.tol = tol
        self.max_iter = max_iter
        self.jac = jac
        self.res = res
        self.v = v
        self.step_size = 1
        self.s = np.zeros_like(start_x)
        self.lx = len(start_x)
        self.x = start_x
        self.count = 0
        self.rl = []
        self.xl = []

    def iterate_ns(self, verbose=True):
        r = self.residual(self.x)
        while self.count < self.max_iter and r > self.tol:
            self.calculate_dir(self.x)
            self.newton_linesearch()
            self.x += self.step_size * self.s
            self.count += 1
            r = self.residual(self.x)
            self.rl.append(r)
            self.xl.append(self.x)
            self.check_condition()
            if verbose:
                print('it:{}   res:{}  steps:{}'.format(self.count, r, self.step_size))

    def calculate_dir(self, xc):
        h = self.hessian_model(xc)
        self.s = - np.linalg.solve(h, self.gradient(xc))
        return self.s

    def gradient(self, x):
        return np.matmul(self.jac(x).transpose(), self.res(x))

    def hessian_model(self, x):
        return np.matmul(self.jac(x).transpose(), self.jac(x)) + np.eye(self.lx) * self.v

    def residual(self, xc):
        return 0.5 * np.dot(self.res(xc), self.res(xc))

    def newton_linesearch(self):
        self.step_size = 0
        for i in range(5):
            df = np.dot(self.gradient(self.x + self.step_size * self.s), self.s)
            ddf = np.dot(self.s, np.matmul(self.hessian_model(self.x + self.step_size * self.s), self.s))
            self.step_size -= df / ddf

    def check_condition(self):
        k = int(len(self.x) / 3)
        a = self.x[2*k:]
        t = self.x[0:2*k]
        if np.any(a < 0) or np.any(t < 0) or np.any(t > 1):
            t[t < 0] = np.random.random(np.sum(t < 0))
            t[t > 1] = np.random.random(np.sum(t > 1))
            a[a < 0] = np.ones(np.sum(a < 0))
            self.x[2 * k:] = a
            self.x[0:2 * k] = t
