import numpy as np
import matplotlib.pyplot as plt

C_CONST = 0.1
V = 10**-1
C_STEP = 0.1
LB = 0
UB = 1


class GaussianPSF:
    def __init__(self, sig):
        self.sig = sig

    def psf(self, l, t):
        return np.exp(-np.sum((t - l) ** 2) / (self.sig ** 2))

    def d_psf(self, l, t, d):
        return - 2 * (t[d] - l[d]) * self.psf(t, l) / self.sig ** 2


def phi(th, ml, psf):
    m, _ = ml.shape
    k, _ = th.shape
    p = np.zeros([m, k])
    for i in range(m):
        for j in range(k):
            p[i, j] = psf.psf(ml[i, :], th[j, :])
    return p


def res(th, ah, ml, s, psf):
    return np.matmul(phi(th, ml, psf), ah) - s


def jac(th, ah, ml, psf):
    m, d = ml.shape
    k, _ = th.shape

    jc = np.zeros([m, k * (d + 1)])

    for i in range(m):
        for dd in range(d):
            for j in range(k):
                jc[i, j + dd * k] = psf.d_psf(ml[i, :], th[j, :], dd) * ah[j]

        for j in range(k):
            jc[i, j + d * k] = psf.psf(ml[i, :], th[j, :])

    return jc


def split_x(x):
    k = int(len(x) / 3)
    return x[0:(k * 2)].reshape([2, k]).transpose(), x[(k * 2):]


def flat_x(t, a):
    k, d = t.shape
    t = np.concatenate([t[:, i].flatten() for i in range(d)])
    return np.concatenate([t, a])


class TwoDimSrProblem:
    def __init__(self, k, va, m_sr, psf):
        self.k = k
        self.m = m_sr**2

        self.t = np.random.random([k, 2])
        self.a = 0.5 * np.ones(k) + va * np.random.random(k) - va / 2

        m1, m2 = np.meshgrid(np.linspace(0, 1, m_sr), np.linspace(0, 1, m_sr))
        self.ml = np.stack([m1.flatten(), m2.flatten()]).transpose()
        self.S = np.matmul(phi(self.t, self.ml, psf), self.a)


class Constraints:
    def __init__(self, lb, ub, c_const):
        self.lb = lb
        self.ub = ub
        self.c_const = c_const

    def constraints_active(self, x):
        return np.any(x < self.lb) or np.any(x > self.ub)

    def penalty(self, x):  # 1-norm constraint penalty function
        return self.c_const * (np.sum(np.abs(x[x < self.lb])) + np.sum(np.abs(x[x > self.ub])))

    def penalty_sq(self, x):  # squared constraint penalty function
        if self.constraints_active(x):
            return self.c_const * (np.sum(x[x < self.lb]**2) + np.sum(x[x > self.ub]**2)) / 2
        else:
            return 0

    def derivative(self, x):
        cg = np.zeros_like(x)
        cg[x < self.lb] = - 1
        cg[x > self.ub] = 1
        return self.c_const * cg

    def derivative_sq(self, x):
        cg = np.zeros_like(x)
        cg[x < self.lb] = x[x < self.lb] - self.lb
        cg[x > self.ub] = x[x > self.ub] - self.ub
        return self.c_const * cg

    def hes_sq(self, x):
        v = np.abs(self.derivative(x))
        h = np.diag(v)
        return h


class LM:
    def __init__(self, start_x, jac_lm, res_lm, v, max_iter, tol, lb=LB, ub=UB, c_const=C_CONST):
        self.tol = tol
        self.max_iter = max_iter
        self.jac = jac_lm
        self.res = res_lm
        self.con = Constraints(lb=lb, ub=ub, c_const=c_const)
        self.v = v
        self.step_size = C_STEP
        self.s = np.zeros_like(start_x)
        self.lx = len(start_x)
        self.x = start_x
        self.count = 0
        self.rl = []
        self.xl = []

    def iterate_ns(self, verbose=True):
        g = np.sum(self.gradient(self.x) ** 2)
        while self.count < self.max_iter and g > self.tol:
            self.calculate_dir(self.x)
            # self.set_step_size()
            self.newton_line_search()
            self.x += self.step_size * self.s
            self.count += 1
            g = np.sum(self.gradient(self.x) ** 2)
            r = self.residual(self.x)
            self.rl.append(r)
            self.xl.append(self.x.copy())
            if verbose:
                print('it:{}   res:{}  steps:{}   grad:{}'.format(self.count, r, self.step_size, g))

    def calculate_dir(self, xc):
        h = self.hessian_model(xc)
        self.s = - np.linalg.solve(h, self.gradient(xc))
        return self.s

    def gradient(self, x):
        g = np.matmul(self.jac(x).transpose(), self.res(x))
        return g + self.con.derivative_sq(x)

    def hessian_model(self, x):
        return np.matmul(self.jac(x).transpose(), self.jac(x)) + np.eye(self.lx) * self.v + self.con.hes_sq(x)

    def residual(self, xc):
        return 0.5 * np.dot(self.res(xc), self.res(xc)) + self.con.penalty_sq(xc)

    def set_step_size(self):
        if self.con.constraints_active(self.x):
            self.step_size = C_STEP
        else:
            self.newton_line_search()

    def newton_line_search(self):
        self.step_size = 0
        for i in range(5):
            df = np.dot(self.gradient(self.x + self.step_size * self.s), self.s)
            ddf = np.dot(self.s, np.matmul(self.hessian_model(self.x + self.step_size * self.s), self.s))
            self.step_size -= df / ddf

    def plot_r(self):
        plt.plot(self.rl)
        plt.yscale('log')
        plt.xlabel('Iterations')
        plt.ylabel('Residual')
        plt.show()

    def check_condition(self):  # only for test purposes, random re-injection 
        k = int(len(self.x) / 3)
        a = self.x[2*k:]
        t = self.x[0:2*k]
        if np.any(a < 0) or np.any(t < 0) or np.any(t > 1):
            t[t < 0] = np.random.random(np.sum(t < 0))
            t[t > 1] = np.random.random(np.sum(t > 1))
            a[a < 0] = np.ones(np.sum(a < 0))
            self.x[2 * k:] = a
            self.x[0:2 * k] = t
            print('re-injected')


class SrSolver:
    def __init__(self, ml, s, psf, max_iter, res_tol):
        self.max_iter = max_iter
        self.ml = ml
        self.S = s
        self.cS = s.copy()
        self.res_tol = res_tol
        self.init_x = np.array([0.5, 0.5, 1])
        self.x = np.array([])
        self.psf = psf
        self.rl = []
        self.xl = []

    def g_res(self, x):
        t, a = split_x(x)
        return res(t, a, self.ml, self.S, self.psf)

    def fi_res(self, x):
        t, a = split_x(x)
        return res(t, a, self.ml, self.cS, self.psf)

    def g_jac(self, x):
        t, a = split_x(x)
        return jac(t, a, self.ml, self.psf)

    def find_insert(self):
        lm = LM(self.init_x, self.g_jac, self.fi_res, V, 10, 10 ** -5)
        lm.iterate_ns(verbose=False)
        return lm.x

    def append_to_x(self, new_x):
        t, a = split_x(self.x)
        tn, an = split_x(new_x)
        t = np.concatenate([t, tn])
        a = np.concatenate([a, an])
        self.x = flat_x(t, a)

    def current_signal(self):
        t, a = split_x(self.x)
        return np.matmul(phi(t, self.ml, self.psf), a)

    def update_cs(self):
        self.cS -= self.current_signal()
        self.cS[self.cS < 0] = 0

    def residual(self):
        return np.dot(self.g_res(self.x), self.g_res(self.x))

    def solve(self):
        count = 0
        r = self.residual()
        self.x = self.find_insert()

        while r > self.res_tol and count < self.max_iter:

            self.update_cs()
            self.append_to_x(self.find_insert())

            lm = LM(self.x, self.g_jac, self.g_res, V, 30, 10 ** - 6)
            lm.iterate_ns(verbose=True)

            self.x = lm.x
            self.rl = self.rl + lm.rl
            self.xl = self.xl + lm.xl

            r = self.residual()
            count += 1
