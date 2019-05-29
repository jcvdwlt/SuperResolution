import src
import numpy as np

np.random.seed(6)

SIG = 0.3
M_SQ = 6
V_A = 0.15
K = 4

psf = src.GaussianPSF(SIG)
sp = src.TwoDimSrProblem(K, V_A, M_SQ, psf)

solver = src.SrSolver(sp.ml, sp.S, psf, 7, 10 ** -6)
solver.solve()

np.save('sr_run', [sp.t, sp.a, sp.ml, sp.S, SIG, solver.xl, solver.rl])

print(len(solver.x) / 3)
