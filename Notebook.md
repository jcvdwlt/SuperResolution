
## Sparse Super Resolution 
Sparse super-resolution entails post-processing images of few sufficiently separated light sources to resolve their positions to a higher accuracy than that of the imaging device.


<img src="figs\start.png" width="800">

Imagine that the four lines the first image represent four point-light sources, with the heigt of each representing its intensity.  The diffraction limit of light, and the lens through which an observation is made, blurr the light signal that is emmitted from the four sources, so the combined signal might look like the middle image.  Capturing and storing an image of the light involves discretization.  If the four light sources are located very close to each other, it may occur that all four sources fall within a 6 by 6 pixel square, resulting the the given image.

The sparse super resolution problem is, given such an image, a model of the *point spread function* (PSF), and the assumtion that there are 'few' sources, find the locatoins and intensities of the light sources.

Let $k$ light sources be located at points $t_i \in \Omega$, where $\Omega$ is a spatial domain bounded by the extremal measurements.  Each source has a positive amplitude $a_i$, so that the object may be modelled by a discrete measure, $$z = \sum_{i = 1}^{k} a_i \delta_{t_i}.$$ 

The measurement device is characterised by $m$ continuous functions $\phi_j$, which aim to approximate the measurement device's PSF at a particular location $j$ so that measurements are given by the convolution,

\begin{align*}
y_j = \int_{\Omega} \phi_j(t)z(dt) = \sum_{i=1}^k a_i \phi_j(t_i), \ j = 1, 2, \dots m.
\end{align*}


Writing $ \textbf{y} = [y_1, y_2, \dots y_m]^T$, $\Phi = [\phi_1(t), \phi_2(t), \dots \phi_m(t)]^T$, the localisation problem is to find a sparse measure $\hat{z} \geq 0$, from the measurements $\textbf{y}$, which is achieved by solving,

\begin{align*}
\text{minimise}\quad\left\|\textbf{y} - \int_{\Omega} \Phi(t) \hat{z} ({\rm d}t) \right\|_2,
\end{align*}

subject to the support of $z$ in $\Omega$.
 


This may be rewritten as,

\begin{align*}
\text{minimise}\quad\left\|\textbf{y} - \hat{\Phi}\hat{\textbf{a}} \right\|_2,
\end{align*}

where the hat notation is used to denote estimates of the values charactering the object of interest and $\hat{\Phi} \in \mathbb{R}^{m\times \hat{k}}$, $\{\hat{\Phi}\}_{ji} = \phi_j(\hat{t}_i)$, $\hat{\textbf{a}} \in \mathbb{R}^{\hat{k}}$.

One approach to solving this problem is via a *Nonlinear Least Squares* (NLS) formulation. 

Let $\Omega \equiv [0, 1] \times [0, 1]$, and with normalisation we may assume  $0 \leq a_i \leq 1$.  Then the probelm is 

\begin{align*}
\text{minimise}&\quad f^*(x) = \frac{1}{2} \left\| r(x) \right\|_{2}^2,  \\ 
\text{s.t.}&\quad0 \leq x_i \leq 1, \quad i=1, 2, \dots \hat{k},
\end{align*}

where $r(x) = \textbf{y} - \hat{\Phi}\hat{\textbf{a}}. $


Consider the unconstrained NLS,

\begin{align*}
\text{minimise}&\quad f(x) = \frac{1}{2} \left\| r(x) \right\|_2^2 + \alpha |\max(0, -x)|_1 + \alpha |\max(0, x-1)|_1,
\end{align*}

(where the $\max$ function is applied pointwise) which allows the use of classic NLS algorithms.

For example, the *Gauss-Newton* (GN) method performs updates using steps $s$ which are solutions to,

\begin{align*}
\mbox{minimise}&\quad f_m(s) = f(x) + \nabla f(x)^T s + \frac{1}{2}s^T B s, 
\end{align*}

where $B =  J(x)^TJ(x)$, $\nabla f(x) = J(x) ^ T r(x) + \alpha G(x)$, 


\begin{align*}
\{G(x)\}_i = \begin{cases}
\ 1 &\text{ if } x_i > 1,\\
-1 &\text{ if } x_i < 0, \\
\ 0 &\text{ otherwise,}
\end{cases}
\end{align*}

and $J \in \mathbb{R}^{m\, \times\, \hat{k}(D + 1)}$ is the Jacobian of $r$.  

A popular alteration of GN is the Levenberg-Marquardt method where a damping parameter $\gamma$ is added to the hessian model $B = J^T J + \gamma I$, which promotes smaller steps and improves conditioning, thereby improving the convergence properties.

In the above,

\begin{align*}
\{J(x)\}_{jw} &= \frac{\partial \phi_j}{\partial \hat{t}_{id}} \hat{a}_i, \quad j = 1, 2, \dots m, \quad i = 1, 2, \dots \hat{k}, \quad d = 1, 2 \quad w = i + (d - 1) \cdot \hat{k},\\
\{J(x)\}_{jw}  &= \phi_j(t_i), \quad j = 1, 2, \dots m, \quad i = 1, 2, \dots \hat{k}, \quad w = i + 2\hat{k}.
\end{align*}

Assuming a Gaussian PSF,  

\begin{align*}
\phi_{j}(t) &= \text{exp}\left({-\frac{\|t - s_{j}\|_2^2}{\sigma^2}}\right), \\
\frac{\partial \phi_j}{\partial t_{id}} &= \frac{2(t_{id} - s_{jd})}{\sigma^2} \phi_{j}(t)
\end{align*}

where $s_j$ is the spatial position of measurement $j$.

So far, the fact that the number of sources $\hat{k}$ is unknown has not been addressed.  A simple approach, insprired by the ADCG method, is to first insert the source which would result in the greatest decrease in the objective.  Then, the PSF-signal of that source is subtracted from the original image (thresholding negative values), to obtain a residual-image.  The subsequent source insertion is done by finding the maximal decrease over the residual-image.  After each insertion, the sources are collectively adjusted until a new minima is found, after which is residaul-image is updated.  This is repeated until the objective reaches a termination tolerance.

INITIALISE: 

$x = \{\}$, $y = y*$

DO:
1. minimise $f(x_{\text{new}})$ for $y^*$ and append $x_{\text{new}}$ to $x$
2. minimise $f(x)$ for measurement $y$
3. update $y^* = \max(0, y - \hat{\Phi}\hat{a})$

WHILE: $f(x) >$ tolerance and iteration limit not reached.

A rudimentary implementation of the Levenberg-Marquardt method is used in (1) and (2), with a newton linesearch to solve the sub-problem.  As the constraint gradients have discontinuities, the linesearch is applied when the constaints are inactive, otherwise the stepsize is dictated by the norm of the calculated search direction.


<img src="figs\sr4.gif" width="800"> 
