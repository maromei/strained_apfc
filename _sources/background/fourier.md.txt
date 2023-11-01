# Fourier Method

(ch:fourier_etd)=
## ETD Scheme

For fourier method transform the base equation into the form

$$
\begin{equation}
\frac{\partial \phi}{\partial t} = \mathcal{L}_m \phi + N(\phi)
\end{equation}
$$ (eq:ode_form)

where $\mathcal{L}_m$ is a linear operator and $N(\phi)$ is a non-linear function of $\phi$.
Then fourier transform it.

$$
\begin{equation}
\frac{\partial \widehat{\phi}_k}{\partial t} = \mathcal{L}_k \widehat{\phi}_k + \widehat{N}_k
\end{equation}
$$ (eqn:ode_fourier_form)

This ODE is solved by

$$
\begin{equation}
\widehat{\phi}_k (t) = e^{\mathcal{L}_k t} \widehat{\phi}_k(0) + e^{\mathcal{L}_k t}
\int\limits_0^t \mathcal{d}t^\prime e^{- \mathcal{L}_k t^\prime} \widehat{N}_k(t^\prime)
\end{equation}
$$

under the assumption that $\mathcal{L}_k$ does not depend on time. With another approximation of
$\widehat{N}_k(t^\prime) \approx \widehat{N}_k(t)$ the equation for $\widehat{\phi}_k (t + \Delta t)$ reads

$$
\begin{equation}
\widehat{\phi}_k (t + \Delta t) =
e^{\mathcal{L}_k \Delta t} \widehat{\phi}_k(t) +
\frac{e^{\mathcal{L}_k \Delta t} - 1}{\mathcal{L}_k} \widehat{N}_k(t)
\end{equation}
$$ (eqn:fourier_approx_sol)

(ch:fourier_imex)=
## IMEX scheme

This takes equation {eq}`eqn:ode_fourier_form` also solves the linear part
implicitly, while the nonlinear part is used explicetly.

$$
\begin{equation}
\frac{\widehat{\phi}_{t+1} - \widehat{\phi}_t}{\tau} =
\widehat{\mathcal{L}} \widehat{\phi}_{t+1} + \widehat{N}_{t}
\end{equation}
$$

which gives an iteration scheme of:

$$
\begin{equation}
\widehat{\phi}_{t+1} = \frac{
    \widehat{\phi}_t + \tau \widehat{N}_t
} {
    1 - \tau \widehat{\mathcal{L}}
}
\end{equation}
$$ (eqn:imex_scheme)
