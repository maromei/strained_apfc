# Hydro APFC

See {cite:t}`hydrodynamics_apfc`

## Equation Gather

Effective Hamiltonian:

$$
\begin{equation}
H = T + F = \int \mathrm{d}\mathbf{r} \; \left[
    \frac{1}{2} n_0 |\mathbf{v}|^2 + \tilde{f}
\right]
\end{equation}
$$

This is the energy density which is almost the same as eq.
{eq}`eqn:apfc_energy_functional`. It has an additional term acting on the
interface of the average density $\sim |\nabla n_0|^2$.

$$
\begin{equation}
    \tilde{f} =
    \frac{B}{2} \Phi + \frac{3D}{4} \Phi^2 +
    \sum\limits_m \left(
        A | \mathcal{G}_m \eta_m |^2 - \frac{3 D}{2} |\eta_m|^4
    \right)
    + f^{(s)} + E
    + \frac{A}{2} |\nabla n_0|^2
\end{equation}
$$

The flow equations for the amplitudes and density have an additional term each.

$$
\begin{aligned}
    \frac{\partial \eta_m}{\partial t} =
        - \mathcal{Q}_m (\eta_m \mathbf{v})
        - \mu_\eta \frac{\delta H}{\delta \eta_m^*} \\
    \frac{\partial n_0}{\partial t} =
        - \nabla (n_0 \mathbf{v})
        + \mu_n \nabla^2 \frac{\delta H}{\delta n_0}
\end{aligned}
$$ (eqn:hydro_apfc_flow_eta_n)

Here the operator $\mathcal{Q}_m$ is defined as:

$$
\begin{equation}
    \mathcal{Q}_m = \nabla + i \mathbf{G}_m
\end{equation}
$$

and $\mu_\eta$ and $\mu_n$ are dissipation parameters. <br>
The flow equation for the velocity field uses the advection derivative
$\frac{D\mathbf{v}}{Dt}$

$$
\begin{equation}
    n_0 \frac{D\mathbf{v}}{Dt} =
        n_0 \left[
            \frac{\partial v}{\partial t} +
            \left( \mathbf{v} \nabla \right) \mathbf{v}
        \right]
    = \mathbf{f} + \mu_\text{S} \nabla^2 \mathbf{v} +
        (\mu_\text{B} - \mu_\text{S}) \nabla \nabla \cdot \mathbf{v}
\end{equation}
$$ (eqn:hydro_apfc_flow_v)

with

$$
\begin{equation}
    \mathbf{f} =
        - n_0 \nabla \frac{\delta F}{\delta n_0} -
        \sum\limits_m \left[
            \eta_m^* \mathcal{Q}_m \frac{\delta F}{\delta \eta_m^*} + \text{c.c.}
        \right]
\end{equation}
$$

The variation for the amplitudes does not change compared to the APFC formulation
in eq. {eq}`eqn:apfc_flow`.

$$
\begin{equation}
    \frac{\delta F}{\delta \eta_m} =
    - | \boldsymbol{G}_m |^2 \left[
        A \mathcal{G}_m^2 \eta_m + B \eta_m + 3 D (\Phi - |\eta_m|^2) \eta_m + \frac{\partial f^s}{\partial \eta_m^*}
    \right]
\end{equation}
$$

with the full flow equation being:

$$
\begin{equation}
    \frac{\partial \eta_m}{\partial t} =
    - \mathcal{Q}_m (\eta_m \mathbf{v})
    + \mu_\eta | \boldsymbol{G}_m |^2 \left[
        A \mathcal{G}_m^2 \eta_m + B \eta_m + 3 D (\Phi - |\eta_m|^2) \eta_m + \frac{\partial f^s}{\partial \eta_m^*}
    \right]
\end{equation}
$$

The variation w.r.t. $n_0$ gaines an additional term:

$$
\begin{gathered}
    \frac{\delta F}{\delta n_0} =
    \left(
        \Delta B^0 + B^x + 3 v \Phi - B^x \nabla^2
    \right) n_0
    + 3 v P - \Phi t
    - t n_0^2 + v n_0^3 \\
    \text{with} \quad
    P = 2 \left(
        \prod\limits_m \eta_m + \prod\limits_m \eta_m^*
    \right)
\end{gathered}
$$
