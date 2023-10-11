# Energy Functional and Flow

Energy Functional {cite:p}`apfc_overview`:

$$
\begin{equation}
    F = \int_\Omega \mathrm{d} \boldsymbol{r} \left[
        \frac{B}{2} \Phi + \frac{3D}{4} \Phi^2 +
        \sum\limits_m \left(
            A | \mathcal{G}_m \eta_m |^2 - \frac{3 D}{2} |\eta_m|^4
        \right)
        + f^s + E
    \right]
\end{equation}
$$ (eqn:apfc_energy_functional)

With {cite:p}`apfc_overview` {cite:p}`2010Yeon_apfc_density`
{cite:p}`2018Ofori_anisotrop_pfc`

$$
\begin{aligned}
A &= B^x \\
B &= \Delta B^0 - 2 t n_0 + 3 v n_0^2 \\
C &= - t + 3 v n_0 \\
D &= v = 1 / 3\\
E &= (\Delta B^0 + B^x) \frac{n_0^2}{2} - t \frac{n_0^3}{3} + v \frac{n_0^4}{4} \\
\Phi &= 2 \sum\limits_m^M |\eta_m|^2 \\
\mathcal{G}_m &= \sqrt\beta \nabla^2 + 2 \mathbb{i} \boldsymbol{G}_m \nabla
\end{aligned}
$$ (eqn:apfc_flow_constants)

For a triangular crystal with one-mode approx.:

$$
\begin{gathered}
f = 2 C (\eta_1 \eta_2 \eta_3 + \eta_1^* \eta_2^* \eta_3^*) \\
\boldsymbol{G}_1 = \begin{bmatrix} - \sqrt{3} / 2 \\ - 1 / 2 \end{bmatrix}, \quad
\boldsymbol{G}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad
\boldsymbol{G}_3 = \begin{bmatrix} \sqrt{3} / 2 \\ - 1 / 2 \end{bmatrix}
\end{gathered}
$$ (eqn:onemodetriangluar)

$L^2$ Gradient Flow for amplitudes $\eta_m$ {cite:p}`apfc_overview`:

$$
\begin{equation}
\frac{\partial \eta_m}{\partial t} = \mathcal{L}_m \frac{\delta F}{\delta \eta_m^*} \approx
- | \boldsymbol{G}_m |^2 \left[
    A \mathcal{G}_m^2 \eta_m + B \eta_m + 3 D (\Phi - |\eta_m|^2) \eta_m + \frac{\partial f^s}{\partial \eta_m^*}
\right]
\end{equation}
$$ (eqn:apfc_flow)

$H^{-1}$ Gradient Flow for the average density $n_0$ {cite:p}`2010Yeon_apfc_density`

$$
\begin{gathered}
\frac{\partial n_0}{\partial t} =
\nabla^2 \frac{\delta F}{\delta n_0} =
\nabla^2 \left[
    \left(
        \Delta B^0 + B^x + 3 v \Phi
    \right) n_0
    + 3 v P - \Phi t
    - t n_0^2 + v n_0^3
\right] \\
\text{with} \quad
P = 2 \left(
    \prod\limits_m \eta_m + \prod\limits_m \eta_m^*
\right)
\end{gathered}
$$ (eqn:apfc_n0_flow)
