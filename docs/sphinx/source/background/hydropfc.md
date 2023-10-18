(ch:hydro_apfc)=
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
$$ (eqn:hydro_non_linear_f)

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

## Numerics

As with all previous simulations, fourier methods with an
[IMEX scheme](ch:fourier_imex) are utilized. However, the flow equations now
have differential operators in the non-linear part. For these finite differences
are used.

```{todo}
Link to and specify which FD methods are used.
(e.g. first order diff -> central difference with forward / backward on boundary)
```

### Simplifications and Limitations

A numerical problem of {eq}`eqn:hydro_apfc_flow_v` is having the average
density field on the left hand side of the equation. Simply dividing by
$n_0$ causes most summands to have two spatially dependent fields. Under the
fourier transform a summand like this will result in a convolution of
frequency dependent fields, which prevents an implicit treatment in the
numerical scheme using the fourier methods.

This problem can be avoided by choosing the $n_0$ on the left hand side to
be constant $n_0 \rightarrow n_{0, \text{init}} = \text{const.}$, resulting in:

$$
\begin{equation}
    n_{0, \text{init}} \frac{D\mathbf{v}}{Dt} =
        n_0 \left[
            \frac{\partial v}{\partial t} +
            \left( \mathbf{v} \nabla \right) \mathbf{v}
        \right]
    = \mathbf{f} + \mu_\text{S} \nabla^2 \mathbf{v} +
        (\mu_\text{B} - \mu_\text{S}) \nabla \nabla \cdot \mathbf{v}
\end{equation}
$$ (eqn:hydro_apfc_flow_v_corrected_n0)

$n_{0, \text{init}}$ will be the value used for $n_0$ on initialization.
This can be done since the flow equation of $n_0$ is set to conserve its value.
Additionally, $n_0$ does not vary a lot in the solid phase, even at the defects.

```{todo}
Reference results for $n_0$ not varying much in the solid phase.
```

```{todo}
Is it even true that it limits us to only the solid phase?
The variation is not too great aournd the initial $n_0$.
```

This limits the simulation to the domain of the solid phase and additional
ideas are needed to describe grain growth. An example of such an approach
is described in the works of {cite:t}`shpfc`.

```{todo}
shpfc paper reading
```

### Scheme

The components of velocity's flow equation can be solved seperately. This
way some parts of the gradients can be treated implicelty.

$$
\begin{align}
    \partial_t \boldsymbol{v}_x &=
    \frac{1}{n_{0, \text{init}}} \left[
        \mu_\text{B} \partial^2_x +
        \left( \mu_\text{B} - \mu_\text{S} \right) \partial^2_y
    \right] \boldsymbol{v}_x +
    \frac{\mu_\text{B} - \mu_\text{S}}{n_{0, \text{init}}} \partial^2_{x,y} \boldsymbol{v}_y
    + \left[
        \frac{\boldsymbol{f}}{n_{0, \text{init}}}
        - \left( \boldsymbol{v} \nabla \right) \boldsymbol{v}
    \right]_x
    \\
    \partial_t \boldsymbol{v}_y &= \frac{1}{n_{0, \text{init}}} \left[
        \left( \mu_\text{B} - \mu_\text{S} \right) \partial^2_x +
        \mu_\text{B} \partial^2_y
    \right] \boldsymbol{v}_y +
    \frac{\mu_\text{B} - \mu_\text{S}}{n_{0, \text{init}}} \partial^2_{x,y} \boldsymbol{v}_x
    + \left[
        \frac{\boldsymbol{f}}{n_{0, \text{init}}}
        - \left( \boldsymbol{v} \nabla \right) \boldsymbol{v}
    \right]_y
\end{align}
$$ (eqn:hapfc_velocity_flow_numeric_scheme)
