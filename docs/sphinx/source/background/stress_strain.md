# Stress and strain

```{note}
Entirely based on {cite:t}`apfc_plastic_motion`. Frequently cited: {cite:t}`separation_estic_plastic_timescales`
```

There are 3 main parts:

- $\boldsymbol{\sigma}$ total stress
- $\sigma_{ij}^n$ stress computed from the amplitude functions
- $\boldsymbol{\varepsilon}$ smooth strain
- $\sigma^{\delta}$ Smooth stress ?

```{todo}
smooth stress right word?
```
(total_stress)=
## Total Stress

Mechanical equilibrium condition:

$$
\begin{equation}
    \nabla \boldsymbol{\sigma} = 0
\end{equation}
$$

```{todo}
Background mechanical eq.
```

(amplitdue_computed_stress)=
## Stress from amplitudes

According to {cite:t}`separation_estic_plastic_timescales`, stress can be
computed via:

$$
\begin{equation}
    \sigma_{ij}^n =
    \langle \sigma_ij \rangle =
    \langle
        (\partial_i (1 + \nabla^2) \eta) \partial_j n
        -
        (1 + \nabla^2) n \partial_{ij} n
    \rangle
\end{equation}
$$

"Using eq. {eq}`eq:density` and [...] integrating over the unit cell we
obtain"{cite}`apfc_plastic_motion`

$$
\begin{equation}
    \sigma_{ij}^n = \sum\limits_{m}^M \left\{
        \left[
            \mathcal{Q}_{m, i} \mathcal{G}_m \eta_m
        \right]
        \left[
            \mathcal{Q}_{m, j}^* \eta^*_{m}
        \right]
        -
        \left[
            \mathcal{G}_m \eta_m
        \right]
        \left[
            \mathcal{Q}_{m, i}^* \mathcal{Q}_{m, j}^* \eta^*_{m}
        \right]
    \right\}
\end{equation}
$$

with

$$
\begin{aligned}
    \mathcal{Q}_{m, i} &= \partial_i + \mathbb{i} G_m^{(i)} \\
    \mathcal{G}_m &= \nabla^2 + 2 \mathbb{i} \boldsymbol{G}_m \nabla
\end{aligned}
$$

(smooth_strain)=
## Smooth Strain

```{todo}
Smooth strain section
```

(smooth_stress)=
## Smooth stress

The stress field corresponding to the [smooth strain](smooth_strain)
$\boldsymbol{\varepsilon}_{ij}$.
