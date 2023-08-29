# Displacement

## Density

The density is described by

$$
\begin{equation}
    n(\boldsymbol{r}, t) = n_0(\boldsymbol{r}, t) +
    \sum\limits_{m=1}^M
    \eta_m(\boldsymbol{r}, t)
    \exp(i \boldsymbol{G}_m \cdot \boldsymbol{r}) + \text{c.c.}
\end{equation}
$$ (eq:density)

Here it is exploited that the inverse crystal direction can be expressed
by the complex conjugate (c.c.). An example of this can be seen in
{numref}`hexagonal_lattice`

```{figure} images/hexagonal_lattice.png
---
class: with-border
name: hexagonal_lattice
---

The crystal structure for a triangular lattice in both real and reciprocal space is
displayed. Blue lines represent single mode approximations, while red lines
represent two mode approximations. Dotted lines show complex conjugates of the
solid lines, which will be exploited by the APFC model. Plot was taken
from {cite:p}`apfc_overview`.
```

## Add displacement

### Simple Calculation

(Done in {cite:t}`hydrodynamics_apfc`, and supported by {cite:t}`apfc_plastic_motion`).

A distortion field $\boldsymbol{u}$ is added s.t.
$n(\boldsymbol{r}, t) \rightarrow n(\boldsymbol{r} - \boldsymbol{u}, t)$.

$$
\begin{aligned}
    n(\boldsymbol{r} - \boldsymbol{u}, t) &=
        n_0(\boldsymbol{r} - \boldsymbol{u}, t) +
        \sum\limits_{m=1}^M \eta_m
        \exp(i \boldsymbol{G}_m \cdot (\boldsymbol{r} - \boldsymbol{u}))
        + \text{c.c.} \\
    &= n_0(\boldsymbol{r} - \boldsymbol{u}, t) +
        \sum\limits_{m=1}^M \overset{\sim}{\eta}_m
        \exp(i \boldsymbol{G}_m \cdot (\boldsymbol{r}))
        + \text{c.c.}
\end{aligned}
$$

Here we made the transition

$$
\begin{equation}
    \eta_m \rightarrow \overset{\sim}{\eta}_m =
        \eta_m \exp(-i \boldsymbol{G}_m \cdot \boldsymbol{u})
\end{equation}
$$

"giving a meaning to the phase of the complex amplitudes" {cite:p}`hydrodynamics_apfc`.

```{todo}
**Question**: When doing calculations with the basic APFC model without any
stressfields and using complex amplitudes $\eta_m$, we already have information
in the phase / complex part of the amplitudes.

Now with the displacement $\boldsymbol{u}$ we add additional information to the
complex part. If we calculate the complex amplitude, then we don't know what
information of the phase is due to the displacement part, and which is due to
the previous complex amplitude.

Ideas to resolve:

- Consider just $\eta_m$ in $\overset{\sim}{\eta}_m$ as real. The phase is exclusive to the displacement.
- If the displacement is given, or known, you could just get this part.
```

```{todo}
**Question**: How does a defect actually look like encoded in the displacement?
```

### By solving flow equation

(Done in {cite:t}`apfc_plastic_motion`. Additionally cited: {cite:t}`amplitude_eq_interact_composition_stress`)

They solve the flow equation {eq}`eqn:apfc_flow` for the density field $n$. <br>
(Assumption: small displacement $\boldsymbol{u}$)

$$
\begin{gathered}
    \overset{\sim}{n}(\boldsymbol{r} - \boldsymbol{u}, t) =
        \sum\limits_\boldsymbol{q}
        \overset{\sim}{\eta}_\boldsymbol{q}(\boldsymbol{r}, \boldsymbol{u}, t)
        e^{i \boldsymbol{q} \boldsymbol{r}} + \varepsilon(||\boldsymbol{u}||^2) \\
    \overset{\sim}{\eta}_\boldsymbol{q}(\boldsymbol{r}, \boldsymbol{u}, t) =
        (1 - i \boldsymbol{q} \boldsymbol{u}) \eta_\boldsymbol{q}(\boldsymbol{r}, t)
        - \left[ \nabla \eta_\boldsymbol{q}(\boldsymbol{r}, t) \right]^\text{T}
        \boldsymbol{u}
\end{gathered}
$$

Next Assumption: slowly varying amplitudes.

```{todo}
**Question**: How does this assumption lead to the form below?
Cited paper: {cite:t}`amplitude_eq_interact_composition_stress`.
```

$$
\begin{equation}
    \overset{\sim}{\eta}_{\boldsymbol{q}}(\boldsymbol{r}, \boldsymbol{u}, t) =
        \eta_\boldsymbol{q}(\boldsymbol{r}, t)
        e^{-i \boldsymbol{q} \boldsymbol{u}}
    \overset{\boldsymbol{u} \rightarrow 0}{\approx}
        \left( 1 - i \boldsymbol{q} \boldsymbol{u} \right)
        \eta_\boldsymbol{q} (\boldsymbol{r}, t)
\end{equation}
$$

## Computing displacement

$\boldsymbol{u}$ can be computed via [Helmholtz Decomposition](helmholtz_decomposition)
{cite}`separation_estic_plastic_timescales`.

$$
\begin{equation}
    u_i = \partial_i \phi + \sum\limits_j \epsilon_{ij} \partial_j \alpha
\end{equation}
$$

With $\phi$ and $\alpha$ can be computed from the [Smooth strain](smooth_strain)

$$
\begin{aligned}
    \nabla^2 \phi &= \text{Tr}\left( \boldsymbol{\varepsilon} \right) \\
    \nabla^4 \alpha &= - 2
        \sum\limits_{i,j} \epsilon_{ij}
        \sum\limits_{k} \partial_{ik} \varepsilon_{jk}
\end{aligned}
$$

and $\epsilon_{ij}$ is the [Levi-Civita Symbol](levi_civita_symbol)

```{todo}
**Question**: What is $\epsilon_{ij}$?

The {cite:t}`separation_estic_plastic_timescales` paper is cited repeatedly.
The answer is probably there.
```

```{todo}
**Question**: Did I interpreted the Einstein Summation notation for
$\alpha$ correctly?
```

```{todo}
Integration to get the actual values of $\phi$ and $\alpha$.
```
