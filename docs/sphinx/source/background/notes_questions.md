# Notes and Questions

## Approaches

The [hydrodynamics](papers/hydrodynamics.md) paper assumes Navier-Stokes-like dissipation
and adds an additional flow equation. The approach needs additional parameters
$\mu_\eta$ and $\mu_n$ (also called $\mu_S$, $\mu_B$) to describe the dissipation.

The plastic motion paper considers how the stress modifies the displacement.
The displacement is computed and then used to update the amplitudes. However,
some parameters need to be determined via solving a diff. eq. with 4th order.
