# Config Reference

| key              | datatype   | description                                                                                                                                                 |
|------------------|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Bx`             | float      | $\Delta B^x$ see eq. {eq}`eqn:apfc_flow_constants`                                                                                                          |
| `dB0`            | float      | $\Delta B^0$ see eq. {eq}`eqn:apfc_flow_constants`                                                                                                          |
| `n0`             | float      | $n_0$ see eq. {eq}`eqn:apfc_flow_constants`                                                                                                                 |
| `t`              | float      | $t$ see eq. {eq}`eqn:apfc_flow_constants`                                                                                                                   |
| `v`              | float      | $v$ see eq. {eq}`eqn:apfc_flow_constants`                                                                                                                   |
| `dt`             | float      | time step                                                                                                                                                   |
| `initRadius`     | float      | initial Radius of the grain / center line initialization of the amplitudes                                                                                  |
| `interfaceWidth` | float      | The initial interfacewidth of the amplitudes                                                                                                                |
| `initEta`        | float      | The initial height of the amplitudes. This can be calculated as a first step in the simulation.                                                             |
| `thetaDiv`       | int        | Divides the $[0, 2 \pi]$ interval into `thetaDiv` parts. Only the first part will be simulated.                                                             |
| `thetaCount`     | int        | How many angles in the interval should be simulated. If set to 1, only $\theta=0$ will be done.                                                             |
| `numPtsX`        | int        | Number of points in x-direction.                                                                                                                            |
| `numPtsY`        | int        | Number of points in y-direction. If set $\le 1$ a 1-dimensional simulation will be done.                                                                    |
| `xlim`           | float      | The domain bounds will be set to $[-\text{xlim}, \text{xlim}]^2$                                                                                            |
| `keepEtaConst`   | boolean    | If `True` only the average density will be computed, while the amplitudes are kepts Constant. Only applies to `simType=n0`                                  |
| `G`              | float[][2] | The reciprical vectors.                                                                                                                                     |
| `numT`           | int        | How many timesteps should bed done.                                                                                                                         |
| `writeEvery`     | int        | After how many timesteps should an output be created.                                                                                                       |
| `eqSteps`        | int        | Does this number of steps before creating output. These steps do not count towards `numT`.                                                                  |
| `simPath`        | string     | The directory where the config file lies, and where the output should be created.                                                                           |
| `simType`        | str        | The type of the simulation. Choices are: "n0", otherwise a basic sim will be done.                                                                          |
| `defects`        | list[dict] | Contains dictionaries describing the defects with which the amplitudes should be initialized with. See the [Defect Reference](code_defect_reference) below. |
| `mu_b` | float | dissipation parameter $\mu_B$ in eq. {eq}`eqn:hydro_apfc_flow_v`. Only relevant for hydro apfc. |
| `mu_s` | float | dissipation parameter $\mu_S$ in eq. {eq}`eqn:hydro_apfc_flow_v`. Only relevant for hydro apfc. |
| `mu_eta` | float | dissipation parameter $\mu_\eta$ in eq. {eq}`eqn:hydro_apfc_flow_eta_n`. Only relevant for hydro apfc. |
| `mu_n0` | float | dissipation parameter $\mu_n$ in eq. {eq}`eqn:hydro_apfc_flow_eta_n`. Only relevant for hydro apfc. |


(code_defect_reference)=
## Defect Reference

| key            | datatype | description                                        |
|----------------|----------|----------------------------------------------------|
| type           | str      | Type of defect. Currently only "line" is valid.    |
| poisson ratio  | float    | poisson ratio used for initializing a line defect. |
| burgers vector | float[2] | burgers vector for defect                          |
| offset         | float[2] | offset from center of domain                       |
