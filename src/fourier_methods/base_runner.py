import numpy as np
from calculations import initialize, params
from manage import read_write as rw


class FFTBaseSim:
    """
    This class is supposed to run a base FFT simulations, where
    :math:`n_0` is constant everywhere.

    For more information see the :ref:`ch:scheme_ampl` section.
    """

    A: float = 0.98  #: param in eq. :eq:`eqn:apfc_flow_constants`
    B: float = 0.044  #: param in eq. :eq:`eqn:apfc_flow_constants`
    C: float = -0.5  #: param in eq. :eq:`eqn:apfc_flow_constants`
    D: float = 0.33333  #: param in eq. :eq:`eqn:apfc_flow_constants`
    beta_sqrt: float = 1.0  #: :math:`\sqrt\beta` in eq. :eq:`eqn:apfc_flow_constants`

    #: The bounds of the domain :math:`[-\text{xlim}, \text{xlim}]^2`
    xlim: int = 400
    pt_count_x = 1000  #: number of points in x-direction
    pt_count_y = 1000  #: number of points in y-direction
    dt: float = 0.5  #: timestep

    #: reciprical vector; see eq. :eq:`eqn:apfc_flow_constants`.
    #: Should have shape :code:`(eta_count, 2)`
    G: np.array = None

    #: the amplitudes. Should have
    #: shape :code:`(eta_count, pt_count_x, pt_count_y)`
    etas: np.array = None

    #: The :math:`\widehat{\mathcal{G}_m^2}` operator.
    #: See eq. :eq:`eqn:g_sq_op_fourier`
    g_sq_hat: np.array = None

    xm: np.array = None  #: x meshgrid
    ym: np.array = None  #: y meshgrid

    kxm: np.array = None  #: x frequency meshgrid
    kym: np.array = None  #: y frequency meshgrid

    eta_count: int = 0  #: number of etas

    def __init__(self, config: dict, con_sim: bool):
        """
        Initializes the simulation. Gets all the parameters from
        the config and initializes all grids.

        Args:
            config (dict): config object
            con_sim (bool): whether the simulations should continue.
                If :code:`True`, it will read the last values
                from a file.
        """

        #########################
        ## VARIABLE ASSIGNMENT ##
        #########################

        self.A = params.A(config)
        self.B = params.B(config)
        self.C = params.C(config)
        self.D = params.D(config)
        self.beta_sqrt = np.sqrt(config.get("beta", 1.0))

        self.xlim = config.get("xlim", self.xlim)
        self.pt_count_x = config.get("numPtsX", self.pt_count_x)
        self.pt_count_y = config.get("numPtsY", self.pt_count_y)
        self.dt = config.get("dt", self.dt)

        self.G = np.array(config["G"])
        self.eta_count = self.G.shape[0]

        ##############
        ## BUILDING ##
        ##############

        self.build(con_sim, config)

    ########################
    ## BUILDING FUNCTIONS ##
    ########################

    def build_grid(self):
        """
        Initializess the grid.
        """

        x = np.linspace(-self.xlim, self.xlim, self.pt_count_x)

        if self.pt_count_y <= 1:
            y = [0.0]
        else:
            y = np.linspace(-self.xlim, self.xlim, self.pt_count_y)

        self.xm, self.ym = np.meshgrid(x, y)

        dx = np.diff(x)[0]
        freq_x = np.fft.fftfreq(len(x), dx)

        if self.pt_count_y <= 1:
            freq_y = [0.0]
        else:
            dy = np.diff(y)[0]
            freq_y = np.fft.fftfreq(len(y), dy)

        self.kxm, self.kym = np.meshgrid(freq_x, freq_y)

    def build_eta(self, config: dict):
        """
        Initializes the amplitudes

        Needs build_grid() to be called before this function!

        Args:
            config (dict): config object
        """

        if self.pt_count_y <= 1:
            self.init_eta_center_line(config)
        else:
            self.init_eta_grain(config)

    def build_gsq_hat(self):
        """
        Initializess the :math:`\widehat{\mathcal{G}^2_m}` operator.

        Needs build_eta() to be called before this function!

        Args:
            config (dict): config object
        """

        shape = (self.eta_count, self.kxm.shape[0], self.kxm.shape[1])

        self.g_sq_hat = np.zeros(shape, dtype=complex)
        for eta_i in range(self.eta_count):
            self.g_sq_hat[eta_i, :, :] = self.g_sq_hat_fnc(eta_i)

    def build(self, con_sim: bool, config: dict):
        """
        Builds the grids, amplitudes and operators.

        Args:
            con_sim (bool): Whether the simulation should continue
                from the last values.
            config (dict): config object.
        """

        self.build_grid()

        if con_sim:
            self.init_eta_file(config)
        else:
            self.build_eta(config)

        self.build_gsq_hat()

    ########################
    ## INIT ETA FUNCTIONS ##
    ########################

    def init_eta_grain(self, config: dict):
        """
        Initializes the amplitudes with the
        :py:meth:`calculations.initialize.single_grain` function.

        Args:
            config (dict): config object
        """

        self.etas = np.zeros(
            (self.eta_count, self.xm.shape[0], self.xm.shape[1]), dtype=complex
        )

        defects: list[dict] = config.get("defects")
        if defects is not None:
            defect_field = self.get_defect_field(defects, config)

        for eta_i in range(self.eta_count):
            self.etas[eta_i, :, :] += initialize.single_grain(
                self.xm,
                self.ym,
                config,
            )

            defect_prod = self.G[eta_i, 0] * defect_field[0]
            defect_prod += self.G[eta_i, 1] * defect_field[1]

            self.etas[eta_i] *= np.exp(complex(0, 1) * defect_prod)

    def get_defect_field(self, defects: list[dict], config: dict) -> np.array:
        """
        Calculates the displacement vector for the line defects

        Args:
            defects (list[dict]): list of dictionaryies containing the info
                for each line defect
            config (dict):

        Returns:
            np.array: [u_x, u_y]
        """

        ux = np.zeros(self.xm.shape)
        uy = np.zeros(self.ym.shape)

        for defect in defects:

            burgers_vector = np.array(defect["burgers_vector"])
            offset = np.array(defect["offset"])

            ux += initialize.line_defect_x(
                self.xm, self.ym, defect["poisson_ratio"], burgers_vector[0], offset
            )
            uy += initialize.line_defect_y(
                self.xm, self.ym, defect["poisson_ratio"], burgers_vector[1], offset
            )

        return np.array([ux, uy])

    def init_eta_center_line(self, config: dict):
        """
        Initializes the amplitudes with the
        :py:meth:`calculations.initialize.center_line` function.

        Args:
            config (dict): config object
        """

        self.etas = np.zeros(
            (self.eta_count, self.xm.shape[0], self.xm.shape[1]), dtype=complex
        )

        for eta_i in range(self.eta_count):
            self.etas[eta_i, :, :] += initialize.center_line(
                self.xm,
                config,
            )

    def init_eta_file(self, config: dict):
        """
        Initializes the amplitudes with the
        :py:meth:`calculations.initialize.load_eta_from_file` function.

        Args:
            config (dict): config object
        """

        shape = (self.eta_count, self.xm.shape[0], self.xm.shape[1])
        self.etas = np.zeros(shape, dtype=complex)

        for eta_i in range(self.eta_count):
            self.etas[eta_i, :, :] += initialize.load_eta_from_file(
                self.xm.shape, config, eta_i
            )

    ###################
    ## SIM FUNCTIONS ##
    ###################

    def g_sq_hat_fnc(self, eta_i: int) -> np.array:
        """
        Calculated the :math:`\widehat{\mathcal{G}_m^2}`
        For one single amplitude.
        See eq. :eq:`eqn:g_sq_op_fourier`.

        Args:
            eta_i (int): amplitude index

        Returns:
            np.array: operator
        """

        ret = self.beta_sqrt * (self.kxm**2 + self.kym**2)
        ret += 2.0 * self.G[eta_i, 0] * self.kxm
        ret += 2.0 * self.G[eta_i, 1] * self.kym

        return ret**2

    def amp_abs_sq_sum(self, eta_i: int) -> np.array:
        """
        Computes

        .. math::

            \sum\limits_j | \eta_j |^2 - | \eta_m |^2

        for one amplitude :math:`\eta_m`

        Args:
            eta_i (int): amplitude index

        Returns:
            np.array:
        """

        sum_ = np.zeros(self.etas[0].shape, dtype=complex)
        for eta_j in range(self.eta_count):

            is_eta_i = int(eta_i == eta_j)
            sum_ += (2.0 - is_eta_i) * self.etas[eta_j] * np.conj(self.etas[eta_j])

        return sum_

    def n_hat(self, eta_i: int) -> np.array:
        """
        Computes :math:`\widehat{N}`
        for one amplitude.

        See :ref:`ch:scheme_ampl`.

        Args:
            eta_i (int): amplitude index

        Returns:
            np.array:
        """

        poss_eta_is = set([i for i in range(self.eta_count)])
        other_etas = list(poss_eta_is.difference({eta_i}))

        eta_conj1 = np.conj(self.etas[other_etas[0]])
        eta_conj2 = np.conj(self.etas[other_etas[1]])

        n = 3.0 * self.D * self.amp_abs_sq_sum(eta_i) * self.etas[eta_i]
        n += 2.0 * self.C * eta_conj1 * eta_conj2
        n = np.fft.fft2(n)

        return -1.0 * n * np.linalg.norm(self.G[eta_i]) ** 2

    def lagr_hat(self, eta_i: int) -> np.array:
        """
        Computes the linear part :math:`\widehat{\mathcal{L}}`
        for one amplitude.

        See :ref:`ch:scheme_ampl`.

        Args:
            eta_i (int): amplitude index

        Returns:
            np.array:
        """

        lagr = self.A * self.g_sq_hat[eta_i] + self.B
        return -1.0 * lagr * np.linalg.norm(self.G[eta_i]) ** 2

    def eta_routine(self, eta_i: int) -> np.array:
        """
        Runs one time step for one single amplitude.
        See :ref:`ch:scheme_ampl`

        Args:
            eta_i (int): amplitude index

        Returns:
            np.array:
        """

        lagr = self.lagr_hat(eta_i)
        n = self.n_hat(eta_i)

        exp_lagr = np.exp(lagr * self.dt)

        n_eta = exp_lagr * np.fft.fft2(self.etas[eta_i])
        n_eta += ((exp_lagr - 1.0) / lagr) * n

        return np.fft.ifft2(n_eta, s=self.etas[0].shape)

    def run_one_step(self):
        """
        Runs one timestep for every amplitude.

        See :ref:`ch:scheme_ampl`.
        """

        n_etas = np.zeros(self.etas.shape, dtype=complex)

        for eta_i in range(self.eta_count):
            n_etas[eta_i, :, :] = self.eta_routine(eta_i)

        self.etas = n_etas

    ##################
    ## IO FUNCTIONS ##
    ##################

    def reset_out_files(self, out_path: str):
        """
        Empties or creates the output files if they don't exist.

        Checks all :code:`out_{eta_i}.txt` files.

        Args:
            out_path (str): directory of the output files.
        """

        for eta_i in range(self.eta_count):
            with open(f"{out_path}/out_{eta_i}.txt", "w") as f:
                f.write("")

    def write(self, out_path: str):
        """
        Writes the current content of
        :code:`etas` into the output files in :code:`out_{eta_i}.txt`.
        Every line corresponds to one flattened amplitude.

        Args:
            out_path (str): Path where the out files are located.
        """

        rw.write_etas(self.etas, out_path, "a")
