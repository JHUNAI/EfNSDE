from .Base import *


class NSDE(eqx.Module):
    #  Base Neural SDE as generator
    initial: eqx.nn.MLP
    vf: VectorField  # drift
    cvf: ControlledVectorField  # diffusion
    readout1: eqx.nn.Linear
    initial_noise_size: int
    noise_size: int

    def __init__(
            self,
            data_size,
            initial_noise_size,
            noise_size,
            hidden_size,
            width_size,
            depth,
            activation,
            final_activation,
            *,
            key,
            **kwargs,
    ):
        super().__init__(**kwargs)
        initial_key, vf_key, cvf_key, readout_key = jr.split(key, 4)
        self.initial = eqx.nn.MLP(
            initial_noise_size, hidden_size, width_size, depth, key=initial_key
        )
        self.vf = VectorField(hidden_size, width_size, depth, True,
                              activation[0], final_activation[0], key=vf_key)
        self.cvf = ControlledVectorField(
            noise_size, hidden_size, width_size, depth, True,
            activation[1], final_activation[1], key=cvf_key
        )
        self.readout1 = eqx.nn.Linear(hidden_size, data_size, use_bias=True, key=readout_key)
        self.initial_noise_size = initial_noise_size
        self.noise_size = noise_size

    @eqx.filter_jit
    def __call__(self, ts, *, key):
        t0 = ts[0]
        t1 = ts[-1]
        # Very large dt0 for computational speed
        dt0 = ts[1] - ts[0]
        init_key, bm_key, drop_key, = jr.split(key, 3)
        init = jr.normal(init_key, (self.initial_noise_size,))
        control = diffrax.VirtualBrownianTree(
            t0=t0, t1=t1, tol=dt0, shape=(self.noise_size,), key=bm_key
        )
        vf = diffrax.ODETerm(self.vf)  # Drift term
        cvf = diffrax.ControlTerm(self.cvf, control)  # Diffusion term
        terms = diffrax.MultiTerm(vf, cvf)
        # ReversibleHeun is a cheap choice of SDE solver. We could also use Euler etc.
        solver = diffrax.ReversibleHeun()
        y0 = self.initial(init)
        saveat = diffrax.SaveAt(ts=ts)
        sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0, y0, saveat=saveat)
        ys = sol.ys  # [time_seq, hidden_dim ]
        # ys is the output of the Neural SDEs.
        ys = jax.vmap(self.readout1)(ys)
        return ys


class NFSDE(eqx.Module):
    #  Fractional SDE - net
    initial: eqx.nn.MLP
    vf: VectorField  # drift
    cvf: ControlledVectorField  # diffusion
    readout1: eqx.nn.Linear
    initial_noise_size: int
    noise_size: int

    def __init__(
            self,
            data_size,
            initial_noise_size,
            noise_size,
            hidden_size,
            width_size,
            depth,
            activation,
            final_activation,
            *,
            key,
            **kwargs,
    ):
        super().__init__(**kwargs)
        initial_key, vf_key, cvf_key, readout_key = jr.split(key, 4)
        self.initial = eqx.nn.MLP(
            initial_noise_size, hidden_size, width_size, depth, key=initial_key
        )
        self.vf = VectorField(hidden_size, width_size, depth, True,
                              activation[0], final_activation[0], key=vf_key)
        self.cvf = ControlledVectorField(
            noise_size, hidden_size, width_size, depth, True,
            activation[1], final_activation[1], key=cvf_key
        )
        self.readout1 = eqx.nn.Linear(hidden_size, data_size, use_bias=True, key=readout_key)
        self.initial_noise_size = initial_noise_size
        self.noise_size = noise_size

    @eqx.filter_jit
    def __call__(self, ts, *, key):
        t0 = ts[0]
        t1 = ts[-1]
        # Very large dt0 for computational speed
        dt0 = ts[1] - ts[0]
        init_key, bm_key, drop_key, = jr.split(key, 3)
        init = jr.normal(init_key, (self.initial_noise_size,))
        control = diffrax.Virtual_Fra_BrownianTree(
            t0=t0, t1=t1, tol=dt0, H=0.721, shape=(self.noise_size,), key=bm_key
        )
        vf = diffrax.ODETerm(self.vf)  # Drift term
        cvf = diffrax.ControlTerm(self.cvf, control)  # Diffusion term
        terms = diffrax.MultiTerm(vf, cvf)
        # ReversibleHeun is a cheap choice of SDE solver. We could also use Euler etc.
        solver = diffrax.ReversibleHeun()
        y0 = self.initial(init)
        saveat = diffrax.SaveAt(ts=ts)
        sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0, y0, saveat=saveat)
        ys = sol.ys  # [time_seq, hidden_dim ]
        # ys is the output of the Neural fractional SDEs.
        ys = jax.vmap(self.readout1)(ys)
        return ys


class NFSDE_sa(eqx.Module):
    #  Fractional SDE - net with self attention
    initial: eqx.nn.MLP
    vf: VectorField  # drift
    cvf: ControlledVectorField  # diffusion
    Yt: eqx.nn.Linear
    qnn: eqx.nn.Linear
    knn: eqx.nn.Linear
    vnn: eqx.nn.Linear
    selfAttention: eqx.nn.MultiheadAttention
    initial_noise_size: int
    noise_size: int

    def __init__(
            self,
            data_size,
            initial_noise_size,
            noise_size,
            hidden_size,
            width_size,
            depth,
            activation,
            final_activation,
            *,
            key,
            **kwargs,
    ):
        super().__init__(**kwargs)
        initial_key, vf_key, cvf_key, Yt_key, sakey = jr.split(key, 5)
        self.initial = eqx.nn.MLP(
            initial_noise_size, hidden_size, width_size, depth, key=initial_key
        )
        self.vf = VectorField(hidden_size, width_size, depth, True,
                              activation[0], final_activation[0], key=vf_key)
        self.cvf = ControlledVectorField(
            noise_size, hidden_size, width_size, depth, True,
            activation[1], final_activation[1], key=cvf_key
        )
        self.Yt = eqx.nn.Linear(hidden_size, hidden_size, use_bias=True, key=Yt_key)

        qkey, kkey, vkey, sannkey = jr.split(sakey, 4)
        self.qnn = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=qkey)
        self.knn = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=kkey)
        self.vnn = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=vkey)

        self.selfAttention = eqx.nn.MultiheadAttention(
            num_heads=12, query_size=hidden_size, key_size=hidden_size,
            value_size=hidden_size, output_size=data_size, key=sannkey
        )
        self.initial_noise_size = initial_noise_size  # define initial noise size
        self.noise_size = noise_size  # define noise size

    @eqx.filter_jit
    def __call__(self, ts, *, key):
        t0 = ts[0]
        t1 = ts[-1]
        # Very large dt0 for computational speed
        dt0 = (ts[1] - ts[0])
        init_key, bm_key, drop_key, = jr.split(key, 3)
        init = jr.normal(init_key, (self.initial_noise_size,))
        control = diffrax.VirtualBrownianTree(
            t0=t0, t1=t1, tol=dt0, shape=(self.noise_size,), levy_area="", key=bm_key
        )
        vf = diffrax.ODETerm(self.vf)  # Drift term
        cvf = diffrax.ControlTerm(self.cvf, control)  # Diffusion term
        terms = diffrax.MultiTerm(vf, cvf)
        # ReversibleHeun is a cheap choice of SDE solver. We could also use Euler etc.
        solver = diffrax.ReversibleHeun()
        y0 = self.initial(init)
        saveat = diffrax.SaveAt(ts=ts)
        sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0, y0, saveat=saveat)
        ys = sol.ys  # [time_seq, hidden_dim ]
        yt = jax.vmap(self.Yt)(ys)
        ytq, ytk, ytv = jax.vmap(self.qnn)(yt), jax.vmap(self.knn)(yt), jax.vmap(self.vnn)(yt)
        ys = self.selfAttention(ytq, ytk, ytv)
        return ys
