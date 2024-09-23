from .Base import *


# ======================Neural CDEs as Discriminator=====================

class NCDE(eqx.Module):
    initial: eqx.nn.MLP
    vf: VectorField
    cvf: ControlledVectorField
    readout: eqx.nn.Linear

    def __init__(self, data_size, hidden_size, width_size, depth,
                 activation, final_activation, *, key, **kwargs):
        super().__init__(**kwargs)
        initial_key, vf_key, cvf_key, readout_key = jr.split(key, 4)

        self.initial = eqx.nn.MLP(
            data_size + 1, hidden_size, width_size, depth, key=initial_key
        )
        self.vf = VectorField(hidden_size, width_size, depth, True,
                              activation[0], final_activation[0], key=vf_key)
        self.cvf = ControlledVectorField(
            data_size, hidden_size, width_size, depth, True,
            activation[1], final_activation[1], key=cvf_key
        )

        self.readout = eqx.nn.Linear(hidden_size, 1, key=readout_key)

    @eqx.filter_jit
    def __call__(self, ts, ys):
        # Interpolate data into a continuous path.
        ys = diffrax.linear_interpolation(
            ts, ys, replace_nans_at_start=0.0, fill_forward_nans_at_end=True
        )
        init = jnp.concatenate([ts[0, None], ys[0]])
        control = diffrax.LinearInterpolation(ts, ys)
        vf = diffrax.ODETerm(self.vf)
        cvf = diffrax.ControlTerm(self.cvf, control)
        terms = diffrax.MultiTerm(vf, cvf)
        solver = diffrax.ReversibleHeun()
        t0 = ts[0]
        t1 = ts[-1]
        dt0 = (ts[1] - ts[0])
        y0 = self.initial(init)
        # Have the discriminator produce an output at both `t0` *and* `t1`.
        # The output at `t0` has only seen the initial point of a sample. This gives
        # additional supervision to the distribution learnt for the initial condition.
        # The output at `t1` has seen the entire path of a sample. This is needed to
        # actually learn the evolving trajectory.
        saveat = diffrax.SaveAt(ts=ts)
        sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0, y0, saveat=saveat)
        ys = sol.ys
        ys = jax.vmap(self.readout)(ys)
        return ys  # Outputs the Discriminator.

    @eqx.filter_jit
    def clip_weights(self):
        leaves, treedef = jax.tree_util.tree_flatten(
            self, is_leaf=lambda x: isinstance(x, eqx.nn.Linear)
        )
        new_leaves = []
        for leaf in leaves:
            if isinstance(leaf, eqx.nn.Linear):
                lim = 1 / leaf.out_features
                leaf = eqx.tree_at(
                    lambda x: x.weight, leaf, leaf.weight.clip(-lim, lim)
                )
            new_leaves.append(leaf)
        return jax.tree_util.tree_unflatten(treedef, new_leaves)


class NCDE_sa(eqx.Module):
    initial: eqx.nn.MLP
    vf: VectorField
    cvf: ControlledVectorField
    QLN: eqx.nn.Linear
    KLN: eqx.nn.Linear
    VLN: eqx.nn.Linear
    selfAttention: eqx.nn.MultiheadAttention
    readout: eqx.nn.Linear

    def __init__(self, data_size, hidden_size, width_size, depth,
                 activation, final_activation, *, key, **kwargs):
        super().__init__(**kwargs)
        initial_key, vf_key, cvf_key, readout_key = jr.split(key, 4)
        ro_key1, ro_key2, ro_key3, sa_key = jr.split(readout_key, 4)

        self.initial = eqx.nn.MLP(
            data_size + 1, hidden_size, width_size, depth, key=initial_key
        )
        self.vf = VectorField(hidden_size, width_size, depth, True,
                              activation[0], final_activation[0], key=vf_key)
        self.cvf = ControlledVectorField(
            data_size, hidden_size, width_size, depth, True,
            activation[1], final_activation[1], key=cvf_key
        )
        self.QLN = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=ro_key1)
        self.KLN = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=ro_key2)
        self.VLN = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=ro_key3)

        self.selfAttention = eqx.nn.MultiheadAttention(
            num_heads=12, query_size=hidden_size, key_size=hidden_size,
            value_size=hidden_size, output_size=hidden_size, key=sa_key
        )
        self.readout = eqx.nn.Linear(hidden_size, 1, key=readout_key)

    @eqx.filter_jit
    def __call__(self, ts, ys):
        # Interpolate data into a continuous path.
        ys = diffrax.linear_interpolation(
            ts, ys, replace_nans_at_start=0.0, fill_forward_nans_at_end=True
        )
        init = jnp.concatenate([ts[0, None], ys[0]])
        control = diffrax.LinearInterpolation(ts, ys)
        vf = diffrax.ODETerm(self.vf)
        cvf = diffrax.ControlTerm(self.cvf, control)
        terms = diffrax.MultiTerm(vf, cvf)
        solver = diffrax.ReversibleHeun()
        t0 = ts[0]
        t1 = ts[-1]
        dt0 = (ts[1] - ts[0])
        y0 = self.initial(init)
        # Have the discriminator produce an output at both `t0` *and* `t1`.
        # The output at `t0` has only seen the initial point of a sample. This gives
        # additional supervision to the distribution learnt for the initial condition.
        # The output at `t1` has seen the entire path of a sample. This is needed to
        # actually learn the evolving trajectory.
        saveat = diffrax.SaveAt(ts=ts)
        sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0, y0, saveat=saveat)
        ys = sol.ys

        # self-attention to decode Ht
        qys = jax.vmap(self.QLN)(ys)
        kys = jax.vmap(self.KLN)(ys)
        vys = jax.vmap(self.VLN)(ys)
        ys = self.selfAttention(query=qys,
                                key_=kys,
                                value=vys)
        ys = jax.vmap(self.readout)(ys)
        return ys  # Outputs the Discriminator.

    @eqx.filter_jit
    def clip_weights(self):
        leaves, treedef = jax.tree_util.tree_flatten(
            self, is_leaf=lambda x: isinstance(x, eqx.nn.Linear)
        )
        new_leaves = []
        for leaf in leaves:
            if isinstance(leaf, eqx.nn.Linear):
                lim = 1 / leaf.out_features
                leaf = eqx.tree_at(
                    lambda x: x.weight, leaf, leaf.weight.clip(-lim, lim)
                )
            new_leaves.append(leaf)
        return jax.tree_util.tree_unflatten(treedef, new_leaves)
