from typing import Union
import jax.random as jr
import jax.numpy as jnp
import diffrax
import equinox as eqx
import jax

# https://github.com/patrick-kidger/equinox

# ======================Activation function======================


# Base type.
class VectorField(eqx.Module):
    scale: Union[int, jnp.ndarray]
    mlp: eqx.nn.MLP

    def __init__(self, hidden_size, width_size, depth, scale,
                 activation, final_activation, *, key, **kwargs):
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if scale:
            self.scale = jr.uniform(scale_key, (hidden_size,), minval=0.9, maxval=1.1)
        else:
            self.scale = 1
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            key=mlp_key,
        )

    @eqx.filter_jit
    def __call__(self, t, y, args):
        return self.scale * self.mlp(jnp.concatenate([t[None], y]))


class ControlledVectorField(eqx.Module):
    scale: Union[int, jnp.ndarray]
    mlp: eqx.nn.MLP
    control_size: int
    hidden_size: int

    def __init__(
            self, control_size, hidden_size, width_size,
            depth, scale, activation, final_activation, *, key, **kwargs
    ):
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if scale:
            self.scale = jr.uniform(
                scale_key, (hidden_size, control_size), minval=0.9, maxval=1.1
            )
        else:
            self.scale = 1
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size * control_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            key=mlp_key,
        )
        self.control_size = control_size
        self.hidden_size = hidden_size

    @eqx.filter_jit
    def __call__(self, t, y, args):
        return self.scale * self.mlp(jnp.concatenate([t[None], y])).reshape(
            self.hidden_size, self.control_size
        )
