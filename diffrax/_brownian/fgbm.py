import math
from typing import Literal, Optional, Union
from typing_extensions import TypeAlias

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax.internal as lxi
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from .._custom_types import (
    BoolScalarLike,
    IntScalarLike,
    levy_tree_transpose,
    LevyArea,
    LevyVal,
    RealScalarLike,
)
from .._misc import (
    is_tuple_of_ints,
    linear_rescale,
    split_by_tree,
)
from .base import AbstractBrownianPath

#
# The notation here comes from section 5.5.2 of
#
# @phdthesis{kidger2021on,
#     title={{O}n {N}eural {D}ifferential {E}quations},
#     author={Patrick Kidger},
#     year={2021},
#     school={University of Oxford},
# }

# We define
# H_{s,t} = 1/(t-s) ( \int_s^t ( W_u - (u-s)/(t-s) W_{s,t} ) du ).
# bhh_t = t * H_{0,t}
# For more details see section 6.1 of
# @phdthesis{foster2020a,
#   publisher = {University of Oxford},
#   school = {University of Oxford},
#   title = {Numerical approximations for stochastic differential equations},
#   author = {Foster, James M.},
#   year = {2020}
# }
# For more about space-time Levy area see Definition 4.2.1.
# For the midpoint rule for generating space-time Levy area see Theorem 6.1.6.
# For the general interpolation rule for space-time Levy area see Theorem 6.1.4.

FloatDouble: TypeAlias = tuple[Float[Array, " *shape"], Float[Array, " *shape"]]
FloatTriple: TypeAlias = tuple[
    Float[Array, " *shape"], Float[Array, " *shape"], Float[Array, " *shape"]
]
_Spline: TypeAlias = Literal["sqrt", "quad", "zero"]


class _State(eqx.Module):
    level: IntScalarLike  # level of the tree
    s: RealScalarLike  # starting time of the interval
    w_s_u_su: FloatTriple  # W_s, W_u, W_{s,u}
    key: PRNGKeyArray
    bhh_s_u_su: Optional[FloatTriple]  # \bar{H}_s, _u, _{s,u}
    bkk_s_u_su: Optional[FloatTriple]  # \bar{K}_s, _u, _{s,u}


def _levy_diff(x0: LevyVal, x1: LevyVal) -> LevyVal:
    r"""Computes $(W_{s,u}, H_{s,u})$ from $(W_s, \bar{H}_{s,u})$ and
    $(W_u, \bar{H}_u)$, where $\bar{H}_u = u * H_u$.

    **Arguments:**

    - `x0`: `LevyVal` at time `s`.
    - `x1`: `LevyVal` at time `u`.

    **Returns:**

    `LevyVal(W_su, H_su)`
    """

    su = jnp.asarray(x1.dt - x0.dt, dtype=x0.W.dtype)
    w_su = x1.W - x0.W
    if x0.H is None or x1.H is None:  # BM only case
        return LevyVal(dt=su, W=w_su, H=None, bar_H=None, K=None, bar_K=None)

    assert (x0.bar_H is not None) and (x1.bar_H is not None)
    # if we are at this point levy_area == "space-time"
    _su = jnp.where(jnp.abs(su) < jnp.finfo(su).eps, jnp.inf, su)
    inverse_su = 1 / _su
    u_bb_s = x1.dt * x0.W - x0.dt * x1.W
    bhh_su = x1.bar_H - x0.bar_H - 0.5 * u_bb_s  # bhh_su = H_{s,u} * (u-s)
    hh_su = inverse_su * bhh_su
    return LevyVal(dt=su, W=w_su, H=hh_su, bar_H=None, K=None, bar_K=None)


def _split_interval(
        pred: BoolScalarLike, x_stu: FloatTriple, x_st_tu: FloatDouble
) -> FloatTriple:
    x_s, x_t, x_u = x_stu
    x_st, x_tu = x_st_tu
    x_s = jnp.where(pred, x_t, x_s)
    x_u = jnp.where(pred, x_u, x_t)
    x_su = jnp.where(pred, x_tu, x_st)
    return x_s, x_u, x_su


class Virtual_Fra_BrownianTree(AbstractBrownianPath):
    t0: RealScalarLike
    t1: RealScalarLike
    tol: RealScalarLike
    H: RealScalarLike
    shape: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    levy_area: LevyArea = eqx.field(static=True)
    key: PyTree[PRNGKeyArray]
    _spline: _Spline = eqx.field(static=True)

    @eqxi.doc_remove_args("_spline")
    def __init__(
            self,
            t0: RealScalarLike,
            t1: RealScalarLike,
            tol: RealScalarLike,
            H: RealScalarLike,
            shape: Union[tuple[int, ...], PyTree[jax.ShapeDtypeStruct]],
            key: PRNGKeyArray,
            levy_area: LevyArea = "",
            _spline: _Spline = "sqrt",
    ):
        (t0, t1) = eqx.error_if((t0, t1), t0 >= t1, "t0 must be strictly less than t1")
        self.t0 = t0
        self.t1 = t1
        # Since we rescale the interval to [0,1],
        # we need to rescale the tolerance too.
        self.tol = tol / (self.t1 - self.t0)

        self.levy_area = levy_area
        self._spline = _spline
        self.H = H  # Hurst parameter: 0< H <1
        self.shape = (
            jax.ShapeDtypeStruct(shape, lxi.default_floating_dtype())
            if is_tuple_of_ints(shape)
            else shape
        )
        if any(
                not jnp.issubdtype(x.dtype, jnp.inexact)
                for x in jtu.tree_leaves(self.shape)
        ):
            raise ValueError(
                "VirtualBrownianTree dtypes all have to be floating-point."
            )
        self.key = split_by_tree(key, self.shape)

    def _denormalise_bm_inc(self, x: LevyVal) -> LevyVal:
        # Rescaling back from [0, 1] to the original interval [t0, t1].
        interval_len = self.t1 - self.t0  # can be any dtype
        sqrt_len = jnp.sqrt(interval_len)

        def mult(z):
            dtype = jnp.result_type(z)
            return jnp.astype(interval_len, dtype) * z

        def sqrt_mult(z):
            # need to cast to dtype of each leaf in PyTree
            dtype = jnp.result_type(z)
            return jnp.astype(sqrt_len, dtype) * z

        return LevyVal(
            dt=jtu.tree_map(mult, x.dt),
            W=jtu.tree_map(sqrt_mult, x.W),
            H=jtu.tree_map(sqrt_mult, x.H),
            bar_H=None,
            K=jtu.tree_map(sqrt_mult, x.K),
            bar_K=None,
        )

    @eqx.filter_jit
    def evaluate(
            self,
            t0: RealScalarLike,
            t1: Optional[RealScalarLike] = None,
            left: bool = True,
            use_levy: bool = False,
    ) -> Union[PyTree[Array], LevyVal]:

        def _is_levy_val(obj):
            return isinstance(obj, LevyVal)

        t0 = eqxi.nondifferentiable(t0, name="t0")
        # map the interval [self.t0, self.t1] onto [0,1]
        t0 = linear_rescale(self.t0, t0, self.t1)
        levy_0 = self._evaluate(t0)
        if t1 is None:
            levy_out = levy_0

        else:
            t1 = eqxi.nondifferentiable(t1, name="t1")
            # map the interval [self.t0, self.t1] onto [0,1]
            t1 = linear_rescale(self.t0, t1, self.t1)
            levy_1 = self._evaluate(t1)
            levy_out = jtu.tree_map(_levy_diff, levy_0, levy_1, is_leaf=_is_levy_val)

        levy_out = levy_tree_transpose(self.shape, self.levy_area, levy_out)
        # now map [0,1] back onto [self.t0, self.t1]
        levy_out = self._denormalise_bm_inc(levy_out)
        assert isinstance(levy_out, LevyVal)
        return levy_out if use_levy else levy_out.W

    def _evaluate(self, r: RealScalarLike) -> PyTree[LevyVal]:
        """Maps the _evaluate_leaf function at time r using self.key onto self.shape"""
        r = eqxi.error_if(
            r,
            (r < 0) | (r > 1),
            "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1].",
        )
        map_func = lambda key, shape: self._evaluate_leaf(key, r, shape)
        return jtu.tree_map(map_func, self.key, self.shape)

    def _evaluate_leaf(
            self,
            key,
            r: RealScalarLike,
            struct: jax.ShapeDtypeStruct,
    ) -> LevyVal:
        """
        Evaluates the levy process at a given time for a given real scalar value and structure.

        Parameters:
        - key: A random key for JAX's random number generator.
        - r: A real scalar value representing the time at which the levy process is evaluated.
        - struct: A structure containing the shape and dtype of the output.

        Returns:
        - LevyVal: An object containing the evaluated levy process values.
        """

        # Initialize necessary parameters
        shape, dtype = struct.shape, struct.dtype
        t0 = jnp.zeros((), dtype)
        r = jnp.asarray(r, dtype)

        # Split the random key and initialize state for the binary search
        state_key, init_key_w = jr.split(key, 2)
        bhh = None
        bkk = None

        w_0 = jnp.zeros(shape, dtype)
        w_1 = jr.normal(init_key_w, shape, dtype)
        w = (w_0, w_1, w_1)

        init_state = _State(
            level=0, s=t0, w_s_u_su=w, key=state_key, bhh_s_u_su=bhh, bkk_s_u_su=bkk
        )

        # Define the condition function for the binary search
        def _cond_fun(_state):
            """Condition for continuing the binary search."""
            return 2.0 ** (-_state.level) > self.tol

        # Define the body function for the binary search
        def _body_fun(_state: _State):
            """Performs a single step of the binary search."""
            # Calculate Brownian motion and related quantities
            (
                _t,
                _w_stu,
                _w_inc,
                _keys,
                _bhh_stu,
                _bhh_st_tu,
                _bkk_stu,
                _bkk_st_tu,
            ) = self._fra_brownian_arch(_state, shape, dtype)

            _level = _state.level + 1
            _cond = r > _t
            _s = jnp.where(_cond, _t, _state.s)
            _key_st, _key_tu = _keys
            _key = jnp.where(_cond, _key_st, _key_tu)

            # Split the interval and update quantities based on the condition
            _w = _split_interval(_cond, _w_stu, _w_inc)
            if not self.levy_area == "":
                assert _bhh_stu is not None and _bhh_st_tu is not None
                _bhh = _split_interval(_cond, _bhh_stu, _bhh_st_tu)
                _bkk = None
            else:
                _bhh = None
                _bkk = None

            return _State(
                level=_level,
                s=_s,
                w_s_u_su=_w,
                key=_key,
                bhh_s_u_su=_bhh,
                bkk_s_u_su=_bkk,
            )

        # Perform the binary search
        final_state = lax.while_loop(_cond_fun, _body_fun, init_state)

        # Calculate the remaining quantities
        s = final_state.s
        su = 2.0 ** -final_state.level

        sr = jax.nn.relu(r - s)  # ReLU activation to ensure non-negativity
        ru = jax.nn.relu(su - sr)  # Adjustments to ensure su = sr + ru

        w_s, w_u, w_su = final_state.w_s_u_su

        # Calculate the Wiener process mean and variance
        if self.levy_area == "":
            # BM only case
            w_mean = w_s + sr / su * w_su
            # Handle different spline methods for the variance
            if self._spline == "sqrt":
                z = jr.normal(final_state.key, shape, dtype)
                bb = jnp.sqrt(sr * ru / su) * z
            elif self._spline == "quad":
                z = jr.normal(final_state.key, shape, dtype)
                bb = (sr * ru / su) * z
            elif self._spline == "zero":
                bb = jnp.zeros(shape, dtype)
            else:
                assert False  # Invalid spline method
            w_r = w_mean + bb
            return LevyVal(dt=r, W=w_r, H=None, bar_H=None, K=None, bar_K=None)

        else:
            assert False  # This branch is not implemented or conditional is not met

        return LevyVal(dt=r, W=w_r, H=hh_r, bar_H=bhh_r, K=None, bar_K=None)

    def _fra_brownian_arch(
            self, _state: _State, shape, dtype
    ) -> tuple[
        RealScalarLike,
        FloatTriple,
        FloatDouble,
        tuple[PRNGKeyArray, PRNGKeyArray],
        Optional[FloatTriple],
        Optional[FloatDouble],
        Optional[FloatTriple],
        Optional[FloatDouble],
    ]:
        r"""For `t = (s+u)/2` evaluates `w_t` and (optionally) `bhh_t`
         conditioned on `w_s`, `w_u`, `bhh_s`, `bhh_u`, where
         `bhh_st` represents $\bar{H}_{s,t} \coloneqq (t-s) H_{s,t}$.
         To avoid cancellation errors, requires an input of `w_su`, `bhh_su`
         and also returns `w_st` and `w_tu` in addition to just `w_t`. Same for `bhh`
         if it is not None.
         Note that the inputs and outputs already contain `bkk`. These values are
         there for the sake of a future extension with "space-time-time" Levy area
         and should be None for now.

        **Arguments:**

        - `_state`: The state of the Brownian tree
        - `shape`:
        - `dtype`:

        **Returns:**

        - `t`: midpoint time
        - `w_stu`: $(W_s, W_t, W_u)$
        - `w_st_tu`: $(W_{s,t}, W_{t,u})$
        - `keys`: a tuple of subinterval keys `(key_st, key_tu)`
        - `bhh_stu`: (optional) $(\bar{H}_s, \bar{H}_t, \bar{H}_u)$
        - `bhh_st_tu`: (optional) $(\bar{H}_{s,t}, \bar{H}_{t,u})$
        - `bkk_stu`: (optional) $(\bar{K}_s, \bar{K}_t, \bar{K}_u)$
        - `bkk_st_tu`: (optional) $(\bar{K}_{s,t}, \bar{K}_{t,u})$
        """
        # 从当前状态的密钥中分割出用于子区间随机数生成的密钥。
        H = self.H

        # def _fractional_brownian_increment(key, shape, dtype, H):
        #     """生成分数阶布朗运动增量的近似方法"""
        #     normal_increment = jr.normal(key, shape, dtype)
        #     scale = 0.5 ** H  # 根据Hurst参数调整增量的尺度
        #     return normal_increment * scale

        key_st, midpoint_key, key_tu = jr.split(_state.key, 3)
        keys = (key_st, key_tu)
        # 计算时间间隔`s`到`u`的长度，并进一步得到时间间隔`s`到`t`的长度。
        su = 2.0 ** -_state.level
        st = su / 2
        # 当前状态的时间点`s`。
        s = _state.s
        # 计算时间点`t=(s+u)/2`。
        t = s + st
        # 计算时间间隔`s`到`u`的根长度。
        root_su = jnp.sqrt(su)

        # 从当前状态获取已知的布朗运动值。
        w_s, w_u, w_su = _state.w_s_u_su

        # 确保当前状态中不包含高斯过程H和K的值（因为它们是可选的）。
        assert _state.bhh_s_u_su is None
        assert _state.bkk_s_u_su is None

        # 计算布朗运动在时间点`t`的平均值。
        mean = 0.5 * w_su
        # 生成布朗运动在时间点`t`的随机增量。
        w_term2 = root_su ** (2*H) / 2 * jr.normal(midpoint_key, shape, dtype)

        # 计算时间点`s`到`t`和`t`到`u`的布朗运动增量。
        w_st = mean + w_term2
        w_tu = mean - w_term2
        w_st_tu = (w_st, w_tu)
        # 计算时间点`t`的布朗运动值。
        w_t = w_s + w_st
        w_stu = (w_s, w_t, w_u)

        # 初始化高斯过程H和K的相关值为None。
        bhh_stu, bhh_st_tu, bkk_stu, bkk_st_tu = None, None, None, None

        # 返回计算得到的所有值。
        return t, w_stu, w_st_tu, keys, bhh_stu, bhh_st_tu, bkk_stu, bkk_st_tu
