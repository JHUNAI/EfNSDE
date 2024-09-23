import jax.random as jr
import math
import Modules as M
import jax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax.numpy as jnp
from typing import Literal
import numpy as np
import os


def default_settings(args):
    # Map the "str" to the function
    A_to_B = lambda x: (Act_Map(x[0]), Act_Map(x[1]))
    args.gen_activation = A_to_B(args.gen_activation)
    args.gen_final_activation = A_to_B(args.gen_final_activation)
    args.dis_activation = A_to_B(args.dis_activation)
    args.dis_final_activation = A_to_B(args.dis_final_activation)

    # Map the Network
    args.Gentype = Choice_Gen(args.Gentype)
    args.Distype = Choice_Dis(args.Distype)
    return args


def data_get(stop_idx: int):
    # from comput_H import pick_best_stock
    # _, _, ys, max_hurst = pick_best_stock()
    # load data
    target_path = "./data/"
    data_path = os.listdir(target_path)
    ys = np.load(target_path + data_path[0])
    max_hurst = float(data_path[0][-9:-4])

    # build data
    ts = np.arange(0, len(ys))
    ts_train = ts[np.newaxis, :stop_idx]
    ys_train = ys[np.newaxis, :stop_idx, np.newaxis]
    ts_test = ts[np.newaxis, stop_idx:]
    ys_test = ys[np.newaxis, stop_idx:, np.newaxis]
    return (ts_train, ys_train), (ts_test, ys_test), max_hurst


def data_to_batch(data, batch_size, key=None):
    suf_key, add_key = jr.split(key, 2)
    ts, ys = data
    len_ys = ys.shape[0]
    loops = math.ceil(len_ys / batch_size)
    add_size = batch_size - (len_ys % batch_size)
    ts = jnp.tile(ts[0, ...], (batch_size, 1))
    suf_idx = jr.permutation(suf_key, len_ys)
    add_idx = jr.permutation(add_key, len_ys)[:add_size]
    new_shuffle_idx = jnp.concatenate([suf_idx, add_idx])
    batch_data = [(ts, ys[new_shuffle_idx[i * batch_size:i * batch_size + batch_size], ...]) for i in range(loops)]
    return batch_data


def Shuffle(data, key=None):
    ys = data
    len_ys = ys.shape[0]
    suf_idx = jr.permutation(key, len_ys)
    ys = ys[suf_idx, ...]
    return ys


def Choice_Gen(tp="NFSDE"):
    """
        Code for the choice of generators.
        Only support:
        NSDE(Neural SDE),
        NFSDE(Neural fractional SDE),
        NFSDEsa(Self attention  Neural fractional SDE).
    """
    assert tp in ["NSDE", "NFSDE", "NFSDE_sa"], "选择Generator错误"
    gens = {
        "NSDE": M.NSDE,
        "NFSDE": M.NFSDE,
        "NFSDE_sa": M.NFSDE_sa
    }
    return gens[tp]


def Choice_Dis(tp="NCDE"):
    """
        Code for the choice of discriminator.
        Only support:
        NCDE(Neural CDE),
        NCDE_sa(Self attention Neural CDE),
    """
    assert tp in ["NCDE", "NCDE_sa"], "选择Discriminator错误"
    diss = {
        "NCDE": M.NCDE,
        "NCDE_sa": M.NCDE_sa,
    }
    return diss[tp]


def lipswish(x):
    return 0.909 * jax.nn.silu(x)


def Act_Map(tp):
    Acts = {
        "tanh": jax.nn.tanh,
        "sigmoid": jax.nn.sigmoid,
        "softplus": jax.nn.softplus,
        "softmax": jax.nn.softmax,
        "leaky_relu": jax.nn.leaky_relu,
        "elu": jax.nn.elu,
        "swish": jax.nn.swish,
        "lips": lipswish
    }
    return Acts[tp]


def calculate_rmse(x, y):
    """
    Calculate the Root Mean Square Error (RMSE) between two sequences x and y.

    Args:
    x (array-like): First sequence.
    y (array-like): Second sequence.

    Returns:
    float: RMSE value.
    """
    x = np.array(x)
    y = np.array(y)
    mse = np.mean((x - y) ** 2)
    rmse = np.sqrt(mse)
    return np.mean(rmse)


def calculate_cdf(sequence):
    """
    Calculate the Cumulative Distribution Function (CDF) of a sequence.

    Args:
    sequence (array-like): Input sequence.

    Returns:
    tuple: Two arrays, sorted values and their corresponding CDF values.
    """
    sequence = np.array(sequence)
    sorted_sequence = np.sort(sequence)
    cdf = np.arange(1, len(sequence) + 1) / len(sequence)
    return sorted_sequence, cdf


# @eqx.filter_jit
# def loss_D(discriminator, generator, ts_i, ys_i, key, step=0):
#     """
#         LOSS with gradient penalty,
#         need to add :  adjoint = diffrax.DirectAdjoint()
#         to solve.
#     """
#
#     batch_size, ts_size = ts_i.shape
#     # 更新随机种子
#     key = jr.fold_in(key, step)
#     key1, key2 = jr.split(key, 2)
#     key = jr.split(key2, batch_size)
#
#     # 生成随机权重 epsilon
#     epsilon = jr.uniform(key1, shape=(batch_size, 1, 1), minval=0.0, maxval=1.0)
#     epsilon = jnp.repeat(epsilon, ts_size, axis=1)
#     # 生成假样本
#     fake_ys_i = jax.vmap(generator)(ts_i, key=key)
#     # 插值真实样本和假样本
#     inteps = epsilon * fake_ys_i + (1 - epsilon) * ys_i
#     # 计算真实样本、假样本和插值样本的判别器分数
#     real_score = jax.vmap(discriminator)(ts_i, ys_i)
#     fake_score = jax.vmap(discriminator)(ts_i, fake_ys_i)
#
#     # 计算插值样本的分数和梯度范数
#     def intp_loss(dis, ts, inteps):
#         return jnp.mean(dis(ts, inteps))  # 确保输出是标量
#
#     def gradients_penalty_loss(dis, ts, inteps):
#         gradients = jax.vmap(jax.grad(intp_loss, argnums=2), (None, 0, 0))(dis, ts, inteps)
#         gradients_norm = jnp.linalg.norm(gradients, axis=1)
#         gradients_norm = jnp.mean(gradients_norm)
#         return gradients_norm
#
#     # 计算插值样本的梯度范数
#     gradients_norm = gradients_penalty_loss(discriminator, ts_i, inteps)
#     # 计算梯度惩罚
#     gradient_penalty = jnp.mean((gradients_norm - 1.0) ** 2)
#     # GANGP的最终损失函数
#     gp_weight = 10.0  # 梯度惩罚的权重
#     d_loss = jnp.mean(fake_score - real_score) #+ gp_weight * gradient_penalty
#     return d_loss
#
#
# @eqx.filter_jit
# def loss_G(generator, discriminator, ts_i, key, step=0):
#     batch_size, ts_size = ts_i.shape
#     # 更新随机种子
#     key = jr.fold_in(key, step)
#     key = jr.split(key, batch_size)
#     fake_ys_i = jax.vmap(generator)(ts_i, key=key)
#     fake_score = jax.vmap(discriminator)(ts_i, fake_ys_i)
#     g_loss = jnp.mean(fake_score)
#     return g_loss

def dataloader(arrays, batch_size, loop, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        key = jr.split(key, 1)[0]
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
        if not loop:
            break


def create_file(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建目录 {path}")
    else:
        print(f"目录 {path} 已经存在")


def increase_update_initial(updates):
    # 获取初始叶子节点的值
    get_initial_leaves = lambda u: jax.tree_util.tree_leaves(u.initial)
    return eqx.tree_at(get_initial_leaves, updates, replace_fn=lambda x: 10 * x)


@eqx.filter_jit
def make_step(
        generator,
        discriminator,
        g_opt_state,
        d_opt_state,
        g_optim,
        d_optim,
        ts_i,
        ys_i,
        key,
        lam,  #
        step,
):
    g_grad, d_grad = grad_loss((generator, discriminator), ts_i, ys_i, key, lam, step)
    g_updates, g_opt_state = g_optim.update(g_grad, g_opt_state)
    d_updates, d_opt_state = d_optim.update(d_grad, d_opt_state)
    g_updates = increase_update_initial(g_updates)
    d_updates = increase_update_initial(d_updates)
    generator = eqx.apply_updates(generator, g_updates)
    discriminator = eqx.apply_updates(discriminator, d_updates)
    discriminator = discriminator.clip_weights()
    return generator, discriminator, g_opt_state, d_opt_state


@eqx.filter_jit
@eqx.filter_grad
def grad_loss(g_d, ts_i, ys_i, key, lam, step):
    generator, discriminator = g_d
    return loss(generator, discriminator, ts_i, ys_i, key, lam, step)


@eqx.filter_jit
def loss(generator, discriminator, ts_i, ys_i, key, lam, step=0):
    batch_size, ts_size = ts_i.shape
    # key
    key = jr.fold_in(key, step)
    key = jr.split(key, batch_size)

    # 生成假样本
    fake_ys_i = jax.vmap(generator)(ts_i, key=key)
    # 计算真实样本、假样本和插值样本的判别器分数
    real_score = jax.vmap(discriminator)(ts_i, ys_i)
    fake_score = jax.vmap(discriminator)(ts_i, fake_ys_i)

    # d_loss = jnp.mean(fake_score - real_score) + lam * jnp.mean((fake_ys_i - ys_i) ** 2)
    d_loss = jnp.mean(fake_score - real_score) + jnp.mean((fake_ys_i - ys_i) ** 2)

    return d_loss


def cosine_annealing_lr(epoch, total_epochs, initial_lr, min_lr):
    """
    余弦退火学习率衰减函数

    :param epoch: 当前迭代轮次
    :param total_epochs: 总共迭代轮次
    :param initial_lr: 初始学习率
    :param min_lr: 最小学习率边界
    :return: 当前epoch的学习率
    """
    lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))
    return lr


def exponential_decay_lr(epoch, initial_lr, decay_rate, decay_steps=1):
    """
    指数学习率衰减函数

    :param epoch: 当前迭代轮次
    :param initial_lr: 初始学习率
    :param decay_rate: 衰减率
    :param decay_steps: 每decay_steps个epoch衰减一次
    :return: 当前epoch的学习率
    """
    lr = initial_lr * math.pow(decay_rate, epoch // decay_steps)
    return lr


def linear_decay_lr(epoch, initial_lr, min_lr, decay_epochs):
    """
    线性学习率衰减函数

    :param epoch: 当前迭代轮次
    :param initial_lr: 初始学习率
    :param min_lr: 最小学习率
    :param decay_epochs: 衰减周期，即多少个epoch后达到min_lr
    :return: 当前epoch的学习率
    """
    lr = max(min_lr, initial_lr - (initial_lr - min_lr) * (epoch / decay_epochs))
    return lr


def polynomial_decay_lr(epoch, initial_lr, end_epoch, power=1.0):
    """
    多项式学习率衰减函数

    :param epoch: 当前迭代轮次
    :param initial_lr: 初始学习率
    :param end_epoch: 训练总轮次
    :param power: 多项式的幂，默认为1（线性）
    :return: 当前epoch的学习率
    """
    lr = initial_lr * ((1.0 - epoch / end_epoch) ** power)
    return lr


@eqx.filter_jit
def fsde_step(
        fsdenet,
        g_opt_state,
        g_optim,
        ts_i,
        key,
        step,
):
    g_grad = fsde_loss(fsdenet, ts_i, key, step=step)
    g_updates, g_opt_state = g_optim.update(g_grad, g_opt_state)
    g_updates = increase_update_initial(g_updates)
    generator = eqx.apply_updates(fsdenet, g_updates)
    return generator, g_opt_state


@eqx.filter_jit
@eqx.filter_grad
def fsde_loss(fsdenet, ts_i, key, step=0):
    batch_size, ts_size = ts_i.shape
    # key
    key = jr.fold_in(key, step)
    key = jr.split(key, batch_size)

    fake_ys_i = jax.vmap(fsdenet)(ts_i, key=key)
    r_hat = jnp.diff(jnp.log(fake_ys_i), axis=1)
    loss = jnp.mean(r_hat)
    return loss


if __name__ == '__main__':
    epoch = 20000
    decay_epochs = 200
    initial_lr = 1.0
    min_lr = .1
    pdl_lrs = [polynomial_decay_lr(i, initial_lr, epoch, power=2.0) for i in range(epoch)]
    ldl_lrs = [linear_decay_lr(i, initial_lr, min_lr, epoch) for i in range(epoch)]
    edl_lrs = [exponential_decay_lr(i, initial_lr, .1, 1) for i in range(epoch)]
    cal_lrs = [cosine_annealing_lr(i, epoch, initial_lr, min_lr) for i in range(epoch)]
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(pdl_lrs, label='polynomial_decay_lr')
    plt.plot(ldl_lrs, label='linear_decay_lr')
    plt.plot(edl_lrs, label='exponential_decay_lr')
    plt.plot(cal_lrs, label='cosine_annealing_lr')
    plt.legend(loc='best')
    plt.show()
