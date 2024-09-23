import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
import tqdm
# import equinox as eqx  # https://github.com/patrick-kidger/equinox
import argparse
from utils import *
import json


def fSDEnet_params(
        Gentype: str = "NFSDE",
        Distype: str = "NCDE",
        is_save: bool = True,
):
    # params for Stock which used to
    initial_noise_size = 5
    noise_size = 10
    hidden_size = 32
    width_size = 32
    depth = 3
    generator_lr = 1e-4
    discriminator_lr = 2e-4
    batch_size = 20
    steps = 10000
    steps_per_print = 100
    seed = 7778
    target_path = f'./Output_models/{Gentype}/'
    Gen_save_path = target_path + 'Generator.eqx'

    # (fhandle, ghandle)
    G_act = ["tanh", "tanh"]
    G_fn_act = ["tanh", "tanh"]
    D_act = ["tanh", "tanh"]
    D_fn_act = ["tanh", "tanh"]

    parser = argparse.ArgumentParser()

    # =================== parameters for network. ===================
    parser.add_argument('--data_size', type=int, required=False,
                        default=1, help='data dim')
    parser.add_argument('--initial_noise_size', type=int, required=False,
                        default=initial_noise_size, help='Initial noise size for NSDEs(G & C)')
    parser.add_argument('--noise_size', type=int, required=False,
                        default=noise_size, help='Noise size for NSDEs(G & C)')
    parser.add_argument('--hidden_size', type=int, required=False,
                        default=hidden_size, help='hidden size for MLP')
    parser.add_argument('--width_size', type=int, required=False,
                        default=width_size, help='width size for MLP')
    parser.add_argument('--depth', type=int, required=False,
                        default=depth, help='depth for MLP')
    parser.add_argument('--stop_idx', type=float, required=False,
                        default=236, help='stop idx for MLP')

    # (A， B) -> A ： fhandle, B : ghandle.
    parser.add_argument('--gen_activation', type=tuple, required=False,
                        default=G_act, help='Generator activation function')
    parser.add_argument('--gen_final_activation', type=tuple, required=False,
                        default=G_fn_act, help='Generator activation function')
    parser.add_argument('--dis_activation', type=tuple, required=False,
                        default=D_act, help='Discriminator activation function')
    parser.add_argument('--dis_final_activation', type=tuple, required=False,
                        default=D_fn_act, help='Discriminator activation function')

    # =================== setting fot training.===================
    # learning rate
    parser.add_argument('--generator_lr', type=float, required=False,
                        default=generator_lr, help='learning rate for generator')
    parser.add_argument('--discriminator_lr', type=float, required=False,
                        default=discriminator_lr, help="learning rate for discriminator")
    # training set
    parser.add_argument('--steps', type=int, required=False,
                        default=steps, help='Training steps')
    parser.add_argument('--steps_per_print', type=int, required=False,
                        default=steps_per_print, help='Show results for every step')
    parser.add_argument('--seed', type=int, required=False,
                        default=seed, help='Model seed')
    parser.add_argument('--batch_size', type=int, required=False,
                        default=batch_size, help='batch_size')
    parser.add_argument('--Gen_save_path', type=str, required=False,
                        default=Gen_save_path, help='path for generator model')
    parser.add_argument('--target_path', type=str, required=False,
                        default=target_path, help='path for generator model')
    # stop rate
    parser.add_argument('--epsilon', type=float, required=False,
                        default=0.036, help='stop ratio')  # .003 -- .004

    # choice for generator and discriminator type : (classic, with RNN, with self attention)
    parser.add_argument('--Gentype', type=str, required=False,
                        default=Gentype, help='generator type')
    parser.add_argument('--Distype', type=str, required=False,
                        default=Distype, help='discriminator type')
    parser.add_argument('--is_save', type=bool, required=False,
                        default=is_save, help='discriminator type')
    create_file(target_path)
    args = parser.parse_args()

    if is_save:  # 将参数转换为字典并保存为JSON文件
        save_path = f"./Output_models/{args.Gentype}/"
        with open(save_path + 'settings.json', 'w') as f:
            json.dump(vars(args), f)
        print(f"{'=' * 7}Save args in {save_path + 'settings.json'} Successful!{'=' * 7}")
    return args


def train_fSDE_Net(args):
    args = default_settings(args)

    # get keys in need
    key = jr.PRNGKey(args.seed)
    (
        Shuffle_key,
        generator_key,
        discriminator_key,
        sample_key,
        dataloader_key,
    ) = jr.split(key, 5)
    (ts, ys), (ts_test, ys_test), max_hurst = data_get(args.stop_idx)
    ts, ys = jnp.array(ts), jnp.array(ys)  # ts :[solve-dim, time]; ys[solve-dim,time,1]
    ts_test, ys_test = jnp.array(ts_test), jnp.array(ys_test)
    dataset_size, _, _ = ys.shape
    trt_lens = jnp.shape(ts)[1]
    tst_lens = jnp.shape(ts_test)[1]

    generator = args.Gentype(
        args.data_size,
        args.initial_noise_size,
        args.noise_size,
        args.hidden_size,
        args.width_size,
        args.depth,
        args.gen_activation,
        args.gen_final_activation,
        key=generator_key
    )

    g_optim = optax.adam(args.generator_lr)
    g_opt_state = g_optim.init(eqx.filter(generator, eqx.is_inexact_array))

    loop = tqdm.tqdm(range(args.steps),
                     desc=f"GEN:(**{args.Gentype}**)DIS(**{args.Distype}**)Training Epoch",
                     total=args.steps)
    RMSE = lambda y, y_: jnp.sum((y - jnp.mean(y_, axis=0)) ** 2) ** .5 / (jnp.sum(y ** 2)) ** .5
    L = []

    for step in loop:
        step = jnp.asarray(step)
        generator, g_opt_state = fsde_step(
            generator,
            g_opt_state,
            g_optim,
            ts,
            key,
            step,
        )
        # compute loss
        show_samples = 10
        ts_to_plot = jnp.tile(ts, show_samples).reshape(show_samples, -1)
        ys_sampled = jax.vmap(generator)(ts_to_plot,
                                         key=jr.split(sample_key, show_samples))[..., 0]
        L.append(RMSE(ys[..., 0], ys_sampled))
        if (step % args.steps_per_print) == 0 or step == args.steps - 1:
            # y is y_ture; y_ is y_pre
            num_batches = 0
            num_batches += 1
            ts_novel = jnp.arange(0, trt_lens + tst_lens)
            show_samples = 10
            ts_to_plot = jnp.tile(ts_novel, show_samples).reshape(show_samples, -1)

            ys_to_plot = ys[..., 0]

            ys_sampled = jax.vmap(generator)(ts_to_plot,
                                             key=jr.split(sample_key, show_samples))[..., 0]
            ys_pre_train = ys_sampled[:, :args.stop_idx]
            ys_pre_test = ys_sampled[:, args.stop_idx:]

            train_err = RMSE(ys_to_plot, ys_pre_train)
            test_err = RMSE(ys_test, ys_pre_test)

            fig, ax = plt.subplots()
            kwargs = dict(label="Real")
            for ti, yi in zip(ts, ys):
                ax.plot(ti, yi, c="dodgerblue", linewidth=0.5, alpha=0.7, **kwargs)
                kwargs = {}
            kwargs = dict(label="Generated")
            for ti, yi in zip(ts_to_plot, ys_sampled):
                ax.plot(ti, yi, c="crimson", linewidth=0.5, alpha=0.7, **kwargs)
                kwargs = {}
            fig.legend()
            fig.tight_layout()
            plt.show()

            loop.set_postfix(Step=step,
                             Train_err=train_err,
                             Test_err=test_err
                             )

            eqx.tree_serialise_leaves(args.Gen_save_path, generator)
            if test_err < args.epsilon and train_err < .06:
                print(f"Find best training model")
                break

    # Save loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(L)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(args.target_path + f"Loss_with_None.png")


if __name__ == '__main__':
    args = fSDEnet_params()
    train_fSDE_Net(args)
