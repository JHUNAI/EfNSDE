# -*- coding:utf-8 -*-
"""
    Code for normal level day to day-ahead solar irradiance generation
"""

import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
import tqdm

from Params import (Stock_GAN_params)
# import equinox as eqx  # https://github.com/patrick-kidger/equinox
from utils import *

# 选择CPU设备
jax.devices('cpu')
jax.config.update('jax_platform_name', 'cpu')
plt.rc('font', family='Times New Roman')


def train(args):
    # Translator args
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

    discriminator = args.Distype(
        args.data_size,
        args.hidden_size,
        args.width_size,
        args.depth,
        args.dis_activation,
        args.dis_final_activation,
        key=discriminator_key
    )

    # Optimizer
    g_optim = optax.rmsprop(args.generator_lr)
    d_optim = optax.rmsprop(-args.discriminator_lr)
    g_opt_state = g_optim.init(eqx.filter(generator, eqx.is_inexact_array))
    d_opt_state = d_optim.init(eqx.filter(discriminator, eqx.is_inexact_array))

    RMSE = lambda y, y_: jnp.sum((y - jnp.mean(y_, axis=0)) ** 2) ** .5 / (jnp.sum(y ** 2)) ** .5
    L = []

    loop = tqdm.tqdm(range(args.steps),
                     desc=f"Training",
                     total=args.steps)
    for step in loop:
        lam = jnp.asarray(cosine_annealing_lr(step, args.steps * .1, 1, .001))
        step = jnp.asarray(step)
        generator, discriminator, g_opt_state, d_opt_state = make_step(
            generator,
            discriminator,
            g_opt_state,
            d_opt_state,
            g_optim,
            d_optim,
            ts,
            ys,
            key,
            lam,
            step,
        )
        # Samples for loss
        show_samples = 10
        ts_to_plot = jnp.tile(ts, show_samples).reshape(show_samples, -1)
        ys_sampled = jax.vmap(generator)(ts_to_plot,
                                         key=jr.split(sample_key, show_samples))[..., 0]
        L.append(RMSE(ys[..., 0], ys_sampled))

        if (step % args.steps_per_print) == 0 or step == args.steps - 1:
            # y is y_ture; y_ is y_pre
            total_score = 0
            num_batches = 0
            score = loss(generator, discriminator, ts, ys, sample_key, lam)
            total_score += score.item()
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

            # Show the Training result
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

            loop.set_postfix(Step=step, Score=score,
                             Train_error=train_err, Test_error=test_err)

            if (test_err < args.epsilon and train_err < .04) or (step == args.steps - 1):
                # Save Models
                if args.is_save:
                    eqx.tree_serialise_leaves(args.Gen_save_path, generator)
                    print(f"{'*' * 10}Save models to {args.Gen_save_path}{'*' * 10}")
                print(f"Find best training model or stop training")
                break

    # Save loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(L)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if args.is_save:
        eqx.tree_serialise_leaves(args.Gen_save_path, generator)
        print(f"{'*' * 10}Save models to {args.Gen_save_path}{'*' * 10}")
        plt.savefig(args.target_path + f"Loss_with_None.png")
    plt.close()





def train_by_params(
        Gentype: list = ["NFSDE", "NFSDE_sa", "NSDE"],
        Distype: list = ["NCDE", "NCDE_sa"],
        is_save: bool = True,
):
    # loop for training
    for Gtp in Gentype:
        for Dtp in Distype:
            print(f"\n{'===' * 25} \n"
                  f"Begin Training: G:{Gtp}, D:{Dtp} \n"
                  f"{'==' * 25}")
            args = Stock_GAN_params(
                Gentype=Gtp,
                Distype=Dtp,
                is_save=is_save,
            )
            train(args)
