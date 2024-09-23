# -*- coding:utf-8 -*-
"""
    Code for normal level day to day-ahead solar irradiance generation
"""


import pandas as pd
import copy
from Params import load_args
# import equinox as eqx  # https://github.com/patrick-kidger/equinox
from utils import *
import akshare as ak
import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.use("pgf")

# 选择CPU设备
jax.devices('cpu')
jax.config.update('jax_platform_name', 'cpu')
plt.rc('font', family='Times New Roman')


def test(args):
    args = default_settings(args)

    # load Basic data
    (ts, ys), (ts_test, ys_test), max_hurst = data_get(args.stop_idx)
    ts, ys = jnp.array(ts), jnp.array(ys)  # ts :[solve-dim, time]; ys[solve-dim,time,1]
    ts_test, ys_test = jnp.array(ts_test), jnp.array(ys_test)

    # define key in need
    key = jr.PRNGKey(args.seed)
    (
        Shuffle_key,
        generator_key,
        discriminator_key,
        sample_key,
        dataloader_key,
    ) = jr.split(key, 5)

    generator = args.Gentype(
        args.data_size,
        args.initial_noise_size,
        args.noise_size,
        args.hidden_size,
        args.width_size,
        args.depth,
        args.gen_activation,
        args.gen_final_activation,
        key=generator_key,
    )
    generator = eqx.tree_deserialise_leaves(args.Gen_save_path, generator)
    model_name = args.Gen_save_path[16:-14]
    # plots
    num_samples = max(len(ys), 10)
    ts_total = jnp.concatenate((ts, ts_test), axis=1)
    ys_total = jnp.concatenate((ys, ys_test), axis=1)[..., 0]
    split_idx = ts.shape[1]

    # generate data
    ts_total_samples = jnp.tile(ts_total, num_samples).reshape(num_samples, -1)
    ys_tr_sampled = jax.vmap(generator)(ts_total_samples, key=jr.split(sample_key, num_samples))[
        ..., 0
    ]
    labels_map = {
        "NFSDE_NCDE": "fNSDE-GAN",
        "NFSDE_NCDE_sa": "Extended fNSDE-Net",
        "NSDE_NCDE": "NSDE-GAN",
        "NSDE_NCDE_sa": "NSDE-self attention G AN"
    }

    stock_zh_a_hist_df = ak.stock_zh_a_hist(
        symbol='300750',
        period="daily",
        start_date="20230528",
        end_date='20240528',
        adjust="")
    times = stock_zh_a_hist_df['日期'].values

    # plot generator samples
    plt.figure(figsize=(12, 6))
    plt.plot(times, ys_tr_sampled[0, ...].T, c="crimson", linewidth=1.5, alpha=0.7,
             label=labels_map[model_name])
    plt.plot(times, ys_tr_sampled.T, c="crimson", linewidth=1.5, alpha=0.7)
    # plot real samples
    plt.plot(times, ys_total[0, :], c="dodgerblue", linewidth=1.5, alpha=0.7, label="CATL's stock price")
    plt.axvline(x=times[ts.shape[1] - 1], color='k')
    # 在测试数据范围内添加阴影
    plt.axvspan(times[0], times[ts_total.shape[1] - 1], facecolor='gray', alpha=0.1, label='Training part')
    plt.axvspan(times[ts.shape[1] - 1], times[ts_total.shape[1] - 1], facecolor='black', alpha=0.3,
                label='Testing part')
    plt.legend(fontsize=15)
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Stock price", fontsize=18)
    plt.tick_params(axis='both', which='both', labelsize=18)
    plt.tight_layout()

    # if args.is_save:
    plt.savefig(args.target_path + f"{labels_map[model_name]}.pdf", transparent=True)
    plt.show()

    # super expend test
    ts_plus = jnp.arange(0, 422)[jnp.newaxis, :]
    ys_ps_samples = jax.vmap(generator)(ts_plus, key=jr.split(sample_key, 1))[..., 0]
    plt.figure(figsize=(15, 6))
    plt.plot(ys[0, ..., 0].T, c="red", linewidth=0.5, alpha=0.9, label="Real")
    plt.plot(ys_ps_samples.T, c="dodgerblue", linewidth=0.5, alpha=0.9, label="Generator")
    plt.axvline(x=ts.shape[1] + .5, color='k')
    plt.legend()
    plt.tight_layout()
    # if args.is_save:
    plt.savefig(args.target_path + f"{labels_map[model_name]}_plus.png")
    plt.show()

    # loss function.
    _di = 3
    MAPE = lambda y, y_: jnp.round(jnp.mean(jnp.abs((y - jnp.mean(y_, axis=0)) / y)), _di)
    strong_err = lambda y, y_: np.round(np.mean((y[-1, :] - y_[-1, :]) ** 2) ** .5, _di)
    weak_err = lambda y, y_: np.round(np.mean(np.abs(y[-1, :] - y_[-1, :])), _di)
    Errors = lambda y, y_: (MAPE(y, y_).tolist(), strong_err(y, y_).tolist(), weak_err(y, y_).tolist())

    # Compute Error
    import time

    def _sample_and_err(
            _ts, _ys,
            _lens,
            _n_samples=10,
            _s_key=sample_key
    ):
        train_lens = _lens

        # generate result of the samples.
        _ts_samples = jnp.tile(_ts, _n_samples).reshape(_n_samples, -1)
        _t1 = time.time()
        _ys_sampled = jax.vmap(generator)(_ts_samples, key=jr.split(s_key, _n_samples))[..., 0]
        _t2 = time.time()
        print(f"Generate {_n_samples} samples in {_t2 - _t1:.3f} s")

        _ys_train_real, _ys_test_real = _ys[:, :train_lens], _ys[:, train_lens:]
        _ys_train_gen, _ys_test_gen = _ys_sampled[:, :train_lens], _ys_sampled[:, train_lens:]

        train_err, train_strong_err, train_weak_err = Errors(_ys_train_real, _ys_train_gen)
        test_err, test_strong_err, test_weak_err = Errors(_ys_test_real, _ys_test_gen)
        Info = {
            "Samples": _n_samples,
            "Train_err": train_err,
            "Train_strong_err": train_strong_err,
            "Train_weak_err": train_weak_err,
            "Test_err": test_err,
            "Test_strong_err": test_strong_err,
            "Test_weak_err": test_weak_err,
        }
        return Info

    from scipy.io import savemat

    def _sample_saved(
            _ts, _ys,
            _lens,
            _n_samples=10,
            _s_key=sample_key
    ):
        train_lens = _lens
        # generate result of the samples.
        _ts_samples = jnp.tile(_ts, _n_samples).reshape(_n_samples, -1)
        _ys_sampled = jax.vmap(generator)(_ts_samples, key=jr.split(s_key, _n_samples))[..., 0]

        _ys_train_real, _ys_test_real = _ys[:, :train_lens], _ys[:, train_lens:]
        _ys_train_gen, _ys_test_gen = _ys_sampled[:, :train_lens], _ys_sampled[:, train_lens:]
        savemat(rf"C:\Users\DTY\Desktop\我的电脑\1研究生成果\Paper_NFSDE股价\期权计算/{model_name}.mat",
                {'data': _ys_test_gen})
        return

    samples = [20, 50, 100, 200, 500]
    Infos = []
    Infos_norm = []

    def _Norm(_Info: dict):
        dfs = (jnp.max(ys_total) - jnp.min(ys_total))
        _Info["Test_strong_err"] = round(_Info["Test_strong_err"] / dfs, 3)
        _Info["Train_strong_err"] = round(_Info["Train_strong_err"] / dfs, 3)
        _Info["Train_weak_err"] = round(_Info["Train_weak_err"] / dfs, 3)
        _Info["Test_weak_err"] = round(_Info["Test_weak_err"] / dfs, 3)
        return _Info

    for n in samples:
        s_key = jr.fold_in(sample_key, n)
        I = _sample_and_err(ts_total, ys_total, split_idx, n, s_key)
        I_ = copy.deepcopy(I)
        Infos += [I]
        Infos_norm.append(_Norm(I_))
    df_Infos = pd.DataFrame(Infos)
    df_Infos_norm = pd.DataFrame(Infos_norm)  # Normalize the strong or weak error
    save_samples = 50
    _sample_saved(ts_total, ys_total, split_idx, save_samples, sample_key)
    return df_Infos, df_Infos_norm


def test_and_save_err(
        Gentype: list = ["NFSDE", "NSDE"],
        Distype: list = ["NCDE", "NCDE_sa"],
):
    def _body(data: pd.DataFrame):
        ifo = pd.DataFrame({"Models": [f"{Gtp}+{Dtp}"] * 5})
        ifo = ifo.join(data, how="outer")
        return ifo

    def _save(_I, _I_norm):
        writer = pd.ExcelWriter('./Output_models/Errors.xlsx', engine='openpyxl')
        combined_df = pd.concat(_I, ignore_index=True)
        combined_df_norm = pd.concat(_I_norm, ignore_index=True)
        combined_df.to_excel(writer, index=False, sheet_name="Error")
        combined_df_norm.to_excel(writer, index=False, sheet_name="Error_Norm")
        writer._save()
        writer.close()
        print(f"Errors have been saved in './Output_models/Errors.xlsx'")
        # print(f"{'====' * 20}\n")
        # print(combined_df)
        # print(f"{'====' * 20}\n")

    # loop for testing
    I, I_norm = [], []
    for Gtp in Gentype:
        for Dtp in Distype:
            print(f"\n{'*' * 5} G:{Gtp}, D:{Dtp} {'*' * 5}")
            model_path = f"./Output_models/{Gtp}_{Dtp}/"
            args = load_args(model_path)
            I1, I2 = test(args)
            I += [_body(I1)]
            I_norm += [_body(I2)]
        _save(I, I_norm)

    print(f"Finish testing!\n"
          "The Errors is saved in ./Output_models/Errors.xlsx")
    return


def test_only_save(args):
    args = default_settings(args)

    # load Basic data
    (ts, ys), (ts_test, ys_test), max_hurst = data_get(args.stop_idx)
    ts, ys = jnp.array(ts), jnp.array(ys)  # ts :[solve-dim, time]; ys[solve-dim,time,1]
    ts_test, ys_test = jnp.array(ts_test), jnp.array(ys_test)

    # define key in need
    key = jr.PRNGKey(args.seed)
    (
        Shuffle_key,
        generator_key,
        discriminator_key,
        sample_key,
        dataloader_key,
    ) = jr.split(key, 5)

    generator = args.Gentype(
        args.data_size,
        args.initial_noise_size,
        args.noise_size,
        args.hidden_size,
        args.width_size,
        args.depth,
        args.gen_activation,
        args.gen_final_activation,
        key=generator_key,
    )
    generator = eqx.tree_deserialise_leaves(args.Gen_save_path, generator)
    model_name = args.Gen_save_path[16:-14]
    # plots
    ts_total = jnp.concatenate((ts, ts_test), axis=1)
    ys_total = jnp.concatenate((ys, ys_test), axis=1)[..., 0]
    split_idx = ts.shape[1]

    from scipy.io import savemat

    def _sample_saved(
            _ts, _ys,
            _lens,
            _n_samples=10,
            _s_key=sample_key
    ):
        train_lens = _lens
        # generate result of the samples.
        _ts_samples = jnp.tile(_ts, _n_samples).reshape(_n_samples, -1)
        _ys_sampled = jax.vmap(generator)(_ts_samples, key=jr.split(_s_key, _n_samples))[..., 0]

        _ys_train_real, _ys_test_real = _ys[:, :train_lens], _ys[:, train_lens:]
        _ys_train_gen, _ys_test_gen = _ys_sampled[:, :train_lens], _ys_sampled[:, train_lens:]
        savemat(rf"C:\Users\DTY\Desktop\我的电脑\1研究生成果\Paper_NFSDE股价\期权计算/{model_name}_{_n_samples}.mat",
                {'data': _ys_test_gen})
        return

    samples = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    # Normalize the strong or weak error
    for n in samples:
        _sample_saved(ts_total, ys_total, split_idx, n, sample_key)
    return


def test_and_save_path(
        Gentype: list = ["NFSDE", "NSDE"],
        Distype: list = ["NCDE", "NCDE_sa"],
):
    # loop for save path.
    for Gtp in Gentype:
        for Dtp in Distype:
            print(f"\n{'*' * 5} G:{Gtp}, D:{Dtp} {'*' * 5}")
            model_path = f"./Output_models/{Gtp}_{Dtp}/"
            args = load_args(model_path)
            test_only_save(args)
    return


if __name__ == "__main__":
    test_and_save_path()
