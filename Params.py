# -*- coding:utf-8 -*-
"""
    Code for model paramsers of SANCDE and NSDE-SAGAN respectly.
"""
import argparse
from utils import *
import json


def Stock_GAN_params(
        Gentype: Literal["NSDE", "NFSDE", "NFSDE_sa"] = "NFSDE_sa",
        Distype: Literal["NCDE", "NCDE_sa"] = "NCDE",
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
    target_path = f'./Output_models/{Gentype}_{Distype}/'
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

    if is_save:     # 将参数转换为字典并保存为JSON文件
        save_path = f"./Output_models/{args.Gentype}_{args.Distype}/"
        with open(save_path + 'settings.json', 'w') as f:
            json.dump(vars(args), f)
        print(f"{'=' * 7}Save args in {save_path + 'settings.json'} Successful!{'=' * 7}")
    return args


def load_args(args_path):
    import argparse
    # load data
    with open(args_path + 'settings.json', 'r') as f:
        saved_args = json.load(f)
    print(f"{'=' * 7}Load args from {args_path} Successful!{'=' * 7}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int, required=False,
                        help='data dim')
    parser.add_argument('--initial_noise_size', type=int, required=False,
                        help='Initial noise size for NSDEs(G & C)')
    parser.add_argument('--noise_size', type=int, required=False,
                        help='Noise size for NSDEs(G & C)')
    parser.add_argument('--hidden_size', type=int, required=False,
                        help='hidden size for MLP')
    parser.add_argument('--width_size', type=int, required=False,
                        help='width size for MLP')
    parser.add_argument('--depth', type=int, required=False,
                        help='depth for MLP')
    parser.add_argument('--stop_idx', type=float, required=False,
                        help='stop idx for MLP')
    parser.add_argument('--gen_activation', type=tuple, required=False,
                        help='Generator activation function')
    parser.add_argument('--gen_final_activation', type=tuple, required=False,
                        help='Generator activation function')
    parser.add_argument('--dis_activation', type=tuple, required=False,
                        help='Discriminator activation function')
    parser.add_argument('--dis_final_activation', type=tuple, required=False,
                        help='Discriminator activation function')
    parser.add_argument('--generator_lr', type=float, required=False,
                        help='learning rate for generator')
    parser.add_argument('--discriminator_lr', type=float, required=False,
                        help="learning rate for discriminator")
    parser.add_argument('--steps', type=int, required=False,
                        help='Training steps')
    parser.add_argument('--steps_per_print', type=int, required=False,
                        help='Show results for every step')
    parser.add_argument('--seed', type=int, required=False,
                        help='Model seed')
    parser.add_argument('--batch_size', type=int, required=False,
                        help='batch_size')
    parser.add_argument('--Gen_save_path', type=str, required=False,
                        help='path for generator model')
    parser.add_argument('--target_path', type=str, required=False,
                        help='path for generator model')
    parser.add_argument('--epsilon', type=float, required=False,
                        help='stop ratio')  # .003 -- .004
    parser.add_argument('--Gentype', type=str, required=False,
                        help='generator type')
    parser.add_argument('--Distype', type=str, required=False,
                        help='discriminator type')
    parser.add_argument('--is_save', type=bool, required=False,
                        help='discriminator type')

    parser.set_defaults(**saved_args)
    args = parser.parse_args()
    return args








