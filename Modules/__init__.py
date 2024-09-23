# -*- coding:utf-8 -*-
"""
    Code for models from Kidger Patrick.
@inproceedings{kidger2021sde1,
    title={{N}eural {SDE}s as {I}nfinite-{D}imensional {GAN}s},
    author={Kidger, Patrick and Foster, James and Li, Xuechen and Lyons, Terry J},
    booktitle = {Proceedings of the 38th International Conference on Machine Learning},
    pages = {5453--5463},
    year = {2021},
    volume = {139},
    series = {Proceedings of Machine Learning Research},
    publisher = {PMLR},
}

@incollection{kidger2021sde2,
    title={{E}fficient and {A}ccurate {G}radients for {N}eural {SDE}s},
    author={Kidger, Patrick and Foster, James and Li, Xuechen and Lyons, Terry},
    booktitle = {Advances in Neural Information Processing Systems 34},
    year = {2021},
    publisher = {Curran Associates, Inc.},
}
"""

from .Generator import NSDE, NFSDE, NFSDE_sa
from .Discriminator import NCDE, NCDE_sa
