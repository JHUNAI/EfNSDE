from Trainings import train_by_params
from Testings import test_and_save_err

# =======================================
# train_by_params: 通过参数训练。
# test_and_save_err: 测试并计算误差。
# =======================================
if __name__ == "__main__":
    # background:
    # Generator: NFSDE, NFSDE_sa, NSDE
    # Discriminator: NCDE, NCDE_sa
    # ========================================
    # Single model train: NFSDE + NCDE_SA
    # ========================================

    Gentype = ["NFSDE", "NSDE"]
    Distype = ["NCDE", "NCDE_sa"]

    # train_by_params(
    #     Gentype,
    #     Distype,
    #     is_save=True,
    # )

    test_and_save_err(
        Gentype,
        Distype,
    )

