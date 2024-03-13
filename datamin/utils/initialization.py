import os
import random
from typing import List

import numpy as np
import torch

from datamin.utils.config import ClassifierType, Config


def seed_everything(seed: int) -> None:
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)

    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def setup_run_out_dir(cfg: Config) -> str:

    if isinstance(cfg.min_config, List):
        id = "continual"
        suffix = "continual_[" + "|".join([str(m) for m in cfg.min_config]) + "]"
    else:
        id = cfg.min_config.minimizer.name
        suffix = str(cfg.min_config)

    parent_dir = os.path.join(cfg.out_dir, cfg.dataset_config.id_str(), id)

    for clf_conf in cfg.eval_config.clf_configs:
        if clf_conf.clf_type == ClassifierType.DP:
            parent_dir = f"{parent_dir}_dp_{clf_conf.clf_dp_noise}"
            break

    cfg_id_str = f"{suffix}.txt"

    if cfg.logger_level != "STDOUT":
        os.makedirs(parent_dir, exist_ok=True)

    return os.path.join(parent_dir, cfg_id_str)
