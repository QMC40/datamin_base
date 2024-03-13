import argparse
import os
from typing import List

import torch
import yaml  # type: ignore[import]
from bunch import Bunch

from datamin.utils.config import (
    AdversaryConfig,
    AdvMinimizerConfig,
    ClassifierConfig,
    Config,
    FeatSelMinimizerConfig,
    IBMMinimizerConfig,
    IterativeMinimizerConfig,
    LoadMinimizerConfig,
    MutualInfMinimizerConfig,
    TreeMinimizerConfig,
    UniformMinimizerConfig,
    get_minimizer_config,
    make_config,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None)

    parser.add_argument(
        "--device", type=str, default="cpu" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--neptune", type=str, default=None)  #
    parser.add_argument("--out-dir", type=str, default="out/")
    parser.add_argument("--seed", type=int, default=100)  #
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--dataset", type=str, default=None)  #
    parser.add_argument("--acs-state", type=str, default="CT")
    parser.add_argument("--acs-year", type=int, default=2014)
    parser.add_argument("--freeze-features", nargs="+", default=[], type=int)

    parser.add_argument(
        "--minimizer",
        type=str,
        default=None,
        choices=["tree", "advtrain", "featsel", "ibm", "iterative", "mi", "uniform"],
    )

    # Load from file
    parser.add_argument("--load-bucketization", type=str, default=None)
    parser.add_argument(
        "--percent", type=float, default=None
    )  # % of training data used

    parser.add_argument(
        "--compute-guarantees", default=False, action="store_true"
    )  # "adver"
    parser.add_argument(
        "--fairness-sens-col", type=int, default=None
    )  # disable tune-wd
    parser.add_argument("--shift-state", type=str, default=None)
    parser.add_argument("--shift-year", type=int, default=None)

    # Classifier & Adversary used in Evaluator (+ methods that utilize them inside)
    parser.add_argument("--clf-model", type=str, default="mlp2")
    parser.add_argument("--clf-epochs", type=int, default=20)
    parser.add_argument("--clf-lr", type=float, default=1e-2)
    parser.add_argument("--adv-model", type=str, default="mlp2")
    parser.add_argument("--adv-epochs", type=int, default=20)
    parser.add_argument("--adv-lr", type=float, default=1e-2)

    # NOTE: when used in Evaluator WD is always tuned
    parser.add_argument("--clf-weight-decay", type=float, default=0.0)
    parser.add_argument("--adv-weight-decay", type=float, default=0.0)

    #############
    # Method-specific configs
    #############

    # Tree
    parser.add_argument("--tree_max_leaf_nodes", type=int, default=20)
    parser.add_argument("--tree_min_sample_leaf", type=int, default=100)
    parser.add_argument("--tree_alpha", type=float, default=0.7)

    # adv_train
    parser.add_argument("--advtrain-max-buckets", type=int, default=5)
    parser.add_argument("--advtrain-n-epochs", type=int, default=20)
    parser.add_argument("--advtrain-inner-steps", type=int, default=1)
    parser.add_argument("--advtrain-weight", type=float, default=0.0)

    # feat_sel
    parser.add_argument("--featsel-k", type=int, default=2)

    # ibm
    parser.add_argument("--ibm-max-tree-depth", type=int, default=None)
    parser.add_argument("--ibm-target", type=float, default=0.7)

    # iterative
    parser.add_argument("--iterative-target", type=float, default=0.8)
    parser.add_argument("--iterative-init-buckets", type=int, default=4)
    parser.add_argument("--iterative-fix-wd", default=False, action="store_true")

    # mutual_inf
    parser.add_argument("--mi-max-buckets", type=int, default=5)
    parser.add_argument("--mi-n-epochs", type=int, default=20)
    parser.add_argument("--mi-weight", type=float, default=0.0)

    # uniform
    parser.add_argument("--uniform-buckets", type=int, default=5)

    args = parser.parse_args()
    return args


# flake8: noqa: C901
def update_config_with_args(config: Config, args: argparse.Namespace) -> Config:

    if args.device is not None:
        config.device = args.device

    if args.neptune is not None:
        config.neptune_config.use_neptune = args.neptune

    if args.out_dir is not None:
        config.out_dir = args.out_dir

    if args.seed is not None:
        config.seed = args.seed

    if args.batch_size is not None:
        config.dataset_config.batch_size = args.batch_size

    if args.dataset is not None:
        config.dataset_config.dataset = args.dataset

    if args.acs_state is not None:
        config.dataset_config.acs_state = args.acs_state

    if args.acs_year is not None:
        config.dataset_config.acs_year = args.acs_year

    if args.shift_state is not None:
        config.dataset_config.shift_state = args.shift_state

    if args.shift_year is not None:
        config.dataset_config.shift_year = args.shift_year

    if args.freeze_features is not None:
        config.dataset_config.freeze_feature = args.freeze_features

    if args.freeze_features is not None:
        config.dataset_config.freeze_feature = args.freeze_features

    if args.percent is not None:
        config.dataset_config.train_percent = args.percent

    if args.compute_guarantees is not None:
        config.compute_guarantees = args.compute_guarantees

    if args.fairness_sens_col is not None:
        config.fairness_sens_col = args.fairness_sens_col

    config.eval_config.clf_configs.append(ClassifierConfig(Bunch()))
    config.eval_config.adv_configs.append(AdversaryConfig(Bunch()))

    assert len(config.eval_config.clf_configs) == 1
    assert len(config.eval_config.adv_configs) == 1
    if args.clf_model is not None:
        config.eval_config.clf_configs[0].clf_model = args.clf_model
    if args.clf_epochs is not None:
        config.eval_config.clf_configs[0].clf_epochs = args.clf_epochs
    if args.clf_lr is not None:
        config.eval_config.clf_configs[0].clf_lr = args.clf_lr
    if args.clf_weight_decay is not None:
        config.eval_config.clf_configs[0].clf_weight_decay = args.clf_weight_decay
    if args.adv_model is not None:
        config.eval_config.adv_configs[0].clf_config.clf_model = args.adv_model
    if args.adv_epochs is not None:
        config.eval_config.adv_configs[0].clf_config.clf_epochs = args.adv_epochs
    if args.adv_lr is not None:
        config.eval_config.adv_configs[0].clf_config.clf_lr = args.adv_lr
    if args.adv_weight_decay is not None:
        config.eval_config.adv_configs[
            0
        ].clf_config.clf_weight_decay = args.adv_weight_decay

    if args.minimizer is not None:
        config.min_config = get_minimizer_config(
            Bunch(minimizer=args.minimizer), config.eval_config
        )[0]

    if isinstance(config.min_config, TreeMinimizerConfig):
        if args.tree_max_leaf_nodes is not None:
            config.min_config.tree_max_leaf_nodes = args.tree_max_leaf_nodes
        if args.tree_min_sample_leaf is not None:
            config.min_config.tree_min_sample_leaf = args.tree_min_sample_leaf
        if args.tree_alpha is not None:
            config.min_config.tree_alpha = args.tree_alpha
    elif isinstance(config.min_config, AdvMinimizerConfig):
        if args.advtrain_max_buckets is not None:
            config.min_config.advtrain_max_buckets = args.advtrain_max_buckets
        if args.advtrain_n_epochs is not None:
            config.min_config.advtrain_n_epochs = args.advtrain_n_epochs
        if args.advtrain_inner_steps is not None:
            config.min_config.advtrain_inner_steps = args.advtrain_inner_steps
        if args.advtrain_weight is not None:
            config.min_config.advtrain_weight = args.advtrain_weight
    elif isinstance(config.min_config, FeatSelMinimizerConfig):
        if args.featsel_k is not None:
            config.min_config.featsel_k = args.featsel_k
    elif isinstance(config.min_config, IBMMinimizerConfig):
        if args.ibm_max_tree_depth is not None:
            config.min_config.ibm_max_tree_depth = args.ibm_max_tree_depth
        if args.ibm_target is not None:
            config.min_config.ibm_target = args.ibm_target
    elif isinstance(config.min_config, IterativeMinimizerConfig):
        if args.iterative_target is not None:
            config.min_config.iterative_target = args.iterative_target
        if args.iterative_init_buckets is not None:
            config.min_config.iterative_init_buckets = args.iterative_init_buckets
        if args.iterative_fix_wd is not None:
            config.min_config.iterative_fix_wd = args.iterative_fix_wd
    elif isinstance(config.min_config, UniformMinimizerConfig):
        if args.uniform_buckets is not None:
            config.min_config.uniform_buckets = args.uniform_buckets
    elif isinstance(config.min_config, LoadMinimizerConfig):
        if args.load_bucketization is not None:
            config.min_config.load_bucketization_path = args.load_bucketization
    elif isinstance(config.min_config, MutualInfMinimizerConfig):
        if args.mi_max_buckets is not None:
            config.min_config.mi_max_buckets = args.mi_max_buckets
        if args.mi_n_epochs is not None:
            config.min_config.mi_n_epochs = args.mi_n_epochs
        if args.mi_inner_steps is not None:
            config.min_config.mi_weight = args.mi_weight
    else:
        raise Exception("Unknown minimizer config type")

    return config


def get_configs() -> List[Config]:
    args = get_args()
    if args.config is not None:

        configs = []
        if os.path.isfile(args.config):
            # Load YAML config
            cfg_bunch = {}
            with open(args.config, "r") as open_file:
                cfg_bunch = yaml.load(open_file, Loader=yaml.FullLoader)
                print(cfg_bunch)

            configs = make_config(**cfg_bunch)
            return configs
        else:
            for filename in os.listdir(args.config):
                f: str = os.path.join(args.config, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    cfg_bunch = {}
                    with open(f, "r") as open_file:
                        cfg_bunch = yaml.load(open_file, Loader=yaml.FullLoader)
                        # print(cfg_bunch)
                    configs.extend(make_config(**cfg_bunch))
            return configs

    else:
        configs = make_config(**Bunch())
        assert len(configs) == 1
        config = update_config_with_args(configs[0], args)
        return [config]
