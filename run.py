# version 2.0

import multiprocessing
import os
import sys
import time
from typing import List, Optional
from datetime import datetime

import neptune.new as neptune  # type: ignore
import numpy as np
import tqdm

from datamin.utils.config import LoadMultiMinimizerConfig
from datamin.bucketization import Bucketization, MultiBucketization
from datamin.classifiers.classifier_factory import get_classifier_from_config
from datamin.dataset import FolktablesDataset
from datamin.evaluator import Evaluator
from datamin.minimizers.minimizer_factory import get_minimizer
from datamin.utils.args_factory import get_configs
from datamin.utils.config import (
    Config,
    DatasetConfig,
    LoadMinimizerConfig,
    MinimizerConfig,
)
from datamin.utils.initialization import seed_everything, setup_run_out_dir
from datamin.utils.logging_utils import CLogger, get_logger


def get_bucketization(
    cfg: MinimizerConfig,
    dataset: FolktablesDataset,
    dataset_config: DatasetConfig,
    logger: CLogger,
    run: Optional[neptune.Run] = None,
) -> Bucketization:
    if isinstance(cfg, LoadMinimizerConfig):
        bucketization = Bucketization(dataset)
        assert cfg.load_bucketization_path is not None
        bucketization.from_json_file(cfg.load_bucketization_path)
    else:
        # Instantiate the minimizer
        logger.info(f"\nRunning data minimizer: {cfg.minimizer.name}")
        minimizer = get_minimizer(cfg, dataset_config, logger, run)
        # Minimize
        minimizer.fit(
            dataset
        )  # should just be ready to call get_bucketization() which can transform

        # Get and save bucketization
        bucketization = minimizer.get_bucketization()
    return bucketization


def get_bucketizations(
    cfg: Config,
    dataset: FolktablesDataset,
    logger: CLogger,
    run: Optional[neptune.Run] = None,
) -> List[Bucketization]:
    bucketizations = []
    if isinstance(cfg.min_config, List):
        for min_config in cfg.min_config:
            bucketization = get_bucketization(
                min_config, dataset, cfg.dataset_config, logger, run
            )
            bucketizations.append(bucketization)
    elif isinstance(cfg.min_config, LoadMultiMinimizerConfig):
        # Get all files in the folder
        files = os.listdir(cfg.min_config.load_bucketization_folder)
        files = [
            os.path.join(cfg.min_config.load_bucketization_folder, f)
            for f in files
            if f.endswith(".txt")
        ]
        # Load all bucketizations
        for f in files:
            bucketization = Bucketization(dataset)
            bucketization.from_json_file(f)
            bucketizations.append(bucketization)

    else:
        bucketization = get_bucketization(
            cfg.min_config, dataset, cfg.dataset_config, logger, run
        )
        bucketizations.append(bucketization)
    return bucketizations


def save_run(cfg: Config) -> None:
    try:
        run(cfg)
    except Exception as e:
        print(e)


def run(cfg: Config) -> None:
    # Set seed
    seed_everything(cfg.seed)

    # Get file_path for output
    file_path = setup_run_out_dir(cfg)

    # Get logger
    logger = get_logger(f"datamin-{file_path}", cfg, file_path)
    logger.info(cfg)

    # Init neptune
    run: Optional[neptune.Run] = None
    if cfg.neptune_config.use_neptune:
        logger.debug("neptune")
        run = neptune.init(
            project="ethsri/data-minimization",
            api_token=cfg.neptune_config.neptune_key,
        )
        run["parameters"] = vars(cfg)
        run["label"] = cfg.neptune_config.neptune_run_label
    else:
        logger.debug("no neptune")
        run = None

    # Fetch dataset
    dataset_config = cfg.dataset_config
    require_all_values_in_train = True  # For PAT we require each class to be present at least once in the training set
    logger.debug("dataset fetched")

    dataset = FolktablesDataset(
        dataset_config,
        dataset_config.batch_size,
        all_values_in_train=require_all_values_in_train,
    )

    if dataset_config.shift_state is not None or dataset_config.shift_year is not None:
        logger.info(
            f"Creating shift dataset {dataset_config.shift_state} {dataset_config.shift_year} --- the transformations were fit on the original one"
        )
        dataset = FolktablesDataset(
            dataset_config,
            dataset_config.batch_size,
            scaler=dataset.scaler,
            oh_enc=dataset.oh_enc,
            feat_data=dataset.feat_data,
        )
        logger.info("Done")

    # Init evaluator
    logger.debug(f"setting up evaluator")
    evaluator = Evaluator(cfg.eval_config, run, dataset, cfg.device)

    # Get clf LB
    logger.debug(f"training")
    train_lb = np.maximum(np.mean(dataset.y_train), 1 - np.mean(dataset.y_train))
    test_lb = np.maximum(np.mean(dataset.y_test), 1 - np.mean(dataset.y_test))
    logger.info(
        f"majority class acc (clf ACC LB): train_acc={train_lb:.3f} test_acc={test_lb:.3f}"
    )

    # Get clf UB
    if cfg.get_clf_upper_bound:
        assert len(cfg.eval_config.clf_configs) > 0
        nb_fts = dataset.X_train_oh.shape[1]
        clf_on_orig = get_classifier_from_config(
            cfg.eval_config.clf_configs[0], nb_fts, device=cfg.device, logger=logger
        )

        clf_on_orig.fit(
            dataset.train_loader,
            cfg.eval_config.clf_configs[0].clf_epochs,
            cfg.eval_config.clf_configs[0].clf_lr,
            tune_wd=True,
            val_loader=dataset.val_loader,
        )
        acc_train = clf_on_orig.score(dataset.train_loader)
        acc_test = clf_on_orig.score(dataset.test_loader)
        logger.info(
            f"[clf_on_orig] original data (clf ACC UB): train_acc={acc_train:.3f}, test_acc={acc_test:.3f}"
        )

    # Get adv LB
    majority_freqs_train, majority_freqs_test = [], []
    for i in dataset.sens_feats:
        freqs_train = (
            np.unique(dataset.X_train_orig[:, i], return_counts=True)[1]
            / dataset.X_train_orig.shape[0]
        )
        freqs_test = (
            np.unique(dataset.X_test_orig[:, i], return_counts=True)[1]
            / dataset.X_test_orig.shape[0]
        )
        majority_freqs_train += [freqs_train.max()]
        majority_freqs_test += [freqs_test.max()]
        logger.info(
            f"feat={dataset.feature_names[i]}, majority_freqs: train={majority_freqs_train[-1]:.3f} test={majority_freqs_test[-1]:.3f}"
        )
    logger.info(
        f"[adv] naive majority freq guess (adv ACC LB): train_acc={np.mean(majority_freqs_train):.3f}, test_acc={np.mean(majority_freqs_test):.3f}"
    )

    bucketizations = get_bucketizations(cfg, dataset, logger, run)

    # Print
    logger.debug("Final bucketizations:")
    for buck in bucketizations:
        buck.print_buckets(logger=logger)
        k_anon_stats = buck._k_anonymity_stats(dataset.train_loader.dataset.tensors[0])
        l_div_stats = buck._l_diversity_stats(
            dataset.train_loader.dataset.tensors[0],
            dataset.train_loader.dataset.tensors[1],
        )
        # k_anon_str = ", ".join([f"{i}={k_anon_stats[i]}" for i in range(len(k_anon_stats))])
        # l_div_str = ", ".join([f"{i}={l_div_stats[i]:.4f}" for i in range(len(l_div_stats))])
        logger.info(
            f"k_anon={min(k_anon_stats)} , l_div={min(l_div_stats)}, size={len(k_anon_stats)}"
        )

    bucketization: Bucketization
    if len(bucketizations) > 1:
        bucketization = MultiBucketization(dataset, bucketizations)  # type: ignore
    else:
        bucketization = bucketizations[0]  # type: ignore

    label = (
        cfg.neptune_config.neptune_run_label
        if cfg.neptune_config.use_neptune
        else str(int(time.time()))
    )
    # TD add bucketization to the run name
    filename = os.path.join(cfg.out_dir, "bucketizations", f"buckets_{label}.json")
    buck_path = os.path.join("bucketizations", file_path)
    os.makedirs(os.path.dirname(buck_path), exist_ok=True)
    if not isinstance(bucketization, MultiBucketization):
        bucketization.to_json_file(buck_path)
    if run is not None:
        logger.info("Uploading bucketization to neptune...")
        run["bucketization"].upload(filename)

    # Evaluate
    logger.info("Final results:")
    tune_wd = True
    results = evaluator.evaluate(
        bucketization,
        verbose=True,
        tune_wd=tune_wd,
        guarantees=cfg.compute_guarantees,
        fairness_sens_col=cfg.fairness_sens_col,
        logger=logger,
    )
    logger.info(results)
    print(f"Results({filename}: {results}")
    if run is not None:
        run["status"] = "done"


###########################
if __name__ == "__main__":
    start_time = datetime.now()
    print("\n\n\nCommand:", " ".join(sys.argv))
    print(f"Run started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n\n")

    configs = get_configs()
    cpu_count = configs[0].num_workers  # cpu_count()
    print(f" Num configs: {len(configs)} Num workers: {cpu_count}")
    if cpu_count > 1:
        with multiprocessing.get_context("spawn").Pool(processes=cpu_count) as pool:
            list(tqdm.tqdm(pool.imap(save_run, configs), total=len(configs)))
            # pool.map(run, configs)
    else:
        for cfg in configs:
            run(cfg)
            
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"\nRun finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total elapsed time: {str(elapsed).split('.')[0]} (hh:mm:ss)")
