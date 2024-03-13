from typing import Optional

import neptune.new as neptune
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from datamin.bucketization import Bucketization
from datamin.classifiers.classifier import Classifier
from datamin.dataset import FolktablesDataset
from datamin.utils.config import IBMMinimizerConfig
from datamin.utils.logging_utils import CLogger, get_print_logger

from .abstract_minimizer import AbstractMinimizer
from .ibm_apt_core.minimizer import GeneralizeToRepresentative


# Wrap to fit IBMs API
class ClassifierWrapper:
    def __init__(self, config: IBMMinimizerConfig, clf_on_orig: Classifier):
        self.device = config.device
        self.batch_size = config.batch_size
        self.clf_on_orig = clf_on_orig

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        data = TensorDataset(
            torch.tensor(X.astype(float)).float(), torch.tensor(y).long()
        )
        loader = DataLoader(data, batch_size=self.batch_size)
        return self.clf_on_orig.score(loader)


# Fit uniformly, directly populates a bucketization
class IbmAptMinimizer(AbstractMinimizer):
    def __init__(
        self,
        config: IBMMinimizerConfig,
        logger: Optional[CLogger] = None,
        run: Optional[neptune.Run] = None,
    ):
        super(IbmAptMinimizer, self).__init__()
        self.max_tree_depth = config.ibm_max_tree_depth
        self.target = config.ibm_target
        self.config = config
        self.run = run
        if logger is None:
            logger = get_print_logger("IBM-Logger")
        self.logger = logger

    def get_bucketization(self) -> Bucketization:
        return self.bucketization

    def fit(self, dataset: FolktablesDataset) -> None:
        # Train on orig
        nb_fts = dataset.X_train_oh.shape[1]
        clf_on_orig = Classifier(
            self.config.device, self.config.clf_config.clf_model, nb_fts
        )
        clf_on_orig.fit(
            dataset.buck_train_loader,
            self.config.clf_config.clf_epochs,
            self.config.clf_config.clf_lr,
            tune_wd=True,
            val_loader=dataset.val_loader,
        )
        acc_train = clf_on_orig.score(dataset.buck_train_loader)
        acc_test = clf_on_orig.score(dataset.test_loader)
        self.logger.info(
            f"[IBM clf_on_orig] original data (clf ACC UB): train_acc={acc_train:.3f}, test_acc={acc_test:.3f}"
        )

        minimizer = GeneralizeToRepresentative(
            ClassifierWrapper(self.config, clf_on_orig),
            categorical_features=[str(x) for x in dataset.disc_feats],
            target_accuracy=self.target,
            max_depth=self.max_tree_depth,
        )

        nb_feats = len(dataset.feat_data)
        names = [str(x) for x in range(nb_feats)]

        self.logger.info("IBM fitting... timeout: " + str(self.config.timeout))
        minimizer.fit(
            dataset.X_train_buck_orig,
            dataset.y_train_buck_orig,
            dataset.X_val_orig,
            dataset.y_val,
            features_names=names,
            timeout=self.config.timeout,
        )
        self.logger.info(f"IBM done, NCP: {minimizer.ncp_}")
        gens = minimizer.generalizations_
        self.logger.info(gens)

        self.bucketization = Bucketization(dataset)

        for i, (
            is_feat_disc,
            name,
            feat_beg,
            feat_end,
            orig_categories,
            _,
        ) in enumerate(dataset.feat_data):
            if is_feat_disc:
                # DISCRETE
                if "untouched" in gens and str(i) in gens["untouched"]:
                    # leave discrete feature unchanged, same #buckets
                    sz = feat_end - feat_beg
                    self.bucketization.add_disc(name, sz, np.arange(sz))
                    continue
                elif "categories" not in gens or str(i) not in gens["categories"]:
                    raise RuntimeError(f"{str(i)} not in gens categories")

                # get proposed buckets
                sz = feat_end - feat_beg
                mapping = np.full(sz, -1)
                buckets = gens["categories"][str(i)]
                new_bucket_id = 0
                for bucket in buckets:
                    for val in bucket:
                        # find old bucket id for this value
                        old_bucket_id = np.where(np.abs(orig_categories - val) < 1e-6)[
                            0
                        ]
                        mapping[old_bucket_id] = new_bucket_id
                    new_bucket_id += 1
                assert len(np.where(mapping == -1)[0]) == 0
                self.bucketization.add_disc(name, len(buckets), mapping)
            else:
                # CONTINUOUS
                if "untouched" in gens and str(i) in gens["untouched"]:
                    # leave continuous feature unchanged? TODO
                    sz = 100
                    borders = np.quantile(
                        dataset.X_train_buck_oh[:, i], np.linspace(0, 1, sz + 1)
                    )
                    borders = borders[1:-1]
                    self.bucketization.add_cont(name, sz, borders)
                    continue
                elif "ranges" not in gens or str(i) not in gens["ranges"]:
                    raise RuntimeError(f"{str(i)} not in gens ranges")

                # get proposed buckets
                ranges = gens["ranges"][str(i)]
                self.bucketization.add_cont(name, len(ranges) + 1, np.asarray(ranges))
        self.logger.info("Done ibm")
