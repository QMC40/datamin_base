from typing import Optional

import neptune.new as neptune
import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    chi2,
    mutual_info_classif,
)

from datamin.bucketization import Bucketization
from datamin.dataset import FolktablesDataset
from datamin.utils.config import FeatSelMinimizerConfig
from datamin.utils.logging_utils import CLogger, get_print_logger

from .abstract_minimizer import AbstractMinimizer


# Made to fit with the rest
class FeatureSelectionMinimizer(AbstractMinimizer):
    def __init__(
        self,
        config: FeatSelMinimizerConfig,
        logger: Optional[CLogger] = None,
        run: Optional[neptune.Run] = None,
    ):
        super(FeatureSelectionMinimizer, self).__init__()
        self.k = config.featsel_k
        self.method = config.method
        self.run = run
        if logger is None:
            logger = get_print_logger("Featsel-Logger")
        self.logger = logger

    def fit(self, dataset: FolktablesDataset) -> None:
        selector = self.get_base_selector()
        selector.fit(dataset.X_train_buck_orig, dataset.y_train_buck_orig)
        support = selector.get_support()
        self.logger.info(f"Support: {support}")

        self.bucketization = Bucketization(dataset)

        assert support is not None

        for i, (is_feat_disc, name, feat_beg, feat_end, _, _) in enumerate(
            dataset.feat_data
        ):
            if support[i]:
                self.logger.info(f"{name} kept")
                if is_feat_disc:
                    sz = feat_end - feat_beg
                    self.bucketization.add_disc(name, sz, np.arange(sz))
                else:
                    self.bucketization.add_cont(name, 100, np.linspace(0, 1, 101)[1:-1])
            else:
                self.logger.info(f"{name} dropped")
                if is_feat_disc:
                    sz = feat_end - feat_beg
                    self.bucketization.add_disc(name, 1, np.full((sz,), 0))
                else:
                    self.bucketization.add_cont(name, 1, [])

    def get_bucketization(self) -> Bucketization:
        return self.bucketization

    def get_base_selector(self):
        if self.method == "chi2":
            method = chi2
        elif self.method == "anova":
            method = f_classif
        elif self.method == "mi":
            method = mutual_info_classif
        else:
            raise ValueError(f"Unknown method {self.method}")

        selector = SelectKBest(method, k=self.k)
        return selector
