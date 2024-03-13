from typing import Optional

import neptune.new as neptune
import numpy as np

from datamin.bucketization import Bucketization
from datamin.dataset import FolktablesDataset
from datamin.utils.config import UniformMinimizerConfig
from datamin.utils.logging_utils import CLogger, get_print_logger

from .abstract_minimizer import AbstractMinimizer


# Fit uniformly, directly populates a bucketization
class UniformMinimizer(AbstractMinimizer):
    def __init__(
        self,
        config: UniformMinimizerConfig,
        logger: Optional[CLogger] = None,
        run: Optional[neptune.Run] = None,
    ):
        super(UniformMinimizer, self).__init__()
        self.run = run
        self.freeze_features = config.freeze_feature
        self.nb_buckets = config.uniform_buckets
        if logger is None:
            logger = get_print_logger("Uniform-Logger")
        self.logger = logger

    def fit(self, dataset: FolktablesDataset) -> None:
        self.bucketization = Bucketization(dataset)

        for i, (is_feat_disc, feat_name, feat_beg, feat_end, _, _) in enumerate(
            dataset.feat_data
        ):
            if is_feat_disc:
                nb_buckets = (
                    1
                    if i in self.freeze_features
                    else min(feat_end - feat_beg, self.nb_buckets)
                )
                # np.random.seed(42)  # TODO Required in case we want to check the same unioform bucketization in MBR
                mapping = np.array(
                    [np.random.randint(nb_buckets) for _ in range(feat_end - feat_beg)]
                )

                # Fix the mapping such that only a prefix of available buckets is used
                uniques = np.unique(mapping)
                fixed_mapping = np.zeros_like(mapping)
                for i, val in enumerate(uniques):
                    fixed_mapping[mapping == val] = i

                self.bucketization.add_disc(feat_name, self.nb_buckets, fixed_mapping)
            else:
                nb_buckets = 1 if i in self.freeze_features else self.nb_buckets
                borders = np.linspace(0, 1, nb_buckets + 1)
                borders = borders[1:-1]
                self.bucketization.add_cont(feat_name, self.nb_buckets, borders)

    def get_bucketization(self) -> Bucketization:
        return self.bucketization
