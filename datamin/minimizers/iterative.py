import time
from typing import Optional, Tuple

import neptune.new as neptune
import numpy as np
from bunch import Bunch
from sklearn.linear_model import LogisticRegression

from datamin.bucketization import Bucketization
from datamin.dataset import FolktablesDataset
from datamin.evaluator import Evaluator
from datamin.utils.config import EvaluatorConfig, IterativeMinimizerConfig
from datamin.utils.logging_utils import CLogger, get_print_logger

from .abstract_minimizer import AbstractMinimizer


# Fit uniformly, directly populates a bucketization
class IterativeMinimizer(AbstractMinimizer):
    def __init__(
        self,
        config: IterativeMinimizerConfig,
        logger: Optional[CLogger] = None,
        run: Optional[neptune.Run] = None,
    ):
        super(IterativeMinimizer, self).__init__()
        self.init_nb_buckets = config.iterative_init_buckets
        self.target = config.iterative_target
        self.config = config
        self.run = run
        self.tune_wd = not config.iterative_fix_wd
        self.timeout = config.timeout
        self.timeout_per_feature = config.timeout_per_feature
        if logger is None:
            logger = get_print_logger("Classifier-Logger")
        self.logger = logger

    def get_bucketization(self) -> Bucketization:
        return self.bucketization

    def _evaluate_and_get_scores(self) -> Tuple[float, float]:
        res = self.evaluator.evaluate(
            self.bucketization, self.logger, tune_wd=self.tune_wd, guarantees=False
        )
        if "adv_recovery" not in res:
            res["adv_recovery"] = res["adv_recovery-ops"]
        return res["clf"][0][1], res["adv_recovery"][0][1]  # validation

    def fit(self, dataset: FolktablesDataset) -> None:
        self.dataset = dataset
        eval_config = EvaluatorConfig(Bunch())

        self.config.clf_config.clf_tune_wd = self.tune_wd
        self.config.adv_config.clf_config.clf_tune_wd = self.tune_wd

        eval_config.clf_configs = [self.config.clf_config]
        eval_config.adv_configs = [self.config.adv_config]
        self.evaluator = Evaluator(eval_config, self.run, self.dataset)

        # TODO: move boundaries for cont.
        nb_fts = len(dataset.feat_data)

        # Train logreg
        logreg = LogisticRegression(random_state=0, max_iter=1000).fit(
            dataset.X_train_buck_oh, dataset.y_train_buck_orig
        )
        self.logreg_coeffs = logreg.coef_[0]
        score_train = logreg.score(dataset.X_train_buck_oh, dataset.y_train_buck_orig)
        score_val = logreg.score(dataset.X_val_oh, dataset.y_val)
        self.logger.info(f"Logreg score: {score_train:.3f} {score_val:.3f}")

        # This is the initial solution
        self.bucketization = Bucketization(dataset)
        for i in range(nb_fts):
            self._set_feature(i, self.init_nb_buckets)

        clf_val, adv_val = self._evaluate_and_get_scores()

        # Sort by ImpactDifferenceEstimate (IDE)
        impact = np.zeros(nb_fts)
        ide = np.zeros(nb_fts)

        self.logger.info("Calculating IDE")
        for i, (_, _, feat_beg, feat_end, _, _) in enumerate(dataset.feat_data):
            self.logger.info(f"Feature {i}")
            impact[i] = np.abs(np.mean(self.logreg_coeffs[feat_beg:feat_end]))

            self._set_feature(i, 1)
            C, A = self._evaluate_and_get_scores()

            ide[i] = (adv_val - A) - (clf_val - C)  # bigger = better
            self.logger.info(f"ide = {ide[i]}")
            self._set_feature(i, self.init_nb_buckets)

        idxs = list(reversed(np.argsort(ide)))
        self.logger.info(f"Sort is: {idxs}")

        start_time = time.time()
        self.logger.info(
            f"Going through features descending w.r.t. IDE, Start_time: {start_time} Timeout: {self.timeout}\n"
        )
        for i in idxs:

            inner_start = time.time()

            if self.timeout is not None and time.time() - start_time > self.timeout:
                self.logger.info("Timeout, stopping")
                break
            if (
                self.timeout_per_feature is not None
                and time.time() - inner_start > self.timeout_per_feature
            ):
                continue

            sz = 0
            self.logger.info(f"Feature {i}")
            is_feat_disc, name, feat_beg, feat_end, _, _ = dataset.feat_data[i]

            do_rollback = False
            size = self.bucketization.buckets[name]["size"]
            assert isinstance(size, int)
            while size > 1:
                sz = size
                self.logger.info(f"Going down to {sz-1}")
                self._set_feature(i, sz - 1)

                C, A = self._evaluate_and_get_scores()
                if C < self.target + 0.001:
                    do_rollback = True
                    break
                adv_val, clf_val = A, C

                if self.timeout is not None and time.time() - start_time > self.timeout:
                    break
                if (
                    self.timeout_per_feature is not None
                    and time.time() - inner_start > self.timeout_per_feature
                ):
                    break

            if do_rollback:
                self.logger.info(f"Going up (rollback) to {sz}")
                self._set_feature(i, sz)
        self.logger.info("Done with all, bucketization is stable")

    def _set_feature(self, i: int, K: int) -> None:
        # Takes feature (i) and adds it to the bucketization (or replaces)
        is_feat_disc, name, feat_beg, feat_end, _, _ = self.dataset.feat_data[i]
        if is_feat_disc:
            sz = feat_end - feat_beg
            if K > sz:
                self.bucketization.add_disc(name, sz, np.arange(sz))
            else:
                coeffs = self.logreg_coeffs[feat_beg:feat_end]
                mapping = self._optimal_k_split(K, coeffs)
                self.bucketization.add_disc(name, K, mapping)
        else:
            borders = np.quantile(
                self.dataset.X_train_buck_oh[:, i], np.linspace(0, 1, K + 1)
            )
            borders = borders[1:-1]
            self.bucketization.add_cont(name, K, borders)

    def _optimal_k_split(self, k: int, coeffs: np.ndarray) -> np.ndarray:
        # Find an optimal split of {coeffs} into exactly k sets s.t.
        # average set variance is minimized --> returns a mapping
        # of each original index in [0, sz) to [0, k) -> bucketization
        sz = len(coeffs)

        coeffs_sorted, coeffs_idxs = np.sort(coeffs), np.argsort(coeffs)

        mapping = np.full(sz, -1)
        dp = np.full((sz, k + 1), 1e9)
        fsts = np.full((sz, k + 1), -1)

        for i in range(sz):
            for kk in range(1, k + 1):
                # Split [0, i] into kk groups
                if kk > i + 1:
                    continue
                if kk == 1:
                    dp[i][1] = coeffs_sorted[: (i + 1)].var()
                    fsts[i][1] = 0
                    continue
                for fst in range(kk - 1, i + 1):
                    cand = (
                        dp[fst - 1][kk - 1] * (kk - 1)
                        + coeffs_sorted[fst : (i + 1)].var()
                    ) / kk
                    if cand < dp[i][kk]:
                        dp[i][kk] = cand
                        fsts[i][kk] = fst

        # Solution?
        starts = []
        curr = sz - 1
        kk = k
        while kk > 0:
            curr = fsts[curr][kk]
            starts.append(curr)
            curr -= 1
            kk -= 1

        # Make mapping
        component = -1
        for i in range(sz):
            if i in starts:
                component += 1
            mapping[coeffs_idxs[i]] = component

        for i in range(sz):
            assert mapping[i] >= 0

        return mapping
