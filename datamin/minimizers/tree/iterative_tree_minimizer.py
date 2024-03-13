import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import neptune.new as neptune
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from datamin.bucketization import Bucketization
from datamin.dataset import FolktablesDataset
from datamin.minimizers.abstract_minimizer import AbstractMinimizer
from datamin.minimizers.tree.tree_minimizer import TreeMinimizer
from datamin.utils.config import DatasetConfig, IterativeTreeMinimizerConfig
from datamin.utils.logging_utils import CLogger, get_print_logger


# Fits a tree minimizer on the smallest dataset that achieves similar results
class IterativeTreeMinimizer(AbstractMinimizer):
    def __init__(
        self,
        config: IterativeTreeMinimizerConfig,
        logger: Optional[CLogger],
        run: Optional[neptune.Run] = None,
    ):
        super(IterativeTreeMinimizer, self).__init__()
        self.run = run
        self.config = config

        self.eps = config.eps
        self.iterative_method = config.iter_method
        self.max_leaf_nodes = config.tree_max_leaf_nodes
        self.min_sample_leaf = config.tree_min_sample_leaf
        self.alpha = config.tree_alpha

        if logger is None:
            logger = get_print_logger("IterativeTree-Logger")
        self.logger = logger

    def set_dataset_config(self, ds_config: DatasetConfig) -> None:
        self.dataset_config = copy.deepcopy(ds_config)

    def fit(self, dataset: FolktablesDataset) -> None:

        self.dataset = dataset

        # Build Forest
        trees: List[TreeMinimizer] = []
        bucketizations: List[Bucketization] = []

        data_percentages = np.arange(0.05, 1.0, 0.05)

        if not self.iterative_method == "bottom_up":
            data_percentages = np.flip(data_percentages)

        has_minimized = False

        for i, perc in enumerate(data_percentages):
            trees.append(TreeMinimizer(self.config, self.logger, self.run))

            curr_dataset_config = self.dataset_config
            curr_dataset_config.bucketization_percent = perc

            curr_dataset = FolktablesDataset(
                curr_dataset_config,
                curr_dataset_config.batch_size,
                all_values_in_train=True,
            )

            trees[i].fit(curr_dataset)

            curr_buck = trees[i].bucketization

            # TODO
            trees[i] = None
            curr_buck.dataset = dataset

            bucketizations.append(curr_buck)

            if len(bucketizations) > 1:
                prior_sims = []
                for prev_buck in bucketizations[:-1]:
                    prior_sims.append(curr_buck.get_similarity(prev_buck))
                sim = curr_buck.get_similarity(bucketizations[-2])
                prior_sims_str = ", ".join([f"{s:.3f}" for s in prior_sims])
                self.logger.info(
                    f"[Iterative Tree] Similarity: {prior_sims_str} at {perc} of the training set"
                )
                if sim > 1 - self.eps:
                    has_minimized = True
                    self.bucketization = bucketizations[-2]
                    self.logger.info(
                        f"[Iterative Tree] Used {perc} of the training set"
                    )
                    break

        if not has_minimized:
            if self.iterative_method == "bottom_up":
                self.bucketization = bucketizations[-1]
            else:
                self.bucketization = bucketizations[0]

    def get_bucketization(self) -> Bucketization:
        return self.bucketization

    def _process_data(self, dataset: FolktablesDataset) -> Dict[str, Any]:

        X_train = dataset.X_train_orig
        X_val = dataset.X_val_orig
        X_test = dataset.X_test_orig

        cont_feats = dataset.cont_feats
        disc_feats = dataset.disc_feats
        feature_names = dataset.feature_names

        # Scale - We don't use the loader later on
        # scaler = dataset.scaler
        # X_train[:, cont_feats] = scaler.transform(X_train[:, cont_feats])
        # X_test[:, cont_feats] = scaler.transform(X_test[:, cont_feats])

        # Categories are made from train set so train set has /all/ categories present at least once
        # (Last col might group all infrequent and unknown (<5) cats)
        # (We assert that if there is an unknown cat in test set there is an infreq column to put it in,
        # so no example has [0,0,0,0] -> important for the tree)
        # Test set might not have some categories represented, but that's irrelevant
        oh_enc = OneHotEncoder(
            sparse=False, handle_unknown="infrequent_if_exist", min_frequency=5
        )  # if under 5 examples
        oh_enc.fit(X_train[:, disc_feats])
        X_train_oh = np.concatenate(
            [X_train[:, cont_feats], oh_enc.transform(X_train[:, disc_feats])], axis=1
        )
        X_val_oh = np.concatenate(
            [X_val[:, cont_feats], oh_enc.transform(X_val[:, disc_feats])], axis=1
        )
        X_test_oh = np.concatenate(
            [X_test[:, cont_feats], oh_enc.transform(X_test[:, disc_feats])], axis=1
        )

        # make ft pos
        ft_pos: Dict[str, Union[int, Tuple[int, int]]] = {}
        beg = 0
        enc_idx = 0
        for i in range(len(cont_feats) + len(disc_feats)):
            name = feature_names[i]
            if i in disc_feats:
                # tot_vals = len(np.unique(X_train[:, i])) # we want categories
                # take into account infrequent categories
                tot_vals = oh_enc.categories_[enc_idx].shape[0]
                inf = oh_enc.infrequent_categories_[enc_idx]
                if inf is not None:
                    tot_vals -= inf.shape[0]
                    tot_vals += 1

                ft_pos[name] = (beg, beg + tot_vals)
                beg += tot_vals
                enc_idx += 1
            elif i in cont_feats:
                ft_pos[name] = beg
                beg += 1
            else:
                assert False

        if oh_enc.infrequent_categories_:
            print(f"NUM INFREQUENT CATEGORIES: {len(oh_enc.infrequent_categories_)}")
        else:
            print(f"NUM INFREQUENT CATEGORIES: {0}")

        inf_cat_rev_mapping: List[Optional[np.ndarray]] = [None] * len(cont_feats)
        for i, cat in enumerate(oh_enc.infrequent_categories_):
            if cat is None:
                inf_cat_rev_mapping.append(None)
            else:
                rev_map = np.array(
                    [int(np.where(elem == oh_enc.categories_[i])[0]) for elem in cat]
                )
                inf_cat_rev_mapping.append(rev_map)

        # checks
        assert enc_idx == len(oh_enc.categories_)
        assert X_train_oh.shape[1] == beg

        cnt_maptozero = 0
        cnt_emptycat = 0
        for it, X in enumerate([X_train_oh, X_test_oh]):
            for v in ft_pos.values():
                if not isinstance(v, tuple):
                    continue
                slic = X[:, v[0] : v[1]]
                maptozero = np.where(slic.sum(1) == 0)[0].shape[0]
                emptycat = np.where(slic.sum(0) == 0)[0].shape[0]
                # print(f'{maptozero} {emptycat}')
                cnt_maptozero += maptozero
                cnt_emptycat += emptycat
            # print('---')
            if it == 0:
                assert cnt_emptycat == 0
        assert cnt_maptozero == 0

        # Reverse the OH-Encoding to get indices directly
        x_train, x_val, x_test = [], [], []
        cat_pos = []
        for new_idx, idx in enumerate(ft_pos.values()):
            if isinstance(idx, tuple):
                # cat
                slc = X_train_oh[:, idx[0] : idx[1]]
                assert slc.max(axis=1).min().item() == 1
                x_train.append(slc.argmax(axis=1) + 1)

                slc = X_val_oh[:, idx[0] : idx[1]]
                assert slc.max(axis=1).min().item() == 1
                x_val.append(slc.argmax(axis=1) + 1)

                slc = X_test_oh[:, idx[0] : idx[1]]
                assert slc.max(axis=1).min().item() == 1
                x_test.append(slc.argmax(axis=1) + 1)

                cat_pos.append(new_idx)
            else:
                # cont
                x_train.append(X_train_oh[:, idx])
                x_val.append(X_val_oh[:, idx])
                x_test.append(X_test_oh[:, idx])

        x_train_stack = np.vstack(x_train).T
        x_val_stack = np.vstack(x_val).T
        x_test_stack = np.vstack(x_test).T

        s_train = x_train_stack[:, dataset.sens_feats]  # type: ignore[call-overload]
        s_val = x_val_stack[:, dataset.sens_feats]  # type: ignore[call-overload]
        s_test = x_test_stack[:, dataset.sens_feats]  # type: ignore[call-overload]

        prep_data = {
            "train": (x_train_stack, s_train.astype(np.int), dataset.y_train.astype(np.int)),  # type: ignore[attr-defined]
            "val": (x_val_stack, s_val.astype(np.int), dataset.y_val.astype(np.int)),  # type: ignore[attr-defined]
            "test": (x_test_stack, s_test.astype(np.int), dataset.y_test.astype(np.int)),  # type: ignore[attr-defined]
            "ft_pos": ft_pos,
            "rev_map": inf_cat_rev_mapping,
        }
        return prep_data
