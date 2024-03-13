from typing import Any, Dict, List, Optional, Tuple, Union

import neptune.new as neptune
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from datamin.bucketization import Bucketization
from datamin.dataset import FolktablesDataset
from datamin.minimizers.abstract_minimizer import AbstractMinimizer
from datamin.utils.config import ForestMinimizerConfig
from datamin.utils.logging_utils import CLogger, get_print_logger

from .tree_util import tree_to_decision_box_boundaries


# Fit a MultiFairGini-Minimizer
class ForestMinimizer(AbstractMinimizer):
    def __init__(
        self,
        config: ForestMinimizerConfig,
        logger: Optional[CLogger],
        run: Optional[neptune.Run] = None,
    ):
        super(ForestMinimizer, self).__init__()
        self.run = run
        self.config = config

        self.max_leaf_nodes = config.tree_max_leaf_nodes
        self.min_sample_leaf = config.tree_min_sample_leaf
        self.alpha = config.tree_alpha
        self.num_trees = config.forest_n_trees

        if logger is None:
            logger = get_print_logger("Forest-Logger")
        self.logger = logger

    def fit(self, dataset: FolktablesDataset) -> None:
        self.dataset = dataset

        processed_data = self._process_data(dataset)

        # Set relevant variables
        cat_pos = np.array(dataset.disc_feats).astype(
            np.int32
        )  # np.array(dataset.sens_feats).astype(np.int32)

        x_train, s_train, y_train = processed_data["train"]
        x_val, s_val, y_val = processed_data["val"]
        x_test, s_test, y_test = processed_data["test"]
        ft_pos = processed_data["ft_pos"]
        rev_map = processed_data["rev_map"]
        # Revert One hot encoding for categoricals

        # Build Forest
        trees: List[DecisionTreeClassifier] = []
        bucketizations: List[Bucketization] = []

        for i in range(self.num_trees):
            trees.append(
                DecisionTreeClassifier(
                    criterion="multi_fair_gini",
                    max_leaf_nodes=self.max_leaf_nodes,
                    random_state=i,
                    min_samples_leaf=self.min_sample_leaf,
                )
            )

            # Subsample dataset
            rand_ids = np.random.choice(
                x_train.shape[0], size=(int(x_train.shape[0] // 1.5),), replace=False
            )
            x_train_sub = x_train[rand_ids]
            y_train_sub = y_train[rand_ids]
            s_train_sub = s_train[rand_ids]

            trees[i].fit(
                x_train_sub, y_train_sub, s_train_sub, cat_pos=cat_pos, alpha=self.alpha
            )

            encoding = tree_to_decision_box_boundaries(
                trees[i], features=ft_pos, rev_map=rev_map
            )

            # Extract Bucketization
            bucketizations.append(Bucketization(dataset))

            dte_items = list(encoding.items())

            for i in dataset.cont_feats:
                sz = len(dte_items[i][1]) + 1
                bucketizations[-1].add_cont(
                    name=dte_items[i][0], sz=sz, borders=dte_items[i][1]
                )
            for j in dataset.disc_feats:
                sz = len(np.unique(dte_items[j][1]))
                bucketizations[-1].add_disc(
                    name=dte_items[j][0], sz=sz, mapping=dte_items[j][1]
                )

        print("Hello")

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
