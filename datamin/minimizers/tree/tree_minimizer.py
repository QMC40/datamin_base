from typing import Any, Dict, List, Optional, Tuple, Union

import neptune.new as neptune
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from datamin.bucketization import Bucketization
from datamin.dataset import FolktablesDataset
from datamin.minimizers.abstract_minimizer import AbstractMinimizer
from datamin.utils.config import TreeMinimizerConfig
from datamin.utils.logging_utils import CLogger, get_print_logger

from .tree_util import tree_to_decision_box_boundaries


# Fit a MultiFairGini-Minimizer
class TreeMinimizer(AbstractMinimizer):
    def __init__(
        self,
        config: TreeMinimizerConfig,
        logger: CLogger,
        run: Optional[neptune.Run] = None,
    ):
        super(TreeMinimizer, self).__init__()
        self.run = run
        self.max_leaf_nodes = config.tree_max_leaf_nodes
        self.min_sample_leaf = config.tree_min_sample_leaf
        self.alpha = config.tree_alpha
        self.min_bucket_threshold = config.min_bucket_threshold
        self.initial_split_factor = config.initial_split_factor
        if logger is None:
            logger = get_print_logger("Tree-Logger")
        self.logger = logger

    # flake8: noqa: C901
    def fit(self, dataset: FolktablesDataset) -> None:
        self.dataset = dataset

        # Simply take (buck, val, test) folds and extract sensitive features
        # (used to do a lot more before it was synced with the main dataset)
        processed_data = self._process_data(dataset)

        # Set relevant variables
        cat_pos = np.array(dataset.disc_feats).astype(
            np.int32
        )  # np.array(dataset.sens_feats).astype(np.int32)

        x_train, s_train, y_train = processed_data["train"]
        x_val, s_val, y_val = processed_data["val"]
        x_test, s_test, y_test = processed_data["test"]
        # Revert One hot encoding for categoricals

        # Build Tree
        T = DecisionTreeClassifier(
            criterion="multi_fair_gini",
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=43,
            min_samples_leaf=self.min_sample_leaf,
        )
        # s_train = np.concatenate((s_train, s_train), axis=1)

        T = T.fit(
            x_train,
            y_train,
            s_train,
            cat_pos=cat_pos,
            alpha=self.alpha,
            initial_split_factor=self.initial_split_factor,
        )

        # Get encoding
        # meta = {"ft_pos": ft_pos, "rev_map": rev_map}
        decision_tree_encoding = tree_to_decision_box_boundaries(T, dataset.feat_data)

        # Extract Bucketization
        self.bucketization = Bucketization(dataset)

        dte_items = list(decision_tree_encoding.items())

        for i in dataset.cont_feats:
            sz = len(dte_items[i][1]) + 1
            self.bucketization.add_cont(
                name=dte_items[i][0], sz=sz, borders=dte_items[i][1]
            )
        for j in dataset.disc_feats:
            sz = len(np.unique(dte_items[j][1]))
            self.bucketization.add_disc(
                name=dte_items[j][0], sz=sz, mapping=dte_items[j][1]
            )

        self.tree = T

        # Optimization: reduce the number of buckets
        if self.min_bucket_threshold > 0:
            stats = self.bucketization.get_stats_on_input(
                self.dataset.X_train_buck_oh, self.dataset.y_train  # type: ignore
            )

            buck_usage = stats["feat_buck_usage"]
            prior_num_buckets = self.bucketization.total_size()
            total = self.dataset.X_train_buck_oh.shape[0]
            for feat, counts in buck_usage.items():
                buckets = self.bucketization.buckets[feat]
                size = buckets["size"]
                assert isinstance(size, int)
                if "mapping" in buckets:
                    expected = total / size
                    mapping = buckets["mapping"]
                    assert isinstance(mapping, List)
                    new_counts: List[int] = []
                    curr_left_idx = -1
                    reassign_list = list(
                        range(len(counts))
                    )  # Contains the index we map to
                    overlap = 0
                    for i, count in enumerate(counts):
                        if count + overlap < self.min_bucket_threshold * expected:
                            left_size = right_size = total
                            if len(new_counts) > 0:
                                left_size = new_counts[-1]
                            if i < len(counts) - 1:
                                right_size = counts[i + 1]
                            if left_size < right_size:  # Merging to the left
                                new_counts[-1] += counts[i] + overlap
                                reassign_list[i] = curr_left_idx
                                overlap = 0
                            else:
                                reassign_list[i] = i + 1
                                overlap += counts[i]
                        else:
                            curr_left_idx = i
                            new_counts.append(count + overlap)  # Merging to the right
                            overlap = 0
                    # Clean the reassign list
                    for i, j in enumerate(reassign_list):
                        if i != j:
                            curr_j = j
                            while reassign_list[curr_j] != curr_j:
                                curr_j = reassign_list[curr_j]
                            reassign_list[i] = curr_j
                    cur_val = -1
                    cur_ctr = -1
                    for i, j in enumerate(reassign_list):
                        if j != cur_val:
                            cur_val = j
                            cur_ctr += 1
                        reassign_list[i] = cur_ctr
                    # Clean up the mapping
                    new_mapping = []
                    for elem in mapping:
                        assert isinstance(elem, int)
                        new_mapping.append(reassign_list[elem])
                    new_size = len(np.unique(reassign_list))
                    self.bucketization.buckets[feat] = {
                        "size": new_size,
                        "mapping": new_mapping,
                    }
                elif "borders" in buckets:
                    expected = total / (size + 1)
                    borders = buckets["borders"]
                    assert isinstance(borders, List)
                    full_borders: List[float] = [0.0] + borders + [1.0]  # type: ignore
                    new_borders: List[float] = []
                    new_counts = []
                    overlap = 0
                    for i, count in enumerate(counts):

                        if count + overlap < self.min_bucket_threshold * expected:
                            left_size = right_size = total
                            if len(new_counts) > 0:
                                left_size = new_counts[-1]
                            if i < len(counts) - 1:
                                right_size = counts[i + 1]
                            if left_size < right_size:  # Merging to the left
                                new_counts[-1] += counts[i] + overlap
                                new_borders[-1] = full_borders[i]
                                overlap = 0
                            else:
                                overlap += counts[i]
                        else:
                            new_borders.append(full_borders[i + 1])
                            new_counts.append(count + overlap)  # Merging to the right
                            overlap = 0
                    # Remove the [...,1.0] at the end
                    self.bucketization.buckets[feat] = {
                        "size": len(new_borders),
                        "borders": new_borders[:-1],
                    }
                else:
                    assert False
            self.logger.info(
                f"Reduced number of buckets from {prior_num_buckets} to {self.bucketization.total_size()}"
            )

    def get_bucketization(self) -> Bucketization:
        return self.bucketization

    def _process_data(self, dataset: FolktablesDataset) -> Dict[str, Any]:
        X_train = dataset.X_train_buck_orig
        y_train = dataset.y_train_buck_orig
        X_val = dataset.X_val_orig
        X_test = dataset.X_test_orig

        # type: ignore[call-overload]
        s_train: np.ndarray = X_train[:, dataset.sens_feats]
        # type: ignore[call-overload]
        s_val: np.ndarray = X_val[:, dataset.sens_feats]
        # type: ignore[call-overload]
        s_test: np.ndarray = X_test[:, dataset.sens_feats]

        prep_data = {
            "train": (X_train, s_train.astype(np.int), y_train.astype(np.int)),  # type: ignore[attr-defined]
            "val": (X_val, s_val.astype(np.int), dataset.y_val.astype(np.int)),  # type: ignore[attr-defined]
            "test": (X_test, s_test.astype(np.int), dataset.y_test.astype(np.int)),  # type: ignore[attr-defined]
        }
        return prep_data
