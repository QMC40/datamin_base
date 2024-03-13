import importlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from folktables import ACSDataSource
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

from datamin.datasets.csv import load_csv
from datamin.datasets.crime import load_crime
from datamin.datasets.loan import load_loan
from datamin.datasets.health_preprocessed import load_health as load_health_preprocessed
from datamin.utils.config import DatasetConfig, SensFeat


class FolktablesDataset:
    all_cont_feats = {
        "ACSIncome": [0, 2, 7],
        "ACSEmployment": [0, 1],
        "ACSPublicCoverage": [0, 1, 14],
        "ACSMobility": [0, 1, 18, 19, 20],
        "ACSTravelTime": [0, 1, 15],
    }

    # Holds:
    #
    # feature_names, cont_feats, disc_feats, tot_feats, sens_feats
    #
    # feat_data list of big tuples
    #
    # original data: X/y_train_orig, X/y_val_orig, X/y_test_orig
    # one-hot data: same but _oh
    # loaders (with one-hot) for train, val, test

    # flake8: noqa: C901
    def __init__(
        self,
        config: DatasetConfig,
        batch_size: int,
        scaler: Optional[MinMaxScaler] = None,
        oh_enc: Optional[OneHotEncoder] = None,
        feat_data: Optional[
            List[Tuple[bool, str, int, int, Optional[int], Optional[np.ndarray]]]
        ] = None,
        add_syn: bool = False,
        all_values_in_train: bool = False,  # Whether all possible values for each feature must be present in the training data
    ):
        cont_feats: List[int] = []
        disc_feats: List[int] = []

        # Get data source
        if config.dataset.endswith(".csv"):
            data, sens_feats = load_csv(config.dataset)
            stratify = True
            features, labels = data["train"]
            assert isinstance(features, np.ndarray)
            assert isinstance(labels, np.ndarray)
            # 0-1 encode labels as int
            labels = (labels == labels.max()).astype(int)
            feature_names = data["feature_names"]

            tot_feats = features.shape[1]
            cont_feats = data["cont_features"]
            disc_feats = [i for i in range(len(feature_names)) if i not in cont_feats]
            tot_feats = len(disc_feats) + len(cont_feats)
            if (
                isinstance(config.sens_feats, SensFeat)
                and config.sens_feats.type != "selection"
            ) or (
                isinstance(config.sens_feats, str) and config.sens_feats != "selection"
            ):
                sens_feats = config.sens_feats.get_feats(
                    cont_feats, disc_feats, tot_feats
                )
            self.nb_out_fts = 2

        elif "ACS" in config.dataset:
            data_source = ACSDataSource(
                survey_year=str(config.acs_year), horizon="1-Year", survey="person"
            )
            acs_data = data_source.get_data(states=[config.acs_state], download=True)
            m = importlib.import_module("folktables")
            ACSClass = getattr(m, config.dataset)

            # Distinguish features
            features, labels, _ = ACSClass.df_to_numpy(acs_data)
            feature_names: List[str] = ACSClass.features
            stratify = True

            cont_feats = FolktablesDataset.all_cont_feats[config.dataset]
            disc_feats = [i for i in range(len(feature_names)) if i not in cont_feats]
            tot_feats = len(disc_feats) + len(cont_feats)
            sens_feats = config.sens_feats.get_feats(cont_feats, disc_feats, tot_feats)

            self.nb_out_fts = 2

        elif "health" in config.dataset:
            if config.dataset == "health_no_transfer":
                data = load_health_preprocessed(
                    label=["max_CharlsonIndex"],
                    transfer=False,
                    fnfpruned=False,
                    p_test=0.0,
                )
                self.nb_out_fts = 2
                stratify = True
            elif config.dataset == "health_pruned_transfer":
                data = load_health_preprocessed(
                    label=["max_CharlsonIndex"],
                    transfer=True,
                    fnfpruned=True,
                    p_test=0.0,
                )
                self.nb_out_fts = 2
                stratify = True
            elif config.dataset == "health_pruned_no_transfer":
                data = load_health_preprocessed(
                    label=["max_CharlsonIndex"],
                    transfer=False,
                    fnfpruned=True,
                    p_test=0.0,
                )
                self.nb_out_fts = 2
                stratify = True
            else:
                assert False, f"No dataset called: {config.dataset}"

            # Set relevant fields
            features, labels = data["train"]
            assert isinstance(features, np.ndarray)
            assert isinstance(labels, np.ndarray)

            feature_names = data["feature_names"]
            tot_feats = features.shape[1]
            cont_feats = data["cont_features"]
            disc_feats = [i for i in range(len(feature_names)) if i not in cont_feats]
            tot_feats = len(disc_feats) + len(cont_feats)
            sens_feats = config.sens_feats.get_feats(cont_feats, disc_feats, tot_feats)

            if config.dataset == "health_no_transfer":
                new_sens_feats = []
                for feat in sens_feats:
                    if feat in [2, 3, 4, 5, 8, 16, 19, 33, 44, 61, 71, 77, 92]:
                        new_sens_feats.append(feat)
                sens_feats = new_sens_feats

        elif "crime" in config.dataset:
            data = load_crime(split="train", args=None, normalize=True, p_test=0.0)
            stratify = True
            features, labels = data["train"]
            assert isinstance(features, np.ndarray)
            assert isinstance(labels, np.ndarray)
            feature_names = data["feature_names"]

            tot_feats = features.shape[1]
            cont_feats = data["cont_features"]
            disc_feats = [i for i in range(len(feature_names)) if i not in cont_feats]
            tot_feats = len(disc_feats) + len(cont_feats)
            sens_feats = config.sens_feats.get_feats(cont_feats, disc_feats, tot_feats)
            self.nb_out_fts = 2

        elif "loan" in config.dataset:
            data = load_loan()
            stratify = True
            features, labels = data["train"]
            assert isinstance(features, np.ndarray)
            assert isinstance(labels, np.ndarray)
            feature_names = data["feature_names"]

            tot_feats = features.shape[1]
            cont_feats = data["cont_features"]
            disc_feats = [i for i in range(len(feature_names)) if i not in cont_feats]
            tot_feats = len(disc_feats) + len(cont_feats)
            sens_feats = config.sens_feats.get_feats(cont_feats, disc_feats, tot_feats)
            self.nb_out_fts = 2
        else:
            assert False, f"Unknown dataset: {config.dataset}"

        X_train: np.ndarray
        X_val: np.ndarray
        X_test: np.ndarray

        # Hand-built ordinal encoder: map all categorical feature values to {1, ..., max} (needed for tree, noop for rest)
        for idx in disc_feats:
            ft_vals = np.unique(features[:, idx])
            new_col = np.zeros_like(features[:, idx], dtype=int)
            for i, val in enumerate(ft_vals):
                new_col[features[:, idx] == val] = i + 1
            features[:, idx] = new_col
            assert new_col.min().item() == 1, "Ordinal encoding must start from 1"

        # Split to get (train, val, test) folds, optionally forcing all unique values for each feature to be in train
        (
            (X_train, y_train),
            nb_preselected_rows,
            (X_val, y_val),
            (X_test, y_test),
        ) = self.split(
            features,
            labels,
            stratify=stratify,
            preselect=all_values_in_train,
            disc_feats=disc_feats,
        )

        assert X_train.shape[1] == tot_feats, "X_train has wrong number of features"
        assert len(np.unique(y_train)) == 2, "y_train should be binary"

        # Permute the order of features such that continuous go first
        perm = cont_feats + disc_feats
        invperm: Dict[int, int] = {}
        for i, idx in enumerate(perm):
            invperm[idx] = i
        X_train = X_train[:, perm]
        X_val = X_val[:, perm]
        X_test = X_test[:, perm]
        self.feature_names: List[str] = [feature_names[i] for i in perm]
        self.cont_feats = [invperm[x] for x in cont_feats]
        self.disc_feats = [invperm[x] for x in disc_feats]
        self.sens_feats = [invperm[x] for x in sens_feats]
        self.tot_feats = tot_feats

        # Fit a scaler (for continuous) and onehot encoder (for categorical features)
        if scaler is None:
            # NOTE: oh sets all to zero, check
            # NOTE: Preserve order when building _oh
            scaler = MinMaxScaler()
            if len(self.cont_feats) > 0:
                scaler.fit(X_train[:, self.cont_feats])
        if oh_enc is None:
            oh_enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
            oh_enc.fit(X_train[:, self.disc_feats])
        self.categories: List[List[int]] = (
            oh_enc.categories_
        )  # List of present categories for each feature

        # Subsample the train fold (used for classifier training on *bucketized* entries)
        if config.train_percent is not None:
            X_train, y_train = self.subsample(
                X_train,
                y_train,
                size=config.train_percent,
                important_prefix_size=nb_preselected_rows,
            )

        # Subsample to create the buck fold (used for minimizer training + adversary training on *full* entries)
        if config.bucketization_percent is not None:
            X_train_buck, y_train_buck = self.subsample(
                X_train,
                y_train,
                size=config.bucketization_percent,
                important_prefix_size=nb_preselected_rows,
            )
        else:
            X_train_buck = X_train
            y_train_buck = y_train

        # Save original data
        self.X_train_buck_orig = X_train_buck
        self.y_train_buck_orig = y_train_buck
        self.X_train_orig = X_train
        self.X_val_orig = X_val
        self.X_test_orig = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        # Save scaled+onehot-encoded versions and corresponding loaders
        # NOTE: This also scales all self.X_*_orig datasets as a side effect (Nope)
        self.train_loader, self.X_train_oh = self.get_loader(
            X_train, y_train, scaler, oh_enc, batch_size, shuffle=True
        )
        self.buck_train_loader, self.X_train_buck_oh = self.get_loader(
            X_train_buck, y_train_buck, scaler, oh_enc, batch_size, shuffle=True
        )

        if config.adv_percent is not None and config.adv_percent < 1.0:
            adv_samples = int(config.adv_percent * len(X_train_buck))
            indices = np.random.choice(len(X_train_buck), adv_samples, replace=False)[
                :adv_samples
            ]
        else:
            indices = range(len(X_train_buck))

        self.adv_train_loader, self.X_train_adv_oh = self.get_loader(
            X_train_buck[indices],
            y_train_buck[indices],
            scaler,
            oh_enc,
            batch_size,
            shuffle=False,
        )

        self.val_loader, self.X_val_oh = self.get_loader(
            X_val, y_val, scaler, oh_enc, batch_size
        )
        self.test_loader, self.X_test_oh = self.get_loader(
            X_test, y_test, scaler, oh_enc, batch_size
        )
        self.tot_feats_oh = self.X_train_oh.shape[1]
        # print('Unique sums of 1hot vecs in test:')
        # print(np.unique(self.X_test_oh[:, 3:].sum(axis=1), return_counts=True))

        # Prepare feat data
        if feat_data is None:
            self.prepare_feat_data()
        else:
            self.feat_data = feat_data

        self.X_train_orig[:, self.cont_feats] = scaler.transform(
            self.X_train_orig[:, self.cont_feats]
        )
        self.X_train_buck_orig[:, self.cont_feats] = scaler.transform(
            self.X_train_buck_orig[:, self.cont_feats]
        )

        # Save scaler and oh_enc for shifts
        self.scaler = scaler
        self.oh_enc = oh_enc

    def get_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scaler: MinMaxScaler,
        oh_enc: OneHotEncoder,
        batch_size: int,
        shuffle: bool = False,
    ) -> Tuple[DataLoader, np.ndarray]:
        if len(self.cont_feats) > 0:
            x_cont_scaled = scaler.transform(X[:, self.cont_feats])
            X_oh = np.concatenate(
                [x_cont_scaled, oh_enc.transform(X[:, self.disc_feats])], axis=1
            )
        else:
            X_oh = oh_enc.transform(X[:, self.disc_feats])

        data = TensorDataset(torch.tensor(X_oh).float(), torch.tensor(y).long())
        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        return loader, X_oh

    def split(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        val_size: float = 0.1,
        test_size: float = 0.3,
        stratify: bool = True,
        preselect: bool = False,
        disc_feats: Optional[List[int]] = None,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        int,
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        # Split given dataset to create (train, val, test) folds -- bucketization data is chosen later
        if preselect:
            preselected_index_set = set()

            # Preselect: For each feature, ensure that each possible value is present (>=5 times) in the train fold
            # NOTE: will likely break for high-granularity real-valued features
            for i in range(features.shape[-1]):
                # Skip continuous features
                if disc_feats is not None and i not in disc_feats:
                    continue

                selected_indices = []
                unique_vals, ret_inv, ret_counts = np.unique(
                    features[:, i], return_inverse=True, return_counts=True
                )
                num_unique_vals = len(unique_vals)

                val_to_idxs: List[List[int]] = [[] for _ in range(num_unique_vals)]
                for idx, val in enumerate(ret_inv):
                    val_to_idxs[val].append(idx)

                for val in range(num_unique_vals):
                    selected_indices.extend(val_to_idxs[val][: min(5, ret_counts[val])])
                preselected_index_set.update(selected_indices)

            preselected_indices = list(preselected_index_set)
            nb_preselected_rows = len(preselected_indices)
            preselected_feats = features[preselected_indices]
            preselected_labels = labels[preselected_indices]
            non_selected_feats = np.delete(features, preselected_indices, axis=0)
            non_selected_labels = np.delete(labels, preselected_indices, axis=0)

            # Select the rest for the training set to achieve 1-val_size-test-size
            target_size = int((1 - val_size - test_size) * features.shape[0])
            assert (
                target_size >= nb_preselected_rows
            ), "Train fold too small to preselect"

            sample = np.random.permutation(target_size - nb_preselected_rows)
            sampled_feats = non_selected_feats[sample]
            sampled_labels = non_selected_labels[sample]

            X_train = np.concatenate((preselected_feats, sampled_feats))
            y_train = np.concatenate((preselected_labels, sampled_labels))

            # Sanity check that all values for each feature are there
            for i in range(features.shape[-1]):
                # Skip continuous features
                if disc_feats is not None and i not in disc_feats:
                    continue

                assert len(np.unique(features[:, i])) == len(
                    np.unique(X_train[:, i])
                ), f"Values missing for feat {i}"

            # Remove newly sampled examples and choose val and test set
            non_selected_feats = np.delete(non_selected_feats, sample, axis=0)
            non_selected_labels = np.delete(non_selected_labels, sample, axis=0)

            test_ratio = test_size / (val_size + test_size)
            X_val, X_test, y_val, y_test = train_test_split(
                non_selected_feats,
                non_selected_labels,
                test_size=test_ratio,
                random_state=42,
            )
        else:
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=42
            )

            rat = val_size / (1 - test_size)
            if stratify:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_trainval,
                    y_trainval,
                    test_size=rat,
                    stratify=y_trainval,
                    random_state=42,
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_trainval, y_trainval, test_size=rat, random_state=42
                )
            nb_preselected_rows = 0

        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_val, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_val, np.ndarray)
        assert isinstance(y_test, np.ndarray)

        return (X_train, y_train), nb_preselected_rows, (X_val, y_val), (X_test, y_test)

    def subsample(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        size: float,
        important_prefix_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Sample a subset of the given data with given size -- make sure that the important prefix is kept
        nb_old = features.shape[0]
        nb_new = int(size * nb_old)
        if important_prefix_size > 0:
            indices = np.arange(nb_new)
            assert (
                nb_new >= important_prefix_size
            ), "Can't keep important prefix when subsampling"
            # TODO: maybe we should take important ones + sample rest randomly?
        else:
            indices = np.random.permutation(nb_old)[:nb_new]
        return (features[indices, :], labels[indices])

    def prepare_feat_data(self) -> None:
        feat_data: List[
            Tuple[bool, str, int, int, Optional[int], Optional[np.ndarray]]
        ] = []
        beg: int = 0
        categories_iter: int = 0
        cats: Optional[int] = None
        unique_vals: Optional[np.ndarray] = None
        for feat_idx in range(self.tot_feats):
            if feat_idx in self.disc_feats:
                tot_vals = len(
                    np.unique(self.X_train_orig[:, feat_idx])
                )  # we want categories
                cats = self.categories[categories_iter]  # type: ignore # TODO should actually be List[np.ndarray]

                categories_iter += 1
                unique_vals = None
            elif feat_idx in self.cont_feats:
                tot_vals = 1
                cats = None
                unique_vals = np.unique(self.X_train_oh[:, beg])  # we want scaled
            else:
                assert False
            # print(f"feature {feat_idx}, tot vals is {tot_vals}")
            data = (
                feat_idx in self.disc_feats,
                self.feature_names[feat_idx],
                beg,
                beg + tot_vals,
                cats,
                unique_vals,
            )
            feat_data.append(data)
            beg += tot_vals
        assert categories_iter == len(self.categories)
        self.feat_data = feat_data

    def get_stats(self) -> str:
        stats = f"Size Orig: {self.X_train_orig.shape} Size for bucketization: {self.X_train_buck_orig.shape} Size val: {self.X_val_orig.shape} Size test: {self.X_test_orig.shape}"
        return stats
