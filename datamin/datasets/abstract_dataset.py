from abc import ABC, abstractmethod
from os import makedirs, path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data

project_root = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))


class AbstractDataset(ABC, torch.utils.data.Dataset):
    @abstractmethod
    def __init__(self, name: str, split: str, p_test: float, p_val: float) -> None:
        if split not in ["train", "test", "validation"]:
            raise ValueError("Unknown dataset split")

        self.split = split
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = path.join(project_root, "data", name)
        self.p_test = p_test
        self.p_val = p_val

        self.X_train: torch.Tensor
        self.X_val: torch.Tensor
        self.X_test: torch.Tensor
        self.y_train: torch.Tensor
        self.y_val: torch.Tensor
        self.y_test: torch.Tensor
        self.protected_train: torch.Tensor
        self.protected_val: torch.Tensor
        self.protected_test: torch.Tensor

        makedirs(self.data_dir, exist_ok=True)

    def _discretize_continuous(
        self,
        features: torch.Tensor,
        continuous: List[int],
        categorical: Dict[str, List[int]],
        rem: List[str],
        all_bins: Optional[List[np.ndarray]] = None,
        k: int = 4,
        column_ids: Optional[Dict[str, int]] = None,
    ) -> Tuple[
        torch.Tensor, List[List[int]], List[np.ndarray], Optional[Dict[str, int]]
    ]:
        new_features = []
        new_categorical: List[List[int]] = []
        new_bins: List[np.ndarray] = []

        beg = 0
        new_column_ids: Optional[Dict[str, int]] = (
            dict() if column_ids is not None else None
        )

        for idx, i in enumerate(continuous):
            q = np.linspace(0, 1, k + 1)
            if all_bins is None:
                bins: np.ndarray = np.quantile(features[:, i].cpu().numpy(), q)
                bins[0], bins[-1] = bins[0] - 1e-2, bins[-1] + 1e-2
                new_bins += [bins]
            else:
                bins = all_bins[idx]

            if column_ids is not None:
                assert new_column_ids is not None
                for col, col_id in column_ids.items():
                    if col_id != i:
                        continue

                    col_to_remove = col

                    for j in range(k):
                        lb, ub = bins[j], bins[j + 1]
                        new_column_ids[f"{col}={lb}-{ub}"] = beg + j

                column_ids.pop(col_to_remove)

            disc = np.digitize(features[:, i].cpu().numpy(), bins)
            disc = np.clip(disc, 1, k)  # test data outside of training bins
            assert np.all(disc >= 1)
            assert np.all(disc <= k)
            one_hot_vals = torch.zeros(features.shape[0], k).to(features.device)
            one_hot_vals[np.arange(features.shape[0]), disc - 1] = 1.0
            new_features += [one_hot_vals]
            new_categorical += [[j for j in range(beg, beg + k)]]
            beg += k

        for col_name, col_ids in categorical.items():
            if col_name in rem:
                if column_ids is not None:
                    cols_to_remove = list(
                        filter(lambda c: c.startswith(col_name), column_ids.keys())
                    )
                    for col_to_remove in cols_to_remove:
                        column_ids.pop(col_to_remove)
                continue

            if column_ids is not None:
                assert new_column_ids is not None
                for col, col_id in column_ids.items():
                    if col.startswith(col_name):
                        new_column_ids[col] = beg + col_id - col_ids[0]
                cols_to_remove = list(
                    filter(lambda c: c.startswith(col_name), column_ids.keys())
                )
                for col_to_remove in cols_to_remove:
                    column_ids.pop(col_to_remove)

            new_features += [features[:, col_ids]]
            new_categorical += [[j for j in range(beg, beg + len(col_ids))]]
            beg += len(col_ids)
        new_features_tensor = torch.cat(new_features, dim=1)

        return new_features_tensor, new_categorical, new_bins, new_column_ids

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index], self.protected[index]

    def __len__(self) -> int:
        return self.labels.size()[0]

    def _normalize(self, columns: Optional[List[int]]) -> None:
        columns = columns if columns is not None else np.arange(self.X_train.shape[1])

        self.mean, self.std = (
            self.X_train.mean(dim=0)[columns],
            self.X_train.std(dim=0)[columns],
        )

        use_minmax = False

        if use_minmax:
            self.minn, self.maxx = (
                self.X_train.min(dim=0)[0][columns],
                self.X_train.max(dim=0)[0][columns],
            )

            self.X_train[:, columns] = (self.X_train[:, columns] - self.minn) / (
                self.maxx - self.minn
            )
            if self.X_val is not None:
                self.X_val[:, columns] = (self.X_val[:, columns] - self.minn) / (
                    self.maxx - self.minn
                )
            if self.X_test is not None:
                self.X_test[:, columns] = (self.X_test[:, columns] - self.minn) / (
                    self.maxx - self.minn
                )
        else:
            self.X_train[:, columns] = (self.X_train[:, columns] - self.mean) / self.std
            if self.X_val is not None and len(self.X_val) > 0:
                self.X_val[:, columns] = (self.X_val[:, columns] - self.mean) / self.std
            if self.X_test is not None and len(self.X_test) > 0:
                self.X_test[:, columns] = (
                    self.X_test[:, columns] - self.mean
                ) / self.std

    def _assign_split(self) -> None:
        if self.split == "train":
            self.features, self.labels, self.protected = (
                self.X_train,
                self.y_train,
                self.protected_train,
            )
        elif self.split == "test":
            self.features, self.labels, self.protected = (
                self.X_test,
                self.y_test,
                self.protected_test,
            )
        elif self.split == "validation":
            self.features, self.labels, self.protected = (
                self.X_val,
                self.y_val,
                self.protected_val,
            )

        self.features = self.features.float()
        self.labels = self.labels.float()
        self.protected = self.protected.long()

    def pos_weight(self, split: str) -> float:
        if split == "train":
            labels = self.y_train
        elif split == "train-val":
            labels = torch.cat((self.y_train, self.y_val))
        else:
            raise ValueError("Unknown split")

        positives: float = torch.sum(labels == 1).float().item()
        negatives: float = torch.sum(labels == 0).float().item()

        assert positives + negatives == labels.shape[0]

        return negatives / positives
