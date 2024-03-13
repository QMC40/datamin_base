from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from datamin.bucketization import Bucketization
from datamin.dataset import FolktablesDataset


class DatasetMetaInformation:
    def __init__(
        self,
        sens_feats: List[int],
        feature_names: List[str],
        feat_data: List[
            Tuple[bool, str, int, int, Optional[int], Optional[np.ndarray]]
        ],
        sens_feat_data: Dict[int, Tuple[int, int]],
        buck_ignored_feats: List[int] = [],
    ) -> None:
        self.sens_feats = sens_feats  # list of sensitive features
        self.feature_names = feature_names  # list of feature names
        self.feat_data = (
            feat_data  # Overview all all features currently in the x of this dataset
        )
        self.sens_feat_data = sens_feat_data  # Ranges all sensitive features currently in the x of this dataset
        self.tot_feats = len(feat_data)  # Total number of features
        self.buck_ignored_feats = (
            buck_ignored_feats  # List of features that are ignored by the bucketization
        )
        self.og_feat_data = (
            feat_data.copy()
        )  # List of the original features in the original dataset -> Can be used to correctly slice from original x


def encode_loader(
    loader: DataLoader,
    bucketization: Bucketization,
    adv_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ignore_feats: Optional[List[int]] = None,
    requires_orig: bool = False,
    shuffle: bool = False,
    adv_use_label: bool = False,
) -> DataLoader:
    x, y = loader.dataset.tensors[0], loader.dataset.tensors[1]  # type: ignore[attr-defined]
    orig_y = y
    z = bucketization.transform(x, ignore_feats).detach()
    if adv_extractor is not None:
        y = adv_extractor(x)  # z -> sens_x (adv), otherwise z->y (clf)

    tensor_tuple = [z, y]
    if requires_orig:
        tensor_tuple.append(x)
    if adv_use_label:
        tensor_tuple.append(orig_y)

    loader = DataLoader(
        TensorDataset(*tensor_tuple), batch_size=loader.batch_size, shuffle=shuffle
    )

    return loader


def append_label_to_loader(loader: DataLoader) -> DataLoader:
    """(z,y,x,y_orig) -> (z*, y, x)"""
    assert len(loader.dataset.tensors) == 4  # type: ignore[attr-defined]
    return DataLoader(
        TensorDataset(
            torch.cat((loader.dataset.tensors[0], loader.dataset.tensors[3].view((-1, 1))), dim=1),  # type: ignore[attr-defined]
            loader.dataset.tensors[1],  # type: ignore[attr-defined]
            loader.dataset.tensors[2],  # type: ignore[attr-defined]
        ),
        batch_size=loader.batch_size,
        shuffle=False,
    )


def remove_orig_from_loader(loader: DataLoader) -> DataLoader:
    """(z,y,x) -> (z,y). Note that y is Sens(x) for adversary and y is y for classifier"""
    assert len(loader.dataset.tensors) == 3  # type: ignore[attr-defined]
    return DataLoader(
        TensorDataset(loader.dataset.tensors[0], loader.dataset.tensors[1]),  # type: ignore[attr-defined]
        batch_size=loader.batch_size,
        shuffle=False,
    )


def get_orig_train_loader(loader: DataLoader) -> DataLoader:
    """(z,y,x) -> (x,y). Note that y is Sens(x) for adversary and y is y for classifier"""
    assert len(loader.dataset.tensors) == 3  # type: ignore[attr-defined]
    return DataLoader(
        TensorDataset(loader.dataset.tensors[2], loader.dataset.tensors[1]),  # type: ignore[attr-defined]
        batch_size=loader.batch_size,
        shuffle=False,
    )


def get_reduced_dataset_metainformation(
    curr_sens_feats: List[int],
    curr_ignored_feats: List[int],
    dataset: FolktablesDataset,
    bucketization: Bucketization,
) -> DatasetMetaInformation:
    """Generates a dataset meta information object with the correct feature sizes when removing certain features

    Args:
        curr_sens_feats (List[int]): Features that are sensitive and should be predicted by the adversary
        curr_ignored_feats (List[int]): Features that should be ignored by the bucketization (i.e. full resolution)
        dataset (DatasetMetaInformation): The dataset to use
        bucketization (Bucketization): The bucketization to use

    Returns:
        DatasetMetaInformation: A dataset meta information object which can be used by the adversary to predict the relevant sensitive features
    """

    stub_dataset = DatasetMetaInformation(
        sens_feats=curr_sens_feats.copy(),
        feature_names=dataset.feature_names.copy(),
        feat_data=dataset.feat_data.copy(),
        sens_feat_data={},
        buck_ignored_feats=curr_ignored_feats.copy(),
    )

    it = 0
    for i, (is_feat_disc, feat_name, feat_beg, feat_end, _, _) in enumerate(
        dataset.feat_data
    ):

        if i in curr_sens_feats:
            stub_dataset.sens_feat_data[i] = (feat_beg, feat_end)

        if i in curr_ignored_feats:
            size = feat_end - feat_beg  # Use original size of the data
        else:
            # Requires the size from the bucketization
            buck_size = bucketization.buckets[feat_name]["size"]
            # size = dataset.feat_data[i][3] - dataset.feat_data[i][2]
            assert isinstance(buck_size, int)
            size = buck_size
        stub_dataset.feat_data[i] = (
            is_feat_disc,
            feat_name,
            it,
            it + size,
            None,
            None,
        )
        it += size

    return stub_dataset
