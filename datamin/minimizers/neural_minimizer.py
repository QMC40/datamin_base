from typing import List, Tuple, Union

import torch

from datamin.bucketization import Bucketization
from datamin.dataset import FolktablesDataset

from .abstract_minimizer import AbstractMinimizer
from .neural_encoders.categorical_encoder import CategoricalEncoder
from .neural_encoders.continuous_encoder import ContinuousEncoder


class NeuralMinimizer(AbstractMinimizer):
    def __init__(self, device: str, max_buckets: int):
        super(NeuralMinimizer, self).__init__()
        self.device = device
        self.max_buckets = max_buckets
        self.dataset: FolktablesDataset

    def _prepare_encoders(
        self, dataset: FolktablesDataset, freeze_features: List[int]
    ) -> None:
        # Prepare encoders
        self.encoders: List[Union[CategoricalEncoder, ContinuousEncoder]] = []
        self.new_nb_fts: int = 0
        for i, (is_feat_disc, _, feat_beg, feat_end, _, _) in enumerate(
            dataset.feat_data
        ):
            if is_feat_disc:
                max_buckets = 1 if i in freeze_features else self.max_buckets  # CLIP
                self.encoders.append(
                    CategoricalEncoder(feat_end - feat_beg, max_buckets).to(self.device)
                )
            else:
                max_buckets = 1 if i in freeze_features else self.max_buckets
                self.encoders.append(
                    ContinuousEncoder(max_buckets, dataset.X_train_oh[:, i]).to(
                        self.device
                    )
                )
                # oh or orig it doesn't matter here (since first k are cont)
            self.new_nb_fts += max_buckets

    # neural minimizers bucketize directly (so we can backprop) instead of using the Bucketization class
    # no reason to call this from outside (you want postprocessed Bucketization)
    def _bucketize(
        self, temp: float, x: torch.Tensor, use_hard: bool = False
    ) -> torch.Tensor:
        z: List[torch.Tensor] = []
        for encoder, (_, _, feat_beg, feat_end, _, _) in zip(
            self.encoders, self.dataset.feat_data
        ):
            feat_z = encoder(x[:, feat_beg:feat_end], temp, use_hard)
            z += [feat_z]
        cat_z = torch.cat(z, dim=1)
        return cat_z

    def _compress(self, buckets: List[int]) -> Tuple[List[int], int]:
        d = {}
        new_buckets = []
        curr = 0
        for buck in buckets:
            if buck not in d:
                d[buck] = curr
                curr += 1
            new_buckets.append(d[buck])
        return new_buckets, curr

    def _to_bucketization(self, print_raw: bool = False) -> Bucketization:
        print(
            f"Transforming neural minimizer to a bucketization (max_buckets={self.max_buckets})"
        )
        if print_raw:
            print("## Also printing raw buckets")
        bucketization = Bucketization(self.dataset)

        print("Bucketization:")
        for encoder, (
            is_feat_disc,
            feat_name,
            feat_beg,
            feat_end,
            _,
            unique_vals,
        ) in zip(self.encoders, self.dataset.feat_data):
            if is_feat_disc:
                tot_vals = feat_end - feat_beg
                buckets: List[int] = []
                for j in range(tot_vals):
                    tmp_x = torch.zeros((1, tot_vals)).to(self.device)
                    tmp_x[0, j] = 1.0
                    with torch.no_grad():
                        buckets.append(int(torch.argmax(encoder(tmp_x, None)).item()))
                if print_raw:
                    print(f"{feat_name} (disc): {buckets}")

                buckets, sz = self._compress(buckets)

                bucketization.add_disc(feat_name, sz, buckets)
            else:
                buckets = []
                assert unique_vals is not None
                for val in unique_vals:
                    tmp_x = torch.tensor([[val]]).float().to(self.device)
                    with torch.no_grad():
                        buck = int(torch.argmax(encoder(tmp_x, None)).item())
                    buckets.append(buck)
                if print_raw:
                    print(f"{feat_name} (cont): {buckets}")

                buckets, sz = self._compress(buckets)

                # get borders, all buckets in [0, sz-1] will be present
                borders = []
                it = 0
                for curr in range(sz - 1):
                    # border between curr and curr+1
                    while buckets[it] == curr:
                        it += 1
                    # between this and last
                    borders.append((unique_vals[it] + unique_vals[it - 1]) / 2)

                bucketization.add_cont(feat_name, sz, borders)

        return bucketization
