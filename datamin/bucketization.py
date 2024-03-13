import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import neptune.new as neptune
import numpy as np
import torch
from torch import Tensor

from datamin.utils.logging_utils import CLogger

from .dataset import FolktablesDataset


class Bucketization:

    # b is a dict, for each feature there's #buckets and mapping/borders
    # 'OCCP': {'size': 2, 'mapping': [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0]}
    # 'AGE': {'size': 3, 'borders': [0.3, 0.8]}
    # cont assumed to be in [0, 1]
    #

    def __init__(self, dataset: FolktablesDataset):
        self.dataset = dataset
        self.buckets: Dict[str, Dict[str, Union[int, List[int], List[float]]]] = {}

    # Calling several times is ok, it just replaces
    def add_disc(
        self, name: str, sz: int, mapping: Union[List[int], np.ndarray]
    ) -> None:
        if len(mapping) > 0:
            assert np.max(mapping) < sz
            assert np.min(mapping) >= 0
        self.buckets[name] = {"size": sz, "mapping": [int(x) for x in mapping]}

    # Calling several times is ok, it just replaces
    def add_cont(
        self, name: str, sz: int, borders: Union[List[float], np.ndarray]
    ) -> None:
        assert len(borders) + 1 == sz
        self.buckets[name] = {"size": sz, "borders": [float(x) for x in borders]}

    def total_size(self, red: Callable[[List[int]], int] = sum) -> int:
        return red([b["size"] for b in self.buckets.values()])  # type: ignore

    def json(self) -> str:
        return json.dumps(self.buckets)  # indent=4?

    def from_json_file(self, path: str) -> None:
        if path is not None:
            with open(path, "r") as f:
                self.buckets = json.load(f)

    def to_json_file(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.json())

    def transform(
        self, x: torch.Tensor, ignore_feats: Optional[List[int]] = None
    ) -> torch.Tensor:  # x -> z
        """Bucketizes x according to the buckets defined in self.buckets, and returns z

        Args:
            x (torch.Tensor): Input Tensor
            ignore_feats (Optional[List[int]], optional): The features that we explcitely dont want to bucketize but keep original. Defaults to None.

        Returns:
            torch.Tensor: (Partially) bucketized tensor z
        """

        bucketed_size = self.total_size()

        if ignore_feats is not None:

            for feat in ignore_feats:
                non_bucket_feat_size = (
                    self.dataset.feat_data[feat][3] - self.dataset.feat_data[feat][2]
                )
                bucket_feat_size = self.buckets[self.dataset.feat_data[feat][1]]["size"]
                assert isinstance(bucket_feat_size, int)
                bucketed_size = bucketed_size + non_bucket_feat_size - bucket_feat_size

        z = torch.zeros((x.shape[0], bucketed_size))
        it = 0

        for i, (is_feat_disc, feat_name, feat_beg, feat_end, _, _) in enumerate(
            self.dataset.feat_data
        ):
            buckets = self.buckets[feat_name]
            size = buckets["size"]
            assert isinstance(size, int)
            assert feat_name in self.buckets

            if ignore_feats is None or i not in ignore_feats:
                if is_feat_disc:
                    mapping = buckets["mapping"]
                    assert isinstance(mapping, list)
                    assert np.min(mapping) >= 0
                    cats = x[:, feat_beg:feat_end].max(dim=1)[1]
                    for j in range(feat_end - feat_beg):
                        bucket_flag = cats == j
                        z[bucket_flag, it + mapping[j]] = 1.0
                else:
                    borders = buckets["borders"]
                    assert isinstance(borders, list)
                    assert len(borders) == size - 1
                    if size == 1:
                        z[:, it] = 1.0
                    else:
                        for j in range(size):
                            if j == 0:
                                bucket_flag = x[:, i] < borders[0]
                            elif j == size - 1:
                                bucket_flag = x[:, i] >= borders[-1]
                            else:
                                bucket_flag = (x[:, i] >= borders[j - 1]) & (
                                    x[:, i] < borders[j]
                                )
                            z[bucket_flag, it + j] = 1.0
            else:
                # Get feat_size
                z[:, it : it + feat_end - feat_beg] = x[:, feat_beg:feat_end]
                size = feat_end - feat_beg
            it += size
        return z

    def get_indices(
        self, z: Tensor
    ) -> Tensor:  # Returns the indices for each bucket in the transformed sample
        indices = torch.zeros((z.shape[0], self.total_size()), dtype=torch.int16)
        it = 0
        for i, (is_feat_disc, feat_name, feat_beg, feat_end, _, _) in enumerate(
            self.dataset.feat_data
        ):
            buckets = self.buckets[feat_name]
            size = buckets["size"]
            assert isinstance(size, int)
            end = it + size
            # Everything is one hot encoded so we can simply argmax
            indices[:, i] = torch.argmax(z[:, it:end], dim=1)
            it = end
        return indices

    def get_stats_on_input(
        self,
        x: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        z: Optional[Tensor] = None,
    ) -> Dict[str, Any]:

        if z is None:
            assert x is not None
            z = self.transform(torch.tensor(x))

        # How frequented are the buckets
        usage_per_bucket = z.sum(dim=0)

        it = 0
        feat_buck_usage: Dict[str, List[float]] = {}
        for name, bucket in self.buckets.items():
            feat_buck_usage[name] = []
            size = bucket["size"]
            assert isinstance(size, int)
            end = it + size
            z_slice = z[:, it:end]
            z_slice_sum = z_slice.sum(dim=0)
            for i, count in enumerate(z_slice_sum):
                feat_buck_usage[name].append(count.item())
            it = end

        feat_buck_norm: Dict[str, List[float]] = {}
        for name, usage in feat_buck_usage.items():
            feat_buck_norm[name] = [x / sum(usage) for x in usage]

        # Number of differently bucketized inputs - Shrinkage
        unique_inputs, ret_indices, counts_per_input = torch.unique(
            z, dim=0, return_counts=True, sorted=True, return_inverse=True
        )

        single_inputs = [
            (val, y[idx] if y is not None else None, x[idx] if x is not None else None)
            for val, idx, count in zip(unique_inputs, ret_indices, counts_per_input)
            if count == 1
        ]

        unique_zip = sorted(zip(counts_per_input, unique_inputs), key=lambda x: x[0])

        num_bucket_inputs = unique_inputs.shape[0]
        # Used fraction of the bucket_grid
        used_bucket_fraction = num_bucket_inputs / self.total_size(red=np.prod)

        # Compute the mean-weighted GCP
        # For each unique point in the generalization compute the NCP and multiply with the counts
        curr_idx = 0
        ncps = np.zeros_like(counts_per_input, dtype=np.float64)
        ncp_per_feats = self.get_ncp_stats()
        for buck, val in self.buckets.items():
            size = val["size"]
            feat_ncps = ncp_per_feats[buck]
            assert isinstance(size, int)
            sl_left, sl_right = curr_idx, curr_idx + size
            sl_ui = unique_inputs[:, sl_left:sl_right]
            ncps += feat_ncps[sl_ui.argmax(axis=1)]
            curr_idx += size

        gcp = (ncps * counts_per_input.numpy()).sum()
        # Most unique bucketized inputs

        stats: Dict[str, Any] = {}
        stats["single_inputs"] = single_inputs
        stats["unique_inputs"] = unique_zip  # type: ignore
        stats["usage_per_bucket"] = usage_per_bucket
        stats["feat_buck_usage"] = feat_buck_usage
        stats["feat_buck_norm"] = feat_buck_norm
        stats["num_bucket_inputs"] = num_bucket_inputs
        stats["used_bucket_fraction"] = used_bucket_fraction
        stats["gcp"] = gcp
        stats["ncps"] = ncps
        return stats

    def get_linkage_statistics(
        self,
        x: Optional[Tensor],
        z: Optional[Tensor],
        relevant_feats_a: List[int],
        relevant_feats_b: List[int],
    ) -> Dict[Tuple[float, ...], Dict[Tuple[float, ...], float]]:

        if z is None:
            assert x is not None
            z = self.transform(torch.tensor(x))

        stats = self.get_stats_on_input(None, None, z=z)

        feat_a_res: Dict[Tuple[float, ...], Dict[Tuple[float, ...], float]] = {}
        feat_b_res: Dict[Tuple[float, ...], Dict[Tuple[float, ...], float]] = {}

        for count, unique_vec in stats["unique_inputs"]:

            feat_a = tuple(unique_vec[relevant_feats_a].tolist())
            feat_b = tuple(unique_vec[relevant_feats_b].tolist())

            if feat_a not in feat_a_res:
                feat_a_res[feat_a] = {}
            if feat_b not in feat_b_res:
                feat_b_res[feat_b] = {}

            if feat_b not in feat_a_res[feat_a]:
                feat_a_res[feat_a][feat_b] = 0.0
            if feat_a not in feat_b_res[feat_b]:
                feat_b_res[feat_b][feat_a] = 0.0

            feat_a_res[feat_a][feat_b] += count.item()
            feat_b_res[feat_b][feat_a] += count.item()

        return feat_a_res

    def get_input_to_bucketization_stats(
        self,
        x: Tensor,
        x_feat_ranges: List[Tuple[int, int]],
        z_feat_ranges: List[Tuple[int, int]],
        z: Optional[Tensor] = None,
        sens_feats: Optional[List[int]] = None,
    ) -> Tuple[
        Dict[int, Dict[Tuple[float, ...], Dict[Tuple[float, ...], float]]],
        Dict[int, Dict[int, torch.Tensor]],
    ]:

        if sens_feats is None:
            sens_feats = self.dataset.sens_feats

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        if z is None:
            z = self.transform(x)

        res: Dict[int, Dict[Tuple[float, ...], Dict[Tuple[float, ...], float]]] = {}

        sens_x_range = [x_feat_ranges[i] for i in sens_feats]
        sens_z_range = [z_feat_ranges[i] for i in sens_feats]

        for i, ((x_start, x_end), (z_start, z_end)) in enumerate(
            zip(sens_x_range, sens_z_range)
        ):
            sens_idx = sens_feats[i]
            res[sens_idx] = {}
            x_slice = x[:, x_start:x_end]
            z_slice = z[:, z_start:z_end]
            unique_z = torch.unique(z_slice, dim=0)
            for z_val in unique_z:
                res[sens_idx][tuple(z_val.tolist())] = {}
                # Select all input indices that have the bucketization z_val
                mask = (z_slice == z_val).all(dim=1)
                for x_unique in torch.unique(x_slice[mask], dim=0):
                    res[sens_idx][tuple(z_val.tolist())][tuple(x_unique.tolist())] = (
                        (x_slice[mask] == x_unique).all(dim=1).sum().item()
                    )

        res_mask: Dict[int, Dict[int, torch.Tensor]] = {}

        for key, val in res.items():
            res_mask[key] = {}
            if key in self.dataset.cont_feats:
                left_bound = 0.0
                feat_idx = self.dataset.feat_data[key][1]
                for buck_idx, right_bound in enumerate(self.buckets[feat_idx]["borders"] + [1.0]):  # type: ignore
                    res_mask[key][buck_idx] = (left_bound, right_bound)  # type: ignore
                    left_bound = right_bound
            else:
                for z_val, x_val in val.items():
                    z_index = int(torch.tensor(z_val).argmax().item())
                    res_mask[key][z_index] = torch.zeros((len(list(x_val.keys())[0])))
                    for x_unique, count in x_val.items():
                        res_mask[key][z_index] += torch.tensor(x_unique)
                    res_mask[key][z_index] = (res_mask[key][z_index] > 0).float()

        return res, res_mask

    def print_buckets(
        self,
        logger: CLogger,
        run: Optional[neptune.Run] = None,
    ) -> None:

        logger.info("================")
        logger.info("Bucketization:")
        for i, (
            is_feat_disc,
            feat_name,
            feat_beg,
            feat_end,
            _,
            unique_vals,
        ) in enumerate(self.dataset.feat_data):
            assert feat_name in self.buckets
            buckets = self.buckets[feat_name]
            if is_feat_disc:
                sz, mapping = buckets["size"], buckets["mapping"]
                logger.info(f"{feat_name} (disc): tot_buckets={sz}, mapping={mapping}")
            else:
                sz, borders = buckets["size"], buckets["borders"]
                assert isinstance(borders, list)
                assert isinstance(sz, int)
                rounded_borders = [round(x, 3) for x in borders]

                logger.info(
                    f"{feat_name} (cont): tot_buckets={sz}, borders={rounded_borders}"
                )
                train_map_to = []
                assert unique_vals is not None
                for val in unique_vals:
                    it = 0
                    while it < sz - 1 and val >= borders[it]:
                        it += 1
                    train_map_to.append(it)
                used_sz = len(np.unique(train_map_to))
                logger.info(
                    f"\t\tTrain set values would map to {used_sz} buckets: {train_map_to}"
                )

        logger.info(f"Used {self.total_size()} buckets in total")
        logger.info("================")
        if run is not None:
            run["nb_buckets_used"].log(self.total_size)  # TODO log at call site instead

    def get_ncp_stats(self) -> Dict[str, np.ndarray]:

        scores_per_feature: Dict[str, np.ndarray] = {}

        for buck, val in self.buckets.items():
            size = val["size"]
            assert isinstance(size, int)
            if "mapping" in val:
                # Disc. features
                scores_per_feature[buck] = np.zeros((size,))
                mapping = val["mapping"]
                assert isinstance(mapping, List)
                unique, count = np.unique(mapping, return_counts=True)
                for i, (uni, cnt) in enumerate(zip(unique, count)):
                    assert uni == i
                    scores_per_feature[buck][uni] = cnt / len(mapping)
            else:
                # Cont feats
                scores_per_feature[buck] = np.zeros((size + 1,))
                borders = val["borders"]
                assert isinstance(borders, List)
                full_borders = [0.0] + borders + [1.0]  # type:ignore
                for i in range(len(full_borders) - 1):
                    scores_per_feature[buck][i] = full_borders[i + 1] - full_borders[i]

        return scores_per_feature

    def _k_anonymity_stats(self, x: torch.Tensor) -> List[int]:
        z = self.transform(x)
        unique_z, count_z = torch.unique(z, dim=0, return_counts=True)
        count_z = sorted(list(count_z.int()))
        return count_z

    def _l_diversity_stats(self, x: torch.Tensor, y: torch.Tensor) -> List[int]:
        # TODO not yet directly l_diversity
        z = self.transform(x)
        unique_z, inverse_z, count_z = torch.unique(
            z, dim=0, return_counts=True, return_inverse=True
        )

        # Go through the dataset and count the number of unique values for each sensitive attribute
        # for each bucket
        l_count_norm = torch.zeros((len(unique_z)))
        for i in range(len(count_z)):
            # Bucket i
            # selected_x = x[inverse_z == i]
            selected_y = y[inverse_z == i]
            _, count_y = torch.unique(selected_y, dim=0, return_counts=True)
            p_qs = count_y / len(selected_y)
            l_count_norm[i] = (-(p_qs * torch.log(p_qs)).sum()).item()

        return l_count_norm

    def is_eps_equal(self, other: "Bucketization", eps: float = 0.05) -> bool:
        similarity = self.get_similarity(other)
        return similarity >= 1 - eps

    def get_similarity(
        self,
        other: "Bucketization",
    ) -> float:
        sim = 0
        for buck, val in self.buckets.items():

            if buck not in other.buckets:
                sim = 0
                break

            if "mapping" in val and "mapping" in other.buckets[buck]:
                # Disc. features
                curr_agg = self._mapping_similarity(
                    val["mapping"], other.buckets[buck]["mapping"]  # type:ignore
                )  # type:ignore
                assert curr_agg / len(val["mapping"]) <= 1  # type:ignore
                sim += curr_agg / len(val["mapping"])  # type:ignore
            elif "borders" in val and "borders" in other.buckets[buck]:
                range = np.linspace(0, 1, 1000)
                self_mapping = np.zeros_like(range)
                other_mapping = np.zeros_like(range)

                for i, elem in sorted(
                    enumerate(val["borders"] + [1.0]), reverse=True  # type:ignore
                ):  # type:ignore
                    self_mapping[range <= elem] = i
                for i, elem in sorted(
                    enumerate(other.buckets[buck]["borders"] + [1.0]),  # type:ignore
                    reverse=True,  # type:ignore
                ):  # type:ignore
                    other_mapping[range <= elem] = i

                curr_agg = self._mapping_similarity(
                    self_mapping.tolist(), other_mapping.tolist()
                )  # type:ignore
                assert curr_agg / 1000 <= 1
                sim += curr_agg / 1000  # type:ignore
            else:
                sim = 0
                break

        assert sim <= len(self.buckets)
        sim /= float(len(self.buckets))  # type:ignore

        return sim

    @staticmethod
    def _mapping_similarity(mapping1: List[int], mapping2: List[int]) -> float:
        assert len(mapping1) == len(mapping2)
        curr_agg = 0
        self_map: Dict[int, List[int]] = {}
        for i, elem in enumerate(mapping1):  # type:ignore
            if elem not in self_map:
                self_map[elem] = []
            self_map[elem].append(i)
        other_map: Dict[int, List[int]] = {}
        for i, elem in enumerate(mapping2):  # type:ignore
            if elem not in other_map:
                other_map[elem] = []
            other_map[elem].append(i)
        for i, self_buck in enumerate(mapping1):  # type:ignore
            other_buck = mapping2[i]  # type:ignore
            self_list = self_map[self_buck]
            other_list = other_map[other_buck]
            union_list = set(self_list + other_list)
            intersect_list = set(self_list).intersection(set(other_list))
            curr_agg += len(intersect_list) / len(union_list)  # type:ignore

        return curr_agg


class MultiBucketization(Bucketization):
    """Combines Multiple Bucketizations into a single one,
    offers data view both as joint k-length bucketization
    or as individual bucketizations"""

    def __init__(self, dataset: FolktablesDataset, bucketizations: List[Bucketization]):
        self.dataset = dataset
        self.bucketizations: List[Bucketization] = bucketizations

    def total_size(self, red: Callable[[List[int]], int] = sum) -> int:
        sums = []
        for bucketization in self.bucketizations:
            buck_size = bucketization.total_size(red)
            sums.append(buck_size)
        return red(sums)

    def individual_sizes(self) -> List[int]:
        return [buck.total_size() for buck in self.bucketizations]

    def json(self) -> str:
        return json.dumps(self.bucketizations)

    def from_json_file(self, path: str) -> None:
        if path is not None:
            with open(path, "r") as f:
                self.bucketizations = json.load(f)

    def to_json_file(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.json())

    def transform(
        self, x: torch.Tensor, ignore_feats: Optional[List[int]] = None
    ) -> torch.Tensor:  # x -> z
        z = []
        for bucketization in self.bucketizations:
            z.append(bucketization.transform(x, ignore_feats))
        return torch.cat(z, dim=1)

    def print_buckets(
        self,
        logger: CLogger,
        run: Optional[neptune.Run] = None,
    ) -> None:
        for bucketization in self.bucketizations:
            bucketization.print_buckets(logger, run)

    def get_indices(
        self, z: Tensor
    ) -> Tensor:  # Returns the indices for each bucket in the transformed sample
        raise NotImplementedError("Not implemented yet")

    def get_stats_on_input(
        self,
        x: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        z: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Not implemented yet")

    def get_linkage_statistics(
        self,
        x: Optional[Tensor],
        z: Optional[Tensor],
        relevant_feats_a: List[int],
        relevant_feats_b: List[int],
    ) -> Dict[Tuple[float, ...], Dict[Tuple[float, ...], float]]:

        raise NotImplementedError("Not implemented yet")

    def get_input_to_bucketization_stats(
        self,
        x: Tensor,
        x_feat_ranges: List[Tuple[int, int]],
        z_feat_ranges: List[Tuple[int, int]],
        z: Optional[Tensor] = None,
        sens_feats: Optional[List[int]] = None,
    ) -> Tuple[
        Dict[int, Dict[Tuple[float, ...], Dict[Tuple[float, ...], float]]],
        Dict[int, Dict[int, torch.Tensor]],
    ]:
        results: List[
            Dict[int, Dict[Tuple[float, ...], Dict[Tuple[float, ...], float]]]
        ] = []
        masks: List[Dict[int, Dict[int, torch.Tensor]]] = []

        # Here z_feat_ranges is actually List[List[Tuple[int,int]]], one for each buck
        assert len(z_feat_ranges) == len(self.bucketizations)

        for i, buck in enumerate(self.bucketizations):
            # TODO Note this now uses buck to bucketize, alternatively we could slice from the given z
            result, mask = buck.get_input_to_bucketization_stats(
                x, x_feat_ranges, z_feat_ranges[i], sens_feats=sens_feats  # type: ignore
            )
            results.append(result)
            masks.append(mask)

        return results, masks  # type: ignore

    def get_ncp_stats(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError("Not implemented yet")

    def is_eps_equal(
        self,
        other: "Bucketization",
        disc_thresh: float = 0.05,
        cont_thresh: float = 0.1,
    ) -> bool:
        raise NotImplementedError("Not implemented yet")


class ContBucketization(Bucketization):
    def __init__(self, dataset: FolktablesDataset):
        super().__init__(dataset)

    def add_cont(
        self, name: str, sz: int, borders: Union[List[float], np.ndarray]
    ) -> None:
        # We now interprete continuous features with values instead of a one-hot vector -> Size is 1
        # We keep 'num_buckets' in case it is needed
        assert len(borders) + 1 == sz
        self.buckets[name] = {
            "size": 1,
            "num_buckets": sz,
            "borders": [float(x) for x in borders],
        }

    def transform_with_mean(self, x: torch.Tensor) -> torch.Tensor:  # x -> z
        z = torch.zeros((x.shape[0], self.total_size()))
        it = 0

        for i, (is_feat_disc, feat_name, feat_beg, feat_end, _, _) in enumerate(
            self.dataset.feat_data
        ):
            assert feat_name in self.buckets
            buckets = self.buckets[feat_name]
            size = buckets["size"]
            assert isinstance(size, int)
            if is_feat_disc:
                mapping = buckets["mapping"]
                assert isinstance(mapping, list)
                assert np.min(mapping) >= 0
                cats = x[:, feat_beg:feat_end].max(dim=1)[1]
                for j in range(feat_end - feat_beg):
                    bucket_flag = cats == j
                    z[bucket_flag, it + mapping[j]] = 1.0
            else:
                borders = buckets["borders"]
                assert isinstance(borders, list)
                assert len(borders) == size - 1
                if size == 1:
                    z[:, it] = 1.0
                else:
                    for j in range(size):
                        if j == 0:
                            bucket_flag = x[:, i] < borders[0]
                            z[bucket_flag, it + j] = borders[0]
                        elif j == size - 1:
                            bucket_flag = x[:, i] >= borders[-1]
                            z[bucket_flag, it + j] = borders[-1]
                        else:
                            bucket_flag = (x[:, i] >= borders[j - 1]) & (
                                x[:, i] < borders[j]
                            )
                            z[bucket_flag, it + j] = (borders[j - 1] + borders[j]) / 2
            it += size
        return z

    def get_indices(
        self, z: Tensor
    ) -> Tensor:  # Returns the indices for each bucket in the transformed sample
        indices = torch.zeros((z.shape[0], self.total_size()), dtype=torch.int16)
        it = 0
        for i, (is_feat_disc, feat_name, feat_beg, feat_end, _, _) in enumerate(
            self.dataset.feat_data
        ):
            buckets = self.buckets[feat_name]
            size = buckets["size"]
            assert isinstance(size, int)
            end = it + size
            z_slice = z[:, it:end]
            if is_feat_disc:
                # Everything is one hot encoded so we can simply argmax
                indices[:, i] = torch.argmax(z_slice, dim=1)
            else:
                # Here we need to get the matrix to one_hot encode first
                borders = buckets["borders"]
                assert isinstance(borders, list)
                oh_encoding = torch.zeros((z.shape[0], size))
                assert len(borders) == size - 1
                if size == 1:
                    oh_encoding[:, :] = 1.0
                else:
                    for j in range(size):
                        if j == 0:
                            bucket_flag = z_slice < borders[0]
                        elif j == size - 1:
                            bucket_flag = z_slice >= borders[-1]
                        else:
                            bucket_flag = (z_slice >= borders[j - 1]) & (
                                z_slice < borders[j]
                            )
                        oh_encoding[bucket_flag, j] = 1
                indices[:, i] = torch.argmax(oh_encoding, dim=1)

        return indices
