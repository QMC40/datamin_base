import typing
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# from ortools.graph.python import min_cost_flow
from torch.utils.data import DataLoader
from tqdm import tqdm

from datamin.adversaries.adversary import Adversary
from datamin.bucketization import Bucketization
from datamin.dataset import FolktablesDataset
from datamin.encoding_utils import get_reduced_dataset_metainformation
from datamin.utils.config import LinkageAlgorithm
from datamin.utils.logging_utils import CLogger


class LinkageAdversary(Adversary):
    """Adversary that tries to link disjoint sets of features S_{a}, S_{b} from a datapoint X.
    As help the adversary has the full set of bucketized data to guide its decision-making.
    Versions: Bucketized -> Bucketized
              Bucketized -> Original # Recovery
                Original -> Bucketized # This could be done in feature stages to show how bucketization affects the adversary
                Original -> Original
    """

    def __init__(
        self,
        device: str,
        model_name: str,
        dataset: FolktablesDataset,
        new_nb_fts: int,
        linkage_alg: LinkageAlgorithm,
        a_feats: List[int],
        b_feats: List[int],
        logger: Optional[CLogger] = None,
    ):
        super().__init__(
            device=device,
            model_name=model_name,
            dataset=dataset,
            new_nb_fts=new_nb_fts,
            logger=logger,
        )
        self.trained: bool = False
        self.a_feats = a_feats
        self.b_feats = b_feats
        self.alg = linkage_alg
        self.n_queries = 1000  # TODO: make config

    def _train(
        self,
        train_loader: DataLoader,
        nb_epochs: int,
        lr: float,
        weight_decay: float,
        val_loader: Optional[DataLoader] = None,
    ) -> None:

        return

    def fit(
        self,
        train_loader: DataLoader,
        nb_epochs: int,
        lr: float,
        tune_wd: bool = False,
        weight_decay: Optional[float] = None,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        if self.trained:  # type: ignore[has-type]
            raise RuntimeError(
                "Are you sure you want to call .fit() on a trained adversary?"
            )
        self.trained = True  # can call during training

        assert isinstance(self.bucketization, Bucketization)

        stub_dataset = get_reduced_dataset_metainformation(
            curr_sens_feats=list(range(len(self.dataset.feat_data))),
            curr_ignored_feats=[],
            dataset=self.dataset,
            bucketization=self.bucketization,
        )

        translated_a_feats = [
            list(
                range(
                    stub_dataset.feat_data[feat_idx][2],
                    stub_dataset.feat_data[feat_idx][3],
                )
            )
            for feat_idx in self.a_feats
        ]
        translated_b_feats = [
            list(
                range(
                    stub_dataset.feat_data[feat_idx][2],
                    stub_dataset.feat_data[feat_idx][3],
                )
            )
            for feat_idx in self.b_feats
        ]
        orig_a_feats = [
            list(
                range(
                    self.dataset.feat_data[feat_idx][2],
                    self.dataset.feat_data[feat_idx][3],
                )
            )
            for feat_idx in self.a_feats
        ]
        orig_b_feats = [
            list(
                range(
                    self.dataset.feat_data[feat_idx][2],
                    self.dataset.feat_data[feat_idx][3],
                )
            )
            for feat_idx in self.b_feats
        ]

        self.translated_a_feats = [
            item for sublist in translated_a_feats for item in sublist
        ]
        self.translated_b_feats = [
            item for sublist in translated_b_feats for item in sublist
        ]
        self.orig_a_feats = [item for sublist in orig_a_feats for item in sublist]
        self.orig_b_feats = [item for sublist in orig_b_feats for item in sublist]

        self.statistics = self.bucketization.get_linkage_statistics(
            x=None,
            z=train_loader.dataset.tensors[0],  # type: ignore[has-type]
            relevant_feats_a=self.translated_a_feats,
            relevant_feats_b=self.translated_b_feats,
        )

        self.trained = True

    def predict(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        # Slice z
        z_a = z[:, self.translated_a_feats]
        z_b = z[:, self.translated_b_feats]

        # Goal: find a matching z_b for each z_a (tensor b_matched) based on self.statistics (obtained from z_train)
        # We use x only to evaluate the final mapping (as z_a and x are aligned)
        # (although the linkage adversary should in principle be allowed to use x_a and x_b)
        b_matched: Optional[List[int]] = None
        if self.alg == LinkageAlgorithm.MATCHING:
            assert False, "Matching adversary not implemented"
            # b_matched = self.predict_matching(z_a, z_b)
            # confidences = np.ones_like(
            #     b_matched
            # )  # TODO: implement this if we use matching adversary again
        elif self.alg == LinkageAlgorithm.SAMPLING:
            b_matched, confidences = self.predict_distributional(z_a, z_b, sample=True)
        elif self.alg == LinkageAlgorithm.MOSTLIKELY:
            b_matched, confidences = self.predict_distributional(z_a, z_b, sample=False)
        elif self.alg == LinkageAlgorithm.RANDOM:
            b_matched = np.arange(z_b.shape[0])
            assert b_matched is not None
            np.random.shuffle(b_matched)  # type: ignore
            confidences = np.ones_like(b_matched)

        assert b_matched is not None
        assert confidences is not None

        correct = np.zeros_like(b_matched)
        for i, linked_idx in enumerate(b_matched):
            correct[i] = (
                (
                    torch.isclose(
                        x[linked_idx, self.orig_b_feats], x[i, self.orig_b_feats]
                    )
                )
                .all()
                .item()
            )

        # Produce results for various quantiles
        results = {}
        percents = [1, 5, 10, 50, 100]
        for p in percents:
            mask = confidences >= np.quantile(confidences, q=1 - p / 100.0)
            results[p] = correct[mask].mean()

        # print(self.alg)
        # print(results)

        # TODO: change the api to return the whole dict properly + sync with how it's done for main adversary
        return results[100]

    def score(self, loader: DataLoader) -> Dict[str, float]:
        if not self.trained:
            raise RuntimeError("Cant call score on not trained adv")
        assert isinstance(self.bucketization, Bucketization)

        with torch.no_grad():

            z = loader.dataset.tensors[0]  # type: ignore
            x = loader.dataset.tensors[2]  # type: ignore

            # Subsample the given fold
            perm = torch.randperm(z.shape[0])[: self.n_queries]
            z, x = z[perm], x[perm]

            # shuffle x-z

            # Predict
            acc = self.predict(z, x)

            return {"acc": acc}

    def get_full_eval(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        logger: Optional[CLogger] = None,
    ) -> Tuple[float, float, float]:
        train_acc = self.score(train_loader)["acc"]
        val_acc = self.score(val_loader)["acc"]
        test_acc = self.score(test_loader)["acc"]

        if logger is not None:
            logger.info(
                f"[Linkage adv - {self.alg}] train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, test_acc={test_acc:.3f}"
            )

        return train_acc, val_acc, test_acc

    # flake8: noqa: C901
    # def predict_matching(self, z_a: torch.Tensor, z_b: torch.Tensor) -> List[int]:

    #     b_elems, b_rev_map, b_counts = torch.unique(
    #         z_b, return_counts=True, return_inverse=True, dim=0
    #     )
    #     b_elems = [tuple(elem.tolist()) for elem in b_elems]

    #     b_inv: Dict[float, List[int]] = {}
    #     for i, elem in enumerate(b_rev_map):
    #         if elem.item() in b_inv:
    #             b_inv[elem.item()].append(i)
    #         else:
    #             b_inv[elem.item()] = [i]

    #     # Via OR
    #     smcf = min_cost_flow.SimpleMinCostFlow()
    #     a_elems, a_counts = torch.unique(z_a, return_counts=True, dim=0)
    #     a_elems = [tuple(elem.tolist()) for elem in a_elems]

    #     # Num nodes Unique A + Unique B
    #     num_a = len(a_elems)
    #     num_b = len(b_elems)

    #     edges_start = []
    #     edges_end = []
    #     capacities = []
    #     unit_cost = []

    #     # Compute overall max
    #     max_cost = 1.0
    #     for feat_a in self.statistics:
    #         for feat_b in self.statistics[feat_a]:
    #             max_cost = max(self.statistics[feat_a][feat_b], max_cost)

    #     for i, a_tup in enumerate(a_elems):
    #         if a_tup in self.statistics:
    #             poss_b = self.statistics[a_tup]
    #             for b_tup, count in poss_b.items():
    #                 try:
    #                     b_ind = b_elems.index(b_tup)
    #                     edges_start.append(i)
    #                     edges_end.append(num_a + b_ind)
    #                     capacities.append(
    #                         min(a_counts[i].item(), b_counts[b_ind].item())
    #                     )
    #                     unit_cost.append(max_cost - count)
    #                 except Exception as e:
    #                     print(f"Couldn't match parts {e}")
    #         else:  # Connect to all possible in b
    #             for j, b_tup in enumerate(b_elems):
    #                 b_ind = b_elems.index(b_tup)
    #                 edges_start.append(i)
    #                 edges_end.append(num_a + b_ind)
    #                 capacities.append(a_counts[i].item())
    #                 unit_cost.append(max_cost)

    #     supply = [a_counts[i].item() for i in range(num_a)] + [
    #         -1 * b_counts[i].item() for i in range(num_b)
    #     ]

    #     all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
    #         edges_start, edges_end, capacities, unit_cost
    #     )
    #     smcf.set_nodes_supply(np.arange(0, len(supply)), supply)

    #     status = smcf.solve()

    #     arc_dict: Dict[int, Dict[int, Tuple[float, float]]] = {}
    #     a_ctr = [0] * num_a
    #     b_ctr = [0] * num_b

    #     if status != smcf.OPTIMAL:
    #         print("There was an issue with the min cost flow input.")
    #         print(f"Status: {status}")
    #         assert False
    #     # print(f'Minimum cost: {smcf.optimal_cost()}')
    #     # print('')
    #     # print(' Arc    Flow / Capacity Cost')
    #     solution_flows = smcf.flows(all_arcs)
    #     costs = solution_flows * unit_cost
    #     for arc, flow, cost in zip(all_arcs, solution_flows, costs):

    #         tail = smcf.tail(arc)
    #         head = smcf.head(arc)

    #         if flow > 0:
    #             if tail in arc_dict:
    #                 curr_max_flow = 0.0
    #                 for he, fl in arc_dict[tail].items():
    #                     curr_max_flow = max(curr_max_flow, fl[1])
    #                 arc_dict[tail][head] = (curr_max_flow, curr_max_flow + flow)
    #             else:
    #                 arc_dict[tail] = {head: (0, flow)}
    #         # print(
    #         #     f'{smcf.tail(arc):1} -> {smcf.head(arc)}  {flow:3}  / {smcf.capacity(arc):3}       {cost}'
    #         # )

    #     ret_idx = []

    #     for i, a_val in enumerate(z_a):
    #         a_tup = tuple(a_val.tolist())
    #         a_idx = a_elems.index(a_tup)
    #         ctr = a_ctr[a_idx]
    #         break_inner = False
    #         for b_idx, counts in arc_dict[a_idx].items():
    #             b_idx -= num_a  # Should start from 0
    #             if break_inner:
    #                 break
    #             if counts[0] <= ctr and ctr < counts[1]:
    #                 rev = b_inv[b_idx][b_ctr[b_idx]]
    #                 b_ctr[b_idx] += 1
    #                 a_ctr[a_idx] += 1

    #                 ret_idx.append(rev)

    #                 break_inner = True

    #     # Correctness checks
    #     assert z_a.shape[0] == torch.tensor(a_ctr).sum()
    #     assert z_b.shape[0] == torch.tensor(b_ctr).sum()
    #     assert z_b.shape[0] == len(ret_idx)

    #     return ret_idx

    # TODO: Add type hints
    @typing.no_type_check
    def predict_distributional(
        self, z_a: torch.Tensor, z_b: torch.Tensor, sample: bool
    ) -> Tuple[List[int], List[int]]:

        # Normalize for sampling
        # Normalize the feat_a_res and feat_b_res
        for feat_a in self.statistics:
            sum = 0.0
            for feat_b in self.statistics[feat_a]:
                sum += self.statistics[feat_a][feat_b]
            for feat_b in self.statistics[feat_a]:
                self.statistics[feat_a][feat_b] /= sum

        a_elems, a_rev_map = torch.unique(z_a, return_inverse=True, dim=0)
        a_elems = [tuple(elem.tolist()) for elem in a_elems]
        a_elems_set = set(a_elems)
        a_elems = np.asarray(a_elems)
        a_inv: Dict[float, List[int]] = {}
        for i, elem in enumerate(a_rev_map):
            if elem.item() in a_inv:
                a_inv[elem.item()].append(i)
            else:
                a_inv[elem.item()] = [i]

        b_elems, b_rev_map = torch.unique(z_b, return_inverse=True, dim=0)
        b_elems = [tuple(elem.tolist()) for elem in b_elems]
        b_elems_set = set(b_elems)
        b_elems = np.asarray(b_elems)
        b_inv: Dict[float, List[int]] = {}
        for i, elem in enumerate(b_rev_map):
            if elem.item() in b_inv:
                b_inv[elem.item()].append(i)
            else:
                b_inv[elem.item()] = [i]

        # Local stats that take into account which b values are present
        local_stats = dict()
        for a_tup, b_distribution in self.statistics.items():
            if a_tup not in a_elems_set:
                continue

            local_stats[a_tup] = dict()
            sum_probs = 0
            for b_tup, prob in b_distribution.items():
                if b_tup in b_elems_set:
                    local_stats[a_tup][b_tup] = prob
                    sum_probs += prob

            for b_tup, prob in local_stats[a_tup].items():
                local_stats[a_tup][b_tup] = prob / sum_probs

        # Pick a more efficient solution for many queries on highly-bucketized data
        # NOTE prelim results use the first impl, second is not fully tested
        # TODO fix duplication between implementations
        if self.n_queries <= 10000 or z_a.shape[0] < len(a_inv) * len(b_inv):
            linked_idxs = []
            confidences = []
            # Go through all of z_a
            for i, a_val in enumerate(tqdm(z_a)):
                a_tup = tuple(a_val.tolist())

                # Build choices and probs
                choices = []
                probs = []
                if a_tup in local_stats and len(local_stats[a_tup]) > 0:
                    poss_b = local_stats[a_tup]
                    choices, probs = zip(*list(poss_b.items()))
                else:  # Uknown a_val, so connect to all possible in b
                    choices = b_elems
                    probs = np.full(len(choices), 1 / len(choices))
                choices = np.asarray(choices)

                # Sample from the conditional distribution or choose the most likely value
                if sample:
                    idx = np.random.choice(len(choices), p=probs)
                else:
                    idx = np.asarray(probs).argmax()
                b_tup = choices[idx]
                conf = probs[idx]

                # From all z_b entries matching this b_tup pick a random one (you can't do better than that)
                try:
                    b_ind = np.where((b_elems == b_tup).all(axis=1))[0].item()
                    possible_indices = b_inv[b_ind]
                    conf *= 1 / len(possible_indices)
                    linked_idx = np.random.choice(possible_indices)
                except Exception as e:
                    self.logger.info(
                        f"Couldn't match parts {e} -> picking a random one"
                    )
                    linked_idx = np.random.choice(len(z_b))
                    conf = 1 / len(z_b)

                linked_idxs.append(linked_idx)
                confidences.append(conf)
        else:
            linked_idxs = np.zeros(z_a.shape[0], dtype=int)
            confidences = np.zeros(z_a.shape[0])

            # Go through all of z_a grouped by value
            for elem_idx, a_positions in a_inv.items():
                a_tup = tuple(a_elems[elem_idx].tolist())

                # Build choices and probs
                choices = []
                probs = []
                if a_tup in local_stats and len(local_stats[a_tup]) > 0:
                    poss_b = local_stats[a_tup]
                    choices, probs = zip(*list(poss_b.items()))
                else:  # Uknown a_val, so connect to all possible in b
                    choices = b_elems
                    probs = np.full(len(choices), 1 / len(choices))
                choices = np.asarray(choices)
                probs = np.asarray(probs)

                # Sample from the conditional distribution or always choose the most likely value
                if sample:
                    idxs = np.random.choice(
                        len(choices), p=probs, size=len(a_positions)
                    )
                else:
                    idxs = np.full(len(a_positions), np.asarray(probs).argmax())
                b_tups = choices[idxs]
                b_confs = probs[idxs]

                # From all z_b entries matching this b_tup pick a random one (you can't do better than that)
                try:
                    linked_idxs_curr = np.zeros(len(a_positions))
                    confidences_curr = np.zeros(len(a_positions))
                    # Go through possible values for b
                    for b_elem_idx, b_positions in b_inv.items():
                        mask = (b_tups == b_elems[b_elem_idx]).all(axis=1)
                        picked_idxs = np.random.choice(b_positions, size=mask.sum())
                        linked_idxs_curr[mask] = picked_idxs
                        confidences_curr[mask] = b_confs[mask] * (1 / len(b_positions))
                except Exception as e:
                    self.logger.info(
                        f"Couldn't match parts {e} -> picking a random one"
                    )
                    linked_idxs_curr = np.random.choice(len(z_b), size=len(a_positions))
                    confidences_curr = np.full(len(a_positions), 1 / len(z_b))

                linked_idxs[a_positions] = linked_idxs_curr
                confidences[a_positions] = confidences_curr
        return np.asarray(linked_idxs), np.asarray(confidences)
