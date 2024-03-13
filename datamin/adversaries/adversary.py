from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datamin.bucketization import Bucketization, MultiBucketization
from datamin.dataset import FolktablesDataset
from datamin.encoding_utils import (
    DatasetMetaInformation,
    append_label_to_loader,
    get_reduced_dataset_metainformation,
    remove_orig_from_loader,
)
from datamin.model_factory import get_model
from datamin.utils.logging_utils import CLogger, get_print_logger
from datamin.utils.utils import Stat, add_to_dict, itemize_dict


class Adversary:
    def __init__(
        self,
        device: str,
        model_name: str,
        dataset: FolktablesDataset,
        new_nb_fts: int,
        only_predict_seen: bool = False,
        use_label: bool = False,
        logger: Optional[CLogger] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.dataset = dataset
        self.new_nb_fts = new_nb_fts
        self.advs: Dict[int, nn.Module] = {}
        self.trained = False
        if logger is None:
            logger = get_print_logger("Adversary-Logger")
        self.logger = logger
        self._init_advs()
        self.only_predict_seen = only_predict_seen
        self.use_label = use_label
        self.masks: Dict[
            int, Dict[int, torch.Tensor]
        ] = {}  # Feature_idx -> [Bucket_idx -> Mask (which mapped to this bucket)]

    def _init_advs(self) -> None:
        self.advs = {}
        for i in self.dataset.sens_feats:
            _, _, feat_beg, feat_end, _, _ = self.dataset.feat_data[i]
            if hasattr(self.dataset, "sens_feat_data"):  # Used with A Datasetstub
                feat_beg, feat_end = self.dataset.sens_feat_data[i]  # type: ignore

            old_nb_buckets = feat_end - feat_beg
            self.advs[i] = get_model(
                self.model_name, self.device, self.new_nb_fts, old_nb_buckets
            )
            if old_nb_buckets == 1 and i in self.dataset.cont_feats:
                # If the feature is continuous and only has one bucket, we use a sigmoid
                self.advs[i] = nn.Sequential(self.advs[i], nn.Sigmoid()).to(self.device)

    def set_bucketization(self, bucketization: Bucketization) -> None:
        self.bucketization = bucketization

    def extract_targets(self, x: torch.Tensor) -> torch.Tensor:
        sens_targets = []
        for i in self.dataset.sens_feats:
            _, _, feat_beg, feat_end, _, _ = self.dataset.feat_data[i]
            if i in self.dataset.cont_feats:
                sens_targets.append(x[:, feat_beg:feat_end].flatten())
            else:
                sens_targets.append(x[:, feat_beg:feat_end].max(dim=1)[1])
        return torch.stack(sens_targets, dim=1)

    def _train(
        self, train_loader: DataLoader, nb_epochs: int, lr: float, weight_decay: float
    ) -> None:
        params = []
        for adv in self.advs.values():
            params += list(adv.parameters())
        opt = optim.Adam(list(params), lr=lr, weight_decay=weight_decay)
        lr_sched = StepLR(opt, step_size=nb_epochs // 2, gamma=0.1)

        for epoch in range(nb_epochs):
            tot_adv_feat_acc = {
                feat_idx: Stat() for feat_idx in self.dataset.sens_feats
            }
            tot_adv_feat_loss = {
                feat_idx: Stat() for feat_idx in self.dataset.sens_feats
            }
            for z, sens_targets in train_loader:
                z, sens_targets = z.to(self.device), sens_targets.to(self.device)
                opt.zero_grad()

                feat_loss, feat_acc, _ = self.predict(z, sens_targets)

                add_to_dict(tot_adv_feat_loss, itemize_dict(feat_loss))
                add_to_dict(tot_adv_feat_acc, itemize_dict(feat_acc))

                loss = torch.mean(torch.stack(list(feat_loss.values())))
                loss.backward()
                opt.step()
                lr_sched.step()
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
        if self.trained:
            raise RuntimeError(
                "Are you sure you want to call .fit() on a trained adversary?"
            )
        self.trained = True  # can call during training

        # Setup data specific meta information
        if self.only_predict_seen:

            ignored_feats = []
            if "buck_ignored_feats" in self.dataset.__dict__:
                ignored_feats = self.dataset.buck_ignored_feats  # type: ignore

            if isinstance(
                self.dataset, DatasetMetaInformation
            ):  # This implies that the adversary is called as a subroutine in a another adversary, hence we use the original feat_data for x ranges
                self.x_feat_ranges = [
                    (
                        self.dataset.og_feat_data[feat][2],
                        self.dataset.og_feat_data[feat][3],
                    )
                    for feat in range(self.dataset.tot_feats)
                ]
            else:
                self.x_feat_ranges = [
                    (self.dataset.feat_data[feat][2], self.dataset.feat_data[feat][3])
                    for feat in range(self.dataset.tot_feats)
                ]

            if isinstance(self.bucketization, MultiBucketization):
                self.z_feat_ranges = []
                for buck in self.bucketization.bucketizations:
                    stub_dataset = get_reduced_dataset_metainformation(
                        curr_sens_feats=self.dataset.sens_feats,
                        curr_ignored_feats=ignored_feats,
                        dataset=self.dataset,
                        bucketization=buck,
                    )
                    self.z_feat_ranges.append(
                        [
                            (
                                stub_dataset.feat_data[feat_idx][2],
                                stub_dataset.feat_data[feat_idx][3],
                            )
                            for feat_idx in range(self.dataset.tot_feats)
                        ]
                    )
            else:
                stub_dataset = get_reduced_dataset_metainformation(
                    curr_sens_feats=self.dataset.sens_feats,
                    curr_ignored_feats=ignored_feats,
                    dataset=self.dataset,
                    bucketization=self.bucketization,
                )
                self.z_feat_ranges = [
                    (  # type: ignore
                        stub_dataset.feat_data[feat_idx][2],
                        stub_dataset.feat_data[feat_idx][3],
                    )
                    for feat_idx in range(self.dataset.tot_feats)
                ]

            _, self.masks = self.bucketization.get_input_to_bucketization_stats(
                x=train_loader.dataset.tensors[2],  # type: ignore
                z=train_loader.dataset.tensors[0],  # type: ignore  # TODO Need z here as we might ignore feats (and the bucketization doesnt account for this)
                x_feat_ranges=self.x_feat_ranges,
                z_feat_ranges=self.z_feat_ranges,  # type: ignore
                sens_feats=self.dataset.sens_feats,
            )

        if self.use_label:
            train_loader = append_label_to_loader(train_loader)
            if val_loader is not None:
                val_loader = append_label_to_loader(val_loader)

        # Get rid of the original x values
        if len(train_loader.dataset.tensors) > 2:  # type: ignore
            train_loader = remove_orig_from_loader(train_loader)

        if val_loader is not None and len(val_loader.dataset.tensors) > 2:  # type: ignore
            val_loader = remove_orig_from_loader(val_loader)

        if tune_wd:
            results: List[Tuple[float, float, Dict[int, nn.Module]]] = []
            for wd in [
                1e-6,
                1e-5,
                1e-4,
                0.001,
                0.002,
                0.004,
                0.006,
                0.008,
                0.01,
                0.02,
                0.05,
            ]:
                self._init_advs()
                self._train(train_loader, nb_epochs, lr, wd)
                with torch.no_grad():
                    assert val_loader is not None
                    train_feat_acc, _, _ = self.score(train_loader)
                    train_acc = np.mean(
                        [train_feat_acc[i].avg() for i in self.dataset.sens_feats]
                    )
                    val_feat_acc, _, _ = self.score(val_loader)
                    val_acc: float = np.mean(
                        [val_feat_acc[i].avg() for i in self.dataset.sens_feats]  # type: ignore
                    )
                    self.logger.info(
                        f"[Trying ADV with wd={wd}] {train_acc:.3f} -> {val_acc:.3f}"
                    )
                results.append((val_acc, wd, self.advs))
            results = sorted(results)  # sort by validation
            weight_decay = results[-1][1]
            self.advs = results[-1][2]
            self.logger.info(f"Chose {weight_decay}")
        else:
            assert weight_decay is not None
            self._init_advs()
            self._train(train_loader, nb_epochs, lr, weight_decay)

        self.trained = True

    def predict(
        self,
        z: torch.Tensor,
        sens_targets: torch.Tensor,
        all_pred_logits: Optional[Dict[int, List[torch.Tensor]]] = None,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], float]:
        loss: Dict[int, torch.Tensor] = {}
        acc: Dict[int, torch.Tensor] = {}
        acc_list = []
        for j, feat_idx in enumerate(self.dataset.sens_feats):
            adv_preds = self._masked_predict(z, feat_idx)
            if feat_idx in self.dataset.disc_feats:
                loss[feat_idx] = F.cross_entropy(adv_preds, sens_targets[:, j].long())
                is_acc = adv_preds.max(dim=1)[1].eq(sens_targets[:, j]).float()
                if all_pred_logits is not None:
                    all_pred_logits[feat_idx].append(adv_preds)
            else:
                loss[feat_idx] = F.mse_loss(
                    adv_preds, sens_targets[:, j].view_as(adv_preds)
                )
                is_acc = (
                    torch.abs(adv_preds.flatten() - sens_targets[:, j]) < 0.05
                ).float()

            acc_list.append(is_acc)
            acc[feat_idx] = is_acc.mean()

        acc_matrix = torch.stack(acc_list)
        acc_matrix = acc_matrix.transpose(0, 1)
        total_corr = (acc_matrix.sum(dim=1) == acc_matrix.shape[1]).float().sum()
        return loss, acc, total_corr.item()  # dict keyed by feature

    def _masked_predict(self, z: torch.Tensor, feat_idx: int) -> torch.Tensor:

        adv_preds: torch.Tensor = self.advs[feat_idx](z)
        curr_lb_mask = torch.zeros_like(adv_preds)
        curr_mask = torch.ones_like(adv_preds)

        if len(self.masks) > 0:
            if isinstance(self.bucketization, MultiBucketization):
                offset = 0
                for i_o, masks in enumerate(
                    self.masks
                ):  # Masks is List[Dict[int, Dicht[int, Tensor]]]
                    feat_start, feat_end = self.z_feat_ranges[i_o][feat_idx]  # type: ignore
                    feat_start += offset
                    feat_end += offset
                    idxs = z[:, feat_start:feat_end].argmax(dim=1)
                    for i, idx in enumerate(idxs):
                        if int(idx.item()) in masks[feat_idx]:  # type: ignore
                            curr_mask[i] *= masks[feat_idx][int(idx.item())]  # type: ignore
                    offset += self.bucketization.bucketizations[i_o].total_size()
            else:
                feat_start, feat_end = self.z_feat_ranges[feat_idx]  # type: ignore
                idxs = z[:, feat_start:feat_end].argmax(dim=1)
                if feat_idx in self.dataset.disc_feats:
                    for i, idx in enumerate(idxs):
                        if int(idx.item()) in self.masks[feat_idx]:
                            curr_mask[i] = self.masks[feat_idx][int(idx.item())]
                else:
                    for i, idx in enumerate(idxs):
                        if int(idx.item()) in self.masks[feat_idx]:
                            curr_lb_mask[i] = self.masks[feat_idx][int(idx.item())][0]
                            curr_mask[i] = self.masks[feat_idx][int(idx.item())][1]

        if feat_idx in self.dataset.disc_feats:
            return adv_preds * curr_mask
        else:
            return torch.clamp(adv_preds, curr_lb_mask, curr_mask)

    def forward(
        self, z: torch.Tensor, feats: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        if not self.trained:
            raise RuntimeError("Can't call forward on not trained adv")
        with torch.no_grad():
            adv_pred_list: List[torch.Tensor] = []
            if feats is None:
                feats = self.dataset.sens_feats
            for j, feat_idx in enumerate(feats):
                adv_preds: torch.Tensor = self._masked_predict(z, feat_idx)
                adv_pred_list.append(adv_preds)
        return adv_pred_list

    def score(
        self, loader: DataLoader, get_quantile_stats: bool = False
    ) -> Tuple[Dict[int, Stat], Dict[int, float], float]:
        # TODO make logits eval optional for training

        if self.use_label and len(loader.dataset.tensors) == 4:
            loader = append_label_to_loader(loader)

        if len(loader.dataset.tensors) > 2:  # type: ignore
            loader = remove_orig_from_loader(loader)

        if not self.trained:
            raise RuntimeError("Cant call score on not trained adv")
        with torch.no_grad():

            tot_feat_acc = {feat_idx: Stat() for feat_idx in self.dataset.sens_feats}
            tot_feat_loss = {feat_idx: Stat() for feat_idx in self.dataset.sens_feats}
            tot_full_feat_loss: OrderedDict[int, List[torch.Tensor]] = OrderedDict()
            for feat in self.dataset.sens_feats:
                tot_full_feat_loss[feat] = []

            fully_correct = 0.0
            for z, sens_targets in loader:
                z, sens_targets = z.to(self.device), sens_targets.to(self.device)
                feat_loss, feat_acc, fc = self.predict(
                    z, sens_targets, tot_full_feat_loss
                )
                fully_correct += fc
                add_to_dict(tot_feat_loss, itemize_dict(feat_loss))
                add_to_dict(tot_feat_acc, itemize_dict(feat_acc))

            mean_dict: Dict[int, float] = {}
            if get_quantile_stats:
                bounds = 1 - torch.tensor(
                    [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1], device=self.device
                )
                quant_acc_dict: Dict[int, Dict[float, float]] = {
                    feat_idx: {} for feat_idx in self.dataset.sens_feats
                }
                mean_dict = {b.item(): 0 for b in bounds}  # type: ignore[assignment]
                for i, (feat, val) in enumerate(tot_full_feat_loss.items()):
                    if feat in self.dataset.cont_feats:
                        continue
                    logit_tensor = torch.cat(val, dim=0)
                    qs = torch.quantile(logit_tensor.max(dim=1).values, q=bounds)
                    for j, q in enumerate(qs):
                        mask = logit_tensor.max(dim=1).values >= q
                        # TODO shift sensitive feature here
                        quant_acc_dict[feat][bounds[j].item()] = (
                            logit_tensor[mask]
                            .cpu()
                            .max(dim=1)[1]
                            .eq(loader.dataset.tensors[1][mask, i])  # type: ignore[attr-defined]
                            .float()
                            .mean()
                        ).item()
                for b in bounds:
                    for feat in self.dataset.sens_feats:
                        if feat in self.dataset.cont_feats:
                            mean_dict[b.item()] += feat_acc[feat].item()  # type: ignore[index]
                        else:
                            mean_dict[b.item()] += quant_acc_dict[feat][b.item()]  # type: ignore[index]
                    mean_dict[b.item()] /= len(self.dataset.sens_feats)  # type: ignore[index]
            else:
                mean_dict = {}

            return tot_feat_acc, mean_dict, fully_correct

    def get_full_eval(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        logger: Optional[CLogger] = None,
    ) -> Tuple[float, float, float]:
        train_acc, train_dist, train_fully_corr = self.score(
            train_loader, get_quantile_stats=True
        )
        val_acc, val_dist, val_fully_corr = self.score(
            val_loader, get_quantile_stats=True
        )
        test_acc, test_dist, test_fully_cor = self.score(
            test_loader, get_quantile_stats=True
        )

        if logger is not None:
            logger.info("[adv test] adv accuracy per feature:")
            for i in self.dataset.sens_feats:
                feat_name = self.dataset.feat_data[i][1]
                acc_train, acc_val, acc_test = (
                    train_acc[i].avg(),
                    val_acc[i].avg(),
                    test_acc[i].avg(),
                )
                logger.info(
                    f"\tfeat={feat_name}: tr= {acc_train:.3f}, va= {acc_val:.3f}, te= {acc_test:.3f}"
                )

        adv_train_acc = np.mean(
            [train_acc[i].avg() for i in self.dataset.sens_feats]
        ).item()
        adv_val_acc = np.mean(
            [val_acc[i].avg() for i in self.dataset.sens_feats]
        ).item()
        adv_test_acc = np.mean(
            [test_acc[i].avg() for i in self.dataset.sens_feats]
        ).item()

        if logger is not None:
            logger.info(
                f"[adv] train_acc={adv_train_acc:.3f}, val_acc={adv_val_acc:.3f}, test_acc={adv_test_acc:.3f}"
            )
            for key, val in test_dist.items():
                logger.info(f"[ADV Quantile] {key:.2f}: {val:.3f}")
            logger.info(
                f"[ADV EXACT] train_acc={train_fully_corr}/{len(train_loader.dataset)}, val_acc={val_fully_corr}/{len(val_loader.dataset)}, test_acc={test_fully_cor}/{len(test_loader.dataset)}"  # type: ignore[attr-defined]
            )  # type: ignore

        return adv_train_acc, adv_val_acc, adv_test_acc

    @staticmethod
    def _bucket_hash(bucket: torch.Tensor) -> int:
        arr = bucket.long().cpu().numpy()
        ret = hash(str(arr))
        return ret

    @staticmethod
    def compute_upper_bound(
        sens_feats: List[int],
        feat_data: List[
            Tuple[bool, str, int, int, Optional[int], Optional[np.ndarray]]
        ],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        delta: float = 0.05,
        eps: float = 0.001,
    ) -> Dict[str, Tuple[float, float]]:
        # train+val -> train, test -> test
        z1, sens1 = train_loader.dataset.tensors  # type: ignore[attr-defined]
        z2, sens2 = val_loader.dataset.tensors  # type: ignore[attr-defined]

        z_train = torch.vstack([z1, z2])
        sens_targets_train = torch.cat([sens1, sens2])

        z_test, sens_targets_test = test_loader.dataset.tensors  # type: ignore[attr-defined]

        z_train, z_test = z_train.long(), z_test.long()

        # go
        print("computing guarantees...")
        unique_buckets_train = torch.unique(z_train, dim=0)
        delta_feat = delta / (sens_targets_train.shape[1]) - eps
        deltas: List[Dict[int, float]] = [
            {} for _ in range(sens_targets_train.shape[1])
        ]

        # compute deltas on train set
        for i in range(sens_targets_train.shape[1]):
            n_values = feat_data[sens_feats[i]][3] - feat_data[sens_feats[i]][2]
            cand_buckets = []
            for bucket_idx, bucket in tqdm(enumerate(unique_buckets_train)):
                bucket_flag = z_train.eq(bucket.reshape(1, -1)).all(dim=1)
                bucket_sens = sens_targets_train[bucket_flag, i]
                n_samples = bucket_sens.shape[0]

                p_hat = torch.zeros(n_values)
                for j in range(n_values):
                    p_hat[j] = torch.mean((bucket_sens == j).float())
                assert (torch.sum(p_hat) - 1.0) < 1e-6

                err_prob_1 = np.exp(-2 * n_samples * (p_hat - 1.0) ** 2).sum()
                if err_prob_1 < delta_feat / len(unique_buckets_train):
                    cand_buckets += [bucket]
            for bucket in cand_buckets:
                deltas[i][Adversary._bucket_hash(bucket)] = delta_feat / len(
                    cand_buckets
                )

        # compute guarantees
        res: Dict[str, Tuple[float, float]] = {}
        for split in ["train", "test"]:
            if split == "train":
                z, sens_targets = z_train, sens_targets_train
            else:
                z, sens_targets = z_test, sens_targets_test

            t_0 = np.sqrt(-np.log(eps) / (2 * z.shape[0]))
            print("t_0: ", t_0)
            ub_err_prob, ub_adv_acc = 0.0, 0.0
            unique_buckets = torch.unique(z, dim=0)
            for i in range(sens_targets.shape[1]):
                mul_err_prob_feat, ub_err_prob_feat, ub_adv_acc_feat = 1.0, 0.0, 0.0
                n_values = feat_data[sens_feats[i]][3] - feat_data[sens_feats[i]][2]

                if len(deltas[i].values()) == 0:
                    ub_err_prob += delta_feat + eps
                    ub_adv_acc += 1.0 / sens_targets.shape[1]
                    continue

                found_bucket = {}
                for bucket_idx, bucket in tqdm(enumerate(unique_buckets)):
                    bucket_flag = z.eq(bucket.reshape(1, -1)).all(dim=1)
                    bucket_sens = sens_targets[bucket_flag, i]
                    n_samples = bucket_sens.shape[0]

                    max_adv_acc, best_err_prob = 1.0, 0.0
                    hsh = Adversary._bucket_hash(bucket)
                    if hsh in deltas[i]:
                        delta_bucket = deltas[i][hsh]
                        found_bucket[hsh] = True

                        p_hat = torch.zeros(n_values)
                        for j in range(n_values):
                            p_hat[j] = torch.mean((bucket_sens == j).float())
                        assert (torch.sum(p_hat) - 1.0) < 1e-6

                        # cp_ub = 0.0
                        # alpha = delta_bucket/n_values
                        # for j in range(n_values):
                        #     k = torch.sum((bucket_sens == j))
                        #     cp_ub = np.maximum(cp_ub, proportion_confint(k, n_samples, alpha=2*alpha, method="beta")[1])

                        target_lo, target_hi = torch.tensor(p_hat.max()), torch.tensor(
                            1.0
                        )
                        for _ in range(40):
                            target_adv_acc = 0.5 * (target_lo + target_hi)
                            err_prob = np.exp(
                                -2 * n_samples * (p_hat - target_adv_acc) ** 2
                            ).sum()
                            if err_prob < delta_bucket:
                                target_hi = target_adv_acc
                                max_adv_acc = target_adv_acc.item()
                            else:
                                target_lo = target_adv_acc
                        best_err_prob = deltas[i][hsh]

                        # print(f'cp_ub={cp_ub}, hoeff_ub={max_adv_acc}')

                    ub_adv_acc_feat += (n_samples / z.shape[0]) * max_adv_acc
                    ub_err_prob_feat += best_err_prob
                    mul_err_prob_feat *= 1.0 - best_err_prob
                for hsh in deltas[i]:
                    if hsh not in found_bucket:
                        ub_err_prob_feat += deltas[i][hsh]
                        mul_err_prob_feat *= 1.0 - deltas[i][hsh]
                mul_err_prob_feat = 1.0 - mul_err_prob_feat

                # print(i, 'ub_adv_acc=%.5f, ub_err_prob=%.10f' % (ub_adv_acc_feat, ub_err_prob_feat))
                # print(i, 'ub_adv_acc=%.5f, mul_err_prob=%.10f' % (ub_adv_acc_feat, mul_err_prob_feat))

                ub_adv_acc_feat = np.clip(ub_adv_acc_feat + t_0, a_min=0.0, a_max=1.0)
                ub_err_prob += ub_err_prob_feat + eps
                ub_adv_acc += ub_adv_acc_feat / sens_targets.shape[1]
            res[split] = (ub_adv_acc, 1 - ub_err_prob)
        return res
