from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from datamin.adversaries.adversary import Adversary
from datamin.bucketization import MultiBucketization
from datamin.dataset import FolktablesDataset
from datamin.encoding_utils import (
    get_reduced_dataset_metainformation,
    remove_orig_from_loader,
)
from datamin.utils.logging_utils import CLogger, get_print_logger
from datamin.utils.utils import Stat, add_to_dict, itemize_dict


class MultiAdversary(Adversary):
    def __init__(
        self,
        device: str,
        model_name: str,
        dataset: FolktablesDataset,
        new_nb_fts: int,
        only_predict_seen: bool = False,
        logger: Optional[CLogger] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.dataset = dataset
        self.new_nb_fts = new_nb_fts
        self.trained = False
        if logger is None:
            logger = get_print_logger("MultiAdversary-Logger")
        self.logger = logger
        self.advs: List[Adversary] = []
        self.only_predict_seen = only_predict_seen
        self.masks: Dict[int, Dict[int, torch.Tensor]] = {}

    def set_bucketization(self, bucketization: MultiBucketization) -> None:
        self.bucketization = bucketization

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

        # Get the new dataset meta information which replaces selected sensitive feature with the non-bucketized version
        self.stub_datasets = []  #
        self.slices: List[Tuple[int, int]] = []
        ctr = 0
        for i, buck in enumerate(self.bucketization.bucketizations):
            stub_dataset = get_reduced_dataset_metainformation(
                curr_sens_feats=self.dataset.sens_feats,
                curr_ignored_feats=[],
                dataset=self.dataset,
                bucketization=buck,
            )
            self.stub_datasets.append(stub_dataset)

            size = buck.total_size()
            curr_start = ctr
            curr_end = ctr + size
            self.slices.append((curr_start, curr_end))
            ctr += size

            sliced_z = train_loader.dataset.tensors[0][:, curr_start:curr_end]  # type: ignore[attr-defined]
            y = train_loader.dataset.tensors[1]  # type: ignore[attr-defined]

            # Create new trainloader that only contains the features relevant to the current adversary
            interm_train_loader = DataLoader(
                TensorDataset(sliced_z, y, train_loader.dataset.tensors[2])  # type: ignore[attr-defined]
                if self.only_predict_seen
                else TensorDataset(sliced_z, y),
                batch_size=train_loader.batch_size,
                shuffle=True,
            )
            if val_loader is not None:
                sliced_z = val_loader.dataset.tensors[0][:, curr_start:curr_end]  # type: ignore[attr-defined]
                y = val_loader.dataset.tensors[1]  # type: ignore[attr-defined]
                interm_val_loader = DataLoader(
                    TensorDataset(sliced_z, y),
                    batch_size=val_loader.batch_size,
                    shuffle=True,
                )
            else:
                interm_val_loader = None

            # Train the adversary
            self.advs.append(
                Adversary(
                    device=self.device,
                    model_name=self.model_name,
                    dataset=stub_dataset,  # type: ignore[arg-type]
                    new_nb_fts=size,
                    only_predict_seen=self.only_predict_seen,
                    logger=self.logger,
                )
            )
            self.advs[i].set_bucketization(buck)
            self.advs[i].fit(
                interm_train_loader,
                nb_epochs,
                lr,
                tune_wd=tune_wd,
                weight_decay=weight_decay,
                val_loader=interm_val_loader,
            )

    def predict(
        self,
        z: torch.Tensor,
        sens_targets: torch.Tensor,
        all_pred_logits: Optional[Dict[int, List[torch.Tensor]]] = None,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], float]:
        loss: Dict[int, torch.Tensor] = {}
        acc: Dict[int, torch.Tensor] = {}
        for j, feat_idx in enumerate(self.dataset.sens_feats):
            losses_in_feat: List[torch.Tensor] = []
            preds_in_feat: List[torch.Tensor] = []
            for i, adv in enumerate(self.advs):
                cur_slice = self.slices[i]

                adv_pred_list = adv.forward(
                    z[:, cur_slice[0] : cur_slice[1]], feats=[feat_idx]
                )
                assert len(adv_pred_list) == 1
                adv_preds = adv_pred_list[0]
                # adv_preds: torch.Tensor = adv.advs[feat_idx](
                #    z[:, cur_slice[0] : cur_slice[1]]
                # )
                preds_in_feat.append(adv_preds)
                losses_in_feat.append(F.cross_entropy(adv_preds, sens_targets[:, j]))

            actual_pred = torch.stack(preds_in_feat, dim=0).mean(dim=0)
            is_acc = actual_pred.max(dim=1)[1].eq(sens_targets[:, j]).float()
            loss[feat_idx] = F.cross_entropy(actual_pred, sens_targets[:, j])
            acc[feat_idx] = is_acc.mean()

            if all_pred_logits is not None:
                all_pred_logits[feat_idx].append(actual_pred)

        return loss, acc, 0.0

    def score(
        self, loader: DataLoader, get_quantile_stats: bool = False
    ) -> Tuple[Dict[int, Stat], Dict[int, float], float]:
        # TODO make logits eval optional for training

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
                        mean_dict[b.item()] += quant_acc_dict[feat][b.item()]  # type: ignore[index]
                    mean_dict[b.item()] /= len(self.dataset.sens_feats)  # type: ignore[index]
            else:
                mean_dict = {}

            return tot_feat_acc, mean_dict, fully_correct
