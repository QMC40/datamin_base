from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from datamin.adversaries.adversary import Adversary
from datamin.bucketization import Bucketization
from datamin.dataset import FolktablesDataset
from datamin.encoding_utils import (
    encode_loader,
    get_reduced_dataset_metainformation,
    remove_orig_from_loader,
)
from datamin.utils.logging_utils import CLogger
from datamin.utils.utils import Stat, add_to_dict, itemize_dict


class IterativeAdversary(Adversary):
    """Adversary that tries to attack sensitive features given several sensitive features in higher resolution."""

    def __init__(
        self,
        device: str,
        model_name: str,
        dataset: FolktablesDataset,
        new_nb_fts: int,
        logger: Optional[CLogger] = None,
        only_predict_seen: bool = False,
    ):
        super().__init__(
            device=device,
            model_name=model_name,
            dataset=dataset,
            new_nb_fts=new_nb_fts,
            logger=logger,
        )
        self.only_predict_seen = only_predict_seen
        self.proxy_adv = Adversary(
            device=device,
            model_name=model_name,
            dataset=dataset,
            new_nb_fts=new_nb_fts,
            logger=logger,
            only_predict_seen=only_predict_seen,
        )
        # TODO Should save whether we want to use our own predictions or the original dataset
        # self.use_own_preds = use_own_preds
        self.advs: Dict[Tuple[int, ...], Adversary] = {}
        self.sens_feats = dataset.sens_feats.copy()
        self.tune_wd = False
        # Bucketization is set via set_bucketization of super class
        self.use_own_preds = False

    def _train(
        self,
        train_loader: DataLoader,
        nb_epochs: int,
        lr: float,
        weight_decay: float,
        val_loader: Optional[DataLoader] = None,
    ) -> None:

        # Here we first train a default adversary to sort sensitive features by their ease of recovery
        train_acc, _, _ = self.proxy_adv.score(train_loader)

        sorted_sensitive_feats = sorted(
            [(idx, acc.avg()) for idx, acc in train_acc.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        print(f"Sorted sensitive features: {sorted_sensitive_feats}")

        # Then we train a new adversary for each sensitive feature in order of importance
        adv_stack: List[List[int]] = [
            [i for i in range(self.dataset.tot_feats) if i not in self.sens_feats]
        ]  # Non-sensitive features

        for j, (feat_idx, _) in enumerate(sorted_sensitive_feats):

            curr_ignored_feats = adv_stack[-1]

            # We actually only need to predict the next sensitive feature
            curr_sens_target_feat = [feat_idx]  # [sorted_sensitive_feats[j + 1][0]]

            if len(adv_stack) == 0:
                adv_stack.append([feat_idx])
            else:
                adv_stack.append(adv_stack[-1] + [feat_idx])

            # Get the new dataset meta information which replaces selected sensitive feature with the non-bucketized version
            stub_dataset = get_reduced_dataset_metainformation(
                curr_sens_feats=curr_sens_target_feat,
                curr_ignored_feats=curr_ignored_feats,
                dataset=self.dataset,
                bucketization=self.bucketization,
            )

            # Get the new dataloader which replaces selected sensitive feature with the non-bucketized version
            new_train_loader = encode_loader(
                self.dataset.train_loader,
                self.bucketization,
                adv_extractor=self.get_sens_feat_extractor(curr_sens_target_feat),
                ignore_feats=curr_ignored_feats,
                requires_orig=self.only_predict_seen,
            )

            new_val_loader = encode_loader(
                self.dataset.val_loader,
                self.bucketization,
                adv_extractor=self.get_sens_feat_extractor(curr_sens_target_feat),
                ignore_feats=curr_ignored_feats,
                requires_orig=self.only_predict_seen,
            )

            new_nb_feats = new_train_loader.dataset.tensors[0].shape[1]  # type: ignore[attr-defined]

            self.advs[tuple(curr_ignored_feats)] = Adversary(
                device=self.device,
                model_name=self.model_name,
                dataset=stub_dataset,  # type: ignore[arg-type]
                new_nb_fts=new_nb_feats,
                only_predict_seen=self.only_predict_seen,
                logger=self.logger,
            )
            self.advs[tuple(curr_ignored_feats)].set_bucketization(self.bucketization)
            self.advs[tuple(curr_ignored_feats)].fit(
                new_train_loader,
                nb_epochs,
                lr,
                tune_wd=self.tune_wd,
                weight_decay=weight_decay,
                val_loader=new_val_loader,
            )
            curr_train_acc, _, _ = self.advs[tuple(curr_ignored_feats)].score(
                new_train_loader
            )
            val_train_acc, _, _ = self.advs[tuple(curr_ignored_feats)].score(
                new_val_loader
            )

            print(f"====== Adv trained with {curr_ignored_feats} ======")
            for feat_idx, acc in curr_train_acc.items():
                print(
                    f"Feat {feat_idx} - Train: {acc.avg():.3f} - Val: {val_train_acc[feat_idx].avg():.3f}"
                )

        self.feature_order = [x[0] for x in sorted_sensitive_feats]  # adv_stack[-1]

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
        self.tune_wd = tune_wd

        train_loader_no_x = remove_orig_from_loader(train_loader)
        assert val_loader is not None
        val_loader_no_x = remove_orig_from_loader(val_loader)

        # Fit the proxy adversary
        self.proxy_adv.set_bucketization(self.bucketization)
        self.proxy_adv.fit(
            train_loader,
            nb_epochs,
            lr,
            tune_wd=tune_wd,
            weight_decay=weight_decay,
            val_loader=val_loader_no_x,
        )

        self.advs[tuple([])] = self.proxy_adv

        results: List[Tuple[float, float, Dict[Tuple[int, ...], nn.Module]]] = []
        for wd in [
            0.0001,
        ]:
            self._train(train_loader_no_x, nb_epochs, lr, wd, val_loader=val_loader)
            with torch.no_grad():
                assert val_loader is not None
                train_feat_acc = self.score(train_loader)
                train_acc = np.mean(
                    [train_feat_acc[i].avg() for i in self.dataset.sens_feats]
                )

                val_feat_acc = self.score(val_loader)
                val_acc: float = np.mean(
                    [val_feat_acc[i].avg() for i in self.dataset.sens_feats]
                )
                self.logger.info(
                    f"[Iterative ADV with wd={wd}] {train_acc:.3f} -> {val_acc:.3f}"
                )
            results.append((val_acc, wd, self.advs))  # type: ignore[has-type]
        results = sorted(results)  # sort by validation

        self.trained = True

    def predict(
        self,
        z: torch.Tensor,
        sens_targets: torch.Tensor,
        orig_x: torch.Tensor,
        all_pred_logits: Optional[Dict[int, List[torch.Tensor]]] = None,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:

        # We iterate over all sensitive features in the order that we trained in
        assert self.feature_order is not None
        curr_order = self.feature_order

        loss: Dict[int, torch.Tensor] = {}
        acc: Dict[int, torch.Tensor] = {}

        curr_sens_feats = self.dataset.sens_feats.copy()
        curr_ignored_feats: List[int] = [
            i for i in range(self.dataset.tot_feats) if i not in curr_sens_feats
        ]  # Non-sensitive features

        for j, feat_idx in enumerate(curr_order):

            # Get the correct data
            proxy_loader = encode_loader(
                DataLoader(
                    TensorDataset(
                        orig_x, sens_targets
                    ),  # y doesnt matter as it gets overriden by the encoder
                    256,
                    shuffle=False,
                ),
                self.bucketization,
                adv_extractor=self.get_sens_feat_extractor(curr_sens_feats),
                ignore_feats=curr_ignored_feats,
            )
            curr_z = proxy_loader.dataset.tensors[0]  # type: ignore[attr-defined]
            # curr_sens = proxy_loader.dataset.tensors[1]  # type: ignore[attr-defined]

            # Predict with the correct adversary
            adv_pred_list = self.advs[tuple(curr_ignored_feats)].forward(
                curr_z, feats=[feat_idx]
            )
            assert len(adv_pred_list) == 1
            adv_preds = adv_pred_list[0]
            # adv_preds: torch.Tensor = self.advs[tuple(curr_ignored_feats)].advs[
            #     feat_idx
            # ](curr_z)

            # Feature handled
            curr_ignored_feats.append(feat_idx)
            curr_sens_feats.remove(feat_idx)

            sens_target_idx = self.dataset.sens_feats.index(
                feat_idx
            )  # Assumes that the sensitive features are sorted

            loss[feat_idx] = F.cross_entropy(
                adv_preds, sens_targets[:, sens_target_idx]
            )
            if all_pred_logits is not None:
                all_pred_logits[feat_idx].append(adv_preds)
            acc[feat_idx] = (
                adv_preds.max(dim=1)[1]
                .eq(sens_targets[:, sens_target_idx])
                .float()
                .mean()
            )

        return loss, acc  # dict keyed by feature

    def score(
        self,
        loader: DataLoader,
    ) -> Dict[int, Stat]:

        if not self.trained:
            raise RuntimeError("Cant call score on not trained adv")
        assert isinstance(self.bucketization, Bucketization)

        with torch.no_grad():

            tot_feat_acc = {feat_idx: Stat() for feat_idx in self.dataset.sens_feats}
            tot_feat_loss = {feat_idx: Stat() for feat_idx in self.dataset.sens_feats}

            for z, sens_targets, x in loader:
                z, sens_targets, x = (
                    z.to(self.device),
                    sens_targets.to(self.device),
                    x.to(self.device),
                )
                feat_loss, feat_acc = self.predict(z, sens_targets, x)
                add_to_dict(tot_feat_loss, itemize_dict(feat_loss))
                add_to_dict(tot_feat_acc, itemize_dict(feat_acc))

            return tot_feat_acc

    def get_full_eval(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        logger: Optional[CLogger] = None,
    ) -> Tuple[float, float, float]:
        train_acc = self.score(train_loader)
        val_acc = self.score(val_loader)
        test_acc = self.score(test_loader)

        # TODO Could/Should evaluate over time

        if logger is not None:
            logger.info("[Iterative adv] adv accuracy per feature:")
            for i in self.dataset.sens_feats:
                feat_name = self.dataset.feat_data[i][1]
                acc_train, acc_val, acc_test = (
                    train_acc[i].avg(),
                    val_acc[i].avg(),
                    test_acc[i].avg(),
                )
                logger.info(
                    f"\tfeat={feat_name}: tr={acc_train:.3f}, va={acc_val:.3f}, te={acc_test:.3f}"
                )

        adv_train_acc = np.mean([train_acc[i].avg() for i in self.dataset.sens_feats])
        adv_val_acc = np.mean([val_acc[i].avg() for i in self.dataset.sens_feats])
        adv_test_acc = np.mean([test_acc[i].avg() for i in self.dataset.sens_feats])

        if logger is not None:
            logger.info(
                f"[Iterative adv] train_acc={adv_train_acc:.3f}, val_acc={adv_val_acc:.3f}, test_acc={adv_test_acc:.3f}"
            )

        return adv_train_acc, adv_val_acc, adv_test_acc

    def get_sens_feat_extractor(
        self, sens_feats: List[int]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def sens_feat_extractor(x: torch.Tensor) -> torch.Tensor:
            sens_targets = []
            for i in sens_feats:
                _, _, feat_beg, feat_end, _, _ = self.dataset.feat_data[i]
                sens_targets.append(x[:, feat_beg:feat_end].max(dim=1)[1])
            return torch.stack(sens_targets, dim=1)

        return sens_feat_extractor
