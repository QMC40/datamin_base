from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from datamin.adversaries.adversary import Adversary
from datamin.bucketization import Bucketization
from datamin.dataset import FolktablesDataset
from datamin.utils.logging_utils import CLogger
from datamin.utils.utils import Stat, add_to_dict, itemize_dict


class OutlierAdversary(Adversary):
    """Adversary that tries to attack points that have rare bucketizations.
    Inspired by the singling out attack in https://arxiv.org/pdf/2211.10459.pdf.

    """

    def __init__(
        self,
        device: str,
        model_name: str,
        dataset: FolktablesDataset,
        new_nb_fts: int,
        logger: Optional[CLogger] = None,
    ):
        super().__init__(
            device=device,
            model_name=model_name,
            dataset=dataset,
            new_nb_fts=new_nb_fts,
            logger=logger,
        )
        self._init_advs()

    def _train(
        self, train_loader: DataLoader, nb_epochs: int, lr: float, weight_decay: float
    ) -> None:
        params = []
        for adv in self.advs.values():  # type: ignore[has-type]
            params += list(adv.parameters())
        opt = optim.Adam(list(params), lr=lr, weight_decay=weight_decay)
        lr_sched = StepLR(opt, step_size=nb_epochs // 2, gamma=0.1)

        # Adv. training here
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

                feat_loss, feat_acc = self.predict(z, sens_targets)

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
        if self.trained:  # type: ignore[has-type]
            raise RuntimeError(
                "Are you sure you want to call .fit() on a trained adversary?"
            )
        self.trained = True  # can call during training

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
                    train_feat_acc, train_single_feat_acc = self.score(train_loader)
                    train_acc = np.mean(
                        [train_feat_acc[i].avg() for i in self.dataset.sens_feats]
                    )
                    train_outlier_acc = np.mean(
                        [
                            train_single_feat_acc[i].avg()
                            for i in self.dataset.sens_feats
                        ]
                    )

                    val_feat_acc, val_single_feat_acc = self.score(val_loader)
                    val_acc: float = np.mean(
                        [val_feat_acc[i].avg() for i in self.dataset.sens_feats]
                    )
                    val_outlier_acc = np.mean(
                        [val_single_feat_acc[i].avg() for i in self.dataset.sens_feats]
                    )
                    self.logger.info(
                        f"[Trying ADV with wd={wd}] {train_acc:.3f} -> {val_acc:.3f} | Outlier: {train_outlier_acc:.3f} -> {val_outlier_acc:.3f}"
                    )
                results.append((val_acc, wd, self.advs))  # type: ignore[has-type]
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
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        loss: Dict[int, torch.Tensor] = {}
        acc: Dict[int, torch.Tensor] = {}
        for j, feat_idx in enumerate(self.dataset.sens_feats):
            adv_preds: torch.Tensor = self.advs[feat_idx](z)
            loss[feat_idx] = F.cross_entropy(adv_preds, sens_targets[:, j])
            if all_pred_logits is not None:
                all_pred_logits[feat_idx].append(adv_preds)
            acc[feat_idx] = (
                adv_preds.max(dim=1)[1].eq(sens_targets[:, j]).float().mean()
            )
        return loss, acc  # dict keyed by feature

    def score(
        self,
        loader: DataLoader,
    ) -> Tuple[Dict[int, Stat], Dict[int, Stat]]:

        if not self.trained:
            raise RuntimeError("Cant call score on not trained adv")
        assert isinstance(self.bucketization, Bucketization)

        with torch.no_grad():

            z: torch.Tensor = loader.dataset.tensors[0]  # type: ignore[has-type]
            x: torch.Tensor = loader.dataset.tensors[1]  # type: ignore[has-type]

            stats = self.bucketization.get_stats_on_input(x, None, z)

            single_inputs_and_counts = stats["single_inputs"]

            single_tot_feat_acc = {
                feat_idx: Stat() for feat_idx in self.dataset.sens_feats
            }
            single_tot_feat_loss = {
                feat_idx: Stat() for feat_idx in self.dataset.sens_feats
            }

            z = torch.stack([z for (z, _, _) in single_inputs_and_counts])
            sens_targets = torch.stack([s for (_, _, s) in single_inputs_and_counts])

            sing_feat_loss, sing_feat_acc = self.predict(z, sens_targets)
            add_to_dict(single_tot_feat_loss, itemize_dict(sing_feat_loss))
            add_to_dict(single_tot_feat_acc, itemize_dict(sing_feat_acc))

            tot_feat_acc = {feat_idx: Stat() for feat_idx in self.dataset.sens_feats}
            tot_feat_loss = {feat_idx: Stat() for feat_idx in self.dataset.sens_feats}

            for z, sens_targets in loader:
                z, sens_targets = z.to(self.device), sens_targets.to(self.device)
                feat_loss, feat_acc = self.predict(z, sens_targets)
                add_to_dict(tot_feat_loss, itemize_dict(feat_loss))
                add_to_dict(tot_feat_acc, itemize_dict(feat_acc))

            return tot_feat_acc, single_tot_feat_acc

    def get_full_eval(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        logger: Optional[CLogger] = None,
    ) -> Tuple[float, float, float]:
        train_acc, train_single_acc = self.score(train_loader)
        val_acc, val_single_acc = self.score(val_loader)
        test_acc, test_single_acc = self.score(test_loader)

        if logger is not None:
            logger.info("[OUTLIER adv test] adv accuracy per feature:")
            for i in self.dataset.sens_feats:
                feat_name = self.dataset.feat_data[i][1]
                acc_train, acc_val, acc_test = (
                    train_acc[i].avg(),
                    val_acc[i].avg(),
                    test_acc[i].avg(),
                )
                acc_single_train, acc_single_val, acc_single_test = (
                    train_single_acc[i].avg(),
                    val_single_acc[i].avg(),
                    test_single_acc[i].avg(),
                )
                logger.info(
                    f"\tfeat={feat_name}: tr={acc_train:.3f}, va={acc_val:.3f}, te={acc_test:.3f} | OUTLIER tr={acc_single_train:.3f}, va={acc_single_val:.3f}, te={acc_single_test:.3f}"
                )

        adv_train_acc = np.mean([train_acc[i].avg() for i in self.dataset.sens_feats])
        adv_val_acc = np.mean([val_acc[i].avg() for i in self.dataset.sens_feats])
        adv_test_acc = np.mean([test_acc[i].avg() for i in self.dataset.sens_feats])

        single_adv_train_acc = np.mean(
            [train_single_acc[i].avg() for i in self.dataset.sens_feats]
        )
        single_adv_val_acc = np.mean(
            [val_single_acc[i].avg() for i in self.dataset.sens_feats]
        )
        single_adv_test_acc = np.mean(
            [test_single_acc[i].avg() for i in self.dataset.sens_feats]
        )

        if logger is not None:
            logger.info(
                f"[OUTLIER adv] train_acc={adv_train_acc:.3f}, val_acc={adv_val_acc:.3f}, test_acc={adv_test_acc:.3f} | OUTLIER train_acc={single_adv_train_acc:.3f}, val_acc={single_adv_val_acc:.3f}, test_acc={single_adv_test_acc:.3f}"
            )

        return adv_train_acc, adv_val_acc, adv_test_acc
