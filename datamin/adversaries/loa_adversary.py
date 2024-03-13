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
    get_orig_train_loader,
    get_reduced_dataset_metainformation,
)
from datamin.utils.logging_utils import CLogger
from datamin.utils.utils import Stat, add_to_dict, itemize_dict


class LeaveOutAdversary(Adversary):
    """Adversary that tries to attack sensitive features given other sensitive features in higher resolution.
    By default it uses all but the currently attacked feature in high-resolution.
    When setting, only_nonsensitive=True, it will only use non-sensitive features in high-resolution.
    """

    def __init__(
        self,
        device: str,
        model_name: str,
        dataset: FolktablesDataset,
        new_nb_fts: int,
        only_nonsensitive: bool = False,
        only_predict_seen: bool = False,
        logger: Optional[CLogger] = None,
    ):
        super().__init__(
            device=device,
            model_name=model_name,
            dataset=dataset,
            new_nb_fts=new_nb_fts,
            logger=logger,
        )
        # self._init_advs()
        # TODO Should save whether we want to use our own predictions or the original dataset
        # self.use_own_preds = use_own_preds
        self.advs: Dict[int, Adversary] = {}
        self.sens_feats = dataset.sens_feats.copy()
        self.tune_wd = False
        self.only_nonsensitive = only_nonsensitive
        self.only_predict_seen = only_predict_seen
        if self.only_nonsensitive:
            self.tag = "LOA-NonSensitive"
        else:
            self.tag = "LOA"
        # Bucketization is set via set_bucketization of super class

    def _train(
        self,
        train_loader: DataLoader,
        nb_epochs: int,
        lr: float,
        weight_decay: float,
        val_loader: Optional[DataLoader] = None,
    ) -> None:

        for feat_idx in self.sens_feats:

            # The features that the adversary will predict on - Only the feature that is currently being attacked
            curr_sens_feats = [feat_idx]

            # Features that will not be bucketized for this prediction i.e. all features except the current sensitive feature
            if self.only_nonsensitive:
                curr_ignored_feats = list(range(self.dataset.tot_feats))
                for i_feat_idx in self.sens_feats:
                    curr_ignored_feats.remove(
                        i_feat_idx
                    )  # Only ignore non-sensitive features during bucketization
            else:
                curr_ignored_feats = list(range(self.dataset.tot_feats))
                curr_ignored_feats.remove(feat_idx)

            # Get the new dataset meta information which replaces selected sensitive feature with the non-bucketized version
            stub_dataset = get_reduced_dataset_metainformation(
                curr_sens_feats=curr_sens_feats,
                curr_ignored_feats=curr_ignored_feats,
                dataset=self.dataset,
                bucketization=self.bucketization,
            )

            # Get the new dataloader which replaces selected sensitive feature with the non-bucketized version
            new_train_loader = encode_loader(
                train_loader,
                self.bucketization,
                adv_extractor=self.get_sens_feat_extractor(curr_sens_feats),
                ignore_feats=curr_ignored_feats,
                requires_orig=self.only_predict_seen,
            )

            new_val_loader = encode_loader(
                self.dataset.val_loader,
                self.bucketization,
                adv_extractor=self.get_sens_feat_extractor(curr_sens_feats),
                ignore_feats=curr_ignored_feats,
            )

            new_nb_feats = new_train_loader.dataset.tensors[0].shape[1]  # type: ignore[attr-defined]

            # Here stub_dataset and only_predict seen are not in sinc - z is not bucketized (as it shouldn't be) but via the only predict seen the adversary gets the wrong information from the bucketization -> TODO Use ignore feats in Adversary stub_dataset

            self.advs[feat_idx] = Adversary(
                device=self.device,
                model_name=self.model_name,
                dataset=stub_dataset,  # type: ignore[arg-type]
                new_nb_fts=new_nb_feats,
                only_predict_seen=self.only_predict_seen,
                logger=self.logger,
            )
            self.advs[feat_idx].set_bucketization(self.bucketization)
            self.advs[feat_idx].fit(
                new_train_loader,
                nb_epochs,
                lr,
                tune_wd=self.tune_wd,
                weight_decay=weight_decay,
                val_loader=new_val_loader,
            )
            curr_train_acc, _, _ = self.advs[feat_idx].score(new_train_loader)
            val_train_acc, _, _ = self.advs[feat_idx].score(new_val_loader)

            print(f"====== Adv trained with {curr_ignored_feats} ======")
            for feat_idx, acc in curr_train_acc.items():
                print(
                    f"Feat {feat_idx} - Train: {acc.avg():.3f} - Val: {val_train_acc[feat_idx].avg():.3f}"
                )
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
        # train_loader_no_x = remove_orig_from_loader(train_loader)
        train_loader_with_x = get_orig_train_loader(train_loader)

        results: List[Tuple[float, float, Dict[int, nn.Module]]] = []
        for wd in [0.0001]:
            self._train(train_loader_with_x, nb_epochs, lr, wd, val_loader=val_loader)
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
                    f"[{self.tag} ADV with wd={wd}] {train_acc:.3f} -> {val_acc:.3f}"
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

        loss: Dict[int, torch.Tensor] = {}
        acc: Dict[int, torch.Tensor] = {}

        curr_sens_feats = self.dataset.sens_feats.copy()
        curr_ignored_feats = []

        for j, feat_idx in enumerate(self.sens_feats):

            if self.only_nonsensitive:
                curr_ignored_feats = list(range(self.dataset.tot_feats))
                for i_feat_idx in self.sens_feats:
                    curr_ignored_feats.remove(
                        i_feat_idx
                    )  # Only ignore non-sensitive features during bucketization
            else:
                curr_ignored_feats = list(range(self.dataset.tot_feats))
                curr_ignored_feats.remove(feat_idx)

            curr_sens_feats = [feat_idx]

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
            curr_sens = proxy_loader.dataset.tensors[1]  # type: ignore[attr-defined]

            # Predict with the correct adversary
            adv_pred_list = self.advs[feat_idx].forward(curr_z, feats=curr_sens_feats)
            assert len(adv_pred_list) == 1
            adv_preds = adv_pred_list[0]

            sens_target_idx = self.dataset.sens_feats.index(
                feat_idx
            )  # Assumes that the sensitive features are sorted

            assert (
                sens_targets[:, sens_target_idx].flatten() == curr_sens.flatten()
            ).all()

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

        if logger is not None:
            logger.info(f"[{self.tag} adv] adv accuracy per feature:")
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
                f"[{self.tag} adv] train_acc={adv_train_acc:.3f}, val_acc={adv_val_acc:.3f}, test_acc={adv_test_acc:.3f}"
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
