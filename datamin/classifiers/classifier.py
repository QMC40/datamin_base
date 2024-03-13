# import pandas as pd
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

# from aif360.sklearn.metrics import average_odds_error
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from datamin.model_factory import get_model
from datamin.utils.logging_utils import CLogger, get_print_logger
from datamin.utils.utils import Stat


class Classifier:
    def __init__(
        self,
        device: str,
        model_name: str,
        nb_fts: int,
        nb_out_fts: int = 2,
        logger: Optional[CLogger] = None,
    ):
        self.clf = get_model(
            model_name, device, nb_fts, nb_out_fts
        )  # might be replaced
        self.device = device
        self.model_name = model_name
        self.trained = False
        self.nb_fts = nb_fts
        self.nb_out_fts = nb_out_fts
        if self.nb_out_fts > 2:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        if logger is None:
            logger = get_print_logger("Classifier-Logger")
        self.logger = logger

    def _train(
        self, train_loader: DataLoader, nb_epochs: int, lr: float, weight_decay: float
    ) -> None:
        opt = optim.Adam(self.clf.parameters(), lr=lr, weight_decay=weight_decay)
        lr_sched = StepLR(opt, step_size=nb_epochs // 2, gamma=0.1)

        for epoch in range(nb_epochs):
            acc = Stat()
            loss = Stat()

            # pbar = tqdm(train_loader)
            for z, y in train_loader:
                z, y = z.to(self.device), y.to(self.device)
                opt.zero_grad()
                clf_preds = self.predict(z)

                # TODO Adapt here!
                if self.nb_out_fts == 2:
                    clf_acc = clf_preds.max(dim=1)[1].eq(y).float().mean()
                    clf_loss = F.cross_entropy(clf_preds, y)
                else:
                    clf_acc = ((clf_preds > 0.5) == y).float().mean()
                    clf_loss = self.criterion(clf_preds, y.float())

                acc += (clf_acc.item(), z.shape[0])
                loss += (clf_loss.item(), z.shape[0])

                clf_loss.backward()
                opt.step()
            # print('[clf] epoch=%d, loss=%.3f acc=%.3f' % (
            #        epoch, loss.avg(), acc.avg()
            #    ))
            lr_sched.step()
        return  # call eval

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.clf(z)

    def fit(
        self,
        train_loader: DataLoader,
        nb_epochs: int,
        lr: float,
        tune_wd: bool = False,
        weight_decay: Optional[float] = None,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        # print(f'Training a classifier for {nb_epochs} epochs')
        if self.trained:
            raise RuntimeError(
                "Are you sure you want to call .fit() on a trained classifier?"
            )
        self.trained = True  # can call during training

        if tune_wd:
            results = []
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
                self.clf = get_model(
                    self.model_name, self.device, self.nb_fts, self.nb_out_fts
                )
                self._train(train_loader, nb_epochs, lr, wd)
                with torch.no_grad():
                    assert val_loader is not None
                    train_acc = self.score(train_loader)
                    val_acc = self.score(val_loader)
                    self.logger.info(
                        f"[Trying CLF with wd={wd}] {train_acc:.3f} -> {val_acc:.3f}"
                    )
                results.append((val_acc, wd, self.clf))
            results = sorted(results)  # sort by validation
            weight_decay = results[-1][1]
            self.clf = results[-1][2]
            self.logger.info(f"Chose {weight_decay}")
        else:
            assert weight_decay is not None
            self.clf = get_model(self.model_name, self.device, self.nb_fts, 2)
            self._train(train_loader, nb_epochs, lr, weight_decay)

        self.trained = True

    def score(self, loader: DataLoader, sens: Optional[torch.Tensor] = None) -> float:
        if sens is not None:
            # The score should be equalized odds with 0/1 sensitive atributes
            return self.equalized_odds(loader, sens)

        if not self.trained:
            raise RuntimeError("Can not call score on not trained clf")
        with torch.no_grad():
            tot_clf_loss, tot_clf_acc = Stat(), Stat()
            for z, y in loader:
                z, y = z.to(self.device), y.to(self.device)
                clf_preds = self.predict(z)

                if self.nb_out_fts == 2:
                    clf_acc = clf_preds.max(dim=1)[1].eq(y).float().mean()
                    clf_loss = F.cross_entropy(clf_preds, y)
                else:
                    clf_acc = ((clf_preds > 0.5) == y).float().mean()
                    clf_loss = self.criterion(clf_preds, y.float())
                tot_clf_loss += (clf_loss.item(), z.shape[0])
                tot_clf_acc += (clf_acc.item(), z.shape[0])
            # print(tot_clf_loss.avg(), tot_clf_acc.avg())
            return tot_clf_acc.avg()

    def equalized_odds(self, loader: DataLoader, sens: torch.Tensor) -> float:
        z, y = loader.dataset.tensors  # type: ignore
        assert isinstance(z, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        preds = self.predict(z)
        assert sens.shape[0] == y.shape[0]
        assert sens.shape[0] == preds.shape[0]

        sens0 = sens == 0
        sens1 = sens == 1
        target_true = y == 1
        target_false = y == 0

        tpr0 = preds[sens0 & target_true].sum() / target_true[sens0].sum()
        tpr1 = preds[sens1 & target_true].sum() / target_true[sens1].sum()
        fpr0 = preds[sens0 & target_false].sum() / target_false[sens0].sum()
        fpr1 = preds[sens1 & target_false].sum() / target_false[sens1].sum()

        eqodd = ((tpr0 - tpr1).abs() + (fpr0 - fpr1).abs()) / 2
        return eqodd.item()
