from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader

from datamin.classifiers.classifier import Classifier
from datamin.model_factory import get_model
from datamin.utils.logging_utils import CLogger, get_print_logger
from datamin.utils.utils import Stat

BATCH_SIZE = 256
MAX_PHYSICAL_BATCH_SIZE = 256
DELTA = 1e-5


class DPClassifier(Classifier):
    def __init__(
        self,
        device: str,
        model_name: str,
        nb_fts: int,
        nb_out_fts: int = 2,
        clf_dp_noise: float = 1.0,
        logger: Optional[CLogger] = None,
    ):
        super().__init__(device, model_name, nb_fts, nb_out_fts)
        self.wd = 0.004
        self.clf_dp_noise = clf_dp_noise
        self.logger = logger if logger is not None else get_print_logger("DPClassifier")

    def _train(
        self, train_loader: DataLoader, nb_epochs: int, lr: float, max_grad_norm: float
    ) -> None:
        opt = optim.Adam(self.clf.parameters(), lr=lr, weight_decay=self.wd)  # type: ignore[has-type]

        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=self.clf,  # type: ignore[has-type]
            optimizer=opt,
            data_loader=train_loader,
            noise_multiplier=self.clf_dp_noise,
            max_grad_norm=max_grad_norm,
        )
        self.logger.info(
            f"Using sigma={optimizer.noise_multiplier} and C={optimizer.max_grad_norm}"
        )

        for epoch in range(nb_epochs):
            acc = Stat()
            loss = Stat()

            bmm_loader = BatchMemoryManager(
                data_loader=data_loader,
                max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                optimizer=optimizer,
            )

            # pbar = tqdm(train_loader)
            for i, (z, y) in enumerate(bmm_loader.data_loader):
                z, y = z.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                clf_preds = model(z)

                assert self.nb_out_fts == 2
                clf_acc = clf_preds.max(dim=1)[1].eq(y).float().mean()
                clf_loss = F.cross_entropy(clf_preds, y)

                acc += (clf_acc.item(), z.shape[0])
                loss += (clf_loss.item(), z.shape[0])

                clf_loss.backward()
                optimizer.step()

            epsilon = privacy_engine.get_epsilon(DELTA)
            self.logger.info(
                f"\tTrain Epoch: {epoch} \t"
                f"Loss: {loss.avg():.6f} "
                f"Acc: {acc.avg() * 100:.6f} "
                f"(ε = {epsilon:.2f}, δ = {DELTA})"
            )

        self.clf = model
        self.pe = privacy_engine
        return

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.clf(z)

    def get_dp_stats(self) -> float:
        return self.pe.get_epsilon(DELTA)

    def fit(
        self,
        train_loader: DataLoader,
        nb_epochs: int,
        lr: float,
        tune_wd: bool = False,  # Is actually tune_dp in this case
        max_grad_norm: Optional[float] = None,
        val_loader: Optional[DataLoader] = None,
    ) -> None:

        if self.trained:  # type: ignore[has-type]
            raise RuntimeError(
                "Are you sure you want to call .fit() on a trained classifier?"
            )
        self.trained = True  # can call during training
        tune_dp = tune_wd
        if tune_dp:
            results = []
            for mgn in [0.1, 0.2, 0.5, 1, 2, 5, 10]:
                self.clf = get_model(
                    self.model_name, self.device, self.nb_fts, self.nb_out_fts
                )
                self._train(train_loader, nb_epochs, lr, mgn)
                with torch.no_grad():
                    assert val_loader is not None
                    train_acc = self.score(train_loader)
                    val_acc = self.score(val_loader)
                    self.logger.info(
                        f"[Trying CLF with Max Grad Norm={mgn}] {train_acc:.3f} -> {val_acc:.3f}"
                    )
                results.append((val_acc, mgn, self.clf, self.pe))
            results = sorted(results)  # sort by validation
            mgn = results[-1][1]
            self.clf = results[-1][2]
            self.pe = results[-1][3]
            self.logger.info(f"Chose Magnitude {mgn}")
        else:
            assert max_grad_norm is not None
            self.clf = get_model(self.model_name, self.device, self.nb_fts, 2)
            self._train(train_loader, nb_epochs, lr, max_grad_norm)

        self.trained = True
