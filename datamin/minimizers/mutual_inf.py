from typing import Optional

import neptune.new as neptune
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from datamin.bucketization import Bucketization
from datamin.classifiers.classifier import Classifier
from datamin.dataset import FolktablesDataset
from datamin.utils.config import MutualInfMinimizerConfig
from datamin.utils.logging_utils import CLogger, get_print_logger
from datamin.utils.utils import Stat, TempScheduler

from .neural_minimizer import NeuralMinimizer


class MutualInformationMinimizer(NeuralMinimizer):
    def __init__(
        self,
        config: MutualInfMinimizerConfig,
        logger: Optional[CLogger] = None,
        run: Optional[neptune.Run] = None,
    ):
        super(MutualInformationMinimizer, self).__init__(
            config.device, config.mi_max_buckets
        )
        self.W = config.mi_weight
        self.config = config
        self.run = run
        self.n_epochs = config.mi_n_epochs
        if logger is None:
            logger = get_print_logger("MI-Logger")
        self.logger = logger

    def get_bucketization(self) -> Bucketization:
        return self._to_bucketization()  # cache?

    def fit(self, dataset: FolktablesDataset) -> None:
        self.dataset = dataset
        # self.evaluator = Evaluator(self.args, self.run, self.dataset)

        self._prepare_encoders(dataset, self.config.freeze_feature)
        self.logger.info(f"new_nb_fts={self.new_nb_fts}")

        nb_out_fts = dataset.nb_out_fts

        # Init clf and adv
        clf = Classifier(
            self.config.device,
            self.config.clf_config.clf_model,
            self.new_nb_fts,
            nb_out_fts,
        )
        # adv = Adversary(
        #     self.config.device,
        #     self.config.adv_config.clf_config.clf_model,
        #     self.dataset,
        #     self.new_nb_fts,
        # )

        # Init optimizers
        enc_params = []
        for enc in self.encoders:
            enc_params += list(enc.parameters())
        clf_params = list(clf.clf.parameters())

        opt = optim.Adam(
            list(enc_params) + list(clf_params),
            lr=self.config.clf_config.clf_lr,
            weight_decay=self.config.clf_config.clf_weight_decay,
        )
        lr_sched = StepLR(opt, step_size=5, gamma=0.1)

        temp_sched = TempScheduler(
            2.0, 0.5, len(dataset.buck_train_loader) * self.n_epochs
        )

        # Evaluate and print buckets at init
        for enc in self.encoders:
            enc.eval()
        bucketization = self._to_bucketization()
        bucketization.print_buckets(self.logger)
        # self.evaluator.evaluate(bucketization, verbose=True, tune_wd=True, guarantees=False)

        # We need a random loader
        ds = self.dataset.buck_train_loader.dataset
        random_loader = DataLoader(
            ds,
            sampler=RandomSampler(ds, replacement=True),  # type: ignore[arg-type]
            batch_size=self.config.batch_size,
            drop_last=True,
        )

        # Main loop
        for epoch in range(self.n_epochs):
            self.logger.info("LR:")
            for param_group in opt.param_groups:
                self.logger.info(param_group["lr"])

            tot_h_ub, tot_h_cond, tot_clf_loss, tot_clf_acc = (
                Stat(),
                Stat(),
                Stat(),
                Stat(),
            )

            pbar = tqdm(dataset.buck_train_loader)

            # Batches
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                temp = temp_sched.get_temp()
                opt.zero_grad()

                x_rnd = next(iter(random_loader))[0].to(self.device)

                # (1) Classifier loss
                for enc in self.encoders:
                    enc.train()

                z = self._bucketize(temp, x, use_hard=True)
                clf_preds = clf.predict(z)

                clf_acc: torch.Tensor = torch.tensor(0, device=self.device)
                if nb_out_fts == 2:
                    clf_acc = clf_preds.max(dim=1)[1].eq(y).float().mean()
                    clf_loss = F.cross_entropy(clf_preds, y)
                else:
                    criterion = torch.nn.BCEWithLogitsLoss()
                    clf_acc = ((clf_preds > 0.5) == y).float().mean()
                    clf_loss = criterion(clf_preds, y.float())

                # (2) MI loss
                for enc in self.encoders:
                    enc.eval()

                h_ub, h_cond = torch.tensor(0.0, device=self.device), torch.tensor(
                    0.0, device=self.device
                )
                for encoder, (_, _, feat_beg, feat_end, _, _) in zip(
                    self.encoders, dataset.feat_data
                ):
                    b, log_probs_x = encoder.sample(x[:, feat_beg:feat_end], temp)
                    rnd_slice = x_rnd[: x.shape[0], feat_beg:feat_end]
                    log_probs_x_rnd = encoder.log_probs(
                        rnd_slice, b.to(self.device), temp
                    )
                    h_ub += -log_probs_x_rnd.mean()
                    h_cond += -log_probs_x.mean()
                mi_ub = h_ub - h_cond

                # Combine, backprop
                loss = (1 - self.W) * clf_loss + self.W * mi_ub
                loss.backward()

                opt.step()
                temp_sched.step()

                # Print and log
                tot_h_ub += h_ub.item()
                tot_h_cond += h_cond.item()
                tot_clf_loss += clf_loss.item()
                tot_clf_acc += clf_acc.item()
                pbar.set_description(
                    "[mi] epoch=%d, temp=%.4f, clf_loss=%.4f, clf_acc=%.4f, h_ub=%.4f, h_cond=%.4f"
                    % (
                        epoch,
                        temp_sched.get_temp(),
                        tot_clf_loss.avg(),
                        tot_clf_acc.avg(),
                        tot_h_ub.avg(),
                        tot_h_cond.avg(),
                    )
                )
                if self.run is not None:
                    self.run["train_mi/batch/mi_ub"].log(mi_ub.item())
                    self.run["train_mi/batch/h_ub"].log(h_ub.item())
                    self.run["train_mi/batch/h_cond"].log(h_cond.item())
                    self.run["train_mi/batch/clf_loss"].log(clf_loss.item())
                    self.run["train_mi/batch/clf_acc"].log(clf_acc.item())
                    self.run["train_mi/batch/total_loss"].log(loss.item())

            # Epoch end
            lr_sched.step()

            for enc in self.encoders:
                enc.eval()

            if (epoch + 1) % 5 == 0 or epoch == self.n_epochs - 1:
                for enc in self.encoders:
                    enc.eval()
                self.logger.info("==================")

                bucketization = self._to_bucketization()
                bucketization.print_buckets(self.logger)
                # self.evaluator.evaluate(bucketization, verbose=False, tune_wd=True, guarantees=False)
