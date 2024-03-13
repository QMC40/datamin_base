from typing import Optional

import neptune.new as neptune
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from datamin.adversaries.adversary import Adversary
from datamin.bucketization import Bucketization
from datamin.classifiers.classifier import Classifier
from datamin.dataset import FolktablesDataset
from datamin.utils.config import AdvMinimizerConfig
from datamin.utils.logging_utils import CLogger, get_print_logger
from datamin.utils.utils import Stat, TempScheduler, add_to_dict, itemize_dict

from .neural_minimizer import NeuralMinimizer


class AdversarialTrainingMinimizer(NeuralMinimizer):
    def __init__(
        self,
        config: AdvMinimizerConfig,
        logger: Optional[CLogger] = None,
        run: Optional[neptune.Run] = None,
    ):
        super(AdversarialTrainingMinimizer, self).__init__(
            config.device, config.advtrain_max_buckets
        )
        self.W = config.advtrain_weight
        self.config = config
        self.run = run
        self.n_epochs = config.advtrain_n_epochs
        if logger is None:
            logger = get_print_logger("Advtrain-Logger")
        self.logger = logger

    def get_bucketization(self) -> Bucketization:
        return self._to_bucketization()  # cache?

    def fit(self, dataset: FolktablesDataset) -> None:
        self.dataset = dataset
        # self.evaluator = Evaluator(self.args, self.run, self.dataset)

        self._prepare_encoders(dataset, self.config.freeze_feature)
        self.logger.info(f"new_nb_fts={self.new_nb_fts}")

        # Init clf and adv
        clf = Classifier(
            self.config.device, self.config.clf_config.clf_model, self.new_nb_fts
        )
        adv = Adversary(
            self.config.device,
            self.config.adv_config.clf_config.clf_model,
            self.dataset,
            self.new_nb_fts,
        )

        # Init optimizers
        enc_params, adv_params = [], []
        for enc in self.encoders:
            enc_params += list(enc.parameters())
        clf_params = list(clf.clf.parameters())
        for a in adv.advs.values():
            adv_params += list(a.parameters())

        opt = optim.Adam(
            list(enc_params) + list(clf_params),
            lr=self.config.clf_config.clf_lr,
            weight_decay=self.config.clf_config.clf_weight_decay,
        )
        lr_sched = StepLR(opt, step_size=5, gamma=0.1)

        opt_adv = optim.Adam(
            list(adv_params),
            lr=self.config.adv_config.clf_config.clf_lr,
            weight_decay=self.config.adv_config.clf_config.clf_weight_decay,
        )
        lr_sched_adv = StepLR(opt_adv, step_size=5, gamma=0.1)

        temp_sched = TempScheduler(
            2.0, 0.5, len(self.dataset.buck_train_loader) * self.n_epochs
        )

        # Evaluate and print buckets at init
        for enc in self.encoders:
            enc.eval()
        bucketization = self._to_bucketization()
        bucketization.print_buckets(self.logger)
        # self.evaluator.evaluate(bucketization, verbose=True, tune_wd=True, guarantees=False)

        # Separate loader for adv
        ds = self.dataset.buck_train_loader.dataset
        adv_train_loader = DataLoader(
            ds,
            sampler=RandomSampler(ds, replacement=True),  # type: ignore
            batch_size=self.config.batch_size,
            drop_last=True,
        )

        # Main loop
        for epoch in range(self.n_epochs):
            self.logger.info("LR:")
            for param_group in opt.param_groups:
                self.logger.info(param_group["lr"])

            tot_clf_loss, tot_clf_acc = Stat(), Stat()
            tot_adv_feat_acc = {
                feat_idx: Stat() for feat_idx in self.dataset.sens_feats
            }
            tot_adv_feat_loss = {
                feat_idx: Stat() for feat_idx in self.dataset.sens_feats
            }

            pbar = tqdm(dataset.train_loader)

            # Batches
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                temp = temp_sched.get_temp()

                # (1) Train Adversary (inner)
                for enc in self.encoders:
                    enc.eval()
                for _ in range(self.config.advtrain_inner_steps):
                    x_adv, y_adv = next(iter(adv_train_loader))
                    opt_adv.zero_grad()
                    z_adv = self._bucketize(temp, x_adv, use_hard=False)

                    sens_targets = adv.extract_targets(x_adv)
                    adv_feat_loss, adv_feat_acc, _ = adv.predict(z_adv, sens_targets)

                    adv_loss = torch.mean(torch.stack(list(adv_feat_loss.values())))
                    adv_acc = torch.mean(torch.stack(list(adv_feat_acc.values())))

                    adv_loss.backward()
                    opt_adv.step()

                # (2) Train Encoder+Classifier (outer)
                opt.zero_grad()
                for enc in self.encoders:
                    enc.train()

                # (2A) Classifier Loss
                z = self._bucketize(temp, x, use_hard=False)
                clf_preds = clf.predict(z)
                clf_acc = clf_preds.max(dim=1)[1].eq(y).float().mean()
                clf_loss = F.cross_entropy(clf_preds, y)

                tot_clf_acc += clf_acc.item()
                tot_clf_loss += clf_loss.item()

                # (2B) Adversary Loss
                sens_targets = adv.extract_targets(x)
                adv_feat_loss, adv_feat_acc, _ = adv.predict(z, sens_targets)

                adv_loss = torch.mean(torch.stack(list(adv_feat_loss.values())))
                adv_acc = torch.mean(torch.stack(list(adv_feat_acc.values())))

                add_to_dict(tot_adv_feat_acc, itemize_dict(adv_feat_acc))
                add_to_dict(tot_adv_feat_loss, itemize_dict(adv_feat_loss))

                # Combine, backprop
                all_losses = (1 - self.W) * clf_loss - self.W * adv_loss
                all_losses.backward()
                opt.step()
                temp_sched.step()

                # Print and log to neptune
                tot_adv_loss = np.mean(
                    [
                        tot_adv_feat_loss[feat_idx].avg()
                        for feat_idx in dataset.sens_feats
                    ]
                )
                tot_adv_acc = np.mean(
                    [
                        tot_adv_feat_acc[feat_idx].avg()
                        for feat_idx in dataset.sens_feats
                    ]
                )
                desc = (
                    "[epoch=%d T=%.4f] (CLF | loss=%.3f, acc=%.3f), (ADV %.2f | loss=%.3f, acc=%.3f)"
                    % (
                        epoch,
                        temp_sched.get_temp(),
                        tot_clf_loss.avg(),
                        tot_clf_acc.avg(),
                        self.W,
                        tot_adv_loss,
                        tot_adv_acc,
                    )
                )
                pbar.set_description(desc)
                if self.run is not None:
                    self.run["adv_train/batch/clf_loss"].log(clf_loss.item())
                    self.run["adv_train/batch/clf_acc"].log(clf_acc.item())

                    self.run["adv_train/batch/adv_loss"].log(adv_loss.item())
                    self.run["adv_train/batch/adv_acc"].log(adv_acc.item())

                    self.run["adv_train/batch/all_losses"].log(all_losses.item())
            # Epoch end
            lr_sched.step()
            lr_sched_adv.step()

            for enc in self.encoders:
                enc.eval()

            if (epoch + 1) % 5 == 0 or epoch == self.n_epochs - 1:
                for enc in self.encoders:
                    enc.eval()
                self.logger.info("==================")

                bucketization = self._to_bucketization()
                bucketization.print_buckets(self.logger)

                # self.evaluator.evaluate(bucketization, verbose=False, tune_wd=True, guarantees=False)
        # Done
        self.logger.info("adv_train done")
