from typing import Dict, List, Optional, Tuple

import neptune.new as neptune  # type: ignore

from datamin.utils.config import AdversaryType, EvaluatorConfig
from datamin.utils.logging_utils import CLogger

from .adversaries.adversary import Adversary
from .adversaries.adversary_factory import get_adversary_from_config
from .bucketization import Bucketization, MultiBucketization
from .classifiers.classifier_factory import get_classifier_from_config
from .dataset import FolktablesDataset
from .encoding_utils import encode_loader


class Evaluator:
    def __init__(
        self,
        config: EvaluatorConfig,
        run: Optional[neptune.Run],
        dataset: FolktablesDataset,
        device: str = "cpu",
    ):
        self.config: EvaluatorConfig = config
        self.run = run
        self.dataset = dataset
        self.device = device

    def evaluate(
        self,
        bucketization: Bucketization,
        logger: CLogger,
        verbose: bool = False,
        tune_wd: bool = False,
        guarantees: bool = False,
        fairness_sens_col: Optional[int] = None,
    ) -> Dict[str, List[Tuple[float, float, float]]]:

        new_nb_fts = bucketization.total_size()
        # bucketization.get_stats_on_input(x=self.dataset.train_loader.dataset.tensors[0])
        logger.info(
            f"----> Evaluating: Training new classifiers and a new adversary on 1-hot buckets ({new_nb_fts} fts) - {self.dataset.train_loader.dataset.tensors[0].shape} points"  # type: ignore[attr-defined]
        )

        accs: List[Tuple[float, float, float]] = []
        adv_accs: List[Tuple[float, float, float]] = []

        # train clf
        clf_train_loader = encode_loader(
            self.dataset.train_loader, bucketization, shuffle=True
        )
        clf_val_loader = encode_loader(self.dataset.val_loader, bucketization)
        clf_test_loader = encode_loader(self.dataset.test_loader, bucketization)

        nb_fts = clf_train_loader.dataset.tensors[0].shape[1]  # type: ignore[attr-defined]
        if len(clf_train_loader.dataset.tensors[1].shape) == 1:  # type: ignore[attr-defined]
            nb_out_fts = 2
        else:
            nb_out_fts = clf_train_loader.dataset.tensors[1].shape[1]  # type: ignore[attr-defined]

        for i, clf_cfg in enumerate(self.config.clf_configs):
            logger.info(f"----> Evaluating: Training classifier {i+1}")
            clf = get_classifier_from_config(
                clf_cfg, nb_fts, nb_out_fts, device=self.device, logger=logger
            )

            if clf_cfg.clf_tune_wd:
                clf.fit(
                    clf_train_loader,
                    clf_cfg.clf_epochs,
                    clf_cfg.clf_lr,
                    tune_wd=clf_cfg.clf_tune_wd,
                    val_loader=clf_val_loader,
                )
            else:
                logger.info(f"Using LR={clf_cfg.clf_lr}")
                clf.fit(
                    clf_train_loader,
                    clf_cfg.clf_epochs,
                    clf_cfg.clf_lr,
                    tune_wd=False,
                    weight_decay=clf_cfg.clf_weight_decay,
                )

            if fairness_sens_col is not None:
                logger.info(
                    "Evaluating fairness (equalized odds) --- will use this metric to tune and report the classifier, ignoring the adversary"
                )
                x, _ = (
                    self.dataset.test_loader.dataset.tensors[0],  # type: ignore[attr-defined]
                    self.dataset.test_loader.dataset.tensors[1],  # type: ignore[attr-defined]
                )

                sens = x[
                    :, fairness_sens_col
                ]  # fairness_sens_col=723 for ACSIncome second 1-hot column in sex,efactor

                clf_test_acc = clf.score(clf_test_loader, sens=sens)
                logger.info(f"clf fairness on test={clf_test_acc:.5f}")
                run = self.run
                if run is not None:
                    run["fairness/test/eqodds"].log(clf_test_acc)
                return {"clf": [(clf_test_acc, clf_test_acc, clf_test_acc)]}
                # DONE

            clf_train_acc = clf.score(clf_train_loader)
            clf_val_acc = clf.score(clf_val_loader)
            clf_test_acc = clf.score(clf_test_loader)

            accs.append((clf_train_acc, clf_val_acc, clf_test_acc))

            logger.info(
                f"[clf] train_acc={clf_train_acc:.3f}, val_acc={clf_val_acc:.3f}, test_acc={clf_test_acc:.3f}"
            )

        # train adv
        for i, adv_cfg in enumerate(self.config.adv_configs):
            adv_config = adv_cfg.clf_config

            if adv_cfg.use_label:
                new_nb_fts = bucketization.total_size() + 1
            else:
                new_nb_fts = bucketization.total_size()

            if isinstance(bucketization, MultiBucketization):
                assert adv_cfg.adv_type in [
                    AdversaryType.RECOVERY,
                    AdversaryType.MULTIRECOVERY,
                ]

            adv = get_adversary_from_config(
                adv_cfg,
                self.dataset,
                new_nb_fts,
                bucketization,
                device=self.device,
                logger=logger,
            )

            logger.info(
                f"Adversary uses {self.dataset.adv_train_loader.dataset.tensors[0].shape} points!"  # type: ignore[attr-defined]
            )

            adv_train_loader = encode_loader(
                self.dataset.adv_train_loader,
                bucketization,
                adv_extractor=adv.extract_targets,
                shuffle=True,
                requires_orig=adv_cfg.requires_original_data,
                adv_use_label=adv_cfg.use_label,
            )
            adv_val_loader = encode_loader(
                self.dataset.val_loader,
                bucketization,
                adv_extractor=adv.extract_targets,
                requires_orig=adv_cfg.requires_original_data,
                adv_use_label=adv_cfg.use_label,
            )
            adv_test_loader = encode_loader(
                self.dataset.test_loader,
                bucketization,
                adv_extractor=adv.extract_targets,
                requires_orig=adv_cfg.requires_original_data,
                adv_use_label=adv_cfg.use_label,
            )

            if adv_config.clf_tune_wd:
                adv.fit(
                    adv_train_loader,
                    adv_config.clf_epochs,
                    adv_config.clf_lr,
                    weight_decay=adv_config.clf_weight_decay,
                    tune_wd=True,
                    val_loader=adv_val_loader,
                )
            else:
                adv.fit(
                    adv_train_loader,
                    adv_config.clf_epochs,
                    adv_config.clf_lr,
                    tune_wd=False,
                    weight_decay=adv_config.clf_weight_decay,
                    val_loader=adv_val_loader,
                )

            adv_train_acc, adv_val_acc, adv_test_acc = adv.get_full_eval(
                train_loader=adv_train_loader,
                val_loader=adv_val_loader,
                test_loader=adv_test_loader,
                logger=logger,
            )

            adv_accs.append((adv_train_acc, adv_val_acc, adv_test_acc))

        # Adversary upper bound? Compute here and also log to neptune
        if guarantees:
            res = Adversary.compute_upper_bound(
                self.dataset.sens_feats,
                self.dataset.feat_data,
                adv_train_loader,  # type: ignore
                adv_val_loader,  # type: ignore
                adv_test_loader,  # type: ignore
            )
            for k in ["train", "test"]:
                ub, conf = res[k]
                print(
                    f"[G --- {k}] adv. acc. is upper-bounded by {ub} with confidence {conf}"
                )
                if self.run is not None:
                    self.run[f"guarantees/{k}/ub"].log(ub)
                    self.run[f"guarantees/{k}/confidence"].log(conf)

        # Save
        if verbose:
            logger.info("==================")
        run = self.run
        if run is not None:
            run["clf/train/acc"].log(accs[0][0])
            run["clf/val/acc"].log(accs[0][1])
            run["clf/test/acc"].log(accs[0][2])
            run["clf/overfitting"].log(accs[0][0] - accs[0][2])

            run["adv/train/acc"].log(adv_accs[0][0])
            run["adv/val/acc"].log(adv_accs[0][1])
            run["adv/test/acc"].log(adv_accs[0][2])
            run["adv/overfitting"].log(adv_accs[0][0] - adv_accs[0][2])

        res_dict = {"clf": accs}

        for i, adv_cfg in enumerate(self.config.adv_configs):
            key = "adv_" + adv_cfg.adv_type.value
            if adv_cfg.only_predict_seen:
                key += "-ops"
            if adv_cfg.use_label:
                key += "-ul"

            res_dict[key] = [adv_accs[i]]

        return res_dict
