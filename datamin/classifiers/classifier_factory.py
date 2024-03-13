from typing import Optional

from datamin.classifiers.classifier import Classifier
from datamin.classifiers.dp_classifier import DPClassifier
from datamin.utils.config import ClassifierConfig, ClassifierType
from datamin.utils.logging_utils import CLogger


def get_classifier_from_config(
    clf_config: ClassifierConfig,
    nb_in_fts: int,
    nb_out_feats: int = 2,
    device: str = "cpu",
    logger: Optional[CLogger] = None,
) -> Classifier:

    if clf_config.clf_type == ClassifierType.NORMAL:
        return Classifier(
            device, clf_config.clf_model, nb_in_fts, nb_out_feats, logger=logger
        )
    if clf_config.clf_type == ClassifierType.DP:
        return DPClassifier(
            device,
            clf_config.clf_model,
            nb_in_fts,
            nb_out_feats,
            clf_dp_noise=clf_config.clf_dp_noise,
            logger=logger,
        )
    else:
        raise ValueError(f"Unknown classifier type {clf_config.clf_type}")
