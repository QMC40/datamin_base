from typing import Optional

from datamin.adversaries.adversary import Adversary
from datamin.adversaries.iterative_adversary import IterativeAdversary
from datamin.adversaries.linkage_adversary import LinkageAdversary
from datamin.adversaries.loa_adversary import LeaveOutAdversary
from datamin.adversaries.multi_adversary import MultiAdversary
from datamin.adversaries.outlier_adversary import OutlierAdversary
from datamin.bucketization import Bucketization
from datamin.dataset import FolktablesDataset
from datamin.utils.config import AdversaryConfig, AdversaryType
from datamin.utils.logging_utils import CLogger


def get_adversary_from_config(
    adv_config: AdversaryConfig,
    dataset: FolktablesDataset,
    nb_feats: int,
    bucketization: Bucketization,
    device: str = "cpu",
    logger: Optional[CLogger] = None,
) -> Adversary:

    if adv_config.adv_type == AdversaryType.RECOVERY:
        adv = Adversary(
            device,
            adv_config.clf_config.clf_model,
            dataset,
            nb_feats,
            only_predict_seen=adv_config.only_predict_seen,
            use_label=adv_config.use_label,
            logger=logger,
        )
    elif adv_config.adv_type == AdversaryType.MULTIRECOVERY:
        adv = MultiAdversary(
            device,
            adv_config.clf_config.clf_model,
            dataset,
            nb_feats,
            only_predict_seen=adv_config.only_predict_seen,
            logger=logger,
        )
    elif adv_config.adv_type == AdversaryType.LINKAGE:
        assert bucketization is not None
        assert adv_config.linkage_algorithm is not None
        assert len(adv_config.linkage_a_features) > 0
        assert len(adv_config.linkage_b_features) > 0
        adv = LinkageAdversary(
            device,
            adv_config.clf_config.clf_model,
            dataset,
            nb_feats,
            adv_config.linkage_algorithm,
            adv_config.linkage_a_features,
            adv_config.linkage_b_features,
            logger=logger,
        )
    elif adv_config.adv_type == AdversaryType.OUTLIER:
        assert bucketization is not None
        adv = OutlierAdversary(
            device, adv_config.clf_config.clf_model, dataset, nb_feats, logger=logger
        )
    elif adv_config.adv_type == AdversaryType.ITERATIVE:
        assert bucketization is not None
        adv = IterativeAdversary(
            device,
            adv_config.clf_config.clf_model,
            dataset,
            nb_feats,
            only_predict_seen=adv_config.only_predict_seen,
            logger=logger,
        )
    elif adv_config.adv_type == AdversaryType.LEAKNONSENSITIVE:
        assert bucketization is not None
        adv = LeaveOutAdversary(
            device,
            adv_config.clf_config.clf_model,
            dataset,
            nb_feats,
            only_nonsensitive=True,
            only_predict_seen=adv_config.only_predict_seen,
            logger=logger,
        )
    elif adv_config.adv_type == AdversaryType.LEAKLEAVEONEOUT:
        assert bucketization is not None
        adv = LeaveOutAdversary(
            device,
            adv_config.clf_config.clf_model,
            dataset,
            nb_feats,
            only_nonsensitive=False,
            only_predict_seen=adv_config.only_predict_seen,
            logger=logger,
        )
    else:
        raise ValueError(f"Unknown adversary type {adv_config.adv_type}")
    adv.set_bucketization(bucketization)  # type: ignore
    return adv
