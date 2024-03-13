from typing import Optional

import neptune.new as neptune

from datamin.utils.config import (
    AdvMinimizerConfig,
    DataAnonymizationMinimizerConfig,
    DatasetConfig,
    FeatSelMinimizerConfig,
    ForestMinimizerConfig,
    IBMMinimizerConfig,
    IterativeMinimizerConfig,
    IterativeTreeMinimizerConfig,
    Minimizer,
    MinimizerConfig,
    MutualInfMinimizerConfig,
    TreeMinimizerConfig,
    UniformMinimizerConfig,
)
from datamin.utils.logging_utils import CLogger

from .abstract_minimizer import AbstractMinimizer
from .adv_train import AdversarialTrainingMinimizer
from .da_minimizer import DataAnonymizationMinimizer
from .feat_sel import FeatureSelectionMinimizer
from .ibm_apt import IbmAptMinimizer
from .iterative import IterativeMinimizer
from .mutual_inf import MutualInformationMinimizer
from .tree.forest_minimizer import ForestMinimizer
from .tree.iterative_tree_minimizer import IterativeTreeMinimizer
from .tree.tree_minimizer import TreeMinimizer
from .uniform import UniformMinimizer


def get_minimizer(
    min_config: MinimizerConfig,
    dataset_config: DatasetConfig,
    logger: CLogger,
    run: Optional[neptune.Run] = None,
) -> AbstractMinimizer:

    if min_config.minimizer == Minimizer.advtrain:
        assert isinstance(min_config, AdvMinimizerConfig)
        return AdversarialTrainingMinimizer(min_config, logger, run)
    elif min_config.minimizer == Minimizer.featsel:
        assert isinstance(min_config, FeatSelMinimizerConfig)
        return FeatureSelectionMinimizer(min_config, logger, run)
    elif min_config.minimizer == Minimizer.ibm:
        assert isinstance(min_config, IBMMinimizerConfig)
        return IbmAptMinimizer(min_config, logger, run)
    elif min_config.minimizer == Minimizer.iterative:
        assert isinstance(min_config, IterativeMinimizerConfig)
        return IterativeMinimizer(min_config, logger, run)
    elif min_config.minimizer == Minimizer.mi:
        assert isinstance(min_config, MutualInfMinimizerConfig)
        return MutualInformationMinimizer(min_config, logger, run)
    elif min_config.minimizer == Minimizer.uniform:
        assert isinstance(min_config, UniformMinimizerConfig)
        return UniformMinimizer(min_config, logger, run)
    elif min_config.minimizer == Minimizer.tree:
        assert isinstance(min_config, TreeMinimizerConfig)
        return TreeMinimizer(min_config, logger, run)
    elif min_config.minimizer == Minimizer.forest:
        assert isinstance(min_config, ForestMinimizerConfig)
        return ForestMinimizer(min_config, logger, run)
    elif min_config.minimizer == Minimizer.iterative_tree:
        assert isinstance(min_config, IterativeTreeMinimizerConfig)
        minimizer = IterativeTreeMinimizer(min_config, logger, run)
        minimizer.set_dataset_config(dataset_config)
        return minimizer
    elif min_config.minimizer == Minimizer.data_anonymization:
        assert isinstance(min_config, DataAnonymizationMinimizerConfig)
        return DataAnonymizationMinimizer(min_config, logger, run)

    else:
        raise RuntimeError(f"Minimizer {min_config.minimizer} not registered")
