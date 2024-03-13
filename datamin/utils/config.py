from __future__ import annotations

import os
from abc import ABC
from enum import Enum
from functools import reduce
from itertools import chain, product, repeat
from typing import Any, Dict, List, Optional, TypeVar, Union

from bunch import Bunch  # type: ignore[import]

T = TypeVar("T")
V = TypeVar("V")


def create_combs_from_list(
    obj: T, exclude: List[str], bunch_cfg: Bunch, new_type: Optional[V] = None
) -> Union[List[T], List[V]]:
    """Iterates over all attributes if it encounters a list creates to cross product of all possible choices
    Returns:
        List: List of all individual Combinations for obj type or specified V type
    """
    # Remove specific list features from product from list
    freeze_feature: Dict[str, Any] = {}
    for attr in exclude:
        if hasattr(obj, attr):
            freeze_feature[attr] = getattr(obj, attr)
            delattr(obj, attr)

    keys, values = zip(*obj.__dict__.items())
    values = [v if isinstance(v, list) else [v] for v in values]

    # Check whether arguments should be connected
    connect_lists = bunch_cfg.get("connect_arg_lists", False)
    if connect_lists:
        lens = [len(v) for v in values]
        max_len = max(lens)
        divides = reduce(lambda a, b: a and b, [max_len % le == 0 for le in lens])
        if divides:
            values = [list(chain(*repeat(v, (max_len // len(v))))) for v in values]
        else:
            raise ValueError(f"Lists cannot be connected {values}")
        combs = [Bunch(zip(keys, v)) for v in zip(*values)]
    else:
        combs = [Bunch(zip(keys, v)) for v in product(*values)]

    # Add frozen features
    for attr, value in freeze_feature.items():
        for comb in combs:
            setattr(comb, attr, value)

    typed_combs: Union[List[T], List[V]] = []  # type: ignore[assignment]
    if new_type is None:
        proto = obj.__class__
    else:
        proto = new_type.__class__  # type: ignore[assignment]
    if isinstance(obj, ClassifierMinimizerConfig):
        eval_config = EvaluatorConfig(
            Bunch({"clf_config": [obj.clf_config], "adv_config": [obj.adv_config]})
        )
        typed_combs = [proto(comb, eval_config) for comb in combs]  # type: ignore
    else:
        typed_combs = [proto(comb) for comb in combs]  # type: ignore
    return typed_combs


class NeptuneConfig:
    use_neptune: bool = False
    neptune_key: Optional[str]
    neptune_run_label: Optional[str]

    def __init__(self, config: Bunch):
        self.neptune_run_label = config.get("run_name", None)
        self.use_neptune = config.get("enable", False)
        if self.use_neptune:
            self.neptune_key = config.get("api_key", None)
            if self.neptune_key is None:
                self.neptune_key = os.getenv("NEPTUNE_API_KEY")
            if self.neptune_key is None:
                raise ValueError("Neptune API key is not provided")

    def __str__(self) -> str:
        ret = ""
        for attr, value in self.__dict__.items():
            ret += f" {attr}: {value}\n"
        return ret


class SensFeat:
    NONE = "none"
    ALL = "all"
    CONT = "cont"
    DISC = "disc"
    DYNAMIC = "dynamic"

    def __init__(self, sens_feats: Union[str, List[int]]) -> None:
        self.dyn_feats = []
        if isinstance(sens_feats, str):
            self.type = sens_feats
        else:
            self.type = SensFeat.DYNAMIC
            self.dyn_feats = sens_feats

    def get_feats(
        self, cont_feats: List[int], disc_feats: List[int], tot_num_feats: int
    ) -> List[int]:
        if self.type == SensFeat.NONE:
            sens_feats = []
        elif self.type == SensFeat.ALL:
            sens_feats = list(range(tot_num_feats))
        elif self.type == SensFeat.DISC:
            sens_feats = disc_feats
        elif self.type == SensFeat.CONT:
            sens_feats = cont_feats
        elif self.type == SensFeat.DYNAMIC:
            sens_feats = self.dyn_feats
        else:
            assert False, f"Unknown sens_feat: {self.type}"

        return sens_feats

    def __str__(self) -> str:
        if self.type == SensFeat.DYNAMIC:
            return f"{self.dyn_feats}"
        else:
            return f"{self.type}"


class DatasetConfig:
    dataset: str
    acs_state: str
    acs_year: int
    shift_state: Optional[str]
    shift_year: Optional[int]
    freeze_feature: List[int]
    train_percent: Optional[
        float
    ]  # How much of the total data is available for training
    bucketization_percent: Optional[
        float
    ]  # How much of the training data is used to determine the bucketization
    adv_percent: Optional[
        float
    ]  # How much of the bucketization data is leaked to train the adversary
    val_split: float
    test_split: float
    add_syn: bool
    sens_feats: SensFeat
    batch_size: int

    def __init__(self, config: Bunch):
        self.dataset = config.get("dataset", None)
        self.acs_state = config.get("acs_state", "CT")
        self.acs_year = config.get("acs_year", 2014)
        self.shift_state = config.get("shift_state", None)
        self.shift_year = config.get("shift_year", None)
        self.freeze_feature = config.get("freeze_feature", [])
        self.add_syn = config.get("add_syn", False)
        self.bucketization_percent = config.get("bucketization_percent", 1.0)
        self.train_percent = config.get("train_percent", 1.0)
        self.adv_percent = config.get("adv_percent", 1.0)
        sens_feats = config.get("sens_feats", "disc")
        if isinstance(sens_feats, SensFeat):
            self.sens_feats = sens_feats
        elif isinstance(sens_feats, str):
            self.sens_feats = SensFeat(sens_feats)
        elif isinstance(sens_feats, List):
            self.sens_feats = SensFeat(sens_feats)
        else:
            raise ValueError(f"Unknown sens_feats {sens_feats}")
        self.val_split = config.get("val_split", 0.1)
        self.test_split = config.get("test_split", 0.3)
        self.batch_size = config.get("batch_size", 256)

    def __str__(self) -> str:
        ret = ""
        for attr, value in self.__dict__.items():
            ret += f" {attr}: {value}\n"
        return ret

    def id_str(self) -> str:
        ret = ""
        for attr in [
            "dataset",
            "acs_state",
            "acs_year",
            "sens_feats",
            "val_split",
            "test_split",
            "train_percent",
            "bucketization_percent",
            "adv_percent",
        ]:
            ret += f"{getattr(self, attr)}_"
        return ret[:-1]


class Minimizer(Enum):
    tree = "tree"
    advtrain = "advtrain"
    featsel = "featsel"
    forest = "forest"
    ibm = "ibm"
    iterative = "iterative"
    mi = "mi"
    uniform = "uniform"
    load = "load"
    iterative_tree = "iterative_tree"
    data_anonymization = "data_anonymization"


class MinimizerConfig(ABC):
    minimizer: Minimizer

    def __str__(self) -> str:
        ret = ""
        for key, value in self.__annotations__.items():
            ret += f"{key}={getattr(self, key)},"
        return ret[:-1]

    # def __str__(self) -> str:
    #     ret = ""
    #     for key, value in self.__dict__.items():
    #         if key != "minimizer":
    #             ret += f"{key}={value},"
    #     return ret[:-1]


class ClassifierMinimizerConfig(MinimizerConfig):
    clf_config: ClassifierConfig
    adv_config: AdversaryConfig
    device: str
    freeze_feature: List[int]
    batch_size: int

    def __init__(self, config: Bunch, eval_config: EvaluatorConfig) -> None:
        self.clf_config = config.get(
            "clf_config",
            eval_config.clf_configs[0]
            if len(eval_config.clf_configs) > 0
            else ClassifierConfig(Bunch({})),
        )

        self.adv_config = config.get(
            "adv_config",
            eval_config.adv_configs[0]
            if len(eval_config.adv_configs) > 0
            else AdversaryConfig(Bunch({})),
        )

        self.device = config.get("device", "cpu")
        self.freeze_feature = config.get("freeze_feature", [])
        self.batch_size = config.get("batch_size", 256)


class AdvMinimizerConfig(ClassifierMinimizerConfig):
    advtrain_max_buckets: int
    advtrain_n_epochs: int
    advtrain_inner_steps: int
    advtrain_weight: float

    def __init__(self, config: Bunch, eval_config: EvaluatorConfig):
        super().__init__(config, eval_config)
        self.minimizer = Minimizer("advtrain")
        self.advtrain_max_buckets = config.get("advtrain_max_buckets", 5)
        self.advtrain_n_epochs = config.get("advtrain_n_epochs", 20)
        self.advtrain_inner_steps = config.get("advtrain_inner_steps", 1)
        self.advtrain_weight = config.get("advtrain_weight", 0.0)

    # def __str__(self) -> str:
    #     ret = ""
    #     for key, value in self.__annotations__.items():
    #         ret += f"{key}={getattr(self, key)},"
    #     return ret[:-1]


class MutualInfMinimizerConfig(ClassifierMinimizerConfig):
    mi_max_buckets: int
    mi_n_epochs: int
    mi_weight: float

    def __init__(self, config: Bunch, eval_config: EvaluatorConfig):
        super().__init__(config, eval_config)
        self.minimizer = Minimizer("mi")
        self.mi_max_buckets = config.get("mi_max_buckets", 5)
        self.mi_n_epochs = config.get("mi_n_epochs", 20)
        self.mi_weight = config.get("mi_weight", 0.0)


class IBMMinimizerConfig(ClassifierMinimizerConfig):
    ibm_max_tree_depth: Optional[int]
    ibm_target: int
    timeout: Optional[int]

    def __init__(self, config: Bunch, eval_config: EvaluatorConfig):
        super().__init__(config, eval_config)
        self.minimizer = Minimizer("ibm")
        self.ibm_max_tree_depth = config.get("ibm_max_tree_depth", 10)
        self.ibm_target = config.get("ibm_target", 0.7)
        self.timeout = config.get("timeout", None)


class IterativeMinimizerConfig(ClassifierMinimizerConfig):
    iterative_target: float
    iterative_init_buckets: int
    iterative_fix_wd: bool
    timeout_per_feature: Optional[int]
    timeout: Optional[int]

    def __init__(self, config: Bunch, eval_config: EvaluatorConfig):
        super().__init__(config, eval_config)
        self.minimizer = Minimizer("iterative")
        self.iterative_target = config.get("iterative_target", 0.8)
        self.iterative_init_buckets = config.get("iterative_init_buckets", 4)
        self.iterative_fix_wd = config.get("iterative_fix_wd", False)
        self.timeout_per_feature = config.get("timeout_per_feature", None)
        self.timeout = config.get("timeout", None)


class FeatSelMinimizerConfig(MinimizerConfig):
    featsel_k: int
    method: str

    def __init__(self, config: Bunch):
        self.minimizer = Minimizer("featsel")
        self.featsel_k = config.get("featsel_k", 2)
        self.method = config.get("method", "anova")


class TreeMinimizerConfig(MinimizerConfig):
    tree_max_leaf_nodes: int
    tree_min_sample_leaf: int
    tree_alpha: float
    min_bucket_threshold: float
    initial_split_factor: float

    def __init__(self, config: Bunch):
        self.minimizer = Minimizer("tree")
        self.tree_max_leaf_nodes = config.get("tree_max_leaf_nodes", 20)
        self.tree_min_sample_leaf = config.get("tree_min_sample_leaf", 100)
        self.tree_alpha = config.get("tree_alpha", 0.7)
        self.min_bucket_threshold = config.get("min_bucket_threshold", 0.0)
        self.initial_split_factor = config.get("initial_split_factor", 1.0)


class ForestMinimizerConfig(TreeMinimizerConfig):
    forest_n_trees: int

    def __init__(self, config: Bunch):
        super().__init__(config)
        self.minimizer = Minimizer("forest")
        self.forest_n_trees = config.get("forest_n_trees", 10)


class IterativeTreeMinimizerConfig(TreeMinimizerConfig):
    iter_method: str

    def __init__(self, config: Bunch):
        super().__init__(config)
        self.minimizer = Minimizer("iterative_tree")
        self.iter_method = config.get("iter_method", "bottom_up")
        self.eps = config.get("eps", 0.1)

    def __str__(self) -> str:
        ret = ""
        for key, value in super().__annotations__.items():
            ret += f"{key}={getattr(self, key)},"
        ret += f"eps={self.eps},"
        return ret[:-1]


class UniformMinimizerConfig(MinimizerConfig):
    uniform_buckets: int
    freeze_feature: List[int]

    def __init__(self, config: Bunch):
        self.minimizer = Minimizer("uniform")
        self.uniform_buckets = config.get("uniform_buckets", 5)
        self.freeze_feature = config.get("freeze_feature", [])


class DataAnonymizationMinimizerConfig(MinimizerConfig):
    anonymization_type: str
    anonymization_k: int

    def __init__(self, config: Bunch):
        self.minimizer = Minimizer("data_anonymization")
        self.anonymization_type = config.get("anonymization_type", "k-anonymity")
        assert self.anonymization_type in ["k-anonymity", "l-diversity", "t-closeness"]
        self.anonymization_k = config.get("anonymization_k", 2)


class LoadMinimizerConfig(MinimizerConfig):
    load_bucketization_path: Optional[str]

    def __init__(self, config: Bunch):
        self.minimizer = Minimizer("load")
        self.load_bucketization_path = config.get("load_bucketization_path", None)

    def __str__(self) -> str:
        rel_parts = self.load_bucketization_path.split("/")[-1]
        rel_parts = ".".join(rel_parts.split(".")[:-1])

        return f"load_buck={rel_parts}"


class LoadMultiMinimizerConfig(MinimizerConfig):
    load_bucketization_folder: Optional[str]

    def __init__(self, config: Bunch):
        self.minimizer = Minimizer("load")
        self.load_bucketization_folder = config.get("load_bucketization_folder", None)

    def __str__(self) -> str:
        return f"load_bucketization_folder={self.load_bucketization_folder[:3]}"


class ClassifierType(Enum):
    NORMAL = "normal"
    DP = "dp"


class ClassifierConfig:
    clf_type: ClassifierType
    clf_model: str
    clf_epochs: int
    clf_lr: float
    clf_weight_decay: float
    clf_tune_wd: bool
    clf_dp_noise: float

    def __init__(self, config: Bunch):
        self.clf_type = ClassifierType(config.get("clf_type", "normal"))
        self.clf_model = config.get("clf_model", "mlp2")
        self.clf_epochs = config.get("clf_epochs", 20)
        self.clf_lr = config.get("clf_lr", 1e-2)
        self.clf_weight_decay = config.get("clf_weight_decay", 0.0)
        self.clf_tune_wd = config.get("clf_tune_wd", True)
        self.clf_dp_noise = config.get("clf_dp_noise", 1.0)


class AdversaryType(Enum):
    RECOVERY = "recovery"
    MULTIRECOVERY = "multirecovery"
    LINKAGE = "linkage"
    OUTLIER = "outlier"
    LEAKNONSENSITIVE = "nonsensitive"
    LEAKLEAVEONEOUT = "oneout"
    ITERATIVE = "iterative"


class LinkageAlgorithm(Enum):
    MATCHING = "matching"
    SAMPLING = "sampling"
    MOSTLIKELY = "mostlikely"
    RANDOM = "random"


class AdversaryConfig:
    adv_type: AdversaryType
    clf_config: ClassifierConfig
    use_only_buck_data: bool  # Train adversary only on data from bucketization
    only_predict_seen: bool  # Only predict x values for z which have been seen in training
    use_label: bool  # Use label in training
    requires_original_data: bool
    linkage_algorithm: Optional[LinkageAlgorithm]
    linkage_a_features: List[int]
    linkage_b_features: List[int]

    def __init__(self, config: Bunch):
        self.adv_type = AdversaryType(config.get("adv_type", "recovery"))
        self.clf_config = ClassifierConfig(config)
        self.use_only_buck_data = config.get("use_only_buck_data", True)
        self.only_predict_seen = config.get("only_predict_seen", True)
        self.use_label = config.get("use_label", False)
        # TODO Clean up this logic
        self.requires_original_data = (
            self.adv_type
            in [
                AdversaryType.RECOVERY,
                AdversaryType.LINKAGE,
                AdversaryType.LEAKLEAVEONEOUT,
                AdversaryType.LEAKNONSENSITIVE,
                AdversaryType.ITERATIVE,
            ]
            or self.only_predict_seen
        )
        self.linkage_algorithm = (
            LinkageAlgorithm(config.get("linkage_algorithm", ""))
            if "linkage_algorithm" in config
            else None
        )
        self.linkage_a_features = config.get("linkage_a_features", [])
        self.linkage_b_features = config.get("linkage_b_features", [])


def get_minimizer_config(
    config: Bunch, eval_config: EvaluatorConfig
) -> List[MinimizerConfig]:
    minimizer = config.get("minimizer", "tree")
    cfg: MinimizerConfig
    if minimizer == "advtrain":
        cfg = AdvMinimizerConfig(config, eval_config)
    elif minimizer == "mi":
        cfg = MutualInfMinimizerConfig(config, eval_config)
    elif minimizer == "ibm":
        cfg = IBMMinimizerConfig(config, eval_config)
    elif minimizer == "iterative":
        cfg = IterativeMinimizerConfig(config, eval_config)
    elif minimizer == "featsel":
        cfg = FeatSelMinimizerConfig(config)
    elif minimizer == "tree":
        cfg = TreeMinimizerConfig(config)
    elif minimizer == "forest":
        cfg = ForestMinimizerConfig(config)
    elif minimizer == "iterative_tree":
        cfg = IterativeTreeMinimizerConfig(config)
    elif minimizer == "uniform":
        cfg = UniformMinimizerConfig(config)
    elif minimizer == "load":
        cfg = LoadMinimizerConfig(config)
    elif minimizer == "load_multi":
        configs = []
        files = os.listdir(config["load_bucketization_folder"])
        files = [
            os.path.join(config["load_bucketization_folder"], f)
            for f in files
            if f.endswith(".txt")
        ]
        # Load all bucketizations
        for f in files:
            cfg = LoadMinimizerConfig(Bunch({"load_bucketization_path": f}))
            configs.append(cfg)

        cfgs: List[MinimizerConfig] = []
        for cfg in configs:
            cfgs.extend(create_combs_from_list(cfg, ["freeze_feature"], config))
        return cfgs

    elif minimizer == "data_anonymization":
        cfg = DataAnonymizationMinimizerConfig(config)
    else:
        raise ValueError(f"Unknown minimizer: {minimizer}")
    cfgs: List[MinimizerConfig] = create_combs_from_list(cfg, ["freeze_feature"], config)  # type: ignore[assignment]
    return cfgs


class EvaluatorConfig:
    clf_configs: List[ClassifierConfig]
    adv_configs: List[AdversaryConfig]

    def __init__(self, config: Bunch):
        self.clf_configs: List[ClassifierConfig] = []
        self.adv_configs: List[AdversaryConfig] = []
        for clf in config.get("classifiers", Bunch()):
            self.clf_configs.append(ClassifierConfig(Bunch(**clf)))
        for adv in config.get("adversaries", Bunch()):
            self.adv_configs.append(AdversaryConfig(Bunch(**adv)))

    def __str__(self) -> str:
        ret = "Classifiers:\n"
        for i, clf in enumerate(self.clf_configs):
            ret += f" Classifier {i}:\n"
            ret += f"  Model: {clf.clf_model} Epochs: {clf.clf_epochs} LR: {clf.clf_lr} WD: {clf.clf_weight_decay}\n"
        for i, adv in enumerate(self.adv_configs):
            ret += f"Adversary {i}:\n"
            ret += f"  Model: {adv.clf_config.clf_model} Epochs: {adv.clf_config.clf_epochs} LR: {adv.clf_config.clf_lr} WD: {adv.clf_config.clf_weight_decay}\n"
        return ret


class Config:
    neptune_config: NeptuneConfig
    dataset_config: DatasetConfig
    eval_config: EvaluatorConfig
    min_config: MinimizerConfig

    seed: int
    device: str
    out_dir: str
    compute_guarantees: bool
    get_clf_upper_bound: bool
    fairness_sens_col: Optional[int]
    num_workers: int
    logger_level: str

    def __init__(self, config: Bunch):
        # Bunch defaults are just for typing

        self.neptune_config = config.get("neptune_config", NeptuneConfig(config))
        self.dataset_config = config.get("dataset_configs", DatasetConfig(config))
        self.eval_config = config.get("eval_configs", EvaluatorConfig(config))
        self.min_config = config.get(
            "min_configs", get_minimizer_config(config, self.eval_config)
        )

        self.seed = config.get("seed", 100)
        self.device = config.get("device", "cpu")
        self.out_dir = config.get("out_dir", "out/")
        self.compute_guarantees = config.get("compute_guarantees", False)
        self.get_clf_upper_bound = config.get("get_clf_upper_bound", False)
        self.fairness_sens_col = config.get("fairness_sens_cols", None)
        self.num_workers = config.get("num_workers", 1)
        self.logger_level = config.get("logger_level", "INFO")

        # TODO Create DeviceFull minimizer subclass
        if isinstance(self.min_config, ClassifierMinimizerConfig):
            self.min_config.device = self.device

    def __str__(self) -> str:
        ret = "CONFIG".center(80, "-") + "\n"
        ret += f"neptune_config:\n{str(self.neptune_config)}"
        ret += f"dataset_config:\n{str(self.dataset_config)}"
        ret += f"eval_config:\n{str(self.eval_config)}"
        if isinstance(self.min_config, List):
            for i, cfg in enumerate(self.min_config):
                ret += f"min_config {i}: {str(cfg)}\n"
        else:
            ret += f"min_config:\n{str(self.min_config)}"
        ret += "\n"
        for k, v in self.__dict__.items():
            if k not in [
                "neptune_config",
                "dataset_config",
                "eval_config",
                "min_config",
            ]:
                ret += f"{k}: {v}\n"
        ret += "CONFIG END".center(80, "-") + "\n"

        return ret


class MetaConfig:
    neptune_config: NeptuneConfig
    dataset_configs: List[DatasetConfig]
    eval_configs: List[EvaluatorConfig]
    min_configs: List[MinimizerConfig]

    seed: int
    device: str
    out_dir: str
    continual_release: bool
    compute_guarantees: bool
    get_clf_upper_bound: bool
    fairness_sens_col: Optional[int]
    connect_arguments: bool
    num_workers: int
    logger_level: str

    def __init__(self, config: Bunch):
        self.neptune_config = NeptuneConfig(config.get("neptune", Bunch()))
        self.dataset_configs = []
        for dataset in config.get("datasets", []):
            meta_dataset_config = DatasetConfig(Bunch(**dataset))
            ds: List[DatasetConfig] = create_combs_from_list(
                meta_dataset_config, ["freeze_feature"], dataset
            )
            self.dataset_configs.extend(ds)

        self.eval_configs = []
        for eval in config.get("evaluations", []):
            self.eval_configs.append(EvaluatorConfig(Bunch(**eval)))

        assert len(self.eval_configs) > 0, "No evaluations specified"
        self.min_configs = []
        for min in config.get("minimizers", []):
            self.min_configs.extend(
                get_minimizer_config(Bunch(**min), self.eval_configs[0])
            )

        self.seed = config.get("seed", 100)
        self.device = config.get("device", "cpu")
        self.out_dir = config.get("out_dir", "out/")
        self.continual_release = config.get("continual_release", False)
        self.compute_guarantees = config.get("compute_guarantees", False)
        self.get_clf_upper_bound = config.get("get_clf_upper_bound", False)
        self.fairness_sens_col = config.get("fairness_sens_cols", None)
        self.num_workers = config.get("num_workers", 1)
        self.logger_level = config.get("logger_level", "INFO")


def make_config(**config: Any) -> List[Config]:
    bunch = Bunch(**config)
    meta_config = MetaConfig(bunch)
    proto_config = Config(Bunch())
    freeze_feats = []
    if meta_config.continual_release:
        freeze_feats = ["min_configs"]
    configs: List[Config] = create_combs_from_list(meta_config, freeze_feats, bunch, proto_config)  # type: ignore
    return configs
