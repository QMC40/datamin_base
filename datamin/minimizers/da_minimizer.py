from typing import Any, Dict, List, Optional, Tuple, Union

import anonypy
import neptune.new as neptune
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from datamin.bucketization import Bucketization
from datamin.dataset import FolktablesDataset
from datamin.minimizers.abstract_minimizer import AbstractMinimizer
from datamin.utils.config import DataAnonymizationMinimizerConfig
from datamin.utils.logging_utils import CLogger, get_print_logger


# Fit a Minimizer
class DataAnonymizationMinimizer(AbstractMinimizer):
    def __init__(
        self,
        config: DataAnonymizationMinimizerConfig,
        logger: CLogger,
        run: Optional[neptune.Run] = None,
    ):
        super(DataAnonymizationMinimizer, self).__init__()
        self.run = run
        self.anonymization_type = config.anonymization_type
        self.anonymization_k = config.anonymization_k
        if logger is None:
            logger = get_print_logger("DataAnonymization-Logger")
        self.logger = logger

    # flake8: noqa: C901
    def fit(self, dataset: FolktablesDataset) -> None:
        self.dataset = dataset

        # Simply take (buck, val, test) folds and extract sensitive features
        # (used to do a lot more before it was synced with the main dataset)
        processed_data = self._process_data(dataset)

        # Extract Bucketization
        self.bucketization = Bucketization(dataset)

        # dte_items = list(decision_tree_encoding.items())

        # for i in dataset.cont_feats:
        #     sz = len(dte_items[i][1]) + 1
        #     self.bucketization.add_cont(
        #         name=dte_items[i][0], sz=sz, borders=dte_items[i][1]
        #     )
        # for j in dataset.disc_feats:
        #     sz = len(np.unique(dte_items[j][1]))
        #     self.bucketization.add_disc(
        #         name=dte_items[j][0], sz=sz, mapping=dte_items[j][1]
        #     )

        # Optimization: reduce the number of buckets

    def get_bucketization(self) -> Bucketization:
        return self.bucketization

    def _process_data(self, dataset: FolktablesDataset) -> Dict[str, Any]:

        # test_Data = [
        #     [6, "1", "1", "1", 20],
        #     [6, "1", "1", "1", 30],
        #     [8, "2", "2", "1", 50],
        #     [8, "1", "2", "2", 35],
        #     [8, "2", "3", "0", 45],
        #     [4, "2", "3", "2", 20],
        #     [4, "1", "3", "2", 20],
        #     [2, "1", "3", "3", 22],
        #     [2, "2", "3", "2", 32],
        # ]
        # test_columns = ["col1", "col2", "col3", "col4", "col5"]
        # test_categorical = set(("col2", "col3", "col4"))
        # test_df = pd.DataFrame(data=test_Data, columns=test_columns)

        # for name in test_categorical:
        #     test_df[name] = test_df[name].astype("category")

        # feature_columns = ["col1", "col2", "col3"]
        # sensitive_column = "col4"

        # p = anonypy.Preserver(test_df, feature_columns, sensitive_column)
        # unique_rows, rows = p.anonymize_k_anonymity(k=2)

        # dfn = pd.DataFrame(unique_rows)
        # dfn = dfn.applymap(lambda x: x[0] if isinstance(x, list) else x)

        # for col in dfn.columns:
        #     if col == "count":
        #         continue
        #     unique_vals = sorted(list(dfn[col].unique()))
        #     rows[col] = rows[col].map(lambda x: unique_vals.index(x[0] if isinstance(x, list) else x))

        # print(dfn)

        columns = [f"col_{x}" for x in list(range(dataset.tot_feats + 1))]
        categorical = set([f"col_{x}" for x in dataset.disc_feats])

        data = dataset.X_train_buck_orig
        y_data = dataset.y_train_buck_orig

        data = np.concatenate((data, y_data.reshape(-1, 1)), axis=1)

        feature_columns = columns
        sensitive_column = f"col_{dataset.tot_feats}"  # the y_label

        # Put into correct pandas format
        df = pd.DataFrame(data=data, columns=columns)

        for name in categorical:
            df[name] = df[name].astype("category")

        p = anonypy.Preserver(df, feature_columns, sensitive_column)
        unique_rows, rows = p.anonymize_k_anonymity(k=self.anonymization_k)

        dfn = pd.DataFrame(unique_rows)
        dfn = dfn.applymap(lambda x: x[0] if isinstance(x, list) else x)

        for col in dfn.columns:
            if col == "count":
                continue
            unique_vals = sorted(list(dfn[col].unique()))
            rows[col] = rows[col].map(
                lambda x: unique_vals.index(x[0] if isinstance(x, list) else x)
            )

        print(dfn)

        X_train = dataset.X_train_buck_orig
        y_train = dataset.y_train_buck_orig
        X_val = dataset.X_val_orig
        X_test = dataset.X_test_orig

        # type: ignore[call-overload]
        s_train: np.ndarray = X_train[:, dataset.sens_feats]
        # type: ignore[call-overload]
        s_val: np.ndarray = X_val[:, dataset.sens_feats]
        # type: ignore[call-overload]
        s_test: np.ndarray = X_test[:, dataset.sens_feats]

        prep_data = {
            "train": (X_train, s_train.astype(np.int), y_train.astype(np.int)),  # type: ignore[attr-defined]
            "val": (X_val, s_val.astype(np.int), dataset.y_val.astype(np.int)),  # type: ignore[attr-defined]
            "test": (X_test, s_test.astype(np.int), dataset.y_test.astype(np.int)),  # type: ignore[attr-defined]
        }
        return prep_data
