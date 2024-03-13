import numpy as np
from typing import Dict, Any
import pandas as pd
from typing import Tuple, Union
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from datamin.datasets.selector.interface import FeatureSelectionApp


def replace_nan(df):
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    df[numerical_cols] = df[numerical_cols].fillna(0)
    df[categorical_cols] = df[categorical_cols].fillna("NA")


def load_csv(path: str) -> Dict[str, Any]:
    df = pd.read_csv(path)

    # Check which columns have NaNs
    nan_columns = df.columns[df.isna().any()].tolist()
    # Check which columns are categorical
    categorical = df.select_dtypes(include="object").columns.tolist()
    categorical_index = [df.columns.get_loc(c) for c in categorical]

    sample_size = 100 if len(df) > 100 else len(df)

    # Get relevant features and target
    while True:
        app = FeatureSelectionApp(
            feature_names=df.columns.to_list(),
            features=df.to_numpy()[:sample_size],
            categorical=categorical_index,
        )
        app.run()
        if app.reset:
            continue
        else:
            break

    protected_indices = app.selection_list.selected
    categorical_indices = app.categorical_list.selected

    target_name = app.target_name.value
    if target_name == "":
        target_name = df.columns.to_list()[-1]
    nan_bahaviour = app.nan_selector.value

    # Map indices to column names
    categorical_features = [df.columns[i] for i in categorical_indices]
    continuous = [
        col for col in df.columns if col not in categorical_features + [target_name]
    ]

    # Feature df
    features = df.drop(columns=[target_name])
    column_names = features.columns.to_list()
    target = df[target_name]

    # Handle NaNs
    if nan_bahaviour == "remove_row":
        features.dropna(inplace=True)
        target = target[features.index]
    elif nan_bahaviour == "remove_column":
        features.drop(nan_columns, axis=1, inplace=True)
    elif nan_bahaviour == "replace_mean":
        features.fillna(features.mean(), inplace=True)
    elif nan_bahaviour == "replace_median":
        features.fillna(features.median(), inplace=True)

    # Shuffle data and labels
    features, target = shuffle(features, target, random_state=0)

    # Get the indices of all continuous features
    continuous_indices = [features.columns.get_loc(c) for c in continuous]

    # Transform all categorical features to numbers but not continuous features
    for col in features.columns:
        if col not in continuous:
            features[col] = features[col].astype("category")
            features[col] = features[col].cat.codes

    # Normalize continuous features to 0-1 range robustly
    # Instantiate scaler
    scaler = MinMaxScaler()
    features[continuous] = scaler.fit_transform(features[continuous])

    # Convert to numpy
    X = features.to_numpy()  # feat_oh
    y = target.to_numpy()

    # Get sizes of one-hot encoded features
    feature_sizes = [len(features[col].unique()) for col in column_names[:-1]]

    # Get dict of feature names to indices
    feature_names = features.columns.to_list()

    ft_pos: Dict[str, Union[int, Tuple[int, int]]] = {}  # type: ignore[assignment]
    ctr = 0
    for i, col in enumerate(column_names[:-1]):
        if col in continuous:
            ft_pos[col] = ctr
            ctr += 1
        else:
            ft_pos[col] = (ctr, ctr + feature_sizes[i])
            ctr += feature_sizes[i]

    data: Dict[str, Any] = {}
    data["train"] = [X, y]
    # data['test'] = [ X_test, c_test, y_test ]
    data["ft_pos"] = ft_pos
    data["feature_names"] = feature_names
    data["cont_features"] = continuous_indices

    return data, protected_indices
