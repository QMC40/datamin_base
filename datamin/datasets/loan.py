import numpy as np
from typing import Dict, Any, Union, Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


def load_loan():
    column_names = [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "Loan_Amount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area",
            "Loan_Status"
            ]

    target_name = "Loan_Status"

    df = pd.read_csv("data/loan/accepted_2015.csv")

    # Feature df
    features = df.drop(columns=[target_name])
    target = df[target_name]

    # Transform target not_recom to 0 else 1
    target = target.apply(lambda x: 0 if x == "N" else 1)

    # Get the indices of all continuous features
    continuous = features.select_dtypes(include=np.number).columns.tolist()
    continuous_indices = [
            features.columns.get_loc(c)
            for c in features.select_dtypes(include=np.number).columns.tolist()
            ]

    # Transform all categorical features to numbers but not continuous features
    for col in features.columns:
        if features[col].dtype == "object":
            features[col] = features[col].astype("category")
            features[col] = features[col].cat.codes

    # Normalize continuous features to 0-1 range robustly

    # Instantiate scaler
    scaler = MinMaxScaler()
    features[continuous] = scaler.fit_transform(features[continuous])

    # for column in continuous:
    #   features[column] = pd.qcut(features[column], q=100, duplicates='drop')

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

    return data

