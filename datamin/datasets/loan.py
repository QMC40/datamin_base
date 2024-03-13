import numpy as np
from typing import Dict, Any, Union, Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


def replace_nan(df):
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    df[numerical_cols] = df[numerical_cols].fillna(0)
    df[categorical_cols] = df[categorical_cols].fillna("NA")


def transform_regions(df):
    # Define region states
    regions = {
        "west": ["CA", "OR", "UT", "WA", "CO", "NV", "AK", "MT", "HI", "WY", "ID"],
        "south_west": ["AZ", "TX", "NM", "OK"],
        "south_east": [
            "GA",
            "NC",
            "VA",
            "FL",
            "KY",
            "SC",
            "LA",
            "AL",
            "WV",
            "DC",
            "AR",
            "DE",
            "MS",
            "TN",
        ],
        "mid_west": [
            "IL",
            "MO",
            "MN",
            "OH",
            "WI",
            "KS",
            "MI",
            "SD",
            "IA",
            "NE",
            "IN",
            "ND",
        ],
        "north_east": ["CT", "NY", "PA", "NJ", "RI", "MA", "MD", "VT", "NH", "ME"],
    }

    for region, states in regions.items():
        df.loc[df["addr_state"].isin(states), "addr_state"] = region

    return df


def transform_loan_status(df):
    bad_loan_status = [
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status: Charged Off",
        "In Grace Period",
        "Late (16–30 days)",
        "Late (31–120 days)",
    ]

    df["loan_status"] = df["loan_status"].apply(
        lambda x: "bad" if x in bad_loan_status else "good"
    )

    return df


def load_loan():
    column_names = [
        "loan_amnt",
        "funded_amnt",
        "funded_amnt_inv",
        "term",
        "int_rate",
        "installment",
        "grade",
        "sub_grade",
        "emp_length",
        "home_ownership",
        "annual_inc",
        "verification_status",
        "pymnt_plan",
        "purpose",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "mths_since_last_delinq",
        "mths_since_last_record",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "initial_list_status",
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_rec_int",
        "total_rec_late_fee",
        "last_pymnt_amnt",
        "collections_12_mths_ex_med",
        "policy_code",
        "application_type",
        "acc_now_delinq",
        "chargeoff_within_12_mths",
        "delinq_amnt",
        "tax_liens",
        "hardship_flag",
        "disbursement_method",
        "issue_d",
        "addr_state",
    ]  # zip_code, 'earliest_cr_line'

    target_name = "loan_status"

    df = pd.read_csv("data/loan/accepted_2015.csv")

    # Feature df
    features = df.drop(columns=[target_name])
    target = df[target_name]

    # Transform target not_recom to 0 else 1
    target = target.apply(lambda x: 0 if x == "bad" else 1)

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
