from os import path
from urllib import request
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .abstract_dataset import AbstractDataset


class CrimeDataset(AbstractDataset):
    column_names = [
        "communityname",
        "state",
        "countyCode",
        "communityCode",
        "fold",
        "population",
        "householdsize",
        "racepctblack",
        "racePctWhite",
        "racePctAsian",
        "racePctHisp",
        "agePct12t21",
        "agePct12t29",
        "agePct16t24",
        "agePct65up",
        "numbUrban",
        "pctUrban",
        "medIncome",
        "pctWWage",
        "pctWFarmSelf",
        "pctWInvInc",
        "pctWSocSec",
        "pctWPubAsst",
        "pctWRetire",
        "medFamInc",
        "perCapInc",
        "whitePerCap",
        "blackPerCap",
        "indianPerCap",
        "AsianPerCap",
        "OtherPerCap",
        "HispPerCap",
        "NumUnderPov",
        "PctPopUnderPov",
        "PctLess9thGrade",
        "PctNotHSGrad",
        "PctBSorMore",
        "PctUnemployed",
        "PctEmploy",
        "PctEmplManu",
        "PctEmplProfServ",
        "PctOccupManu",
        "PctOccupMgmtProf",
        "MalePctDivorce",
        "MalePctNevMarr",
        "FemalePctDiv",
        "TotalPctDiv",
        "PersPerFam",
        "PctFam2Par",
        "PctKids2Par",
        "PctYoungKids2Par",
        "PctTeen2Par",
        "PctWorkMomYoungKids",
        "PctWorkMom",
        "NumKidsBornNeverMar",
        "PctKidsBornNeverMar",
        "NumImmig",
        "PctImmigRecent",
        "PctImmigRec5",
        "PctImmigRec8",
        "PctImmigRec10",
        "PctRecentImmig",
        "PctRecImmig5",
        "PctRecImmig8",
        "PctRecImmig10",
        "PctSpeakEnglOnly",
        "PctNotSpeakEnglWell",
        "PctLargHouseFam",
        "PctLargHouseOccup",
        "PersPerOccupHous",
        "PersPerOwnOccHous",
        "PersPerRentOccHous",
        "PctPersOwnOccup",
        "PctPersDenseHous",
        "PctHousLess3BR",
        "MedNumBR",
        "HousVacant",
        "PctHousOccup",
        "PctHousOwnOcc",
        "PctVacantBoarded",
        "PctVacMore6Mos",
        "MedYrHousBuilt",
        "PctHousNoPhone",
        "PctWOFullPlumb",
        "OwnOccLowQuart",
        "OwnOccMedVal",
        "OwnOccHiQuart",
        "OwnOccQrange",
        "RentLowQ",
        "RentMedian",
        "RentHighQ",
        "RentQrange",
        "MedRent",
        "MedRentPctHousInc",
        "MedOwnCostPctInc",
        "MedOwnCostPctIncNoMtg",
        "NumInShelters",
        "NumStreet",
        "PctForeignBorn",
        "PctBornSameState",
        "PctSameHouse85",
        "PctSameCity85",
        "PctSameState85",
        "LemasSwornFT",
        "LemasSwFTPerPop",
        "LemasSwFTFieldOps",
        "LemasSwFTFieldPerPop",
        "LemasTotalReq",
        "LemasTotReqPerPop",
        "PolicReqPerOffic",
        "PolicPerPop",
        "RacialMatchCommPol",
        "PctPolicWhite",
        "PctPolicBlack",
        "PctPolicHisp",
        "PctPolicAsian",
        "PctPolicMinor",
        "OfficAssgnDrugUnits",
        "NumKindsDrugsSeiz",
        "PolicAveOTWorked",
        "LandArea",
        "PopDens",
        "PctUsePubTrans",
        "PolicCars",
        "PolicOperBudg",
        "LemasPctPolicOnPatr",
        "LemasGangUnitDeploy",
        "LemasPctOfficDrugUn",
        "PolicBudgPerPop",
        "murders",
        "murdPerPop",
        "rapes",
        "rapesPerPop",
        "robberies",
        "robbbPerPop",
        "assaults",
        "assaultPerPop",
        "burglaries",
        "burglPerPop",
        "larcenies",
        "larcPerPop",
        "autoTheft",
        "autoTheftPerPop",
        "arsons",
        "arsonsPerPop",
        "ViolentCrimesPerPop",
        "nonViolPerPop",
    ]

    def __init__(
        self, split, args=None, normalize=True, p_test=0.0, p_val=0.0, preprocess=True
    ):
        super().__init__("crime", split, p_test, p_val)

        data_file = path.join(self.data_dir, "communities.data")

        self.X_train = None
        self.X_val = None
        self.X_test = None

        if not path.exists(data_file):
            request.urlretrieve(
                "http://archive.ics.uci.edu/ml/machine-learning-databases/00211/CommViolPredUnnormalizedData.txt",
                data_file,
            )

        dataset = pd.read_csv(
            data_file, sep=",", header=None, names=CrimeDataset.column_names
        )
        # remove features that are not predictive
        dataset.drop(
            ["communityname", "countyCode", "communityCode", "fold"],
            axis=1,
            inplace=True,
        )
        # remove all other potential goal variables
        dataset.drop(
            [
                "murders",
                "murdPerPop",
                "rapes",
                "rapesPerPop",
                "robberies",
                "robbbPerPop",
                "assaults",
                "assaultPerPop",
                "burglaries",
                "burglPerPop",
                "larcenies",
                "larcPerPop",
                "autoTheft",
                "autoTheftPerPop",
                "arsons",
                "arsonsPerPop",
                "nonViolPerPop",
            ],
            axis=1,
            inplace=True,
        )
        dataset.replace(to_replace="?", value=np.nan, inplace=True)
        # drop rows with missing labels
        dataset.dropna(axis=0, subset=["ViolentCrimesPerPop"], inplace=True)
        # drop columns with missing values
        dataset.dropna(axis=1, inplace=True)
        features, labels = (
            dataset.drop("ViolentCrimesPerPop", axis=1),
            dataset["ViolentCrimesPerPop"],
        )

        continuous_vars = []
        self.categorical_columns = []

        self.protected_unique = 2
        protected = np.less(
            features["racePctWhite"] / 5,
            features["racepctblack"]
            + features["racePctAsian"]
            + features["racePctHisp"],
        )

        # Remove the protected attributes
        features = features.drop(
            ["racePctWhite", "racepctblack", "racePctAsian", "racePctHisp"], axis=1
        )

        for col in features.columns:
            if features[col].isnull().sum() > 0:
                features.drop(col, axis=1, inplace=True)
            else:
                if features[col].dtype == np.object:
                    self.categorical_columns += [col]
                else:
                    continuous_vars += [col]

        for col in features.columns:
            if features[col].dtype == "object":
                features[col] = features[col].astype("category")
                features[col] = features[col].cat.codes

        # Instantiate scaler
        scaler = MinMaxScaler()
        features[continuous_vars] = scaler.fit_transform(features[continuous_vars])

        # features = pd.get_dummies(features, columns=self.categorical_columns, prefix_sep='=')
        self.continuous_columns = [
            features.columns.get_loc(var) for var in continuous_vars
        ]

        self.column_ids = {col: idx for idx, col in enumerate(features.columns)}

        #########################
        self.ft_pos: Dict[str, Union[int, Tuple[int, int]]] = self.column_ids  # type: ignore[assignment]
        ctr = 0
        for i, feat in enumerate(features.columns):
            if i in self.continuous_columns:
                self.ft_pos[feat] = ctr
                ctr += 1
            else:
                self.ft_pos[feat] = (ctr, ctr + len(features[feat].unique()))
                ctr += len(features[feat].unique())

        self.selected_cont_features = self.continuous_columns
        self.selected_feature_names = features.columns.tolist()

        ##########################

        features = torch.tensor(features.values.astype(np.float32), device=self.device)
        labels = torch.tensor(labels.values.astype(np.float32), device=self.device)
        protected = torch.tensor(protected.values.astype(np.bool), device=self.device)

        # binarize labels
        labels = labels < labels.median()

        if p_test > 1e-6:
            (
                X_train,
                self.X_test,
                y_train,
                self.y_test,
                protected_train,
                self.protected_test,
            ) = train_test_split(
                features, labels, protected, test_size=self.p_test, random_state=0
            )
        else:
            self.X_train, self.y_train, self.protected_train = (
                features,
                labels,
                protected,
            )
        if p_val > 1e-6:
            (
                self.X_train,
                self.X_val,
                self.y_train,
                self.y_val,
                self.protected_train,
                self.protected_val,
            ) = train_test_split(
                X_train, y_train, protected_train, test_size=self.p_val, random_state=0
            )
        else:
            self.X_train, self.y_train, self.protected_train = (
                features,
                labels,
                protected,
            )

        # Select all
        selected_column_ids = np.arange(0, self.X_train.shape[-1], 1)

        self.X_train = self.X_train[:, selected_column_ids]
        self.X_train = self.X_train.to(args.device)
        self.y_train = self.y_train.to(args.device)
        self.protected_train = self.protected_train.to(args.device)

        if p_val > 1e-6:
            self.X_val = self.X_val[:, selected_column_ids]
            self.X_val = self.X_val.to(args.device)
            self.y_val = self.y_val.to(args.device)
            self.protected_val = self.protected_val.to(args.device)

        if p_test > 1e-6:
            self.X_test = self.X_test[:, selected_column_ids]
            self.X_test = self.X_test.to(args.device)
            self.y_test = self.y_test.to(args.device)
            self.protected_test = self.protected_test.to(args.device)

        self._assign_split()


def load_crime(
    split, args=None, normalize=True, p_test=0.0, p_val=0.0, preprocess=True
):
    args = argparse.Namespace()
    args.load = True
    args.device = "cpu"

    train_dataset = CrimeDataset(split, args, normalize, p_test, p_val, preprocess)

    c_train = train_dataset.protected_train.numpy()
    X_train = train_dataset.X_train.numpy()
    y_train = train_dataset.y_train.numpy()

    # c_test = train_dataset.protected_test.numpy()
    # X_test = train_dataset.X_test.numpy()
    # y_test = train_dataset.y_test.numpy()

    # Append the original sensitive feature""
    X_train = np.concatenate(
        (X_train, c_train.astype(np.float32).reshape((-1, 1))), axis=1
    )

    data: Dict[str, Any] = {}
    data["train"] = [X_train, y_train]
    # data['test'] = [ X_test, c_test, y_test ]
    data["ft_pos"] = train_dataset.ft_pos
    data["feature_names"] = train_dataset.selected_feature_names + ["PROTECTED_ATTR"]
    data["cont_features"] = train_dataset.selected_cont_features

    return data
