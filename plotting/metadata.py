metadata = {
    "ACSIncome": (0.644, 0.8303, 0.475),
    "ACSEmployment": (0.571, 0.814, 0.725),
    "ACSTravelTime": (0.545, 0.689, 0.603),
    "ACSPublicCoverage": (0.701, 0.769, 0.757),
    "health_no_transfer": (0.8016, 0.882, 0.605),
    "health_pruned_no_transfer": (0.8006, 0.865, 0.510),
    "health_pruned_transfer": (0.799, 0.815, 0.691),
    "crime": (0.503, 0.801335, 0.3583),
    "loan": (0.8156, 0.990, 0.55085),
}

global_methods = [
    "tree",
    "uniform",
    "mi",
    "advtrain",
    "iterative",
    "ibm",
    "featsel",
    "anova",
    "chi2",
]

adv_methods = ["nonsensitive-ops", "oneout-ops", "iterative-ops"]

method2label = {
    "tree": "PAT",
    "uniform": "Uniform",
    "ibm": "Apt",
    "mi": "MutualInf",
    "advtrain": "AdvTrain",
    "iterative": "Iterative",
    "featsel": "FeatSel",
    "tree-ops": "PAT",
    "uniform-ops": "Uniform",
    "ibm-ops": "Apt",
    "mi-ops": "MutualInf",
    "advtrain-ops": "AdvTrain",
    "iterative-ops": "Partial-Sensitive Knowledge (A5)",
    "featsel-ops": "FeatSel",
    "nonsensitive-ops": "Non-Sensitive Knowledge (A3)",
    "oneout-ops": "Leave-one-out (A4)",
    "[8]": "SEX",
    "[3]": "MAR",
    "disc": "All discrete",
    "100%": "100% (A1)",
}

settings = ["health_pruned_no_transfer", "health_pruned_transfer"]
