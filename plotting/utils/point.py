# Point datastructure


class Point:

    setting_fields = [
        "method",
        "tree_max_leaf_nodes",
        "tree_min_sample_leaf",
        "tree_alpha",
        "min_bucket_threshold",
        "initial_split_factor",
        "dp_noise",
        "dataset_name",
        "dataset_state",
        "dataset_year",
        "sens_feats",
        "dataset_val_percentage",
    ]

    def __init__(self, dict):
        if isinstance(dict, Point):
            self.dict = dict.dict
        else:
            self.dict = dict

    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, value):
        self.dict[key] = value

    def __str__(self):
        return str(self.dict)

    def red_str(self):
        point_str = ""
        for k, v in self.dict.items():
            if k.split("_")[0] in [
                "dataset",
                "clf",
                "adv",
                "quantiles",
                "sens",
                "hash",
            ]:
                continue
            point_str += f"{k}={v}, "
        return point_str[:-2]

    def __repr__(self):
        return str(self.dict)

    def __len__(self):
        return len(self.dict)

    def __iter__(self):
        return iter(self.dict)

    def __contains__(self, key):
        return key in self.dict

    def eq_except(self, other, key):
        # Checks whether dicts are equal except for the given key
        for k in self.setting_fields:
            if k != key and self.dict[k] != other.dict[k]:
                return False
        return True
