import argparse
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple

from tinydb import Query, TinyDB


# flake8: noqa: C901
def parse_files_to_db(args: argparse.Namespace):
    dir = args.dir
    # Remove the old database
    if os.path.exists("temp_db.json") and args.clear_db:
        os.remove("temp_db.json")
    db = TinyDB("temp_db.json")

    key_query = Query()

    for root, dirs, files in os.walk(dir):
        # Get Dataset information
        for name in files:
            print(f"working on file: {name}")
            with open(os.path.join(root, name)) as fp:
                # Method_name is the last directory level
                method_name = root.split("/")[-1]
                method_split = method_name.split("_")
                if len(method_split) > 1:
                    method_name = method_split[0]
                    assert method_split[1] == "dp"
                    dp_noise = method_split[2]
                else:
                    dp_noise = 0

                if args.plot_over != "method" and method_name not in args.method:
                    continue
                # Dataset is the second to last directory level
                dataset = root.split("/")[-2]

                dataset_dict = get_dataset_dict_from_str(dataset)

                # Settings are in the file name separated by commata
                settings = ".".join(name.split(".")[:-1]).split(",")

                lines = fp.readlines()
                line_holder: str = ""

                quantile_lines: List[Tuple[float, float]] = []
                feat_lines: List[Dict[str, Tuple[float, ...]]] = []
                feat_baseline: Dict[str, Tuple[float, ...]] = {}
                dp_dict: Dict[str, float] = {}
                on_tab: bool = False
                curr_feat_vals: Dict[str, Tuple[float, ...]] = {}

                for line in lines:
                    if line.startswith("Used ") and line.endswith("buckets in total\n"):
                        vals = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
                        assert len(vals) == 1
                        feat_baseline["num_buckets"] = vals[0]

                    if line.startswith("Adversary uses torch.Size("):
                        vals = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
                        assert len(vals) == 2
                        feat_baseline["num_samples"] = vals[0]

                    if line.startswith("feat="):
                        vals = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
                        feat_name = line.split("=")[1].split(" ")[0][:-1]
                        feat_baseline[feat_name] = tuple([float(v) for v in vals])

                    if line.startswith("\tfeat="):
                        vals = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
                        feat_name = line.split("=")[1].split(" ")[0][:-1]
                        curr_feat_vals[feat_name] = tuple([float(v) for v in vals[-3:]])
                        on_tab = True
                    else:
                        if on_tab:
                            feat_lines.append(curr_feat_vals)
                            curr_feat_vals = {}
                            on_tab = False

                    if line.startswith("[ADV Quantile]"):
                        vals = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
                        assert len(vals) == 2
                        quantile_lines.append(tuple([float(v) for v in vals]))

                    if "ε = " in line:
                        vals = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
                        assert len(vals) == 6
                        dp_dict["dp_eps"] = float(vals[-3])
                        dp_dict["dp_delta"] = float(f"{vals[-2]}e{vals[-1]}")

                line_holder = lines[-1]

                # split settings on =
                settings_dict = {"method": method_name}
                for setting in settings:
                    res = setting.split("=")
                    if len(res) > 2:
                        assert res[0] == "load_buck"
                        setting = "=".join(res[1:])

                    k, v = setting.split("=")
                    settings_dict[k] = v
                for k, v in dp_dict.items():
                    settings_dict[k] = v
                settings_dict["dp_noise"] = dp_noise

                settings_dict.update(dataset_dict)

                serialized_dict = json.dumps(settings_dict, sort_keys=True)

                hash_id = hashlib.sha1(serialized_dict.encode()).hexdigest()

                settings_dict["hash_id"] = hash_id

                values = re.findall(r"[-+]?(?:\d*\.*\d+)", line_holder)

                adv_res_dict = {}
                first_ops = True
                if len(quantile_lines) > 0:
                    for quant, q_acc in quantile_lines:
                        if "quantiles" + f"_{quant}" in adv_res_dict:
                            if first_ops:
                                print(
                                    "Warning: Duplicate quantile, assuming ops is the first one."
                                )
                                continue
                            else:
                                print(
                                    "Warning: Duplicate quantile, assuming ops is the second one."
                                )
                        adv_res_dict["quantiles" + f"_{quant}"] = q_acc

                if len(feat_baseline) > 0:
                    for feat, feat_base in feat_baseline.items():
                        if feat in ["num_buckets", "num_samples"]:
                            adv_res_dict[feat] = feat_base
                        else:
                            adv_res_dict[
                                "adv_method_baseline" + "_baseline" + f"_{feat}"  # Uff
                            ] = feat_base

                try:
                    val_dict = eval(line_holder)

                    value = val_dict["clf"]
                    triple = value[0]
                    assert len(triple) == 3
                    settings_dict["clf_train"] = float(triple[0])
                    settings_dict["clf_val"] = float(triple[1])
                    settings_dict["clf_test"] = float(triple[2])

                    # TODO How to ensure ordering?
                    i = 0
                    for key, value in val_dict.items():
                        if "adv" in key or "clf" not in key:
                            triple = value[0]
                            assert len(triple) == 3

                            assert isinstance(key, str)
                            if key.startswith("adv_"):
                                key = key.split("_")[1]
                            key_idx = (
                                "adv_method_recovery"
                                if key == "adv"
                                else "adv_method_" + key
                            )

                            adv_res_dict[key_idx] = (
                                float(triple[0]),
                                float(triple[1]),
                                float(triple[2]),
                            )

                            if len(feat_lines) > 0:
                                feat_data = feat_lines[i]
                                improv_agg = 0
                                for name, vals in feat_data.items():
                                    adv_res_dict[
                                        key_idx
                                        + f"_{settings_dict['sens_feats']}"
                                        + f"_{name}"
                                    ] = vals
                                    if name in feat_baseline:
                                        improvement = vals[-1] - feat_baseline[name][-1]
                                        adv_res_dict[
                                            key_idx
                                            + "_improvement"
                                            + f"_{settings_dict['sens_feats']}"
                                            + f"_{name}"
                                        ] = improvement
                                        improv_agg += improvement
                                adv_res_dict[
                                    key_idx
                                    + "_improvement"
                                    + f"_{settings_dict['sens_feats']}"
                                    + "_mean"
                                ] = improv_agg / len(feat_data)

                                i += 1
                        else:
                            continue

                    settings_dict.update(adv_res_dict)
                    # Append to db
                    assert key_query is not None
                    db.upsert(settings_dict, key_query.hash_id == hash_id)

                except Exception as e:
                    print("Error:", e)
                    print("Line:", line_holder)
                    print("Values:", values)
                    print("Settings:", settings)
                    print("Settings dict:", settings_dict)
                    print("Dataset dict:", dataset_dict)

    return db


def get_dataset_dict_from_str(dataset: str) -> Dict[str, str]:
    dataset_ids = dataset.split("_")
    dataset_dict: Dict[str, str] = {}
    
    print(len(dataset_ids))

    if len(dataset_ids) == 9:  # Full information
        if dataset_ids[-6] in ["disc", "cont", "all"] or (
            dataset_ids[-6][0] == "[" and dataset_ids[-6][-1] == "]"
        ):
            dataset_dict["dataset_name"] = "_".join(dataset_ids[:-8])
            dataset_dict["dataset_state"] = dataset_ids[-8]
            dataset_dict["dataset_year"] = dataset_ids[-7]
            dataset_dict["sens_feats"] = dataset_ids[-6]
            dataset_dict["dataset_val_percentage"] = dataset_ids[-5]
            dataset_dict["dataset_test_percentage"] = dataset_ids[-4]
            dataset_dict["dataset_train_percentage"] = dataset_ids[-3]
            dataset_dict["dataset_buck_percentage"] = dataset_ids[-2]
            dataset_dict["dataset_adv_percentage"] = dataset_ids[-1]
        else:
            assert False, "Invalid dataset string"

    elif len(dataset_ids) == 8:  # No sens feats or no adv_percentage
        if "disc" in dataset_ids or "cont" in dataset_ids or "all" in dataset_ids:
            dataset_dict["dataset_name"] = "_".join(dataset_ids[:-7])
            dataset_dict["dataset_state"] = dataset_ids[-7]
            dataset_dict["dataset_year"] = dataset_ids[-6]
            dataset_dict["sens_feats"] = dataset_ids[-5]
            dataset_dict["dataset_val_percentage"] = dataset_ids[-4]
            dataset_dict["dataset_test_percentage"] = dataset_ids[-3]
            dataset_dict["dataset_train_percentage"] = dataset_ids[-2]
            dataset_dict["dataset_buck_percentage"] = dataset_ids[-1]
            dataset_dict["dataset_adv_percentage"] = "1.0"
        else:
            dataset_dict["dataset_name"] = "_".join(dataset_ids[:-6])
            dataset_dict["dataset_state"] = dataset_ids[-6]
            dataset_dict["dataset_year"] = dataset_ids[-5]
            dataset_dict["sens_feats"] = "disc"
            dataset_dict["dataset_val_percentage"] = dataset_ids[-4]
            dataset_dict["dataset_test_percentage"] = dataset_ids[-3]
            dataset_dict["dataset_train_percentage"] = dataset_ids[-2]
            dataset_dict["dataset_buck_percentage"] = dataset_ids[-1]
            dataset_dict["dataset_adv_percentage"] = "1.0"
    else:  #
        assert len(dataset_ids) == 7
        dataset_dict["dataset_name"] = "_".join(dataset_ids[:-6])
        dataset_dict["dataset_state"] = dataset_ids[-6]
        dataset_dict["dataset_year"] = dataset_ids[-5]
        dataset_dict["sens_feats"] = "disc"
        dataset_dict["dataset_val_percentage"] = dataset_ids[-4]
        dataset_dict["dataset_test_percentage"] = dataset_ids[-3]
        dataset_dict["dataset_train_percentage"] = dataset_ids[-2]
        dataset_dict["dataset_buck_percentage"] = dataset_ids[-1]
        dataset_dict["dataset_adv_percentage"] = "1.0"

    return dataset_dict
