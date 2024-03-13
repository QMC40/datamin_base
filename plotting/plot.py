import argparse
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from tinydb import Query, TinyDB

from metadata import (
    adv_methods,
    global_methods,
    metadata,
    method2label,
    settings,
)
from utils.args import get_args
from utils.db import parse_files_to_db
from utils.plotting import plot
from utils.point import Point


# flake8: noqa: C901
def go(it, db, args):
    if args.datasets is None:
        q = Query()
        args.datasets = sorted(list(set([entry["dataset_name"] for entry in db.all()])))

    for dataset in args.datasets:
        q = Query()
        methods = []
        db_data = None
        db_data = db.search(q.dataset_name == dataset)

        def base_data_extractor(
            entry: Dict[str, Tuple[float, ...]], plot_over: str, idx: int
        ) -> Dict[str, Dict[int, Dict[str, Tuple[float, ...]]]]:
            method = entry[plot_over]

            column_id = "adv_method_recovery-ops"
            adv_vals = entry[column_id]
            if args.improvement:
                column_id += "_improvement_disc_mean"
                mean_err = entry[column_id]
                adv_vals = (mean_err, mean_err, mean_err)

            res = {
                "train": (
                    entry["clf_train"],
                    adv_vals[0],
                ),  # TODO
                "val": (entry["clf_val"], adv_vals[1]),
                "test": (entry["clf_test"], adv_vals[2]),
            }
            id_dict = {idx: res}
            method_dict = {method: id_dict}
            return method_dict

        data_ex = base_data_extractor

        if args.plot_over == "method":
            methods = list(set([entry["method"] for entry in db.all()]))
            new_methods = []
            for m in global_methods:
                if m in methods:
                    new_methods.append(m)
            methods = new_methods
        elif args.plot_over == "quantiles":
            methods = sorted(
                list(
                    set(
                        [
                            key.split("_")[1]
                            for entry in db.all()
                            for key in entry.keys()
                            if key.startswith("quantiles")
                        ]
                    )
                )
            )
            methods = [f"{(1-float(method))*100:.0f}%" for method in methods]

            def quant_data_extractor(
                entry: Dict[str, Tuple[float, ...]], plot_over: str, idx: int
            ) -> Dict[str, Dict[int, Dict[str, Tuple[float, ...]]]]:
                res: Dict[str, Dict[int, Dict[str, Tuple[float, ...]]]] = {}
                for k, q_val in entry.items():
                    if k.startswith("quantiles"):
                        method = k.split("_")[1]
                        method = f"{(1-float(method))*100:.0f}%"
                        if method not in res:
                            res[method] = {}
                        res[method][idx] = {
                            "train": (entry["clf_train"], q_val),
                            "val": (entry["clf_val"], q_val),
                            "test": (entry["clf_test"], q_val),
                        }
                return res

            data_ex = quant_data_extractor
        elif args.plot_over == "adv_method":
            methods = sorted(
                list(
                    set(
                        [
                            key.split("_")[2]
                            for entry in db.all()
                            for key in entry.keys()
                            if key.startswith("adv_method")
                        ]
                    )
                )
            )
            new_methods = []
            for m in adv_methods:
                if m in methods:
                    new_methods.append(m)
            methods = new_methods

            def adv_method_data_extractor(
                entry: Dict[str, Tuple[float, ...]], plot_over: str, idx: int
            ) -> Dict[str, Dict[int, Dict[str, Tuple[float, ...]]]]:
                res: Dict[str, Dict[int, Dict[str, Tuple[float, ...]]]] = {}
                for k, val in entry.items():
                    if k.startswith("adv_method"):
                        k_split = k.split("_")
                        if len(k_split) > 3:
                            continue
                        method = k_split[2]
                        if method not in res:
                            res[method] = {}
                        res[method][idx] = {
                            "train": (entry["clf_train"], val[0]),
                            "val": (entry["clf_val"], val[1]),
                            "test": (entry["clf_test"], val[2]),
                        }
                return res

            data_ex = adv_method_data_extractor
        elif args.plot_over.startswith("feat_single_"):
            selected_feat = args.plot_over.split("_")[-1]
            methods = sorted(
                list(
                    set(
                        [
                            key.split("_")[
                                3
                            ]  # TODO plotting over adversary requires this to be 2
                            for entry in db.all()
                            for key in entry.keys()
                            if key.endswith(selected_feat)
                        ]
                    )
                )
            )
            methods = [method2label[m] if m in method2label else m for m in methods]

            def feat_single_data_extractor(
                entry: Dict[str, Tuple[float, ...]], plot_over: str, idx: int
            ) -> Dict[str, Dict[int, Dict[str, Tuple[float, ...]]]]:
                res: Dict[str, Dict[int, Dict[str, Tuple[float, ...]]]] = {}
                for k, val in entry.items():
                    if k.endswith(selected_feat):
                        k_split = k.split("_")
                        method = k_split[3]
                        method = (
                            method2label[method] if method in method2label else method
                        )
                        if method not in res:
                            res[method] = {}
                        if method == "baseline":
                            val = (val[0], val[1], val[1])

                        res[method][idx] = {
                            "train": (entry["clf_train"], val[0]),
                            "val": (entry["clf_val"], val[1]),
                            "test": (entry["clf_test"], val[2]),
                        }
                return res

            data_ex = feat_single_data_extractor
        # Plot all features in one plot for a single method e.g. feat_iterative, feat_recovery
        elif args.plot_over.startswith("feat_"):
            adv_method = args.plot_over.split("_")[1]

            methods = sorted(
                list(
                    set(
                        [
                            key.split("_")[-1]
                            for entry in db.all()
                            for key in entry.keys()
                            if key.startswith(f"adv_method_{adv_method}_")
                        ]
                    )
                )
            )

            def feat_data_extractor(
                entry: Dict[str, Tuple[float, ...]], plot_over: str, idx: int
            ) -> Dict[str, Dict[int, Dict[str, Tuple[float, ...]]]]:
                res: Dict[str, Dict[int, Dict[str, Tuple[float, ...]]]] = {}
                for k, val in entry.items():
                    if (
                        k.startswith(f"adv_method_{adv_method}_")
                        and not "improvement" in k
                    ):
                        k_split = k.split("_")
                        method = k_split[-1]
                        if method not in res:
                            res[method] = {}
                        res[method][idx] = {
                            "train": (entry["clf_train"], val[0]),
                            "val": (entry["clf_val"], val[1]),
                            "test": (entry["clf_test"], val[2]),
                        }
                return res

            data_ex = feat_data_extractor
        else:
            methods = sorted(list(set([entry[args.plot_over] for entry in db.all()])))

        data = {}
        for k in methods:
            data[k] = {}

        pareto_train = {}
        for k in methods:
            pareto_train[k] = {}

        pareto_test = {}
        for k in methods:
            pareto_test[k] = {}

        pareto_val = {}
        for k in methods:
            pareto_val[k] = {}

        for i, d in enumerate(db_data):
            new_data = data_ex(d, args.plot_over, i)
            for method in new_data.keys():
                new_data[method][i]["point"] = Point(d)
            for method in new_data.keys():
                data[method].update(new_data[method])

            # data.update(data_ex(d, args.plot_over, i))

        data = {k: v for k, v in data.items() if v}

        pareto_field = "val"  # switch to val
        reference_method = ""

        base_methods = data.keys()
        if args.reference_method:
            base_methods = [args.reference_method]
            args.reference_points = True

        tree_methods = []

        for method in base_methods:
            filtered_train = []
            filtered_test = []
            filtered_val = []
            for i in data[method].keys():
                on_pareto = True
                for j in data[method].keys():
                    if i == j:
                        continue
                    p1 = (
                        data[method][j][pareto_field][0]
                        > data[method][i][pareto_field][0]
                    )
                    p2 = (
                        data[method][j][pareto_field][1]
                        < data[method][i][pareto_field][1]
                    )
                    if p1 and p2:
                        # print(f"Point {i} killed by {j}")
                        on_pareto = False
                        break
                if on_pareto:
                    # Always print out the selected settings

                    if method == "tree":
                        tree_methods.append(
                            (
                                int(data[method][i]["point"]["tree_max_leaf_nodes"]),
                                float(data[method][i]["point"]["tree_alpha"]),
                            )
                        )

                    # Plot test points though
                    filtered_test.append(
                        (
                            data[method][i]["test"][0],
                            data[method][i]["test"][1],
                            i,
                            Point(data[method][i]["point"]),
                        )
                    )
                    filtered_val.append(
                        (
                            data[method][i]["val"][0],
                            data[method][i]["val"][1],
                            i,
                            Point(data[method][i]["point"]),
                        )
                    )
                    filtered_train.append(
                        (
                            data[method][i]["train"][0],
                            data[method][i]["train"][1],
                            i,
                            Point(data[method][i]["point"]),
                        )
                    )
            pareto_train[method] = filtered_train
            pareto_test[method] = filtered_test
            pareto_val[method] = filtered_val

            if args.reference_points:
                reference_method = method
                break

        tree_methods = sorted(tree_methods)

        for tree_setting in tree_methods:
            print(f"Tree setting: {tree_setting}")

        if args.reference_points:
            for method in data.keys():
                if method == reference_method:
                    continue
                filtered_train = []
                filtered_test = []
                filtered_val = []
                for _, _, i, point in pareto_train[reference_method]:
                    print(f"Point {i} {point.red_str()}")

                    relevant_point = [
                        np
                        for k, np in data[method].items()
                        if point.eq_except(np["point"], args.plot_over)
                    ][0]

                    filtered_test.append(
                        (relevant_point["test"][0], relevant_point["test"][1], i)
                    )
                    filtered_val.append(
                        (relevant_point["val"][0], relevant_point["val"][1], i)
                    )
                    filtered_train.append(
                        (
                            relevant_point["train"][0],
                            relevant_point["train"][1],
                            i,
                        )
                    )
                pareto_train[method] = filtered_train
                pareto_test[method] = filtered_test
                pareto_val[method] = filtered_val

        # Draw baseline
        lines = []
        if "baseline" in methods:
            if len(pareto_test["baseline"]) > 0:
                lines.append((pareto_test["baseline"][0][1], "majority baseline"))
            del pareto_test["baseline"]
        # PLOT
        TAG = args.tag
        dataset_name = dataset.split("_")[0]
        dataset_name = dataset_name[0].upper() + dataset_name[1:]
        if "health" in dataset:
            if len(dataset.split("_")) > 3:
                dataset_name = f"{dataset_name}, pruned"
        header_text = (
            f"{dataset_name}" + f"{args.header_suffix}"
        )  # , varying over {args.plot_over}"
        name = f"{dataset}_{args.dir.replace('/','_')}_{args.plot_over}"
        if args.method is not None:
            for method in args.method:
                name += f"_{method}"
        ex = "pdf"
        basedir = os.path.join(args.out_dir, TAG)

        plot(
            it,
            args,
            metadata[dataset],
            pareto_test,
            f"{basedir}/{name}.{ex}",
            methods,
            header_text,
            lines=lines,
            pareto=True,
        )
        if args.plot_train_val:
            header_text = f"{header_text}, train"
            name = f"{dataset}_{args.dir.replace('/','_')}_{args.plot_over}_train"
            plot(
                it,
                args,
                metadata[dataset],
                pareto_train,
                f"{basedir}/{name}.{ex}",
                methods,
                header_text,
                lines=lines,
                pareto=True,
            )
            header_text = f"{header_text}, val"
            name = f"{dataset}_{args.dir.replace('/','_')}_{args.plot_over}_val"
            plot(
                it,
                args,
                metadata[dataset],
                pareto_val,
                f"{basedir}/{name}.{ex}",
                methods,
                header_text,
                lines=lines,
                pareto=True,
            )


##########################
if __name__ == "__main__":
    args = get_args()
    it = 0

    db = parse_files_to_db(args)

    print("Num of db entries:", len(db))

    if args.threshold_acc > 0:
        # Take the feature select method with attribute all$
        q = Query()
        res = db.search(q.method == "featsel" and q.featsel_k == "all")
        if len(res) == 0:
            print("No results for feature selection")
            exit(0)
        elif len(res) > 1:
            print("Multiple results for feature selection")
            exit(0)
        else:
            res = res[0]
            print("Found Accuracy:", res["clf_test"])
        acc = res["clf_test"]
        target_acc = args.threshold_acc * acc
        print("Threshold accuracy:", target_acc)
        best_res = db.search(q.clf_test >= target_acc)
        # Sort by recovery ops
        best_res = sorted(
            best_res, key=lambda x: x["adv_method_recovery-ops"][1], reverse=False
        )
        print("Best results:")
        for i, res in enumerate(best_res[:3]):
            method_str = res["method"]
            for k, v in res.items():
                if k.startswith(res["method"] + "_"):
                    method_str += f" {'_'.join(k.split('_')[1:])}: {v}"

            print(
                f"{i}: Acc-Val: {res['clf_val']} Acc-Test: {res['clf_test']} Adv Acc. {res['adv_method_recovery-ops'][1]:.3f} Method: {method_str}"
            )

        print(
            "Note that you can find the corresponding generalizations directly from the stored run results."
        )

    elif not args.load:
        go(it, db, args)
        # plt.show()
    else:
        print("Done loading data into database")
        print("Run again without --load to plot")
        print("Run with --clear-db to clear the database")
