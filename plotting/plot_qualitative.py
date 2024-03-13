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

from plotting.metadata import (
    adv_methods,
    global_methods,
    metadata,
    method2label,
    settings,
)
from plotting.utils.args import get_args
from plotting.utils.db import parse_files_to_db


def plot_qual(it, args, meta, data, filename, methods, header):
    x = sorted([float(x) for x in methods])
    clf = [x[0] for x in data]
    adv = [x[1] for x in data]
    buckets = [x[2] for x in data]

    colors = ["#fb8072", "#80b1d3", "#fdb462"]

    labelsize = 18
    ticksize = 14
    legendsize = 14

    fig, ax = plt.subplots()
    # Set size of figure
    fig.set_size_inches(10, 7)

    lines = []
    labels = []

    if not args.adv_only:
        l1 = ax.plot(x, clf, color=colors[0], marker="o")
        lines.append(l1[0])
        labels.append("Clf.")

    l2 = ax.plot(x, adv, color=colors[1], marker="*")
    lines.append(l2[0])
    labels.append("Adv.")

    # ax.legend()
    ax.set_xlabel(meta["x_label"], fontsize=labelsize)
    ax.set_ylabel("Acc.", fontsize=labelsize)
    ax.set_title(
        header,
        {
            "fontsize": labelsize,
            "fontweight": "bold",
            "verticalalignment": "baseline",
            "horizontalalignment": "center",
        },
        pad=25,
    )
    plt.setp(ax.get_xticklabels(), fontsize=ticksize)
    plt.setp(ax.get_yticklabels(), fontsize=ticksize)

    if args.right_axis:
        ax2 = ax.twinx()
        l3 = ax2.plot(x, buckets, color=colors[2], marker=".")
        ax2.set_ylabel("Num. Buckets", fontsize=20)
        # ax2.legend()

        plt.setp(ax2.get_xticklabels(), fontsize=ticksize)
        plt.setp(ax2.get_yticklabels(), fontsize=ticksize)
        lines.append(l3[0])
        labels.append("Buckets")
        loc = "upper right"
    else:
        ax.set_xscale("log")
        loc = "lower right"

    plt.legend(lines, labels, loc=loc, prop={"size": legendsize})

    plt.tight_layout()
    # plt.show()

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    plt.savefig(filename)


# flake8: noqa: C901
def go(it, db, args):

    if args.datasets is None:
        q = Query()
        args.datasets = sorted(list(set([entry["dataset_name"] for entry in db.all()])))

    for dataset in args.datasets:

        q = Query()
        methods = []
        db_data = None

        filter_dict = {}

        if args.filters is None:
            filter_dict = {}
        else:
            filter_dict = args.filters.copy()
        filter_dict["dataset_name"] = dataset

        db_data = db.search(q.fragment(filter_dict))

        def base_data_extractor(
            entry: Dict[str, Tuple[float, ...]], plot_over: str, idx: int
        ) -> Dict[str, Dict[int, Dict[str, Tuple[float, ...]]]]:
            method = entry[plot_over]

            column_id = "adv_method_recovery-ops"
            adv_vals = entry[column_id]

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

        methods = sorted(list(set([entry[args.plot_over] for entry in db.all()])))

        data = {}
        for k in methods:
            data[k] = {}

        train = {}
        for k in methods:
            train[k] = {}

        test = {}
        for k in methods:
            test[k] = {}

        buckets = {}
        for k in methods:
            buckets[k] = {}

        for i, d in enumerate(db_data):
            new_data = data_ex(d, args.plot_over, i)
            for method in new_data.keys():
                new_data[method][i]["point"] = d
            for method in new_data.keys():
                data[method].update(new_data[method])
            # data.update(data_ex(d, args.plot_over, i))

        data = {str(float(k)): v for k, v in data.items() if v}

        base_methods = data.keys()
        if args.reference_method:
            base_methods = [args.reference_method]
            args.reference_points = True

        x_values = sorted([float(x) for x in base_methods])

        filtered_train = []
        filtered_test = []

        for method in x_values:
            str_method = str(method)
            for i in data[str_method].keys():
                filtered_test.append(
                    (
                        data[str_method][i]["test"][0],
                        data[str_method][i]["test"][1],
                        int(data[str_method][i]["point"]["num_buckets"]),
                    )
                )
                filtered_train.append(
                    (
                        data[str_method][i]["train"][0],
                        data[str_method][i]["train"][1],
                        int(data[str_method][i]["point"]["num_buckets"]),
                    )
                )

        filter_str = ",".join([f"{k} = {v}" for k, v in args.filters.items()])
        filter_str = filter_str.replace("tree_max_leaf_nodes", "k*")
        filter_str = filter_str.replace("tree_alpha", "alpha")

        # PLOT
        TAG = args.tag
        dataset_name = dataset.split("_")[0]
        dataset_name = dataset_name[0].upper() + dataset_name[1:] + ", " + filter_str
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
        name += "_" + filter_str
        ex = "pdf"
        basedir = os.path.join(args.out_dir, TAG)

        meta = {}
        args.right_axis = True
        args.adv_only = False

        if args.plot_over == "tree_alpha":
            meta["x_label"] = "Alpha"
        elif args.plot_over == "tree_max_leaf_nodes":
            meta["x_label"] = "k*"
        elif args.plot_over == "num_samples":
            meta["x_label"] = "Number of Samples"
            args.right_axis = False
            args.adv_only = True
        else:
            meta["x_label"] = args.plot_over
            args.right_axis = False
            args.adv_only = True

        plot_qual(
            it,
            args,
            meta,
            filtered_test,
            f"{basedir}/{name}.{ex}",
            methods,
            header_text,
        )


##########################
if __name__ == "__main__":

    args = get_args()
    it = 0

    db = parse_files_to_db(args)

    print("Num of db entries:", len(db))

    if not args.load:
        go(it, db, args)
        # plt.show()
    else:
        print("Done loading data into database")
        print("Run again without --load to plot")
        print("Run with --clear-db to clear the database")
