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


def plot_dp(it, args, meta, data, lines, filename, methods, header):
    x = methods

    colors = ["#fb8072", "#80b1d3", "#fdb462"]

    labelsize = 18
    ticksize = 14
    legendsize = 14

    fig, ax = plt.subplots()
    # Set size of figure
    fig.set_size_inches(10, 7)

    for c, method in enumerate(x):
        method_x = []
        method_y = []

        for k, v in data[method].items():
            method_x.append(v["test"][0])
            method_y.append(v["test"][1])

        method_x, method_y = zip(*sorted(zip(method_x, method_y)))

        new_x = []
        new_y = []

        # Only keep points on pareto front
        for i in range(len(method_x)):
            on_par = True
            for j in range(len(method_x)):
                if i == j:
                    continue
                if method_x[i] >= method_x[j] and method_y[i] < method_y[j]:
                    on_par = False
                    break
            if on_par:
                new_x.append(method_x[i])
                new_y.append(method_y[i])

        method_x = new_x
        method_y = new_y

        ax.plot(
            method_x,
            method_y,
            color=colors[c],
            marker="o",
            markerfacecolor=colors[c],
            markeredgecolor=colors[c],
            markersize=10,
            label=method,
        )

        if method in lines:
            # Plot line
            plt.axhline(y=lines[method], color=colors[c], ls="dashed")

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

    ax.set_xscale("log")
    loc = "lower right"

    plt.legend(loc=loc, prop={"size": legendsize})

    plt.tight_layout()
    plt.show()

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

            eps = 100

            if "dp_eps" in entry.keys():
                eps = entry["dp_eps"]
            else:
                return {}

            res = {
                "train": (
                    eps,
                    entry["clf_train"],
                ),
                "val": (eps, entry["clf_val"]),
                "test": (eps, entry["clf_test"]),
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

        lines = {}

        for i, d in enumerate(db_data):
            new_data = data_ex(d, args.plot_over, i)
            for method in new_data.keys():
                new_data[method][i]["point"] = d
            for method in new_data.keys():
                data[method].update(new_data[method])
            # data.update(data_ex(d, args.plot_over, i))
            if len(new_data) == 0:  # NO eps specified
                method = d[args.plot_over]
                if method not in lines.keys():
                    lines[method] = d["clf_test"]
                else:
                    lines[method] = max(lines[method], d["clf_test"])

        # PLOT
        TAG = "DP"  # args.tag
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
        ex = "png"
        basedir = os.path.join(args.out_dir, TAG)

        meta = {"x_label": "ε", "y_label": "Accuracy"}

        plot_dp(
            it,
            args,
            meta,
            data,
            lines,
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
