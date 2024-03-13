import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from metadata import method2label

from .misc import F


# flake8: noqa: C901
def plot(
    it, args, meta, data, filename, methods, header, lines=None, aux=None, pareto=False
):
    cL, aL = meta[0], meta[2]
    cU, aU = meta[1], 1.0

    if args.zoom:
        min_clf = 1
        max_clf = 0
        min_adv = 1
        for dps in data.values():
            for dp in dps:
                if dp[0] < min_clf:
                    min_clf = dp[0]
                if dp[0] > max_clf:
                    max_clf = dp[0]
                if dp[1] < min_adv:
                    min_adv = dp[1]
        meta = (min_clf, max_clf, min_adv)
        print(f"Zooming in to {meta}")
    labelsize = 28
    ticksize = 24
    legendsize = 19
    plt.clf()
    if len(methods) > 10:
        f = plt.figure(it, figsize=(20, 14))
        labelsize *= 2
        ticksize *= 2
        legendsize *= 1.5
    else:
        f = plt.figure(it, figsize=(10, 7))

    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["font.family"] = "sans-serif"

    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # plt.grid(axis='y')

    ax = plt.gca()
    # alpha = 0.97
    # ax.yaxis.grid(True, color=(1,1,1))
    # ax.set_facecolor((alpha, alpha, alpha))
    # sns.set_style("dark")

    # remove bounding lines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    scheme4 = [
        "#9b2a2e",
        "#c7b98b",
        "#789a69",
        "#b6d7a8",
        "#31c1c9",
        "#eb8904",
        "#016280",
        "#338500",
        "#e0bf00",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#ffffb3",
        "#bebada",
        "#8dd3c7",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#000000",
    ]

    scheme = list(reversed(scheme4))

    colors = {}
    # add colors to domains
    ii = 0
    for k in methods:
        colors[k] = scheme[ii % len(scheme)]
        ii += 1

    me = methods  # [methods[-1]] + methods[:-1]  # featsel

    ####################################################################

    maxclf = meta[1]

    plt.title(
        header,
        {
            "fontsize": labelsize // 1.5,
            "fontweight": "bold",
            "verticalalignment": "baseline",
            "horizontalalignment": "center",
        },
        pad=25,
    )

    # just set (CLF, ADV) pairs
    for name in me:
        if name not in data:
            continue
        c = colors[name]

        # vals = sorted(data[k].values())
        # clfs = [x[0] for x in vals]
        # advs = [x[1] for x in vals]
        # Plot also points that are not pareto?
        # plt.scatter(clfs, advs, label=k, color=c, linewidth=2, alpha=0.5)

        # plt.plot(clfs, advs, color=c, linewidth=1, opacity=0.5)

        # Prepare data

        if aux is not None:
            tmp = sorted(enumerate(data[name]), key=lambda x: x[0])

            clfs = [x[1][0] for x in tmp]
            advs = [x[1][1] for x in tmp]
            # ids = [x[1][2] for x in tmp]
        else:
            vals = sorted(data[name])
            clfs = [x[0] for x in vals]
            advs = [x[1] for x in vals]
            # ids = [x[2] for x in vals]

        if pareto:
            # Plot only pareto of given points
            filtered = []
            n = len(data[name])
            for i in range(n):
                on_pareto = True
                for j in range(n):
                    p1 = data[name][j][0] > data[name][i][0]
                    p2 = data[name][j][1] < data[name][i][1]
                    if p1 and p2:
                        # print(f"Inner Point {i} killed by {j}")
                        on_pareto = False
                        break
                if on_pareto:
                    filtered.append(data[name][i])
            # connect with lines only the pareto points
            vals = sorted(filtered)
            clfs = [x[0] for x in vals]
            advs = [x[1] for x in vals]

        # PLOT IDS?
        if lines is not None:
            for line in lines:
                plt.axhline(y=F(line[0]), color="gray", ls="dashed")
                plt.text(
                    0.01,
                    F(line[0]) / meta[2] - 0.125,
                    line[1],
                    rotation=0,
                    color="gray",
                    transform=ax.transAxes,
                )

        # If worse than uniform we can fall back to the majority classifier
        if not args.improvement:
            advs = np.maximum(advs, meta[2])
        else:
            advs = F(advs)

        plt.scatter(
            F(clfs),
            F(advs),
            color=c,
            linewidth=4,
            alpha=1,
            label=method2label[name] if name in method2label else name,
        )

        plt.plot(F(clfs), F(advs), color=c, linewidth=2, alpha=1)

        if len(clfs) > 0:
            maxclf = max(maxclf, max(clfs))

    ######################################################################

    # plt.axvline(x=meta[0], color='gray', label='clf lb, majority', ls='dotted')
    # plt.axvline(x=meta[1], color='gray', label='clf ub, fulldata', ls='dashed')
    # plt.axhline(y=meta[2], color='gray', label='adv lb, naive', ls='dashdot')

    loc = "lower right"
    plt.xlim(F(maxclf + 0.02), F(meta[0] - 0.01))

    if args.improvement:
        aU = 0.0
        for dps in data.values():
            for dp in dps:
                aU = max(aU, dp[1])
        print(f"Max adv: {aU}")
        plt.ylim(-0.01, aU + 0.1)
        loc = "upper right"
    else:
        plt.ylim(-0.01, F(meta[2] - 0.03))

    # plt.legend(fontsize=SZ)
    # plt.xlabel('Perturbation Radius', fontsize=SZ)
    # plt.ylabel('Certified Robustness (%)', fontsize=SZ)

    if args.frame_legend:
        plt.legend(
            fontsize=legendsize,
            labelspacing=0.5,
            loc=loc,
            fancybox=True,
            shadow=True,
            ncol=1,
        )
    else:
        plt.legend(fontsize=legendsize, frameon=False, loc=loc, labelspacing=0.5)

    plt.xlabel("Classifier Error", fontsize=labelsize)  # labelpad 10?
    plt.ylabel(
        "Mean Adv. Improvement" if args.improvement else "Adversary Error",
        fontsize=labelsize,
    )
    # plt.title(f'{setting}2014', fontsize=labelsize)

    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.12)

    # ax.yaxis.set_label_coords(0.01, 1.02)
    plt.locator_params(axis="x", nbins=4)
    plt.locator_params(axis="y", nbins=4)

    plt.setp(ax.get_xticklabels(), fontsize=ticksize)
    plt.setp(ax.get_yticklabels(), fontsize=ticksize)

    if args.plot_stars:
        star_clf = [cU]
        star_adv = [aU]

        if not cL < meta[0] and args.zoom:
            star_clf.append(cL)
            star_adv.append(aL)

        plt.scatter(
            F(star_clf),
            F(star_adv),
            s=300,
            marker=(5, 1),  # type: ignore
            color="darkgray",
            zorder=200,
            edgecolors="gray",
        )
        if args.zoom:
            if cL < meta[0]:
                plt.scatter(
                    F([meta[0] - 0.005]),
                    F([meta[2]]),
                    s=300,
                    marker=">",  # type: ignore
                    color="darkgray",
                    zorder=200,
                    edgecolors="gray",
                )

    # plt.yticks([20,40,60,80,100], ['20%', '40%','60%','80%','100%'])
    # sns.despine()
    # plt.tight_layout(pad=1)

    f.tight_layout()

    # Create directory if it not exists
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    plt.savefig(filename)  # , pad_inches=0.0, bbox_inches="tight", dpi=300)
    # plt.show()
