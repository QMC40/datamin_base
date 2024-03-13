import argparse


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        assert values is not None
        for value in values:
            key, value = value.split("=")
            key = key.strip()
            value = value.strip()
            getattr(namespace, self.dest)[key] = value


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, default="results/acs", help="Root directory of results"
    )
    parser.add_argument(
        "--plot_over", type=str, default="method", help="What to plot over"
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Only load data into database, don't plot yet.",
    )
    parser.add_argument(
        "--clear-db",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Clear database before loading new data",
    )
    parser.add_argument(
        "--plot-stars",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Plot stars indicating lower and upper bounds",
    )
    parser.add_argument(
        "--header_suffix",
        type=str,
        default="",
        help="Suffix to add to the header string",
    )
    parser.add_argument(
        "--zoom", default=False, action="store_true", help="Zoom in on the plot"
    )
    parser.add_argument(
        "--reference_points",
        default=False,
        action="store_true",
        help="Select reference points and then plot all other methods over these points",
    )
    parser.add_argument(
        "--reference_method",
        type=str,
        default=None,
        help="Method to use as reference for reference points",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out_test/plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--tag", type=str, default="TEST", help="Tag to add to the output directory"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=str,
        help="Which datasets to plot, Default: all in directory",
    )
    parser.add_argument(
        "--method",
        nargs="*",
        type=str,
        help="Which method to plot over in case we specify certain other attributes",
    )
    parser.add_argument(
        "--plot-train-val", action="store_true", help="Plot train and val data"
    )
    parser.add_argument(
        "--improvement", action="store_true", help="Plot improvement over baseline"
    )
    parser.add_argument(
        "--frame-legend", action="store_true", help="Put legend in a frame and scale"
    )

    parser.add_argument(
        "--threshold_acc",
        type=float,
        default=-1,
        help="Threshold accuracy to get the best privacy for a given percentage of max accuracy",
    )

    parser.add_argument("-f", "--filters", nargs="*", action=ParseKwargs)

    args = parser.parse_args()

    if args.plot_over != "method":
        assert len(args.method) > 0, "Plotting over unclear method"

    return args
