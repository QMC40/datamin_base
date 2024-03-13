import os

import matplotlib.pyplot as plt


def addlabels(x, loc, y):
    for i in range(len(x)):
        plt.text(i, loc[i] + 0.05, y[i], ha="center", weight="bold", fontsize=18)


if __name__ == "__main__":
    # Plotting

    header = "ACSEmployment, CA 2014, Alpha=0.7, k*=20"
    x_labels = [
        "Age",
        "Education",
        "Marrital Status",
        "Relationship",
        "Disability",
        "Empl. of Parents",
        "Citizenship",
        "Mobility Status",
        "Military Service",
        "Ancestry Code",
        "Nativity",
        "Hearing difficulty",
        "Vision difficulty",
        "Cognitive difficulty",
        "Sex",
        "Racial code",
    ]
    buckets = [7, 5, 1, 4, 2, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1]
    orig_size = [100, 24, 5, 18, 2, 9, 5, 4, 5, 5, 2, 2, 2, 3, 2, 9]
    percentage = [f"{x/y:.2f}" for x, y in zip(buckets, orig_size)]

    fig = plt.figure()

    labelsize = 24
    ticksize = 20
    legendsize = 14

    fig, ax = plt.subplots()
    # Set size of figure
    fig.set_size_inches(16, 12)
    ax.set_ylabel("Number of buckets", fontsize=labelsize, fontweight="bold")

    ax.set_title(
        header,
        {
            "fontsize": 28,
            "fontweight": "bold",
            "verticalalignment": "baseline",
            "horizontalalignment": "center",
        },
        pad=25,
    )
    plt.xticks(rotation=45, ha="right", weight="bold")
    plt.setp(ax.get_xticklabels(), fontsize=ticksize)
    plt.setp(ax.get_yticklabels(), fontsize=ticksize)
    ax.bar(x_labels, buckets)

    # calling the function to add value labels
    addlabels(x_labels, buckets, percentage)
    plt.tight_layout()
    # plt.show()
    # create dir if not exists
    path = "plots_test/QUALITATIVE/buckets.pdf"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    plt.savefig(path)
