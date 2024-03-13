# Minimizer training data size

## Description

The plot provides an overview of the reconstruction error of the A1 adversaries for a fixed data minimization algorithm and various configurations across various sizes of data used to fit the minimizer. The reconstruction error is measured as the mean error of the A1 adversary across all sensitive attributes. We show this for PAT on ACSEmployment. 

## Special points

The plot contains two special points:

- The grey star at the topr right corner corresponds to a fully minimized dataset (all points are equal). The respective classification accuracy and reconstruction error correspond to the majority labels across the dataset for each attribute.
- The grey circle at the bottom left corner corresponds to the original dataset (no minimization). The respective classification accuracy corresponds to the accuracy of the original dataset, upper bounding the accuracy of any minimization algorithm. The reconstruction error is 0 (no error) as the adversary sees non-minimized data.

## Axes

The x-axis denotes the absolute classification accuracy on the downstream task when using the minimization algorithm. The y-axis denotes the reconstruction error of the A1 adversary.

- X-axis: [0,1] (classification accuracy)
- Y-axis: [0,1] (reconstruction error) (1-accuracy)

## Interpretation

We note that this plot is highly dependent on both chosen minimizer as well as dataset. In particualr we show on how little data one can train the actual minimizer (not the downstream classifier) before "a bad minimization" starts to affect the utility and privacy of the resulting minimized dataset. While PAT has proven to be quite stable, only degrading at very low levels of training data, this is in practice highly dependent on the dataset and the minimizer used. Having such a plot allows you to potentially collect less full resolution data for the minimizer which is overall preferable. At the same time sometimes collecting more upfront data can also enable better evaluations and re-trainings over a longer period of time which might also beneficial.


