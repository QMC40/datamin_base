# (A1 adversary) Reconstruction 

## Description

The plot provides an overview of the reconstruction error of the A1 adversary for various data minimization algorithms and configurations. The reconstruction error is measured as the mean error of the A1 adversary across all sensitive attributes. The exemplary plot shows this for ACSEmployment.

## Special points

The plot contains two special points:

- The grey star at the topr right corner corresponds to a fully minimized dataset (all points are equal). The respective classification accuracy and reconstruction error correspond to the majority labels across the dataset for each attribute.
- The grey circle at the bottom left corner corresponds to the original dataset (no minimization). The respective classification accuracy corresponds to the accuracy of the original dataset, upper bounding the accuracy of any minimization algorithm. The reconstruction error is 0 (no error) as the adversary sees non-minimized data.

## Axes

The x-axis denotes the absolute classification accuracy on the downstream task when using the minimization algorithm. The y-axis denotes the reconstruction error of the A1 adversary.

- X-axis: [0,1] (classification accuracy)
- Y-axis: [0,1] (reconstruction error) (1-accuracy)

## Interpretation

We note that the reconstruction error is a measure of how well the adversary can reconstruct the original sensitive attributes from the minimized data. The lower the reconstruction error, the better the adversary can reconstruct the original sensitive attributes. The plot shows that the reconstruction error is generally lower for minimization algorithms with higher classification accuracy. This is expected, as the minimization algorithms aim to maintain the classification accuracy while minimizing the information about the sensitive attributes. **Notably, many algorithms show an elbo curve where we can trade significant gains in classification accuracy for only marginal increases in reconstruction error. We recommend to particularly investigate points on the elbow curve as they might provide a good trade-off between classification accuracy and privacy.**

