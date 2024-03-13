# (A2 adversary) High-certainty Reconstruction 

## Description

The plot provides an overview of the high-certainty reconstruction error of the A2 adversary for a fixed data minimization algorithm and various configurations. We measure the high-certainty reconstruction error as the mean error of the A2 adversary across all sensitive attributes. We rank certainties across all attribute predictions for an attribute by using the absolute logits of the adversary predictions. We then show several thresholds for the top x% of the most certain predictions.
The exemplary plot shows this for ACSPublicCoverage and the AdvTrain minimizer.

## Special points

The plot contains two special points:

- The grey star at the topr right corner corresponds to a fully minimized dataset (all points are equal). The respective classification accuracy and reconstruction error correspond to the majority labels across the dataset for each attribute.
- The grey circle at the bottom left corner corresponds to the original dataset (no minimization). The respective classification accuracy corresponds to the accuracy of the original dataset, upper bounding the accuracy of any minimization algorithm. The reconstruction error is 0 (no error) as the adversary sees non-minimized data.

## Axes

The x-axis denotes the absolute classification accuracy on the downstream task when using the minimization algorithm. The y-axis denotes the reconstruction error of the A1 adversary.

- X-axis: [0,1] (classification accuracy)
- Y-axis: [0,1] (reconstruction error) (1-accuracy)

## Interpretation

The main take-away from this plot is whether or not an adversary is able to make more certain predictions for a specific subset of the data (e.g., a specific group of people). The lower the high-certainty reconstruction error, the better the adversary can reconstruct the original sensitive attributes from the minimized data. The plot shows that the high-certainty reconstruction error for lower threshholds is generally lower than for higher threshholds. In case there are specific groups of people that are more likely to be reconstructed, we recommend to investigate the corresponding threshholds and data points that can be reconstructed with more certainty in more detail.

