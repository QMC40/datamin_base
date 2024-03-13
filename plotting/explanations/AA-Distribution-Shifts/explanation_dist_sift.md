# Distributiuon Shifts

## Description

The plot provides an overview of the reconstruction error of A1 adversaries for a fixed data minimization algorithm and various configurations under a distribution shift of the respective dataset. The reconstruction error is measured as the mean error of the A1 adversary across all sensitive attributes.

## Axes

The x-axis denotes the absolute classification accuracy on the downstream task when using the minimization algorithm. The y-axis denotes the reconstruction error of the A1 adversary.

- X-axis: [0,1] (classification accuracy)
- Y-axis: [0,1] (reconstruction error) (1-accuracy)

## Interpretation

For this plot we train a minimizer once on the ACSEmployment split from 2014 and then use the learned minimization to minimize the ACSEmployment split from 2015-2018. While PAT performs well under this real-world scenario it is important to note that monitoring such distribution drifts is extremely important in practice. In particular we can use this plot to figure out when we should potentially re-train a new minimizer to adapt to the new distribution of the data, in order to ensure appropriate levels of accuracy and privacy protection.


