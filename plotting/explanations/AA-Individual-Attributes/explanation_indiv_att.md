# Individual attributes

## Description

The plot provides an overview of the reconstruction error of the A1 adversaries for a fixed data minimization algorithm and various configurations. In particular instead of reporting the mean error across all sensitive attributes we report the error for each individual sensitive attribute. This allows us to investigate if the minimization algorithm is protecting all sensitive attributes well or whether there are specific attributes that require more attention. WE show this for PAT on ACSIncome.

## Axes

The x-axis denotes the absolute classification accuracy on the downstream task when using the minimization algorithm. The y-axis denotes the reconstruction error of the A1 adversary.

- X-axis: [0,1] (classification accuracy)
- Y-axis: [0,1] (reconstruction error) (1-accuracy)

## Interpretation

We can see in the plot how some attributes have noticably higher reconstruction errors than others (e.g. OCCP). This is primarily due there being a significantly smaller base rate of the largest class (there are many occupations but only two values for SEX). In practice we can use this plot to find specific cutoff-values that achieve good privacy protection for attributes that we are specifically interested in. Further it provides a good overview of the trade-off between privacy and utility for each individual attribute, and the representativeness of the mean aggregate error across all attributes in A1.
