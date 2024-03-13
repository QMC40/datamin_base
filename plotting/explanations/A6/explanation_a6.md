# (A6 adversary) Multi-Breach

## Description

In practice we might have multiple minimized versions of the same dataset. The A6 adversary models the scenario where the adversary has access to these versions and tries to reconstruct the original data. The reconstruction error is the mean error across all sensitive attributes. We show 4 scenarios using various PAT and Uniform minimizer configurations.

## Special points

We do not have specific special points for this table. However it is important to compare the reconstruction error of the A6 adversary with the reconstruction error of the individual A1 adversaries.

## Columns and Rows

In our table we have 4 scenarios and 2 rows. The first row contains the reconstruction error of the individual A1 adversaries. The second row contains the reconstruction error of the A6 adversary.

## Interpretation

We can see across all scenarios that the reconstruction error of the A6 adversary is lower than the reconstruction error of the individual A1 adversaries. This is expected as the A6 adversary has access to more information. However if it is significantly lower it may indicate that the used minimization algorithm is brittle against multiple minimized data releases which is important to consider before applying it in practice.
