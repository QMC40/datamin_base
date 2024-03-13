# (A8 adversary) Singling Out

## Description

Singling out refers to an adversaries capability of finding a single individual in the minimized data. We test this using several settings of PAT on ACSEmployment. 

## Special points


## Columns and Rows

In our table we have 7 scenarios and 2 rows. I nthe first row we givbe the summed up number of of individual buckets (a bucket refers to a possible value for a single attribute in the minimized data) for each setting. In the second row we give the utilization of the least frequent bucket in this setting. The value "X (Y)" indicates that the lest frequent minimized data representation has X occurrences in the minimized dataset. If there are multiple least frequent representations we denote their count by Y.

## Interpretation

This table shows how the least frequent bucket is utilized in the minimized data. We can see that the least frequent bucket is utilized more often in the original data than in the minimized data. This is expected as the minimization algorithm tries to reduce the number of unique values. However with a certain size of buckets we cannot prevent that there are some unique data points in the minimized data. Especially when considering data minimization from a perspective of k-anonymity (or similar concepts) it is important to consider these counts.
