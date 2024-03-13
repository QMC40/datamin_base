# (A7 adversary) Linkability

## Description

With linkability we refer to the ability of an adversary to use the minimized data release to link two attributes in the original data. In our example we therefore assume that the adversar wants to link Occupation and Place of birth on the original data. We test this using several settings of PAT, Uniform, and AdvTrain on ACSIncome.

## Special points

Using the randomized baseline we can see that we have an average linkage success of $0.24$ (meaning that we can link the two attributes in $24\%$ of the cases).

## Columns and Rows

In our table we have 7 scenarios and 2 rows. I nthe first row we givbe the summed up number of of individual buckets (a bucket refers to a possible value for a single attribute in the minimized data) for each setting. In the second row we give the average linkage success across all settings.

## Interpretation

We can clearly see how higher bucketization (i.e., higher resolution minimized data) leads to a higher linkage success. This is expected as the adversary has more information to work with. There is not much to prevent this as more informative generalization naturally allow an adversary to build a more detailed statistical model (increasing the linkage success).
