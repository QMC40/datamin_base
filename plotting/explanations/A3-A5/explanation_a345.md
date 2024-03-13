# (A3-A5 adversaries) Reconstruction with Side Information

## Description

The plot provides an overview of the reconstruction error of the A3, A4, and A5 adversaries for a fixed data minimization algorithm and various configurations. The reconstruction error is measured as the mean error of the A3, A4, and A5 adversaries across all sensitive attributes. 

## Special points

The plot contains two special points:

- The grey star at the topr right corner corresponds to a fully minimized dataset (all points are equal). The respective classification accuracy and reconstruction error correspond to the majority labels across the dataset for each attribute.
- The grey circle at the bottom left corner corresponds to the original dataset (no minimization). The respective classification accuracy corresponds to the accuracy of the original dataset, upper bounding the accuracy of any minimization algorithm. The reconstruction error is 0 (no error) as the adversary sees non-minimized data.

## Axes

The x-axis denotes the absolute classification accuracy on the downstream task when using the minimization algorithm. The y-axis denotes the reconstruction error of the A1 adversary.

- X-axis: [0,1] (classification accuracy)
- Y-axis: [0,1] (reconstruction error) (1-accuracy)

## Interpretation

We first explain the three adversaries and their respective reconstruction errors:

- A3: The adversary gets both the minimized data as well as all non-sensitive attributes (in full resolution). The reconstruction error is the mean error across all sensitive attributes.
- A4: The adversary gets both the minimized data as well as all but the target sensitive attribute (in full resolution). The reconstruction error is the mean error across all sensitive attributes. This models the worst case scenario where the adversary has access to all but the target sensitive attribute.
- A5: The adversary gets both the minimized data as well as all non-sensitive attributes (in full resolution). It has access to some full-resolution sensitive attributes (specified by a parameter k). It models the intermediate case between A3 and A4. For our evaluation in a single plot we average over all possible choices of k.

The key takeaway is how well a generalization performs under additional knowledge. A4 models the worst case scenario where the adversary has access to all but the target sensitive attribute. A3 models the a more realsitic public information scenario commonly assumed. In practice we commonly see how data minimization can protect reasonably well against all three scenarios. However it is important to investigate unexpected behavior in the reconstruction error as it may indicate a noticably weaker protection against potential adversarial scenarios.


