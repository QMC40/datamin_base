# From Principle to Practice: Vertical Data Minimization for Machine Learning

This is the repository accompanying our IEEE S&P 2024 paper ["From Principle to Practice: Vertical Data Minimization for Machine Learning"](https://www.sri.inf.ethz.ch/publications/staab2023datamin).

## Setup

To install the library simply execute `./setup.sh` which will install the `datamin-env` conda environment (conda must be installed beforehand). In case you have any issues with the isntall we provide step-by-step annotations within `setup.sh` to allow for a manual installation. Please note that the script additionally installs our modified version of the `scikit-learn` library (code in `sktree`, installed within `datamin-env`). In case the script should not work simply follow the steps in `./setup.sh` by hand. In case you have issues with our `scikit-learn` version make sure you always fully uninstall any version of the library before attempting to build `sktree`.

## Running experiments

We provide all our experiment configuration files in the folder `configs`. Assuming you are in the `datamin-env` environment you can run a specific config via

`python3 ./run.py --config <path-to-config>`

Please note that the Apt minimizer (`..._ibm.yaml`) requires the non-modified version of `scikit-learn`. You can simply uninstall and re-install `scikit-learn` via

```bash
pip uninstall scikit-learn
pip install scikit-learn
```

To later reinstall the modified version of `scikit-learn` execute

```bash
cd sktree
./build.sh
cd ..
```

## Experiment format

### Input

All experiments are described via self-contained `.yaml` config files. The general format of these configs can be exemplary seen at `configs/acs/acs_base.yaml` which runs the ACS experiments presented in the paper using all of our own methods (requires the custom `scikit-learn` for `PAT`, in configs referred to as `tree`). Note that `.yaml` files actually specify meta-configs that create individual experiments as cross-products of all lists that are specified in them (e.g., across the three ACS datasets). 

### Output

We generally store results in the following format: 

```
<specified_out_path>/<dataset>_<state>_<year>_<personal_attr>_<val_split>_<test_split>_<train_percent>_<bucketization_percent>_<adv_percent>/<minimizer>_[<param=val>].txt
``` 

Our plotting script uses this format to properly parse files and their contents into a temporary database. Note that most (not all) of the relevant information of a run is summarized in the final line of a file which contains classification accuracies as well as adversarial accuracies under all specified classifiers and adversaries.

Also note that we keep this format even for other datasets that do not have state / year (Health, Crime, etc.) to maintain consistency in the parsing (however they must not be explicitly declared in the respective config and will fall back to default values). 

## Plotting results

All plotting is done via the `./plotting/plot.py` python script. We provide additional `./plotting/plot_base.sh` (generates the base plots known from our paper) and `./plotting/plot_ablation.sh` (generates ablation plots from the paper) scripts. If you want to create your own plots, we recommend orienting yourself based on the commands used in these scripts.

## Simplified usage

### Selecting your personal attributes and accuracy thresholds

We provide a meta-config that allows interactive personal attribute selection via `configs/selection.yaml`. It expects the path to a `.csv` file containing the dataset (preferably with a header row). We note that we provide a [small toy loan dataset](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/) alongside in `data_example/loan_data_set.csv` on which we do not evaluate otherwise. When running it via 

```
python3 ./run.py --config configs/selection.yaml
```

it will let you choose which attributes you consider as personal and run a pre-selected set of minimizers (and various configurations) on them. You can visualize the corresponding results via 

```
python3 ./plotting/plot.py --clear-db --plot_over method --dir results/selection --out_dir selection_plots --tag MY_TAG 
```

Further, when adding the `--threshold_acc 0.xx` flag to the plotting script it will also directly output the generalization achieving the best average privacy (over selected attributes) while maintaining at least `0.xx` of the non-minimized accuracy on the classification task (note that this requires the `feature_selection` minimizer with the setting `all` to be part of the minimizer runs; which is the default). We also note that feature selection with the setting `all` corresponds to exactly the original prediction task (which can be taken as a point of comparison). 
If, after initial data exploration, you plan to run the same experiment multiple times, we recommend writing a specific config for it. 

Following the scripts `./plotting/plot.py` and `./plotting/plot_ablation.sh` we can generate a wide variety of visualizations and metrics across minimizer evaluations. We provide a set of explanations for all major plots and tables that can be created with our scripts in the `/explanations` folder.

### Important notes
- The way datasets are implemented using a dynamic selection interface at the moment mandates that only a single worker works at a time. If you want to run with more workers (as we highly recommend for more practical use-cases), we recommend reading the new dataset section below to allow the programmatic loading of your dataset.
- In line with literature in this area we assume that the target attribute `y` is binary.


## New datasets

All datasets in this project are combined and selected in `datamin/dataset.py`. In particular, we extended the ACS `FolktablesDataset` to also represent other datasets. In case you have a dataset that requires preprocessing you can add you respective dataset starting in the `__init__` method. Make sure that you follow the same format as the other datasets such as `health` via `load_health_preprocessed` or the generic `.csv` loader `load_csv`. In particular, we assume the following dataset dict when initializing the `FolktablesDataset` class:

- `train`: A tuple of `(X,y)` containing the feature matrix `X` and the target vector `y`. (for ACS we also directly include the sensitive attributes `s`)
- `ft_pos`: A dictionary mapping feature names to their positions in the feature matrix. Each value is either a single int or a tuple of ints `(a,b)` which then refers to the range in the feature vector (1-hot encoded for categoricals).
- `feature_names`: A list of feature names.
- `cont_features`: A list of indices of continuous features in the feature matrix.

Note that the `train` key actually contains all data (including validation and test) and the respective splits are done after loading the dataset in `datamin/dataset.py` (the same holds for data scaling which is automatically applied).

## New minimizers

In case you want to implement a new minimizer you can do so directly by extending the `datamin/minimizers/abstract_minimizer.py` class. In particular, you need to implement the `fit` method which takes a `FolktablesDataset` and computes a `Bucketization` on the dataset. Additionally, you should implement the  `get_bucketization` method that returns the respective bucketization. We have several simple examples to get you started, most notably the `UniformMinimizer` in `datamin/minimizers/uniform.py`. 

After implementing your minimizer you can register it in the `datamin/minimizers/minimizer_factory.py` to make it accessible via a common interface. You can then also add the required configuration (e.g., hyperparameters) to `datamin/utils/config.py` to write configs for it (`datamin/utils/config.py` already contains many examples to get you started). 

## New adversaries

Implementing a new adversary is similar to implementing a new minimizer. You can extend the `datamin/adversaries/adversary.py` class (we include several such extensions in `datamin/adversaries/`). 

Again you should register your adversary in the `datamin/adversaries/adversary_factory.py` to make it accessible via a common interface. Lastly, you should add the required configuration (e.g., hyperparameters) to `datamin/utils/config.py` to write configs for it (`datamin/utils/config.py` already contains many examples to get you started).

## Citation

If you found this code or our paper helpful, please consider citing us:

```bibtex
@article{staab2024datamin, 
    title={From Principle to Practice: Vertical Data Minimization for Machine Learning}, 
    author={Staab, Robin and Jovanović, Nikola and Balunović, Mislav and Vechev, Martin}, 
    year = {2024}, 
    booktitle = {45th IEEE Symposium on Security and Privacy}, 
    publisher = {{IEEE}} 
}
```