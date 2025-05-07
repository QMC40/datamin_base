# Reproduction and further analysis of 'From Principle to Practice: Vertical Data Minimization for Machine Learning'

> **Repo for the IEEE S&P 2024 paper**  
> Staab *et al.* “From Principle to Practice: Vertical Data Minimization for Machine Learning.” [[PDF]](https://www.sri.inf.ethz.ch/publications/staab2023datamin)

---
## Quick-Start (TL;DR)
```bash
# 1. clone
$ git clone https://github.com/QMC40/datamin_base.git && cd datamin

# 2. create the Conda env and build our sklearn-tree fork
$ ./setup.sh               # installs env "datamin-env"
$ conda activate datamin-env

# 3. sanity-check experiment
$ python run.py --config configs/loan_base.yaml
```
If it finishes without tracebacks you are good to go.  See **Plotting results** below for recreating the paper’s figures.

---
## Supported Platforms
| Works out-of-the-box | How to run |
|----------------------|------------|
| **Linux** (Ubuntu, Fedora, Arch, …) | follow commands verbatim |
| **macOS (≥ 11)**      | install Xcode CLT (`xcode-select --install`) so C-extensions build |
| **Windows 10/11**     | **use WSL 2 + Ubuntu 22.04** and run all commands inside the WSL shell |
| Any Docker / HPC node | mount the repo and run; GPU **not required** |

> **Why no native Windows?**  `sktree/` builds Cython/NumPy extensions that depend on a POSIX-like tool-chain.  Using WSL gives you that environment with zero changes to the codebase.

---
## Prerequisites
* Miniconda ≥ 23 (or Anaconda) <https://docs.conda.io/en/latest/miniconda.html>
* `git`, `bash`, C/C++ compiler (GCC ≥ 9 **or** Clang ≥ 11)
* Python >= 3.9 (handled by Conda)

---
## Setup
### 1 Automated (recommended)
```bash
./setup.sh           # builds env, installs dependencies & sktree fork
conda activate datamin-env
```
*The script is heavily commented—open it if you hit trouble.*

### 2 Manual (when the script fails)
```bash
conda create -n datamin-env python=3.10 -y
conda activate datamin-env
pip install -r requirements.txt           # pure-Python deps

# build our scikit-learn fork (PAT)
cd sktree && ./build.sh && cd ..
```

<details>
<summary>Typical build hiccups & fixes</summary>

| Symptom | Fix |
|---------|-----|
| `gcc: command not found` | `sudo apt install build-essential` (Linux) / install Xcode CLT (macOS) |
| `error: numpy headers not found` | `pip install --no-binary :all: numpy` inside the env, then re-run build |
| `fatal error: Python.h: No such file` | `conda install python-devel` (RPM) or ensure `python-dev` headers are present |
</details>

---
## Running Experiments
All configs live in **`configs/`**:
```bash
python run.py --config configs/acs/acs_base.yaml     # example
```
**Multiple experiments?**  Point `--config` to a *meta-config* listing parameter *lists*—`run.py` crowns all permutations.

### Switching between PAT (`sktree`) and vanilla scikit-learn
Some minimizers (e.g. `*_ibm.yaml`) require upstream sklearn.  Keep two envs **or** uninstall/reinstall:
```bash
# switch to vanilla
pip uninstall scikit-learn sktree -y
pip install scikit-learn==1.4.2

# switch back to PAT
cd sktree && ./build.sh && cd ..
```

---
## Experiment I/O conventions
*Input* meta-configs ⇒ Cartesian product ⇒ *run-folder/filename*:
```
<out>/<dataset>_<state>_<year>_<personal>_<val>_<test>_<train%>_<bucket%>_<adv%>/
       <minimizer>_[param=val].txt
```
The **last line** in each `.txt` summarises accuracy vs. privacy metrics; plotting relies on this.

---
## Plotting Results
```bash
bash plotting/plot_base.sh        # paper’s Fig.-1-to-3
bash plotting/plot_ablation.sh    # ablation appendix

# ad-hoc
python plotting/plot.py   --clear-db   --dir results/acs   --plot_over minimizer   --out_dir my_plots
```
Add `--threshold_acc 0.85` to surface the best privacy-preserving generalization ≥ 85 % of original accuracy.

---
## Simplified Workflow on *Your* Data
### 1 Interactive attribute selection
```bash
python run.py --config configs/selection.yaml               data_path=/absolute/path/to/your.csv
```
Choose personal attributes ⮕ runs a curated minimizer set.

### 2 Visualize & pick thresholds
```bash
python plotting/plot.py   --dir results/selection --plot_over method --clear-db   --out_dir selection_plots --tag FIRST_LOOK   --threshold_acc 0.90
```
### 3 Lock a dedicated config
Copy a YAML from `configs/`, freeze the chosen attributes/thresholds, and commit it to version control so reruns are reproducible.

---
## Extending the Framework
### New dataset loader
1. Drop CSV or parquet under `data/`.
2. In `datamin/dataset.py`, add a `load_<name>()` that returns the dict:
   * `train`: `(X, y)` (and `s` for sensitive attrs if available)
   * `ft_pos`, `feature_names`, `cont_features`
3. Register the loader in `dataset_map`.
4. Write a YAML under `configs/` referencing `dataset: <name>`.

### New minimizer / adversary
Subclass the respective abstract base, implement `fit()` & `get_bucketization()`, then register in `<x>_factory.py`.  Provide default h-params in `utils/config.py` and reference it in configs.

---
## Troubleshooting / FAQ
| Question | Answer |
|----------|--------|
| *Can I use a GPU?* | Not needed—the heavy lifting is tree-based and CPU-bound. |
| *How do I parallelise?* | Add `n_jobs: <int>` in the YAML.  **Exception**: interactive dataset selection (`selection.yaml`) forces single-process. |
| *ModuleNotFoundError: datamin* | `pip install -e .` at repo root to make it an editable package |
| *Plots are empty* | Did you run with `--clear-db` after new experiments?  The plot DB caches parses. |

---
## License
Distributed under the Apache 2.0 License.  See `LICENSE` for details.
