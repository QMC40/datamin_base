# Plot over different architectures
# Table
# Plot over continuous features
python3 ./plotting/plot.py --clear-db --dir ablation/continuous_feats --out_dir plots_test --tag CONT_FEATS --zoom --header_suffix ", CA 2014, continuous"
# Plot improvement graphs
python3 ./plotting/plot.py --clear-db --improvement --dir results/acs/ACSIncome_CA_2014_disc_0.1_0.3_1.0_1.0/ --out_dir plots_test --tag IMPROVEMENT --zoom --header_suffix ", CA 2014" --no-plot-stars
python3 ./plotting/plot.py --clear-db --improvement --dir results/acs/ACSEmployment_CA_2014_disc_0.1_0.3_1.0_1.0 --out_dir plots_test --tag IMPROVEMENT --zoom --header_suffix ", CA 2014" --no-plot-stars
python3 ./plotting/plot.py --clear-db --improvement --dir results/acs/ACSPublicCoverage_CA_2014_disc_0.1_0.3_1.0_1.0 --out_dir plots_test --tag IMPROVEMENT --zoom --header_suffix ", CA 2014" --no-plot-stars

# Qualitative ablation plots
## Ablation for Hyperparameters
python3 ./plotting/plot_qualitative.py --clear-db -f "tree_max_leaf_nodes=20" --plot_over tree_alpha --method tree --dir results/acs/ACSEmployment_CA_2014_disc_0.1_0.3_1.0_1.0 --out_dir plots_test --tag ALPHA
python3 ./plotting/plot_qualitative.py --clear-db -f "tree_max_leaf_nodes=10" --plot_over tree_alpha --method tree --dir results/acs/ACSEmployment_CA_2014_disc_0.1_0.3_1.0_1.0 --out_dir plots_test --tag ALPHA
python3 ./plotting/plot_qualitative.py --clear-db -f "tree_alpha=0.7" --plot_over tree_max_leaf_nodes --method tree --dir results/acs/ACSEmployment_CA_2014_disc_0.1_0.3_1.0_1.0 --out_dir plots_test --tag ALPHA
## Qualitative

# Bucketization size plots
## Adversarial TR TE plots
python3 ./plotting/plot_qualitative.py --clear-db -f tree_alpha=0.7 tree_max_leaf_nodes=50 dataset_buck_percentage=1 --plot_over num_samples --method tree --dir ablation/percentages/ACSIncome_CA_2014_disc_0.1_0.3_1.0_1.0 --out_dir plots_test --tag PERCENTAGES
## Multiple bucketization sizes
python3 ./plotting/plot.py --clear-db --plot_over dataset_adv_percentage --method tree --dir ablation/percentages/ACSIncome_CA_2014_disc_0.1_0.3_1.0_0.05 --out_dir plots_test --tag BUCK_ABL --header_suffix ", CA 2014, PAT, 0.05"
