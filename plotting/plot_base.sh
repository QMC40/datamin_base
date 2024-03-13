# compare methods
python3 ./plotting/plot.py --clear-db --plot_over method --dir results/acs --header_suffix ", CA 2014" --out_dir plots --tag BASE 
python3 ./plotting/plot.py --clear-db --plot_over method --dir results/health --out_dir plots --tag BASE 

# High certainty recovery
python3 ./plotting/plot.py --clear-db --plot_over quantiles --method advtrain --dir results/acs/ --header_suffix ", CA 2014" --out_dir plots --tag HIGHCERT

# Mean ablation
python3 ./plotting/plot.py --clear-db --load --plot_over feat_single_SEX --method advtrain --dir results/acs/ACSIncome_CA_2014_disc_0.1_0.3_1.0_1.0
python3 ./plotting/plot.py --no-clear-db --plot_over feat_single_SEX --method advtrain --dir results/ablation/single_sens_feat_sex/ --header_suffix ", SEX, Advtrain" --out_dir plots --tag MEAN --no-plot-stars

python3 ./plotting/plot.py --clear-db --load --plot_over feat_single_SEX --method tree --dir results/acs/ACSIncome_CA_2014_disc_0.1_0.3_1.0_1.0
python3 ./plotting/plot.py --no-clear-db --plot_over feat_single_SEX --method tree --dir results/ablation/single_sens_feat_sex/ --header_suffix ", SEX, PAT" --out_dir plots --tag MEAN --no-plot-stars

python3 ./plotting/plot.py --clear-db --load --plot_over feat_single_MAR --method advtrain --dir results/acs/ACSIncome_CA_2014_disc_0.1_0.3_1.0_1.0
python3 ./plotting/plot.py --no-clear-db --plot_over feat_single_MAR --method advtrain --dir results/ablation/single_sens_feat/ --header_suffix ", MAR, Advtrain" --out_dir plots --tag MEAN --no-plot-stars

python3 ./plotting/plot.py --clear-db --load --plot_over feat_single_MAR --method tree --dir results/acs/ACSIncome_CA_2014_disc_0.1_0.3_1.0_1.0
python3 ./plotting/plot.py --no-clear-db --plot_over feat_single_MAR --method tree --dir results/ablation/single_sens_feat/ --header_suffix ", MAR, PAT" --out_dir plots --tag MEAN --no-plot-stars

# compare adv_methods
python3 ./plotting/plot.py --clear-db --plot_over adv_method --method tree --dir results/ablation/acs_adversaries_ops/ --out_dir plots --tag ADVMETHODS --zoom --reference_method "oneout-ops" --header_suffix ", CA 2014"
# python3 ./plotting/plot.py --clear-db --plot_over adv_method --method tree --dir results/ablation/acs_adversaries_ops/ --out_dir plots --tag ADVMETHODS
# python3 ./plotting/plot.py --clear-db --plot_over adv_method --method advtrain --dir results/ablation/acs_adversaries_ops/ --out_dir plots --tag ADVMETHODS
# python3 ./plotting/plot.py --clear-db --plot_over adv_method --method mi --dir results/ablation/acs_adversaries_ops/ --out_dir plots --tag ADVMETHODS
# python3 ./plotting/plot.py --clear-db --plot_over adv_method --method uniform --dir results/ablation/acs_adversaries_ops/ --out_dir plots --tag ADVMETHODS

# Plot all features in one plot for a single method e.g. feat_iterative, feat_recovery
python3 ./plotting/plot.py --clear-db --plot_over feat_recovery-ops --method tree --dir results/acs/ACSIncome_CA_2014_disc_0.1_0.3_1.0_1.0/ --out_dir plots_test --tag MEANADV --zoom --no-plot-stars --frame-legend --header_suffix ", CA 2014"
python3 ./plotting/plot.py --clear-db --plot_over feat_recovery-ops --method tree --dir results/acs/ACSEmployment_CA_2014_disc_0.1_0.3_1.0_1.0/ --out_dir plots_test --tag MEANADV --zoom --no-plot-stars --frame-legend --header_suffix ", CA 2014"
python3 ./plotting/plot.py --clear-db --plot_over feat_recovery-ops --method tree --dir results/acs/ACSPublicCoverage_CA_2014_disc_0.1_0.3_1.0_1.0/ --out_dir plots_test --tag MEANADV --zoom --no-plot-stars --frame-legend --header_suffix ", CA 2014"