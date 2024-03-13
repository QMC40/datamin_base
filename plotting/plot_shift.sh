# SHIFT
python3 ./plotting/plot.py --load --dir final_results/acs_shift
python3 ./plotting/plot.py --no-clear-db --plot_over dataset_year --method tree --dir final_results/acs_shift_base --out_dir plots --tag SHIFTPLOTS
# Feature selection
python3 ./plotting/plot.py --plot_over method --dir final_results/acs_feature_selection --out_dir plots --tag feature_selection --zoom