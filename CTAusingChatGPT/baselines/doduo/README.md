# Baselines

This folder contains the code to run the DODUO method introduced in the paper [Annotating Columns with Pre-trained Language Models](https://arxiv.org/abs/2104.01785). The code is taken from their official [github repository](https://github.com/megagonlabs/doduo). The "model.py" and "dataset.py" files are modified to automatically run three runs with different random seeds of the model, while all other files remain similar and are a copy of the original files found in DODUO's github repository. To run the training for three different random seeds, the "run_all_doduo_runs.sh" file needs to be executed.

# Preparation of data

The files "Preprocessing-cpa-lm.ipynb" and "Preprocessing-cta-lm.ipynb" can be used to format the SOTAB benchmark into the correct input format to be used when running the DODUO baseline.