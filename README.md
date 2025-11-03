# nvisii-dataset

Utilities for NVISII-based synthetic data workflows.

## Install into existing env
```bash
pip install "nvisii-dataset @ git+https://github.com/overlab-kevin/nvisii-dataset.git"
```

## Clone locally for testing
```bash
git clone https://gitlab.theoverlab.com/spatial/nvisii_data.git
cd nvisii_data
micromamba create -n nd python=3.10 # recommend using micromamba
micromamba activate nd
pip install .
python visualize_dataset.py [path to dataset root] [dataset phase (train/val/test)]
