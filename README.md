# GRAFT

This repository contains code I developed for the 2025 Virtual Cell Challenge, and related tasks.

The relatedness_transfer/ directory contains the code for the method I finally submitted, which was a Kernel Ridge Regression approach that fused kernels from multiple external sources to estimate how similar the effects of different perturbations should be to one another.

The bayesopt/ directory contains code for some later attempts at active learning and Bayesian optimization for perturbation effect prediction.

The gnn/ directory contains code for an earlier Graph Neural Network based approach. The graft/ directory contains code for an even earlier approach, based on both GNNs and VAEs to aid in cross-dataset transfer. The cell_state_flow/ directory contains a few brief attempts at a flow matching based approach. Most other directories contain similarly brief attempts.

The scripts/ directory contains most of the miscellaneous utility scripts.
