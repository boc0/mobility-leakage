# Secrets Everywhere: Auditing Memorization in Mobility Prediction Models

This is code for the work Secrets Everywhere: Auditing Memorization in Mobility Prediction Models.
It evaluates deep trajectory next-location prediction models for memorization of training data.

## Project Structure

The project compares multiple model architectures. Within the `2-predictive_models/` folder, 
each architecture's folder contains the code to train and evaluate that model, with the 
corresponding usage instructions in `USAGE.md`.

- `2-predictive_models/DeepMove/`
    - `codes/`            # scripts
    - `USAGE.md`
- `2-predictive_models/Graph-Flashback/`
    - `USAGE.md`
    - scripts
- `2-predictive_models/LSTPM/`
    - `train/`            # scripts
    - `USAGE.md`
- `2-predictive_models/markov/`
    - `USAGE.md`
    - scripts
- `2-predictive_models/computation.py`          # script for training and evaluating all models
- `3-result_analysis.py`      # script for analyzing results and generating the figures seen in the paper
