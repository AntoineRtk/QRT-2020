# QRT-2020
Stock Return Prediction Challenge

This repository provides a Python script for training and evaluating Random Forest and XGBoost models using cross-validation on the QRT 2020 dataset.

# Random Forest and XGBoost Model Training with train.py

## Features

The dataset is preprocessed following a hierarchy, as described in the notebook. Several features are crafted, including conditional mean, cumulative returns, volatility features, and cross-level differences. The exhaustive list of features is:

- `RET_1, RET_2, RET_3, RET_4, RET_5`: Past returns
- `VOLUME_1, VOLUME_2, ..., VOLUME_20`: Volume data
- `RET_X_Y_mean, RET_X_Y_median, RET_X_Y_std`: Aggregated features across different groups (`X`, `Y` representing different levels like sector, industry group, etc.)
- `CUM_RETURNS`: Cumulative sum of returns over 5 periods
- `CUM_SQUARED_RETURNS`: Sum of squared returns over 5 periods
- `RET_VOLATILITY_SECTOR_mean`: Mean sector return volatility
- `RET_VOLATILITY_SECTOR_std`: Standard deviation of sector return volatility
- `VOLUME_VOLATILITY_SECTOR_mean`: Mean sector volume volatility
- `VOLUME_VOLATILITY_SECTOR_std`: Standard deviation of sector volume volatility
- `ROLLING_STD_5`: Rolling standard deviation over 5 periods
- `VOLATILITY_RATIO`: Ratio of rolling standard deviation to sector volatility mean
- `VOL_Z`: Normalized volume based on stock mean and standard deviation
- `RET_1_SECTOR_DATE_mean`: Mean of RET_1 within a sector on a given date
- `SECTOR_INDUSTRY_GROUP_DIFF`: Difference between the mean sector and industry group returns
- `SECTOR_INDUSTRY_DIFF`: Difference between the mean sector and industry returns

## Installation

Ensure you have Python installed along with the required dependencies. You can install the necessary packages using:

```sh
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Usage

Run the script with the following command:

```sh
python -m train.py --features RET_1,VOLUME_1,CUM_SQUARED_RETURNS --n_estimators 200 --depth 8 --random_seed 42 --output results.json
```

### Arguments

- `--features`: Comma-separated list of features to be used in training.
- `--n_estimators`: Number of trees in the Random Forest model (default: 200).
- `--depth`: Maximum depth of the trees (default: 8).
- `--random_seed`: Random seed for reproducibility (default: 42).
- `--output`: File to save the results (default: `results.json`).

## Output

The script saves results in a JSON file containing:

- Selected parameters
- Accuracy of both models (mean and standard deviation across folds)

