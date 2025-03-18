import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import argparse
from sklearn.ensemble import RandomForestClassifier
import time
import itertools
import random
from xgboost import XGBClassifier
import json

def main():
    parser = argparse.ArgumentParser(description='Train a Random Forest model with specified parameters.')
    parser.add_argument('--features', type=str, required=True, help='Comma-separated list of features (e.g., RET_1,VOLUME_1,CUM_SQUARED_RETURNS)')
    parser.add_argument('--n_estimators', type=int, default=200, help='Number of estimators for the Random Forest (default: 200)')
    parser.add_argument('--depth', type=int, default=8, help='Max depth of the Random Forest (default: 8)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output', type=str, default='results.json', help='File to save the results (default: results.json)')

    args = parser.parse_args()

    # Parse features
    features = args.features.split(',')
    print(f'Features: {features}')
    print(f'Max depth: {args.depth}')
    print(f'Number of estimators: {args.n_estimators}')
    print(f'Random seed: {args.random_seed}')

    x_train = pd.read_csv('x_train.csv', index_col='ID')
    y_train = pd.read_csv('y_train.csv', index_col='ID')
    train = pd.concat([x_train, y_train], axis=1)
    
    # Filling NaN values
    ret_columns = [f'RET_{i}' for i in range(1, 6)]
    for col in ret_columns:
        train[col] = train.groupby(['SUB_INDUSTRY', 'DATE'])[col].transform(lambda x: x.fillna(x.median()))
    vol_columns = [f'VOLUME_{i}' for i in range(1, 6)]
    for col in vol_columns:
        train[col] = train.groupby(['DATE', 'SUB_INDUSTRY'])[col].transform(lambda x: x.fillna(x.median()))
        train[col] = train.groupby(['DATE', 'INDUSTRY'])[col].transform(lambda x: x.fillna(x.median()))
        train[col] = train.groupby(['DATE', 'INDUSTRY_GROUP'])[col].transform(lambda x: x.fillna(x.median()))
        
    columns_to_check = ret_columns + vol_columns
    missing_rows_count = train[columns_to_check].isna().any(axis=1).sum()
    train.dropna(subset=columns_to_check, inplace=True)
    
    target = "RET"
    
    # Conditional aggregated features
    shifts = [1, 2, 3, 4, 5]  # Choose some different shifts
    statistics = ['mean', 'median', 'std']  # the type of stat
    
    gb_features = [['DATE', 'SECTOR'], ['INDUSTRY_GROUP', 'DATE'], ['SECTOR', 'INDUSTRY_GROUP']]
    target_feature = 'RET'
    for gb in gb_features:
        tmp_name = '_'.join(gb)
        for shift in shifts:
            for stat in statistics:
                name = f'{target_feature}_{shift}_{tmp_name}_{stat}'
                feat = f'{target_feature}_{shift}'
                train[name] = train.groupby(gb)[feat].transform(stat)
    
    # Cumulative Returns and Squared Returns
    train['CUM_RETURNS'] = train[[f'RET_{i}' for i in range(1, 6)]].sum(1)
    train['CUM_SQUARED_RETURNS'] = (train[[f'RET_{i}' for i in range(1, 6)]] ** 2).sum(1)
    
    # Volatility Features
    volatility = train.groupby(["SECTOR", "DATE"])[[f'RET_{i}' for i in range(1, 6)]].transform('mean')
    train['RET_VOLATILITY_SECTOR_mean'] = volatility.std(axis=1)
    volatility = train.groupby(["SECTOR", "DATE"])[[f'RET_{i}' for i in range(1, 6)]].transform('std')
    train['RET_VOLATILITY_SECTOR_std'] = volatility.std(axis=1)
    volatility = train.groupby(["SECTOR", "DATE"])[[f'VOLUME_{i}' for i in range(1, 21)]].transform('mean')
    train['VOLUME_VOLATILITY_SECTOR_mean'] = volatility.std(axis=1)
    volatility = train.groupby(["SECTOR", "DATE"])[[f'VOLUME_{i}' for i in range(1, 21)]].transform('std')
    train['VOLUME_VOLATILITY_SECTOR_std'] = volatility.std(axis=1)
    
    train['ROLLING_STD_5'] = train[[f'RET_{i}' for i in range(1, 6)]].std(axis=1)
    train['VOLATILITY_RATIO'] = train['ROLLING_STD_5'] / train['VOLATILITY_SECTOR_mean']
    
    # Normalized Volume Feature
    train['VOL_Z'] = (train['VOLUME_1'] - train.groupby('STOCK')['VOLUME_1'].transform('mean')) / \
                  train.groupby('STOCK')['VOLUME_1'].transform('std')
    
    # Cross-level Differences
    train['SECTOR_INDUSTRY_GROUP_DIFF'] = train['RET_1_SECTOR_DATE_mean'] - train.groupby(['INDUSTRY_GROUP', 'DATE'])['RET_1'].transform("mean")
    train['SECTOR_INDUSTRY_DIFF'] = train['RET_1_SECTOR_DATE_mean'] - train.groupby(['INDUSTRY', 'DATE'])['RET_1'].transform("mean")
    
    train.dropna(axis="index", inplace=True)
    
    train_dates = train['DATE'].unique()
    n_splits = 4
    
    splits = KFold(n_splits=n_splits, random_state=args.random_seed, shuffle=True).split(train_dates)
    
    acc_xgb = []
    acc_rf = []
    
    # Iterate over each fold
    for i, (local_train_dates_ids, local_test_dates_ids) in enumerate(splits):
        local_train_dates = train_dates[local_train_dates_ids]
        local_test_dates = train_dates[local_test_dates_ids]

        # Get the indices of the train and test sets for this fold
        local_train_ids = train['DATE'].isin(local_train_dates)
        local_test_ids = train['DATE'].isin(local_test_dates)

        # Split the data into local train and test sets
        X_local_train = train.loc[local_train_ids, features]
        y_local_train = train.loc[local_train_ids, target]
        X_local_test = train.loc[local_test_ids, features]
        y_local_test = train.loc[local_test_ids, target]
    
        # Initialize and train the models
        params = { # Mostly for regularization
            "lambda": 0.7,
            "gamma": 0.7,
            "colsample_bytree": 0.3,
        }
        xgb = XGBClassifier(eta = 0.01, max_depth = args.depth, subsample = 0.5, alpha = 0.7, random_state = args.random_seed, **params)
        xgb.fit(X_local_train, y_local_train)
        
        rf = RandomForestClassifier(n_estimators = args.n_estimators, max_depth = args.depth, random_state = args.random_seed)
        rf.fit(X_local_train, y_local_train)

        # Evaluate on the test set
        y_pred_xgb = xgb.predict(X_local_test)
        accuracy = accuracy_score(y_local_test, y_pred_xgb)
        acc_xgb.append(accuracy)
        
        y_pred_rf = rf.predict(X_local_test)
        accuracy = accuracy_score(y_local_test, y_pred_rf)
        acc_rf.append(accuracy)

    std_xgb = np.std(acc_xgb)
    accuracy_xgb = np.mean(acc_xgb)
    std_rf = np.std(acc_rf)
    accuracy_rf = np.mean(acc_rf)
    results = {
        'parameters': {
            'features': features,
            'n_estimators': args.n_estimators,
            'depth': args.depth,
            'random_seed': args.random_seed,
        },
        'accuracy': {
            'random_forest': {'mean': accuracy_rf, 'std': std_rf},
            'xgboost': {'mean': accuracy_xgb, 'std': std_xgb}
        }
    }
    with open(f'{args.output}', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()