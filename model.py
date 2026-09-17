"""
Trees, Ensembles and SVMs with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_boston
import os
import tempfile
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

ISLP_BASE = "https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/"

def load_islp(name):
    # Save the dataset in the system temporary directory.
    path = os.path.join(tempfile.gettempdir(), f"islp_{name}.csv")

    # Download the file only if it is not already present.
    if not os.path.exists(path):
        url = f"{ISLP_BASE}{name}.csv"
        urllib.request.urlretrieve(url, path)

    # Load and return the dataset as a pandas DataFrame.
    return pd.read_csv(path)

def load_boston():
    # Load the Boston housing dataset.
    return load_islp("Boston")

def split_xy(df, target):
    # Separate the target column from the feature columns.
    X = df.drop(columns=[target])
    y = df[target]

    return X, y

def train_test(X, y, test_size=0.25, random_state=0):
    # Split the data into training and test sets.
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

# Step 2 - regression_tree
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

def fit_tree(X, y, max_depth=None, ccp_alpha=0.0):
    # Fit a regression tree with a fixed random state for reproducibility.
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        ccp_alpha=ccp_alpha,
        random_state=0
    )

    return tree.fit(X, y)

def tree_summary(tree, feature_names):
    # Get the index of the feature used for the root split.
    root_feature_idx = tree.tree_.feature[0]

    # Translate the feature index into its feature name.
    root_feature = feature_names[root_feature_idx]

    # Get the threshold used at the root split.
    root_threshold = round(tree.tree_.threshold[0], 3)

    return {
        "n_leaves": tree.get_n_leaves(),
        "depth": tree.get_depth(),
        "root_feature": root_feature,
        "root_threshold": root_threshold
    }

def rmse(model, X, y):
    # Compute predictions and return the root mean squared error.
    predictions = model.predict(X)
    return round(np.sqrt(mean_squared_error(y, predictions)), 3)

# Step 3 - cost_complexity_pruning
from sklearn.model_selection import cross_val_score

def pruning_path(X, y):
    # Compute the cost-complexity pruning path.
    tree = DecisionTreeRegressor(random_state=0)
    path = tree.cost_complexity_pruning_path(X, y)

    # Exclude the final alpha, which produces the root-only tree.
    alphas = path.ccp_alphas[:-1]

    # Count the leaves for the tree fitted at each alpha.
    n_leaves = [
        fit_tree(X, y, ccp_alpha=alpha).get_n_leaves()
        for alpha in alphas
    ]

    return alphas, n_leaves

def cv_prune(X, y, alphas, cv):
    # Compute cross-validated mean squared error for each pruning alpha.
    mses = []

    for alpha in alphas:
        scores = cross_val_score(
            fit_tree(X, y, ccp_alpha=alpha).__class__(
                max_depth=None,
                ccp_alpha=alpha,
                random_state=0
            ),
            X,
            y,
            cv=cv,
            scoring="neg_mean_squared_error"
        )

        # Negate the sklearn scores to obtain MSE and round to 2 decimals.
        mses.append(round(-np.mean(scores), 2))

    return mses

def best_alpha(alphas, mses):
    # Return the alpha corresponding to the smallest cross-validated MSE.
    return alphas[int(np.argmin(mses))]

def pruned_tree(X, y, cv):
    # Build a reduced alpha grid using the specified path slicing rule.
    alphas, _ = pruning_path(X, y)
    step = max(1, len(alphas) // 30)
    grid = alphas[::step]

    # Evaluate the candidate alphas by cross-validation.
    mses = cv_prune(X, y, grid, cv)

    # Fit the final tree using the alpha with the lowest MSE.
    alpha = best_alpha(grid, mses)

    return fit_tree(X, y, ccp_alpha=alpha)

# Step 4 - bagging_and_forests
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

def bagging(X, y, n_estimators=200):
    # Fit a bagging ensemble using decision trees as the base estimator.
    model = BaggingRegressor(
        DecisionTreeRegressor(random_state=0),
        n_estimators=n_estimators,
        oob_score=True,
        random_state=0
    )

    return model.fit(X, y)

def random_forest(X, y, max_features, n_estimators=200):
    # Fit a random forest with the specified maximum number/fraction of features.
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        oob_score=True,
        random_state=0
    )

    return model.fit(X, y)

def oob_rmse(model, y):
    # Compute RMSE using the out-of-bag predictions.
    return round(
        np.sqrt(mean_squared_error(y, model.oob_prediction_)),
        3
    )

def oob_by_max_features(X, y, options):
    # Fit a random forest for each max_features option and record its OOB RMSE.
    return {
        option: oob_rmse(random_forest(X, y, option), y)
        for option in options
    }

