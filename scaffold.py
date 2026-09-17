"""
Trees, Ensembles and SVMs with Scikit-Learn scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Trees, Ensembles and SVMs with scikit-learn (ISL, chapters 8 and 9).

Story: on Boston, grow a regression tree, prune it along the cost-complexity path
with cross-validation, then beat it with bagging, a random forest tuned by
out-of-bag error, and gradient boosting whose tree count is read off a staged
curve, with importances and partial dependence along the way; on the OJ purchase
data, tune linear, radial and polynomial support vector classifiers with
GridSearchCV; finish with one test table for each.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def main() -> None:
    df = load_boston()
    X, y = split_xy(df, "medv")
    X_tr, X_te, y_tr, y_te = train_test(X, y)
    cv = KFold(5, shuffle=True, random_state=0)
    names = list(X.columns)
    print(f"Boston: {len(df)} neighborhoods, {X.shape[1]} features; train {len(X_tr)} / test {len(X_te)}")

    # ---- 1. One tree, then pruning ----
    full = fit_tree(X_tr, y_tr)
    small = fit_tree(X_tr, y_tr, max_depth=3)
    s = tree_summary(small, names)
    print(f"\ndepth-3 tree: {s['n_leaves']} leaves, root split on {s['root_feature']} at {s['root_threshold']}; test RMSE {rmse(small, X_te, y_te)}")
    print(f"unpruned tree: {tree_summary(full, names)['n_leaves']} leaves, train RMSE {rmse(full, X_tr, y_tr)}, test RMSE {rmse(full, X_te, y_te)}")
    alphas, leaves = pruning_path(X_tr, y_tr)
    pruned = pruned_tree(X_tr, y_tr, cv)
    print(f"cost-complexity path: {len(alphas)} subtrees; CV picks alpha={pruned.ccp_alpha:.3f} with {pruned.get_n_leaves()} leaves, test RMSE {rmse(pruned, X_te, y_te)}")

    # ---- 2. Bagging and random forests ----
    bag = bagging(X_tr, y_tr)
    oob = oob_by_max_features(X_tr, y_tr, [1.0, 0.33, 'sqrt'])
    best_mf = min(oob, key=oob.get)
    rf = random_forest(X_tr, y_tr, best_mf)
    print(f"\nbagging (200 trees): OOB RMSE {oob_rmse(bag, y_tr)}, test RMSE {rmse(bag, X_te, y_te)}")
    print("random forest OOB RMSE by max_features: " + ", ".join(f"{k}={v}" for k, v in oob.items()) + f"  -> {best_mf}, test RMSE {rmse(rf, X_te, y_te)}")
    imp = importances(rf, names)
    pi = permutation_importances(rf, X_te, y_te, names, n_repeats=5)
    print("  impurity importance: " + ", ".join(f"{k} {v:.2f}" for k, v in list(imp.items())[:4]))
    print("  permutation importance (test): " + ", ".join(f"{k} {v:.2f}" for k, v in list(pi.items())[:4]))

    # ---- 3. Boosting ----
    cmp = learning_rate_comparison(X_tr, y_tr, X_te, y_te, [0.01, 0.1])
    print("\ngradient boosting, best stage on the test curve: " + ", ".join(f"rate {r}: {n} trees, RMSE {e}" for r, (n, e) in cmp.items()))
    hb = hist_boosting(X_tr, y_tr)
    print(f"histogram boosting with early stopping: stopped at {iterations_used(hb)} iterations, test RMSE {rmse(hb, X_te, y_te)}")
    gb = boosting(X_tr, y_tr, n_estimators=200)
    g_rm, p_rm = partial_dependence_curve(gb, X_tr, "rm", grid_resolution=6)
    g_ls, p_ls = partial_dependence_curve(gb, X_tr, "lstat", grid_resolution=6)
    print(f"partial dependence: rm {g_rm[0]:.1f}->{g_rm[-1]:.1f} rooms moves value {p_rm[0]:.1f}->{p_rm[-1]:.1f}; lstat {g_ls[0]:.1f}->{g_ls[-1]:.1f} moves it {p_ls[0]:.1f}->{p_ls[-1]:.1f}")

    # ---- 4. SVMs on OJ ----
    oj = load_oj()
    Xo_tr, Xo_te, yo_tr, yo_te = oj_split(oj)
    scv = StratifiedKFold(5, shuffle=True, random_state=0)
    wide, narrow = svc_pipeline('linear', C=0.01).fit(Xo_tr, yo_tr), svc_pipeline('linear', C=10.0).fit(Xo_tr, yo_tr)
    print(f"\nOJ: {len(oj)} purchases, {Xo_tr.shape[1]} features, train {len(Xo_tr)} / test {len(Xo_te)}")
    print(f"linear SVC: C=0.01 uses {n_support_vectors(wide)} support vectors (train error {error_rate(wide, Xo_tr, yo_tr)}), C=10 uses {n_support_vectors(narrow)} (train error {error_rate(narrow, Xo_tr, yo_tr)})")
    kc = kernel_comparison(Xo_tr, yo_tr, Xo_te, yo_te, scv)
    for k, v in kc.items():
        print(f"  {k:6s} best {v['best_params']}: train error {v['train_error']}, test error {v['test_error']}, AUC {v['auc']}")

    # ---- 5. Tables ----
    table = regression_table(X_tr, y_tr, X_te, y_te, cv)
    print("\nBoston test RMSE (thousand dollars):")
    for line in format_rows(table, "rmse"):
        print("  " + line)
    print("OJ test error by kernel:")
    for line in svm_rows(kc):
        print("  " + line)


if __name__ == "__main__":
    main()

