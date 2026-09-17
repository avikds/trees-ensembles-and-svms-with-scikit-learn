# Trees, Ensembles and SVMs with Scikit-Learn

Chapters 8 and 9 of An Introduction to Statistical Learning with the real library. On the Boston housing data you grow a regression tree, prune it along the cost-complexity path with cross-validation, then beat it with bagging, random forests tuned by out-of-bag error, and gradient boosting whose number of trees is read off a staged test curve, reading feature importances and partial dependence along the way. On the OJ purchase data you tune linear, radial and polynomial support vector classifiers with GridSearchCV, count support vectors, and compare kernels by test error and ROC AUC. Every method ends in one honest table.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_boston
- [x] **2.** regression_tree
- [x] **3.** cost_complexity_pruning
- [x] **4.** bagging_and_forests
- [x] **5.** feature_importance
- [x] **6.** gradient_boosting
- [x] **7.** early_stopping_and_partial_dependence
- [x] **8.** oj_data
- [x] **9.** linear_svc
- [x] **10.** kernels
- [x] **11.** final_tables

## Results

```
Boston: 506 neighborhoods, 12 features; train 379 / test 127

depth-3 tree: 8 leaves, root split on lstat at 8.13; test RMSE 5.458
unpruned tree: 362 leaves, train RMSE 0.0, test RMSE 5.384
cost-complexity path: 352 subtrees; CV picks alpha=0.118 with 35 leaves, test RMSE 5.021

bagging (200 trees): OOB RMSE 3.222, test RMSE 4.106
random forest OOB RMSE by max_features: 1.0=3.217, 0.33=2.908, sqrt=2.908  -> 0.33, test RMSE 4.729
  impurity importance: lstat 0.29, rm 0.27, ptratio 0.07, crim 0.07
  permutation importance (test): lstat 0.29, rm 0.26, dis 0.04, ptratio 0.03

gradient boosting, best stage on the test curve: rate 0.01: 500 trees, RMSE 4.087, rate 0.1: 335 trees, RMSE 3.649
histogram boosting with early stopping: stopped at 251 iterations, test RMSE 4.638
partial dependence: rm 5.4->7.6 rooms moves value -3.0->8.6; lstat 3.5->27.1 moves it 8.9->-6.7

OJ: 1070 purchases, 17 features, train 800 / test 270
linear SVC: C=0.01 uses 436 support vectors (train error 0.1625), C=10 uses 332 (train error 0.16)
  linear best {'svc__C': 1}: train error 0.1588, test error 0.1741, AUC 0.8991
  rbf    best {'svc__C': 1, 'svc__gamma': 0.01}: train error 0.1575, test error 0.1593, AUC 0.9038
  poly   best {'svc__C': 10, 'svc__degree': 3}: train error 0.1425, test error 0.1778, AUC 0.8863

Boston test RMSE (thousand dollars):
  boosting   rmse=  3.667 setting=379
  bagging    rmse=  4.106 setting=200
  hist       rmse=  4.638 setting=251
  forest     rmse=  4.729 setting=0.33
  tree       rmse=  5.021 setting=35
OJ test error by kernel:
  rbf        test_error=0.1593 auc=0.9038 support_vectors=410
  linear     test_error=0.1741 auc=0.8991 support_vectors=334
  poly       test_error=0.1778 auc=0.8863 support_vectors=338
```
