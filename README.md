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

---

Built on Deep-ML.
