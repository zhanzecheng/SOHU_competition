# -*- coding: utf-8 -*-
"""
# @Time    : 2018/7/9 上午9:48
# @Author  : zhanzecheng
# @File    : lightgbm_model.py
# @Software: PyCharm
"""

import lightgbm as lgbm
from model.model_basic import BasicStaticModel

class LightGbmModel(BasicStaticModel):
    def __init__(self, num_folds=5):
        lgbm_params = {'objective': 'multiclass',
                       'bagging_seed': 10,
                       'boosting_type': 'gbdt',
                       'feature_fraction': 0.9,
                       'feature_fraction_seed': 10,
                       'lambda_l1': 0.5,
                       'lambda_l2': 0.5,
                       'learning_rate': 0.01,
                       'metric': 'multi_logloss',
                       'min_child_weight': 1,
                       'min_split_gain': 0,
                       'min_sum_hessian_in_leaf': 0.1,
                       'num_leaves': 64,
                       'num_thread': -1,
                       'num_class': 3,
                       'verbose': 1}
        BasicStaticModel.__init__(self, lgbm_params, num_folds, 'lightGBM')

    def create_model(self, kfold_X_train, y_train, kfold_X_valid, y_test, test):

        dtrain = lgbm.Dataset(kfold_X_train, label=y_train)
        dwatch = lgbm.Dataset(kfold_X_valid, label=y_test)

        best = lgbm.train(self.params, dtrain, num_boost_round=4000, verbose_eval=100, valid_sets=dwatch,
                          early_stopping_rounds=100)
        # 对验证集predict

        pred = best.predict(kfold_X_valid)
        results = best.predict(test)

        return pred, results, best