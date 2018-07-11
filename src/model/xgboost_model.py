# -*- coding: utf-8 -*-
"""
# @Time    : 2018/7/9 上午9:24
# @Author  : zhanzecheng
# @File    : xgboost_model.py
# @Software: PyCharm
"""
import xgboost as xgb
from model.model_basic import BasicStaticModel

class XgboostModel(BasicStaticModel):
    def __init__(self, num_folds=5):

        xgb_params = {
            'updater': 'grow_gpu',
            'booster': 'gbtree',
            'lambda': 0.1,
            'gamma': 0.7,
            'max_depth': 7,
            'nthread': -1,
            'subsample': 0.5,
            'silent': 0,
            'eta': 0.01,
            'scale_pos_weight': 1,
            'gpu_id': 0,
            'alpha': 0.1,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'objective': 'multi:softprob',
            'num_class': 3,
        }
        BasicStaticModel.__init__(self, xgb_params, num_folds, 'xgboost')
    def create_model(self, kfold_X_train, y_train, kfold_X_valid, y_test, test):
        dtrain = xgb.DMatrix(kfold_X_train, label=y_train)
        dvalid = xgb.DMatrix(kfold_X_valid, label=y_test)

        ddtest = xgb.DMatrix(test)

        dwatch = xgb.DMatrix(kfold_X_valid, label=y_test)

        watchlist = [(dtrain, 'train'), (dwatch, 'test')]

        best= xgb.train(self.params, dtrain, 2000, watchlist, verbose_eval=100, early_stopping_rounds=150)
        # 对验证集predict

        pred = best.predict(dvalid)
        results = best.predict(ddtest)

        return pred, results, best