# -*- coding: utf-8 -*-
"""
# @Time    : 2018/7/9 上午9:58
# @Author  : zhanzecheng
# @File    : catboost_model.py
# @Software: PyCharm
"""
from catboost import CatBoostClassifier
from model.model_basic import BasicStaticModel

class CatboostModel(BasicStaticModel):
    def __init__(self, num_folds=5):
        params = {}
        BasicStaticModel.__init__(self, params, num_folds, 'catboost')

    def create_model(self, kfold_X_train, y_train, kfold_X_valid, y_test, test):


        best = CatBoostClassifier(loss_function='MultiClassOneVsAll', learning_rate=0.07940735491731761, depth=8)
        best.fit(kfold_X_train, y_train)

        # 对验证集predict
        pred = best.predict_proba(kfold_X_valid)
        results = best.predict_proba(test)

        return pred, results, best