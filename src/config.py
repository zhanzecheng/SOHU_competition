# -*- coding: utf-8 -*-
"""
# @Time    : 2018/7/7 下午3:38
# @Author  : zhanzecheng
# @File    : config.py
# @Software: PyCharm
"""
from model.attention_model import AttentionModel
from model.covlstm_model import CovlstmModel
from model.dpcnn_model import DpcnnModel
from model.lstmcov_model import LstmCovModel
from model.lstmgru_model import LstmgruModel
from model.textCNN_model import TextCnnModel
from model.capsule_model import CapsuleModel
from model.catboost_model import CatboostModel
# from model.lightgbm_model import LightGbmModel
from model.xgboost_model import XgboostModel

class Config:

    def __init__(self):

        self.model = {
            'model1' : AttentionModel,
            'model2' : CovlstmModel,
            'model3' : DpcnnModel,
            'model4' : LstmCovModel,
            'model5' : LstmgruModel,
            'model6' : TextCnnModel,
            'model7' : CapsuleModel,
            'model8' : CatboostModel,
            # 'model9' : LightGbmModel,
            'model10' : XgboostModel
        }

        self.EMBED_SIZES = 300
        self.MAX_LEN = 1000
        self.OCR_LEN = 400
        self.BATCH_SIZE = 64
        self.EPOCH = 2


        self.TEXT_X = '../data/News_cut_test_text.txt'
        self.ITEM_TO_ID = '../data/item_to_id_small.pkl'
        self.ID_TO_ITEM = '../data/id_to_item_small.pkl'
        self.EMBEDDING_FILE = "../data/chinese_txt"
        self.TRAIN_X = '../data/All_cut_train_text.txt'
        self.OCR_TRAIN_X = '../data/All_ocr_train_cut_text.txt'
        self.OCR_TEST_X = '../data/All_ocr_test_cut_text.txt'
        self.TEST_FILE = '../data/All_cut_test_text.txt'
        self.FEATURES_FILE = '../data/train_features.pkl'
        self.FEATURES_test_FILE = '../data/test_features.pkl'
        self.OCR_FEATURES_FILE = '../data/ocr_train_features.pkl'
        self.OCR_FEATURES_test_FILE = '../data/ocr_test_features.pkl'
        self.STOP_WORD_FILE = '../data/stopword.txt'
        self.ORIGIN_LABEL_FILE = '../data/News_pic_label_train.txt'
