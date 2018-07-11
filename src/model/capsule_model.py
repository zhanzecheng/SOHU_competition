# -*- coding: utf-8 -*-
"""
# @Time    : 2018/7/9 上午9:24
# @Author  : zhanzecheng
# @File    : capsule_model.py
# @Software: PyCharm
"""

from keras.layers import *
from keras.models import *
from model.model_basic import BasicModel
from model.model_component import Capsule

class CapsuleModel(BasicModel):
    def __init__(self, maxLen, ocrLen, max_features, init_embedding_matrix, name='basicModel', num_flods=4, batch_size=64):
        BasicModel.__init__(self, maxLen, ocrLen, max_features, init_embedding_matrix, name='CapsuleModel', num_flods=num_flods,
                            batch_size=batch_size)
    def create_model(self):
        Routings = 5
        Num_capsule = 10
        Dim_capsule = 16
        dropout_p = 0.5

        main_input = Input(shape=(self.maxLen,), name='news')
        embedding = Embedding(self.max_features, self.embed_size, weights=[self.embedding_matrix], trainable=False, name='embedding')
        x = embedding(main_input)
        x = SpatialDropout1D(0.5)(x)
        x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
        capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                          share_weights=True)(x)

        ocr_input = Input(shape=(self.ocrLen,),  name='ocr')
        ocr_embedding_layer = embedding(ocr_input)
        ocr_embedding_layer = SpatialDropout1D(0.25)(ocr_embedding_layer)
        ocr_rnn_1 = Bidirectional(GRU(recurrent_units // 2, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(
            ocr_embedding_layer)
        ocr_rnn_2 = Bidirectional(GRU(recurrent_units // 2, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(
            ocr_rnn_1)
        ocr_maxpool = GlobalMaxPooling1D()(ocr_rnn_2)
        ocr_average = GlobalAveragePooling1D()(ocr_rnn_2)

        capsule = Flatten()(capsule)
        capsule = concatenate([capsule, ocr_maxpool, ocr_average], axis=1)
        capsule = Dropout(dropout_p)(capsule)
        dense2 = Dense(3, activation="softmax")(capsule)
        res_model = Model(inputs=[main_input, ocr_input], outputs=dense2)
        return res_model