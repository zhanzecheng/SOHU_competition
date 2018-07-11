# -*- coding: utf-8 -*-
"""
# @Time    : 2018/7/6 下午2:46
# @Author  : zhanzecheng
# @File    : covlstm_model.py
# @Software: PyCharm
"""

from keras.models import *
from keras.layers import *
from model.model_basic import BasicModel


class CovlstmModel(BasicModel):
    def __init__(self, maxLen, ocrLen, max_features, init_embedding_matrix, name='basicModel', num_flods=4, batch_size=64):
        BasicModel.__init__(self, maxLen, ocrLen, max_features, init_embedding_matrix, name='CovLstm', num_flods=num_flods,
                            batch_size=batch_size)
    def create_model(self):
        main_input = Input(shape=(self.maxLen,), name='news')
        embedding = Embedding(self.max_features, self.embed_size, weights=[self.embedding_matrix], trainable=False, name='embedding')
        x = embedding(main_input)
        x = SpatialDropout1D(0.5)(x)
        x = Conv1D(256, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
        x = Bidirectional(LSTM(60, return_sequences=True))(x)
        x = Bidirectional(GRU(60, return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)

        recurrent_units = 60
        ocr_input = Input(shape=(self.ocrLen,),  name='ocr')
        ocr_embedding_layer = embedding(ocr_input)
        ocr_embedding_layer = SpatialDropout1D(0.25)(ocr_embedding_layer)
        ocr_rnn_1 = Bidirectional(GRU(recurrent_units // 2, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(
            ocr_embedding_layer)
        ocr_rnn_2 = Bidirectional(GRU(recurrent_units // 2, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(
            ocr_rnn_1)
        ocr_maxpool = GlobalMaxPooling1D()(ocr_rnn_2)
        ocr_average = GlobalAveragePooling1D()(ocr_rnn_2)

        concat2 = concatenate([avg_pool, max_pool, ocr_maxpool, ocr_average], axis=-1)
        concat2 = Dropout(0.5)(concat2)
        dense2 = Dense(3, activation="softmax")(concat2)
        res_model = Model(inputs=[main_input, ocr_input], outputs=dense2)
        # res_model = Model(inputs=[main_input], outputs=main_output)
        return res_model



