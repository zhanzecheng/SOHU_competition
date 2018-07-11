# -*- coding: utf-8 -*-
"""
# @Time    : 2018/7/7 上午11:22
# @Author  : zhanzecheng
# @File    : textCNN_model.py
# @Software: PyCharm
"""

from keras.layers import *
from keras.models import *
from model.model_basic import BasicModel

class TextCnnModel(BasicModel):
    def __init__(self, maxLen, ocrLen, max_features, init_embedding_matrix, name='basicModel', num_flods=4, batch_size=64):
        BasicModel.__init__(self, maxLen, ocrLen, max_features, init_embedding_matrix, name='textCNN', num_flods=num_flods,
                            batch_size=batch_size)
    def create_model(self):
        main_input = Input(shape=(self.maxLen,), name='news')
        embedding = Embedding(self.max_features, self.embed_size, weights=[self.embedding_matrix], trainable=False, name='embedding')
        x = embedding(main_input)
        # cnn0 = Convolution1D(embed_size, 3, padding="same", strides=2, activation='relu')(x)
        # cnn0 = MaxPool1D(pool_size=4)(cnn0)
        cnn1 = Conv1D(self.embed_size, 4, padding="valid", strides=1, activation='relu')(x)
        cnn1 = MaxPooling1D(pool_size=self.maxLen - 4 + 1)(cnn1)
        cnn1 = Flatten()(cnn1)
        cnn2 = Conv1D(self.embed_size, 5, padding="valid", strides=1, activation='relu')(x)
        cnn2 = MaxPooling1D(pool_size=self.maxLen - 5 + 1)(cnn2)
        cnn2 = Flatten()(cnn2)
        cnn3 = Conv1D(self.embed_size, 6, padding="valid", strides=1, activation='relu')(x)
        cnn3 = MaxPooling1D(pool_size=self.maxLen - 6 + 1)(cnn3)
        cnn3 = Flatten()(cnn3)

        recurrent_units = 60
        ocr_input = Input(shape=(self.ocrLen,), name='ocr')
        ocr_embedding_layer = embedding(ocr_input)
        ocr_embedding_layer = SpatialDropout1D(0.25)(ocr_embedding_layer)
        ocr_rnn_1 = Bidirectional(GRU(recurrent_units // 2, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(
            ocr_embedding_layer)
        ocr_rnn_2 = Bidirectional(GRU(recurrent_units // 2, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(
            ocr_rnn_1)
        ocr_maxpool = GlobalMaxPooling1D()(ocr_rnn_2)
        ocr_average = GlobalAveragePooling1D()(ocr_rnn_2)

        all_views = concatenate([cnn1, cnn2, cnn3, ocr_maxpool, ocr_average], axis=1)
        x = Dropout(0.5)(all_views)
        dense2 = Dense(3, activation="softmax")(x)
        res_model = Model(inputs=[main_input], outputs=dense2)
        return res_model
