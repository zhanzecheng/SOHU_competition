# -*- coding: utf-8 -*-
"""
# @Time    : 2018/7/6 上午11:50
# @Author  : zhanzecheng
# @File    : attention_model.py
# @Software: PyCharm
"""
from keras.layers import *
from keras.models import *
from model.model_component import AttentionWeightedAverage
from model.model_basic import BasicModel
from keras.utils.vis_utils import plot_model

class AttentionModel(BasicModel):
    def __init__(self, maxLen, ocrLen, max_features, init_embedding_matrix, name='basicModel', num_flods=4, batch_size=64):
        BasicModel.__init__(self, maxLen, ocrLen, max_features, init_embedding_matrix, name='Attention', num_flods=num_flods, batch_size=batch_size)
    def create_model(self):
        recurrent_units = 60
        main_input = Input(shape=(self.maxLen,), name='news')
        embedding = Embedding(self.max_features, self.embed_size, weights=[self.embedding_matrix], trainable=False, name='embedding')
        embedding_layer = embedding(main_input)
        embedding_layer = SpatialDropout1D(0.25)(embedding_layer)

        rnn_1 = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(
            embedding_layer)
        x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(rnn_1)
        # x = concatenate([rnn_1, rnn_2], axis=2)

        last = Lambda(lambda t: t[:, -1], name='last')(x)
        maxpool = GlobalMaxPooling1D()(x)
        attn = AttentionWeightedAverage()(x)
        average = GlobalAveragePooling1D()(x)


        ocr_input = Input(shape=(self.ocrLen,),  name='ocr')
        ocr_embedding_layer = embedding(ocr_input)
        ocr_embedding_layer = SpatialDropout1D(0.25)(ocr_embedding_layer)
        ocr_rnn_1 = Bidirectional(GRU(recurrent_units // 2, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(
            ocr_embedding_layer)
        ocr_rnn_2 = Bidirectional(GRU(recurrent_units // 2, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(ocr_rnn_1)
        ocr_maxpool = GlobalMaxPooling1D()(ocr_rnn_2)
        ocr_attn = AttentionWeightedAverage()(ocr_rnn_2)


        all_views = concatenate([last, maxpool, attn, average, ocr_maxpool, ocr_attn], axis=1)
        x = Dropout(0.5)(all_views)
        dense2 = Dense(3, activation="softmax")(x)
        res_model = Model(inputs=[main_input, ocr_input], outputs=dense2)
        plot_model(model, to_file="model.png", show_shapes=True)
        # res_model = Model(inputs=[main_input], outputs=main_output)
        return res_model
