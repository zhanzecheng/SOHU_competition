# -*- coding: utf-8 -*-
"""
# @Time    : 2018/7/6 下午2:53
# @Author  : zhanzecheng
# @File    : dpcnn_model.py
# @Software: PyCharm
"""
from keras.models import *
from keras.layers import *
from model.model_basic import BasicModel

filter_nr = 64
filter_size = 3
max_pool_size = 3
max_pool_strides = 2
dense_nr = 128
spatial_dropout = 0.5
dense_dropout = 0.5

class DpcnnModel(BasicModel):
    def __init__(self, maxLen, ocrLen, max_features, init_embedding_matrix, name='basicModel', num_flods=4, batch_size=64):
        BasicModel.__init__(self, maxLen, ocrLen, max_features, init_embedding_matrix, name='dpcnn', num_flods=num_flods,
                            batch_size=batch_size)
    def create_model(self):
        main_input = Input(shape=(self.maxLen,), name='news')
        embedding = Embedding(self.max_features, self.embed_size, weights=[self.embedding_matrix], trainable=False, name='embedding')
        x = embedding(main_input)
        block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(x)
        block1 = BatchNormalization()(block1)
        block1 = PReLU()(block1)
        block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1)
        block1 = BatchNormalization()(block1)
        block1 = PReLU()(block1)

        # we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
        # if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
        resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear')(x)
        resize_emb = PReLU()(resize_emb)

        block1_output = add([block1, resize_emb])
        block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

        block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1_output)
        block2 = BatchNormalization()(block2)
        block2 = PReLU()(block2)
        block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2)
        block2 = BatchNormalization()(block2)
        block2 = PReLU()(block2)

        block2_output = add([block2, block1_output])
        block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

        block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2_output)
        block3 = BatchNormalization()(block3)
        block3 = PReLU()(block3)
        block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3)
        block3 = BatchNormalization()(block3)
        block3 = PReLU()(block3)

        block3_output = add([block3, block2_output])
        block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

        block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3_output)
        block4 = BatchNormalization()(block4)
        block4 = PReLU()(block4)
        block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block4)
        block4 = BatchNormalization()(block4)
        block4 = PReLU()(block4)

        output = add([block4, block3_output])
        output = GlobalMaxPooling1D()(output)
        output = Dense(dense_nr, activation='linear')(output)
        output = BatchNormalization()(output)
        output = PReLU()(output)

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

        output = concatenate([output, ocr_maxpool, ocr_average], axis=-1)

        output = Dropout(dense_dropout)(output)
        dense2 = Dense(3, activation="softmax")(output)
        res_model = Model(inputs=[main_input], outputs=dense2)
        return res_model

