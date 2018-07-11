# -*- coding: utf-8 -*-
"""
# @Time    : 2018/7/7 上午11:44
# @Author  : zhanzecheng
# @File    : model_basic.py
# @Software: PyCharm
"""

import numpy as np
import pickle
from keras.models import *
from model.snapshot import SnapshotCallbackBuilder
from sklearn.model_selection import KFold



class BasicModel:
    '''
    basic class of all models
    '''

    def __init__(self, maxLen, ocrLen, max_features, init_embedding_matrix, name='basicModel', num_flods=4, batch_size=64):
        """
        parameters initialize
        :param maxLen:
        :param max_features:
        :param init_embedding_matrix:
        """
        self.name = name
        self.ocrLen = ocrLen
        self.batch_size = batch_size
        self.maxLen = maxLen
        self.max_features = max_features
        self.embedding_matrix = init_embedding_matrix
        self.embed_size = len(init_embedding_matrix[0])

        self.num_folds =  num_flods
        self.kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=10)

        M = 3  # number of snapshots
        alpha_zero = 5e-4  # initial learning rate
        self.snap_epoch = 12
        self.snapshot = SnapshotCallbackBuilder(self.snap_epoch, M, alpha_zero)
        

        self.model = self.create_model()

    def create_model(self):
        pass



    def train_predict(self, train, train_y, test, option=3, true_length=48480):
        """
        we use KFold way to train our model and save the model
        :param train: 
        :return: 
        """
        name = self.name
        model_name = '../ckpt/' + name
        if not os.path.exists(model_name):
            os.mkdir(model_name)
        self.model.save_weights(model_name + '/init_weight.h5')

        count_kflod = 0
        predict = np.zeros((test['news'].shape[0], 3))
        oof_predict = np.zeros((train['news'].shape[0], 3))
        scores = []
        for train_index, test_index in self.kf.split(train['news']):
            
            kfold_X_train = {}
            kfold_X_valid = {}
            model_prefix = model_name + '/' + str(count_kflod)
            if not os.path.exists(model_prefix):
                os.mkdir(model_prefix)


            y_train, y_test = train_y[train_index], train_y[test_index]
            
            
            self.model.load_weights(model_name + '/init_weight.h5')
            
            for c in ['news', 'ocr']:
                kfold_X_train[c] = train[c][train_index]
                kfold_X_valid[c] = train[c][test_index]


            if option == 1:
                # 冻结embedding， 并且使用snapshot的方式来训练模型
                self.model.get_layer('embedding').trainable = False
                adam_optimizer = optimizers.Adam(lr=1e-3, clipvalue=2.0)
                self.model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
                self.model.summary()
                self.model.fit(kfold_X_train, y_train, batch_size=self.batch_size, epochs=self.snap_epoch, verbose=1,
                          validation_data=(kfold_X_valid, y_test),
                          callbacks=self.snapshot.get_callbacks(model_save_place=model_prefix))

            elif option == 2:
                # 前期冻结embedding层，训练好参数后，开放enbedding层并且使用snapshot的方式来训练模型
                self.model.get_layer('embedding').trainable = False
                adam_optimizer = optimizers.Adam(lr=1e-3, clipvalue=2)
                self.model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
                self.model.summary()
                self.model.fit(kfold_X_train, y_train, batch_size=self.batch_size, epochs=4, verbose=1,
                          validation_data=(kfold_X_valid, y_test))

                self.model.get_layer('embedding').trainable = True
                self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                self.model.fit(kfold_X_train, y_train, batch_size=self.batch_size, epochs=self.snap_epoch, verbose=1,
                               validation_data=(kfold_X_valid, y_test),
                               callbacks=self.snapshot.get_callbacks(model_save_place=model_prefix))

            else:
                # 前期冻结embedding层，训练好参数后，开放enbedding层继续训练模型
                self.model.get_layer('embedding').trainable = False
                adam_optimizer = optimizers.Adam(lr=1e-3, clipvalue=2.4)
                self.model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
                self.model.summary()
                self.model.fit(kfold_X_train, y_train, batch_size=self.batch_size, epochs=6, verbose=1,
                               validation_data=(kfold_X_valid, y_test))
                adam_optimizer = optimizers.Adam(lr=1e-4, clipvalue=1.5)

                self.model.get_layer('embedding').trainable = True
                self.model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
                self.model.fit(kfold_X_train, y_train, batch_size=self.batch_size, epochs=5, verbose=1,
                               validation_data=(kfold_X_valid, y_test))


                self.model.save_weights(model_prefix + '/' + str(count_kflod) + 'model.h5')

            evaluations = []
            for i in os.listdir(model_prefix):
                if '.h5' in i:
                    evaluations.append(i)
            print(evaluations)

            preds1 = np.zeros((test['news'].shape[0], 3))
            preds2 = np.zeros((len(kfold_X_valid['news']), 3))
            for run, i in enumerate(evaluations):
                self.model.load_weights(os.path.join(model_prefix, i))
                preds1 += self.model.predict(test, verbose=1) / len(evaluations)
                preds2 += self.model.predict(kfold_X_valid, batch_size=128) / len(evaluations)

                # model.save_weights('./ckpt/DNN_SNAP/' + str(count_kflod) + 'DNN.h5')

            # results = model.predict(test, verbose=1)

            predict += preds1 / self.num_folds
            oof_predict[test_index] = preds2

            accuracy = self.check_accuracy(oof_predict[test_index], y_test, test_index, true_length)
            print('the kflod cv is : ', str(accuracy))
            count_kflod += 1
            scores.append(accuracy)

        print('total scores is ', np.mean(scores))

        with open('../data/result/' + name + '_oof_' + str(np.mean(scores)) + '.txt', 'wb') as f:
            pickle.dump(oof_predict, f)

        with open('../data/result/' + name + '_pre_' + str(np.mean(scores)) + '.txt', 'wb') as f:
            pickle.dump(predict, f)

        print('done')

    def check_accuracy(self, pred, label, test_index, true_length):
        right = 0
        total = 0
        for count, re in enumerate(pred):
            cc = test_index[count]
            if cc >= true_length:
                continue
            total += 1
            flag = np.argmax(re)
            if int(flag) == int(np.argmax(label[count])):
                right += 1
        return right / total

class BasicStaticModel:
    """

    """
    def __init__(self, params, num_folds=5, name='BasicModel'):

        self.params = params
        self.name = name
        self.num_folds = num_folds
        self.kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=10)


    def create_model(self, kfold_X_train, y_train, kfold_X_valid, y_test, test):
        pass

    def train_predict(self, train, train_y, test, true_length=48480):
        """

        :param train:
        :param train_y:
        :param test:
        :param func:
        :param option:
        :param name:
        :return:
        """
        name = self.name
        count_kflod = 0
        model_name = '../ckpt/' + name
        if not os.path.exists(model_name):
            os.mkdir(model_name)
        predict = np.zeros((test.shape[0], 3))
        oof_predict = np.zeros((train.shape[0], 3))
        scores = []
        for train_index, test_index in self.kf.split(train):


            model_prefix = model_name + '/' + str(count_kflod)
            if not os.path.exists(model_prefix):
                os.mkdir(model_prefix)

            y_train, y_test = train_y[train_index], train_y[test_index]
            kfold_X_train, kfold_X_valid = train[train_index], train[test_index]

            pred, results, best = self.create_model(kfold_X_train, y_train, kfold_X_valid, y_test, test)

            best.save_model(model_prefix + '/' + str(count_kflod) + 'model.h5')

            accuracy_rate = self.check_accuracy(pred, y_test, test_index, true_length)

            print('Test error using softmax = {}'.format(accuracy_rate))

            predict += results / self.num_folds

            oof_predict[test_index] = pred

            scores.append(accuracy_rate)

            count_kflod += 1

        print('total scores is ', np.mean(scores))

        with open('../data/result/' + name + '_oof_' + str(np.mean(scores)) + '.txt', 'wb') as f:
            pickle.dump(oof_predict, f)

        with open('../data/result/' + name + '_pre_' + str(np.mean(scores)) + '.txt', 'wb') as f:
            pickle.dump(predict, f)

        print('done')

    def check_accuracy(self, pred, label, test_index, true_length):
        right = 0
        total = 0
        for count, re in enumerate(pred):
            cc = test_index[count]
            if cc >= true_length:
                continue
            total += 1
            flag = np.argmax(re)
            if int(flag) == int(np.argmax(label[count])):
                right += 1
        return right / total
