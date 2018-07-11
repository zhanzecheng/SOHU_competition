# -*- coding: utf-8 -*-
"""
# @Time    : 2018/7/7 下午7:39
# @Author  : zhanzecheng
# @File    : ensemble_stacking.py
# @Software: PyCharm
"""

import pickle
import glob
from keras.utils import np_utils
from keras.layers import *
from model.snapshot import SnapshotCallbackBuilder
from keras.models import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

TRAIN_X = '../data/All_train_file.txt'
TEXT_X = '../data/' + 'News_cut_validate_text.txt'
def data_prepare():
    oof_filename = []
    test_filename = []

    # load oof train and oof test
    filenames = glob.glob('../data/result/*oof*')
    for filename in filenames:
        oof_filename.append(filename)
        test_filename.append(filename.replace('oof', 'test'))

    oof_data = []
    test_data = []

    for tra, tes in zip(oof_filename, test_filename):
        with open(tra, 'rb') as f:
            oof_data.extend(pickle.load(f))
        with open(tes, 'rb') as f:
            test_data.extend(pickle.load(f))

    # load text feature
    train_y = []



    with open(TRAIN_X, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split('\t')
            label = int(line[2])
            train_y.append(label)

    with open('../data/' + 'test_features.pkl', 'rb') as f:
        test_features = pickle.load(f)
    with open('../data/features.pkl', 'rb') as f:
        features = pickle.load(f)

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    test_features = scaler.transform(test_features)


    train_y = np_utils.to_categorical(train_y)

    with open('../data/train_x_250.pkl', 'rb') as f:
        train_x = pickle.load(f)

    with open('../data/' + 'test_x_250.pkl', 'rb') as f:
        test_x = pickle.load(f)

    length = len(train_x)

    assert length == 48480
    train_x = np.concatenate((train_x, features, oof_data))
    test_x = np.concatenate((test_x, test_features, test_data))

    train = {}
    test = {}
    train['news'] = train_x
    test['news'] = test_x
    return train, train_y, test


def get_model(train_x):
    input_shape = Input(shape=(train_x.shape[1],), name='news')
    x = Dense(256, activation='relu')(input_shape)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(3, activation="softmax")(x)
    res_model = Model(inputs=[input_shape], outputs=x)
    return res_model

def check_accuracy(pred, label, test_index):
    right = 0
    total = 0
    for count, re in enumerate(pred):
        cc = test_index[count]
        if cc >= 48480:
            continue
        total += 1
        flag = np.argmax(re)
        if int(flag) == int(np.argmax(label[count])):
            right += 1
    return right / total

BATCH_SIZE = 64

# 第一次stacking
def stacking_first(train, train_y, test):
    savepath = './stack_/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    count_kflod = 0
    num_folds = 6
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10)
    predict = np.zeros((test['news'].shape[0], 3))
    oof_predict = np.zeros((train['news'].shape[0], 3))
    scores = []
    for train_index, test_index in kf.split(train['news']):

        kfold_X_train = {}
        kfold_X_valid = {}

        y_train, y_test = train_y[train_index], train_y[test_index]

        for c in ['news']:
            kfold_X_train[c] = train[c][train_index]
            kfold_X_valid[c] = train[c][test_index]

        test_watch = []
        test_label = []
        for i in test_index:
            if i < 48480:
                test_watch.append(train[i])
                test_label.append(train_y[i])
        test_watch = np.array(test_watch)
        test_label = np.array(test_label)

        model_prefix = savepath + 'DNN' + str(count_kflod)
        if not os.path.exists(model_prefix):
            os.mkdir(model_prefix)

        M = 4  # number of snapshots
        alpha_zero = 1e-3  # initial learning rate
        snap_epoch = 16
        snapshot = SnapshotCallbackBuilder(snap_epoch, M, alpha_zero)

        res_model = get_model(train['news'])
        res_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # res_model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1,  class_weight=class_weight)
        res_model.fit(kfold_X_train, y_train, batch_size=BATCH_SIZE, epochs=snap_epoch, verbose=1,
                      validation_data=(test_watch, test_label),
                      callbacks=snapshot.get_callbacks(model_save_place=model_prefix))

        evaluations = []
        for i in os.listdir(model_prefix):
            if '.h5' in i:
                evaluations.append(i)

        preds1 = np.zeros((test['news'].shape[0], 3))
        preds2 = np.zeros((len(kfold_X_valid['news']), 3))
        for run, i in enumerate(evaluations):
            res_model.load_weights(os.path.join(model_prefix, i))
            preds1 += res_model.predict(test, verbose=1) / len(evaluations)
            preds2 += res_model.predict(kfold_X_valid, batch_size=128) / len(evaluations)

        predict += preds1 / num_folds
        oof_predict[test_index] = preds2

        accuracy = check_accuracy(oof_predict[test_index], y_test, test_index)
        print('the kflod cv is : ', str(accuracy))
        count_kflod += 1
        scores.append(accuracy)
    print('total scores is ', np.mean(scores))
    return predict

# 使用pseudo-labeling做第二次stacking
def stacking_pseudo(train, train_y, test, results):
    answer = np.zeros((results.shape[0], 1))
    for count in range(len(results)):
        answer[count] = np.argmax(results[count])
    answer = np_utils.to_categorical(answer)
    train_y = np.concatenate([train_y, answer], axis=0)
    train['news'] = np.concatenate([train['news'], test['news']], axis=0)


    savepath = './pesudo_/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    count_kflod = 0
    num_folds = 6
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10)
    predict = np.zeros((test['news'].shape[0], 3))
    oof_predict = np.zeros((train['news'].shape[0], 3))
    scores = []
    for train_index, test_index in kf.split(train['news']):

        kfold_X_train = {}
        kfold_X_valid = {}

        y_train, y_test = train_y[train_index], train_y[test_index]

        for c in ['news']:
            kfold_X_train[c] = train[c][train_index]
            kfold_X_valid[c] = train[c][test_index]

        test_watch = []
        test_label = []
        for i in test_index:
            if i < 48480:
                test_watch.append(train[i])
                test_label.append(train_y[i])
        test_watch = np.array(test_watch)
        test_label = np.array(test_label)

        model_prefix = savepath + 'DNN' + str(count_kflod)
        if not os.path.exists(model_prefix):
            os.mkdir(model_prefix)

        M = 4  # number of snapshots
        alpha_zero = 1e-3  # initial learning rate
        snap_epoch = 16
        snapshot = SnapshotCallbackBuilder(snap_epoch, M, alpha_zero)

        res_model = get_model(train['news'])
        res_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # res_model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1,  class_weight=class_weight)
        res_model.fit(kfold_X_train, y_train, batch_size=BATCH_SIZE, epochs=snap_epoch, verbose=1,
                      validation_data=(test_watch, test_label),
                      callbacks=snapshot.get_callbacks(model_save_place=model_prefix))

        evaluations = []
        for i in os.listdir(model_prefix):
            if '.h5' in i:
                evaluations.append(i)
        print(evaluations)

        preds1 = np.zeros((test['news'].shape[0], 3))
        preds2 = np.zeros((len(kfold_X_valid['news']), 3))
        for run, i in enumerate(evaluations):
            res_model.load_weights(os.path.join(model_prefix, i))
            preds1 += res_model.predict(test, verbose=1) / len(evaluations)
            preds2 += res_model.predict(kfold_X_valid, batch_size=128) / len(evaluations)

        predict += preds1 / num_folds
        oof_predict[test_index] = preds2

        accuracy = check_accuracy(oof_predict[test_index], y_test, test_index)
        print('the kflod cv is : ', str(accuracy))
        count_kflod += 1
        scores.append(accuracy)
    print('total scores is ', np.mean(scores))
    return predict

def save_result(predict):
    with open('../data/pickle.pkl', 'wb') as f:
        pickle.dump(predict, f)

    results = predict
    count_zero = 0
    count_two = 0
    count_one = 0
    with open(TEXT_X, 'r', encoding='utf-8') as f, open('../data/' + 'result.txt', 'w', encoding='utf-8') as d:
        lines = f.readlines()
        for count, line in enumerate(lines):
            line = line.strip()
            line = line.split('\t')
            id = line[0]
            flag = np.argmax(results[count])
            if flag == 1:
                count_one += 1
            elif flag == 0:
                count_zero += 1
            elif flag == 2:
                count_two += 1
            d.write(id + '\t' + str(flag) + '\t' + 'NULL' + '\t' + 'NULL')
            d.write('\n')
    print(count_one)
    print(count_one / len(results))
    print(count_zero / len(results))
    print(count_two / len(results))

if __name__ == '__main__':
    train, train_y, test = data_prepare()
    predicts = stacking_first(train, train_y, test)
    predicts = stacking_pseudo(train, train_y, test, predicts)
    save_result(predicts)