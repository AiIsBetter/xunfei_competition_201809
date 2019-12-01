#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2018.09.19
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
import config
from NN_ffm import NNFM
from DataReader import FeatureDictionary, DataParser
sys.path.append("..")
import gc
from reduce_memory_parallel import reduce_mem_usage_parallel

def _load_data(debug,max_len):
    if debug :
        nrows = 10000
    else:
        nrows = None
    dfTrain = pd.read_csv(config.TRAIN_FILE,nrows = nrows)
    dfTest = pd.read_csv(config.TEST_FILE,nrows = nrows)

    cols = [c for c in dfTrain.columns if c not in ["instance_id", "click"]]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain["click"].values
    X_test = dfTest[cols].values
    ids_test = dfTest["instance_id"].values

    def read_data_from_bin_to_dict(path, col_name, part,nrows,sizes=None):
        data_arr = {}
        dynamic_array = {}
        if sizes != None:
            data_arr[col_name] = np.fromfile(path + col_name + '_' + str(part) + '.bin',count = nrows, dtype=np.int).reshape(-1,sizes)
        else:
            data_arr[col_name] = np.fromfile(path + col_name + '_' + str(part) + '.bin',count = nrows, dtype=np.int)
        return data_arr

    dynamic_index_dict_train = {}
    dynamic_array = []
    xx = 1 if debug else 6
    bin_nrows = nrows*max_len if debug else -1
    for part in range(xx):
        dynamic_array.append( read_data_from_bin_to_dict('../../data/', config.DYNAMIC_CATEGORICAL_COLS+'_vector_feature_train', part, bin_nrows,max_len))
    dynamic_index_dict_train[config.DYNAMIC_CATEGORICAL_COLS+ '_vector_feature_train'] = np.concatenate([item[ config.DYNAMIC_CATEGORICAL_COLS+'_vector_feature_train'] for item in dynamic_array], axis=0)

    dynamic_index_dict_test = {}
    dynamic_array = []
    for part in range(1):
        dynamic_array.append( read_data_from_bin_to_dict('../../data/', config.DYNAMIC_CATEGORICAL_COLS+'_vector_feature_test', part,bin_nrows, max_len))
    dynamic_index_dict = {}
    dynamic_index_dict_test[config.DYNAMIC_CATEGORICAL_COLS+ '_vector_feature_test'] = np.concatenate([item[ config.DYNAMIC_CATEGORICAL_COLS+'_vector_feature_test'] for item in dynamic_array], axis=0)

    f = open('../../data/'+config.DYNAMIC_CATEGORICAL_COLS+'_vector_dict.csv', 'r')
    dynamic_featrue_size = len(f.readlines())
    f.close()
    # if debug :
    #     dynamic_index_dict_train['user_tags_vector_feature_train'] = dynamic_index_dict_train['user_tags_vector_feature_train'][0:10000]
    #     dynamic_index_dict_test['user_tags_vector_feature_test'] = dynamic_index_dict_test['user_tags_vector_feature_test'][0:10000]

    dynamic_length_array = []
    dynamic_length_train = {}
    xx = 1 if debug else 6
    bin_nrows = nrows if debug else -1
    for part in range(xx):
        dynamic_length_array.append( read_data_from_bin_to_dict('../../data/', config.DYNAMIC_CATEGORICAL_COLS+'_vector_length_train', part, bin_nrows,None))
    dynamic_length_train[ config.DYNAMIC_CATEGORICAL_COLS+'_vector_length_train'] = np.concatenate([item[config.DYNAMIC_CATEGORICAL_COLS+ '_vector_length_train'] for item in dynamic_length_array], axis=0)


    dynamic_length_array = []
    dynamic_length_test = {}
    for part in range(1):
        dynamic_length_array.append( read_data_from_bin_to_dict('../../data/',config.DYNAMIC_CATEGORICAL_COLS+ '_vector_length_test', part,bin_nrows, None))
    dynamic_length_test[ config.DYNAMIC_CATEGORICAL_COLS+'_vector_length_test'] = np.concatenate([item[config.DYNAMIC_CATEGORICAL_COLS+ '_vector_length_test'] for item in dynamic_length_array], axis=0)





    return dfTrain, dfTest,dynamic_index_dict_train,dynamic_index_dict_test, dynamic_featrue_size,dynamic_length_train,dynamic_length_test,\
           X_train, y_train, X_test, ids_test


def _run_base_model_dfm(dfTrain, dfTest, dynamic_index_dict_train,dynamic_index_dict_test,dynamic_featrue_size,
                        dynamic_length_train, dynamic_length_test,folds, dfm_params):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train,  y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test,  ids_test = data_parser.parse(df=dfTest)
    print('convet finshed!')
    dfm_params["static_feature_size"] = fd.feat_dict_size
    dfm_params["field_size"] = Xi_train.shape[1]+1

    y_train_meta = np.zeros((dfTrain.shape[0]), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    del dfTrain, dfTest
    gc.collect()
    _get = lambda x, l: [x[i] for i in l]
    gini_results_cv = np.zeros(len(folds), dtype=float)
    # gini_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    # gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    for i, (train_idx, valid_idx) in enumerate(folds):
        # Xi_train_, y_train_ = _get(Xi_train, train_idx),  _get(y_train, train_idx)
        # Xi_valid_,  y_valid_ = _get(Xi_train, valid_idx), _get(y_train, valid_idx)

        Xi_train_, y_train_ = Xi_train.iloc[train_idx], y_train.iloc[train_idx]
        Xi_valid_, y_valid_ = Xi_train.iloc[valid_idx], y_train.iloc[valid_idx]
        Xi_dynamic,Xi_dynamic_valid = dynamic_index_dict_train[config.DYNAMIC_CATEGORICAL_COLS+'_vector_feature_train'][train_idx], \
                                      dynamic_index_dict_train[config.DYNAMIC_CATEGORICAL_COLS+'_vector_feature_train'][valid_idx]
        Xi_dynamic_length, Xi_dynamic_length_valid = dynamic_length_train[config.DYNAMIC_CATEGORICAL_COLS+'_vector_length_train'][train_idx], \
                                                     dynamic_length_train[config.DYNAMIC_CATEGORICAL_COLS+'_vector_length_train'][valid_idx]
        nffm = NNFM(**nffm_params)

        nffm.fit(Xi_train_,  y_train_, Xi_valid_, y_valid_,Xi_dynamic,Xi_dynamic_valid,
                 Xi_dynamic_length, Xi_dynamic_length_valid,early_stopping=True)

        # y_train_meta = y_train_meta.reshape(-1,1)
        y_train_meta[valid_idx] = nffm.predict(Xi_valid_,Xi_dynamic_valid,Xi_dynamic_length_valid).reshape(-1)
        y_test_meta[:,0] += nffm.predict(Xi_test,dynamic_index_dict_test[config.DYNAMIC_CATEGORICAL_COLS+'_vector_feature_test'],
                                         dynamic_length_test[config.DYNAMIC_CATEGORICAL_COLS+'_vector_length_test']).reshape(-1)

        gini_results_cv[i] = log_loss(y_valid_, y_train_meta[valid_idx])
        # gini_results_epoch_train[i] = dfm.train_result
        # gini_results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(len(folds))


    clf_str = 'NFFM'
    print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, gini_results_cv.mean(), gini_results_cv.std())
    _make_submission(ids_test, y_test_meta, filename)

    # _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    return y_train_meta, y_test_meta


def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"instance_id": ids, "predicted_score": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s"%model_name)
    plt.legend(legends)
    plt.savefig("./fig/%s.png"%model_name)
    plt.close()

debug = False
max_len = 367
# load data
dfTrain, dfTest, dynamic_index_dict_train,dynamic_index_dict_test,\
dynamic_featrue_size,dynamic_length_train,dynamic_length_test,X_train, y_train, X_test, ids_test = _load_data(debug,max_len)

# folds
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(dfTrain, dfTrain[['click']]))


# ------------------ NFFM Model ------------------
# params
nffm_params = {
    # "use_fm": True,
    # "use_deep": True,
    "embedding_size": 4,

    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5,0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 1,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.005,
    "verbose": True,
    "eval_metric": log_loss,
    "greater_is_better":False,
    "random_seed": config.RANDOM_SEED,
    "static_categorical_col":config.STATIC_CATEGORICAL_COLS,
    "dynamic_categorical_col":config.DYNAMIC_CATEGORICAL_COLS,
    "dynamic_featrue_size":dynamic_featrue_size,
    "dynamic_max_len":max_len
}
# nffm_params = {
#
# }
dfTrain = reduce_mem_usage_parallel(dfTrain,10)
dfTest = reduce_mem_usage_parallel(dfTest,10)
print('reduce finish!')
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest,dynamic_index_dict_train,dynamic_index_dict_test,dynamic_featrue_size,
                                               dynamic_length_train, dynamic_length_test,folds, nffm_params)




