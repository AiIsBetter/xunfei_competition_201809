#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2018.09.19
import pandas as pd
import numpy as np
import gc

class FeatureDictionary(object):
    def __init__(self, trainfile=None, testfile=None,
                 dfTrain=None, dfTest=None, numeric_cols=[], ignore_cols=[]):
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        assert not ((testfile is None) and (dfTest is None)), "testfile or dfTest at least one is set"
        assert not ((testfile is not None) and (dfTest is not None)), "only one can be set"
        self.trainfile = trainfile
        self.testfile = testfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        if self.dfTrain is None:
            dfTrain = pd.read_csv(self.trainfile)
        else:
            dfTrain = self.dfTrain
        if self.dfTest is None:
            dfTest = pd.read_csv(self.testfile)
        else:
            dfTest = self.dfTest
        df = pd.concat([dfTrain, dfTest])
        self.feat_dict = {}
        self.feat_dict_size = {}
        for col in df.columns:

            if col in self.ignore_cols:
                continue
            us = df[col].unique()
            self.feat_dict[col] = dict(zip(us, range(0, len(us))))
            self.feat_dict_size[col] = len(us)
            a = 1
        # self.feat_dim = tc


class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, infile=None, df=None, has_label=False):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"
        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)
        if has_label:
            y = dfi["click"]
            dfi.drop(["instance_id", "click"], axis=1, inplace=True)
        else:
            ids = dfi["instance_id"].values.tolist()
            dfi.drop(["instance_id"], axis=1, inplace=True)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        # dfv = dfi.copy()
        # for col in dfi.columns:
        #     if col in self.feat_dict.ignore_cols:
        #         dfi.drop(col, axis=1, inplace=True)
        #         dfv.drop(col, axis=1, inplace=True)
        #         continue
        #     if col in self.feat_dict.numeric_cols:
        #         dfi[col] = self.feat_dict.feat_dict[col]
        #         dfv[col] = (dfv[col] - dfv[col].min()) / (dfv[col].max() - dfv[col].min())
        #         # dfi[col] = dfi[col].astype(np.float32)
        #     else:
        #         dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
        #         dfv[col] = 1.
        # print(dfi.info(verbose=True, null_counts=True))

        a = dfi.columns.tolist()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                continue
            dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
        # Xi = dfi.values.tolist()
        # del dfi
        # gc.collect()

        if has_label:
            return dfi,  y
        else:
            return dfi,  ids

