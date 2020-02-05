#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2018.09.19

# 线下测试与调试版本
from contextlib import contextmanager
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import gc
import math
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from reduce_memory_parallel import reduce_mem_usage_parallel
from sklearn.metrics import log_loss
import random
import time
import re
from functools import partial
import multiprocessing as mp
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def onehot_feature_process(train_data, test_data, begin_num, filter_num = 100):
    count_dict = {}
    index_dict = {}
    filter_set = set()
    begin_index = begin_num
    for d in train_data:
        if d not in count_dict.keys():
            count_dict[d] = 0
        count_dict[d] += 1
    for key in count_dict:
        if count_dict[key] < filter_num:
            filter_set.add(key)
    train_res = []
    for d in train_data:
        if d in filter_set:
            d = '-2'
        if d not in index_dict.keys():
            index_dict[d] = begin_index
            begin_index += 1
        train_res.append(index_dict[d])
    if '-2' not in  index_dict.keys():
        index_dict['-2'] = begin_index
    test1_res = []
    for d in test_data:
        if d in filter_set or d not in index_dict.keys():
            d = '-2'
        test1_res.append(index_dict[d])
    return np.array(train_res), np.array(test1_res), index_dict
def label_encode(df):
    cat_col = []
    replace = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead', 'creative_has_deeplink']
    for feat in replace:
        df[feat] = df[feat].replace([False, True], [0, 1])
    encoder = ['city', 'province', 'make', 'model', 'osv',  'adid', 'advert_id', 'orderid',
               'advert_industry_inner_one','advert_industry_inner_two', 'campaign_id', 'creative_id', 'app_cate_id',
               'app_id', 'inner_slot_id', 'advert_name', 'f_channel', 'creative_tp_dnf']
    # ,'top1','top2','top3','top4','top5','top10','advert_industry_inner',
    col_encoder = LabelEncoder()
    df_train = df[df['click']>-1]
    df_test = df[df['click'] == -1]
    for feat in encoder:
        df_train[feat],df_test[feat] ,index_dict= onehot_feature_process(df_train[feat], df_test[feat], 1, filter_num=50)
        # col_encoder.fit(df_train[feat])
        # df_train[feat] = col_encoder.transform(df_train[feat])
        # df_test[feat] = col_encoder.transform(df_test[feat])
    df = pd.concat([df_train,df_test],axis = 0)
    for feature in ['app_cate_id', 'app_id']:
        df[feature].fillna(-1, inplace=True)
    # if (mean_num != -1):
    #     df[feature].fillna(-1, inplace=True)
    # for feature in df.columns:
    #
    #     if df[feature].dtype == 'object':
    #         df[feature].fillna('-1',inplace = True)
    #         le = LabelEncoder()
    #         le.fit(df[feature])
    #         df[feature] = le.transform(df[feature])
    #         cat_col.append(feature)
    #     elif df[feature].dtype == 'bool':
    #         df[feature].fillna(-1, inplace=True)
    #         # df[feature] = df[feature].astype(int)
    #         le = LabelEncoder()
    #         le.fit(df[feature])
    #         df[feature] = le.transform(df[feature])
    #     elif feature in ['app_cate_id','app_id']:
    #         mean_num = df[feature].mean()
    #         if (mean_num != -1):
    #             df[feature].fillna(-1,inplace = True)
    return df,cat_col



def vector_feature_process(df,begin_num = 1, max_len = 30, filter_num = 100):
    count_dict = {}
    index_dict = {}
    filter_set = set()
    begin_index = begin_num
    df_train = df[df['click']>-1]['user_tags']
    df_test = df[df['click']== -1]['user_tags']
    # print "dict counting"
    for d in df_train:
        xs = d.split(',')
        for x in xs:
            if x =='':
                continue
            if x not in count_dict.keys():
                count_dict[x] = 0
            count_dict[x] += 1
    for key in count_dict:
        if count_dict[key] < filter_num:
            filter_set.add(key)

    # count_dict = sorted(count_dict.items(),key = lambda x:x[1],reverse = True)


    train_res = []
    for d in df_train:
        xs = d.split(',')
        row = [0] * max_len
        for i, x in enumerate(xs):
            if x =='':
                continue
            if x in filter_set:
                x = '-2'
            if x not in index_dict.keys():
            # if not index_dict.has_key(x):
                index_dict[x] = begin_index
                begin_index += 1
            row[i] = index_dict[x]
        train_res.append(row)
    if '-2' not in index_dict.keys():
        index_dict['-2'] = begin_index
    test_res = []
    for d in df_test:
        row = [0] * max_len
        xs = d.split(',')
        for i, x in enumerate(xs):
            if x =='':
                continue
            if x in filter_set or x not in index_dict.keys():
                x = '-2'
            row[i] = index_dict[x]
        test_res.append(row)

    return np.array(train_res),np.array(test_res), index_dict


def gen_count_dict(data, labels, begin, end):
    total_dict = {}
    pos_dict = {}
    for i, d in enumerate(data):
        if i >= begin and i < end:
            continue
        xs = d.split(',')
        if '' in xs:
            xs.remove('')
        for x in xs:
            if x not in total_dict.keys():
                total_dict[x] = 0
            if x not in pos_dict.keys():
                pos_dict[x] = 0
            total_dict[x] += 1
            if labels[i] == 1:
                pos_dict[x] += 1
    return total_dict, pos_dict


def combine_to_one(data1, data2):
    assert len(data1) == len(data2)
    new_res = []
    for i, d in enumerate(data1):

        x1 = d.split(',')
        x2 = str(data2[i])

        new_x = ''
        for xs in x1:
            if xs == '':
                continue
            tmp = xs + '|' + x2 + ','
            new_x += tmp
        new_x = new_x[0:len(new_x) - 1]
        new_res.append(new_x)
    return new_res


def click_count( train_data, count_train_data, count_labels, split_points, periods):
    print(periods, "part counting")
    # print split_points[i], split_points[i+1]
    tmp = []
    train = []
    total_dict, pos_dict = gen_count_dict(count_train_data, count_labels, split_points[periods],
                                          split_points[periods + 1])
    for j in range(split_points[periods], split_points[periods + 1]):
        xs = train_data[j].split(',')
        t = []
        for x in xs:
            if x not in pos_dict.keys():
                t.append(0)
                continue
            t.append(pos_dict[x] + 1)
        tmp.append(max(t))
    train.extend(tmp)
    train_res = pd.DataFrame({'tmp_'+str(periods): train})
    del train,tmp
    gc.collect()
    return train_res
#数据分片处理，对每片分别训练预测，然后求平均
def main(debug = False):
    num_rows = 20000 if debug else None


    aid = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_one', 'advert_industry_inner_two', 'advert_name',
           'campaign_id', 'creative_id',
           'creative_type', 'creative_tp_dnf']

    mid = ['app_cate_id', 'app_id', 'inner_slot_id']
    # 用户信息
    uid = ['user_tags']
    # 上下文信息
    tid = ['city', 'province', 'osv', 'os', 'make', 'model']
    all_id = aid + mid  + tid
    # user_id = ['user_tags', 'gd', 'ag', 'mzag', 'mzgd', 'ocpates', 'mz_ag', 'mz_gd']
    user_id = ['user_tags']
    part = 5
    with timer('process pre process '):
        round1_iflyad_train = pd.read_csv('../../data/round1_iflyad_train.txt', delimiter="\t", nrows=num_rows)
        round1_iflyad_test = pd.read_csv('../../data/round1_iflyad_test_feature.txt', delimiter="\t", nrows=num_rows)
        print(round1_iflyad_test.info(verbose=True, null_counts=True))

        round1_iflyad_test['click'] = -1
        df = pd.concat([round1_iflyad_test, round1_iflyad_train]).reset_index(drop=True)
        if part == 0:
            df['advert_industry_inner_one'] = df['advert_industry_inner'].apply(lambda x: int(x.split('_')[0]))
            df['advert_industry_inner_two'] = df['advert_industry_inner'].apply(lambda x: int(x.split('_')[1]))
            df.drop('advert_industry_inner',axis = 1,inplace = True)
            df['time_day'] = df['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
            df['time_hours'] = df['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))
            df['time_day'][df['time_day'] < 27] = df['time_day'][df['time_day'] < 27] + 31

            df['osv'] = df['osv'].fillna(str(-1))
            df['app_cate_id'] = df['app_cate_id'].fillna(-1)
            df['app_id'] = df['app_id'].fillna(-1)
            df['click'] = df['click'].fillna(-1)
            df['user_tags'] = df['user_tags'].fillna(str(-1))
            df['f_channel'] = df['f_channel'].fillna(str(-1))
            df.drop(['app_paid', 'os_name'], axis=1, inplace=True)

            # ####################机型填充#####################

            df['make'] = df['make'].fillna(str(-1))
            df['model'] = df['model'].fillna(str(-1))

            df['creative_width'] = df['creative_width']/10
            df['creative_height'] = df['creative_height']/10

            print(df.info(verbose=True, null_counts=True))

            def hasNumbers(inputString):
                return bool(re.search(r'\d', inputString))
            def convert_up_low(x):
                x = x.upper().lower()
                return x
            df['make'] = df['make'].apply(lambda x: convert_up_low(x))
            df['make_copy'] = df['make']
            df['model'] = df['model'].apply(lambda x: convert_up_low(x))
            df['has_number'] = df['make'].apply(lambda x: hasNumbers(x))
            df['tags_num'] = df['user_tags'].apply(lambda x: len(x.split(',')))

            df.loc[(df['make']!='360')&(df['make']!='-1')&(df['make']!='8848')&df['has_number'],'make'] = '-1'
            df_train = df[df.click >-1]
            df_test = df[df.click ==-1]
            phone_name = list(df_train['make'].unique())
            model_name = list(df_train['model'].unique())
            phone_name.remove('-1')
            model_name.remove('-1')
            # 建品牌名称字典
            phone_name_dict = {}
            print('建品牌名称字典')
            for i in phone_name:
                phone_name_dict[i] = []
            for index, row in df_train.iterrows():
                if row['make'] ==  row['model'] :
                    continue
                if row['make'] != '-1' and row['model'] != '-1':
                    if row['make'] in phone_name_dict.keys():
                        if row['model'] not in phone_name_dict[row['make']]:
                            phone_name_dict[row['make']].append(row['model'])
            # 根据机型填充品牌
            print('根据机型填充品牌')
            for index, row in df.iterrows():
                if index % 10000 == 0:
                    print(index)
                if row['make'] == '-1' and row['model'] == '-1':
                    continue
                if row['make_copy'] ==  row['model'] and row['make'] !='-1' and row['model'] !=-1:
                    for num,key in enumerate(phone_name_dict):
                        if row['model']  in phone_name_dict[key]:
                            df.loc[index,['make']]  = key
                            break
                    continue
                if row['make'] == '-1' and row['model'] != '-1':
                    for num,key in enumerate(phone_name_dict):
                        if row['model']  in phone_name_dict[key]:
                            df.loc[index,['make']]  = key
                            break
            # 剩余的品牌空缺通过机型包含的品牌字符进行填充
            print('剩余的品牌空缺通过机型包含的品牌字符进行填充')
            for index, row in df.iterrows():
                if index % 10000 == 0:
                    print(index)
                if row['make'] == '-1' and row['make_copy'] == '-1':
                    continue
                if row['make_copy'] !='-1' and row['make'] == '-1':
                    for num, key in enumerate(phone_name_dict):
                        if key in row['make_copy']:
                            df.loc[index, ['make']] = key
                            break

            df.loc[(df['make_copy']!='-1')&(df['make']=='-1'),'make']='-2'
            print('make fill finished')
            df, cat_col = label_encode(df)
            print(df.info(verbose=True, null_counts=True))
            df.drop(['time','make_copy','has_number'], axis=1, inplace=True)
            df.to_csv('../../data/original_filled_one_hot_data.csv', index=False)
        # ##################################################

        # begin_num = 1
        # print(df.info(verbose=True, null_counts=True))
        # df_top = property_feature(df.copy())
        # df = df.merge(df_top, on='instance_id', how='left')





        # def get_vector_feature_len(df):
        #     res = []
        #     df['user_len'] = 0
        #     for index, d in enumerate(df['user_tags']):
        #         cnt = 0
        #         d = d.split(',')
        #         if '' in d:
        #             d.remove('')
        #         cnt = len(d)
        #         res.append(cnt)
        #     res = pd.DataFrame({'feature_len': res})
        #     df = pd.concat([df, res], axis=1)
        #     return df
        #
        # df = get_vector_feature_len(df)
        ##################################通过特征重要性过滤user_tags####################################
        if part == 1:
            df = pd.read_csv('../../data/original_filled_one_hot_data.csv')
            df = df[df['click']>-1]
            # count_dict = {}
            # for d in df['user_tags']:
            #     xs = d.split(',')
            #     for x in xs:
            #         if x not in  count_dict:
            #             count_dict[x] = 0
            #         count_dict[x] += 1
            #


            # train_x = df[['city']]
            train_y=df.pop('click')
            train_y = pd.DataFrame(train_y, columns=['click'])
            train_y['click'] = train_y['click'].astype(np.float16)
            cv_encode=CountVectorizer(token_pattern = u'(?u)\\b\\w+\\b')

            feature = 'user_tags'
            cv_encode.fit(df[feature])
            train_x = cv_encode.transform(df[feature])
            cv_encode.vocabulary_.update({'-1':cv_encode.vocabulary_.pop("1")})
            # train_x = sparse.hstack((train_x, train_a))

            clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=2000,
                                       max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                                       min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True)
            # train_x = scipy.sparse.csr_matrix(train_x)
            train_x = train_x.astype(np.float32)
            clf.fit(train_x, train_y, eval_set=[(train_x,train_y)], eval_metric='logloss',early_stopping_rounds=100)
            feature_importances = clf.feature_importances_
            def get_key (dict, value):
                return [k for k, v in dict.items() if v == value]
            a = sorted(zip(range(0,train_x.shape[1]), list(clf.feature_importances_)), key=lambda x: x[1],reverse = True)
            f = open('top_uset_tags.txt','w')
            for i in range(len(feature_importances)):
                name_value =  a[i][0]
                tmp = get_key(cv_encode.vocabulary_, name_value)
                f.writelines(tmp[0]+','+str(a[i][1])+'\n')
            f.close()
            del df
            gc.collect()
            #####################################################################################
            df = pd.read_csv('../../data/original_filled_one_hot_data.csv')
            f = open('top_uset_tags.txt', 'r')
            all_name = f.readlines()
            feature_name = []
            for i in all_name:
                tmp = i.strip('\n').split(',')
                if (int(tmp[1]) > 100):
                    feature_name.append(tmp[0])
                else:
                    break

            def select_top(x):
                tmp = x.split(',')
                if '' in tmp:
                    tmp.remove('')
                str1 = ''
                for i in tmp:
                    if i in feature_name and str1 != '':
                        str1 = str1 + ',' + i
                    elif i in feature_name and str1 == '':
                        str1 = i
                x = str1
                if x == '':
                    x = '-1'
                return x

            df['user_tags'] = df['user_tags'].apply(lambda x: select_top(x))
            df.to_csv('../../data/original_filled_one_hot_data.csv', index=False)
        ##########################组合特征出现次数#############################
        if part ==2:

            # 先分割user_tags为多列
            def has_str(x):
                a = x.split(',')
                if '' in a:
                    a.remove('')
                str_bool = 0
                for i in a:
                    str_bool = bool(re.search('[a-z]', i))
                    if str_bool :
                        break
                return str_bool


            def select(x,name):
                a = x.split(',')
                for i in a:
                    num = i.rfind('_')
                    j = i[0:num]
                    if  j  == name:
                        return i
                return '-1'
            def delete(x,name):
                a = x.split(',')
                c =''
                for i in a:
                    if not bool(re.search('[a-z]', i)):
                        if c=='':
                            c = i
                        else:
                            c = c+','+i
                if c=='':
                    c = '0'
                return c

            all_id = aid + mid + tid+uid
            df = pd.read_csv('../../data/original_filled_one_hot_data.csv',  nrows=num_rows)

            original_columns = df.columns
            original_columns = [i for i in original_columns if i != 'instance_id' and i !='click'and i !='user_tags']

            # df['has_str'] = df['user_tags'].apply(lambda x: has_str(x))
            # df['gd'] = df[df['has_str']]['user_tags'].apply(lambda x:select(x,'gd'))
            # df['gd'].fillna('-1',inplace = True)
            #
            # df['ag'] = df[df['has_str']]['user_tags'].apply(lambda x: select(x, 'ag'))
            # df['ag'].fillna('-1', inplace=True)
            #
            # df['mzag'] = df[df['has_str']]['user_tags'].apply(lambda x: select(x, 'mzag'))
            # df['mzag'].fillna('-1', inplace=True)
            #
            # df['mzgd'] = df[df['has_str']]['user_tags'].apply(lambda x: select(x, 'mzgd'))
            # df['mzgd'].fillna('-1', inplace=True)
            #
            # df['ocpates'] = df[df['has_str']]['user_tags'].apply(lambda x: select(x, 'ocpates'))
            # df['ocpates'].fillna('-1', inplace=True)
            #
            # df['mz_ag'] = df[df['has_str']]['user_tags'].apply(lambda x: select(x, 'mz_ag'))
            # df['mz_ag'].fillna('-1', inplace=True)
            #
            # df['mz_gd'] = df[df['has_str']]['user_tags'].apply(lambda x: select(x, 'mz_gd'))
            # df['mz_gd'].fillna('-1', inplace=True)
            #
            # name = ['gd','ag','mzag','mzgd','ocpates','mz_ag','mz_gd']
            # df['user_tags'] = df['user_tags'].apply(lambda x: delete(x, name))

            df_train = df[df['click'] > -1].reset_index(drop=True)
            df_test = df[df['click'] == -1].reset_index(drop=True)
            df_train = reduce_mem_usage_parallel(df_train,10)
            df_test = reduce_mem_usage_parallel(df_test,10)
            def count_combine_feature_times(train_data_1, train_data_2, test_data_1, test_data_2,train_name):
                total_dict = {}
                feature_count_dict = {}
                for i, d in enumerate(train_data_1):
                    xs1 = d.split(',')
                    x2 = str(train_data_2[i])
                    for x1 in xs1:
                        ke = x1 + '|' + x2
                        if ke not in total_dict.keys():
                            total_dict[ke] = 0
                        total_dict[ke] += 1
                    a =1
                for i, d in enumerate(test_data_1):
                    xs1 = d.split(',')
                    x2 =  str(test_data_2[i])
                    for x1 in xs1:
                        ke = x1 + '|' + x2
                        if ke not in total_dict.keys():
                            total_dict[ke] = 0
                        total_dict[ke] += 1

                for key in total_dict:
                    if total_dict[key] not in feature_count_dict.keys():
                        feature_count_dict[total_dict[key]] = 0
                    feature_count_dict[total_dict[key]] += 1

                train_res = []
                for i, d in enumerate(train_data_1):
                    t = []
                    xs1 = d.split(',')
                    x2 =  str(train_data_2[i])
                    for x1 in xs1:
                        ke = x1 + '|' + x2
                        t.append(total_dict[ke])
                    train_res.append(max(t))
                test_res = []
                for i, d in enumerate(test_data_1):
                    t = []
                    xs1 = d.split(',')
                    x2 =  str(test_data_2[i])
                    for x1 in xs1:
                        ke = x1 + '|' + x2
                        t.append(total_dict[ke])
                    test_res.append(max(t))

                train_res = pd.DataFrame({train_name: train_res})
                test_res = pd.DataFrame({train_name: test_res})
                del total_dict
                gc.collect()
                return train_res, test_res, feature_count_dict
            for user_fea in user_id:
                for feature in all_id:
                    print(feature, ' times counting')
                    train, test,  feature_count_dict = count_combine_feature_times(df_train[user_fea], df_train[feature],df_test[user_fea], df_test[feature],user_fea+'_' +feature+'_times')
                    train[user_fea+'_' + feature + '_times'] = train[user_fea+'_' + feature + '_times'].apply(lambda x:int(math.log(1 + x * x)))
                    test[user_fea+'_' + feature + '_times'] = test[user_fea+'_' + feature + '_times'].apply(lambda x: int(math.log(1 + x * x)))
                    df_train = pd.concat([df_train,train],axis = 1)
                    df_test = pd.concat([df_test, test], axis = 1)
                    log_f_dict = {}
                    for key in feature_count_dict:
                        new_key = int(math.log(1 + key * key))
                        if new_key not in log_f_dict.keys():
                            log_f_dict[new_key] = 0
                        log_f_dict[new_key] += feature_count_dict[key]
                    del train,test,feature_count_dict
                    gc.collect()
            df = pd.concat([df_train,df_test],axis = 0)
            print(df.info(verbose=True, null_counts=True))
            # df.drop(original_columns,axis = 1,inplace = True)
            df.to_csv('../../data/original_count_combine_feature_times.csv',index = False)
        ##########################正样本组合特征出现次数#############################
        if part ==3:


            # 先分割user_tags为多列
            def has_str(x):
                a = x.split(',')
                if '' in a:
                    a.remove('')
                str_bool = 0
                for i in a:
                    str_bool = bool(re.search('[a-z]', i))
                    if str_bool :
                        break
                return str_bool


            def select(x,name):
                a = x.split(',')
                for i in a:
                    num = i.rfind('_')
                    j = i[0:num]
                    if  j  == name:
                        return i
                return '-1'
            def delete(x,name):
                a = x.split(',')
                c =''
                for i in a:
                    if not bool(re.search('[a-z]', i)):
                        if c=='':
                            c = i
                        else:
                            c = c+','+i
                return c

            df = pd.read_csv('../../data/original_count_combine_feature_times.csv', nrows=num_rows)

            print(df.info(verbose=True, null_counts=True))
            df_train = df[df['click'] > -1].reset_index(drop=True)
            df_test = df[df['click'] == -1].reset_index(drop=True)
            df_train = reduce_mem_usage_parallel(df_train,10)
            df_test = reduce_mem_usage_parallel(df_test,10)

            def count_pos_feature(train_data, test1_data,labels, k, train_name,test_only=False,):
                # 正样本特征出现次数统计
                nums = len(train_data)
                last = nums

                interval = last // k
                split_points = []
                for i in range(k):
                    split_points.append(i * interval)
                split_points.append(last)
                count_train_data = train_data[0:last]
                count_labels = labels[0:last]
                t0 = time.time()
                train_array = np.ones(nums)* -1
                if not test_only:
                    periods = [0,1,2,3,4]
                    func = partial(click_count,train_data,count_train_data,count_labels,split_points)
                    pool = mp.Pool()
                    train = pool.map(func,periods)
                    pool.close()
                    pool.join()
                for i in range(len(train)):
                    a = int(train[i].columns.values[0][-1:])
                    train_array[split_points[a]:split_points[a + 1]] = train[i].values.reshape(1,-1)

                total_dict, pos_dict = gen_count_dict(count_train_data, count_labels, 1, 0)
                count_dict = {-1: 0}
                for key in pos_dict:
                    if pos_dict[key] not in  count_dict.keys():
                        count_dict[pos_dict[key]] = 0
                    count_dict[pos_dict[key]] += 1

                test = []
                for d in test1_data:
                    xs = d.split(',')
                    t = []
                    for x in xs:
                        if x  not in  pos_dict.keys():
                            t.append(0)
                            continue
                        t.append(pos_dict[x] + 1)
                    test.append(max(t))
                train_res = pd.DataFrame({train_name: train_array})
                test_res = pd.DataFrame({train_name: test})
                print(train_name,"____","done in {:.0f}s".format(time.time() - t0))
                del pos_dict,total_dict,train_array,test,split_points
                gc.collect()


                return train_res, test_res, count_dict


            for user_fea in user_id:
                for feature in all_id:
                    print(feature, ' click times counting')
                    new_train_data = combine_to_one(df_train[user_fea], df_train[feature])
                    new_test_data = combine_to_one(df_test[user_fea], df_test[feature])
                    train, test,  feature_count_dict = count_pos_feature(new_train_data, new_test_data,df_train['click'], 5,user_fea+'_' +feature+'_click_times')
                    train[user_fea+'_' + feature + '_click_times'] = train[user_fea+'_' +feature+'_click_times'].apply(lambda x:int(math.log(1 + x * x)))
                    test[user_fea+'_' + feature + '_click_times'] = test[user_fea+'_' +feature+'_click_times'].apply(lambda x: int(math.log(1 + x * x)))
                    df_train = pd.concat([df_train,train],axis = 1)
                    df_test = pd.concat([df_test, test], axis = 1)
                    log_f_dict = {}
                    for key in feature_count_dict:
                        new_key = int(math.log(1 + key * key))
                        if new_key not in log_f_dict.keys():
                            log_f_dict[new_key] = 0
                        log_f_dict[new_key] += feature_count_dict[key]
                    del train,test,feature_count_dict,new_train_data,new_test_data
                    gc.collect()
            # delete_id = ['has_str', 'gd', 'ag', 'mzag', 'mzgd', 'ocpates', 'mz_ag', 'mz_gd']
            # df.drop(original_columns, axis=1, inplace=True)
            # df.drop(delete_id, axis=1, inplace=True)
            df = pd.concat([df_train,df_test],axis = 0)
            df.to_csv('../../data/original_count_pos_feature.csv', index=False)
            # df,cat_col = label_encode(df)
            # df.drop(['time'],axis = 1,inplace = True)

        ##########################样本组合特征编码#############################
        if part == 4:
            def onehot_combine_process(train_data_1, train_data_2, test1_data_1, test1_data_2, begin_num,train_name, filter_num=100):
                count_dict = {}
                index_dict = {}
                filter_set = set()
                begin_index = begin_num
                for i, d in enumerate(train_data_1):
                    id_1 = str(train_data_1[i])
                    id_2 = str(train_data_2[i])
                    t_id = id_1 + '|' + id_2
                    if t_id not in count_dict.keys():
                        count_dict[t_id] = 0
                    count_dict[t_id] += 1
                for key in count_dict:
                    if count_dict[key] < filter_num:
                        filter_set.add(key)
                train = []
                for i, d in enumerate(train_data_1):
                    id_1 = str(train_data_1[i])
                    id_2 = str(train_data_2[i])
                    t_id = id_1 + '|' + id_2
                    if t_id in filter_set:
                        t_id = '-2'
                    if t_id not in index_dict.keys():
                        index_dict[t_id] = begin_index
                        begin_index += 1
                    train.append(index_dict[t_id])
                if '-2'  not in  index_dict.keys():
                    index_dict['-2'] = begin_index
                test = []
                for i, d in enumerate(test1_data_1):
                    id_1 = str(test1_data_1[i])
                    id_2 = str(test1_data_1[i])
                    t_id = id_1 + '|' + id_2
                    if t_id in filter_set or t_id not in index_dict.keys():
                        t_id = '-2'
                    test.append(index_dict[t_id])

                train = pd.DataFrame({train_name: train})
                test = pd.DataFrame({train_name: test})
                del begin_index,count_dict
                gc.collect()
                return train, test

            aid = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_one', 'advert_industry_inner_two',
                   'advert_name',
                   'campaign_id', 'creative_id',
                   'creative_type', 'creative_tp_dnf']

            mid = ['app_cate_id', 'app_id', 'inner_slot_id']
            # 用户信息
            uid = ['user_tags']
            # 上下文信息
            tid = ['city', 'province', 'osv', 'os', 'make', 'model']
            df = pd.read_csv('../../data/original_filled_one_hot_data.csv', nrows=num_rows)

            df_train = df[df['click'] > -1].reset_index(drop=True)
            df_test = df[df['click'] == -1].reset_index(drop=True)
            df_train = reduce_mem_usage_parallel(df_train, 10)
            df_test = reduce_mem_usage_parallel(df_test, 10)
            f1_id = mid+tid
            for f1 in f1_id:
                for f2 in aid:
                    print(f1 +'_'+f2+'_one_hot')
                    t0 = time.time()
                    train, test = onehot_combine_process(df_train[f1], df_train[f2],
                                                                                     df_test[f1], df_test[f2],
                                                                                     1,f1 +'_'+f2+'_one_hot')
                    df_train = pd.concat([df_train, train], axis=1)
                    df_test = pd.concat([df_test, test], axis=1)
                    del train,test
                    gc.collect()
                    print(f1 +'_'+f2+'_one_hot', "____", "done in {:.0f}s".format(time.time() - t0))
            df = pd.concat([df_train, df_test], axis=0)
            df.to_csv('../../data/original_onehot_combine_feature.csv', index=False)

        if part == 5:
            def write_data_into_parts(data, root_path, nums=5100000):
                l = data.shape[0] // nums
                for i in range(l + 1):
                    begin = i * nums
                    end = min(nums * (i + 1), data.shape[0])
                    t_data = data[begin:end]
                    t_data.tofile(root_path + '_' + str(i) + '.bin')

            def write_dict(data_path, data):
                fw = open(data_path, 'w')
                for key in data:
                    fw.write(str(key) + ',' + str(data[key]) + '\n')
                fw.close()

            def get_vector_feature_len(data):
                res = []
                for d in data:
                    cnt = 0
                    for item in d:
                        if item != 0:
                            cnt += 1
                    res.append(cnt)
                return np.array(res)
            df = pd.read_csv('../../data/original_count_combine_feature_times.csv',
                             nrows=num_rows)

            orginal_col = [i for i in df.columns if i != 'instance_id']
            df = df[['instance_id', 'user_tags', 'click']]

            ##########################向量特征#############################
            for feature in df.columns:
                if feature != 'user_tags':
                    continue
                def splitx(x):
                    if isinstance(x, str):
                        return x.split(',')
                tmp = df['user_tags'].apply(splitx).values
                max_len = 0
                for i in tmp:
                    if len(i) >max_len:
                        max_len = len(i)
            begin_num = 1
            train_res,test_res,f_dict = vector_feature_process(df,begin_num, max_len)
            write_dict('../../data/user_tags_vector_dict.csv', f_dict)
            write_data_into_parts(train_res, '../../data/user_tags_vector_feature_train', nums=200000)
            write_data_into_parts(test_res, '../../data/user_tags_vector_feature_test', nums=200000)
            train_res_lengths = get_vector_feature_len(train_res)
            # test1_res_lengths = get_vector_feature_len(test1_res)
            test2_res_lengths = get_vector_feature_len(test_res)

            write_data_into_parts(train_res_lengths,'../../data/user_tags_vector_length_train', nums=200000)
            write_data_into_parts(test2_res_lengths,'../../data/user_tags_vector_length_test', nums=200000)
            # df_top = property_feature(df.copy())
            # df = df.merge(df_top, on='instance_id', how='left')
            # df.drop(['user_tags'])
            print(df.info(verbose=True, null_counts=True))

            a = 1
# def somefunc(str_1, str_2, iterable_iterm):
#     print("%s %s %d" % (str_1, str_2, iterable_iterm))
#
#
# def a():
#     iterable = [1, 2, 3, 4, 5]
#     pool = Pool()
#     str_1 = "This"
#     str_2 = "is"
#     func = partial(somefunc, str_1, str_2)
#     pool.map(func, iterable)
#     pool.close()
#     pool.join()
if __name__ == "__main__":


    with timer("Full model run"):
        main(debug = False)
