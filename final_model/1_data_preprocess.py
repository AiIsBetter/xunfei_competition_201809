#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2018.09.19


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
    encoder = [ 'make', 'model', 'adid', 'advert_id', 'orderid',
               'advert_industry_inner_one','advert_industry_inner_two', 'campaign_id', 'creative_id', 'app_cate_id',
               'app_id', 'inner_slot_id', 'advert_name', 'f_channel', 'creative_tp_dnf']
    # ,'top1','top2','top3','top4','top5','top10','advert_industry_inner',
    col_encoder = LabelEncoder()
    df_train = df[df['click']>-1]
    df_test = df[df['click'] == -1]
    for feat in encoder:
        df_train[feat],df_test[feat] ,index_dict= onehot_feature_process(df_train[feat], df_test[feat], 1, filter_num=50)

    df = pd.concat([df_train,df_test],axis = 0)
    for feature in ['app_cate_id', 'app_id']:
        df[feature].fillna(-1, inplace=True)

    return df,cat_col
def osv_split(osv,osv1,osv2,osv3,split_name):
    a = osv.split(split_name)
    if len(a) == 1:
        osv1 = int(a[0])
        osv2 = -1
        osv3 = -1
    elif len(a) == 2:
        osv1 = int(a[0])
        osv2 = int(a[1])
        osv3 = -1
    elif len(a) == 3:
        osv1 = int(a[0])
        osv2 = int(a[1])
        osv3 = int(a[2])
    return osv1,osv2,osv3

#数据分片处理，对每片分别训练预测，然后求平均
def main(debug = False,part = 0):
    num_rows = 10000 if debug else None

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

    with timer('process pre process '):


        if part == 0:
            round1_iflyad_train = pd.read_csv('../data/round1_iflyad_train.txt', delimiter="\t", nrows=num_rows)
            round1_iflyad_test = pd.read_csv('../data/round1_iflyad_test_feature.txt', delimiter="\t", nrows=num_rows)
            # print(round1_iflyad_train.info(verbose=True, null_counts=True))

            round1_iflyad_test['click'] = -1
            df = pd.concat([round1_iflyad_test, round1_iflyad_train]).reset_index(drop=True)
            df['advert_industry_inner_one'] = df['advert_industry_inner'].apply(lambda x: int(x.split('_')[0]))
            df['advert_industry_inner_two'] = df['advert_industry_inner'].apply(lambda x: int(x.split('_')[1]))

            df.drop('advert_industry_inner',axis = 1,inplace = True)
            df['time_day'] = df['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
            df['time_hours'] = df['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))
            # 0 - 6 > --1 | 7 - 12 > --2 | 13 - 18 > --3 | 19 - 24 > --4
            df.loc[(df['time_hours']>=0) & (df['time_hours']<=6),'time_hours'] = 1
            df.loc[(df['time_hours'] >= 7) & (df['time_hours'] <= 12), 'time_hours'] = 2
            df.loc[(df['time_hours'] >= 13) & (df['time_hours'] <= 18), 'time_hours'] = 3
            df.loc[(df['time_hours'] >= 19) & (df['time_hours'] <= 24), 'time_hours'] = 4

            df['time_day'][df['time_day'] < 27] = df['time_day'][df['time_day'] < 27] + 31
            count = 0
            city_arrays = []
            #########################city细分##############################
            print('city细分')
            df['city'] = df['city'].apply(lambda x: str(x))
            df['province'] = df['province'].apply(lambda x: str(x))
            df['city_province']='0'
            df['city_city'] = '0'
            df['city_zone'] = '0'
            df['province_zone'] = '0'
            for index, row in df.iterrows():
                if index % 10000 == 0:
                    print(index)
                city_array = {}
                city_array['city_province'] = row['city'][5:7]
                city_array['city_city'] = row['city'][7:9]
                city_array['city_zone'] = row['city'][9:11]
                city_array['province_zone'] = row['province'][9:11]
                city_arrays.append(city_array)
            city_arrays = pd.DataFrame(city_arrays)
            df.drop(['city', 'province','city_province','city_city','city_zone','province_zone'], axis=1, inplace=True)
            df = pd.concat([df,city_arrays],axis = 1)

            ######################################系统版本号细分############################################
            print('系统版本号细分')
            df['osv'] = df['osv'].fillna(str(-1))
            df['osv1'] = -1
            df['osv2'] = -1
            df['osv3'] = -1
            osv_arrays = []
            for index, row in df.iterrows():
                if index % 10000 == 0:
                    print(index)
                osv_array = {}

                if len(row['osv'])>=10 or 'iOS' in row['osv']:
                    # print(row['osv'])
                    if 'android' in row['osv']:
                        row['osv'] = row['osv'].strip('android')
                        if '_' in row['osv']:
                            osv1, osv2, osv3 = osv_split(row['osv'], row['osv1'], row['osv2'], row['osv3'], '_')
                            osv_array['osv1'] = osv1
                            osv_array['osv2'] = osv2
                            osv_array['osv3'] = osv3
                        elif '.' in row['osv']:
                            osv1, osv2, osv3 = osv_split(row['osv'], row['osv1'], row['osv2'], row['osv3'], '.')
                            osv_array['osv1'] = osv1
                            osv_array['osv2'] = osv2
                            osv_array['osv3'] = osv3
                        else:
                            row['osv1'] = -1
                            row['osv2'] = -1
                            row['osv3'] = -1
                    elif 'Android' in row['osv']:
                        # print(row['osv'])
                        row['osv'] = row['osv'].strip('Android')
                        if '_' in row['osv']:
                            osv1, osv2, osv3 = osv_split(row['osv'], row['osv1'], row['osv2'], row['osv3'], '_')
                            osv_array['osv1'] = osv1
                            osv_array['osv2'] = osv2
                            osv_array['osv3'] = osv3
                        elif '.' in row['osv']:
                            osv1, osv2, osv3 = osv_split(row['osv'], row['osv1'], row['osv2'], row['osv3'], '.')
                            osv_array['osv1'] = osv1
                            osv_array['osv2'] = osv2
                            osv_array['osv3'] = osv3
                        else:
                            osv_array['osv1'] = -1
                            osv_array['osv2'] = -1
                            osv_array['osv3'] = -1
                    elif 'iPhone OS' in row['osv']:
                        if 'iPhone OS' in row['osv']:
                            row['osv'] = row['osv'].strip('iPhone OS')
                            if '_' in row['osv']:
                                osv1, osv2, osv3 = osv_split(row['osv'], row['osv1'], row['osv2'], row['osv3'], '_')
                                osv_array['osv1'] = osv1
                                osv_array['osv2'] = osv2
                                osv_array['osv3'] = osv3
                            elif '.' in row['osv']:
                                osv1, osv2, osv3 = osv_split(row['osv'], row['osv1'], row['osv2'], row['osv3'], '.')
                                osv_array['osv1'] = osv1
                                osv_array['osv2'] = osv2
                                osv_array['osv3'] = osv3
                            else:
                                osv_array['osv1'] = -1
                                osv_array['osv2'] = -1
                                osv_array['osv3'] = -1
                    elif 'iOS' in row['osv']:
                        if 'iPhone OS' in row['osv']:
                            row['osv'] = row['osv'].strip('iPhone OS')
                            if '_' in row['osv']:
                                osv1, osv2, osv3 = osv_split(row['osv'], row['osv1'], row['osv2'], row['osv3'], '_')
                                osv_array['osv1'] = osv1
                                osv_array['osv2'] = osv2
                                osv_array['osv3'] = osv3
                            elif '.' in row['osv']:
                                osv1, osv2, osv3 = osv_split(row['osv'], row['osv1'], row['osv2'], row['osv3'], '.')
                                osv_array['osv1'] = osv1
                                osv_array['osv2'] = osv2
                                osv_array['osv3'] = osv3
                            else:
                                osv_array['osv1'] = -1
                                osv_array['osv2'] = -1
                                osv_array['osv3'] = -1
                    osv_arrays.append(osv_array)
                    continue

                if '_' in row['osv']:
                    osv1, osv2, osv3  = osv_split(row['osv'], row['osv1'], row['osv2'], row['osv3'], '_')
                    osv_array['osv1'] = osv1
                    osv_array['osv2'] = osv2
                    osv_array['osv3'] = osv3
                elif '.' in row['osv']:
                    osv1, osv2, osv3  = osv_split(row['osv'], row['osv1'], row['osv2'], row['osv3'], '.')
                    osv_array['osv1'] = osv1
                    osv_array['osv2'] = osv2
                    osv_array['osv3'] = osv3
                else:
                    osv_array['osv1'] = -1
                    osv_array['osv2'] = -1
                    osv_array['osv3'] = -1

                osv_arrays.append(osv_array)
            osv_arrays = pd.DataFrame(osv_arrays)
            osv_arrays = osv_arrays.fillna(-1)
            df.drop(['osv','osv1','osv2','osv3'],axis = 1,inplace = True)
            df = pd.concat([df,osv_arrays],axis = 1)
            # print(df.info(verbose=True, null_counts=True))
            #####################缺失值填充#####################
            print('缺失值填充')
            df['app_cate_id'] = df['app_cate_id'].fillna(-1)
            df['app_id'] = df['app_id'].fillna(-1)
            df['click'] = df['click'].fillna(-1)
            df['user_tags'] = df['user_tags'].fillna(str(-1))
            df['f_channel'] = df['f_channel'].fillna(str(-1))
            df.drop(['app_paid', 'os_name'], axis=1, inplace=True)
            #####################机型填充#####################
            print('机型填充')
            df['make'] = df['make'].fillna(str(-1))
            df['model'] = df['model'].fillna(str(-1))
            df['creative_width'] = df['creative_width']/10
            df['creative_height'] = df['creative_height']/10
            # print(df.info(verbose=True, null_counts=True))
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
            a = df_train['make'].unique()
            phone_name_dict = {}
            print('  建品牌名称字典')
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
            print('  根据机型填充品牌')
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

            print('  剩余的品牌空缺通过机型包含的品牌字符进行填充')
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
            xiaomi = ['xiaomi','mi','mi max','redmi','redmi pro','mi note lte','xiaomi,mi note lte,virgo',
                      'mi note lte','mi note pro','redmipro','xiaomi,mi max,hydrogen','xiaomi+']
            oppo = ['oppo', 'ｏρρｏ']
            vivo = ['vivo', 'vivo nex s','vivo nex a','vivo nex']
            apple = ['iphone', 'apple','x-apple','iphone+x','苹果六']
            lajiao = ['红辣椒，xiaolajiao']
            honor = ['honor','潇爸的荣耀','x-honor']
            print('  品牌合并去噪音')
            for index, row in df.iterrows():
                if index % 10000 == 0:
                    print(index)
                if row['make'] in xiaomi:
                    row['make'] = 'xiaomi'
                elif row['make'] in oppo:
                    row['make'] = 'oppo'
                elif row['make'] in vivo:
                    row['make'] = 'vivo'
                elif row['make'] in apple:
                    row['make'] = 'apple'
                elif row['make'] in lajiao:
                    row['make'] = 'lajiao'
                elif row['make'] in honor:
                    row['make'] = 'honor'
            df.loc[(df['make_copy']!='-1')&(df['make']=='-1'),'make']='-2'
            print('make fill finished')
            df, cat_col = label_encode(df)
            print(df.info(verbose=True, null_counts=True))
            df.drop(['make_copy','has_number','province_zone'], axis=1, inplace=True)
            df.to_csv('../data/original_filled_labelencode_data.csv', index=False)
        ##################################通过特征重要性过滤user_tags####################################
        if part == 1:
            df = pd.read_csv('../../data/original_filled_labelencode_data.csv',nrows = num_rows)
            # df.drop('province_zone',axis = 1, inplace = True)
            print(df.info(verbose=True, null_counts=True))
            encoder = ['city_city','city_province','city_zone']
            df['city_city'] = df['city_city'].fillna(-1)
            df['city_province'] = df['city_province'].fillna(-1)
            df['city_zone'] = df['city_zone'].fillna(-1)
            df['province_zone'] = df['province_zone'].fillna(-1)
            # df = df.to_csv('../data/original_filled_labelencode_data.csv',index = False)
            col_encoder = LabelEncoder()
            df_train = df[df['click'] > -1]
            df_test = df[df['click'] == -1]
            for feat in encoder:
                print (feat)
                df_train[feat], df_test[feat], index_dict = onehot_feature_process(df_train[feat], df_test[feat], 1,
                                                                                   filter_num=1)
            df = pd.concat([df_train, df_test], axis=0)
            print(df.info(verbose=True, null_counts=True))
            df = df[df['click']>-1]

            train_y=df.pop('click')
            train_y = pd.DataFrame(train_y, columns=['click'])
            train_y['click'] = train_y['click'].astype(np.float16)
            cv_encode=CountVectorizer(token_pattern = u'(?u)\\b\\w+\\b')

            feature = 'user_tags'
            cv_encode.fit(df[feature])
            train_x = cv_encode.transform(df[feature])

            # cv_encode.vocabulary_.update({'-1':cv_encode.vocabulary_.pop("1")})
            # train_x = sparse.hstack((train_x, train_a))
            clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=2000,
                                       max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                                       min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000,  silent=True,device='gpu')
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
            ######################################根据重要性过滤###############################################
            df = pd.read_csv('../../data/original_filled_labelencode_data.csv',nrows = num_rows)
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
            df.to_csv('../data/original_filled_labelencode_data.csv', index=False)
        ##########################样本组合特征编码#############################
        # if part == 2:
        #     def onehot_combine_process(train_data_1, train_data_2, test1_data_1, test1_data_2, begin_num,
        #                                train_name, filter_num=100):
        #         count_dict = {}
        #         index_dict = {}
        #         filter_set = set()
        #         begin_index = begin_num
        #         for i, d in enumerate(train_data_1):
        #             id_1 = str(train_data_1[i])
        #             id_2 = str(train_data_2[i])
        #             t_id = id_1 + '|' + id_2
        #             if t_id not in count_dict.keys():
        #                 count_dict[t_id] = 0
        #             count_dict[t_id] += 1
        #         for key in count_dict:
        #             if count_dict[key] < filter_num:
        #                 filter_set.add(key)
        #         train = []
        #         for i, d in enumerate(train_data_1):
        #             id_1 = str(train_data_1[i])
        #             id_2 = str(train_data_2[i])
        #             t_id = id_1 + '|' + id_2
        #             if t_id in filter_set:
        #                 t_id = '-2'
        #             if t_id not in index_dict.keys():
        #                 index_dict[t_id] = begin_index
        #                 begin_index += 1
        #             train.append(index_dict[t_id])
        #         if '-2' not in index_dict.keys():
        #             index_dict['-2'] = begin_index
        #         test = []
        #         for i, d in enumerate(test1_data_1):
        #             id_1 = str(test1_data_1[i])
        #             id_2 = str(test1_data_1[i])
        #             t_id = id_1 + '|' + id_2
        #             if t_id in filter_set or t_id not in index_dict.keys():
        #                 t_id = '-2'
        #             test.append(index_dict[t_id])
        #
        #         train = pd.DataFrame({train_name: train})
        #         test = pd.DataFrame({train_name: test})
        #         del begin_index, count_dict
        #         gc.collect()
        #         return train, test
        #
        #     aid = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_one', 'advert_industry_inner_two',
        #            'advert_name',
        #            'campaign_id', 'creative_id',
        #            'creative_type', 'creative_tp_dnf']
        #
        #     mid = ['app_cate_id', 'app_id', 'inner_slot_id']
        #     # 用户信息
        #     uid = ['user_tags']
        #     # 上下文信息
        #     tid = ['city_city', 'city_province','city_zone', 'osv1', 'osv2', 'osv3', 'os', 'make', 'model']
        #     df = pd.read_csv('../data/original_filled_labelencode_data.csv', nrows=num_rows)
        #     df['city_city'] = df['city_city'].fillna(-1)
        #     df['city_province'] = df['city_province'].fillna(-1)
        #     df['city_zone'] = df['city_zone'].fillna(-1)
        #     df['province_zone'] = df['province_zone'].fillna(-1)
        #     df_train = df[df['click'] > -1].reset_index(drop=True)
        #     df_test = df[df['click'] == -1].reset_index(drop=True)
        #     df_train = reduce_mem_usage_parallel(df_train, 10)
        #     df_test = reduce_mem_usage_parallel(df_test, 10)
        #     f1_id = uid + mid + tid
        #     for f1 in f1_id:
        #         for f2 in aid:
        #             print(f1 + '_' + f2 + '_one_hot')
        #             t0 = time.time()
        #             train, test = onehot_combine_process(df_train[f1], df_train[f2],
        #                                                  df_test[f1], df_test[f2],
        #                                                  1, f1 + '_' + f2 + '_one_hot')
        #             df_train = pd.concat([df_train, train], axis=1)
        #             df_test = pd.concat([df_test, test], axis=1)
        #             del train, test
        #             gc.collect()
        #             print(f1 + '_' + f2 + '_one_hot', "____", "done in {:.0f}s".format(time.time() - t0))
        #     df = pd.concat([df_train, df_test], axis=0)
        #     df.to_csv('../data/original_filled_labelencode_data.csv', index=False)

if __name__ == "__main__":
    with timer("Full model run"):
        main(debug = False,part = 0)
