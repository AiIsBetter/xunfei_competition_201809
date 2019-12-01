#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2018.09.19
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import gc
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


round1_iflyad_train = pd.read_csv('../data/original_filled_labelencode_data.csv')

df_click = round1_iflyad_train[['instance_id','click']]
round1_iflyad_test = round1_iflyad_train[round1_iflyad_train['click']==-1].reset_index(drop=True)
round1_iflyad_train = round1_iflyad_train[round1_iflyad_train['click']>-1].reset_index(drop=True)

id_set = ['user_tags','adid','advert_id','orderid' ,'osv1', 'osv2', 'osv3', 'make', 'model', 'creative_id', 'creative_type', 'app_id',
             'creative_tp_dnf', 'app_cate_id', 'advert_industry_inner_one'
    , 'advert_industry_inner_two','creative_has_deeplink','creative_is_jump','creative_is_download','creative_is_js','creative_is_voicead','creative_width',
          'creative_height'
    , 'inner_slot_id','instance_id','click']

# id_set = ['user_tags','adid','advert_id','instance_id','click']
# id_set = ['user_tags','instance_id','click','adid']
round1_iflyad_train = round1_iflyad_train[id_set]
round1_iflyad_test = round1_iflyad_test[id_set]

id_set.remove('instance_id')
id_set.remove('click')
id_set.remove('user_tags')
id_name_dict = {}
id_name_dict2 = {}
count_dict = {}
click_num_dict = {}


def extra_same_elem(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    iset = set1.intersection(set2)
    return list(iset)
# 循环每一列特征值
for id_name in id_set:
    count = 0
    c = round1_iflyad_train[id_name].value_counts()
    id_num= list(round1_iflyad_train[id_name].value_counts().index._data)

    count_dict = {}
    index_dict = {}
    filter_set = set()
    begin_index = 1
    round1_iflyad_train['user_tags'].fillna('-1',inplace = True)

    click_num_dict[id_name]= {}
    id_name_dict[id_name] = {}
    id_name_dict2[id_name] = {}
    count_dict[id_name] = {}
    # for id in id_num:
    # 统计每个user_tags 中每个id 的点击次数
    for id in id_num:
        id_name_dict[id_name][id] = {}
        id_name_dict2[id_name][id] = {}
        click_num_dict[id_name][id]  = {}
        count_dict[id_name][id] = {}
    t0 = time.time()
    ##每行id列统计
    for i,row in round1_iflyad_train.iterrows():
        if i%10000 == 0:
            print(i)
        fea_name = round1_iflyad_train['user_tags'][i].split(',')
        if '' in fea_name:
            fea_name.remove('')
        bca = count_dict[id_name][row[id_name]].keys()
        bvd = click_num_dict[id_name][row[id_name]].keys()
        row_click = row['click']
        row_idname = row[id_name]
        for fea in fea_name:
            # 统计每个特征列中的每个instance_id number 中，每个user_tags 中的点击次数,逐行统计相加计算总数
            # count_dict对应总数，click_num_dict对应总数中点击的次数
            if fea not in bca:
                count_dict[id_name][row_idname][fea]=0
            count_dict[id_name][row_idname][fea] += 1
            if fea not in bvd:
                click_num_dict[id_name][row_idname][fea]=0
            click_num_dict[id_name][row_idname][fea] += row_click
        if i % 10000 == 0:
            print('id feature :',id_name,'点击数字典生成中', "____",str(i) ,"done in {:.0f}s".format(time.time() - t0))
            t0 = time.time()


    count_id_num = 0
    for id in id_num:
        t0 = time.time()
        # 讲统计的字典里面的值转为dataframe，方便整体计算
        # id_name_dict[id_name][id] = {}
        click_ratio_tmp = pd.DataFrame([click_num_dict[id_name][id]]).T.reset_index()
        click_ratio_tmp2 = pd.DataFrame([count_dict[id_name][id]]).T.reset_index()
        # click_ratio.rename(columns={'index':'user_tops','0':'click_ratio'},inplace = True)
        click_ratio_tmp.columns = ['user_tops','click_num']
        click_ratio_tmp2.columns = ['user_tops', 'total_num']
        click_ratio_tmp = click_ratio_tmp.merge(click_ratio_tmp2,on = 'user_tops',how = 'left')
        click_ratio_tmp['click_ratio'] = click_ratio_tmp['click_num']/click_ratio_tmp['total_num']

        # 只统计样本总数大于500的点击率分布，样本数太少的不统计打分
        click_ratio = click_ratio_tmp[(click_ratio_tmp['total_num'] > 500)]
        # 将统计的整数除以整个特征集的长度，相当于给点击率加一个权值，样本量大的权重高
        click_ratio['score'] = (click_ratio['total_num'] / round1_iflyad_train.shape[0]) * click_ratio['click_ratio']
        click_ratio.sort_values(by = 'click_ratio',axis = 0,ascending = False,inplace = True)
        # click_ratio_top = click_ratio.iloc[0:50].reset_index(drop=True)
        min_num = 10
        tmp2 = 0
        # id_name_dict存入每个特征列每个id类包含的每个user_tags类单独的点击率
        for i in range(click_ratio.shape[0]):
            if i>min_num-1:
                break
            id_name_dict[id_name][id][click_ratio['user_tops'].values[i]] = click_ratio['score'].values[i]
        tmp2 = sum(click_ratio['score'].values)

        # id_name_dict存入每个特征列中每个id类的总分
        # id_name_dict[id_name][id] = tmp1
        id_name_dict2[id_name][id] = tmp2
        print('id feature :',id_name,' id num length: ', str(len(id_num)), "____",str(count_id_num) ,"done in {:.0f}s".format(time.time() - t0))
        count_id_num +=1
    gc.collect()

# 对训练集进行统计
round1_iflyad_train['user_tags'] = round1_iflyad_train['user_tags'].apply(lambda x:x.split(','))
# 先取出所有行user_tags的值
round1_iflyad_train_u = round1_iflyad_train['user_tags'].values
user_tops_list = {}
user_tops_sum = {}
for i in id_set:
    user_tops_list[i] = []
    user_tops_sum[i] = []
# 每一行将上述统计特征按id对应user_tags的点击率和得分存入
for index in range(round1_iflyad_train.shape[0]):
    # t0 = time.time()
    r_list = round1_iflyad_train_u[index]
    # 每一列特征处理一次
    for id_name in id_set:
        r_id_num = round1_iflyad_train[id_name][index]
        if r_id_num not in id_name_dict[id_name].keys():
            user_tops_list[id_name].append(0)
        else:
            tmp = list(id_name_dict[id_name][r_id_num].keys())
            # 每个特征行和列，对应的user_tags里面，有多少和id_name_dict统计过打分的user_tags相同的，都取出来
            same_tmp = extra_same_elem(r_list, tmp)
            if len(same_tmp)>0:
                count = 0
                # 把相同的id的打分取出来让存入累加作为该行特征
                for i in range(len(same_tmp)):
                    count +=id_name_dict[id_name][r_id_num][tmp[tmp.index(same_tmp[i])]]
                user_tops_list[id_name].append(count)
            else:
                user_tops_list[id_name].append(0)
        # 统计的id对应的总分，直接存入该行作为特征
        if r_id_num not in id_name_dict2[id_name].keys():
            user_tops_sum[id_name].append(0)
        else:
            tmp2 = id_name_dict2[id_name][r_id_num]
            user_tops_sum[id_name].append(tmp2)
    if index % 10000 == 0:
        print('id feature :',id_name,' train set', "____",str(index) ,"done in {:.0f}s".format(time.time() - t0))
        t0 = time.time()
for i in id_set:
    round1_iflyad_train[i + '_user_ratio'] = user_tops_list[i]
    round1_iflyad_train[i + '_user_ratio_sum'] = user_tops_sum[i]

# 对测试集进行统计
user_tops_sum = {}
user_tops_list = {}
for i in id_set:
    user_tops_list[i] = []
    user_tops_sum[i] = []
round1_iflyad_test['user_tags'] = round1_iflyad_test['user_tags'].apply(lambda x:x.split(','))
round1_iflyad_test_u = list(round1_iflyad_test['user_tags'].values)
for index in range(round1_iflyad_test.shape[0]):
    user_tmp = row['user_tags'].strip('').split(',')
    r_list = round1_iflyad_test_u[index]
    for id_name in id_set:
        r_id_num = round1_iflyad_test[id_name][index]
        if r_id_num not in id_name_dict[id_name].keys():
            user_tops_list[id_name].append(0)
        else:
            tmp = list(id_name_dict[id_name][r_id_num].keys())
            same_tmp = extra_same_elem(r_list, tmp)
            if len(same_tmp)>0:
                count = 0
                for i in range(len(same_tmp)):
                    count +=id_name_dict[id_name][r_id_num][tmp[tmp.index(same_tmp[i])]]
                user_tops_list[id_name].append(count)
            else:
                user_tops_list[id_name].append(0)
        if r_id_num not in id_name_dict2[id_name].keys():
            user_tops_sum[id_name].append(0)
        else:
            tmp2 = id_name_dict2[id_name][r_id_num]
            user_tops_sum[id_name].append(tmp2)

    if index % 10000 == 0:
        print('id feature :',id_name,' test set', "____", str(index), "done in {:.0f}s".format(time.time() - t0))
        t0 = time.time()
for i in id_set:
    round1_iflyad_test[i + '_user_ratio'] = user_tops_list[i]
    round1_iflyad_test[i + '_user_ratio_sum'] = user_tops_sum[i]
# round1_iflyad_test[id_name + '_user_tops'] = user_tops_list
# 合并训练测试集后作为新特征
data = pd.concat([round1_iflyad_train,round1_iflyad_test])
data.drop(['user_tags','click'],axis = 1,inplace = True)
data.drop(id_set,axis = 1,inplace = True)
data = data.reset_index(drop=True)
data = data.merge(df_click,on = 'instance_id',how = 'left')
data.to_csv('original_filled_labelencode_user_flag_sum.csv',index = False)
