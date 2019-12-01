#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2018.09.19

import pandas as pd
import gc
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
round1_iflyad_train = pd.read_csv('../data/original_filled_labelencode_data.csv')

print(round1_iflyad_train.info(verbose=True, null_counts=True))
# round1_iflyad_train = pd.read_csv('original_filled_labelencode_user_tops.csv')
# print(round1_iflyad_train.info(verbose=True, null_counts=True))
df_click = round1_iflyad_train[['instance_id','click']]
round1_iflyad_test = round1_iflyad_train[round1_iflyad_train['click']==-1].reset_index(drop=True)
round1_iflyad_train = round1_iflyad_train[round1_iflyad_train['click']>-1].reset_index(drop=True)
round1_iflyad_test = round1_iflyad_test.iloc[0:20000].reset_index(drop=True)
round1_iflyad_train = round1_iflyad_train.iloc[0:20000].reset_index(drop=True)
# round1_iflyad_train = pd.read_csv('../data/round1_iflyad_train.txt',delimiter="\t",nrows = 50000)
# id_set = ['user_tags','adid','advert_id','orderid' ,'osv1', 'osv2', 'osv3', 'make', 'model', 'creative_id', 'creative_type', 'app_id',
#              'creative_tp_dnf', 'app_cate_id', 'advert_industry_inner_one'
#     , 'advert_industry_inner_two'
#     , 'inner_slot_id','instance_id','click']
id_set = ['user_tags','instance_id','click','creative_has_deeplink','creative_is_jump','creative_is_download','creative_is_js','creative_is_voicead','creative_width',
          'creative_height']
# id_set = ['user_tags','instance_id','click','creative_has_deeplink']
round1_iflyad_train = round1_iflyad_train[id_set]
round1_iflyad_test = round1_iflyad_test[id_set]
# 跟2_user_tags_feature原理相同，不过统计flag id类特征
id_set.remove('instance_id')
id_set.remove('click')
id_set.remove('user_tags')
id_name_dict = {}
id_name_dict2 = {}
count_dict = {}
click_num_dict = {}
for id_name in id_set:
    count = 0
    c = round1_iflyad_train[id_name].value_counts()
    id_num= list(round1_iflyad_train[id_name].value_counts().index._data)

    count_dict = {}
    index_dict = {}
    filter_set = set()
    begin_index = 1
    round1_iflyad_train['user_tags'].fillna('-1',inplace = True)
    def extra_same_elem(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        iset = set1.intersection(set2)
        return list(iset)

    click_num_dict[id_name]= {}
    id_name_dict[id_name] = {}
    id_name_dict2[id_name] = {}
    count_dict[id_name] = {}
    # for id in id_num:
    #   # 计每个user_tags 中每个id 的点击次数
    for id in id_num:
        id_name_dict[id_name][id] = {}
        id_name_dict2[id_name][id] = {}
        click_num_dict[id_name][id]  = {}
        count_dict[id_name][id] = {}
    t0 = time.time()
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
            # 统计每个id number 中 每个user_tags 中的点击次数
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
        # id_name_dict[id_name][id] = {}
        click_ratio_tmp = pd.DataFrame([click_num_dict[id_name][id]]).T.reset_index()
        click_ratio_tmp2 = pd.DataFrame([count_dict[id_name][id]]).T.reset_index()
        # click_ratio.rename(columns={'index':'user_tops','0':'click_ratio'},inplace = True)
        click_ratio_tmp.columns = ['user_tops','click_num']
        click_ratio_tmp2.columns = ['user_tops', 'total_num']

        click_ratio_tmp = click_ratio_tmp.merge(click_ratio_tmp2,on = 'user_tops',how = 'left')
        click_ratio = click_ratio_tmp[(click_ratio_tmp['total_num'] > 500)]
        # click_ratio_tmp['click_ratio'] = click_ratio_tmp['click_num']/click_ratio_tmp['total_num']

        # 统计样本总数大于500的点击率分布
        # click_ratio = click_ratio_tmp[(click_ratio_tmp['total_num'] > 500)]
        # click_ratio['score'] = (click_ratio['total_num'] / round1_iflyad_train.shape[0]) * click_ratio['click_ratio']
        # click_ratio.sort_values(by = 'click_ratio',axis = 0,ascending = False,inplace = True)
        # click_ratio_top = click_ratio.iloc[0:50].reset_index(drop=True)
        for i in range(click_ratio.shape[0]):
            id_name_dict[id_name][id][click_ratio['user_tops'].values[i]] = click_ratio['click_num'].values[i]
            id_name_dict2[id_name][id][click_ratio['user_tops'].values[i]] = click_ratio['total_num'].values[i]

        print('id feature :',id_name,' id num length: ', str(len(id_num)), "____",str(count_id_num) ,"done in {:.0f}s".format(time.time() - t0))
        count_id_num +=1
    gc.collect()

def extra_same_elem(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    iset = set1.intersection(set2)
    return list(iset)
# for i in id_set:
#     round1_iflyad_train[i+'_user_tops'] = 0
#     round1_iflyad_test[i+'_user_tops'] = 0

round1_iflyad_train['user_tags'] = round1_iflyad_train['user_tags'].apply(lambda x:x.split(','))
round1_iflyad_train_u = round1_iflyad_train['user_tags'].values
user_tops_list = {}
user_tops_sum = {}
for i in id_set:
    user_tops_list[i] = []
    user_tops_sum[i] = []
for index in range(round1_iflyad_train.shape[0]):
    # t0 = time.time()
    r_list = round1_iflyad_train_u[index]
    for id_name in id_set:
        r_id_num = round1_iflyad_train[id_name][index]
        if r_id_num not in id_name_dict[id_name].keys():
            user_tops_list[id_name].append(0)
            user_tops_sum[id_name].append(0)
        else:
            tmp = list(id_name_dict[id_name][r_id_num].keys())
            same_tmp = extra_same_elem(r_list, tmp)
            if len(same_tmp)>0:
                count = 0
                count2 = 0
                for i in range(len(same_tmp)):
                    same_index = tmp[tmp.index(same_tmp[i])]
                    count +=id_name_dict[id_name][r_id_num][same_index]
                    count2 += id_name_dict2[id_name][r_id_num][same_index]
                user_tops_list[id_name].append(count)
                user_tops_sum[id_name].append(count2)
            else:
                user_tops_list[id_name].append(0)
                user_tops_sum[id_name].append(0)
    if index % 10000 == 0:
        print('id feature :',id_name,' train set', "____",str(index) ,"done in {:.0f}s".format(time.time() - t0))
        t0 = time.time()
for i in id_set:
    round1_iflyad_train[i + '_flag_click'] = user_tops_list[i]
    round1_iflyad_train[i + '_flag_sum'] = user_tops_sum[i]

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
            user_tops_sum[id_name].append(0)
        else:
            tmp = list(id_name_dict[id_name][r_id_num].keys())
            same_tmp = extra_same_elem(r_list, tmp)
            if len(same_tmp)>0:
                count = 0
                count2 = 0

                for i in range(len(same_tmp)):
                    same_index = tmp[tmp.index(same_tmp[i])]
                    count +=id_name_dict[id_name][r_id_num][same_index]
                    count2 += id_name_dict2[id_name][r_id_num][same_index]
                user_tops_list[id_name].append(count)
                user_tops_sum[id_name].append(count2)
            else:
                user_tops_list[id_name].append(0)
                user_tops_sum[id_name].append(0)
    if index % 10000 == 0:
        print('id feature :',id_name,' test set', "____", str(index), "done in {:.0f}s".format(time.time() - t0))
        t0 = time.time()
for i in id_set:
    round1_iflyad_test[i + '_flag_click'] = user_tops_list[i]
    round1_iflyad_test[i + '_flag_sum'] = user_tops_sum[i]
# round1_iflyad_test[id_name + '_user_tops'] = user_tops_list
data = pd.concat([round1_iflyad_train,round1_iflyad_test])
data.drop(['user_tags','click'],axis = 1,inplace = True)
data.drop(id_set,axis = 1,inplace = True)
data = data.reset_index(drop=True)
data = data.merge(df_click,on = 'instance_id',how = 'left')
data.to_csv('original_filled_labelencode_flag_sum.csv',index = False)
