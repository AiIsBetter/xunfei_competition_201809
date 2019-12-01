#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2018.09.19
import pandas as pd
import numpy as np
aid = ['adid','advert_id','orderid','advert_industry_inner_one','advert_industry_inner_two','advert_name','campaign_id','creative_id',
           'creative_type','creative_tp_dnf','creative_has_deeplink','creative_is_jump','creative_is_download',
            'creative_is_js','creative_is_voicead','creative_width','creative_height','advert_industry_inner_one',
           'advert_industry_inner_two','f_channel']
# 媒体信息
mid = ['app_cate_id','app_id','inner_slot_id']
# 用户信息
uid = ['user_tags']
# 上下文信息
tid = ['city','carrier','province','nnt','devtype','osv','os','make','model']
# 补充
b_id = ['user_tags','time_day','time_hours','tags_num']
all_id = aid+mid+tid+b_id
num_rows = None
df0 = pd.read_csv('../../data/original_filled_one_hot_data.csv',  nrows=num_rows)
df0.drop(['user_tags','time_day'],axis = 1 ,inplace = True)


df1 = pd.read_csv('../../data/original_count_combine_feature_times.csv',  nrows=num_rows)
df1.drop(all_id,axis = 1 ,inplace = True)

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


# df1, cat_col = label_encode(df1)
df1.drop('click',axis = 1,inplace =  True)
aid = ['adid','advert_id','orderid','advert_industry_inner_one','advert_industry_inner_two','advert_name','campaign_id','creative_id',
           'creative_type','creative_tp_dnf','creative_has_deeplink','creative_is_jump','creative_is_download',
            'creative_is_js','creative_is_voicead','creative_width','creative_height','advert_industry_inner_one',
           'advert_industry_inner_two','f_channel']
# 媒体信息
mid = ['app_cate_id','app_id','inner_slot_id']
# 用户信息
uid = ['user_tags']
# 上下文信息
tid = ['city','carrier','province','nnt','devtype','osv','os','make','model']
# 补充
b_id = ['user_tags','time_day','time_hours','tags_num','click', 'user_tags_adid_times', 'user_tags_advert_id_times',
       'user_tags_orderid_times', 'user_tags_advert_industry_inner_one_times',
       'user_tags_advert_industry_inner_two_times',
       'user_tags_advert_name_times', 'user_tags_campaign_id_times',
       'user_tags_creative_id_times', 'user_tags_creative_type_times',
       'user_tags_creative_tp_dnf_times', 'user_tags_app_cate_id_times',
       'user_tags_app_id_times', 'user_tags_inner_slot_id_times',
       'user_tags_city_times', 'user_tags_province_times',
       'user_tags_osv_times', 'user_tags_os_times', 'user_tags_make_times',
       'user_tags_model_times', 'user_tags_user_tags_times']
all_id = aid+mid+tid+b_id

df2 = pd.read_csv('../../data/original_count_pos_feature.csv',  nrows=num_rows)
df2.drop(all_id,axis = 1 ,inplace = True)


aid = ['adid','advert_id','orderid','advert_industry_inner_one','advert_industry_inner_two','advert_name','campaign_id','creative_id',
           'creative_type','creative_tp_dnf','creative_has_deeplink','creative_is_jump','creative_is_download',
            'creative_is_js','creative_is_voicead','creative_width','creative_height','advert_industry_inner_one',
           'advert_industry_inner_two','f_channel']
# 媒体信息
mid = ['app_cate_id','app_id','inner_slot_id']
# 用户信息
uid = ['user_tags']
# 上下文信息
tid = ['city','carrier','province','nnt','devtype','osv','os','make','model']
# 补充
b_id = ['user_tags','time_day','time_hours','tags_num','click']
all_id = aid+mid+tid+b_id

df3 = pd.read_csv('../../data/original_onehot_combine_feature.csv',  nrows=num_rows)
df3.drop(all_id,axis = 1 ,inplace = True)

df = pd.merge(df0,df1,on = 'instance_id',how = 'left')
df = pd.merge(df,df2,on = 'instance_id',how = 'left')
df = pd.merge(df,df3,on = 'instance_id',how = 'left')

# df.to_csv('../../data/onehot_all_feature.csv',index = False)
# a = 1

# df = pd.read_csv('../../data/onehot_all_feature.csv')
df_train = df[df['click'] > -1].reset_index(drop=True)
df_test = df[df['click'] == -1].reset_index(drop=True)
df_train.to_csv('../../data/onehot_all_feature_train.csv',index = False)
df_test.to_csv('../../data/onehot_all_feature_test.csv',index = False)
b = 1
# ['instance_id', 'city', 'province', 'carrier', 'devtype', 'make', 'model', 'nnt', 'os', 'osv', 'adid', 'advert_id', 'orderid', 'campaign_id', 'creative_id', 'creative_tp_dnf', 'app_cate_id', 'f_channel', 'app_id', 'inner_slot_id', 'creative_type', 'creative_width', 'creative_height', 'creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead', 'creative_has_deeplink', 'advert_name', 'click', 'advert_industry_inner_one', 'advert_industry_inner_two', 'time_hours', 'tags_num', 'user_tags', 'gd', 'ag', 'mzag', 'mzgd', 'ocpates', 'mz_ag', 'mz_gd', 'user_tags_adid_times_x', 'user_tags_advert_id_times_x', 'user_tags_orderid_times_x', 'user_tags_advert_industry_inner_one_times_x', 'user_tags_advert_industry_inner_two_times_x', 'user_tags_advert_name_times_x', 'user_tags_campaign_id_times_x', 'user_tags_creative_id_times_x', 'user_tags_creative_type_times_x', 'user_tags_creative_tp_dnf_times_x', 'user_tags_app_cate_id_times_x', 'user_tags_app_id_times_x', 'user_tags_inner_slot_id_times_x', 'user_tags_user_tags_times_x', 'user_tags_city_times_x', 'user_tags_province_times_x', 'user_tags_osv_times_x', 'user_tags_os_times_x', 'user_tags_make_times_x', 'user_tags_model_times_x', 'gd_adid_times_x', 'gd_advert_id_times_x', 'gd_orderid_times_x', 'gd_advert_industry_inner_one_times_x', 'gd_advert_industry_inner_two_times_x', 'gd_advert_name_times_x', 'gd_campaign_id_times_x', 'gd_creative_id_times_x', 'gd_creative_type_times_x', 'gd_creative_tp_dnf_times_x', 'gd_app_cate_id_times_x', 'gd_app_id_times_x', 'gd_inner_slot_id_times_x', 'gd_user_tags_times_x', 'gd_city_times_x', 'gd_province_times_x', 'gd_osv_times_x', 'gd_os_times_x', 'gd_make_times_x', 'gd_model_times_x', 'user_tags_adid_times_y', 'user_tags_advert_id_times_y', 'user_tags_orderid_times_y', 'user_tags_advert_industry_inner_one_times_y', 'user_tags_advert_industry_inner_two_times_y', 'user_tags_advert_name_times_y', 'user_tags_campaign_id_times_y', 'user_tags_creative_id_times_y', 'user_tags_creative_type_times_y', 'user_tags_creative_tp_dnf_times_y', 'user_tags_app_cate_id_times_y', 'user_tags_app_id_times_y', 'user_tags_inner_slot_id_times_y', 'user_tags_user_tags_times_y', 'user_tags_city_times_y', 'user_tags_province_times_y', 'user_tags_osv_times_y', 'user_tags_os_times_y', 'user_tags_make_times_y', 'user_tags_model_times_y', 'gd_adid_times_y', 'gd_advert_id_times_y', 'gd_orderid_times_y', 'gd_advert_industry_inner_one_times_y', 'gd_advert_industry_inner_two_times_y', 'gd_advert_name_times_y', 'gd_campaign_id_times_y', 'gd_creative_id_times_y', 'gd_creative_type_times_y', 'gd_creative_tp_dnf_times_y', 'gd_app_cate_id_times_y', 'gd_app_id_times_y', 'gd_inner_slot_id_times_y', 'gd_user_tags_times_y', 'gd_city_times_y', 'gd_province_times_y', 'gd_osv_times_y', 'gd_os_times_y', 'gd_make_times_y', 'gd_model_times_y', 'app_cate_id_adid_one_hot', 'app_cate_id_advert_id_one_hot', 'app_cate_id_orderid_one_hot', 'app_cate_id_advert_industry_inner_one_one_hot', 'app_cate_id_advert_industry_inner_two_one_hot', 'app_cate_id_advert_name_one_hot', 'app_cate_id_campaign_id_one_hot', 'app_cate_id_creative_id_one_hot', 'app_cate_id_creative_type_one_hot', 'app_cate_id_creative_tp_dnf_one_hot', 'app_id_adid_one_hot', 'app_id_advert_id_one_hot', 'app_id_orderid_one_hot', 'app_id_advert_industry_inner_one_one_hot', 'app_id_advert_industry_inner_two_one_hot', 'app_id_advert_name_one_hot', 'app_id_campaign_id_one_hot', 'app_id_creative_id_one_hot', 'app_id_creative_type_one_hot', 'app_id_creative_tp_dnf_one_hot', 'inner_slot_id_adid_one_hot', 'inner_slot_id_advert_id_one_hot', 'inner_slot_id_orderid_one_hot', 'inner_slot_id_advert_industry_inner_one_one_hot', 'inner_slot_id_advert_industry_inner_two_one_hot', 'inner_slot_id_advert_name_one_hot', 'inner_slot_id_campaign_id_one_hot', 'inner_slot_id_creative_id_one_hot', 'inner_slot_id_creative_type_one_hot', 'inner_slot_id_creative_tp_dnf_one_hot', 'city_adid_one_hot', 'city_advert_id_one_hot', 'city_orderid_one_hot', 'city_advert_industry_inner_one_one_hot', 'city_advert_industry_inner_two_one_hot', 'city_advert_name_one_hot', 'city_campaign_id_one_hot', 'city_creative_id_one_hot', 'city_creative_type_one_hot', 'city_creative_tp_dnf_one_hot', 'province_adid_one_hot', 'province_advert_id_one_hot', 'province_orderid_one_hot', 'province_advert_industry_inner_one_one_hot', 'province_advert_industry_inner_two_one_hot', 'province_advert_name_one_hot', 'province_campaign_id_one_hot', 'province_creative_id_one_hot', 'province_creative_type_one_hot', 'province_creative_tp_dnf_one_hot', 'osv_adid_one_hot', 'osv_advert_id_one_hot', 'osv_orderid_one_hot', 'osv_advert_industry_inner_one_one_hot', 'osv_advert_industry_inner_two_one_hot', 'osv_advert_name_one_hot', 'osv_campaign_id_one_hot', 'osv_creative_id_one_hot', 'osv_creative_type_one_hot', 'osv_creative_tp_dnf_one_hot', 'os_adid_one_hot', 'os_advert_id_one_hot', 'os_orderid_one_hot', 'os_advert_industry_inner_one_one_hot', 'os_advert_industry_inner_two_one_hot', 'os_advert_name_one_hot', 'os_campaign_id_one_hot', 'os_creative_id_one_hot', 'os_creative_type_one_hot', 'os_creative_tp_dnf_one_hot', 'make_adid_one_hot', 'make_advert_id_one_hot', 'make_orderid_one_hot', 'make_advert_industry_inner_one_one_hot', 'make_advert_industry_inner_two_one_hot', 'make_advert_name_one_hot', 'make_campaign_id_one_hot', 'make_creative_id_one_hot', 'make_creative_type_one_hot', 'make_creative_tp_dnf_one_hot', 'model_adid_one_hot', 'model_advert_id_one_hot', 'model_orderid_one_hot', 'model_advert_industry_inner_one_one_hot', 'model_advert_industry_inner_two_one_hot', 'model_advert_name_one_hot', 'model_campaign_id_one_hot', 'model_creative_id_one_hot', 'model_creative_type_one_hot', 'model_creative_tp_dnf_one_hot']
