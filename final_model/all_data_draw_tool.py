#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2018.09.19

from scipy import stats
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot
import plotly.plotly as py
from sklearn.preprocessing import LabelEncoder
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from plotly import tools
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# bins = [0.38,1.35,2.198,3.044,3.890,4.737,5.584,6.44]
# py.sign_in('Aigege', 'h1gkryviJ97QpIATnb01') # Replace the username, and API key with your credentials.
# py.sign_in('Aigege1', 'ANMsyr7Zw5PDpqvUr05l') # Replace the username, and API key with your credentials.
##################################train and test#################################################################
# def cat_CTR(col,application):
#     plt.figure(figsize=(10, 8))
#     sns.kdeplot(application.loc[application['click'] == 0, col], label='target == 0')
#     sns.kdeplot(application.loc[application['click'] == 1, col], label='target == 1')
#     plt.xlabel(col);
#     plt.ylabel('Density');
#     plt.title('{} Distribution of click'.format(col))
#     plt.savefig('data_analysis/category/' + col + '_category.png')
#     plt.close()
# # Function to explore the numeric data
# def numeric(col,train,test,cat = False):
#     plt.figure(figsize=(12,10))
#     ax =plt.subplot(2, 1, 1)
#     plt.title("Train Distribution of "+col)
#     sns.distplot(train[col].dropna(),ax = ax)
#     ax = plt.subplot(2, 1, 2)
#     sns.distplot(test[col].dropna(),ax = ax)
#     plt.title("Test Distribution of " + col)
#     plt.show()
#     if cat:
#         plt.savefig('data_analysis/numeric/' + 'cat_' +col + '_numeric.png')
#     else:
#         plt.savefig('data_analysis/numeric/' + col + '_numeric.png')
#     plt.show()
#     plt.close()
# round1_iflyad_train = pd.read_csv('../data/round1_iflyad_train.txt',delimiter="\t")
# round1_iflyad_test = pd.read_csv('../data/round1_iflyad_test_feature.txt',delimiter="\t")
# print(round1_iflyad_train.info(verbose=True,null_counts=True))
# count = 0
# judge = 0
# for feature in round1_iflyad_train.columns:
#     if feature == 'click':
#         continue
#     if(round1_iflyad_train[feature].dtypes == 'object'):
#         round1_iflyad_train[feature].fillna('-1', inplace=True)
#         le = LabelEncoder()
#         le.fit(round1_iflyad_train[feature])
#         round1_iflyad_train[feature] = le.transform(round1_iflyad_train[feature])
#         round1_iflyad_test[feature].fillna('-1', inplace=True)
#         le = LabelEncoder()
#         le.fit(round1_iflyad_test[feature])
#         round1_iflyad_test[feature] = le.transform(round1_iflyad_test[feature])
#         numeric(feature, round1_iflyad_train,round1_iflyad_test,cat = True)
#         cat_CTR(feature, round1_iflyad_train)
#         count = count +1
#         print(str(count)+'obj_'+feature)
#     else:
#         count = count + 1
#         numeric(feature,round1_iflyad_train,round1_iflyad_test)
#         print(str(count)+feature)
############################################################hour-time######################################################################
# round1_iflyad_train = pd.read_csv('../data/round1_iflyad_train.txt',delimiter="\t")
# print(round1_iflyad_train.info(verbose=True,null_counts=True))
# round1_iflyad_train['time_day'] = round1_iflyad_train['time'].apply(lambda x : int(time.strftime("%d", time.localtime(x))))
# round1_iflyad_train['time_hour'] = round1_iflyad_train['time'].apply(lambda x : int(time.strftime("%H", time.localtime(x))))
# print(round1_iflyad_train.info(verbose=True,null_counts=True))
#
# plt.figure(figsize=(10, 8))
# sns.distplot(round1_iflyad_train.loc[round1_iflyad_train['click'] == 1,'time_hour'],label = 'clikc = 1',color = 'r')
# sns.distplot(round1_iflyad_train.loc[round1_iflyad_train['click'] == 0,'time_hour'],label = 'clikc = 0',color = 'b')
# plt.title('Distribution of day num click  ')
#
# plt.xlabel('time')
# plt.ylabel('num click')
# plt.legend(loc='best')
# plt.savefig('data_analysis/numeric/' + 'day_time_click' + '_numeric.png')
# plt.close()
############################################################day-time######################################################################
# round1_iflyad_train = pd.read_csv('../data/round1_iflyad_train.txt',delimiter="\t")
# print(round1_iflyad_train.info(verbose=True,null_counts=True))
# round1_iflyad_train['time'] = (round1_iflyad_train['time']-2190000000)/100000
# bins = [0.38,1.35,2.198,3.044,3.890,4.737,5.584,6.44]
# round1_iflyad_train['time'] = pd.cut(round1_iflyad_train['time'], bins, labels=[0,1,2,3,4,5,6])
# print(round1_iflyad_train.info(verbose=True,null_counts=True))
# r = round1_iflyad_train.groupby('time')['click'].count()
# plt.figure(figsize=(10, 8))
# sns.distplot(round1_iflyad_train.loc[round1_iflyad_train['click'] == 1,'time'],label = 'clikc = 1')
# sns.distplot(round1_iflyad_train.loc[round1_iflyad_train['click'] == 0,'time'],label = 'clikc = 0')
# plt.title('Distribution of day num click  ')
# plt.xlabel('time')
# plt.ylabel('num click')
# plt.show()
# plt.savefig('data_analysis/numeric/' + 'time_click' + '_numeric.png')
# plt.close()
############################################################null distribution######################################################################
# round1_iflyad_train = pd.read_csv('../data/round1_iflyad_train.txt',delimiter="\t")
# round1_iflyad_test = pd.read_csv('../data/round1_iflyad_test_feature.txt',delimiter="\t")
# # round1_iflyad_train['num_null'] = (round1_iflyad_train.isnull()).sum(axis=1)
# # round1_iflyad_test['num_null'] = (round1_iflyad_test.isnull()).sum(axis=1)
# # a = round1_iflyad_train.groupby('num_null')['num_null'].count()
# # b = round1_iflyad_test.groupby('num_null')['num_null'].count()
# round1_iflyad_train['n_null'] = (round1_iflyad_train.isnull()).sum(axis=1)
# round1_iflyad_test['n_null'] = (round1_iflyad_test.isnull()).sum(axis=1)
# train = round1_iflyad_train[['n_null']]
# train = train.sort_values(by = 'n_null')
# test = round1_iflyad_test[['n_null']]
# test = test.sort_values(by = 'n_null')
# train_x = np.array(range(0,train.shape[0]))
# train_y = train['n_null'].values
# test_x = np.array(range(0,test.shape[0]))
# test_y = test['n_null'].values
# fig = plt.figure(figsize=(16,9))
# ax1 = fig.add_subplot(2,1,1)
# plt.scatter(train_x,train_y,c='b',linewidths = 1)
# plt.xlabel('instance_id')
# plt.ylabel('num_null')
# plt.title('Distribution of round1_iflyad_train num null ')
# ax1 = fig.add_subplot(2,1,2)
# plt.scatter(test_x,test_y,c='b',linewidths = 1)
# plt.xlabel('instance_id')
# plt.ylabel('num_null')
# plt.title('Distribution of round1_iflyad_test num ull')
# plt.savefig('data_analysis/numeric/' + 'num_null' + '_numeric.png')
# plt.close()
############################################################nuu dis######################################################################
#  # 统计train正负样本分布
# round1_iflyad_train = pd.read_csv('../data/round1_iflyad_train.txt',delimiter="\t")
# fig = plt.figure(figsize=(10,9))
# ax1 = fig.add_subplot(1,1,1)
# train_target1 = round1_iflyad_train.click.value_counts()
# ratio = train_target1[1]/(train_target1[0]+train_target1[1])
# train_target1 = pd.DataFrame({'all_click':train_target1})
# train_target1.plot(kind = 'bar',ax = ax1,title =( 'round1_iflyad_train+click ratio:'+str(ratio)))
# plt.savefig('data_analysis/numeric/' + 'click ratio' + '_numeric.png')
# plt.close()
# #每列缺失值分布，根据不同分布情况填充缺失值
# num_null = []
# for feature in round1_iflyad_train.columns:
#     temp = round1_iflyad_train[feature].isnull().value_counts()
#     if(temp.shape[0]>1):
#         num_null.append(temp[True])
#     else:
#         num_null.append(0)
# num_null  = [x/round1_iflyad_train.shape[0] for x in num_null ]
# num_null = dict(zip(round1_iflyad_train.columns,num_null))
# num_null = sorted( num_null.items(), key=lambda temp: temp[1], reverse=True)
# num_null = pd.DataFrame(num_null).T
# b = num_null.iloc[0].tolist()
# a = dict(zip(range(round1_iflyad_train.shape[1]),num_null.iloc[0].tolist()))
# num_null.rename(columns = a,inplace=True)
# num_null.drop([0],inplace = True)
# fig = plt.figure(figsize=(12,10))
# ax1 = fig.add_subplot(1,1,1)
# num_null.plot(ax=ax1,kind = 'bar',title = 'num null ratio of feature')
# plt.ylabel( 'ratio')
# plt.xlabel ( 'feature')
# plt.legend(loc = 'best')
# plt.savefig('data_analysis/numeric/' + 'null_feature_ratio' + '_numeric.png')
# plt.close()
############################################province city#######################################################
# round1_iflyad_train = pd.read_csv('../data/round1_iflyad_train.txt',delimiter="\t")
# # a = round1_iflyad_train.city.unique()
# # temp = round1_iflyad_train.groupby(['city'],as_index=False)['click'].agg({'city_click_count':'count'})
# le = LabelEncoder()
# le.fit(round1_iflyad_train['city'])
# round1_iflyad_train['city'] = le.transform(round1_iflyad_train['city'])
# temp1 = round1_iflyad_train['city'].value_counts().reset_index()
# temp1 = temp1.rename(index = str,columns = {'index':'city','city':'city_num'})
# round1_iflyad_train = round1_iflyad_train.merge(temp1,on = 'city',how = 'left')
# round1_iflyad_train[(round1_iflyad_train['city_num']>=0) & (round1_iflyad_train['city_num']<=70)] = 0
# round1_iflyad_train[(round1_iflyad_train['city_num']>70) & (round1_iflyad_train['city_num']<=300)] = 1
# round1_iflyad_train[(round1_iflyad_train['city_num']>300) & (round1_iflyad_train['city_num']<=850)] = 2
# round1_iflyad_train[(round1_iflyad_train['city_num']>850) & (round1_iflyad_train['city_num']<=2500)] = 3
# round1_iflyad_train[(round1_iflyad_train['city_num']>2500) & (round1_iflyad_train['city_num']<=7000)] = 4
# round1_iflyad_train[(round1_iflyad_train['city_num']>7000)] = 5
#
# values = list(round1_iflyad_train['city_num'].unique())
# for v in values:
#     round1_iflyad_train[str('city_num') + '_' + str(v)] = (round1_iflyad_train['city_num'] == v).astype(np.uint8)
# round1_iflyad_train.drop('city_num',axis = 1,inplace = True)
#
# # 0,70,300,850,2500,7000,+++
# df = pd.DataFrame({'SurvS':temp1})
# df.plot(kind = 'bar',logy=True)
# plt.show()
#############################################################################################
# train = pd.read_csv('../data/round1_iflyad_train.txt',delimiter="\t")
# grouped_df = train.groupby(["period", "time_hours"])["click"].aggregate("mean").reset_index()
# grouped_df = grouped_df.pivot('period', 'time_hours', 'click')
# plt.figure(figsize=(12,6))
# sns.heatmap(grouped_df)
# plt.title("CVR of Day Vs Hour")
# plt.show()
# #####################################################user_tags多值特征分析-用户每个属性对应单个id num 点击率#########################################
# round1_iflyad_train = pd.read_csv('../data/round1_iflyad_train.txt',delimiter="\t",nrows = 50000)
# id_name ='advert_industry_inner_one'
# c = round1_iflyad_train[id_name].value_counts()
# adid_id = list(round1_iflyad_train[id_name].value_counts().index._data)[0:100]
#
# # f = open('top_uset_tags.txt', 'r')
# # all_name = f.readlines()
# # feature_name = []
# # for i in all_name:
# #     tmp = i.strip('\n').split(',')
# #     if (int(tmp[1]) > -1):
# #         feature_name.append(tmp[0])
# #     else:
# #         break
# count_dict = {}
# index_dict = {}
# filter_set = set()
# begin_index = 1
# round1_iflyad_train['user_tags'].fillna('-1',inplace = True)
#
#
# def extra_same_elem(list1, list2):
#     set1 = set(list1)
#     set2 = set(list2)
#     iset = set1.intersection(set2)
#     return list(iset)
#
# for id in adid_id:
#     for i, d in enumerate(round1_iflyad_train['user_tags']):
#         if i%10000 == 0:
#             print(i)
#         if round1_iflyad_train[id_name][i] != id :
#             continue
#         fea_name = round1_iflyad_train['user_tags'][i].split(',')
#         if '' in fea_name:
#             fea_name.remove('')
#         for fea in fea_name:
#             if fea not in count_dict.keys():
#                 count_dict[fea] = 0
#             count_dict[fea] += 1
#     c = count_dict.items()
#     count_dict_sort = sorted(count_dict.items(),key = lambda x:x[1],reverse = True)
#     count_dict_top20 = count_dict_sort
#     count_dict_top20 = [i[0] for i in count_dict_top20]
#     click_ratio_dict = {}
#     for index, row in round1_iflyad_train.iterrows():
#         if index%10000 == 0:
#             print(index)
#         if row[id_name]!= id :
#             continue
#         tmp = row['user_tags'].split(',')
#         if '' in tmp:
#             tmp.remove('')
#         same_tmp = extra_same_elem(count_dict_top20, tmp)
#         for i in same_tmp:
#             if i not in click_ratio_dict:
#                 click_ratio_dict[i] = 0
#             click_ratio_dict[i] += row['click']
#     click_ratio = pd.DataFrame([click_ratio_dict]).T.reset_index()
#     # click_ratio.rename(columns={'index':'user_tops','0':'click_ratio'},inplace = True)
#     click_ratio.columns = ['user_tops','click_num']
#     click_ratio['total_num'] = 0
#     for index, row in click_ratio.iterrows():
#         click_ratio['total_num'][index] = count_dict[row['user_tops']]
#     click_ratio['click_ratio'] = click_ratio['click_num']/click_ratio['total_num']
#     click_ratio = click_ratio[(click_ratio['total_num'] > 100)]
#     click_ratio.sort_values(by = 'click_ratio',axis = 0,ascending = False,inplace = True)
#     click_ratio_top = click_ratio.iloc[0:50].reset_index(drop=True)
#
#
#     plt.figure(figsize=(12,12))
#     sns.barplot(x = 'click_ratio', y = 'user_tops', data=click_ratio_top, orient='h',order = click_ratio_top['user_tops'])
#     # ax2 = ax.twinx()
#     # sns.barplot(y='total_num',x='click_ratio', data=click_ratio, orient='h')
#     plt.title('样本总数从大到小的点击率分布_'+id_name+'_'+str(id))
#     plt.xlabel('click_ratio')
#     plt.ylabel('user_tags')
#     plt.legend(loc='best')
#     # plt.savefig('data_analysis/user_tags/adid_'+ str(id) + '_user_tags_cilck_ratio.tiff')
#     plt.show()
#     plt.close()
########################################跟上面的方法一样，对应预处理的数据集做了调整，循环所有id类特征###############################################
# round1_iflyad_train = pd.read_csv('../data/original_filled_labelencode_data.csv',nrows = 200000)
# # round1_iflyad_train = pd.read_csv('../data/round1_iflyad_train.txt',delimiter="\t",nrows = 50000)
#
# id_set = ['adid','advert_id','orderid' ,'osv1', 'osv2', 'osv3', 'make', 'model', 'creative_id', 'creative_type', 'app_id',
#              'creative_tp_dnf', 'app_cate_id', 'advert_industry_inner_one'
#     , 'advert_industry_inner_two'
#     , 'inner_slot_id']
#
#
# for id_name in id_set:
#     count = 0
#     c = round1_iflyad_train[id_name].value_counts()
#     id_num= list(round1_iflyad_train[id_name].value_counts().index._data)
#
#     # f = open('top_uset_tags.txt', 'r')
#     # all_name = f.readlines()
#     # feature_name = []
#     # for i in all_name:
#     #     tmp = i.strip('\n').split(',')
#     #     if (int(tmp[1]) > -1):
#     #         feature_name.append(tmp[0])
#     #     else:
#     #         break
#     count_dict = {}
#     index_dict = {}
#     filter_set = set()
#     begin_index = 1
#     round1_iflyad_train['user_tags'].fillna('-1',inplace = True)
#
#
#     def extra_same_elem(list1, list2):
#         set1 = set(list1)
#         set2 = set(list2)
#         iset = set1.intersection(set2)
#         return list(iset)
#
#     for id in id_num:
#         for i, d in enumerate(round1_iflyad_train['user_tags']):
#             if i%10000 == 0:
#                 print(i)
#             if round1_iflyad_train[id_name][i] != id :
#                 continue
#             fea_name = round1_iflyad_train['user_tags'][i].split(',')
#             if '' in fea_name:
#                 fea_name.remove('')
#             for fea in fea_name:
#                 if fea not in count_dict.keys():
#                     count_dict[fea] = 0
#                 count_dict[fea] += 1
#         c = count_dict.items()
#         count_dict_sort = sorted(count_dict.items(),key = lambda x:x[1],reverse = True)
#         count_dict_top20 = count_dict_sort
#         count_dict_top20 = [i[0] for i in count_dict_top20]
#         click_ratio_dict = {}
#         for index, row in round1_iflyad_train.iterrows():
#             if index%10000 == 0:
#                 print(index)
#             if row[id_name]!= id :
#                 continue
#             tmp = row['user_tags'].split(',')
#             if '' in tmp:
#                 tmp.remove('')
#             same_tmp = extra_same_elem(count_dict_top20, tmp)
#             for i in same_tmp:
#                 if i not in click_ratio_dict:
#                     click_ratio_dict[i] = 0
#                 click_ratio_dict[i] += row['click']
#         click_ratio = pd.DataFrame([click_ratio_dict]).T.reset_index()
#         # click_ratio.rename(columns={'index':'user_tops','0':'click_ratio'},inplace = True)
#         click_ratio.columns = ['user_tops','click_num']
#         click_ratio['total_num'] = 0
#         for index, row in click_ratio.iterrows():
#             click_ratio['total_num'][index] = count_dict[row['user_tops']]
#         click_ratio['click_ratio'] = click_ratio['click_num']/click_ratio['total_num']
#         click_ratio = click_ratio[(click_ratio['total_num'] > 100)]
#         click_ratio.sort_values(by = 'click_ratio',axis = 0,ascending = False,inplace = True)
#         click_ratio_top = click_ratio.iloc[0:50].reset_index(drop=True)
#
#
#         plt.figure(figsize=(12,12))
#         sns.barplot(x = 'click_ratio', y = 'user_tops', data=click_ratio_top, orient='h',order = click_ratio_top['user_tops'])
#         # ax2 = ax.twinx()
#         # sns.barplot(y='total_num',x='click_ratio', data=click_ratio, orient='h')
#         plt.title('_'.join([id_name,str(id),'中每个user_tags点击率分布(降序)']))
#         plt.xlabel('click_ratio')
#         plt.ylabel('user_tags_idnum')
#         # plt.legend(loc='best')
#         # plt.savefig('_'.join(['data_analysis/user_tags/',id_name,'idnum',str(id), 'user_tags_cilck_ratio.tiff']))
#         plt.show()
#         plt.close()
#         count += 1
#         if count>10:
#             break
#############################################user_tags多值特征分析-用户每个属性对应单个id num 点击率#################################################
# round1_iflyad_train = pd.read_csv('../data/original_filled_labelencode_data.csv',nrows = 200000)
# # round1_iflyad_train = pd.read_csv('../data/round1_iflyad_train.txt',delimiter="\t",nrows = 50000)
#
# id_set = ['adid','advert_id','orderid' ,'osv1', 'osv2', 'osv3', 'make', 'model', 'creative_id', 'creative_type', 'app_id',
#              'creative_tp_dnf', 'app_cate_id', 'advert_industry_inner_one'
#     , 'advert_industry_inner_two'
#     , 'inner_slot_id']
#
#
# for id_name in id_set:
#     count = 0
#     c = round1_iflyad_train[id_name].value_counts()
#     id_num= list(round1_iflyad_train[id_name].value_counts().index._data)
#
#     count_dict = {}
#     index_dict = {}
#     filter_set = set()
#     begin_index = 1
#     round1_iflyad_train['user_tags'].fillna('-1',inplace = True)
#
#     def extra_same_elem(list1, list2):
#         set1 = set(list1)
#         set2 = set(list2)
#         iset = set1.intersection(set2)
#         return list(iset)
#
#     for id in id_num:
#         for i, d in enumerate(round1_iflyad_train['user_tags']):
#             if i%10000 == 0:
#                 print(i)
#             if round1_iflyad_train[id_name][i] != id :
#                 continue
#             fea_name = round1_iflyad_train['user_tags'][i].split(',')
#             if '' in fea_name:
#                 fea_name.remove('')
#             for fea in fea_name:
#                 if fea not in count_dict.keys():
#                     count_dict[fea] = 0
#                 count_dict[fea] += 1
#
#         count_dict_sort = sorted(count_dict.items(),key = lambda x:x[1],reverse = True)
#         count_dict_top20 = count_dict_sort
#         count_dict_top20 = [i[0] for i in count_dict_top20]
#         click_ratio_dict = {}
#         for index, row in round1_iflyad_train.iterrows():
#             if index%10000 == 0:
#                 print(index)
#             if row[id_name]!= id :
#                 continue
#             tmp = row['user_tags'].split(',')
#             if '' in tmp:
#                 tmp.remove('')
#             same_tmp = extra_same_elem(count_dict_top20, tmp)
#             for i in same_tmp:
#                 if i not in click_ratio_dict:
#                     click_ratio_dict[i] = 0
#                 click_ratio_dict[i] += row['click']
#         click_ratio = pd.DataFrame([click_ratio_dict]).T.reset_index()
#         # click_ratio.rename(columns={'index':'user_tops','0':'click_ratio'},inplace = True)
#         click_ratio.columns = ['user_tops','click_num']
#         click_ratio['total_num'] = 0
#         for index, row in click_ratio.iterrows():
#             click_ratio['total_num'][index] = count_dict[row['user_tops']]
#         click_ratio['click_ratio'] = click_ratio['click_num']/click_ratio['total_num']
#         click_ratio = click_ratio[(click_ratio['total_num'] > 100)]
#         click_ratio.sort_values(by = 'click_ratio',axis = 0,ascending = False,inplace = True)
#         click_ratio_top = click_ratio.iloc[0:50].reset_index(drop=True)
#
#
#         plt.figure(figsize=(12,12))
#         sns.barplot(x = 'click_ratio', y = 'user_tops', data=click_ratio_top, orient='h',order = click_ratio_top['user_tops'])
#         # ax2 = ax.twinx()
#         # sns.barplot(y='total_num',x='click_ratio', data=click_ratio, orient='h')
#         plt.title('_'.join([id_name,str(id),'中每个user_tags点击率分布(降序)']))
#         plt.xlabel('click_ratio')
#         plt.ylabel('user_tags_idnum')
#         # plt.legend(loc='best')
#         # plt.savefig('_'.join(['data_analysis/user_tags/',id_name,'idnum',str(id), 'user_tags_cilck_ratio.tiff']))
#         plt.show()
#         plt.close()
#         count += 1
#         if count>10:
#             break
