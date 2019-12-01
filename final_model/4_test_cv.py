#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2018.09.19
from contextlib import contextmanager
import numpy as np
import pandas as pd
import time
import datetime
import gc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, log_loss
from scipy import sparse
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.stats import ranksums
from reduce_memory_parallel import reduce_mem_usage_parallel
from sklearn.preprocessing import LabelEncoder

# @contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
def property_feature(org):
    def splitx(x):
        if  isinstance(x, float):
            return str(x)
        else:
            return x.split(',')
    org['user_tags'].fillna('-1',inplace = True)
    tmp=org['user_tags'].apply(splitx).values
    print('dict prepare ')
    property_dict={}
    property_list=[]
    for i in tmp:
        property_list+=i
    for i in property_list:
        if i in property_dict:
            property_dict[i]+=1
        else:
            property_dict[i] = 1
    print('dict finish')
    def top(x):
        propertys=x.split(',')
        cnt=[property_dict[i] for i in propertys]
        res=sorted(zip(propertys,cnt),key=lambda x:x[1],reverse=True)
        top1=res[0][0]
        top2 = '_'.join([i[0] for i in res[:2]])
        top3 = '_'.join([i[0] for i in res[:3]])
        top4 = '_'.join([i[0] for i in res[:4]])
        top5='_'.join([i[0] for i in res[:5]])
        top10 = '_'.join([i[0] for i in res[:10]])
        return (top1,top2,top3,top4,top5,top10)
    org['top']=org['user_tags'].apply(top)
    print('top finish')
    org['top1']=org['top'].apply(lambda x:x[0])
    org['top2'] = org['top'].apply(lambda x: x[1])
    org['top3'] = org['top'].apply(lambda x: x[2])
    org['top4'] = org['top'].apply(lambda x: x[3])
    org['top5'] = org['top'].apply(lambda x: x[4])
    org['top10'] = org['top'].apply(lambda x: x[5])
    a = org[['instance_id','top1','top2','top3','top4','top5','top10']]
    return a

def main (debug = False,part = 1):
    num_rows = 60000 if debug else None
    if part ==1:
        data = pd.read_csv('../data/original_filled_labelencode_data.csv',nrows = num_rows)

        aid = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_one', 'advert_industry_inner_two',
               'advert_name',
               'campaign_id', 'creative_id',
               'creative_type', 'creative_tp_dnf']

        mid = ['app_cate_id', 'app_id', 'inner_slot_id']
        # 用户信息
        uid = ['user_tags']
        # 上下文信息
        tid = ['city_city', 'city_province', 'city_zone', 'osv1', 'osv2', 'osv3', 'os', 'make', 'model']
        f1_id = uid + mid + tid
        del_col = []
        for f1 in f1_id:
            for f2 in aid:
                del_col.append(f1 + '_' + f2 + '_one_hot')
        data.drop(del_col,axis = 1,inplace = True)
        # data.drop(drop_col,axis = 1,inplace = True)
        # data = pd.concat([data,data_time],axis=1)
        # del train,test
        # gc.collect()
        # data['time_day'] = data['time'].apply(lambda x : int(time.strftime("%d", time.localtime(x))))
        # data['time_hour'] = data['time'].apply(lambda x : int(time.strftime("%H", time.localtime(x))))
        df_top = property_feature(data.copy())
        data = data.merge(df_top,on = 'instance_id',how = 'left')
        del df_top
        gc.collect()
        encoder = ['top1','top2','top3','top4','top5','top10']
        col_encoder = LabelEncoder()
        for feat in encoder:
            col_encoder.fit(data[feat])
            data[feat] = col_encoder.transform(data[feat])
        # data = id_top_feature(data, len(train), 100)
        # 历史点击率
        # data = pd.read_csv('../data/original_data.csv',nrows = None)
        print(data.info(verbose=True, null_counts=True))
        # train = data[data['time_day']<34]
        # test = data[data['time_day']==34]
        # test.drop('click',axis = 1,inplace = True)
        # train.to_csv('../data/original_data_train.csv',index = False)
        # test.to_csv('../data/original_data_test.csv',index = False)
        data['period'] = data['time_day']
        # 时间转换
        data['period'][data['period']<27] = data['period'][data['period']<27] + 31
        data.drop(['time_day','user_tags'],axis = 1,inplace = True)
        # df_top = property_feature(data.copy())
        # data = data.merge(df_top,on = 'instance_id',how = 'left')
        data = reduce_mem_usage_parallel(data,num_worker=10)
        print(data.info(verbose=True, null_counts=True))
        #########################################################################
        # print(data.info(verbose=True, null_counts=True))
        for feat_1 in ['advert_id','advert_industry_inner_one','advert_industry_inner_two','advert_name','campaign_id', 'creative_height'
                      , 'creative_tp_dnf', 'creative_width', 'f_channel']:
            gc.collect()
            res=pd.DataFrame()
            temp=data[[feat_1,'period','click']]
            for period in range(27,35):
                if period == 27:
                    count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].count()).reset_index(name=feat_1+'_all')
                    count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].sum()).reset_index(name=feat_1+'_1')
                else:
                    count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<period).values].count()).reset_index(name=feat_1+'_all')
                    count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<period).values].sum()).reset_index(name=feat_1+'_1')
                count[feat_1+'_1']=count1[feat_1+'_1']
                count.fillna(value=0, inplace=True)
                count[feat_1+'_rate'] = round(count[feat_1+'_1'] / count[feat_1+'_all'], 5)
                count['period']=period
                count.drop([feat_1+'_all', feat_1+'_1'],axis=1,inplace=True)
                count.fillna(value=0, inplace=True)
                res=res.append(count,ignore_index=True)
            print(feat_1,' over')
            data = pd.merge(data,res, how='left', on=[feat_1,'period'])
        # #####################
        round1_iflyad_train =data[data['click'] >-1]
        temp1 = round1_iflyad_train['city_city'].value_counts().reset_index()
        temp1 = temp1.rename(index = str,columns = {'index':'city_city','city_city':'city_num'})
        data = data.merge(temp1,on = 'city_city',how = 'left')
        del temp1,round1_iflyad_train
        gc.collect()

        data.city_num[(data['city_num']>=0) & (data['city_num']<=70)] = 0
        data.city_num[(data['city_num']>70) & (data['city_num']<=300)] = 1
        data.city_num[(data['city_num']>300) & (data['city_num']<=850)] = 2
        data.city_num[(data['city_num']>850) & (data['city_num']<=2500)] = 3
        data.city_num[(data['city_num']>2500) & (data['city_num']<=7000)] = 4
        data.city_num[(data['city_num']>7000)] = 5
        values = list(data['city_num'].unique())

        for v in values:
            data[str('city_num') + '_' + str(v)] = (data['city_num'] == v).astype(np.uint8)
        data.drop('city_num',axis = 1,inplace = True)
        print(data.info(verbose=True, null_counts=True))
        data = reduce_mem_usage_parallel(data,num_worker=10)
        ####################
        def agg_count_feature_all_day1(data):
            print('1')

            for feat_2 in ['campaign_id','creative_id','creative_tp_dnf']:
                for feat_1 in ['creative_is_jump', 'creative_is_download', 'creative_is_js',
                               'creative_is_voicead', 'creative_has_deeplink','creative_width','creative_height']:
                    t0 = time.time()
                    gc.collect()
                    res = pd.DataFrame()
                    temp = data[[feat_1,feat_2,'click','period']]
                    for period in range(27, 35):
                        if period == 27:
                            count = temp[(temp['click']>-1) & (temp['period']<=period)].groupby( feat_2, as_index=False)[feat_1].agg({feat_2+'_'+feat_1 + '_mean': 'mean'})
                        else:
                            count = temp[(temp['click']>-1) & (temp['period']<period)].groupby( feat_2, as_index=False)[feat_1].agg({feat_2+'_'+feat_1 + '_mean': 'mean'})
                        count.fillna(value=0, inplace=True)
                        count['period'] = period
                        count.fillna(value=0, inplace=True)
                        res = res.append(count, ignore_index=True)
                    data = data.merge(res,on =[feat_2,'period'],how = 'left')
                    print(feat_2,feat_1, ' over')
                    del temp,count
                    gc.collect()
                    print(feat_2 + '_' + feat_1 + '_1', "____", "done in {:.0f}s".format(time.time() - t0))
            data = reduce_mem_usage_parallel(data, num_worker=10)
            print('原始id占比与出现次数')
            # 原始id占比与出现次数
            ratio_set = ['adid', 'campaign_id', 'creative_id', 'creative_type',
                         'advert_industry_inner_one'
                , 'advert_industry_inner_two',  'make', 'model', 'app_id', 'app_cate_id'
                , 'creative_tp_dnf', 'inner_slot_id', 'osv1', 'osv2','osv3','os',]

            for i in  range(len(ratio_set)):
                t0 = time.time()
                feat_1 = ratio_set[i]
                train = data[data['period']<34]
                test = data[data['period']==34]
                temp = train[[feat_1, 'period', 'click']]

                count = temp.groupby(feat_1,as_index = False)[feat_1].agg({feat_1+ '_cnt': 'count'})
                count.fillna(value = 0,inplace = True)
                temp1 = count.groupby(feat_1,as_index = False)[feat_1+ '_cnt'].agg({feat_1+ '_sum': 'sum'})
                count = count.merge(temp1,on = feat_1,how = 'left')
                count[feat_1+'_cat_ratio'] = count[feat_1+ '_cnt']/count[feat_1+ '_sum']
                count.fillna(value=0, inplace=True)

                print(feat_1, ' over')
                train = pd.merge(train, count, how='left', on=feat_1)
                test = pd.merge(test, count, how='left', on=feat_1)
                data = pd.concat([test,train],axis = 0)
                del train,test,count,temp
                gc.collect()
                print(feat_1, "____", "done in {:.0f}s".format(time.time() - t0))
            data = reduce_mem_usage_parallel(data, num_worker=10)

            print('组合id占比与出现次数')
            # 组合id占比与出现次数
            # ratio_set = ['adid', 'advert_id', 'campaign_id', 'creative_id', 'creative_type', 'advert_industry_inner',
            #              'advert_industry_inner_one'
            #     , 'advert_industry_inner_two', 'creative_type', 'province', 'city', 'make', 'model', 'app_id', 'app_cate_id'
            #     , 'creative_tp_dnf', 'inner_slot_id', 'osv', 'os', 'make', 'model']

            # ['creative_id','creative_type'],['app_cate_id','app_id'],['city','make'],['adid','campaign_id']

            ratio_set = ['adid', 'campaign_id', 'creative_id',
                         'advert_industry_inner_one'
                , 'advert_industry_inner_two',  'make', 'model', 'app_id', 'app_cate_id'
                , 'creative_tp_dnf', 'inner_slot_id','osv1', 'osv2','osv3', 'os', ]
            for i in  range(len(ratio_set)):
                for j in range(i+1,len(ratio_set)):
                    t0 = time.time()
                    feat_1 = ratio_set[i]
                    feat_2 = ratio_set[j]
                    train = data[data['period']<34]
                    test = data[data['period']==34]
                    temp = train[[feat_1,feat_2, 'period', 'click']]

                    count = temp.groupby([feat_1,feat_2],as_index = False)[feat_2].agg({'_'.join([feat_1,feat_2])+ '_cnt': 'count'})
                    count.fillna(value = 0,inplace = True)
                    temp1 = count.groupby(feat_1,as_index = False)['_'.join([feat_1,feat_2])+ '_cnt'].agg({'_'.join([feat_1,feat_2])+ '_sum': 'sum'})
                    count = count.merge(temp1,on = feat_1,how = 'left')
                    count[feat_1+feat_2+'_cat_ratio'] = count['_'.join([feat_1,feat_2])+ '_cnt']/count['_'.join([feat_1,feat_2])+ '_sum']
                    count.fillna(value=0, inplace=True)

                    print(feat_1,feat_2, ' over')
                    train = pd.merge(train, count, how='left', on=[feat_1,feat_2])
                    test = pd.merge(test, count, how='left', on=[feat_1, feat_2])
                    data = pd.concat([test,train],axis = 0)
                    del train,test,count,temp
                    gc.collect()
                    print(feat_1 + '_' + feat_2 + '_组合id占比与出现次数', "____", "done in {:.0f}s".format(time.time() - t0))
            data = reduce_mem_usage_parallel(data, num_worker=10)
            print('时间统计')
            # 时间统计
            # for feat_1 in ['user_tags','adid', 'campaign_id','creative_id','creative_type','creative_tp_dnf']:
            # ratio_set = ['adid', 'osv1', 'osv2','osv3' ,'make', 'model', 'creative_id', 'creative_type','app_id',
            #       'creative_tp_dnf',   'app_cate_id','advert_industry_inner_one'
            #     , 'advert_industry_inner_two'
            #     ,  'inner_slot_id','os', 'campaign_id']
            ratio_set = ['adid', 'osv1', 'osv2', 'osv3', 'make', 'model', 'creative_id', 'creative_type', 'app_id',
                         'creative_tp_dnf', 'app_cate_id', 'advert_industry_inner_one'
                , 'advert_industry_inner_two'
                , 'inner_slot_id']

            for feat_1 in ratio_set:
                print(feat_1,'per hour over!')
                train = data[data['period']<34]
                test = data[data['period'] == 34]
                hours_user_click = train.groupby([feat_1, 'time_hours'], as_index=False)['click'].agg({feat_1+'_per_hour_sum': 'count'})
                train = train.merge( hours_user_click, on=[feat_1,  'time_hours'], how='left')
                test = test.merge(hours_user_click, on=[feat_1,  'time_hours'], how='left')
                data = pd.concat([test,train],axis = 0)
                data[feat_1 + '_per_hour_sum'].fillna(0, inplace=True)

            data = reduce_mem_usage_parallel(data, num_worker=10)
            count = 0
            for i in  range(len(ratio_set)):
                if count>9:
                    break
                for j in range(i+1,len(ratio_set)):
                    t0 = time.time()
                    feat_1 = ratio_set[i]
                    feat_2 = ratio_set[j]
                    train = data[data['period']<34]
                    test = data[data['period'] == 34]
                    hours_user_click = train.groupby([feat_1,feat_2, 'time_hours'], as_index=False)['click'].agg({feat_1+'_'+feat_2+'_per_hour_sum': 'count'})
                    train = train.merge( hours_user_click, on=[feat_1,feat_2,  'time_hours'], how='left')
                    test = test.merge(hours_user_click, on=[feat_1,feat_2,  'time_hours'], how='left')
                    data = pd.concat([test,train],axis = 0)
                    data[feat_1+'_'+feat_2+'_per_hour_sum'].fillna(0, inplace=True)
                    del train, test
                    gc.collect()
                    print(feat_1 + '_' + feat_2 + '_时间统计', "____", "done in {:.0f}s".format(time.time() - t0))
                count +=1
            data.to_csv('final_feature_labelencode_data.csv', index=False)
            data = reduce_mem_usage_parallel(data, num_worker=10)
            return data
        data = agg_count_feature_all_day1(data)

    if part == 2:
        del_col = []
        # 加入2_user_tags_feature-id,3_user_tags_feature-flagid计算的特征
        round1_iflyad_train = pd.read_csv('original_filled_labelencode_user_tops_sum.csv',nrows = num_rows)
        round1_iflyad_train.drop('click',axis = 1, inplace = True)
        flag_id = pd.read_csv('original_filled_labelencode_flag_sum.csv', nrows=num_rows)
        flag_id.drop('click', axis=1, inplace=True)
        flag_user_id = pd.read_csv('original_filled_labelencode_user_flag_sum.csv', nrows=num_rows)
        flag_user_id.drop('click', axis=1, inplace=True)
        data = pd.read_csv('final_feature_labelencode_data.csv',nrows = num_rows)
        ratio_set = ['adid', 'osv1', 'osv2', 'osv3', 'make', 'model', 'creative_id', 'creative_type', 'app_id',
                     'creative_tp_dnf', 'app_cate_id', 'advert_industry_inner_one'
            , 'advert_industry_inner_two'
            , 'inner_slot_id']
        count = 0
        for i in range(len(ratio_set)):
            if count > 9:
                break
            for j in range(i + 1, len(ratio_set)):
                feat_1 = ratio_set[i]
                feat_2 = ratio_set[j]
                del_col.append(feat_1+'_'+feat_2+'_per_hour_sum')
            count +=1
        data.drop(del_col,axis = 1,inplace = True)
        data = reduce_mem_usage_parallel(data, num_worker=10)
        # 讯飞赛冠军公开的三个特征，不过加进去以后对我的模型没啥效果.
        # 广告
        adid_nuq = ['model', 'make', 'os', 'osv1', 'osv2', 'osv3', 'f_channel', 'app_id', 'carrier', 'nnt',
                    'devtype',
                    'app_cate_id', 'inner_slot_id']
        for feat in adid_nuq:
            print('广告:', feat)
            gp1 = data.groupby('adid')[feat].nunique().reset_index().rename(columns={feat: "adid_%s_nuq_num" % feat})
            gp2 = data.groupby(feat)['adid'].nunique().reset_index().rename(columns={'adid': "%s_adid_nuq_num" % feat})
            data = pd.merge(data, gp1, how='left', on=['adid'])
            data = pd.merge(data, gp2, how='left', on=[feat])
        ## 广告主
        advert_id_nuq = ['model', 'make', 'os',  'osv1', 'osv2', 'osv3', 'f_channel', 'app_id', 'carrier', 'nnt',
                         'devtype',
                         'app_cate_id', 'inner_slot_id']
        for fea in advert_id_nuq:
            print('广告主:', fea)
            gp1 = data.groupby('advert_id')[fea].nunique().reset_index().rename(columns={fea: "advert_id_%s_nuq_num" % fea})
            gp2 = data.groupby(fea)['advert_id'].nunique().reset_index().rename(
                columns={'advert_id': "%s_advert_id_nuq_num" % fea})
            data = pd.merge(data, gp1, how='left', on=['advert_id'])
            data = pd.merge(data, gp2, how='left', on=[fea])
        ## app_id
        app_id_nuq = ['model', 'make', 'os','osv1', 'osv2', 'osv3','f_channel', 'carrier', 'nnt', 'devtype',
                      'app_cate_id', 'inner_slot_id']
        for fea in app_id_nuq:
            print ('app_id:',fea)
            gp1 = data.groupby('app_id')[fea].nunique().reset_index().rename(columns={fea: "app_id_%s_nuq_num" % fea})
            gp2 = data.groupby(fea)['app_id'].nunique().reset_index().rename(columns={'app_id': "%s_app_id_nuq_num" % fea})
            data = pd.merge(data, gp1, how='left', on=['app_id'])
            data = pd.merge(data, gp2, how='left', on=[fea])
        ####################################################################
        data = data.merge(round1_iflyad_train, on='instance_id', how='left')
        data = data.merge(flag_id, on='instance_id', how='left')
        data = data.merge(flag_user_id, on='instance_id', how='left')
        data = reduce_mem_usage_parallel(data, num_worker=10)
        data.to_csv('final_feature_merge_data.csv', index=False)
    if part == 3:
        # 特征选择加过滤
        df = pd.read_csv('final_feature_merge_data.csv')
        df.loc[df['click'] == -1, 'click'] = np.nan
        if debug:
            df_train = df[df['click'].notnull()].iloc[0:10000].reset_index(drop=True)
            df_test = df[df['click'].isnull()].iloc[0:10000].reset_index(drop=True)
            df = pd.concat([df_train, df_test], axis=0)
        df = reduce_mem_usage_parallel(df, 10)
        def corr_feature_with_target(feature, target, debug=False):
            c0 = feature[target == 0].dropna()
            c1 = feature[target == 1].dropna()
            if set(feature.unique()) == set([0, 1]):
                diff = abs(c0.mean(axis=0) - c1.mean(axis=0))
            else:
                if (debug):
                    diff = abs(c0.mean(axis=0) - c1.mean(axis=0))
                else:
                    diff = abs(c0.median(axis=0) - c1.median(axis=0))
            # 样本量20以下为小样本情况
            p = ranksums(c0, c1)[1] if ((len(c0) >= 20) & (len(c1) >= 20)) else 2
            return [diff, p]

        # Removing empty features
        nun = df.nunique()
        empty = list(nun[nun <= 1].index)
        print('Before removing empty or constant features there are {0:d} features'.format(df.shape[1]))
        df.drop(empty, axis=1, inplace=True)
        print('After removing empty or constant features there are {0:d} features'.format(df.shape[1]))

        # Removing features with the same distribution on 0 and 1 classes
        corr = pd.DataFrame(index=['diff', 'p'])
        ind = df[df['click'].notnull()].index
        for c in df.columns.drop('click'):
            corr[c] = corr_feature_with_target(df.loc[ind, c], df.loc[ind, 'click'], debug)
        corr = corr.T
        corr['diff_norm'] = abs(corr['diff'] / df.mean(axis=0))

        to_del_1 = corr[((corr['diff'] == 0) & (corr['p'] > .05))].index
        to_del_2 = corr[((corr['diff_norm'] < .5) & (corr['p'] > .05))]
        for i in to_del_1:
            if (i in to_del_2.index):
                to_del_2 = to_del_2.drop(i)

        to_del = list(to_del_1) + list(to_del_2.index)
        if 'instance_id' in to_del:
            to_del.remove('instance_id')

        df.drop(to_del, axis=1, inplace=True)
        print('After removing features with the same distribution on 0 and 1 classes there are {0:d} features'.format(
            df.shape[1]))

        # Removing features with not the same distribution on train and test datasets
        corr_test = pd.DataFrame(index=['diff', 'p'])
        # 提取train和test里面click，非空的为1，空为0
        target = df['click'].notnull().astype(int)

        for c in df.columns.drop('click'):
            corr_test[c] = corr_feature_with_target(df[c], target, debug)

        corr_test = corr_test.T
        corr_test['diff_norm'] = abs(corr_test['diff'] / df.mean(axis=0))
        # P = < 0.05，故拒绝原假设下，认为分布有差异
        bad_features = corr_test[((corr_test['p'] < .05) & (corr_test['diff_norm'] > 1))].index
        bad_features = corr.loc[bad_features][corr['diff_norm'] == 0].index

        df.drop(bad_features, axis=1, inplace=True)
        print(
            'After removing features with not the same distribution on train and test datasets there are {0:d} features'.format(
                df.shape[1]))
        del corr, corr_test
        gc.collect()
        # Removing features not interesting for classifier
        clf = lgb.LGBMClassifier(random_state=0)
        train_index = df[df['click'].notnull()].index
        train_columns = df.drop('click', axis=1).columns

        score = 0.1
        new_columns = []
        while score < .4250:
            train_columns = train_columns.drop(new_columns)
            print(train_columns.shape[0])
            clf.fit(df.iloc[train_index][train_columns], df.iloc[train_index][['click']])
            f_imp = pd.Series(clf.feature_importances_, index=train_columns)
            score = log_loss(df.iloc[train_index][['click']],
                             clf.predict_proba(df.iloc[train_index][train_columns])[:, 1])
            new_columns = f_imp[f_imp > 0].index
            print(score)
        df.drop(train_columns, axis=1, inplace=True)
        print('After removing features not interesting for classifier there are {0:d} features'.format(df.shape[1]))
        # df.to_csv('feature_selected_less0.425.csv', index=False)
        data_original = pd.read_csv('../data/original_filled_labelencode_data.csv')
        data_user = data_original[['instance_id','user_tags']]
        df = df.merge(data_user,on = 'instance_id', how = 'left')
        df.to_csv('feature_selected_less0.425+user_tags.csv',index = False)
        print('save_finshed')
        del data_original
        gc.collect()

    if part ==4:
        # 做stacking，得到的预测值作为新的特征放入原始特征集合。
        data = pd.read_csv('feature_selected_less0.425+user_tags.csv',nrows = num_rows)
        if debug:
            data_train = data[data['click'].notnull()].iloc[0:10000].reset_index(drop=True)
            data_test = data[data['click'].isnull()].iloc[0:10000].reset_index(drop=True)
            data = pd.concat([data_train,data_test],axis = 0)
        data['click'].fillna(-1,inplace = True)
        data.fillna(0, inplace=True)
        data = reduce_mem_usage_parallel(data,num_worker=10)
        one_hot_feature = ['adid', 'advert_id', 'make', 'model','carrier']
        drop = [    "instance_id", "click",
            "period",'time']
        label_enc = LabelEncoder()
        for feature in one_hot_feature:
            label_enc.fit(data[feature])
            try:
                data[feature] = label_enc.transform(data[feature].apply(int))

            except:
                data[feature] = label_enc.transform(data[feature])
            print(feature, '__________finished!')
        train = data[data['click']>-1]
        test = data[data['click']==-1]
        y_train = train.loc[:,'click']
        # y_class = train.loc[:,'click_class']
        res = test.loc[:, ['instance_id']]
        train_lgb = train[['instance_id']]
        test_lgb = test[['instance_id']]
        train_lgb['stack_lgb'] = 0
        test_lgb['stack_lgb'] = 0
        train.drop(drop, axis=1, inplace=True)
        test.drop(drop, axis=1, inplace=True)
        data.drop(drop, axis=1, inplace=True)
        ##################################ONEhot并转为稀疏矩阵，节省内存###########################################
        cv = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
        cv.fit(data['user_tags'])
        train_a = cv.transform(train['user_tags'])
        test_a = cv.transform(test['user_tags'])
        train.drop('user_tags', axis=1, inplace=True)
        test.drop('user_tags', axis=1, inplace=True)
        data.drop('user_tags', axis=1, inplace=True)
        feats = [f for f in train.columns]
        train_x = train.to_sparse().to_coo()
        test_x = test.to_sparse().to_coo()
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
        print('one-hot prepared !')
        del data,train,test
        gc.collect()
        #############################################################################################################
        y_loc_train = y_train.values
        # 模型部分
        model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=2000,
                                   max_bin=250, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                                   min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, device = 'gpu', silent=True)
        # model = lgb.LGBMClassifier(
        #     boosting_type='gbdt',
        #     objective='binary',
        #     n_estimators=5000,
        #     learning_rate=0.02,
        #     num_leaves=30,
        #     max_bin=250,
        #     max_depth=-1,
        #     min_child_samples=70,
        #     subsample=1.0,
        #     subsample_freq=1,
        #     colsample_bytree=0.05,
        #     min_gain_to_split=0.5,
        #     reg_lambda=100.0,
        #     reg_alpha=0.0,
        #     scale_pos_weight=1,
        #     is_unbalance=False,
        #     # n_jobs = cpu_count() - 1,
        #     device='gpu'
        # )
        # 五折交叉训练，构造五个模型
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1024)
        # skf=list(StratifiedKFold(y_loc_train, n_splits=5, shuffle=True, random_state=1024))
        baseloss = []
        loss = 0
        oof_preds = np.zeros(train_x.shape[0])
        cv_index = folds.split(train_x, y_loc_train)
        def get_oof(clf_name, train_x, y_loc_train, test_x,  cv_index, model,num_folds):

            oof_train = np.zeros((train_x.shape[0],))
            oof_test = np.zeros((test_x.shape[0],))
            oof_test_skf = np.empty((num_folds, test_x.shape[0]))
            # tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            for i, (train_index, test_index) in enumerate(cv_index):
                print ('kfold num:',str(i))
                # clf.train(x_tr, y_tr, x_te, y_te)
                clf = model.fit(train_x.tocsr()[train_index], y_loc_train[train_index],
                                eval_names=['train', 'valid'],
                                eval_metric='logloss',
                                eval_set=[(train_x.tocsr()[train_index], y_loc_train[train_index]),
                                          (train_x.tocsr()[test_index], y_loc_train[test_index])],
                                early_stopping_rounds=100, verbose=100)
                # 存入交叉验证结果
                oof_train[test_index] = clf.predict_proba(train_x.tocsr()[test_index], num_iteration=clf.best_iteration_)[:, 1]
                # 存入测试集结果
                oof_test_skf[i, :] = clf.predict_proba(test_x, num_iteration=clf.best_iteration_)[:, 1]
                del clf
                gc.collect()
                print(str(i),'__finsihed!')
            # 测试集结果取均值保存为一列，每个模型一列
            oof_test[:] = oof_test_skf.mean(axis=0)
            gc.collect()
            return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

        xg_oof_train, xg_oof_test = get_oof('xg', train_x, y_loc_train, test_x, cv_index,model,num_folds=5)
        train_lgb['stack_lgb'] = xg_oof_train
        test_lgb['stack_lgb'] = xg_oof_test
        data = pd.concat([train_lgb,test_lgb],axis = 0).reset_index(drop = True)
        data[['instance_id','stack_lgb']].to_csv('stacking_feature.csv',index = False)
        print('save finished')

    if part == 5:
        # 最终特征训练模型,通过cv进行本地验证
        data = pd.read_csv('feature_selected_less0.425+user_tags.csv', nrows=num_rows)
        if debug:
            data_train = data[data['click'].notnull()].iloc[0:10000].reset_index(drop=True)
            data_test = data[data['click'].isnull()].iloc[0:10000].reset_index(drop=True)
            data = pd.concat([data_train, data_test], axis=0)
        data['click'].fillna(-1, inplace=True)
        data.fillna(0, inplace=True)
        print(data.info(verbose=True, null_counts=True))
        # 将stacking的结果作为新特征加入原始特征一起训练
        stack_fea = pd.read_csv('stacking_feature.csv',nrows = num_rows)
        data = data.merge(stack_fea,on = 'instance_id',how = 'left')
        data = reduce_mem_usage_parallel(data, num_worker=10)
        one_hot_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_one', 'advert_industry_inner_two',
                           'advert_name', 'campaign_id', 'creative_id',
                           'creative_type', 'creative_tp_dnf', 'app_cate_id', 'app_id', 'f_channel',
                           'inner_slot_id', 'carrier',
                           'city_city', 'city_province', 'city_zone', 'osv1', 'osv2', 'osv3', 'os', 'make', 'model',
                           'nnt',
                           'creative_width', 'creative_height']
        one_hot_feature = ['adid', 'advert_id', 'make', 'model', 'carrier']
        count_feature = ['user_tags']
        drop = ["instance_id", "click",
                "period", 'time']
        label_enc = LabelEncoder()
        for feature in one_hot_feature:
            label_enc.fit(data[feature])
            try:
                data[feature] = label_enc.transform(data[feature].apply(int))
            except:
                data[feature] = label_enc.transform(data[feature])
            print(feature, '__________finished!')

        train = data[data['click'] > -1]
        test = data[data['click'] == -1]

        y_train = train.loc[:, 'click']
        # y_class = train.loc[:,'click_class']
        res = test.loc[:, ['instance_id']]

        train.drop(drop, axis=1, inplace=True)
        test.drop(drop, axis=1, inplace=True)
        data.drop(drop, axis=1, inplace=True)

        ##################################ONEhot并转为系数矩阵，节省内存###########################################
        cv = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
        cv.fit(data['user_tags'])
        train_a = cv.transform(train['user_tags'])
        test_a = cv.transform(test['user_tags'])
        train.drop('user_tags', axis=1, inplace=True)
        test.drop('user_tags', axis=1, inplace=True)
        data.drop('user_tags', axis=1, inplace=True)
        feats = [f for f in train.columns]
        train_x = train.to_sparse().to_coo()
        test_x = test.to_sparse().to_coo()
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
        print('one-hot prepared !')
        del data, train, test
        gc.collect()
        #############################################################################################################
        y_loc_train = y_train.values
        # X_loc_test = test.values
        # y_loc_class =y_class.values
        # 模型部分
        model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05,
                                   n_estimators=2000,
                                   max_bin=250, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                                   min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, device='gpu',
                                   silent=True)
        # model = lgb.LGBMClassifier(
        #     boosting_type='gbdt',
        #     objective='binary',
        #     n_estimators=5000,
        #     learning_rate=0.02,
        #     num_leaves=30,
        #     max_bin=250,
        #     max_depth=-1,
        #     min_child_samples=70,
        #     subsample=1.0,
        #     subsample_freq=1,
        #     colsample_bytree=0.05,
        #     min_gain_to_split=0.5,
        #     reg_lambda=100.0,
        #     reg_alpha=0.0,
        #     scale_pos_weight=1,
        #     is_unbalance=False,
        #     # n_jobs = cpu_count() - 1,
        #     device='gpu'
        # )
        # 五折交叉训练，构造五个模型

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1024)
        # skf=list(StratifiedKFold(y_loc_train, n_splits=5, shuffle=True, random_state=1024))
        baseloss = []
        loss = 0
        cv_index = folds.split(train_x, y_loc_train)
        # feats.append('stack_lgb')
        feature_importances = dict(zip(feats,np.zeros(len(feats))))
        feature_drop_no_im = dict(zip(feats,np.zeros(len(feats))))
        del cv_index
        gc.collect()
        cv_index = folds.split(train_x, y_loc_train)
        oof_preds = np.zeros(train_x.shape[0])
        for i, (train_index, test_index) in enumerate(cv_index):
            print("Fold", i)
            clf = model.fit(train_x.tocsr()[train_index], y_loc_train[train_index],
                                  eval_names =['train','valid'],
                                  eval_metric='logloss',
                                  eval_set=[(train_x.tocsr()[train_index], y_loc_train[train_index]),
                                            (train_x.tocsr()[test_index], y_loc_train[test_index])],early_stopping_rounds=100,verbose=100)
            baseloss.append(clf.best_score_['valid']['binary_logloss'])
            loss += clf.best_score_['valid']['binary_logloss']
            oof_preds[test_index] = clf.predict_proba(train_x.tocsr()[test_index], num_iteration=clf.best_iteration_)[:, 1]
            test_pred= clf.predict_proba(test_x, num_iteration=clf.best_iteration_)[:, 1]
            print('test mean:', test_pred.mean())
            res['prob_%s' % str(i)] = test_pred
            feature_importances1 = dict(zip(feats, clf.feature_importances_))
            for i in feature_importances1:
                feature_importances[i] = feature_importances[i] + feature_importances1[i]
                if (feature_importances1[i] == 0):
                    feature_drop_no_im[i] = feature_drop_no_im[i] + 1

        for i in feature_importances:
            feature_importances[i] = feature_importances[i]/5
        feature_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        file = open('feature_importance_select.txt', 'w')
        for i in range(len(feature_importances)):
            file.write(str(feature_importances[i]) + '\n')
        file.close()
        print('logloss:', baseloss)
        print('meanloss', loss/5)
        print('Full logloss score %.6f' % log_loss(y_loc_train, oof_preds))
        # 加权平均
        res['predicted_score'] = 0
        for i in range(5):
            res['predicted_score'] += res['prob_%s' % str(i)]
        res['predicted_score'] = res['predicted_score']/5

        # 提交结果
        mean = res['predicted_score'].mean()
        print('mean:',mean)
        now = datetime.datetime.now()
        now = now.strftime('%m-%d-%H-%M')
        res[['instance_id', 'predicted_score']].to_csv("lgb_baseline_%s.csv" % now, index=False)
if __name__ == "__main__":
    main(debug = True,part =1)
