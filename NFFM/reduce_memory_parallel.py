import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import gc
import copy
from functools import partial
import warnings

warnings.simplefilter(action='ignore', category=RuntimeWarning)
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    return df

def reduce_mem_usage_parallel(df_original,num_worker):
    print('reduce_mem_usage_parallel start!')
    chunk_size = df_original.columns.shape[0]
    start_mem = df_original.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    if df_original.columns.shape[0]>500:
        group_chunk = []
        for  name in df_original.columns:
            group_chunk.append(df_original[[name]])
        with mp.Pool(num_worker) as executor:
            df_temp = executor.map(reduce_mem_usage,group_chunk)
        del group_chunk
        gc.collect()
        df_original = pd.concat(df_temp,axis = 1)
        end_mem = df_original.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        del df_temp
        gc.collect()
    else:
        df_original = reduce_mem_usage(df_original)
        end_mem = df_original.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df_original