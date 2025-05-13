import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from pandarallel import pandarallel

from utils import Logger

pd.set_option('display.max_columns', None)  # 在打印 `DataFrame` 时显示所有的列，不进行省略
pd.set_option('display.max_rows', None)

pandarallel.initialize()  # 这是对 `pandarallel`（一个加速 `pandas` 操作的并行计算库）的初始化调用。

warnings.filterwarnings('ignore')

seed = 2020

# 命令行参数
parser = argparse.ArgumentParser(description='排序特征')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'排序特征，mode: {mode}')


def func_if_sum(x):  # 这段函数用于构造一个推荐排序特征，衡量“当前文章与用户最近兴趣的相似程度”。用户越是近期点击的文章，对当前候选文章的影响越大。根据用户过往点击的文章，对当前文章的相似度加权求和。
    user_id = x['user_id']
    article_id = x['article_id']

    interacted_items = user_item_dict[user_id]  # 取出该用户的历史点击文章列表。
    interacted_items = interacted_items[::-1]  # 将点击序列倒序，也就是说最近点击的文章排在前面。

    sim_sum = 0
    for loc, i in enumerate(interacted_items):  # 遍历用户点击的每篇历史文章 i，loc 表示其在倒序列表中的位置。
        try:
            sim_sum += item_sim[i][article_id] * (0.7**loc)  # 取出文章 i 与当前文章 article_id 的相似度，做一个加权（时间衰减因子）：
            # 最近看的文章 loc=0，权重最大（$0.7^0=1$）；
            # 越往前看的文章，权重越小（如 $0.7^1=0.7$，$0.7^2=0.49$）。
        except Exception as e:
            pass
    return sim_sum


def func_if_last(x):  # 计算当前候选文章与用户最近一次点击的文章之间的相似度。
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = item_sim[last_item][article_id]
    except Exception as e:
        pass
    return sim


def func_binetwork_sim_last(x):  # 计算当前候选文章与用户最近一次点击文章之间的双网络相似度（bi-network similarity）。
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = binetwork_sim[last_item][article_id]
    except Exception as e:
        pass
    return sim


def consine_distance(vector1, vector2):  # 计算两个向量之间的余弦相似度（cosine similarity），用于衡量它们的方向相似程度。
    if type(vector1) != np.ndarray or type(vector2) != np.ndarray:  # 先判断两个输入是否都是 numpy 数组。不是就返回 -1。
        return -1
    distance = np.dot(vector1, vector2) / \
        (np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    return distance


def func_w2w_sum(x, num):  # 用来计算用户最近点击的若干文章与候选文章之间的向量相似度总和，常用于推荐排序特征构造，尤其在使用文章的 word2vec 时。
    user_id = x['user_id']
    article_id = x['article_id']

    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1][:num]

    sim_sum = 0
    for loc, i in enumerate(interacted_items):
        try:
            sim_sum += consine_distance(article_vec_map[article_id],
                                        article_vec_map[i])
        except Exception as e:
            pass
    return sim_sum


def func_w2w_last_sim(x):  # 当前候选文章与用户最后点击的文章之间的向量相似度（使用余弦相似度），特别是在基于文章的 word2vec 或其他向量表示时。
    user_id = x['user_id']
    article_id = x['article_id']

    last_item = user_item_dict[user_id][-1]

    sim = 0
    try:
        sim = consine_distance(article_vec_map[article_id],
                               article_vec_map[last_item])
    except Exception as e:
        pass
    return sim


if __name__ == '__main__':
    if mode == 'valid':
        df_feature = pd.read_pickle('../user_data/data/offline/recall.pkl')
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')

    else:
        df_feature = pd.read_pickle('../user_data/data/online/recall.pkl')
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')

    # 文章特征
    log.debug(f'df_feature.shape: {df_feature.shape}')

    df_article = pd.read_csv('../tcdata/articles.csv')
    df_article['created_at_ts'] = df_article['created_at_ts'] / 1000
    df_article['created_at_ts'] = df_article['created_at_ts'].astype('int')
    df_feature = df_feature.merge(df_article, how='left')
    df_feature['created_at_datetime'] = pd.to_datetime(
        df_feature['created_at_ts'], unit='s')  # df_feature['created_at_ts']：这是 DataFrame 中包含时间戳的列（时间戳单位是秒）。
    # 将时间戳转换为 datetime 类型（即可读的日期时间格式）。具体来说，这个过程涉及将表示**自1970年1月1日 UTC 起的秒数（UNIX时间戳）**转换为标准的日期和时间表示形式。


    log.debug(f'df_article.head(): {df_article.head()}')
    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 历史记录相关特征
    df_click.sort_values(['user_id', 'click_timestamp'], inplace=True)  # 处理用户点击行为日志，将点击时间戳转换为可读时间格式，并提取点击小时信息
    df_click.rename(columns={'click_article_id': 'article_id'}, inplace=True)
    df_click = df_click.merge(df_article, how='left')

    df_click['click_timestamp'] = df_click['click_timestamp'] / 1000
    df_click['click_datetime'] = pd.to_datetime(df_click['click_timestamp'],
                                                unit='s',
                                                errors='coerce')
    df_click['click_datetime_hour'] = df_click['click_datetime'].dt.hour

    # 用户点击文章的创建时间差的平均值
    df_click['user_id_click_article_created_at_ts_diff'] = df_click.groupby(
        ['user_id'])['created_at_ts'].diff()
    df_temp = df_click.groupby([
        'user_id'
    ])['user_id_click_article_created_at_ts_diff'].mean().reset_index()
    df_temp.columns = [
        'user_id', 'user_id_click_article_created_at_ts_diff_mean'
    ]
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 用户点击文章的时间差的平均值
    df_click['user_id_click_diff'] = df_click.groupby(
        ['user_id'])['click_timestamp'].diff()
    df_temp = df_click.groupby(['user_id'
                                ])['user_id_click_diff'].mean().reset_index()
    df_temp.columns = ['user_id', 'user_id_click_diff_mean']
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击文章的创建时间差的统计值
    df_click['click_timestamp_created_at_ts_diff'] = df_click[
        'click_timestamp'] - df_click['created_at_ts']

    df_temp = df_click.groupby(
        ['user_id'])['click_timestamp_created_at_ts_diff'].agg({
            'user_click_timestamp_created_at_ts_diff_mean':
            'mean',
            'user_click_timestamp_created_at_ts_diff_std':
            'std'
        }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 click_datetime_hour 统计值
    df_temp = df_click.groupby(['user_id'])['click_datetime_hour'].agg({
        'user_click_datetime_hour_std':
        'std'
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 words_count 统计值
    df_temp = df_click.groupby(['user_id'])['words_count'].agg({
        'user_clicked_article_words_count_mean':
        'mean',
        'user_click_last_article_words_count':
        lambda x: x.iloc[-1]
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 created_at_ts 统计值
    # 统计每个用户点击的文章的发布时间（created_at_ts）特征，例如：
    # 用户点击的最后一篇文章的发布时间；
    # 用户点击过的所有文章中最晚发布的一篇文章的发布时间。
    df_temp = df_click.groupby('user_id')['created_at_ts'].agg({
        'user_click_last_article_created_time':
        lambda x: x.iloc[-1],
        'user_clicked_article_created_time_max':
        'max',
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 click_timestamp 统计值
    df_temp = df_click.groupby('user_id')['click_timestamp'].agg({
        'user_click_last_article_click_time':
        lambda x: x.iloc[-1],
        'user_clicked_article_click_time_mean':
        'mean',
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    df_feature['user_last_click_created_at_ts_diff'] = df_feature[
        'created_at_ts'] - df_feature['user_click_last_article_created_time']
    df_feature['user_last_click_timestamp_diff'] = df_feature[
        'created_at_ts'] - df_feature['user_click_last_article_click_time']
    df_feature['user_last_click_words_count_diff'] = df_feature[
        'words_count'] - df_feature['user_click_last_article_words_count']

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 计数统计。计算不同分组组合下的记录数量
    for f in [['user_id'], ['article_id'], ['user_id', 'category_id']]:  # 分别是按 user_id 分组、按 article_id 分组以及按 user_id 和 category_id 组合分组
        df_temp = df_click.groupby(f).size().reset_index()
        df_temp.columns = f + ['{}_cnt'.format('_'.join(f))]  # 生成一个新的列名，用于表示该分组组合下的计数结果。例如，如果 f 是 ['user_id']，则新列名是 user_id_cnt

        df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 召回相关特征
    ## itemcf 相关
    user_item_ = df_click.groupby('user_id')['article_id'].agg(
        list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['article_id']))

    if mode == 'valid':
        f = open('../user_data/sim/offline/itemcf_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()
    else:
        f = open('../user_data/sim/online/itemcf_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()

    # 用户历史点击物品与待预测物品相似度
    df_feature['user_clicked_article_itemcf_sim_sum'] = df_feature[[
        'user_id', 'article_id'
    ]].parallel_apply(func_if_sum, axis=1)  # parallel_apply 是一种并行计算的方式.从 df_feature 数据框里选取 user_id 和 article_id 这两列，返回一个新的只包含这两列的 DataFrame。
    df_feature['user_last_click_article_itemcf_sim'] = df_feature[[
        'user_id', 'article_id'
    ]].parallel_apply(func_if_last, axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    ## binetwork 相关
    if mode == 'valid':
        f = open('../user_data/sim/offline/binetwork_sim.pkl', 'rb')
        binetwork_sim = pickle.load(f)
        f.close()
    else:
        f = open('../user_data/sim/online/binetwork_sim.pkl', 'rb')
        binetwork_sim = pickle.load(f)
        f.close()

    df_feature['user_last_click_article_binetwork_sim'] = df_feature[[
        'user_id', 'article_id'
    ]].parallel_apply(func_binetwork_sim_last, axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    ## w2v 相关
    if mode == 'valid':
        f = open('../user_data/data/offline/article_w2v.pkl', 'rb')
        article_vec_map = pickle.load(f)
        f.close()
    else:
        f = open('../user_data/data/online/article_w2v.pkl', 'rb')
        article_vec_map = pickle.load(f)
        f.close()

    df_feature['user_last_click_article_w2v_sim'] = df_feature[[
        'user_id', 'article_id'
    ]].parallel_apply(func_w2w_last_sim, axis=1)
    df_feature['user_click_article_w2w_sim_sum_2'] = df_feature[[
        'user_id', 'article_id'
    ]].parallel_apply(lambda x: func_w2w_sum(x, 2), axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 保存特征文件
    if mode == 'valid':
        df_feature.to_pickle('../user_data/data/offline/feature.pkl')

    else:
        df_feature.to_pickle('../user_data/data/online/feature.pkl')
