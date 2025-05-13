import argparse
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from itertools import permutations
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='召回合并')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'召回合并: {mode}')


def mms(df):
    user_score_max = {}
    user_score_min = {}

    # 获取用户下的相似度的最大值和最小值
    for user_id, g in df[['user_id', 'sim_score']].groupby('user_id'):
        scores = g['sim_score'].values.tolist()
        user_score_max[user_id] = scores[0]
        user_score_min[user_id] = scores[-1]

    ans = []
    for user_id, sim_score in tqdm(df[['user_id', 'sim_score']].values):
        ans.append((sim_score - user_score_min[user_id]) /
                   (user_score_max[user_id] - user_score_min[user_id]) +
                   10**-3)
    return ans


def recall_result_sim(df1_, df2_):  # 其主要功能是计算两个数据框（df1_ 和 df2_）中，根据用户 ID 分组后的文章 ID 集合的召回率。
    # 召回率用于衡量在第一个数据框中推荐给用户的文章，有多少在第二个数据框中也出现了。
    df1 = df1_.copy()
    df2 = df2_.copy()

    user_item_ = df1.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict1 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    user_item_ = df2.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict2 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    cnt = 0
    hit_cnt = 0

    for user in user_item_dict1.keys():
        item_set1 = user_item_dict1[user]

        cnt += len(item_set1)

        if user in user_item_dict2:
            item_set2 = user_item_dict2[user]

            inters = item_set1 & item_set2
            hit_cnt += len(inters)

    return hit_cnt / cnt


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        recall_path = '../user_data/data/offline'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        recall_path = '../user_data/data/online'

    log.debug(f'max_threads {max_threads}')

    recall_methods = ['itemcf', 'w2v', 'binetwork']

    weights = {'itemcf': 1, 'binetwork': 1, 'w2v': 0.1}
    recall_list = []
    recall_dict = {}
    for recall_method in recall_methods:  # 召回结果处理。调用 mms 函数对相似度得分进行处理，接着乘以该召回方法的权重。
        recall_result = pd.read_pickle(
            f'{recall_path}/recall_{recall_method}.pkl')
        weight = weights[recall_method]

        recall_result['sim_score'] = mms(recall_result)
        recall_result['sim_score'] = recall_result['sim_score'] * weight

        recall_list.append(recall_result)
        recall_dict[recall_method] = recall_result

    # 求相似度
    for recall_method1, recall_method2 in permutations(recall_methods, 2):
        score = recall_result_sim(recall_dict[recall_method1],
                                  recall_dict[recall_method2])
        log.debug(f'召回相似度 {recall_method1}-{recall_method2}: {score}')

    # 合并召回结果。运用 permutations 函数生成召回方法的所有排列组合。
    recall_final = pd.concat(recall_list, sort=False)
    recall_score = recall_final[['user_id', 'article_id',
                                 'sim_score']].groupby([
                                     'user_id', 'article_id'
                                 ])['sim_score'].sum().reset_index()
    # 合并召回结果
    # 把所有召回结果合并成一个数据框 recall_final。
    # 按照 user_id 和 article_id 对相似度得分进行分组求和。
    # 去除 user_id 和 article_id 的重复项，再将相似度得分合并回 recall_final。
    # 按照 user_id 升序、sim_score 降序对结果进行排序，并记录日志。
    recall_final = recall_final[['user_id', 'article_id', 'label'
                                 ]].drop_duplicates(['user_id', 'article_id'])
    recall_final = recall_final.merge(recall_score, how='left')

    recall_final.sort_values(['user_id', 'sim_score'],
                             inplace=True,
                             ascending=[True, False])

    log.debug(f'recall_final.shape: {recall_final.shape}')
    log.debug(f'recall_final: {recall_final.head()}')

    # 删除无正样本的用户
    # 按照 user_id 对 recall_final 进行分组。
    # 遍历每个用户组，若标签列存在缺失值或者标签总和为 1，则将该用户组添加到 useful_recall 列表中。
    # 合并 useful_recall 列表中的数据框，再次按照 user_id 和 sim_score 排序并重置索引。
    gg = recall_final.groupby(['user_id'])
    useful_recall = []

    for user_id, g in tqdm(gg):
        if g['label'].isnull().sum() > 0:  # 处理标签存在缺失值的情况
            useful_recall.append(g)
        else:  # 处理标签无缺失值的情况
            label_sum = g['label'].sum()
            if label_sum > 1:  # 该用户组的正样本数量超过了预期（在很多业务场景下，每个用户通常只有一个正样本），这可能是数据异常，打印错误信息并记录用户 ID。
                print('error', user_id)
            elif label_sum == 1:
                useful_recall.append(g)

    df_useful_recall = pd.concat(useful_recall, sort=False)
    log.debug(f'df_useful_recall: {df_useful_recall.head()}')

    df_useful_recall = df_useful_recall.sort_values(
        ['user_id', 'sim_score'], ascending=[True,
                                             False]).reset_index(drop=True)

    # 计算相关指标
    if mode == 'valid':
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_useful_recall[df_useful_recall['label'].notnull()], total)
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50

        log.debug(
            f'召回合并后指标: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
    # 统计每个用户的召回数量，并计算平均召回数量，记录日志。
    # 统计非空标签的分布情况，记录日志。
    df = df_useful_recall['user_id'].value_counts().reset_index()
    df.columns = ['user_id', 'cnt']
    log.debug(f"平均每个用户召回数量：{df['cnt'].mean()}")

    log.debug(
        f"标签分布: {df_useful_recall[df_useful_recall['label'].notnull()]['label'].value_counts()}"
    )

    # 保存到本地
    if mode == 'valid':
        df_useful_recall.to_pickle('../user_data/data/offline/recall.pkl')
    else:
        df_useful_recall.to_pickle('../user_data/data/online/recall.pkl')
