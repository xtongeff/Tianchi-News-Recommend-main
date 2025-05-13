import argparse
import math
import os
import pickle
import random
import signal
from collections import defaultdict
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='binetwork 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'binetwork 召回，mode: {mode}')


# 基于用户的点击行为构建物品-物品相似度矩阵（Item-Item Similarity Matrix）。
# 它采用了协同过滤（Collaborative Filtering）的方式，即利用用户的历史交互行为计算物品之间的相关性。
def cal_sim(df):
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        list).reset_index()  # 计算用户-物品交互字典
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    item_user_ = df.groupby('click_article_id')['user_id'].agg(
        list).reset_index()  # 计算物品-用户倒排表
    item_user_dict = dict(
        zip(item_user_['click_article_id'], item_user_['user_id']))

    sim_dict = {}  # 计算物品相似度
    # 计算逻辑
    # 构造物品-用户倒排表，即哪些用户点击了哪些文章。
    # 遍历所有文章，找到点击过它的所有用户。
    # 遍历这些用户，获取他们的点击历史，计算文章之间的共现关系。
    # 使用对数惩罚项，削弱流行文章和超级活跃用户的影响。
    # 最终得到物品-物品相似度字典。

    for item, users in tqdm(item_user_dict.items()):  # item 是当前正在计算相似度的文章。users 是点击过 item 的用户列表。
        sim_dict.setdefault(item, {})

        for user in users:
            tmp_len = len(user_item_dict[user])  # # 该用户点击的文章总数
            for relate_item in user_item_dict[user]:  # # 遍历该用户点击的所有文章
                sim_dict[item].setdefault(relate_item, 0)
                sim_dict[item][relate_item] += 1 / \
                    (math.log(len(users)+1) * math.log(tmp_len+1))

    return sim_dict, user_item_dict  # sim_dict：物品相似度字典，格式：
    # { article1: {article2: similarity, article3: similarity, ...},
    #   article2: {article1: similarity, article4: similarity, ...}, ... }


@multitasking.task
def recall(df_query, item_sim, user_item_dict, worker_id):
    # df_query：查询数据集，包含用户ID 和 实际点击的文章ID。
    # item_sim：物品相似度字典，存储文章之间的相似度。
    # user_item_dict：用户-文章交互字典，存储用户点击的文章历史。
    # worker_id：当前任务的唯一标识，用于区分不同进程的输出文件。
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = {}

        if user_id not in user_item_dict:  # 过滤无历史数据的用户
            continue

        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[::-1][:1]  #  逆序：最近的点击行为更重要，所以取最近的一篇文章 [:1]。

        for _, item in enumerate(interacted_items):
            for relate_item, wij in sorted(item_sim[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:100]:  # 找到 item 相关的所有物品及其相似度 wij。按相似度 wij 降序排序，取前 100 个最相似的文章。
                if relate_item not in interacted_items:  # 过滤掉用户已经看过的文章，避免推荐重复文章。
                    rank.setdefault(relate_item, 0)  # 如果 relate_item 没有出现在 rank，就初始化为 0。
                    rank[relate_item] += wij  # 累加 wij，即多个相似文章都推荐 relate_item，它的权重会更高。

        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:50]  # 取前 50 个最相关物品
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1  # 用户实际点击的文章，设 label = 1。

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs('../user_data/tmp/binetwork', exist_ok=True)
    df_data.to_pickle(f'../user_data/tmp/binetwork/{worker_id}.pkl')


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/sim/offline', exist_ok=True)
        sim_pkl_file = '../user_data/sim/offline/binetwork_sim.pkl'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/sim/online', exist_ok=True)
        sim_pkl_file = '../user_data/sim/online/binetwork_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    item_sim, user_item_dict = cal_sim(df_click)
    f = open(sim_pkl_file, 'wb')
    pickle.dump(item_sim, f)
    f.close()

    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('../user_data/tmp/binetwork'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, item_sim, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('../user_data/tmp/binetwork'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'binetwork: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_binetwork.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_binetwork.pkl')
