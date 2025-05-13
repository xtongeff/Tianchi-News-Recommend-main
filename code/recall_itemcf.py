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
parser = argparse.ArgumentParser(description='itemcf 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'itemcf 召回，mode: {mode}')


def cal_sim(df):
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    item_cnt = defaultdict(int)
    sim_dict = {}

    for _, items in tqdm(user_item_dict.items()):
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_dict.setdefault(item, {})

            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue

                sim_dict[item].setdefault(relate_item, 0)

                # 位置信息权重
                # 考虑文章的正向顺序点击和反向顺序点击
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                loc_weight = loc_alpha * (0.9**(np.abs(loc2 - loc1) - 1))  # # 衡量序列中的相对位置影响，使得越接近的点击权重越大，提高推荐质量。0.9的n次方。

                sim_dict[item][relate_item] += loc_weight  / \
                    math.log(1 + len(items))

    for item, relate_items in tqdm(sim_dict.items()):
        for relate_item, cij in relate_items.items():
            sim_dict[item][relate_item] = cij / \
                math.sqrt(item_cnt[item] * item_cnt[relate_item])

    return sim_dict, user_item_dict


@multitasking.task
def recall(df_query, item_sim, user_item_dict, worker_id):
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = {}  # # user_id：用户 ID。item_id：测试集中的目标物品（若 -1，表示无目标物品）

        if user_id not in user_item_dict:  # 如果用户没有历史点击记录，跳过该用户。
            continue

        interacted_items = user_item_dict[user_id]   # 获取用户点击的文章（按时间倒序）。
        interacted_items = interacted_items[::-1][:2]  # 只取最近的 2 篇文章（用于计算相似度）。之前按照click_timestamp排序过了。

        for loc, item in enumerate(interacted_items):
            for relate_item, wij in sorted(item_sim[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:200]:  # 找到相似物品（最多 200 个）。
                if relate_item not in interacted_items:  # 如果该物品未被用户点击过，则加入推荐池：
                    rank.setdefault(relate_item, 0)  # 初始化 relate_item 的分数。
                    rank[relate_item] += wij * (0.7**loc)  # wij：物品 item 和 relate_item 的相似度。(0.7^loc)：位置权重，越新的点击文章，权重越大。loc=0（最近的一篇），权重 1.0loc=1（次新的），权重 0.7

        sim_items = sorted(rank.items(), key=lambda d: d[1],
                           reverse=True)[:100]
        item_ids = [item[0] for item in sim_items]  # 按相似度排序，选出 Top 100 物品。分别提取 article_id 和 sim_score。
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()  # 构造用户 - 文章推荐数据。
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        # 添加label作为监督学习数据
        if item_id == -1:
            df_temp['label'] = np.nan  # 该用户在测试集中无目标物品 → label=NaN（未标注）。
        else:
            df_temp['label'] = 0  # 默认 label=0（未点击）。
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1  # 如果 article_id 是目标物品，则 label=1（用户点击过）。注意区分item_id和item_ids.

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]  # 格式调整
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs('../user_data/tmp/itemcf', exist_ok=True)
    df_data.to_pickle(f'../user_data/tmp/itemcf/{worker_id}.pkl')


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/sim/offline', exist_ok=True)
        sim_pkl_file = '../user_data/sim/offline/itemcf_sim.pkl'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/sim/online', exist_ok=True)
        sim_pkl_file = '../user_data/sim/online/itemcf_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    item_sim, user_item_dict = cal_sim(df_click)  # item_sim：存储 物品之间的相似度（核心召回依据）。user_item_dict：存储 用户的点击行为（用于构建用户的兴趣画像）。
    f = open(sim_pkl_file, 'wb')
    pickle.dump(item_sim, f)
    f.close()

    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹.删除旧的召回结果，确保不会影响新的召回计算。
    for path, _, file_list in os.walk('../user_data/tmp/itemcf'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):  # 按用户划分子任务
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]  # df_temp 选取这些用户的查询数据。
        recall(df_temp, item_sim, user_item_dict, i)  # 计算推荐列表。重新生成了新的召回数据。

    multitasking.wait_for_tasks()  # 等待所有并行召回任务完成。
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('../user_data/tmp/itemcf'):
        for file_name in file_list:  # 读取所有召回数据并合并。
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')  # 按 user_id 升序、sim_score 降序排列，确保每个用户的最相关推荐项排在前面。

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()  # 计算有点击行为的唯一用户数量

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)  # 命中率（Hitrate）@k，表示用户点击的文章是否出现在推荐列表的前 k 个。mrr_k：平均倒数排名（Mean Reciprocal Rank），表示正确推荐的排名情况。

        log.debug(
            f'itemcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_itemcf.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_itemcf.pkl')
