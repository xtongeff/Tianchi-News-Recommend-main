import argparse
import math
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='w2v 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'w2v 召回，mode: {mode}')


def word2vec(df_, f1, f2, model_path):  # f1：作为主键的字段，例如 user_id。f2：作为序列特征的字段，例如 article_id。model_path：模型存储路径。
    df = df_.copy()  # 点击数据
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})  # 将 f2 字段的值按照 f1 进行分组，并转换成列表形式。

    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()  # 获取所有文章的用户列表（转换成 sentences 形式）。
    del tmp['{}_{}_list'.format(f1, f2)]  # 删除 tmp 里的 list 列，避免后续干扰，不影响 sentences。

    words = []
    for i in range(len(sentences)):
        x = [str(x) for x in sentences[i]]  # 转换为字符串
        sentences[i] = x  # 重新赋值
        words += x  # 记录所有出现过的词

    if os.path.exists(f'{model_path}/w2v.m'):
        model = Word2Vec.load(f'{model_path}/w2v.m')  # 若 w2v.m 存在，则直接加载。
    else:  # 否则，训练一个新的 Word2Vec 模型：
        model = Word2Vec(sentences=sentences,
                         size=256,
                         window=3,
                         min_count=1,
                         sg=1,
                         hs=0,
                         seed=seed,
                         negative=5,
                         workers=10,
                         iter=1)
        model.save(f'{model_path}/w2v.m')

    article_vec_map = {}
    for word in set(words):
        if word in model:
            article_vec_map[int(word)] = model[word]  # 将 word 对应的向量存入 article_vec_map。

    return article_vec_map


@multitasking.task
def recall(df_query, article_vec_map, article_index, user_item_dict,
           worker_id):  # 基于 Word2Vec 召回相似文章的多线程任务 recall()，用于为每个用户生成推荐文章列表，并将结果保存为 .pkl 文件。
    # df_query：包含用户的查询数据 (user_id, item_id)（用户点击文章的数据）。article_vec_map：文章 ID 到向量的映射 ({article_id: vector})。
    # article_index：Annoy（近似最近邻）索引，用于查找相似文章。user_item_dict：字典，记录每个用户交互过的文章 {user_id: [article_id1, article_id2, ...]}。
    # worker_id：线程 ID（用于区分不同任务的输出文件）。
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = defaultdict(int)  # 用于存储推荐文章的得分。

        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[-1:]  # 取该用户最近1 篇交互文章，作为查询文章（假设最近阅读的文章最能反映兴趣）。

        for item in interacted_items:
            article_vec = article_vec_map[item]  # 获取该文章的向量。

            item_ids, distances = article_index.get_nns_by_vector(
                article_vec, 100, include_distances=True)  # Annoy 检索：找到与 article_vec 最相似的 100 篇文章，返回：
            # item_ids：相似文章的 ID。distances：相似文章的距离（越小越相似）。
            sim_scores = [2 - distance for distance in distances]  # Annoy 计算的是距离，转换为相似度分数（2 - 距离，保证分数为正）。

            for relate_item, wij in zip(item_ids, sim_scores):
                if relate_item not in interacted_items:  # 若 relate_item 已经被用户交互过，则不推荐。
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij  # 将相似文章的分数累加到 rank。

        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:50]  # 选取前 50 篇相似文章
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        if item_id == -1:
            df_temp['label'] = np.nan  # 若用户 没有点击文章，label = np.nan（表示无监督）。
        else:
            df_temp['label'] = 0  # 默认不匹配。
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1  # 若推荐的 article_id 正好是用户点击过的 item_id

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs('../user_data/tmp/w2v', exist_ok=True)
    df_data.to_pickle('../user_data/tmp/w2v/{}.pkl'.format(worker_id))


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/data/offline', exist_ok=True)
        os.makedirs('../user_data/model/offline', exist_ok=True)

        w2v_file = '../user_data/data/offline/article_w2v.pkl'
        model_path = '../user_data/model/offline'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/data/online', exist_ok=True)
        os.makedirs('../user_data/model/online', exist_ok=True)

        w2v_file = '../user_data/data/online/article_w2v.pkl'
        model_path = '../user_data/model/online'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    article_vec_map = word2vec(df_click, 'user_id', 'click_article_id',
                               model_path)
    f = open(w2v_file, 'wb')
    pickle.dump(article_vec_map, f)
    f.close()

    # 将 embedding 建立索引
    article_index = AnnoyIndex(256, 'angular')  # 创建 Annoy 近似最近邻索引
    article_index.set_seed(2020)  # 使结果可复现

    for article_id, emb in tqdm(article_vec_map.items()):
        article_index.add_item(article_id, emb)  # 遍历 article_vec_map，将每篇文章的向量添加到索引。

    article_index.build(100)  # 构建 100 棵树（提高查找精度）。

    user_item_ = df_click.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()  # 构建用户交互记录
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('../tmp/w2v'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, article_vec_map, article_index, user_item_dict, i)

    multitasking.wait_for_tasks()  # 合并召回数据
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('../user_data/tmp/w2v'):
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
            f'w2v: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_w2v.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_w2v.pkl')
