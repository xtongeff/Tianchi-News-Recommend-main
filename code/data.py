import argparse
import os
import random
from random import sample

import pandas as pd
from tqdm import tqdm

from utils import Logger

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='数据处理')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'数据处理，mode: {mode}')

# 线下模式 (offline)
# 📌 目标
# 构造验证集，用于线下评估推荐效果。
# 需要从训练集中抽取部分用户的部分数据作为验证集，模拟真实推荐场景。
# 📌 处理方式
# 随机选择 50,000 个用户 作为验证集用户 (val_users)。
# 将这些用户的最后一次点击行为分离出来，作为 df_valid_query：
# 这部分数据相当于用户的最后一次查询，用于评估推荐系统能否准确预测他们会点击的文章。
# 其余的点击行为仍然留在训练集中 (df_train_click)，用于训练推荐模型。
# 测试集 df_test_query 仍然是所有 test_users，click_article_id = -1。
# 最终df_query包括df_valid_query和df_test_query，只有df_test_query 的click_article_id = -1。
# 训练数据 df_click 直接包含 df_train_click和df_test_click，按user_id和click_timestamp排序。
# 📌 适用场景
# 用于离线模型评估，通过 df_valid_query 计算推荐准确率。
# 可用于调试模型，检查推荐效果。

#  线上模式 (online)
# 📌 目标
# 构造测试集，用于线上预测推荐结果。
# 不做数据拆分，直接将 train_click_log 和 testB_click_log 结合，确保模型在完整数据上训练并进行预测。
# 📌 处理方式
# 不抽取验证集，所有 train_click_log 数据都保留。
# 测试集 df_test_query 仍然是所有 test_users，click_article_id = -1。
# 训练数据 df_click 直接包含 train_click_log + testB_click_log，不删除任何行为记录，按user_id和click_timestamp排序。。
# 📌 适用场景
# 用于线上预测推荐结果，即模型最终的提交结果。
# 确保最大程度利用数据，而不是保留部分数据用于验证。
def data_offline(df_train_click, df_test_click):
    train_users = df_train_click['user_id'].values.tolist()
    # 随机采样出一部分样本
    val_users = sample(train_users, 50000)
    log.debug(f'val_users num: {len(set(val_users))}')

    # 训练集用户 抽出行为数据最后一条作为线下验证集
    click_list = []
    valid_query_list = []

    groups = df_train_click.groupby(['user_id'])
    for user_id, g in tqdm(groups):
        if user_id in val_users:
            valid_query = g.tail(1)
            valid_query_list.append(
                valid_query[['user_id', 'click_article_id']])

            train_click = g.head(g.shape[0] - 1)
            click_list.append(train_click)
        else:
            click_list.append(g)

    df_train_click = pd.concat(click_list, sort=False)
    df_valid_query = pd.concat(valid_query_list, sort=False)

    test_users = df_test_click['user_id'].unique()
    test_query_list = []

    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])

    df_query = pd.concat([df_valid_query, df_test_query],
                         sort=False).reset_index(drop=True)
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    os.makedirs('../user_data/data/offline', exist_ok=True)

    df_click.to_pickle('../user_data/data/offline/click.pkl')
    df_query.to_pickle('../user_data/data/offline/query.pkl')


def data_online(df_train_click, df_test_click):
    test_users = df_test_click['user_id'].unique()
    test_query_list = []

    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])

    df_query = df_test_query
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    os.makedirs('../user_data/data/online', exist_ok=True)

    df_click.to_pickle('../user_data/data/online/click.pkl')
    df_query.to_pickle('../user_data/data/online/query.pkl')


if __name__ == '__main__':
    df_train_click = pd.read_csv('../tcdata/train_click_log.csv')
    df_test_click = pd.read_csv('../tcdata/testB_click_log.csv')

    log.debug(
        f'df_train_click shape: {df_train_click.shape}, df_test_click shape: {df_test_click.shape}'
    )

    if mode == 'valid':
        data_offline(df_train_click, df_test_click)
    else:
        data_online(df_train_click, df_test_click)
