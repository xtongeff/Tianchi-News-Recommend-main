# import packages
import time, math, os
from tqdm import tqdm
import gc
import pickle
import random
from datetime import datetime
from operator import itemgetter
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
import collections
warnings.filterwarnings('ignore')

# 日期:2025-02-17 16:36:20。分数:0.1026
data_path = './download/'
save_path = './output/'

# 节约内存的一个标配函数
# 对输入的pandas数据框df进行内存优化，通过将数据框中数值类型的列转换为合适的较小数据类型，从而减少数据框占用的内存空间。
def reduce_mem(df):
    starttime = time.time()   # 记录开始时间
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']  # 定义数值类型列表
    start_mem = df.memory_usage().sum() / 1024**2  # 计算数据框初始内存使用量（单位：MB）
    for col in df.columns:  # 遍历数据框的每一列
        col_type = df[col].dtypes   # 获取当前列的数据类型
        if col_type in numerics:  # 如果当前列的数据类型属于数值类型
            c_min = df[col].min()  # 计算当前列的最小值
            c_max = df[col].max()  # 计算当前列的最大值
            if pd.isnull(c_min) or pd.isnull(c_max):  # 如果最小值或最大值为缺失值，则跳过当前列
                continue
            if str(col_type)[:3] == 'int':  # 如果当前列的数据类型是整数类型
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:   # 判断是否可以转换为np.int8类型.
# np.iinfo(dtype),用来获取整数类型的机器限制信息的。其中，dtype 是一个整数数据类型的 NumPy 对象，比如 np.int8、np.uint16 等。
# np.iinfo() 函数返回一个 iinfo 对象，该对象有以下几个常用属性：min：该整数类型能表示的最小值。max：该整数类型能表示的最大值。bits：该整数类型占用的位数。
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:  # 判断是否可以转换为np.int64类型
                    df[col] = df[col].astype(np.int64)
            else:  # 如果当前列的数据类型是浮点类型
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:  # 判断是否可以转换为np.float16类型
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:  # 判断是否可以转换为np.float32类型
                    df[col] = df[col].astype(np.float32)
                else:  # 否则，保持为np.float64类型
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2   # 计算数据框优化后的内存使用量（单位：MB）
    # 打印内存使用情况和时间花费信息
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                           100*(start_mem-end_mem)/start_mem,
                                                                                                           (time.time()-starttime)/60))
    return df


# debug模式：从训练集中划出一部分数据来调试代码
def get_all_click_sample(data_path, sample_nums=10000):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
        sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
    """
    all_click = pd.read_csv(data_path + 'train_click_log.csv')
    all_user_ids = all_click.user_id.unique()   # 获取all_click数据框中user_id列的所有唯一值，并存储在all_user_ids数组中。

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]   # 创建一个布尔索引，用于判断all_click数据框中每一行的user_id是否在sample_user_ids数组中。根据上述布尔索引筛选出all_click数据框中user_id在sample_user_ids数组中的所有行，更新all_click数据框。

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))  # 使用pandas库的drop_duplicates函数去除all_click数据框中user_id、click_article_id和click_timestamp三列组合重复的记录。
    return all_click


# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path='./data_raw/', offline=True):
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')

        all_click = trn_click.append(tst_click)

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click


# 全量训练集
all_click_df = get_all_click_df(data_path, offline=False)


# 获取 用户 - 文章 - 点击时间字典
# 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
def get_user_item_time(click_df):
    click_df = click_df.sort_values('click_timestamp')  # 使用sort_values方法对click_df数据框按照click_timestamp列进行升序排序

    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))  # 使用zip函数将df中的click_article_id列和click_timestamp列的元素一一对应组合成元组，然后将这些元组转换为列表并返回。

    user_item_time_df = click_df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(
        lambda x: make_item_time_pair(x)) \
        .reset_index().rename(columns={0: 'item_time_list'})  # groupby方法按照user_id对click_df数据框进行分组，只选择click_article_id和click_timestamp两列。
    # 对每个用户组应用make_item_time_pair函数，将每个用户点击的文章 ID 和对应的时间戳组合成一个列表。
    # 在 groupby 操作后，数据的索引变为多级索引（user_id 作为一级索引），.reset_index() 会把索引重置成普通的整数索引。
    # apply 返回的 Series 默认会被命名为 0（因为它只有一个列），所以在这里通过 .rename(columns={0: 'item_time_list'}) 把这个列的名称改为 item_time_list。
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    # 使用zip函数将user_item_time_df数据框中的user_id列和item_time_list列的元素一一对应组合成元组，然后将这些元组转换为字典，其中键为用户 ID，值为该用户点击的文章 ID 和时间戳的列表。

    return user_item_time_dict


# 获取近期点击最多的文章
def get_item_topk_click(click_df, k):
    # .value_counts()：对 click_article_id 列中的每个文章 ID 进行计数，统计每篇文章的点击次数。
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


# itemcf的物品相似度计算
def itemcf_sim(df):
    """
        文章与文章之间的相似性矩阵计算
        :param df: 数据表
        :item_created_time_dict:  文章创建时间的字典
        return : 文章与文章的相似性矩阵
        思路: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习)， 在多路召回部分会加上关联规则的召回策略
    """

    user_item_time_dict = get_user_item_time(df)  # 这个函数假设返回一个字典，其中每个 user 对应的值是一个列表，包含该用户点击的物品及时间的元组 (item_id, click_time)

    # 计算物品相似度
    i2i_sim = {}  # 用于存储物品间的相似度矩阵。它是一个字典，键是物品 ID，值是一个字典，表示与该物品相似的其他物品及其相似度。
    item_cnt = defaultdict(int)  # 用于记录每个物品被点击的次数。它是一个默认字典，键是物品 ID，值是点击次数。
    for user, item_time_list in tqdm(user_item_time_dict.items()): # tqdm 是一个用于在 Python 中添加进度条的库，它可以实时显示循环、迭代等操作的进度
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for i, i_click_time in item_time_list:
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})  # 确保物品 i 在相似度矩阵中有一个空字典来存储与其他物品的相似度。
            for j, j_click_time in item_time_list:
                if (i == j):
                    continue
                i2i_sim[i].setdefault(j, 0)

                i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)  # 这里使用了对数加权，假设每个用户的点击列表越长，物品之间的关联性越小，因此通过 len(item_time_list) 对其进行加权。

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])  # 对物品间的相似度进行归一化.这样可以消除物品被点击次数不同带来的偏差，确保每个物品的相似度计算不受其总点击次数的影响。

    # 将得到的相似性矩阵保存到本地
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))

    return i2i_sim_


i2i_sim = itemcf_sim(all_click_df)


# 基于商品的召回i2i
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
        :param i2i_sim: 字典，文章相似性矩阵
        :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        return: 召回的文章列表 {item1:score1, item2: score2...}
        注意: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习)， 在多路召回部分会加上关联规则的召回策略
    """

    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]  # 从 user_item_time_dict 中获取指定用户的历史点击文章列表。
    user_hist_items_ = {user_id for user_id, _ in user_hist_items}  # 将用户历史点击文章的 ID 存储在一个集合中，方便后续判断某篇文章是否已经被用户点击过。

    item_rank = {}  # 一个字典，用于存储召回文章及其得分。
    for loc, (i, click_time) in enumerate(user_hist_items):  # 外层循环遍历用户的历史点击文章。
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:  # 内层循环遍历与当前文章 i 相似度最高的前 sim_item_topk 篇文章 j，并获取它们之间的相似度得分 wij。
            if j in user_hist_items_:  # 如果文章 j 已经被用户点击过，则跳过该文章。
                continue

            item_rank.setdefault(j, 0)
            item_rank[j] += wij  # 确保 item_rank 中存在文章 j 的键，如果不存在则初始化为 0，然后将相似度得分 wij 累加到该文章的得分上。

    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items():  # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = - i - 100  # 随便给个负数就行。给补全的文章赋予一个负数得分，以区分基于相似度召回的文章。
            if len(item_rank) == recall_item_num:   # 当召回的文章数量达到 recall_item_num 时，停止补全。
                break

    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]  # 对 item_rank 字典按照得分从高到低进行排序，并选取前 recall_item_num 篇文章。

    return item_rank



# 给每个用户根据物品的协同过滤推荐文章
# 定义
user_recall_items_dict = collections.defaultdict(dict)

# 获取 用户 - 文章 - 点击时间的字典
user_item_time_dict = get_user_item_time(all_click_df)

# 去取文章相似度
i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))

# 相似文章的数量
sim_item_topk = 10

# 召回文章数量
recall_item_num = 10

# 用户热度补全
item_topk_click = get_item_topk_click(all_click_df, k=50)

for user in tqdm(all_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim,
                                                        sim_item_topk, recall_item_num, item_topk_click)



# 将字典的形式转换成df
user_item_score_list = []

for user, items in tqdm(user_recall_items_dict.items()):
    for item, score in items:
        user_item_score_list.append([user, item, score])

recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])


# 生成提交文件
def submit(recall_df, topk=5, model_name=None):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])  # 使用 sort_values 方法对 recall_df 按照 user_id 列和 pred_score 列进行排序。先按 user_id 排序，确保同一用户的记录相邻，再按 pred_score 排序。
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')  # 使用 groupby 方法按 user_id 对数据框进行分组。
# 对每个用户组内的 pred_score 列使用 rank 方法进行排名。ascending=False 表示按得分从高到低排名，method='first' 表示当得分相同时，按数据出现的先后顺序排名。排名结果存储在新列 rank 中。

    # 判断是不是每个用户都有5篇文章及以上
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())  # 对每个用户组应用一个匿名函数，该函数返回该用户组内 rank 列的最大值，即该用户的文章最大排名。
    assert tmp.min() >= topk

    del recall_df['pred_score']  # 删除 pred_score 列，因为它只用于排序，不参与提交。
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()
    # 筛选出 rank 列小于等于 topk 的行。
    # 使用 set_index 方法将 user_id 和 rank 列设置为索引。
    # 使用 unstack 方法将 rank 列的索引转换为列，实现数据的重塑。
    # 使用 reset_index 方法将索引还原为列。

    # 使用列表推导式处理列名。如果列名是整数类型，则保持不变；否则保持原列名。droplevel(0)用于移除多级索引的第一级。
    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    # 按照提交格式定义列名
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2',
                                    3: 'article_3', 4: 'article_4', 5: 'article_5'})

    save_name = save_path + model_name + '_' + datetime.today().strftime('%m-%d') + '.csv'
    submit.to_csv(save_name, index=False, header=True)



# 获取测试集
tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
tst_users = tst_click['user_id'].unique()

# 从所有的召回数据中将测试集中的用户选出来
tst_recall = recall_df[recall_df['user_id'].isin(tst_users)]  # isin() 是 Pandas 数据框列的一个方法，用于检查该列中的每个元素是否存在于指定的集合（这里是 tst_users 数组）中。

# 生成提交文件
submit(tst_recall, topk=5, model_name='itemcf_baseline')