import pandas as pd
import numpy as np

from collections import defaultdict
import os, math, warnings, math, pickle
from tqdm import tqdm
import faiss
import collections
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from keras_preprocessing.sequence import pad_sequences

from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss
from deepmatch.utils import sampledsoftmaxloss, NegativeSampler
import tensorflow as tf

from collections import Counter
warnings.filterwarnings('ignore')


data_path = './download/'
save_path = './output/'
# 做召回评估的一个标志, 如果不进行评估就是直接使用全量数据进行召回
metric_recall = True

# 在一般的推荐系统比赛中读取数据部分主要分为三种模式， 不同的模式对应的不同的数据集：
# Debug模式： 这个的目的是帮助我们基于数据先搭建一个简易的baseline并跑通。
# 线下验证模式： 这个的目的是帮助我们在线下基于已有的训练集数据， 来选择好合适的模型和一些超参数。
# 线上模式：使用的训练数据集是全量的数据集。

# 下面就分别对这三种不同的数据读取模式先建立不同的代导入函数， 方便后面针对不同的模式下导入数据。
# debug模式： 从训练集中划出一部分数据来调试代码
def get_all_click_sample(data_path, sample_nums=10000):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
        sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
    """
    all_click = pd.read_csv(data_path + 'train_click_log.csv')
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click


# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path='./download/', offline=True):
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')

        all_click = trn_click.append(tst_click)

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click


# 读取文章的基本属性
def get_item_info_df(data_path):
    item_info_df = pd.read_csv(data_path + 'articles.csv')

    # 为了方便与训练集中的click_article_id拼接，需要把article_id修改成click_article_id
    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})

    return item_info_df


# 读取文章的Embedding数据
def get_item_emb_dict(data_path):
    item_emb_df = pd.read_csv(data_path + 'articles_emb.csv')

    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]  # 从 DataFrame 的列名中筛选出包含 'emb' 字符串的列名。item_emb_cols 是一个列表，包含了所有与文章嵌入相关的列名。
    # 将嵌入数据转换为NumPy数组。通过 item_emb_df[item_emb_cols] 从 DataFrame 中提取出包含嵌入数据的列，并将其转换为一个 NumPy 数组 item_emb_np。
    # np.ascontiguousarray() 可以确保返回的 NumPy 数组是内存连续的，这通常有助于提升计算效率。
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])
    # 进行归一化
    # 这行代码对嵌入数据进行 归一化，即将每个嵌入向量的范数（模长）调整为 1。
    # np.linalg.norm(item_emb_np, axis=1, keepdims=True) 计算每个向量的 L2 范数（即向量的模），axis=1 表示沿着行方向计算范数（对每个嵌入向量），keepdims=True 保持结果的维度与原数组一致。
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    # 保存 item_emb_dict
    # 将 item_emb_df['article_id']（文章ID）和 item_emb_np（归一化后的嵌入向量）配对，创建一个字典 item_emb_dict，其中每个文章ID都映射到对应的嵌入向量。
    item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))
    # pickle 库将 item_emb_dict 字典保存为一个 .pkl 文件。save_path + 'item_content_emb.pkl' 是保存文件的路径（需要定义 save_path）。'wb' 表示以二进制写模式打开文件。
    pickle.dump(item_emb_dict, open(save_path + 'item_content_emb.pkl', 'wb'))

    return item_emb_dict

# 对输入的 NumPy 数组进行 最大最小归一化
max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))

# 采样数据
# all_click_df = get_all_click_sample(data_path)

# 全量训练集
all_click_df = get_all_click_df(offline=False)

# 对时间戳进行归一化,用于在关联规则的时候计算权重
all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)

item_info_df = get_item_info_df(data_path)

# item_emb_dict = get_item_emb_dict(data_path)


# 工具函数，获取用户-文章-时间函数。这个在基于关联规则的用户协同过滤的时候会用到。
# 根据点击时间获取用户的点击文章序列   {user1: {item1: time1, item2: time2..}...}
def get_user_item_time(click_df):
    click_df = click_df.sort_values('click_timestamp')

    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    user_item_time_df = click_df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(
        lambda x: make_item_time_pair(x)) \
        .reset_index().rename(columns={0: 'item_time_list'})  # 此时，生成的列的名称为 0（这是因为 apply 返回的是一个包含列表的列）。将这列的名称改为 item_time_list。
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))

    return user_item_time_dict

# 获取文章-用户-时间函数。这个在基于关联规则的文章协同过滤的时候会用到。
# 根据时间获取商品被点击的用户序列  {item1: {user1: time1, user2: time2...}...}
# 这里的时间是用户点击当前商品的时间，好像没有直接的关系。
def get_item_user_time_dict(click_df):
    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['click_timestamp']))

    click_df = click_df.sort_values('click_timestamp')
    item_user_time_df = click_df.groupby('click_article_id')['user_id', 'click_timestamp'].apply(
        lambda x: make_user_time_pair(x)) \
        .reset_index().rename(columns={0: 'user_time_list'})

    item_user_time_dict = dict(zip(item_user_time_df['click_article_id'], item_user_time_df['user_time_list']))
    return item_user_time_dict

# 获取历史和最后一次点击
# 这个在评估召回结果， 特征工程和制作标签转成监督学习测试集的时候回用到
# 获取当前数据的历史点击和最后一次点击
def get_hist_and_last_click(all_click):
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)   # groupby('user_id') 将数据按照 user_id 进行分组，然后 tail(1) 提取每个用户最后一次点击的记录。

    # 如果用户只有一个点击，hist为空了，会导致训练的时候这个用户不可见，此时默认泄露一下
    def hist_func(user_df):  # 该函数用于从每个用户的点击记录中提取历史点击（不包括最后一次点击）。
        if len(user_df) == 1:  # 这行代码判断该用户是否只有一次点击。如果用户只有一次点击，那么历史数据为空，直接返回该用户的点击记录。这个情况是为了防止在训练模型时，某些只有一次点击的用户数据无法作为训练样本的问题。
            return user_df
        else:  # 如果该用户有多次点击，则返回该用户除了最后一次点击外的所有点击数据
            return user_df[:-1]

    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)

    return click_hist_df, click_last_df


# 获取文章属性特征
# 获取文章id对应的基本属性，保存成字典的形式，方便后面召回阶段，冷启动阶段直接使用
def get_item_info_dict(item_info_df):
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler)

    item_type_dict = dict(zip(item_info_df['click_article_id'], item_info_df['category_id']))
    item_words_dict = dict(zip(item_info_df['click_article_id'], item_info_df['words_count']))
    item_created_time_dict = dict(zip(item_info_df['click_article_id'], item_info_df['created_at_ts']))

    return item_type_dict, item_words_dict, item_created_time_dict


# 获取用户历史点击的文章信息
def get_user_hist_item_info_dict(all_click):
    # 获取user_id对应的用户历史点击文章类型的集合字典
    # 对于每个用户，应用 set 聚合函数，将用户所有点击的 category_id 聚合成一个集合（集合会去重，所以每个用户只会包含其点击过的独特类别）。这是为了得到每个用户点击过的所有商品类别。
    user_hist_item_typs = all_click.groupby('user_id')['category_id'].agg(set).reset_index()
    user_hist_item_typs_dict = dict(zip(user_hist_item_typs['user_id'], user_hist_item_typs['category_id']))  # 将 user_id 和 category_id 两列配对成一个个元组

    # 获取user_id对应的用户点击文章的集合
    user_hist_item_ids_dict = all_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
    user_hist_item_ids_dict = dict(zip(user_hist_item_ids_dict['user_id'], user_hist_item_ids_dict['click_article_id']))

    # 获取user_id对应的用户历史点击的文章的平均字数字典
    user_hist_item_words = all_click.groupby('user_id')['words_count'].agg('mean').reset_index()
    user_hist_item_words_dict = dict(zip(user_hist_item_words['user_id'], user_hist_item_words['words_count']))

    # 获取user_id对应的用户最后一次点击的文章的创建时间
    # 对每个用户的点击记录应用 lambda 函数，x.iloc[-1] 选择每个用户分组中的最后一条记录的 created_at_ts。
    all_click_ = all_click.sort_values('click_timestamp')
    user_last_item_created_time = all_click_.groupby('user_id')['created_at_ts'].apply(
        lambda x: x.iloc[-1]).reset_index()

    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    user_last_item_created_time['created_at_ts'] = user_last_item_created_time[['created_at_ts']].apply(max_min_scaler)

    user_last_item_created_time_dict = dict(zip(user_last_item_created_time['user_id'], \
                                                user_last_item_created_time['created_at_ts']))

    return user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict


# 获取点击次数最多的Top-k个文章
# 获取近期点击最多的文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


# 定义多路召回字典
# 获取文章的属性信息，保存成字典的形式方便查询
item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df)

# 定义一个多路召回的字典，将各路召回的结果都保存在这个字典当中
user_multi_recall_dict =  {'itemcf_sim_itemcf_recall': {},
                           'embedding_sim_item_recall': {},
                           'youtubednn_recall': {},
                           'youtubednn_usercf_recall': {},
                           'cold_start_recall': {}}

# 提取最后一次点击作为召回评估，如果不需要做召回评估直接使用全量的训练集进行召回(线下验证模型)
# 如果不是召回评估，直接使用全量数据进行召回，不用将最后一次提取出来
trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)

# 召回效果评估
# 做完了召回有时候也需要对当前的召回方法或者参数进行调整以达到更好的召回效果，因为召回的结果决定了最终排序的上限，下面也会提供一个召回评估的方法
# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
# 这段代码的目的是计算 Top-K 召回的命中率（hit rate），用于评估推荐系统的性能。
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)  # 这行代码计算 user_recall_items_dict 的长度，即用户数。user_recall_items_dict 是一个字典，键是 user_id，值是推荐系统为该用户召回的商品列表。

    for k in range(10, topk + 1, 10):  # 从 10 开始，每次增加 10，直到达到 topk。
        hit_num = 0  #  hit_num 记录命中的次数。每当推荐系统的前 K 个商品中出现用户最后一次点击的商品时，hit_num 会增加。
        for user, item_list in user_recall_items_dict.items():
            # 获取前k个召回的结果
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]  # 提取出前 k 个推荐商品的 ID
            if last_click_item_dict[user] in set(tmp_recall_items):  # last_click_item_dict[user] 获取该用户的最后一次点击商品 ID
                hit_num += 1

        hit_rate = round(hit_num * 1.0 / user_num, 5)  # 命中次数与用户总数的比值，保留五位小数。
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)


# 计算相似性矩阵
# 这一部分主要是通过协同过滤以及向量检索得到相似性矩阵，相似性矩阵主要分为user2user和item2item，下面依次获取基于itemCF的item2item的相似性矩阵。
# itemCF i2i_sim
# 借鉴KDD2020的去偏商品推荐，在计算item2item相似性矩阵时，使用关联规则，使得计算的文章的相似性还考虑到了:
# 用户点击的时间权重
# 用户点击的顺序权重
# 文章创建的时间权重
def itemcf_sim(df, item_created_time_dict):
    """
        文章与文章之间的相似性矩阵计算
        :param df: 数据表
        :item_created_time_dict:  文章创建时间的字典
        return : 文章与文章的相似性矩阵

        思路: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习) + 关联规则
    """

    user_item_time_dict = get_user_item_time(df)

    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for loc1, (i, i_click_time) in enumerate(item_time_list):  # loc1 是文章 i 在该用户点击序列中的索引位置（从 0 开始）。
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if (i == j):
                    continue

                # 考虑文章的正向顺序点击和反向顺序点击
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                # 位置信息权重，其中的参数可以调节
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                # 点击时间权重，其中的参数可以调节
                click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))
                # 两篇文章创建时间的权重，其中的参数可以调节
                created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                i2i_sim[i].setdefault(j, 0)
                # 考虑多种因素的权重计算最终的文章之间的相似度
                i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(
                    len(item_time_list) + 1)

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():  #  i 是文章 ID，related_items 是一个字典，存储了与 i 相关的文章及其相似度。
        for j, wij in related_items.items():  # j 是与文章 i 相关的文章 ID。wij 是文章 i 和文章 j 的初始相似度（未归一化）。
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

    # 将得到的相似性矩阵保存到本地
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))

    return i2i_sim_

# i2i_sim = itemcf_sim(all_click_df, item_created_time_dict)



# userCF u2u_sim
# 在计算用户之间的相似度的时候，也可以使用一些简单的关联规则，比如用户活跃度权重，这里将用户的点击次数作为用户活跃度的指标
def get_user_activate_degree_dict(all_click_df):
    all_click_df_ = all_click_df.groupby('user_id')['click_article_id'].count().reset_index()

    # 用户活跃度归一化
    mm = MinMaxScaler()
    # mm.fit_transform 对all_click_df_['click_article_id']列的数据进行归一化操作，将每个用户的点击次数映射到[0, 1]范围内。
    all_click_df_['click_article_id'] = mm.fit_transform(all_click_df_[['click_article_id']])
    user_activate_degree_dict = dict(zip(all_click_df_['user_id'], all_click_df_['click_article_id']))

    return user_activate_degree_dict


def usercf_sim(all_click_df, user_activate_degree_dict):
    """
        用户相似性矩阵计算
        :param all_click_df: 数据表
        :param user_activate_degree_dict: 用户活跃度的字典
        return 用户相似性矩阵

        思路: 基于用户的协同过滤(详细请参考上一期推荐系统基础的组队学习) + 关联规则
    """
    item_user_time_dict = get_item_user_time_dict(all_click_df)

    u2u_sim = {}
    user_cnt = defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for u, click_time in user_time_list:
            user_cnt[u] += 1
            u2u_sim.setdefault(u, {})
            for v, click_time in user_time_list:
                u2u_sim[u].setdefault(v, 0)
                if u == v:
                    continue
                # 用户平均活跃度作为活跃度的权重，这里的式子也可以改善
                activate_weight = 100 * 0.5 * (user_activate_degree_dict[u] + user_activate_degree_dict[v])
                u2u_sim[u][v] += activate_weight / math.log(len(user_time_list) + 1)

    u2u_sim_ = u2u_sim.copy()
    for u, related_users in u2u_sim.items():
        for v, wij in related_users.items():
            u2u_sim_[u][v] = wij / math.sqrt(user_cnt[u] * user_cnt[v])

    # 将得到的相似性矩阵保存到本地
    pickle.dump(u2u_sim_, open(save_path + 'usercf_u2u_sim.pkl', 'wb'))

    return u2u_sim_

# 由于usercf计算时候太耗费内存了，这里就不直接运行了
# 如果是采样的话，是可以运行的
# user_activate_degree_dict = get_user_activate_degree_dict(all_click_df)
# u2u_sim = usercf_sim(all_click_df, user_activate_degree_dict)


# item embedding sim
# 使用Embedding计算item之间的相似度是为了后续冷启动的时候可以获取未出现在点击数据中的文章。faiss就是用来加速计算某个查询向量最相似的topk个索引向量。
# faiss使用了PCA（主成分分析）和PQ(Product quantization乘积量化)两种技术进行向量压缩和编码
# 向量检索相似度计算
# topk指的是每个item, faiss搜索后返回最相似的topk个item
def embdding_sim(click_df, item_emb_df, save_path, topk):
    """
        基于内容的文章embedding相似性矩阵计算
        :param click_df: 数据表
        :param item_emb_df: 文章的embedding
        :param save_path: 保存路径
        :patam topk: 找最相似的topk篇
        return 文章相似性矩阵

        思路: 对于每一篇文章， 基于embedding的相似性返回topk个与其最相似的文章， 只不过由于文章数量太多，这里用了faiss进行加速
    """

    # 文章索引与文章id的字典映射
    item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df['article_id']))

    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
    # 向量进行单位化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    # 建立faiss索引。
    # 这里使用了Faiss库来加速相似性计算。IndexFlatIP表示使用内积（dot product）来计算相似度，并创建Faiss索引。add()方法将所有文章的embedding向量添加到索引中。
    item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
    item_index.add(item_emb_np)
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度。使用search()方法对所有文章进行相似性查询。它会返回每篇文章的相似度值和对应的文章索引。
    sim, idx = item_index.search(item_emb_np, topk)  # 返回的是列表

    # 将向量检索的结果保存成原始id的对应关系
    item_sim_dict = collections.defaultdict(dict)  # 这里的item_sim_dict是一个嵌套字典，用来存储每篇文章与其他文章的相似度。
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np)), sim, idx)):
        target_raw_id = item_idx_2_rawid_dict[target_idx]  # target_raw_id是当前文章的ID
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = item_idx_2_rawid_dict[rele_idx]  # rele_raw_id是与当前文章相似的文章ID。sim_value是相似度值（内积）。
            item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id,
                                                                                                 0) + sim_value
            # 函数返回计算出的item_sim_dict字典，其中存储了每篇文章与其他文章的相似度，键是文章ID，值是一个字典，表示与该文章相似的文章及其相似度值。

    # 保存i2i相似度矩阵
    pickle.dump(item_sim_dict, open(save_path + 'emb_i2i_sim.pkl', 'wb'))

    return item_sim_dict


item_emb_df = pd.read_csv(data_path + '/articles_emb.csv')
# emb_i2i_sim = embdding_sim(all_click_df, item_emb_df, save_path, topk=10) # topk可以自行设置


# 上面的各种召回方式一部分在基于用户已经看得文章的基础上去召回与这些文章相似的一些文章。还有一部分是根据用户的相似性进行推荐，对于某用户推荐与其相似的其他用户看过的文章。
# 还有一种思路是类似矩阵分解的思路，先计算出用户和文章的embedding之后，就可以直接算用户和文章的相似度， 根据这个相似度进行推荐， 比如YouTube DNN。
# 获取双塔召回时的训练验证数据
# negsample指的是通过滑窗构建样本的时候，负样本的数量
def gen_data_set(data, negsample=0):
    data.sort_values("click_timestamp", inplace=True)
    item_ids = data['click_article_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):  # 按 user_id 进行分组，每个用户对应一个 hist（该用户的点击历史）。tqdm(...) 用于显示循环进度条，提升可视化体验。
        pos_list = hist['click_article_id'].tolist()  # pos_list: 该用户点击的文章 ID 列表，按时间顺序排列。

        if negsample > 0:  # （即需要负采样）
            candidate_set = list(set(item_ids) - set(pos_list))  # 用户没看过的文章里面选择负样本
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)  # 对于每个正样本，选择n个负样本

        # 长度只有一个的时候，这条数据需要同时加入训练集和测试集，不然的话最终学到的embedding就会有缺失。
        # [pos_list[0]]：用户的历史点击（仅一个）
        # pos_list[0]：目标文章 ID
        # 1：正样本标签
        # len(pos_list)：历史点击长度
        if len(pos_list) == 1:
            train_set.append((reviewerID, [pos_list[0]], pos_list[0], 1, len(pos_list)))
            test_set.append((reviewerID, [pos_list[0]], pos_list[0], 1, len(pos_list)))

        # 滑窗构造正负样本
        for i in range(1, len(pos_list)):  # 从 pos_list 第二个元素开始，遍历用户点击的文章列表（滑窗）。
            hist = pos_list[:i]  # hist：当前滑窗的历史点击序列。

            if i != len(pos_list) - 1:  # 如果 i 不是最后一个点击记录，则加入训练集
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1,
                                  len(hist[::-1])))  # 正样本 [user_id, his_item, pos_item, label, len(his_item)]
                for negi in range(negsample):  # 从负样本列表中取 negsample 个负样本
                    train_set.append((reviewerID, hist[::-1], neg_list[i * negsample + negi], 0,
                                      len(hist[::-1])))  # 负样本 [user_id, his_item, neg_item, label, len(his_item)]
            else:  # 如果 i 是最后一个点击记录，则加入 test_set（只加入正样本）。
                test_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1])))

    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set


# 将输入的数据进行padding，使得序列特征的长度都一致。
def gen_model_input(train_set, user_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])  # 提取 user_id，转换为 NumPy 数组。
    train_seq = [line[1] for line in train_set]  # 提取用户的历史点击序列。
    train_iid = np.array([line[2] for line in train_set])  # 提取目标文章 ID（正/负样本）。
    train_label = np.array([line[3] for line in train_set])  # 提取样本标签（1：正样本，0：负样本）。
    train_hist_len = np.array([line[4] for line in train_set])  # 提取历史点击序列的长度。

    # pad_sequences 作用：
    # maxlen=seq_max_len：将所有历史点击序列填充到 seq_max_len 长度。
    # padding='post'：在序列末尾填充 0（确保所有输入的形状一致）。
    # truncating='post'：如果序列超过 seq_max_len，则从末尾截断。
    # value=0：填充值设为 0（表示无点击）。
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "click_article_id": train_iid, "hist_article_id": train_seq_pad,
                         "hist_len": train_hist_len}

    return train_model_input, train_label


def youtubednn_u2i_dict(data, topk=20):
    sparse_features = ["click_article_id", "user_id"]
    SEQ_LEN = 30  # 用户点击序列的长度，短的填充，长的截断

    user_profile_ = data[["user_id"]].drop_duplicates('user_id')  # 提取 user_id 和 click_article_id 的唯一值，得到用户和物品的基本信息。
    item_profile_ = data[["click_article_id"]].drop_duplicates('click_article_id')

    # 类别编码
    features = ["click_article_id", "user_id"]
    feature_max_idx = {}  # 记录每个特征的最大索引值，用于定义 Embedding 层的输入维度。

    for feature in features:
        lbe = LabelEncoder()  # 使用 LabelEncoder 进行 ID 编码，将 user_id 和 click_article_id 转换为从 0 开始的整数索引，以便模型处理。
        data[feature] = lbe.fit_transform(data[feature])
        feature_max_idx[feature] = data[feature].max() + 1

    # 提取user和item的画像，这里具体选择哪些特征还需要进一步的分析和考虑
    user_profile = data[["user_id"]].drop_duplicates('user_id')  # 重新提取 user_id 和 click_article_id 的唯一值（编码后的）。
    item_profile = data[["click_article_id"]].drop_duplicates('click_article_id')

    user_index_2_rawid = dict(zip(user_profile['user_id'], user_profile_['user_id']))  # 建立索引到原始 ID 的映射。user_id 的索引到原始 ID 的映射。
    item_index_2_rawid = dict(zip(item_profile['click_article_id'], item_profile_['click_article_id']))  # click_article_id 的索引到原始 ID 的映射。

    # 划分训练和测试集
    # 由于深度学习需要的数据量通常都是非常大的，所以为了保证召回的效果，往往会通过滑窗的形式扩充训练样本
    train_set, test_set = gen_data_set(data, 0)   # gen_data_set：基于用户点击数据 构造训练集和测试集（滑窗方法）。
    # 整理输入数据，具体的操作可以看上面的函数
    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)   # gen_model_input：填充/截断用户点击序列，并生成模型输入。
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    # 确定Embedding的维度
    embedding_dim = 16

    # 将数据整理成模型可以直接输入的形式
    # SparseFeat：稀疏特征（用户 ID 和文章 ID）。
    # VarLenSparseFeat：变长序列特征（用户历史点击文章）。
    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                            VarLenSparseFeat(
                                SparseFeat('hist_article_id', feature_max_idx['click_article_id'], embedding_dim,
                                           embedding_name="click_article_id"), SEQ_LEN, 'mean', 'hist_len'), ]
    item_feature_columns = [SparseFeat('click_article_id', feature_max_idx['click_article_id'], embedding_dim)]

    # 模型的定义
    # num_sampled: 负采样时的样本数量
    # YoutubeDNN：YouTube DNN 模型（基于 DeepMatch 库）。
    # num_sampled=5：负采样 5 个负样本。
    # user_dnn_hidden_units=(64, embedding_dim)：用户侧 DNN 结构。
    train_counter = Counter(train_model_input['click_article_id'])
    item_count = [train_counter.get(i, 0) for i in range(item_feature_columns[0].vocabulary_size)]
    sampler_config = NegativeSampler('frequency', num_sampled=5, item_name="click_article_id", item_count=item_count)


    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    else:
        K.set_learning_phase(True)

    model = YoutubeDNN(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, 16, embedding_dim),
                       sampler_config=sampler_config)

    # model = YoutubeDNN(user_feature_columns, item_feature_columns, num_neg_samples=5,
    #                    user_dnn_hidden_units=(64, embedding_dim))
    # 模型编译
    model.compile(optimizer="adam", loss=sampledsoftmaxloss)

    # 模型训练，这里可以定义验证集的比例，如果设置为0的话就是全量数据直接进行训练.validation_split=0.0：不划分验证集。
    history = model.fit(train_model_input, train_label, batch_size=256, epochs=1, verbose=1, validation_split=0.0)

    # 训练完模型之后,提取训练的Embedding，包括user端和item端
    test_user_model_input = test_model_input
    all_item_model_input = {"click_article_id": item_profile['click_article_id'].values}

    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)  # 用户 Embedding 提取模型。
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)  # 物品 Embedding 提取模型。

    # 保存当前的item_embedding 和 user_embedding 排序的时候可能能够用到，但是需要注意保存的时候需要和原始的id对应
    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    # embedding保存之前归一化一下.进行归一化，确保向量的模为 1，方便后续相似性计算。
    user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
    item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

    # 将Embedding转换成字典的形式方便查询。将索引转换回原始 ID，并存入字典。
    raw_user_id_emb_dict = {user_index_2_rawid[k]: \
                                v for k, v in zip(user_profile['user_id'], user_embs)}
    raw_item_id_emb_dict = {item_index_2_rawid[k]: \
                                v for k, v in zip(item_profile['click_article_id'], item_embs)}
    # 将Embedding保存到本地
    pickle.dump(raw_user_id_emb_dict, open(save_path + 'user_youtube_emb.pkl', 'wb'))
    pickle.dump(raw_item_id_emb_dict, open(save_path + 'item_youtube_emb.pkl', 'wb'))

    # faiss紧邻搜索，通过user_embedding 搜索与其相似性最高的topk个item。
    # IndexFlatIP：使用内积（Inner Product）计算相似度。
    index = faiss.IndexFlatIP(embedding_dim)
    # 上面已经进行了归一化，这里可以不进行归一化了
    #     faiss.normalize_L2(user_embs)
    #     faiss.normalize_L2(item_embs)
    index.add(item_embs)  # 将item向量构建索引
    sim, idx = index.search(np.ascontiguousarray(user_embs), topk)  # 通过user去查询最相似的topk个item

    # 遍历 Faiss 召回的 Top-K 结果，构造用户召回字典。
    user_recall_items_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(test_user_model_input['user_id'], sim, idx)):
        target_raw_id = user_index_2_rawid[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = item_index_2_rawid[rele_idx]
            user_recall_items_dict[target_raw_id][rele_raw_id] = user_recall_items_dict.get(target_raw_id, {}) \
                                                                     .get(rele_raw_id, 0) + sim_value

    user_recall_items_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True) for k, v in
                              user_recall_items_dict.items()}
    # 将召回的结果进行排序

    # 保存召回的结果
    # 这里是直接通过向量的方式得到了召回结果，相比于上面的召回方法，上面的只是得到了i2i及u2u的相似性矩阵，还需要进行协同过滤召回才能得到召回结果
    # 可以直接对这个召回结果进行评估，为了方便可以统一写一个评估函数对所有的召回结果进行评估
    pickle.dump(user_recall_items_dict, open(save_path + 'youtube_u2i_dict.pkl', 'wb'))
    return user_recall_items_dict



# 由于这里需要做召回评估，所以讲训练集中的最后一次点击都提取了出来。
# if not metric_recall:   # False：不进行召回效果评估，直接执行召回。
#     user_multi_recall_dict['youtubednn_recall'] = youtubednn_u2i_dict(all_click_df, topk=20)
# else:  # True：需要评估召回效果，则划分数据集并计算指标。
#     trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
#     user_multi_recall_dict['youtubednn_recall'] = youtubednn_u2i_dict(trn_hist_click_df, topk=20)
#     # 召回效果评估
#     metrics_recall(user_multi_recall_dict['youtubednn_recall'], trn_last_click_df, topk=20)



# itemCF recall
# 考虑相似文章与历史点击文章顺序的权重(细节看代码)
# 考虑文章创建时间的权重，也就是考虑相似文章与历史点击文章创建时间差的权重
# 考虑文章内容相似度权重
# 基于商品的召回i2i
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click,
                         item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: {item1: time1, item2: time2..}...}
        :param i2i_sim: 字典，文章相似性矩阵
        :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵

        return: 召回的文章列表 {item1:score1, item2: score2...}

    """
    # 获取用户历史交互的文章
    # user_item_time_dict：存储用户点击的文章及其时间，格式：{user1: {item1: time1, item2: time2, ...}, user2: {item3: time3, ...}, ...}
    # user_hist_items：获取指定 user_id 的历史点击文章序列。
    user_hist_items = user_item_time_dict[user_id]

    item_rank = {}
    # i 是用户点击过的文章 ID，click_time 是点击时间。
    # loc 记录文章在用户历史序列中的索引，用于计算位置权重。
    for loc, (i, click_time) in enumerate(user_hist_items):
        # i2i_sim[i]：物品相似度矩阵，存储文章 i 与其他文章 j 的相似度
        if i not in i2i_sim:  # 避免 KeyError
            print(f"Warning: Item {i} not found in i2i_sim, skipping...")
            continue
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items:  # 如果 j 已被用户点击过，跳过，避免重复推荐。
                continue

            # 文章创建时间差权重。0.8 ** 差值 让较新文章的权重更高。
            created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
            # 相似文章和历史点击文章序列中历史文章所在的位置权重。使得最近点击的文章影响力较大。
            loc_weight = (0.9 ** (len(user_hist_items) - loc))

            content_weight = 1.0
            # emb_i2i_sim：基于内容 embedding 计算的文章相似矩阵，用于提升推荐结果的准确性。
            # content_weight 初始值为 1.0，如果 i 和 j 存在内容相似度，增加权重。
            if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                content_weight += emb_i2i_sim[i][j]
            if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                content_weight += emb_i2i_sim[j][i]

            item_rank.setdefault(j, 0)
            # created_time_weight：时间差影响。
            # loc_weight：点击顺序影响。
            # content_weight：内容相似度影响。
            # wij：文章协同过滤相似度。
            item_rank[j] += created_time_weight * loc_weight * content_weight * wij

    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):  # item_topk_click：全局热门文章列表（点击次数最多）。
            if item in item_rank.items():  # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = - i - 100  # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break

    # 按得分降序排序，取前recall_item_num篇文章作为最终推荐结果。
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return item_rank


# itemCF sim召回
# 先进行itemcf召回, 为了召回评估，所以提取最后一次点击

# 该代码实现了基于 ItemCF（物品协同过滤）的召回，并在需要时评估召回效果。主要流程如下：
#
# 数据准备：
#
# 若 metric_recall=True，则将用户的最后一次点击提取出来，用于评估召回效果。
# 构建 user_item_time_dict，用于存储用户的点击历史。
# 加载协同过滤相似度矩阵：
#
# i2i_sim：基于物品协同过滤计算的文章相似度矩阵。
# emb_i2i_sim：基于内容 Embedding 计算的文章相似度矩阵。
# 召回计算：
#
# 设定 sim_item_topk=20（每篇文章选择相似度最高的 20 篇）和 recall_item_num=10（最终召回 10 篇文章）。
# 获取点击次数最多的前 50 篇文章，用于补全召回列表。
# 遍历所有用户，调用 item_based_recommend 计算推荐结果，并存入 user_recall_items_dict。
# 结果存储与评估：
#
# 召回结果保存至 user_multi_recall_dict，并序列化到本地文件。
# 若 metric_recall=True，则基于用户最后一次点击数据，计算召回效果指标。
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)

i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))
emb_i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl', 'rb'))

sim_item_topk = 20
recall_item_num = 10
item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

# for user in tqdm(trn_hist_click_df['user_id'].unique()):
#     user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, \
#                                                         i2i_sim, sim_item_topk, recall_item_num, \
#                                                         item_topk_click, item_created_time_dict, emb_i2i_sim)
#
# user_multi_recall_dict['itemcf_sim_itemcf_recall'] = user_recall_items_dict
# pickle.dump(user_multi_recall_dict['itemcf_sim_itemcf_recall'], open(save_path + 'itemcf_recall_dict.pkl', 'wb'))
#
# if metric_recall:
#     # 召回效果评估
#     metrics_recall(user_multi_recall_dict['itemcf_sim_itemcf_recall'], trn_last_click_df, topk=recall_item_num)



# embedding sim 召回
# 这里是为了召回评估，所以提取最后一次点击
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl', 'rb'))

sim_item_topk = 20
recall_item_num = 10

item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

# for user in tqdm(trn_hist_click_df['user_id'].unique()):
#     user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, sim_item_topk,
#                                                         recall_item_num, item_topk_click, item_created_time_dict,
#                                                         emb_i2i_sim)
#
# user_multi_recall_dict['embedding_sim_item_recall'] = user_recall_items_dict
# pickle.dump(user_multi_recall_dict['embedding_sim_item_recall'],
#             open(save_path + 'embedding_sim_item_recall.pkl', 'wb'))
#
# if metric_recall:
#     # 召回效果评估
#     metrics_recall(user_multi_recall_dict['embedding_sim_item_recall'], trn_last_click_df, topk=recall_item_num)






# 基于用户的召回 u2u2i
def user_based_recommend(user_id, user_item_time_dict, u2u_sim, sim_user_topk, recall_item_num,
                         item_topk_click, item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: {item1: time1, item2: time2..}...}
        :param u2u_sim: 字典，文章相似性矩阵
        :param sim_user_topk: 整数， 选择与当前用户最相似的前k个用户
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        :param item_created_time_dict: 文章创建时间列表
        :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵

        return: 召回的文章列表 {item1:score1, item2: score2...}
    """
    # 历史交互
    user_item_time_list = user_item_time_dict[user_id]  # {item1: time1, item2: time2...}
    user_hist_items = set([i for i, t in user_item_time_list])  # 存在一个用户与某篇文章的多次交互， 这里得去重

    items_rank = {}
    for sim_u, wuv in sorted(u2u_sim[user_id].items(), key=lambda x: x[1], reverse=True)[:sim_user_topk]:
        for i, click_time in user_item_time_dict[sim_u]:
            if i in user_hist_items:
                continue
            items_rank.setdefault(i, 0)

            loc_weight = 1.0
            content_weight = 1.0
            created_time_weight = 1.0

            # 当前文章与该用户看的历史文章进行一个权重交互
            for loc, (j, click_time) in enumerate(user_item_time_list):
                # 点击时的相对位置权重
                loc_weight += 0.9 ** (len(user_item_time_list) - loc)
                # 内容相似性权重
                if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                    content_weight += emb_i2i_sim[i][j]
                if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                    content_weight += emb_i2i_sim[j][i]

                # 创建时间差权重
                created_time_weight += np.exp(0.8 * np.abs(item_created_time_dict[i] - item_created_time_dict[j]))

            items_rank[i] += loc_weight * content_weight * created_time_weight * wuv

    # 热度补全
    if len(items_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in items_rank.items():  # 填充的item应该不在原来的列表中
                continue
            items_rank[item] = - i - 100  # 随便给个复数就行
            if len(items_rank) == recall_item_num:
                break

    items_rank = sorted(items_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return items_rank



# userCF sim召回
# 这里是为了召回评估，所以提取最后一次点击
# 由于usercf中计算user之间的相似度的过程太费内存了，全量数据这里就没有跑，跑了一个采样之后的数据
# if metric_recall:
#     trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
# else:
#     trn_hist_click_df = all_click_df
#
# user_recall_items_dict = collections.defaultdict(dict)
# user_item_time_dict = get_user_item_time(trn_hist_click_df)
#
# u2u_sim = pickle.load(open(save_path + 'usercf_u2u_sim.pkl', 'rb'))
#
# sim_user_topk = 20
# recall_item_num = 10
# item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)
#
# for user in tqdm(trn_hist_click_df['user_id'].unique()):
#     user_recall_items_dict[user] = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk, \
#                                                         recall_item_num, item_topk_click, item_created_time_dict,
#                                                         emb_i2i_sim)
#
# pickle.dump(user_recall_items_dict, open(save_path + 'usercf_u2u2i_recall.pkl', 'wb'))
#
# if metric_recall:
#     # 召回效果评估
#     metrics_recall(user_recall_items_dict, trn_last_click_df, topk=recall_item_num)



# user embedding sim召回
# 下面使用了YoutubeDNN过程中产生的user embedding来进行向量检索每个user最相似的topk个user，在使用这里得到的u2u的相似性矩阵，使用usercf进行召回，具体代码如下
# 使用Embedding的方式获取u2u的相似性矩阵
# topk指的是每个user, faiss搜索后返回最相似的topk个user
def u2u_embdding_sim(click_df, user_emb_dict, save_path, topk):
    user_list = []
    user_emb_list = []
    for user_id, user_emb in user_emb_dict.items():
        user_list.append(user_id)
        user_emb_list.append(user_emb)  # 从 user_emb_dict 提取用户 ID 和对应的嵌入向量。

    user_index_2_rawid_dict = {k: v for k, v in zip(range(len(user_list)), user_list)}

    user_emb_np = np.array(user_emb_list, dtype=np.float32)  # 使用 NumPy 将用户嵌入转换为数组格式 (user_emb_np)，方便进行计算。

    # 建立faiss索引。使用 FAISS 进行相似性计算。
    user_index = faiss.IndexFlatIP(user_emb_np.shape[1])
    user_index.add(user_emb_np)
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = user_index.search(user_emb_np, topk)  # 返回的是列表

    # 将向量检索的结果保存成原始id的对应关系
    user_sim_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(user_emb_np)), sim, idx)):
        target_raw_id = user_index_2_rawid_dict[target_idx]  # 通过 user_index_2_rawid_dict 映射索引位置到原始用户 ID。
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = user_index_2_rawid_dict[rele_idx]
            user_sim_dict[target_raw_id][rele_raw_id] = user_sim_dict.get(target_raw_id, {}).get(rele_raw_id,
                                                                                                 0) + sim_value

    # 保存i2i相似度矩阵
    # pickle.dump(user_sim_dict, open(save_path + 'youtube_u2u_sim.pkl', 'wb'))
    # return user_sim_dict


# 读取YoutubeDNN过程中产生的user embedding, 然后使用faiss计算用户之间的相似度
# 这里需要注意，这里得到的user embedding其实并不是很好，因为YoutubeDNN中使用的是用户点击序列来训练的user embedding,
# 如果序列普遍都比较短的话，其实效果并不是很好
user_emb_dict = pickle.load(open(save_path + 'user_youtube_emb.pkl', 'rb'))
u2u_sim = u2u_embdding_sim(all_click_df, user_emb_dict, save_path, topk=10)


# 通过YoutubeDNN得到的user_embedding
# 使用召回评估函数验证当前召回方式的效果
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
u2u_sim = pickle.load(open(save_path + 'youtube_u2u_sim.pkl', 'rb'))

sim_user_topk = 20
recall_item_num = 10

# item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)
# for user in tqdm(trn_hist_click_df['user_id'].unique()):
#     user_recall_items_dict[user] = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk, \
#                                                         recall_item_num, item_topk_click, item_created_time_dict,
#                                                         emb_i2i_sim)
#
# user_multi_recall_dict['youtubednn_usercf_recall'] = user_recall_items_dict
# pickle.dump(user_multi_recall_dict['youtubednn_usercf_recall'], open(save_path + 'youtubednn_usercf_recall.pkl', 'wb'))
#
# if metric_recall:
#     # 召回效果评估
#     metrics_recall(user_multi_recall_dict['youtubednn_usercf_recall'], trn_last_click_df, topk=recall_item_num)




# 冷启动问题
# 1，文章冷启动(没有冷启动的探索问题）
# 其实我们这里不是为了做文章的冷启动而做冷启动，而是猜测用户可能会点击一些没有在log数据中出现的文章，我们要做的就是如何从将近27万的文章中选择一些文章作为用户冷启动的文章
# 下面给出一些参考的方案。
# 首先基于Embedding召回一部分与用户历史相似的文章
# 从基于Embedding召回的文章中通过一些规则过滤掉一些文章，使得留下的文章用户更可能点击。我们这里的规则，可以是，留下那些与用户历史点击文章主题相同的文章，
# 或者字数相差不大的文章。并且留下的文章尽量是与测试集用户最后一次点击时间更接近的文章，或者是当天的文章也行。
# 2，用户冷启动
# 这里对测试集中的用户点击数据进行分析会发现，测试集中有百分之20的用户只有一次点击，那么这些点击特别少的用户的召回是不是可以单独做一些策略上的补充呢？
# 或者是在排序后直接基于规则加上一些文章呢？这些都可以去尝试，这里没有提供具体的做法。

# 这里看似和基于embedding计算的item之间相似度然后做itemcf是一致的，但是现在我们的目的不一样，我们这里的目的是找到相似的向量，
# 并且还没有出现在log日志中的商品，再加上一些其他的冷启动的策略，这里需要召回的数量会偏多一点。
# 先进行itemcf召回，这里不需要做召回评估，这里只是一种策略
trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl','rb'))

sim_item_topk = 150
recall_item_num = 100 # 稍微召回多一点文章，便于后续的规则筛选

item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)
# for user in tqdm(trn_hist_click_df['user_id'].unique()):
#     user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, sim_item_topk,
#                                                         recall_item_num, item_topk_click,item_created_time_dict, emb_i2i_sim)
# pickle.dump(user_recall_items_dict, open(save_path + 'cold_start_items_raw_dict.pkl', 'wb'))


# 基于规则进行文章过滤
# 保留文章主题与用户历史浏览主题相似的文章
# 保留文章字数与用户历史浏览文章字数相差不大的文章
# 保留最后一次点击当天的文章
# 按照相似度返回最终的结果

def get_click_article_ids_set(all_click_df):
    return set(all_click_df.click_article_id.values)


def cold_start_items(user_recall_items_dict, user_hist_item_typs_dict, user_hist_item_words_dict, \
                     user_last_item_created_time_dict, item_type_dict, item_words_dict,
                     item_created_time_dict, click_article_ids_set, recall_item_num):
    """
        冷启动的情况下召回一些文章
        :param user_recall_items_dict: 基于内容embedding相似性召回来的很多文章， 字典， {user1: [item1, item2, ..], }
        :param user_hist_item_typs_dict: 字典， 用户点击的文章的主题映射
        :param user_hist_item_words_dict: 字典， 用户点击的历史文章的字数映射
        :param user_last_item_created_time_idct: 字典，用户点击的历史文章创建时间映射
        :param item_tpye_idct: 字典，文章主题映射
        :param item_words_dict: 字典，文章字数映射
        :param item_created_time_dict: 字典， 文章创建时间映射
        :param click_article_ids_set: 集合，用户点击过得文章, 也就是日志里面出现过的文章
        :param recall_item_num: 召回文章的数量， 这个指的是没有出现在日志里面的文章数量
    """

    cold_start_user_items_dict = {}
    for user, item_list in tqdm(user_recall_items_dict.items()):
        cold_start_user_items_dict.setdefault(user, [])
        for item, score in item_list:
            # 获取历史文章信息
            hist_item_type_set = user_hist_item_typs_dict[user]
            hist_mean_words = user_hist_item_words_dict[user]
            hist_last_item_created_time = user_last_item_created_time_dict[user]
            hist_last_item_created_time = datetime.fromtimestamp(hist_last_item_created_time)

            # 获取当前召回文章的信息
            curr_item_type = item_type_dict[item]
            curr_item_words = item_words_dict[item]
            curr_item_created_time = item_created_time_dict[item]
            curr_item_created_time = datetime.fromtimestamp(curr_item_created_time)

            # 首先，文章不能出现在用户的历史点击中， 然后根据文章主题，文章单词数，文章创建时间进行筛选
            if curr_item_type not in hist_item_type_set or \
                    item in click_article_ids_set or \
                    abs(curr_item_words - hist_mean_words) > 200 or \
                    abs((curr_item_created_time - hist_last_item_created_time).days) > 90:
                continue

            cold_start_user_items_dict[user].append((item, score))  # {user1: [(item1, score1), (item2, score2)..]...}

    # 需要控制一下冷启动召回的数量
    cold_start_user_items_dict = {k: sorted(v, key=lambda x: x[1], reverse=True)[:recall_item_num] \
                                  for k, v in cold_start_user_items_dict.items()}

    pickle.dump(cold_start_user_items_dict, open(save_path + 'cold_start_user_items_dict.pkl', 'wb'))

    return cold_start_user_items_dict



all_click_df_ = all_click_df.copy()
all_click_df_ = all_click_df_.merge(item_info_df, how='left', on='click_article_id')
user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict = get_user_hist_item_info_dict(all_click_df_)
click_article_ids_set = get_click_article_ids_set(all_click_df)
# 需要注意的是
# 这里使用了很多规则来筛选冷启动的文章，所以前面再召回的阶段就应该尽可能的多召回一些文章，否则很容易被删掉
# cold_start_user_items_dict = cold_start_items(user_recall_items_dict, user_hist_item_typs_dict, user_hist_item_words_dict, \
#                                               user_last_item_created_time_dict, item_type_dict, item_words_dict, \
#                                               item_created_time_dict, click_article_ids_set, recall_item_num)
#
# user_multi_recall_dict['cold_start_recall'] = cold_start_user_items_dict



# 多路召回合并
# 在做召回评估的时候就会发现有些召回的效果不错有些召回的效果很差，所以对每一路召回的结果，我们可以认为的定义一些权重，来做最终的相似度融合
def combine_recall_results(user_multi_recall_dict, weight_dict=None, topk=25):
    final_recall_items_dict = {}

    # 对每一种召回结果按照用户进行归一化，方便后面多种召回结果，相同用户的物品之间权重相加
    def norm_user_recall_items_sim(sorted_item_list):
        # 如果冷启动中没有文章或者只有一篇文章，直接返回，出现这种情况的原因可能是冷启动召回的文章数量太少了，
        # 基于规则筛选之后就没有文章了, 这里还可以做一些其他的策略性的筛选
        if len(sorted_item_list) < 2:
            return sorted_item_list

        min_sim = sorted_item_list[-1][1]
        max_sim = sorted_item_list[0][1]

        norm_sorted_item_list = []
        for item, score in sorted_item_list:
            if max_sim > 0:
                norm_score = 1.0 * (score - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 1.0
            else:
                norm_score = 0.0
            norm_sorted_item_list.append((item, norm_score))

        return norm_sorted_item_list

    print('多路召回合并...')
    for method, user_recall_items in tqdm(user_multi_recall_dict.items()):
        print(method + '...')
        # 在计算最终召回结果的时候，也可以为每一种召回结果设置一个权重
        if weight_dict == None:
            recall_method_weight = 1
        else:
            recall_method_weight = weight_dict[method]

        for user_id, sorted_item_list in user_recall_items.items():  # 进行归一化
            user_recall_items[user_id] = norm_user_recall_items_sim(sorted_item_list)

        for user_id, sorted_item_list in user_recall_items.items():
            # print('user_id')
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in sorted_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0)
                final_recall_items_dict[user_id][item] += recall_method_weight * score

    final_recall_items_dict_rank = {}
    # 多路召回时也可以控制最终的召回数量
    for user, recall_item_dict in final_recall_items_dict.items():
        final_recall_items_dict_rank[user] = sorted(recall_item_dict.items(), key=lambda x: x[1], reverse=True)[:topk]

    # 将多路召回后的最终结果字典保存到本地
    pickle.dump(final_recall_items_dict, open(os.path.join(save_path, 'final_recall_items_dict.pkl'), 'wb'))

    return final_recall_items_dict_rank


# 这里直接对多路召回的权重给了一个相同的值，其实可以根据前面召回的情况来调整参数的值
weight_dict = {'itemcf_sim_itemcf_recall': 1.0,
               'embedding_sim_item_recall': 1.0,
               'youtubednn_recall': 1.0,
               'youtubednn_usercf_recall': 1.0,
               'cold_start_recall': 1.0}

# 最终合并之后每个用户召回150个商品进行排序
final_recall_items_dict_rank = combine_recall_results(user_multi_recall_dict, weight_dict, topk=150)
















