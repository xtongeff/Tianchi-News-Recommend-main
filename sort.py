# 排序阶段选择了三个比较有代表性的排序模型，它们分别是：
#
# LGB的排序模型
# LGB的分类模型
# 深度学习的分类模型DIN
# 得到了最终的排序模型输出的结果之后，还选择了两种比较经典的模型集成的方法：
#
# 输出结果加权融合
# Staking（将模型的输出结果再使用一个简单模型进行预测）
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import gc, os
import time
from datetime import datetime
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# 读取排序特征
data_path = './download/'
save_path = './output1/'
offline = False

# 重新读取数据的时候，发现click_article_id是一个浮点数，所以将其转换成int类型
trn_user_item_feats_df = pd.read_csv(save_path + 'trn_user_item_feats_df.csv')
trn_user_item_feats_df['click_article_id'] = trn_user_item_feats_df['click_article_id'].astype(int)

if offline:
    val_user_item_feats_df = pd.read_csv(save_path + 'val_user_item_feats_df.csv')
    val_user_item_feats_df['click_article_id'] = val_user_item_feats_df['click_article_id'].astype(int)
else:
    val_user_item_feats_df = None

tst_user_item_feats_df = pd.read_csv(save_path + 'tst_user_item_feats_df.csv')
tst_user_item_feats_df['click_article_id'] = tst_user_item_feats_df['click_article_id'].astype(int)

# 做特征的时候为了方便，给测试集也打上了一个无效的标签，这里直接删掉就行
del tst_user_item_feats_df['label']


# 返回排序后的结果
def submit(recall_df, topk=5, model_name=None):
    # 先对 recall_df 按 user_id 升序，再按 pred_score 降序排序。
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 判断是不是每个用户都有5篇文章及以上
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    assert tmp.min() >= topk

    del recall_df['pred_score']
    # 仅保留 rank 在前 topk=5 的文章。unstack(-1)：将 rank 这一层索引转换为列，使每个 user_id 的推荐文章展开成多列。
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()
    # droplevel(0) 移除 unstack 生成的多级索引，保持扁平化列名。
    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    # 按照提交格式定义列名
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2',
                                    3: 'article_3', 4: 'article_4', 5: 'article_5'})

    save_name = save_path + model_name + '_' + datetime.today().strftime('%m-%d') + '.csv'
    submit.to_csv(save_name, index=False, header=True)


# 排序结果归一化
def norm_sim(sim_df, weight=0.0):
    # print(sim_df.head())
    min_sim = sim_df.min()
    max_sim = sim_df.max()
    if max_sim == min_sim:
        sim_df = sim_df.apply(lambda sim: 1.0)
    else:
        sim_df = sim_df.apply(lambda sim: 1.0 * (sim - min_sim) / (max_sim - min_sim))

    sim_df = sim_df.apply(lambda sim: sim + weight)  # plus one
    return sim_df


# LGB排序模型
# 防止中间出错之后重新读取数据
trn_user_item_feats_df_rank_model = trn_user_item_feats_df.copy()

if offline:
    val_user_item_feats_df_rank_model = val_user_item_feats_df.copy()

tst_user_item_feats_df_rank_model = tst_user_item_feats_df.copy()

# 定义特征列
lgb_cols = ['sim0', 'time_diff0', 'word_diff0', 'sim_max', 'sim_min', 'sim_sum',
            'sim_mean', 'score', 'click_size', 'time_diff_mean', 'active_level',
            'click_environment', 'click_deviceGroup', 'click_os', 'click_country',
            'click_region', 'click_referrer_type', 'user_time_hob1', 'user_time_hob2',
            'words_hbo', 'category_id', 'created_at_ts', 'words_count']

# 排序模型分组
trn_user_item_feats_df_rank_model.sort_values(by=['user_id'], inplace=True)
g_train = trn_user_item_feats_df_rank_model.groupby(['user_id'], as_index=False).count()["label"].values

if offline:
    val_user_item_feats_df_rank_model.sort_values(by=['user_id'], inplace=True)
    g_val = val_user_item_feats_df_rank_model.groupby(['user_id'], as_index=False).count()["label"].values

# 排序模型定义
lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                            max_depth=-1, n_estimators=100, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                            learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=16)

# 排序模型训练
if offline:
    callbacks = [early_stopping(stopping_rounds=50)]
    lgb_ranker.fit(trn_user_item_feats_df_rank_model[lgb_cols], trn_user_item_feats_df_rank_model['label'],
                   group=g_train,
                   eval_set=[(val_user_item_feats_df_rank_model[lgb_cols], val_user_item_feats_df_rank_model['label'])],
                   eval_group=[g_val], eval_at=[1, 2, 3, 4, 5], eval_metric=['ndcg', ], callbacks=callbacks, )
else:
    lgb_ranker.fit(trn_user_item_feats_df[lgb_cols], trn_user_item_feats_df['label'], group=g_train)

# 模型预测
tst_user_item_feats_df['pred_score'] = lgb_ranker.predict(tst_user_item_feats_df[lgb_cols],
                                                          num_iteration=lgb_ranker.best_iteration_)

# 将这里的排序结果保存一份，用户后面的模型融合
tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']].to_csv(save_path + 'lgb_ranker_score.csv',
                                                                             index=False)

# 预测结果重新排序, 及生成提交结果
rank_results = tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']]
rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
submit(rank_results, topk=5, model_name='lgb_ranker')


# 五折交叉验证，这里的五折交叉是以用户为目标进行五折划分
#  这一部分与前面的单独训练和验证是分开的
def get_kfold_users(trn_df, n=5):  # 按用户 ID 进行 5 折交叉划分
    user_ids = trn_df['user_id'].unique()  # 取出所有 user_id，然后按照 切片 方法，均匀地分成 n=5 组。这样可以保证每个折叠中的用户是相对独立的，避免数据泄漏。
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set


k_fold = 5
trn_df = trn_user_item_feats_df_rank_model
user_set = get_kfold_users(trn_df, n=k_fold)  # 获取五折交叉验证的用户集合，用于后续分层抽样。

score_list = []  # 用于存储每折交叉验证的 验证集预测结果。
score_df = trn_df[['user_id', 'click_article_id', 'label']]  # 保存原始数据的 user_id、click_article_id 和 label，用于合并交叉验证预测结果。
sub_preds = np.zeros(tst_user_item_feats_df_rank_model.shape[
                         0])  # 用于存储 测试集的预测结果，后面通过 五折平均 得到最终预测分数。创建一个全 0 的 NumPy 数组 sub_preds，其长度等于测试集的样本数。

# 五折交叉验证，并将中间结果保存用于staking
for n_fold, valid_user in enumerate(user_set):
    train_idx = trn_df[~trn_df['user_id'].isin(valid_user)]  # add slide user，除了 valid_user 以外的用户作为 训练集。
    valid_idx = trn_df[trn_df['user_id'].isin(valid_user)]  # 当前折的 valid_user 作为 验证集。

    # 训练集与验证集的用户分组。由于 LGBMRanker 需要 按用户分组 进行排序任务，因此需要统计每个用户的样本数量。
    # g_train 和 g_val 记录了训练集和验证集中每个用户的交互样本数。
    train_idx.sort_values(by=['user_id'], inplace=True)
    g_train = train_idx.groupby(['user_id'], as_index=False).count()["label"].values

    valid_idx.sort_values(by=['user_id'], inplace=True)
    g_val = valid_idx.groupby(['user_id'], as_index=False).count()["label"].values

    # 定义模型
    lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                max_depth=-1, n_estimators=100, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                                learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=16)
    # 训练模型
    callbacks = [early_stopping(stopping_rounds=50)]
    lgb_ranker.fit(train_idx[lgb_cols], train_idx['label'], group=g_train,
                   eval_set=[(valid_idx[lgb_cols], valid_idx['label'])], eval_group=[g_val],
                   eval_at=[1, 2, 3, 4, 5], eval_metric=['ndcg', ], callbacks=callbacks, )

    # 预测验证集结果
    valid_idx['pred_score'] = lgb_ranker.predict(valid_idx[lgb_cols], num_iteration=lgb_ranker.best_iteration_)

    # 对输出结果进行归一化
    valid_idx['pred_score'] = valid_idx[['pred_score']].transform(lambda x: norm_sim(x))

    valid_idx.sort_values(by=['user_id', 'pred_score'])  # 计算排名
    valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 将验证集的预测结果放到一个列表中，后面进行拼接。记录每折的结果
    score_list.append(valid_idx[['user_id', 'click_article_id', 'pred_score', 'pred_rank']])

    # 如果是线上测试，需要计算每次交叉验证的结果相加，最后求平均
    if not offline:
        sub_preds += lgb_ranker.predict(tst_user_item_feats_df_rank_model[lgb_cols], lgb_ranker.best_iteration_)

score_df_ = pd.concat(score_list, axis=0)
score_df = score_df.merge(score_df_, how='left', on=['user_id', 'click_article_id'])  # 合并五折交叉验证的预测结果。
# 保存训练集交叉验证产生的新特征
score_df[['user_id', 'click_article_id', 'pred_score', 'pred_rank', 'label']].to_csv(
    save_path + 'trn_lgb_ranker_feats.csv', index=False)

# 测试集的预测结果，多次交叉验证求平均,将预测的score和对应的rank特征保存，可以用于后面的staking，这里还可以构造其他更多的特征
tst_user_item_feats_df_rank_model['pred_score'] = sub_preds / k_fold  # 测试集的最终预测分数 = 五折的平均值，并进行归一化。
tst_user_item_feats_df_rank_model['pred_score'] = tst_user_item_feats_df_rank_model['pred_score'].transform(
    lambda x: norm_sim(x))
tst_user_item_feats_df_rank_model.sort_values(by=['user_id', 'pred_score'])  # 对测试集进行排序，计算最终排名 pred_rank。
tst_user_item_feats_df_rank_model['pred_rank'] = tst_user_item_feats_df_rank_model.groupby(['user_id'])[
    'pred_score'].rank(ascending=False, method='first')

# 保存测试集交叉验证的新特征
tst_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score', 'pred_rank']].to_csv(
    save_path + 'tst_lgb_ranker_feats.csv', index=False)

# 预测结果重新排序, 及生成提交结果
# 单模型生成提交结果
rank_results = tst_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score']]
rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
submit(rank_results, topk=5, model_name='lgb_ranker')

# LGB分类模型
# 模型及参数的定义
lgb_Classfication = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                       max_depth=-1, n_estimators=500, subsample=0.7, colsample_bytree=0.7,
                                       subsample_freq=1,
                                       learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=16,
                                       verbose=10)

# 模型训练
if offline:
    callbacks = [early_stopping(stopping_rounds=50)]
    lgb_Classfication.fit(trn_user_item_feats_df_rank_model[lgb_cols], trn_user_item_feats_df_rank_model['label'],
                          eval_set=[(val_user_item_feats_df_rank_model[lgb_cols],
                                     val_user_item_feats_df_rank_model['label'])],
                          eval_metric=['auc', ], callbacks=callbacks, )
else:
    lgb_Classfication.fit(trn_user_item_feats_df_rank_model[lgb_cols], trn_user_item_feats_df_rank_model['label'])

# 模型预测
tst_user_item_feats_df['pred_score'] = lgb_Classfication.predict_proba(tst_user_item_feats_df[lgb_cols])[:, 1]

# 将这里的排序结果保存一份，用户后面的模型融合
tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']].to_csv(save_path + 'lgb_cls_score.csv',
                                                                             index=False)

# 预测结果重新排序, 及生成提交结果
rank_results = tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']]
rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
submit(rank_results, topk=5, model_name='lgb_cls')


# 五折交叉验证，这里的五折交叉是以用户为目标进行五折划分
#  这一部分与前面的单独训练和验证是分开的
def get_kfold_users(trn_df, n=5):
    user_ids = trn_df['user_id'].unique()
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set


k_fold = 5
trn_df = trn_user_item_feats_df_rank_model
user_set = get_kfold_users(trn_df, n=k_fold)

score_list = []
score_df = trn_df[['user_id', 'click_article_id', 'label']]
sub_preds = np.zeros(tst_user_item_feats_df_rank_model.shape[0])

# 五折交叉验证，并将中间结果保存用于staking
for n_fold, valid_user in enumerate(user_set):
    train_idx = trn_df[~trn_df['user_id'].isin(valid_user)]  # add slide user
    valid_idx = trn_df[trn_df['user_id'].isin(valid_user)]

    # 模型及参数的定义
    lgb_Classfication = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                           max_depth=-1, n_estimators=100, subsample=0.7, colsample_bytree=0.7,
                                           subsample_freq=1,
                                           learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=16,
                                           verbose=10)
    # 训练模型
    callbacks = [early_stopping(stopping_rounds=50)]
    lgb_Classfication.fit(train_idx[lgb_cols], train_idx['label'], eval_set=[(valid_idx[lgb_cols], valid_idx['label'])],
                          eval_metric=['auc', ], callbacks=callbacks, )

    # 预测验证集结果
    valid_idx['pred_score'] = lgb_Classfication.predict_proba(valid_idx[lgb_cols],
                                                              num_iteration=lgb_Classfication.best_iteration_)[:, 1]

    # 对输出结果进行归一化 分类模型输出的值本身就是一个概率值不需要进行归一化
    # valid_idx['pred_score'] = valid_idx[['pred_score']].transform(lambda x: norm_sim(x))

    valid_idx.sort_values(by=['user_id', 'pred_score'])
    valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 将验证集的预测结果放到一个列表中，后面进行拼接
    score_list.append(valid_idx[['user_id', 'click_article_id', 'pred_score', 'pred_rank']])

    # 如果是线上测试，需要计算每次交叉验证的结果相加，最后求平均
    if not offline:
        sub_preds += lgb_Classfication.predict_proba(tst_user_item_feats_df_rank_model[lgb_cols],
                                                     num_iteration=lgb_Classfication.best_iteration_)[:, 1]

score_df_ = pd.concat(score_list, axis=0)
score_df = score_df.merge(score_df_, how='left', on=['user_id', 'click_article_id'])
# 保存训练集交叉验证产生的新特征
score_df[['user_id', 'click_article_id', 'pred_score', 'pred_rank', 'label']].to_csv(
    save_path + 'trn_lgb_cls_feats.csv', index=False)

# 测试集的预测结果，多次交叉验证求平均,将预测的score和对应的rank特征保存，可以用于后面的staking，这里还可以构造其他更多的特征
tst_user_item_feats_df_rank_model['pred_score'] = sub_preds / k_fold
tst_user_item_feats_df_rank_model['pred_score'] = tst_user_item_feats_df_rank_model['pred_score'].transform(
    lambda x: norm_sim(x))
tst_user_item_feats_df_rank_model.sort_values(by=['user_id', 'pred_score'])
tst_user_item_feats_df_rank_model['pred_rank'] = tst_user_item_feats_df_rank_model.groupby(['user_id'])[
    'pred_score'].rank(ascending=False, method='first')

# 保存测试集交叉验证的新特征
tst_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score', 'pred_rank']].to_csv(
    save_path + 'tst_lgb_cls_feats.csv', index=False)

# 预测结果重新排序, 及生成提交结果
rank_results = tst_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score']]
rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
submit(rank_results, topk=5, model_name='lgb_cls')

# DIN模型
# 用户的历史点击行为列表,这个是为后面的DIN模型服务的.
if offline:
    all_data = pd.read_csv('./download/train_click_log.csv')
else:  # 走这条
    trn_data = pd.read_csv('./download/train_click_log.csv')
    tst_data = pd.read_csv('./download/testA_click_log.csv')
    all_data = trn_data.append(tst_data)

hist_click = all_data[['user_id', 'click_article_id']].groupby('user_id').agg({list}).reset_index()
his_behavior_df = pd.DataFrame()
his_behavior_df['user_id'] = hist_click['user_id']
his_behavior_df['hist_click_article_id'] = hist_click['click_article_id']

trn_user_item_feats_df_din_model = trn_user_item_feats_df.copy()

if offline:
    val_user_item_feats_df_din_model = val_user_item_feats_df.copy()
else:
    val_user_item_feats_df_din_model = None

tst_user_item_feats_df_din_model = tst_user_item_feats_df.copy()

trn_user_item_feats_df_din_model = trn_user_item_feats_df_din_model.merge(his_behavior_df, on='user_id')

if offline:
    val_user_item_feats_df_din_model = val_user_item_feats_df_din_model.merge(his_behavior_df, on='user_id')
else:
    val_user_item_feats_df_din_model = None

tst_user_item_feats_df_din_model = tst_user_item_feats_df_din_model.merge(his_behavior_df, on='user_id')

# DIN模型简介
# 我们下面尝试使用DIN模型， DIN的全称是Deep Interest Network， 这是阿里2018年基于前面的深度学习模型无法表达用户多样化的兴趣而提出的一个模型，
# 它可以通过考虑【给定的候选广告】和【用户的历史行为】的相关性，来计算用户兴趣的表示向量。具体来说就是通过引入局部激活单元，
# 通过软搜索历史行为的相关部分来关注相关的用户兴趣，并采用加权和来获得有关候选广告的用户兴趣的表示。与候选广告相关性较高的行为会获得较高的激活权重，
# 并支配着用户兴趣。该表示向量在不同广告上有所不同，大大提高了模型的表达能力。

# 1，首先，我们要处理数据集， 得到数据， 由于我们是基于用户过去的行为去预测用户是否点击当前文章， 所以我们需要把数据的特征列划分成数值型特征，
# 离散型特征和历史行为特征列三部分， 对于每一部分， DIN模型的处理会有不同。
#     1.对于离散型特征， 在我们的数据集中就是那些类别型的特征， 比如user_id这种， 这种类别型特征， 我们首先要经过embedding处理得到每个特征的低维稠密型
# 表示。
#     2.对于用户历史行为特征列， 比如文章id， 文章的类别等这种， 同样的我们需要先经过embedding处理， 只不过和上面不一样的地方是，对于这种特征，
# 我们在得到每个特征的embedding表示之后， 还需要通过一个Attention_layer计算用户的历史行为和当前候选文章的相关性以此得到当前用户的embedding向量。
# 对于每个用户， 历史行为序列长度会不一样，我们首先要通过SparseFeat函数指明这些类别型特征， 然后还需要通过VarLenSparseFeat函数再进行序列填充，
# 使得每个用户的历史序列一样长。
#     3.对于连续型特征列， 我们只需要用DenseFeat函数来指明列名和维度即可。
# 2，处理完特征列之后， 我们把相应的数据与列进行对应，就得到了最后的数据。

# 下面根据具体的代码感受一下， 逻辑是这样， 首先我们需要写一个数据准备函数， 在这里面就是根据上面的具体步骤准备数据， 得到数据和特征列，
# 然后就是建立DIN模型并训练， 最后基于模型进行测试。
# 导入deepctr
from deepctr.models import DIN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from keras_preprocessing.sequence import pad_sequences

from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import tensorflow as tf

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 数据准备函数
def get_din_feats_columns(df, dense_fea, sparse_fea, behavior_fea, his_behavior_fea, emb_dim=32, max_len=100):
    """
    数据准备函数:
    df: 数据集
    dense_fea: 数值型特征列
    sparse_fea: 离散型特征列
    behavior_fea: 用户的候选行为特征列
    his_behavior_fea: 用户的历史行为特征列
    embedding_dim: embedding的维度， 这里为了简单， 统一把离散型特征列采用一样的隐向量维度
    max_len: 用户序列的最大长度
    """

    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].max() + 1, embedding_dim=emb_dim) for feat
                              in sparse_fea]  # 构造离散特征

    dense_feature_columns = [DenseFeat(feat, 1, ) for feat in dense_fea]  # 构造数值特征

    var_feature_columns = [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=df['click_article_id'].max() + 1,
                                                       embedding_dim=emb_dim, embedding_name='click_article_id'),
                                            maxlen=max_len) for feat in hist_behavior_fea]  # 构造序列特征

    dnn_feature_columns = sparse_feature_columns + dense_feature_columns + var_feature_columns  # 整合所有特征

    # 建立x, x是一个字典的形式。构造模型输入 x
    x = {}
    for name in get_feature_names(dnn_feature_columns):  # 遍历所有特征名称：
        if name in his_behavior_fea:  # 如果特征在 his_behavior_fea（历史行为）中：
            # 这是历史行为序列
            his_list = [l for l in df[name]]
            x[name] = pad_sequences(his_list, maxlen=max_len, padding='post')  # 二维数组。进行序列填充，确保序列长度一致（短的补零）
        else:
            x[name] = df[name].values  # 其他特征直接取

    return x, dnn_feature_columns  # x：字典形式的输入数据，dnn_feature_columns：DIN 模型的特征列定义


# 把特征分开
sparse_fea = ['user_id', 'click_article_id', 'category_id', 'click_environment', 'click_deviceGroup',
              'click_os', 'click_country', 'click_region', 'click_referrer_type', 'is_cat_hab']  # 离散特征

behavior_fea = ['click_article_id']  # 行为特征

hist_behavior_fea = ['hist_click_article_id']  # 历史行为特征

dense_fea = ['sim0', 'time_diff0', 'word_diff0', 'sim_max', 'sim_min', 'sim_sum', 'sim_mean', 'score',
             'rank','click_size','time_diff_mean','active_level','user_time_hob1','user_time_hob2',
             'words_hbo','words_count']  # 数值特征

# dense特征进行归一化, 神经网络训练都需要将数值进行归一化处理
mm = MinMaxScaler()

# 下面是做一些特殊处理，当在其他的地方出现无效值的时候，不处理无法进行归一化，刚开始可以先把他注释掉，在运行了下面的代码
# 之后如果发现报错，应该先去想办法处理如何不出现inf之类的值
# trn_user_item_feats_df_din_model.replace([np.inf, -np.inf], 0, inplace=True)
# tst_user_item_feats_df_din_model.replace([np.inf, -np.inf], 0, inplace=True)

for feat in dense_fea:
    trn_user_item_feats_df_din_model[feat] = mm.fit_transform(trn_user_item_feats_df_din_model[[feat]])

    if val_user_item_feats_df_din_model is not None:
        val_user_item_feats_df_din_model[feat] = mm.fit_transform(val_user_item_feats_df_din_model[[feat]])

    tst_user_item_feats_df_din_model[feat] = mm.fit_transform(tst_user_item_feats_df_din_model[[feat]])

# 准备训练数据
x_trn, dnn_feature_columns = get_din_feats_columns(trn_user_item_feats_df_din_model, dense_fea,
                                                   sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
y_trn = trn_user_item_feats_df_din_model['label'].values

if offline:
    # 准备验证数据
    x_val, dnn_feature_columns = get_din_feats_columns(val_user_item_feats_df_din_model, dense_fea,
                                                       sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
    y_val = val_user_item_feats_df_din_model['label'].values

dense_fea = [x for x in dense_fea if x != 'label']
x_tst, dnn_feature_columns = get_din_feats_columns(tst_user_item_feats_df_din_model, dense_fea,
                                                   sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)

# 建立模型
model = DIN(dnn_feature_columns, behavior_fea)

# 查看模型结构
model.summary()

# 模型编译
model.compile('adam', 'binary_crossentropy',metrics=['binary_crossentropy', tf.keras.metrics.AUC()])

# 模型训练
if offline:
    history = model.fit(x_trn, y_trn, verbose=1, epochs=10, validation_data=(x_val, y_val) , batch_size=256)
else:
    # 也可以使用上面的语句用自己采样出来的验证集
    # history = model.fit(x_trn, y_trn, verbose=1, epochs=3, validation_split=0.3, batch_size=256)
    history = model.fit(x_trn, y_trn, verbose=1, epochs=2, batch_size=256)

# 模型预测
tst_user_item_feats_df_din_model['pred_score'] = model.predict(x_tst, verbose=1, batch_size=256)
tst_user_item_feats_df_din_model[['user_id', 'click_article_id', 'pred_score']].to_csv(save_path + 'din_rank_score.csv', index=False)

# 预测结果重新排序, 及生成提交结果
rank_results = tst_user_item_feats_df_din_model[['user_id', 'click_article_id', 'pred_score']]
submit(rank_results, topk=5, model_name='din')


# 五折交叉验证，这里的五折交叉是以用户为目标进行五折划分
#  这一部分与前面的单独训练和验证是分开的
def get_kfold_users(trn_df, n=5):
    user_ids = trn_df['user_id'].unique()
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set


k_fold = 5
trn_df = trn_user_item_feats_df_din_model
user_set = get_kfold_users(trn_df, n=k_fold)

score_list = []
score_df = trn_df[['user_id', 'click_article_id', 'label']]
sub_preds = np.zeros(tst_user_item_feats_df_rank_model.shape[0])

dense_fea = [x for x in dense_fea if x != 'label']
x_tst, dnn_feature_columns = get_din_feats_columns(tst_user_item_feats_df_din_model, dense_fea,
                                                   sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)

# 五折交叉验证，并将中间结果保存用于staking
for n_fold, valid_user in enumerate(user_set):
    train_idx = trn_df[~trn_df['user_id'].isin(valid_user)]  # add slide user
    valid_idx = trn_df[trn_df['user_id'].isin(valid_user)]

    # 准备训练数据
    x_trn, dnn_feature_columns = get_din_feats_columns(train_idx, dense_fea,
                                                       sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
    y_trn = train_idx['label'].values

    # 准备验证数据
    x_val, dnn_feature_columns = get_din_feats_columns(valid_idx, dense_fea,
                                                       sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
    y_val = valid_idx['label'].values

    history = model.fit(x_trn, y_trn, verbose=1, epochs=2, validation_data=(x_val, y_val), batch_size=256)

    # 预测验证集结果
    valid_idx['pred_score'] = model.predict(x_val, verbose=1, batch_size=256)

    valid_idx.sort_values(by=['user_id', 'pred_score'])
    valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 将验证集的预测结果放到一个列表中，后面进行拼接
    score_list.append(valid_idx[['user_id', 'click_article_id', 'pred_score', 'pred_rank']])

    # 如果是线上测试，需要计算每次交叉验证的结果相加，最后求平均
    if not offline:
        sub_preds += model.predict(x_tst, verbose=1, batch_size=256)[:, 0]

score_df_ = pd.concat(score_list, axis=0)
score_df = score_df.merge(score_df_, how='left', on=['user_id', 'click_article_id'])
# 保存训练集交叉验证产生的新特征
score_df[['user_id', 'click_article_id', 'pred_score', 'pred_rank', 'label']].to_csv(
    save_path + 'trn_din_cls_feats.csv', index=False)

# 测试集的预测结果，多次交叉验证求平均,将预测的score和对应的rank特征保存，可以用于后面的staking，这里还可以构造其他更多的特征
tst_user_item_feats_df_din_model['pred_score'] = sub_preds / k_fold
tst_user_item_feats_df_din_model['pred_score'] = tst_user_item_feats_df_din_model['pred_score'].transform(
    lambda x: norm_sim(x))
tst_user_item_feats_df_din_model.sort_values(by=['user_id', 'pred_score'])
tst_user_item_feats_df_din_model['pred_rank'] = tst_user_item_feats_df_din_model.groupby(['user_id'])[
    'pred_score'].rank(ascending=False, method='first')

# 保存测试集交叉验证的新特征
tst_user_item_feats_df_din_model[['user_id', 'click_article_id', 'pred_score', 'pred_rank']].to_csv(
    save_path + 'tst_din_cls_feats.csv', index=False)


# 模型融合
# 加权融合
# 读取多个模型的排序结果文件
lgb_ranker = pd.read_csv(save_path + 'lgb_ranker_score.csv')
lgb_cls = pd.read_csv(save_path + 'lgb_cls_score.csv')
din_ranker = pd.read_csv(save_path + 'din_rank_score.csv')
# 这里也可以换成交叉验证输出的测试结果进行加权融合

rank_model = {'lgb_ranker': lgb_ranker,
              'lgb_cls': lgb_cls,
              'din_ranker': din_ranker}

def get_ensumble_predict_topk(rank_model, topk=5):
    final_recall = rank_model['lgb_cls'].append(rank_model['din_ranker'])
    rank_model['lgb_ranker']['pred_score'] = rank_model['lgb_ranker']['pred_score'].transform(lambda x: norm_sim(x))

    final_recall = final_recall.append(rank_model['lgb_ranker'])
    final_recall = final_recall.groupby(['user_id', 'click_article_id'])['pred_score'].sum().reset_index()

    submit(final_recall, topk=topk, model_name='ensemble_fuse')

get_ensumble_predict_topk(rank_model)

# Staking
# 读取多个模型的交叉验证生成的结果文件
# 训练集
trn_lgb_ranker_feats = pd.read_csv(save_path + 'trn_lgb_ranker_feats.csv')
trn_lgb_cls_feats = pd.read_csv(save_path + 'trn_lgb_cls_feats.csv')
trn_din_cls_feats = pd.read_csv(save_path + 'trn_din_cls_feats.csv')

# 测试集
tst_lgb_ranker_feats = pd.read_csv(save_path + 'tst_lgb_ranker_feats.csv')
tst_lgb_cls_feats = pd.read_csv(save_path + 'tst_lgb_cls_feats.csv')
tst_din_cls_feats = pd.read_csv(save_path + 'tst_din_cls_feats.csv')

# 将多个模型输出的特征进行拼接

finall_trn_ranker_feats = trn_lgb_ranker_feats[['user_id', 'click_article_id', 'label']]
finall_tst_ranker_feats = tst_lgb_ranker_feats[['user_id', 'click_article_id']]

for idx, trn_model in enumerate([trn_lgb_ranker_feats, trn_lgb_cls_feats, trn_din_cls_feats]):
    for feat in [ 'pred_score', 'pred_rank']:
        col_name = feat + '_' + str(idx)
        finall_trn_ranker_feats[col_name] = trn_model[feat]

for idx, tst_model in enumerate([tst_lgb_ranker_feats, tst_lgb_cls_feats, tst_din_cls_feats]):
    for feat in [ 'pred_score', 'pred_rank']:
        col_name = feat + '_' + str(idx)
        finall_tst_ranker_feats[col_name] = tst_model[feat]

# 定义一个逻辑回归模型再次拟合交叉验证产生的特征对测试集进行预测
# 这里需要注意的是，在做交叉验证的时候可以构造多一些与输出预测值相关的特征，来丰富这里简单模型的特征
from sklearn.linear_model import LogisticRegression

feat_cols = ['pred_score_0', 'pred_rank_0', 'pred_score_1', 'pred_rank_1', 'pred_score_2', 'pred_rank_2']

trn_x = finall_trn_ranker_feats[feat_cols]
trn_y = finall_trn_ranker_feats['label']

tst_x = finall_tst_ranker_feats[feat_cols]

# 定义模型
lr = LogisticRegression()

# 模型训练
lr.fit(trn_x, trn_y)

# 模型预测
finall_tst_ranker_feats['pred_score'] = lr.predict_proba(tst_x)[:, 1]

# 预测结果重新排序, 及生成提交结果
rank_results = finall_tst_ranker_feats[['user_id', 'click_article_id', 'pred_score']]
submit(rank_results, topk=5, model_name='ensumble_staking')
