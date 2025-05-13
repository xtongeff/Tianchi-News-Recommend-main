import argparse
import gc
import os
import random
import warnings

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils import Logger, evaluate, gen_sub

warnings.filterwarnings('ignore')

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='lightgbm 排序')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'lightgbm 排序，mode: {mode}')


# 使用 LightGBM 模型对给定的特征数据 df_feature 进行训练，并在测试数据上进行预测，
# 同时会进行模型评估和生成提交文件。函数采用 5 折交叉验证的方式进行训练，最终输出特征重要性、线下评估指标，
def train_model(df_feature, df_query):
    df_train = df_feature[df_feature['label'].notnull()]
    df_test = df_feature[df_feature['label'].isnull()]

    del df_feature
    gc.collect()

    ycol = 'label'
    feature_names = list(
        filter(
            lambda x: x not in [ycol, 'created_at_datetime', 'click_datetime'],
            df_train.columns))  # 排除目标列 label 以及 created_at_datetime 和 click_datetime 这两列。
    feature_names.sort()

    model = lgb.LGBMClassifier(num_leaves=64,
                               max_depth=10,
                               learning_rate=0.05,
                               n_estimators=10000,
                               subsample=0.8,
                               feature_fraction=0.8,
                               reg_alpha=0.5,
                               reg_lambda=0.5,
                               random_state=seed,
                               importance_type='gain',
                               metric=None)

    oof = []  # 用于存储每一折验证集的预测结果。
    prediction = df_test[['user_id', 'article_id']]
    prediction['pred'] = 0  # 用于存储测试集的预测结果，初始预测值都设为 0。
    df_importance_list = []  # 用于存储每一折训练得到的特征重要性数据。

    # 训练模型
    # 对于每一折：
    # 划分训练集和验证集。
    # 使用训练集对模型进行训练，同时在训练集和验证集上进行评估，使用 AUC 作为评估指标，并设置早停机制。
    # 对验证集进行预测，将预测结果存储在 df_oof 中，并添加到 oof 列表。
    # 对测试集进行预测，将预测结果累加到 prediction 中，并取平均值。
    # 计算特征重要性，存储在 df_importance 中，并添加到 df_importance_list 列表。
    # 将训练好的模型保存到文件中。
    # 将数据划分为 5 折意味着使用GroupKFold交叉验证策略把数据集df_train按照用户user_id进行分组，然后将其分成 5 个大致相等的子集，每个子集都包含不同用户的数据。在每次迭代中，其中 4 个子集被用作训练集，剩下的 1 个子集被用作验证集，这样可以进行 5 次不同的训练和验证，以评估模型在不同数据划分下的性能，减少模型评估的方差，提高模型评估的准确性和稳定性。
    kfold = GroupKFold(n_splits=5)
    for fold_id, (trn_idx, val_idx) in enumerate(
            kfold.split(df_train[feature_names], df_train[ycol],
                        df_train['user_id'])):
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][ycol]

        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][ycol]

        log.debug(
            f'\nFold_{fold_id + 1} Training ================================\n'
        )

        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=100,
                              eval_metric='auc',
                              early_stopping_rounds=100)

        pred_val = lgb_model.predict_proba(
            X_val, num_iteration=lgb_model.best_iteration_)[:, 1]
        df_oof = df_train.iloc[val_idx][['user_id', 'article_id', ycol]].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)

        pred_test = lgb_model.predict_proba(  # 使用已经训练好的 LightGBM 模型 lgb_model 对测试数据 df_test 进行预测，并且获取每个样本属于正类别的概率。
            df_test[feature_names], num_iteration=lgb_model.best_iteration_)[:,
                                                                             1]
        prediction['pred'] += pred_test / 5

        df_importance = pd.DataFrame({
            'feature_name':
            feature_names,
            'importance':
            lgb_model.feature_importances_,
        })
        df_importance_list.append(df_importance)

        joblib.dump(model, f'../user_data/model/lgb{fold_id}.pkl')

    # 特征重要性
    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby([
        'feature_name'
    ])['importance'].agg('mean').sort_values(ascending=False).reset_index()
    log.debug(f'importance: {df_importance}')

    # 生成线下
    df_oof = pd.concat(oof)
    df_oof.sort_values(['user_id', 'pred'],
                       inplace=True,
                       ascending=[True, False])
    log.debug(f'df_oof.head: {df_oof.head()}')

    # 计算相关指标
    total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_oof, total)
    log.debug(
        f'{hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )

    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('../prediction_result', exist_ok=True)
    df_sub.to_csv(f'../prediction_result/result.csv', index=False)


def online_predict(df_test):  # 功能是利用预先训练好的模型对测试数据进行预测，并生成提交文件。
    # 该函数会对测试数据进行特征筛选，然后使用 5 折交叉验证训练得到的模型进行预测，最后将预测结果进行平均处理，生成提交文件并保存到指定路径。
    ycol = 'label'
    feature_names = list(
        filter(
            lambda x: x not in [ycol, 'created_at_datetime', 'click_datetime'],
            df_test.columns))  # 排除目标列 label 以及 created_at_datetime 和 click_datetime 这两列。
    feature_names.sort()

    prediction = df_test[['user_id', 'article_id']]
    prediction['pred'] = 0

    for fold_id in tqdm(range(5)):  # 循环 5 次，代表 5 折交叉验证。
        model = joblib.load(f'../user_data/model/lgb{fold_id}.pkl')  # 借助 joblib.load 函数从指定路径加载第 fold_id 折训练好的 LightGBM 模型文件。
        pred_test = model.predict_proba(df_test[feature_names])[:, 1]  # 利用加载的模型对 df_test 中的特征列 feature_names 进行预测。
        # predict_proba 方法会返回每个样本属于各个类别的概率，这里选取索引为 1 的列，也就是正类别的概率，将结果存储在 pred_test 中。
        prediction['pred'] += pred_test / 5

    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('../prediction_result', exist_ok=True)
    df_sub.to_csv(f'../prediction_result/result.csv', index=False)


if __name__ == '__main__':
    if mode == 'valid':  # 对数据中的 object 类型特征进行编码，然后调用 train_model 函数进行模型训练。
        # object 类型的数据实际上是 Python 对象的引用，这意味着它可以存储任意类型的 Python 对象，例如字符串、列表、字典、自定义对象等。
        df_feature = pd.read_pickle('../user_data/data/offline/feature.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        for f in df_feature.select_dtypes('object').columns:
            lbl = LabelEncoder()
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))

        train_model(df_feature, df_query)
    else:
        df_feature = pd.read_pickle('../user_data/data/online/feature.pkl')
        online_predict(df_feature)
