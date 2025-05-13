import matplotlib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='SimHei', size=13)

import os,gc,re,warnings,sys
warnings.filterwarnings("ignore")

# 读取数据
# path = './data/' # 自定义的路径
path = './download/' # 天池平台路径

#####train
trn_click = pd.read_csv(path+'train_click_log.csv')
#trn_click = pd.read_csv(path+'train_click_log.csv', names=['user_id','item_id','click_time','click_environment','click_deviceGroup','click_os','click_country','click_region','click_referrer_type'])
item_df = pd.read_csv(path+'articles.csv')
item_df = item_df.rename(columns={'article_id': 'click_article_id'})  #重命名，方便后续match
item_emb_df = pd.read_csv(path+'articles_emb.csv')

#####test
tst_click = pd.read_csv(path+'testA_click_log.csv')

# 数据预处理
# 对每个用户的点击时间戳进行排序
trn_click['rank'] = trn_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
tst_click['rank'] = tst_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)

#计算用户点击文章的次数，并添加新的一列count。transform('count') 是一种聚合操作，计算每个分组（即每个用户）在 click_timestamp 列中的非缺失值（NaN）的数量，也就是每个用户的点击次数。
# #这里 transform 的作用是让结果按行重复返回，即每一行都会得到相应用户的点击次数。
trn_click['click_cnts'] = trn_click.groupby(['user_id'])['click_timestamp'].transform('count')
tst_click['click_cnts'] = tst_click.groupby(['user_id'])['click_timestamp'].transform('count')

# 数据浏览
# 用户点击日志文件_训练集
trn_click = trn_click.merge(item_df, how='left', on=['click_article_id'])
print(trn_click.head())

#用户点击日志信息
trn_click.info()

# 用于生成数据框（DataFrame）中各个数值型列的统计汇总信息。
print(trn_click.describe())

#训练集中的用户数量为20w
print(trn_click.user_id.nunique())

print(trn_click.groupby('user_id')['click_article_id'].count().min())  # 训练集里面每个用户至少点击了两篇文章

plt.figure()
plt.figure(figsize=(15, 20))  # 绘图窗口的大小为宽 15 英寸，高 20 英寸。
i = 1  # 初始化子图索引
for col in ['click_article_id', 'click_timestamp', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country',
            'click_region', 'click_referrer_type', 'rank', 'click_cnts']:  # for 循环遍历指定的列名列表。
    plot_envs = plt.subplot(5, 2, i)  # 创建一个 5x2 的子图网格，并在每次循环时选择下一个子图。
    i += 1  # 每次循环后，i 增加 1，以便选择下一个子图的位置。
    v = trn_click[col].value_counts().reset_index()[:10]  # 通过 [:10] 获取前 10 个最常见的值和它们的出现次数。
    fig = sns.barplot(x=v['index'], y=v[col])   # 绘制柱状图,x=v['index'] 表示 x 轴是类别（从 v 数据框中获取的索引），y=v[col] 是对应的计数值。
    for item in fig.get_xticklabels():   # 旋转 x 轴标签
        item.set_rotation(90)
    plt.title(col)  # 设置子图标题
plt.tight_layout()  # 调整子图布局
plt.show()  # 显示图形  # 此处click_cnts直方图表示的是每篇文章对应用户的点击次数累计图

# 从点击环境click_environment来看，仅有2102次（占0.19%）点击环境为1；仅有25894次（占2.3%）点击环境为2；剩余（占97.6%）点击环境为4。
print(trn_click['click_environment'].value_counts())

# 从点击设备组click_deviceGroup来看，设备1占大部分（61%），设备3占36%。
print(trn_click['click_deviceGroup'].value_counts())

tst_click = tst_click.merge(item_df, how='left', on=['click_article_id'])
print(tst_click.head())

print(tst_click.describe())

#测试集中的用户数量为5w
print(tst_click.user_id.nunique())

print(tst_click.groupby('user_id')['click_article_id'].count().min())  # 注意测试集里面有只点击过一次文章的用户

#新闻文章数据集浏览
print(item_df.head().append(item_df.tail()))

# item_df['words_count']选择 item_df 数据框中的 words_count 列，它可能表示文章的词数（word count）。.value_counts()统计 words_count 列中每个唯一值的出现次数，并按降序排列。
print(item_df['words_count'].value_counts())

print(item_df['category_id'].nunique())     # 计算 category_id 列中唯一值的个数，即不同的文章主题数量。461个文章主题
item_df['category_id'].hist()  # 这行代码绘制 category_id 的直方图（Histogram），用于查看文章类别的分布情况。

print(item_df.shape)  # 364047篇文章 (364047, 4)

# 新闻文章embedding向量表示
print(item_emb_df.head())

print(item_emb_df.shape)  # (295141, 251)

# 数据分析
# 用户重复点击
#####merge
user_click_merge = trn_click.append(tst_click)

#用户重复点击
# 使用 groupby 方法按照 user_id 和 click_article_id 这两列进行分组。
# 针对分组后的数据，对 click_timestamp 列使用 agg 方法进行聚合操作，这里聚合函数为 count，也就是统计每个分组中 click_timestamp 的数量。
# 使用 reset_index 方法将分组索引转换为普通列，使得结果是一个普通的 DataFrame 结构。
user_click_count = user_click_merge.groupby(['user_id', 'click_article_id'])['click_timestamp'].agg({'count'}).reset_index()
print(user_click_count[:10])

print(user_click_count[user_click_count['count']>7])  # 从 user_click_count 数据框中筛选出 count 列的值大于 7 的所有行，并将这些行打印输出。

print(user_click_count['count'].unique())  # 找出 user_click_count 数据框中 count 列的所有唯一值，并将这些唯一值打印输出

#用户点击新闻次数.
# loc 索引器从 user_click_count 数据框中选取所有行（: 表示选取所有行）的 count 列..value_counts() 是 pandas.Series 对象的一个方法，它会对 Series 中的每个唯一值进行计数.
# 主要功能是统计 user_click_count 数据框中 count 列各个唯一值出现的频数
print(user_click_count.loc[:,'count'].value_counts())
# 可以看出：有1605541（约占99.2%）的用户未重复阅读过文章，仅有极少数用户重复点击过某篇文章。 这个也可以单独制作成特征

# 用户点击环境变化分析
def plot_envs(df, cols, r, c):
    plt.figure()  # 用来初始化绘图的画布
    plt.figure(figsize=(10, 5))  # 设置画布的大小。
    i = 1  # 初始化子图索引，从第一个子图开始绘制。
    for col in cols:
        plt.subplot(r, c, i)  # 为每个环境特征创建一个子图，r 和 c 控制子图的行和列数，i 代表当前子图的索引。
        i += 1
        v = df[col].value_counts().reset_index()  # 对当前列（col）进行频数统计，返回一个包含值和计数的 DataFrame，reset_index() 是为了将统计结果转化为可访问的 DataFrame 格式。
        fig = sns.barplot(x=v['index'], y=v[col])  # 使用 Seaborn 的 barplot 绘制柱状图，横轴是环境特征的不同值，纵轴是它们的频数。
        for item in fig.get_xticklabels():
            item.set_rotation(90)  # 这一段是调整横轴标签的显示角度，旋转标签使其竖直，防止重叠。
        plt.title(col)
    plt.tight_layout()  # 调整子图的布局，使其不重叠。
    plt.show()


# 分析用户点击环境变化是否明显，这里随机采样10个用户分析这些用户的点击环境分布
sample_user_ids = np.random.choice(tst_click['user_id'].unique(), size=10, replace=False)  # 从 tst_click 数据中随机选择 10 个用户 ID，replace=False 表示不重复选择。
sample_users = user_click_merge[user_click_merge['user_id'].isin(sample_user_ids)]  # 使用 sample_user_ids 中的用户 ID 从 user_click_merge 数据中提取对应的用户点击数据。
cols = ['click_environment','click_deviceGroup', 'click_os', 'click_country', 'click_region','click_referrer_type']
for _, user_df in sample_users.groupby('user_id'):  # 对于每个随机选择的用户，按 user_id 对数据进行分组。
    plot_envs(user_df, cols, 2, 3)
# 可以看出绝大多数数的用户的点击环境是比较固定的。思路：可以基于这些环境的统计特征来代表该用户本身的属性


# 用户点击新闻数量的分布
user_click_item_count = sorted(user_click_merge.groupby('user_id')['click_article_id'].count(), reverse=True)
plt.plot(user_click_item_count)
plt.show()

#点击次数在前50的用户。点击次数排前50的用户的点击次数都在100次以上。思路：我们可以定义点击次数大于等于100次的用户为活跃用户
plt.plot(user_click_item_count[:50])
plt.show()

#点击次数排名在[25000:50000]之间。看出点击次数小于等于两次的用户非常的多，这些用户可以认为是非活跃用户
plt.plot(user_click_item_count[25000:50000])
plt.show()

# 新闻点击次数分析
item_click_count = sorted(user_click_merge.groupby('click_article_id')['user_id'].count(), reverse=True)
plt.plot(item_click_count)
plt.show()

# 可以看出点击次数最多的前100篇新闻，点击次数大于1000次
plt.plot(item_click_count[:100])
plt.show()

# 点击次数最多的前20篇新闻，点击次数大于2500。思路：可以定义这些新闻为热门新闻
plt.plot(item_click_count[:20])
plt.show()

# 可以发现很多新闻只被点击过一两次。思路：可以定义这些新闻是冷门新闻
plt.plot(item_click_count[3500:])
plt.show()

# 新闻共现频次：两篇新闻连续出现的次数
tmp = user_click_merge.sort_values('click_timestamp')  # 按点击时间排序
tmp['next_item'] = tmp.groupby(['user_id'])['click_article_id'].transform(lambda x:x.shift(-1)) # 对于每个用户的点击记录，使用 shift(-1) 创建一个新列 next_item，该列表示当前点击的下一篇文章
union_item = tmp.groupby(['click_article_id','next_item'])['click_timestamp'].agg({'count'}).reset_index().sort_values('count', ascending=False)  # 计算文章对的点击次数
union_item[['count']].describe()  # 生成 count 列的统计描述，包括计数、均值、标准差、最小值、四分位数等。这有助于了解文章对的点击次数的分布情况。
# 由统计数据可以看出，平均共现次数3.18，最高为2202。
# 说明用户看的新闻，相关性是比较强的。

#画个图直观地看一看
x = union_item['click_article_id']
y = union_item['count']
plt.scatter(x, y)
plt.show()

# 大概有75000个pair至少共现一次
plt.plot(union_item['count'].values[40000:])  # 这表示对 count 数组进行切片，从索引 40000 开始，直到数组的末尾。这意味着你只关注 count 数组的后面部分
plt.show()

# 新闻文章信息
# 不同类型的新闻出现的次数
plt.plot(user_click_merge['category_id'].value_counts().values)
plt.show()

#出现次数比较少的新闻类型, 有些新闻类型，基本上就出现过几次
plt.plot(user_click_merge['category_id'].value_counts().values[:150])
plt.show()

#新闻字数的描述性统计
user_click_merge['words_count'].describe()

plt.plot(user_click_merge['words_count'].values)
plt.show()


# 用户点击的新闻类型的偏好
plt.plot(sorted(user_click_merge.groupby('user_id')['category_id'].nunique(), reverse=True))
plt.show()
# 从上图中可以看出有一小部分用户阅读类型是极其广泛的，大部分人都处在20个新闻类型以下。

user_click_merge.groupby('user_id')['category_id'].nunique().reset_index().describe()

# 用户查看文章的长度的分布
plt.plot(sorted(user_click_merge.groupby('user_id')['words_count'].mean(), reverse=True))
plt.show()
# 从上图中可以发现有一小部分人看的文章平均词数非常高，也有一小部分人看的平均文章次数非常低。
# 大多数人偏好于阅读字数在200-400字之间的新闻。

#挑出大多数人的区间仔细看看
plt.plot(sorted(user_click_merge.groupby('user_id')['words_count'].mean(), reverse=True)[1000:45000])
plt.show()
# 可以发现大多数人都是看250字以下的文章

#更加详细的参数
user_click_merge.groupby('user_id')['words_count'].mean().reset_index().describe()

# 用户点击新闻的时间分析
#为了更好的可视化，这里把时间进行归一化操作
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()  # 该实例将在后续用于将时间戳数据进行归一化
user_click_merge['click_timestamp'] = mm.fit_transform(user_click_merge[['click_timestamp']])  # 对 click_timestamp 进行归一化
user_click_merge['created_at_ts'] = mm.fit_transform(user_click_merge[['created_at_ts']])

user_click_merge = user_click_merge.sort_values('click_timestamp')  # 对数据按 click_timestamp 排序

print(user_click_merge.head())

def mean_diff_time_func(df, col):  # 定义了一个计算时间差平均值的函数.df: 需要处理的 DataFrame，包含了时间数据。.col: 要计算时间差的列名
    df = pd.DataFrame(df, columns={col})  # 我们从输入的 df 中只提取指定的 col 列,，并将其转换为新的 DataFrame。
    df['time_shift1'] = df[col].shift(1).fillna(0)  # 将 col 列中的数据向下移动一位，创建一个新的列 time_shift1。这意味着每一行的 time_shift1 对应于前一行的 col 值。
    # fillna(0)：填充 NaN 值。shift(1) 会在第一行生成 NaN，因此用 0 来填充这个 NaN
    df['diff_time'] = abs(df[col] - df['time_shift1'])
    return df['diff_time'].mean()

# 点击时间差的平均值
# 照 user_id 列进行分组。进行分组后，代码选择了两个列：click_timestamp 和 created_at_ts。这意味着，接下来对每个用户的数据只会考虑这两个列。
# apply 函数用于对每个用户的分组数据进行操作，操作的内容是使用 mean_diff_time_func 函数来计算某个时间差的均值。
mean_diff_click_time = user_click_merge.groupby('user_id')['click_timestamp', 'created_at_ts'].apply(lambda x: mean_diff_time_func(x, 'click_timestamp'))

plt.plot(sorted(mean_diff_click_time.values, reverse=True))
plt.show()
# 从上图可以发现不同用户点击文章的时间差是有差异的

# 前后点击文章的创建时间差的平均值
mean_diff_created_time = user_click_merge.groupby('user_id')['click_timestamp', 'created_at_ts'].apply(lambda x: mean_diff_time_func(x, 'created_at_ts'))

plt.plot(sorted(mean_diff_created_time.values, reverse=True))
plt.show()
# 从图中可以发现用户先后点击文章，文章的创建时间也是有差异的


from gensim.models import Word2Vec
import logging, pickle


# 需要注意这里模型只迭代了一次
def trian_item_word2vec(click_df, embed_size=16, save_name='item_w2v_emb.pkl', split_char=' '):
    click_df = click_df.sort_values('click_timestamp')
    # 只有转换成字符串才可以进行训练
    click_df['click_article_id'] = click_df['click_article_id'].astype(str)
    # 转换成句子的形式。将每个用户点击过的文章ID转换成一个列表。reset_index() 会将分组后的结果恢复为一个普通的 DataFrame，使 user_id 成为列而不是索引。否则，分组后 user_id 会成为索引。
    docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
    docs = docs['click_article_id'].values.tolist()

    # 为了方便查看训练的进度，这里设定一个log信息
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

    # 这里的参数对训练得到的向量影响也很大,默认负采样为5
    w2v = Word2Vec(docs, vector_size=16, sg=1, window=5, seed=2020, workers=24, min_count=1, epochs=10)

    # 保存成字典的形式
    item_w2v_emb_dict = {k: w2v.wv[k] for k in click_df['click_article_id']}

    return item_w2v_emb_dict


item_w2v_emb_dict = trian_item_word2vec(user_click_merge)

# 随机选择5个用户，查看这些用户前后查看文章的相似性
sub_user_ids = np.random.choice(user_click_merge.user_id.unique(), size=15, replace=False)
sub_user_info = user_click_merge[user_click_merge['user_id'].isin(sub_user_ids)]

print(sub_user_info.head())


# 上一个版本，这个函数使用的是赛题提供的词向量，但是由于给出的embedding并不是所有的数据的embedding，所以运行下面画图函数的时候会报keyerror的错误
# 为了防止出现这个错误，这里修改为使用word2vec训练得到的词向量进行可视化
def get_item_sim_list(df):
    sim_list = []
    item_list = df['click_article_id'].values
    for i in range(0, len(item_list)-1):
        emb1 = item_w2v_emb_dict[str(item_list[i])]  # 需要注意的是word2vec训练时候使用的是str类型的数据
        emb2 = item_w2v_emb_dict[str(item_list[i+1])]
        sim_list.append(np.dot(emb1,emb2)/(np.linalg.norm(emb1)*(np.linalg.norm(emb2))))
    sim_list.append(0)
    return sim_list


for _, user_df in sub_user_info.groupby('user_id'):
    item_sim_list = get_item_sim_list(user_df)
    plt.plot(item_sim_list)
    plt.show()

# 这里由于对词向量的训练迭代次数不是很多，所以看到的可视化结果不是很准确，可以训练更多次来观察具体的现象。