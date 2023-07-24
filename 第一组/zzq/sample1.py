Train = pd.read_csv('E:\\data1.csv', delimiter=',')  # 训练集
Test = pd.read_csv('E:\\data2.csv', delimiter=',')  # 测试集
print('Train shape: ', Train.shape)
print('Test shape: ', Test.shape)
Combine = pd.concat([Train, Test])  # 将测试集和训练集合并
print('Combine shape: ', Combine.shape)
print(Combine.isna().sum()) # 统计每一行缺失值
Combine = Combine.fillna(Combine.mode().iloc[0, :])  # 用每一列出现最多的数据填充。

# 匿名特征处理：D_11。拆分为length、width和high
series1 = Combine['D_11'].str.split('*', expand=True)
Combine['length'] = series1[0]
Combine['width'] = series1[1]
Combine['high'] = series1[2]
 
Combine['length'] = Combine['length'].astype(float)
Combine['width'] = Combine['width'].astype(float)
Combine['high'] = Combine['high'].astype(float)


def One_Hot(OneHotCol):
    new_cols = []
    for old_col in OneHotCol:
        new_cols += sorted(['{0}_{1}'.format(old_col, str(x).lower()) for x in set(Combine[old_col].values)])
    ec = OneHotEncoder()
    ec.fit(Combine[OneHotCol].values)
    # list(Combine.index.values)  # 取出Combine的索引
    OneHotCode = pd.DataFrame(ec.transform(Combine[OneHotCol]).toarray(), columns=new_cols,
                              index=list(Combine.index.values)).astype(int)
    return OneHotCode

OneHotCol = ['carCode', 'color', 'country', 'maketype', 'oiltype', 'D_7', 'D_8', 'D_9', 'D_10', 'D_13']
OneHotCode = One_Hot(OneHotCol)
# 合并Combine和OneHotCode
Combine = pd.concat([Combine, OneHotCode], axis=1)

# 日期格式转换函数
def date_proc(x):
    month = int(x[4:6])
    if month == 0:
        month = 1
    if len(x) == 6:
        return x[:4] + '-' + str(month)
    else:
        return x[:4] + '-' + str(month) + '-' + x[6:]
    
# 日期特征提取函数（提取年、月、日、周几这些特征）
def date_transform(df, fea_col):
    for f in tqdm(fea_col):
        df[f] = pd.to_datetime(df[f].astype('str').apply(date_proc))
        df[f + '_year'] = df[f].dt.year
        df[f + '_month'] = df[f].dt.month
        df[f + '_day'] = df[f].dt.day
        df[f + '_dayofweek'] = df[f].dt.dayofweek
    return (df)

Date = ['registerDate', 'tradeTime', 'licenseDate']
Combine = date_transform(Combine, Date)


Combine = Combine[Combine['D_12'].notna()]
Combine['D_12'].astype('str').apply(date_proc)
Combine['D_12'] = pd.to_datetime(Train['D_12'])
Combine['D_12_year'] = Combine['D_12'].dt.year
Combine['D_12_month'] = Combine['D_12'].dt.month


# 对提取的日期特征进行One-hot编码
OneHotCol2 = ['registerDate_year', 'registerDate_month', 'registerDate_dayofweek', 'tradeTime_year', 'tradeTime_month',
             'tradeTime_dayofweek', 'licenseDate_year', 'licenseDate_month', 'licenseDate_dayofweek', 'D_12_year',
             'D_12_month']
 
OneHotCode2 = One_Hot(OneHotCol2)
Combine = pd.concat([Combine, OneHotCode2], axis=1)


# 构建特征：汽车使用天数
Combine['used_time1'] = (pd.to_datetime(Combine['tradeTime'], format='%Y%m%d', errors='coerce') -
                      pd.to_datetime(Combine['registerDate'], format='%Y%m%d', errors='coerce')).dt.days
# 构建特征：汽车注册日期距今天数
Combine['used_time2'] = (
        pd.datetime.now() - pd.to_datetime(Combine['registerDate'], format='%Y%m%d', errors='coerce')).dt.days
# 构建特征：汽车上线日期距今天数
Combine['used_time3'] = (pd.datetime.now() - pd.to_datetime(Combine['tradeTime'], format='%Y%m%d', errors='coerce')).dt.days


# 数据分桶函数
def cut_group(df, cols, num_bins=50):
    for col in cols:
        all_range = int(df[col].max() - df[col].min())
        # ceil():返回一个数的上取整数；floor():返回一个数的下舍整数
        bin = [np.ceil(df[col].min() - 1) + np.floor(i * all_range / num_bins) for i in range(num_bins + 2)]
        # bin是一个列表，区间两端的选取就是跟据bin里的数据决定。如第一个区间就是[bin[0], bin[1]]
        df[col + '_bin'] = pd.cut(df[col], bin, labels=False)
    return df
 
 
# 对汽车使用天数，汽车注册日期距今天数 ，汽车上线日期距今天数进行数据分桶
CutCol = ['used_time1', 'used_time2', 'used_time3']
Combine = cut_group(Combine, CutCol, 50)


AllCol = Combine.columns
Train = Combine.iloc[:len(Train), :][AllCol]
a = dict(Train.corr()['price'])  # 各变量与price变量的相关性
asortlist = sorted(a.items(), key=lambda x: x[1], reverse=True)  # 以字典的值为基准对字典的项进行排序
for i in asortlist:
    print(i)


# 特征交叉函数
def cross_feature(df, fea_col, Nfea_col):
    for i in tqdm(fea_col):  # 遍历分类特征
        for j in tqdm(Nfea_col):  # 遍历数值特征
            # 调用groupby（）函数，以参数i分组，之后，用agg函数对数据做一些聚合操作（求最大值、最小值、中位数）
            feat = df.groupby(i, as_index=False)[j].agg({
                '{}_{}_max'.format(i, j): 'max',  # 最大值
                '{}_{}_min'.format(i, j): 'min',  # 最小值
                '{}_{}_median'.format(i, j): 'median',  # 中位数
            })
            df = df.merge(feat, on=i, how='left')
    return (df)
 
# 挑选与Price相关程度高的非匿名变量和匿名变量作特征交叉
Cross_fea = ['newprice', 'displacement', 'width', 'length', 'maketype', 'maketype_3', 'modelyear']
Cross_Nfea = ['D_1', 'D_10_3', 'D_7', 'D_7_5', 'D_10', 'D_4', 'D_12']
Combine = cross_feature(Combine, Cross_fea, Cross_Nfea)

MeanEncol = ['model', 'brand', 'registerDate', 'tradeTime']
# 如果是回归场景，那么target_type='regression';如果是分类场景，那么target_type='classification'
MeanFit = MeanEncoder(MeanEncol, target_type='regression')
XTrain = MeanFit.fit_transform(XTrain, YTrain)
XTest = MeanFit.transform(XTest)

# K折目标编码，
# 回归场景中，对目标进行编码的常用方式：最小值、最大值、中位数、均值、求和、标准差、偏度、峰度、中位数绝对偏差
XTrain['price'] = Train['price']
EncCol = []
StatDefaultDict = {
    'max': XTrain['price'].max(),
    'min': XTrain['price'].min(),
    'median': XTrain['price'].median(),
    'mean': XTrain['price'].mean(),
    'sum': XTrain['price'].sum(),
    'std': XTrain['price'].std(),
    'skew': XTrain['price'].skew(),
    'kurt': XTrain['price'].kurt(),
    'mad': XTrain['price'].mad()
}
# 采用最大值、最小值、均值对目标特征price分别进行编码
 
EncStat = ['max', 'min', 'mean']
# 分为10折
KF = KFold(n_splits=10, shuffle=True, random_state=2023)
for f in tqdm(['serial', 'brand', 'registerDate_year', 'tradeTime_year', 'mileage', 'model']):
    EncDict = {}
    for stat in EncStat:
        EncDict['{}_target_{}'.format(f, stat)] = stat
        XTrain['{}_target_{}'.format(f, stat)] = 0
        XTest['{}_target_{}'.format(f, stat)] = 0
        EncCol.append('{}_target_{}'.format(f, stat))
    for i, (TrnIndex, ValIndex) in enumerate(KF.split(XTrain, YTrain)):
        TrnX, ValX = XTrain.iloc[TrnIndex].reset_index(drop=True), XTrain.iloc[ValIndex].reset_index(drop=True)
        EncDF = TrnX.groupby(f, as_index=False)['price'].agg(EncDict)
        ValX = ValX[[f]].merge(EncDF, on=f, how='left')
        TestX = XTest[[f]].merge(EncDF, on=f, how='left')
        for stat in EncStat:
            ValX['{}_target_{}'.format(f, stat)] = ValX['{}_target_{}'.format(f, stat)].fillna(StatDefaultDict[stat])
            TestX['{}_target_{}'.format(f, stat)] = TestX['{}_target_{}'.format(f, stat)].fillna(StatDefaultDict[stat])
            XTrain.loc[ValIndex, '{}_target_{}'.format(f, stat)] = ValX['{}_target_{}'.format(f, stat)].values
            XTest['{}_target_{}'.format(f, stat)] += TestX['{}_target_{}'.format(f, stat)].values / KF.n_splits


# 归一化（极差法）
Scaler = MinMaxScaler()
Scaler.fit(pd.concat([XTrain, XTest]).values)
CombineScaler = Scaler.transform(pd.concat([XTrain, XTest]).values)
print('CombineScaler shape: ', CombineScaler.shape)
# 调用sklearn库中decomposition模块中的PCA算法包对数据进行降维操作
# PCA降维
PCA = decomposition.PCA(n_components=550)
CombinePCA = PCA.fit_transform(CombineScaler)
XTrainPCA = CombinePCA[:len(XTrain)]
XTestPCA = CombinePCA[len(XTrain):]
 
YTrain = Train['price'].values
print('CombinePCA shape: ', CombinePCA.shape)