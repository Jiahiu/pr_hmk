from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, auc, roc_auc_score

import pandas as pd
import numpy as np

df = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
df.drop()
# 数据清洗
numerical_cols_df = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
for column in numerical_cols_df:
    # 计算Q1和Q3
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    # 计算IQR
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 替换异常值为NaN或者某个特定值，例如中位数
    # df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), np.nan, df[column])
    # 或者用更稳健的方法，例如用中位数替换异常值
    median = df[column].median()
    df[column] = np.where(
        (df[column] < lower_bound) | (df[column] > upper_bound), median, df[column]
    )

categorical_cols_df = df.select_dtypes(include=["object"]).columns.tolist()
df[categorical_cols_df] = (
    df[categorical_cols_df]
    .apply(lambda x: x.astype("category"))
    .apply(lambda x: x.cat.codes)
)

categorical_cols_ts = test.select_dtypes(include=["object"]).columns.tolist()
test[categorical_cols_ts] = (
    test[categorical_cols_ts]
    .apply(lambda x: x.astype("category"))
    .apply(lambda x: x.cat.codes)
)

X = df.drop(columns=["id", "subscribe"])  # drop丢弃
Y = df["subscribe"]
test = test.drop(columns="id")

# 划分训练及测试集
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
# 建立模型
gbm = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.01,
    boosting_type="gbdt",
    objective="binary",
    max_depth=-1,
    random_state=2022,
    metric="auc",
)
# 交叉验证
result1 = []
mean_score1 = 0
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=2022)
for train_index, test_index in kf.split(X):
    x_train = X.iloc[train_index]
    # print(x_train.shape)
    y_train = Y.iloc[train_index]
    x_test = X.iloc[test_index]
    y_test = Y.iloc[test_index]
    gbm.fit(x_train, y_train)
    y_pred1 = gbm.predict_proba((x_test), num_iteration=gbm.best_iteration_)[:, 1]
    # print(y_pred1)
    # print("验证集AUC:{}".format(roc_auc_score(y_test, y_pred1)))
    mean_score1 += roc_auc_score(y_test, y_pred1) / n_folds
    y_pred_final1 = gbm.predict_proba((test), num_iteration=gbm.best_iteration_)[:, 1]
    y_pred_test1 = y_pred_final1
    result1.append(y_pred_test1)
# 模型评估
print("mean 验证集auc:{}".format(mean_score1))
# cat_pre1 = sum(result1) / n_folds
# ret1 = pd.DataFrame(cat_pre1, columns=["subscribe"])
# ret1["subscribe"] = np.where(ret1["subscribe"] > 0.5, "yes", "no").astype("str")
# ret1.to_csv("/GBM预测.csv", index=False)

# 0.89
