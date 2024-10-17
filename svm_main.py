from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# 数据加载
df = pd.read_csv("./data/train.csv")
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

from sklearn.preprocessing import LabelEncoder

# 为训练集的分类特征创建LabelEncoder实例
label_encoders = {}
for column in categorical_cols_df:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # 保存每个列的编码器，以便以后使用


X = df.drop(columns=["id", "subscribe"])  # drop丢弃
Y = df["subscribe"]

# 划分训练及测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# 创建一个 SVM 分类器
svm_model = SVC(
    C=1.0,  # 正则化参数，较大的C值表示较低的正则化强度
    kernel="rbf",  # 核函数，默认是‘rbf’，但是你可以尝试‘linear’, ‘poly’, ‘sigmoid’
    probability=True,  # 是否启用概率估计
    random_state=2022,
)

# 训练模型
svm_model.fit(x_train, y_train)

# 进行预测
y_pred = svm_model.predict(x_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 0.86
