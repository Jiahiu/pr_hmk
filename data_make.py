import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F


df = pd.read_csv("./data/train.csv")

cat_col = df.select_dtypes(include=["object"]).columns.tolist()
# print(len(cat_col))
encoder = OneHotEncoder()
encoded_categ = encoder.fit_transform(df[cat_col])
# print(encoded_categ.shape[1])
num_col = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
df[num_col] = df[num_col].apply(pd.to_numeric, errors="coerce")

fname = encoder.get_feature_names_out(cat_col)
# 将编码后的数据与原始数据合并
x1 = df[num_col]
x2 = pd.DataFrame(encoded_categ.toarray(), columns=fname)
encoded_df = pd.concat(
    [x1, x2],
    axis=1,
)

# 转torch
X = torch.tensor(
    encoded_df.drop(columns=["id", "subscribe_yes", "subscribe_no"]).values,
    dtype=torch.float32,
)
X = X.unsqueeze(1)
# print(X.shape)
yo = torch.tensor(
    encoded_df[["subscribe_yes", "subscribe_no"]].values, dtype=torch.float32
)
y = torch.argmax(yo, dim=1)
print(y.shape)
# y.unsqueeze(1)
y = y.float()
dataset = TensorDataset(X, y)

# 确定训练集和验证集的大小
total_size = len(dataset)
train_size = int(0.8 * total_size)  # 80% 用于训练
val_size = total_size - train_size  # 剩余的20% 用于验证

# 划分训练集和验证集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


class RNNMLP(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_layers, output_size):
        super(RNNMLP, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers

        # RNN层
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers=rnn_layers, batch_first=True
        )

        # 由于RNN的输出是序列，我们需要在RNN后使用一个全连接层来将序列转换为单个输出
        self.fc1 = nn.Linear(hidden_size, 64)  # RNN层到隐藏层1
        self.fc2 = nn.Linear(64, output_size)  # 隐藏层1到输出层

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.rnn_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播RNN
        out, _ = self.rnn(x, h0)

        # 由于RNN的输出是序列，我们取最后一个时间步长的输出
        out = out[:, -1, :]
        # print(out.shape)
        # 通过剩余的全连接层
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)

        # 使用softmax激活函数进行多分类
        out = torch.softmax(out, dim=1)
        out = torch.argmax(out, dim=1).float()
        return out


input_size = 63
hidden_size = 50
rnn_layers = 1  # RNN层的数量
output_size = 2  # 输出层的类别数

model = RNNMLP(input_size, hidden_size, rnn_layers, output_size)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算二元交叉熵损失
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        # 计算Focal Loss中的调节因子
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # 根据设定的reduction参数来应用损失的缩减方式
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# 初始化Focal Loss
criterion = FocalLoss(gamma=2, alpha=0.25)
# 定义损失函数和优化器
# criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_and_validate_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs
):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            # print(output)
            loss = criterion(output, target)
            loss = torch.tensor(loss, requires_grad=True)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            # 计算准确率
            total_train_correct += (output == target).sum().item()
            total_train_samples += target.size(0)
        average_train_loss = total_train_loss / len(train_loader)
        train_accuracy = total_train_correct / total_train_samples

        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        total_val_samples = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                total_val_loss += loss.item()
                # 计算准确率
                total_val_correct += (output == target).sum().item()
                total_val_samples += target.size(0)
        average_val_loss = total_val_loss / len(val_loader)
        val_accuracy = total_val_correct / total_val_samples

        print(
            f"Epoch {epoch+1}, Train Loss: {average_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {average_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )


# 训练和验证模型
train_and_validate_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=60
)
