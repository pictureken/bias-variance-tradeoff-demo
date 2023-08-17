import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from . import calculate


# 3層の全結合ニューラルネットワーク
class NeuralNet(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class NeuralNetSimulator(calculate.Simulator):
    def train(self, model, optimizer, criterion, iteration, device):
        self.model = model
        self.model.train()
        self.device = device
        tensor_x = torch.from_numpy(self.x.astype(np.float32)).float()
        tensor_x = torch.stack([torch.ones(tensor_x.shape), tensor_x], 1)
        tensor_y = torch.from_numpy(self.y.astype(np.float32)).float()
        for i in range(iteration):
            optimizer.zero_grad()
            y_pred = self.model(tensor_x.to(self.device))
            loss = criterion(y_pred.reshape(tensor_y.shape), tensor_y.to(self.device))
            loss.backward()
            optimizer.step()
            print(f"epoch={i},loss={loss}")

    def test(self):
        self.model.eval()
        tensor_x = torch.from_numpy(self.x_test.astype(np.float32)).float()
        tensor_x = torch.stack([torch.ones(tensor_x.shape), tensor_x], 1)
        with torch.no_grad():
            self.y_pred = (
                self.model(tensor_x.to(self.device))
                .cpu()
                .data.numpy()
                .T[0]
                .reshape(-1, 1)
            )


def task(
    is_fix: bool,
    toy_sample_size: int = 100,
    std: float = 1.0,
    test_sample_size: int = 50,
    trial: int = 100,
    iteration: int = 5000,
    hidden_size: int = 35,
):
    y_pred_list = []
    # Simulatorクラス呼び出し
    simulator = NeuralNetSimulator(is_fix=is_fix)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(in_features=2, hidden_size=hidden_size, out_features=1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.MSELoss()
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    for i in range(trial):
        # 学習データを生成
        simulator.create_toy_data(toy_sample_size=toy_sample_size, std=std)
        # 評価用データを生成
        simulator.create_test_data(test_sample_size=test_sample_size)
        # モデルの学習
        simulator.train(model, optimizer, criterion, iteration, device)
        # 評価
        simulator.test()
        y_pred_list.append(simulator.y_pred)
        # 20個の予測結果をプロット
        if i < 20:
            plt.plot(
                simulator.x_test,
                simulator.y_pred.reshape(-1, 1),
                c="orange",
            )
    # 全ての予測結果の平均をプロット
    plt.plot(
        simulator.x_test, np.asarray(y_pred_list).mean(axis=0), c="green", label="mean"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-1.5, 1.5)

    # 20個の予測結果の平均と真の関数をプロット
    plt.subplot(1, 2, 2)
    plt.plot(simulator.x_test, simulator.y_true, c="red", label="gt")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(
        simulator.x_test, np.asarray(y_pred_list).mean(axis=0), c="green", label="mean"
    )
    plt.ylim(-1.5, 1.5)
    plt.legend()
    os.makedirs(f"./outputs/nn/{is_fix}", exist_ok=True)
    plt.savefig(f"./outputs/nn/{is_fix}/d{hidden_size}.png")
    plt.close()

    # 二乗誤差とそれに対応する偏りと分散の算出
    mse, bias2, variance = simulator.bias_variance_tradeoff(y_pred_list)

    return {"mse": mse, "bias2": bias2, "variance": variance}


if __name__ == "__main__":
    hidden_size_list = [5, 10, 15, 17, 20, 22, 25, 35, 75, 100, 1000, 10000]
    task(is_fix=False)
